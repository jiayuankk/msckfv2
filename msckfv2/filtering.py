"""
Filtering module for MSCKF+SKNet fusion.

This module provides various Kalman filter implementations:
- Extended_Kalman_Filter: Traditional EKF for baseline comparison
- KalmanNet_Filter: Neural network-based Kalman filter
- Split_KalmanNet_Filter: SKNet that outputs Pk and Sk separately
- Dual_Mode_Filter: Runs both baseline and SKNet in parallel for comparison

Key formulas for SKNet-based update:
    K = Pk @ H^T @ Sk
    delta_x = K @ r
    x_post = x_pred + delta_x
"""
from dnn import DNN_KalmanNet_GSS, DNN_SKalmanNet_GSS, KNet_architecture_v2
from model import GSSModel
import torch
import torch.nn.functional as F
import numpy as np

class Extended_Kalman_Filter():
    
    def __init__(self, GSSModel:GSSModel):
        
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.init_state = GSSModel.init_state
        self.Q = GSSModel.cov_q
        self.R = GSSModel.cov_r
        
        self.init_cov = torch.zeros((self.x_dim, self.x_dim))
        self.state_history = self.init_state.detach().clone()
        self.reset(clean_history=True)   

    def reset(self, clean_history=False):
        self.state_post = self.init_state.detach().clone()
        self.cov_post = self.init_cov.detach().clone()
        self.state_history = torch.cat((self.state_history, self.state_post), axis=1)
        if clean_history:
            self.state_history = self.init_state.detach().clone()     
            self.cov_trace_history = torch.zeros((1,))

    def filtering(self, observation):
        with torch.no_grad():
            # print(self.GSSModel.r2)
            # observation: column vector
            x_last = self.state_post
            x_predict = self.GSSModel.f(x_last)

            y_predict = self.GSSModel.g(x_predict)
            residual = observation - y_predict

            F_jacob = self.GSSModel.Jacobian_f(x_last)
            H_jacob = self.GSSModel.Jacobian_g(x_predict)
            cov_pred = (F_jacob @ self.cov_post @ torch.transpose(F_jacob, 0, 1)) + self.Q

            K_gain = cov_pred @ torch.transpose(H_jacob, 0, 1) @ \
                torch.linalg.inv(H_jacob@cov_pred@torch.transpose(H_jacob, 0, 1) + self.R)

            x_post = x_predict + (K_gain @ residual)

            cov_post = (torch.eye(self.x_dim) - K_gain @ H_jacob) @ cov_pred
            cov_trace = torch.trace(cov_post)

            self.state_post = x_post.detach().clone()
            self.cov_post = cov_post.detach().clone()
            self.state_history = torch.cat((self.state_history, x_post.clone()), axis=1)    
            self.cov_trace_history = torch.cat((self.cov_trace_history, cov_trace.reshape(-1).clone()))

            self.pk = cov_pred

class KalmanNet_Filter():
    def __init__(self, GSSModel:GSSModel):
        
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.kf_net = DNN_KalmanNet_GSS(self.x_dim, self.y_dim)
        self.init_state = GSSModel.init_state
        self.reset(clean_history=True)
        

    def reset(self, clean_history=False):
        self.dnn_first = True
        self.kf_net.initialize_hidden()
        self.state_post = self.init_state.detach().clone()
        if clean_history:
            self.state_history = self.init_state.detach().clone()  
            self.cov_trace_history = torch.zeros((1,))      
        self.state_history = torch.cat((self.state_history, self.state_post), axis=1)
          

    def filtering(self, observation):
        # observation: column vector

        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        x_last = self.state_post
        x_predict = self.GSSModel.f(x_last)

        if self.dnn_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()

        y_predict = self.GSSModel.g(x_predict)
        residual = observation - y_predict

        ## input 1: x_{k-1 | k-1} - x_{k-1 | k-2}
        state_inno = self.state_post_past - self.state_pred_past
        ## input 2: residual
        ## input 3: x_k - x_{k-1}
        diff_state = self.state_post - self.state_post_past
        ## input 4: y_k - y_{k-1}
        diff_obs = observation - self.obs_past

        K_gain = self.kf_net(state_inno, residual, diff_state, diff_obs)

        # state_inno_in = F.normalize(state_inno, p=2, dim=0, eps=1e-12)
        # residual_in = F.normalize(residual, p=2, dim=0, eps=1e-12)
        # diff_state_in = F.normalize(diff_state, p=2, dim=0, eps=1e-12)
        # diff_obs_in = F.normalize(diff_obs, p=2, dim=0, eps=1e-12)
        # # residual_in = residual
        # # diff_obs_in = diff_obs
        # K_gain = self.kf_net(state_inno_in, residual_in, diff_state_in, diff_obs_in)


        x_post = x_predict + (K_gain @ residual)

        self.dnn_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), axis=1)


class KalmanNet_Filter_v2():
    def __init__(self, GSSModel:GSSModel):
        
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.kf_net = KNet_architecture_v2(self.x_dim, self.y_dim)
        self.init_state = GSSModel.init_state
        self.reset(clean_history=True)
        

    def reset(self, clean_history=False):
        self.dnn_first = True
        self.kf_net.initialize_hidden()
        self.state_post = self.init_state.detach().clone()
        if clean_history:
            self.state_history = self.init_state.detach().clone()
            self.cov_trace_history = torch.zeros((1,))     
        self.state_history = torch.cat((self.state_history, self.state_post), axis=1)


    def filtering(self, observation):
        # observation: column vector

        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        x_last = self.state_post
        x_predict = self.GSSModel.f(x_last)

        if self.dnn_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()

        y_predict = self.GSSModel.g(x_predict)
        residual = observation - y_predict

        ## input 1: x_{k-1 | k-1} - x_{k-1 | k-2}
        state_inno = self.state_post_past - self.state_pred_past
        ## input 2: residual
        ## input 3: x_k - x_{k-1}
        diff_state = self.state_post - self.state_post_past
        ## input 4: y_k - y_{k-1}
        diff_obs = observation - self.obs_past

        K_gain = self.kf_net(diff_obs, residual, diff_state, state_inno)
        
        # state_inno_in = F.normalize(state_inno, p=2, dim=0, eps=1e-12)
        # residual_in = F.normalize(residual, p=2, dim=0, eps=1e-12)
        # diff_state_in = F.normalize(diff_state, p=2, dim=0, eps=1e-12)
        # diff_obs_in = F.normalize(diff_obs, p=2, dim=0, eps=1e-12)
        # # residual_in = residual
        # # diff_obs_in = diff_obs
        # K_gain = self.kf_net(state_inno_in, residual_in, diff_state_in, diff_obs_in)  

        x_post = x_predict + (K_gain @ residual)

        self.dnn_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), axis=1)



class Split_KalmanNet_Filter():
    def __init__(self, GSSModel:GSSModel):
        
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.kf_net = DNN_SKalmanNet_GSS(self.x_dim, self.y_dim)
        self.init_state = GSSModel.init_state
        self.reset(clean_history=True)
  

    def reset(self, clean_history=False):
        self.dnn_first = True
        self.kf_net.initialize_hidden()
        self.state_post = self.init_state.detach().clone()
        if clean_history:
            self.state_history = self.init_state.detach().clone()        
        self.state_history = torch.cat((self.state_history, self.state_post), axis=1)


    def filtering(self, observation):
        # observation: column vector

        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        x_last = self.state_post
        x_predict = self.GSSModel.f(x_last)

        if self.dnn_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()

        y_predict = self.GSSModel.g(x_predict)
        residual = observation - y_predict

        ## input 1: x_{k-1 | k-1} - x_{k-1 | k-2}
        state_inno = self.state_post_past - self.state_pred_past
        ## input 2: residual
        ## input 3: x_k - x_{k-1}
        diff_state = self.state_post - self.state_post_past
        ## input 4: y_k - y_{k-1}
        diff_obs = observation - self.obs_past
        ## input 6: Jacobian
        H_jacob = self.GSSModel.Jacobian_g(x_predict)     
        ## input 5: linearization error
        # linearization_error = H_jacob@x_predict
        linearization_error = y_predict - H_jacob@x_predict

        # Reshape inputs for DNN: (Batch=1, dim)
        # Column vectors (dim, 1) need to be transposed to (1, dim)
        state_inno_in = state_inno.T if state_inno.dim() == 2 else state_inno.unsqueeze(0)
        residual_in = residual.T if residual.dim() == 2 else residual.unsqueeze(0)
        diff_state_in = diff_state.T if diff_state.dim() == 2 else diff_state.unsqueeze(0)
        diff_obs_in = diff_obs.T if diff_obs.dim() == 2 else diff_obs.unsqueeze(0)
        linearization_error_in = linearization_error.T if linearization_error.dim() == 2 else linearization_error.unsqueeze(0)
        H_jacob_in = H_jacob.reshape((1, -1))  # (1, x_dim*y_dim)
        
        (Pk, Sk) = self.kf_net(state_inno_in, residual_in, diff_state_in, diff_obs_in, linearization_error_in, H_jacob_in)

        # Reshape Pk and Sk from (1, dim*dim) to (dim, dim)
        Pk = Pk.reshape((self.x_dim, self.x_dim))
        Sk = Sk.reshape((self.y_dim, self.y_dim))

        # KEY FORMULA: Kalman gain from SKNet outputs
        # K = Pk @ H^T @ Sk
        K_gain = Pk @ torch.transpose(H_jacob, 0, 1) @ Sk

        # State update: delta_x = K @ r, x_post = x_pred + delta_x
        x_post = x_predict + (K_gain @ residual)

        self.dnn_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), axis=1)


class Dual_Mode_Filter():
    """
    Dual-mode filter that runs both baseline EKF and SKNet in parallel.
    
    This allows direct comparison of both methods on the same data,
    producing two independent closed-loop trajectories.
    
    The SKNet-based update uses:
        K = Pk @ H^T @ Sk
        delta_x = K @ r
        x_post = x_pred + delta_x
    
    This replaces the traditional:
        S = H @ P @ H^T + R
        K = P @ H^T @ S^{-1}
    """
    
    def __init__(self, GSSModel: GSSModel, collect_training_data=True):
        """
        Initialize dual-mode filter.
        
        Args:
            GSSModel: State space model
            collect_training_data: Whether to collect data for SKNet training
        """
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel
        
        # Initialize baseline EKF
        self.ekf = Extended_Kalman_Filter(GSSModel)
        
        # Initialize SKNet filter
        self.sknet = Split_KalmanNet_Filter(GSSModel)
        
        # Training data collection
        self.collect_training_data = collect_training_data
        self.training_data = []
        
        # Track both trajectories
        self.baseline_history = []
        self.sknet_history = []
        
    def reset(self, clean_history=False):
        """Reset both filters."""
        self.ekf.reset(clean_history)
        self.sknet.reset(clean_history)
        if clean_history:
            self.training_data = []
            self.baseline_history = []
            self.sknet_history = []
    
    def filtering(self, observation):
        """
        Run filtering with both baseline and SKNet modes.
        
        Args:
            observation: Observation vector (column vector)
            
        Returns:
            results: Dictionary with 'baseline' and 'sknet' state estimates
        """
        results = {}
        
        # Run baseline EKF
        self.ekf.filtering(observation)
        results['baseline'] = {
            'state': self.ekf.state_post.detach().clone(),
            'cov': self.ekf.cov_post.detach().clone() if hasattr(self.ekf, 'cov_post') else None
        }
        self.baseline_history.append(self.ekf.state_post.detach().clone())
        
        # Collect training data before SKNet update
        if self.collect_training_data:
            self._collect_training_sample(observation)
        
        # Run SKNet filter
        self.sknet.filtering(observation)
        results['sknet'] = {
            'state': self.sknet.state_post.detach().clone()
        }
        self.sknet_history.append(self.sknet.state_post.detach().clone())
        
        return results
    
    def _collect_training_sample(self, observation):
        """
        Collect training data from baseline EKF for SKNet training.
        
        The collected data includes:
        - H: Jacobian matrix
        - r: Residual vector
        - P_active: EKF predicted covariance (teacher signal for Pk)
        - S: Innovation covariance (teacher signal for Sk^{-1})
        - Feature inputs for SKNet
        """
        # Get values from EKF
        x_predict = self.GSSModel.f(self.ekf.state_post)
        y_predict = self.GSSModel.g(x_predict)
        residual = observation - y_predict
        
        H_jacob = self.GSSModel.Jacobian_g(x_predict)
        F_jacob = self.GSSModel.Jacobian_f(self.ekf.state_post)
        
        # Predicted covariance
        P_pred = (F_jacob @ self.ekf.cov_post @ torch.transpose(F_jacob, 0, 1)) + self.ekf.Q
        
        # Innovation covariance
        S = H_jacob @ P_pred @ torch.transpose(H_jacob, 0, 1) + self.ekf.R
        
        # Compute feature inputs
        state_inno = torch.zeros_like(self.ekf.state_post)
        diff_state = torch.zeros_like(self.ekf.state_post)
        diff_obs = torch.zeros_like(observation)
        lin_error = y_predict - H_jacob @ x_predict
        
        if hasattr(self.sknet, 'state_post_past') and hasattr(self.sknet, 'state_pred_past'):
            state_inno = self.sknet.state_post_past - self.sknet.state_pred_past
            diff_state = self.sknet.state_post - self.sknet.state_post_past
        if hasattr(self.sknet, 'obs_past'):
            diff_obs = observation - self.sknet.obs_past
        
        sample = {
            'H': H_jacob.detach().clone(),
            'r': residual.detach().clone(),
            'state_pred': x_predict.detach().clone(),
            'P_active': P_pred.detach().clone(),
            'S': S.detach().clone(),
            'state_inno': state_inno.detach().clone(),
            'diff_state': diff_state.detach().clone(),
            'diff_obs': diff_obs.detach().clone(),
            'lin_error': lin_error.detach().clone(),
            'H_flat': H_jacob.reshape(-1).detach().clone()
        }
        self.training_data.append(sample)
    
    def get_training_data(self):
        """Get collected training data."""
        return self.training_data
    
    def get_trajectories(self):
        """Get both trajectories."""
        return {
            'baseline': torch.stack(self.baseline_history, dim=1) if self.baseline_history else None,
            'sknet': torch.stack(self.sknet_history, dim=1) if self.sknet_history else None
        }


class MSCKF_SKNet_Filter():
    """
    MSCKF-specific SKNet filter that replaces the EKF update with SKNet.
    
    This class is designed to work with variable-dimensional states
    that occur in MSCKF (where state dimension changes as camera states
    are added/removed).
    
    Key difference from Split_KalmanNet_Filter:
    - Handles variable state dimensions
    - Works with external H and r (from MSCKF feature processing)
    - Uses external state prediction (from IMU propagation)
    """
    
    def __init__(self, max_state_dim=147, max_obs_dim=100, device='cpu'):
        """
        Initialize MSCKF-specific SKNet filter.
        
        Args:
            max_state_dim: Maximum state dimension (for padding)
            max_obs_dim: Maximum observation dimension (for padding)
            device: Device for computation
        """
        self.max_state_dim = max_state_dim
        self.max_obs_dim = max_obs_dim
        self.device = torch.device(device)
        
        # Initialize SKNet
        self.kf_net = DNN_SKalmanNet_GSS(x_dim=max_state_dim, y_dim=max_obs_dim)
        self.kf_net.to(self.device)
        
        self.reset()
    
    def reset(self):
        """Reset filter state."""
        self.kf_net.initialize_hidden()
        self.state_post_past = None
        self.state_pred_past = None
        self.obs_past = None
        self.is_first = True
    
    def compute_kalman_gain(self, H, r, state_pred, P_pred=None):
        """
        Compute Kalman gain using SKNet.
        
        Args:
            H: Measurement Jacobian (numpy array)
            r: Measurement residual (numpy array)
            state_pred: Predicted state (numpy array)
            P_pred: Predicted covariance (optional, for reference)
            
        Returns:
            K: Kalman gain matrix (numpy array)
            Pk: State covariance from SKNet (numpy array)
            Sk: Innovation factor from SKNet (numpy array)
        """
        # Pad inputs to fixed dimensions
        H_padded = self._pad_matrix(H, self.max_obs_dim, self.max_state_dim)
        r_padded = self._pad_vector(r, self.max_obs_dim)
        state_padded = self._pad_vector(state_pred, self.max_state_dim)
        
        # Construct features
        if self.is_first:
            self.state_post_past = np.zeros(self.max_state_dim)
            self.state_pred_past = np.zeros(self.max_state_dim)
            self.obs_past = np.zeros(self.max_obs_dim)
            self.state_post_current = state_padded.copy()
        
        state_inno = self.state_post_past - self.state_pred_past
        diff_state = state_padded - self.state_post_past if not self.is_first else np.zeros(self.max_state_dim)
        diff_obs = r_padded - self.obs_past if not self.is_first else np.zeros(self.max_obs_dim)
        lin_error = np.zeros(self.max_obs_dim)  # Simplified
        H_flat = H_padded.flatten()
        
        # Convert to tensors
        state_inno_t = torch.tensor(state_inno, dtype=torch.float32).unsqueeze(0).to(self.device)
        r_t = torch.tensor(r_padded, dtype=torch.float32).unsqueeze(0).to(self.device)
        diff_state_t = torch.tensor(diff_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        diff_obs_t = torch.tensor(diff_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        lin_error_t = torch.tensor(lin_error, dtype=torch.float32).unsqueeze(0).to(self.device)
        H_flat_t = torch.tensor(H_flat, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            Pk_flat, Sk_flat = self.kf_net(state_inno_t, r_t, diff_state_t, diff_obs_t, lin_error_t, H_flat_t)
        
        # Reshape outputs
        Pk = Pk_flat.squeeze(0).cpu().numpy().reshape(self.max_state_dim, self.max_state_dim)
        Sk = Sk_flat.squeeze(0).cpu().numpy().reshape(self.max_obs_dim, self.max_obs_dim)
        
        # Compute Kalman gain: K = Pk @ H^T @ Sk
        K = Pk @ H_padded.T @ Sk
        
        # Extract relevant portion
        actual_state_dim = len(state_pred)
        actual_obs_dim = len(r)
        K_actual = K[:actual_state_dim, :actual_obs_dim]
        Pk_actual = Pk[:actual_state_dim, :actual_state_dim]
        Sk_actual = Sk[:actual_obs_dim, :actual_obs_dim]
        
        # Update history
        self._update_history(state_padded, r_padded)
        
        return K_actual, Pk_actual, Sk_actual
    
    def _pad_vector(self, v, target_len):
        """Pad vector to target length."""
        v = np.array(v).flatten()
        if len(v) >= target_len:
            return v[:target_len]
        return np.pad(v, (0, target_len - len(v)))
    
    def _pad_matrix(self, M, target_rows, target_cols):
        """Pad matrix to target dimensions."""
        M = np.array(M)
        result = np.zeros((target_rows, target_cols))
        rows = min(M.shape[0], target_rows)
        cols = min(M.shape[1], target_cols)
        result[:rows, :cols] = M[:rows, :cols]
        return result
    
    def _update_history(self, state_current, obs_current):
        """Update history for next step."""
        if hasattr(self, 'state_post_current'):
            self.state_post_past = self.state_post_current.copy()
        else:
            self.state_post_past = np.zeros(self.max_state_dim)
        
        self.state_pred_past = state_current.copy()
        self.state_post_current = state_current.copy()
        self.obs_past = obs_current.copy()
        self.is_first = False
    
    def load_model(self, model_path):
        """Load pretrained model weights."""
        state_dict = torch.load(model_path, map_location=self.device)
        self.kf_net.load_state_dict(state_dict)
        self.kf_net.eval()
    
    def save_model(self, model_path):
        """Save model weights."""
        torch.save(self.kf_net.state_dict(), model_path)