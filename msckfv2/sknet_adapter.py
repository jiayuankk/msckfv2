"""
SKNet Adapter for MSCKF Integration.

This module provides the interface between the MSCKF filter and the SKNet
neural network. It handles:
1. Feature construction for SKNet input
2. Pk and Sk output processing
3. Consistency between training and inference feature computation
"""
import torch
import torch.nn as nn
import numpy as np
from dnn import DNN_SKalmanNet_GSS


class SKNetAdapter:
    """
    Adapter class that interfaces SKNet with MSCKF for measurement updates.
    
    The SKNet outputs Pk (predicted state covariance) and Sk (innovation 
    covariance inverse factor) which are used to compute the Kalman gain:
    
        K = Pk @ H^T @ Sk
        delta_x = K @ r
    
    Training and inference use identical feature construction to ensure
    consistency.
    """
    
    def __init__(self, state_dim, obs_dim, model_path=None, device='cpu'):
        """
        Initialize the SKNet adapter.
        
        Args:
            state_dim: Dimension of the state error vector
            obs_dim: Dimension of the observation residual
            model_path: Path to pretrained model weights (optional)
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.device = torch.device(device)
        
        # Initialize SKNet model
        self.model = DNN_SKalmanNet_GSS(x_dim=state_dim, y_dim=obs_dim)
        self.model.to(self.device)
        
        # Load pretrained weights if available
        if model_path is not None:
            self.load_model(model_path)
        
        # Initialize history for feature computation
        self._reset_history()
        
    def _reset_history(self):
        """Reset internal state history for feature computation."""
        self.state_post_past = None
        self.state_pred_past = None
        self.obs_past = None
        self.is_first = True
        
    def reset(self):
        """Reset adapter state for a new sequence."""
        self._reset_history()
        self.model.initialize_hidden()
        
    def load_model(self, model_path):
        """
        Load pretrained model weights.
        
        Args:
            model_path: Path to saved model state dict
        """
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Loaded SKNet model from {model_path}")
        
    def save_model(self, model_path):
        """
        Save model weights.
        
        Args:
            model_path: Path to save model state dict
        """
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved SKNet model to {model_path}")
    
    def forward(self, P_pred, H, r, state_pred, prev_info):
        """
        Compute Pk and Sk using SKNet for measurement update.
        
        Args:
            P_pred: Predicted state covariance matrix (numpy array)
            H: Measurement Jacobian matrix (numpy array)
            r: Measurement residual vector (numpy array)
            state_pred: Predicted state vector (numpy array)
            prev_info: Dictionary with previous state information
            
        Returns:
            Pk: State covariance estimate from SKNet (numpy array)
            Sk: Innovation covariance inverse factor from SKNet (numpy array)
        """
        # Construct features (same as training)
        features = self._construct_features(H, r, state_pred)
        
        # Convert to tensors
        state_inno = torch.tensor(features['state_inno'], dtype=torch.float32).unsqueeze(0).to(self.device)
        residual = torch.tensor(features['residual'], dtype=torch.float32).unsqueeze(0).to(self.device)
        diff_state = torch.tensor(features['diff_state'], dtype=torch.float32).unsqueeze(0).to(self.device)
        diff_obs = torch.tensor(features['diff_obs'], dtype=torch.float32).unsqueeze(0).to(self.device)
        lin_error = torch.tensor(features['lin_error'], dtype=torch.float32).unsqueeze(0).to(self.device)
        H_flat = torch.tensor(features['H_flat'], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Forward pass through SKNet
        with torch.no_grad():
            Pk_flat, Sk_flat = self.model(
                state_inno, residual, diff_state, diff_obs, lin_error, H_flat
            )
        
        # Reshape outputs to matrices
        Pk = Pk_flat.squeeze(0).cpu().numpy().reshape(self.state_dim, self.state_dim)
        Sk = Sk_flat.squeeze(0).cpu().numpy().reshape(self.obs_dim, self.obs_dim)
        
        # Ensure Pk is symmetric positive semi-definite
        Pk = self._ensure_spd(Pk)
        
        # Update history
        self._update_history(state_pred, r)
        
        return Pk, Sk
    
    def _construct_features(self, H, r, state_pred):
        """
        Construct input features for SKNet.
        
        IMPORTANT: This function must be identical for training and inference
        to ensure consistency.
        
        Features:
        - state_inno: x_{k-1|k-1} - x_{k-1|k-2} (state innovation)
        - residual: z - h(x_pred) (observation residual)
        - diff_state: x_k - x_{k-1} (state difference)
        - diff_obs: y_k - y_{k-1} (observation difference)
        - lin_error: h(x) - H @ x (linearization error)
        - H_flat: flattened Jacobian matrix
        
        Args:
            H: Measurement Jacobian (m x n)
            r: Measurement residual (m,)
            state_pred: Predicted state (n,)
            
        Returns:
            Dictionary of feature vectors
        """
        state_dim = self.state_dim
        obs_dim = len(r) if hasattr(r, '__len__') else self.obs_dim
        
        # Pad or truncate state_pred to match state_dim
        if len(state_pred) < state_dim:
            state_pred = np.pad(state_pred, (0, state_dim - len(state_pred)))
        else:
            state_pred = state_pred[:state_dim]
        
        if self.is_first:
            # Initialize history with zeros for first step
            self.state_post_past = np.zeros(state_dim)
            self.state_pred_past = np.zeros(state_dim)
            self.obs_past = np.zeros(obs_dim)
            self.state_post_current = state_pred.copy()
        
        # Feature 1: State innovation (x_{k-1|k-1} - x_{k-1|k-2})
        state_inno = self.state_post_past - self.state_pred_past
        
        # Feature 2: Residual (already provided)
        residual = r.copy() if hasattr(r, 'copy') else np.array([r])
        
        # Feature 3: State difference (x_k - x_{k-1})
        diff_state = state_pred - self.state_post_past if not self.is_first else np.zeros(state_dim)
        
        # Feature 4: Observation difference (y_k - y_{k-1})
        diff_obs = residual - self.obs_past if not self.is_first else np.zeros(obs_dim)
        
        # Feature 5: Linearization error
        # For MSCKF, this approximates h(x) - H @ x
        # In practice, we compute it as the nonlinearity in the measurement model
        has_sufficient_dims = H.shape[1] <= len(state_pred)
        if has_sufficient_dims:
            y_pred = H @ state_pred[:H.shape[1]]
            lin_error = residual + y_pred - H @ state_pred[:H.shape[1]]
        else:
            y_pred = np.zeros(len(r))
            lin_error = np.zeros(obs_dim)
        
        # Feature 6: Flattened Jacobian
        # Pad/reshape H to match expected dimensions
        H_padded = self._pad_jacobian(H)
        H_flat = H_padded.flatten()
        
        # Ensure all features have correct dimensions
        state_inno = self._pad_or_truncate(state_inno, state_dim)
        diff_state = self._pad_or_truncate(diff_state, state_dim)
        residual = self._pad_or_truncate(residual, obs_dim)
        diff_obs = self._pad_or_truncate(diff_obs, obs_dim)
        lin_error = self._pad_or_truncate(lin_error, obs_dim)
        
        return {
            'state_inno': state_inno.astype(np.float32),
            'residual': residual.astype(np.float32),
            'diff_state': diff_state.astype(np.float32),
            'diff_obs': diff_obs.astype(np.float32),
            'lin_error': lin_error.astype(np.float32),
            'H_flat': H_flat.astype(np.float32)
        }
    
    def _pad_jacobian(self, H):
        """
        Pad or truncate Jacobian to match expected dimensions.
        
        Args:
            H: Jacobian matrix (m x n)
            
        Returns:
            Padded Jacobian (obs_dim x state_dim)
        """
        H_padded = np.zeros((self.obs_dim, self.state_dim))
        
        rows = min(H.shape[0], self.obs_dim)
        cols = min(H.shape[1], self.state_dim)
        
        H_padded[:rows, :cols] = H[:rows, :cols]
        
        return H_padded
    
    def _pad_or_truncate(self, arr, target_len):
        """
        Pad or truncate array to target length.
        
        Args:
            arr: Input array
            target_len: Target length
            
        Returns:
            Array of target length
        """
        arr = np.array(arr).flatten()
        if len(arr) < target_len:
            return np.pad(arr, (0, target_len - len(arr)))
        else:
            return arr[:target_len]
    
    def _update_history(self, state_current, obs_current):
        """
        Update history for next step's feature computation.
        
        Args:
            state_current: Current state estimate
            obs_current: Current observation residual
        """
        # Store current as past for next step
        if hasattr(self, 'state_post_current'):
            self.state_post_past = self.state_post_current.copy()
        else:
            self.state_post_past = np.zeros(self.state_dim)
            
        self.state_pred_past = self._pad_or_truncate(state_current, self.state_dim).copy()
        self.state_post_current = self._pad_or_truncate(state_current, self.state_dim).copy()
        self.obs_past = self._pad_or_truncate(obs_current, self.obs_dim).copy()
        self.is_first = False
    
    def _ensure_spd(self, P):
        """
        Ensure matrix is symmetric positive semi-definite.
        
        Args:
            P: Input matrix
            
        Returns:
            SPD matrix
        """
        # Make symmetric
        P = (P + P.T) / 2
        
        # Fix negative eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        P = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return P
    
    def train_mode(self):
        """Set model to training mode."""
        self.model.train()
        
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()


class SKNetTrainer:
    """
    Trainer class for SKNet using MSCKF-collected data.
    
    Training uses supervised learning where the teacher signals are:
    - Pk: Traditional EKF predicted covariance (P_active)
    - Sk: Inverse of innovation covariance (S^{-1})
    
    The loss encourages SKNet to match the traditional EKF behavior while
    potentially learning to handle model mismatches and non-linearities.
    """
    
    def __init__(self, sknet_adapter, learning_rate=1e-3, weight_decay=0):
        """
        Initialize trainer.
        
        Args:
            sknet_adapter: SKNetAdapter instance
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.adapter = sknet_adapter
        self.model = sknet_adapter.model
        self.device = sknet_adapter.device
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.loss_history = []
        
    def train_step(self, batch):
        """
        Single training step.
        
        Args:
            batch: Dictionary containing:
                - state_inno: (batch, state_dim)
                - residual: (batch, obs_dim)
                - diff_state: (batch, state_dim)
                - diff_obs: (batch, obs_dim)
                - lin_error: (batch, obs_dim)
                - H_flat: (batch, state_dim * obs_dim)
                - P_target: (batch, state_dim * state_dim) - Teacher signal for Pk
                - S_inv_target: (batch, obs_dim * obs_dim) - Teacher signal for Sk
                
        Returns:
            loss: Scalar loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device
        state_inno = batch['state_inno'].to(self.device)
        residual = batch['residual'].to(self.device)
        diff_state = batch['diff_state'].to(self.device)
        diff_obs = batch['diff_obs'].to(self.device)
        lin_error = batch['lin_error'].to(self.device)
        H_flat = batch['H_flat'].to(self.device)
        P_target = batch['P_target'].to(self.device)
        S_inv_target = batch['S_inv_target'].to(self.device)
        
        # Forward pass
        Pk_flat, Sk_flat = self.model(
            state_inno, residual, diff_state, diff_obs, lin_error, H_flat
        )
        
        # Compute loss
        loss_P = torch.mean((Pk_flat - P_target) ** 2)
        loss_S = torch.mean((Sk_flat - S_inv_target) ** 2)
        loss = loss_P + loss_S
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        return loss.item()
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Args:
            dataloader: PyTorch DataLoader with training data
            
        Returns:
            avg_loss: Average loss over the epoch
        """
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self, dataloader):
        """
        Validate model on validation data.
        
        Args:
            dataloader: PyTorch DataLoader with validation data
            
        Returns:
            avg_loss: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                state_inno = batch['state_inno'].to(self.device)
                residual = batch['residual'].to(self.device)
                diff_state = batch['diff_state'].to(self.device)
                diff_obs = batch['diff_obs'].to(self.device)
                lin_error = batch['lin_error'].to(self.device)
                H_flat = batch['H_flat'].to(self.device)
                P_target = batch['P_target'].to(self.device)
                S_inv_target = batch['S_inv_target'].to(self.device)
                
                Pk_flat, Sk_flat = self.model(
                    state_inno, residual, diff_state, diff_obs, lin_error, H_flat
                )
                
                loss_P = torch.mean((Pk_flat - P_target) ** 2)
                loss_S = torch.mean((Sk_flat - S_inv_target) ** 2)
                loss = loss_P + loss_S
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0


def prepare_training_data(msckf_training_data, state_dim, obs_dim):
    """
    Prepare training data from MSCKF collected samples.
    
    This function ensures consistency between training and inference by
    using the same feature construction logic.
    
    Args:
        msckf_training_data: List of dictionaries from MSCKF.training_data
        state_dim: Expected state dimension
        obs_dim: Expected observation dimension
        
    Returns:
        training_batches: List of batch dictionaries ready for training
    """
    batches = []
    
    adapter = SKNetAdapter(state_dim, obs_dim)
    
    for i, sample in enumerate(msckf_training_data):
        H = sample['H']
        r = sample['r']
        state_pred = sample['state_pred']
        P_active = sample['P_active']
        S = sample['S']
        
        # Construct features using the same logic as inference
        features = adapter._construct_features(H, r, state_pred)
        adapter._update_history(state_pred, r)
        
        # Compute teacher signals
        # Pk target: the EKF predicted covariance (padded to state_dim)
        P_target = np.zeros((state_dim, state_dim))
        P_rows = min(P_active.shape[0], state_dim)
        P_cols = min(P_active.shape[1], state_dim)
        P_target[:P_rows, :P_cols] = P_active[:P_rows, :P_cols]
        
        # Sk target: inverse of innovation covariance (padded to obs_dim)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        S_inv_target = np.zeros((obs_dim, obs_dim))
        S_rows = min(S_inv.shape[0], obs_dim)
        S_cols = min(S_inv.shape[1], obs_dim)
        S_inv_target[:S_rows, :S_cols] = S_inv[:S_rows, :S_cols]
        
        batch = {
            'state_inno': torch.tensor(features['state_inno']).unsqueeze(0),
            'residual': torch.tensor(features['residual']).unsqueeze(0),
            'diff_state': torch.tensor(features['diff_state']).unsqueeze(0),
            'diff_obs': torch.tensor(features['diff_obs']).unsqueeze(0),
            'lin_error': torch.tensor(features['lin_error']).unsqueeze(0),
            'H_flat': torch.tensor(features['H_flat']).unsqueeze(0),
            'P_target': torch.tensor(P_target.flatten(), dtype=torch.float32).unsqueeze(0),
            'S_inv_target': torch.tensor(S_inv_target.flatten(), dtype=torch.float32).unsqueeze(0)
        }
        batches.append(batch)
    
    return batches


def collate_batches(batches):
    """
    Collate multiple batches into a single batch for efficient training.
    
    Args:
        batches: List of batch dictionaries
        
    Returns:
        Collated batch dictionary
    """
    if len(batches) == 0:
        return None
    
    collated = {}
    for key in batches[0].keys():
        collated[key] = torch.cat([b[key] for b in batches], dim=0)
    
    return collated
