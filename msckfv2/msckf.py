"""
Multi-State Constraint Kalman Filter (MSCKF) for Visual-Inertial Odometry.
Implements both baseline MSCKF and MSCKF+SKNet fusion modes.

The MSCKF algorithm fuses IMU measurements with visual feature observations
to estimate the pose and velocity of the IMU frame.
"""
import numpy as np
from collections import namedtuple
from scipy.linalg import null_space

from utils import (
    Isometry3d, to_rotation, to_quaternion, quaternion_multiply,
    quaternion_normalize, small_angle_quaternion, skew, rodrigues,
    chi_square_test, IMUStateIndex, CamStateIndex
)
from feature import Feature


class IMUState:
    """
    State of the IMU including orientation, position, velocity, and biases.
    """
    # Gravity vector
    gravity = np.array([0., 0., -9.81])
    
    def __init__(self, state_id=0):
        self.id = state_id
        
        # Time when the state is recorded
        self.timestamp = 0.0
        
        # Orientation (quaternion [w, x, y, z])
        # Takes a vector from world frame to IMU frame
        self.orientation = np.array([1.0, 0., 0., 0.])
        
        # Position of the IMU frame in the world frame
        self.position = np.zeros(3)
        
        # Velocity of the IMU frame in the world frame
        self.velocity = np.zeros(3)
        
        # Biases
        self.gyro_bias = np.zeros(3)
        self.acc_bias = np.zeros(3)
        
        # Transformation from IMU to camera
        self.R_imu_cam0 = np.eye(3)
        self.t_cam0_imu = np.zeros(3)
        
        # Process noise
        self.gyro_noise = 0.005 ** 2
        self.acc_noise = 0.05 ** 2
        self.gyro_bias_noise = 0.001 ** 2
        self.acc_bias_noise = 0.01 ** 2


class CamState:
    """
    State of a camera, including orientation and position.
    """
    def __init__(self, state_id=0):
        self.id = state_id
        self.timestamp = 0.0
        
        # Orientation (quaternion [w, x, y, z])
        # Takes a vector from world frame to camera frame
        self.orientation = np.array([1.0, 0., 0., 0.])
        
        # Position of the camera in the world frame
        self.position = np.zeros(3)
        
        # Extrinsic calibration (for clone)
        self.R_imu_cam0 = np.eye(3)
        self.t_cam0_imu = np.zeros(3)


class MSCKF:
    """
    Multi-State Constraint Kalman Filter implementation.
    
    Supports two modes:
    - mode='baseline': Traditional MSCKF using EKF update equations
    - mode='sknet': Uses SKNet to compute Pk and Sk for Kalman gain
    """
    
    def __init__(self, config, mode='baseline', sknet_adapter=None):
        """
        Initialize MSCKF.
        
        Args:
            config: Configuration object with calibration and filter parameters
            mode: 'baseline' for traditional MSCKF, 'sknet' for SKNet fusion
            sknet_adapter: SKNet adapter instance (required if mode='sknet')
        """
        self.config = config
        self.mode = mode
        self.sknet_adapter = sknet_adapter
        
        if mode == 'sknet' and sknet_adapter is None:
            raise ValueError("sknet_adapter required for mode='sknet'")
        
        # IMU state
        self.imu_state = IMUState()
        self._init_imu_state()
        
        # Camera states {id: CamState}
        self.cam_states = {}
        
        # State covariance matrix
        self.state_cov = np.zeros((IMUStateIndex.DIM, IMUStateIndex.DIM))
        self._init_state_covariance()
        
        # Map of features {id: Feature}
        self.map_server = {}
        
        # Tracking state ID
        self.state_server_id = 0
        
        # Feature calibration
        Feature.R_cam0_cam1 = config.T_cn_cnm1[:3, :3]
        Feature.t_cam0_cam1 = config.T_cn_cnm1[:3, 3]
        
        # For trajectory output
        self.position_history = []
        self.orientation_history = []
        self.timestamp_history = []
        
        # For training data collection
        self.collect_training_data = False
        self.training_data = []
        
        # Store observation noise variance
        self.observation_noise = config.observation_noise
        
        # Online reset flag
        self.is_first_frame = True
        self.first_observation = None
        
    def _init_imu_state(self):
        """Initialize IMU state from configuration."""
        config = self.config
        
        # Set noise parameters
        self.imu_state.gyro_noise = config.gyro_noise
        self.imu_state.acc_noise = config.acc_noise
        self.imu_state.gyro_bias_noise = config.gyro_bias_noise
        self.imu_state.acc_bias_noise = config.acc_bias_noise
        
        # Set extrinsic transformation
        T_imu_cam0 = config.T_imu_cam0
        self.imu_state.R_imu_cam0 = T_imu_cam0[:3, :3]
        self.imu_state.t_cam0_imu = T_imu_cam0[:3, 3]
        
        # Set initial velocity
        self.imu_state.velocity = config.velocity.copy()
        
        # Set gravity
        IMUState.gravity = config.gravity
        
    def _init_state_covariance(self):
        """Initialize state covariance matrix."""
        config = self.config
        
        # Initial covariance for orientation (small uncertainty)
        self.state_cov[0:3, 0:3] = np.eye(3) * 1e-6
        
        # Initial covariance for position (small uncertainty)
        self.state_cov[3:6, 3:6] = np.eye(3) * 1e-6
        
        # Initial covariance for velocity
        self.state_cov[6:9, 6:9] = np.eye(3) * config.velocity_cov
        
        # Initial covariance for gyro bias
        self.state_cov[9:12, 9:12] = np.eye(3) * config.gyro_bias_cov
        
        # Initial covariance for acc bias
        self.state_cov[12:15, 12:15] = np.eye(3) * config.acc_bias_cov
        
        # Initial covariance for extrinsic rotation
        self.state_cov[15:18, 15:18] = np.eye(3) * config.extrinsic_rotation_cov
        
        # Initial covariance for extrinsic translation
        self.state_cov[18:21, 18:21] = np.eye(3) * config.extrinsic_translation_cov

    def initialize_gravity_and_bias(self, imu_data):
        """
        Initialize gravity vector and IMU biases using stationary IMU data.
        
        Args:
            imu_data: List of IMU measurements during stationary period
        """
        if len(imu_data) < 200:
            return False
        
        # Compute mean acceleration and angular velocity
        sum_angular_vel = np.zeros(3)
        sum_linear_acc = np.zeros(3)
        
        for msg in imu_data:
            sum_angular_vel += msg.angular_velocity
            sum_linear_acc += msg.linear_acceleration
            
        gyro_mean = sum_angular_vel / len(imu_data)
        acc_mean = sum_linear_acc / len(imu_data)
        
        # Initialize gyro bias
        self.imu_state.gyro_bias = gyro_mean
        
        # Compute gravity direction in IMU frame
        gravity_norm = np.linalg.norm(acc_mean)
        gravity_imu = acc_mean / gravity_norm
        
        # Compute initial orientation (align gravity)
        IMUState.gravity = np.array([0, 0, -gravity_norm])
        
        # Find rotation that transforms gravity_imu to [0, 0, -1]
        target = np.array([0, 0, -1])
        
        if np.allclose(gravity_imu, target):
            self.imu_state.orientation = np.array([1., 0., 0., 0.])
        elif np.allclose(gravity_imu, -target):
            self.imu_state.orientation = np.array([0., 1., 0., 0.])
        else:
            axis = np.cross(gravity_imu, target)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.clip(np.dot(gravity_imu, target), -1, 1))
            
            w = np.cos(angle / 2)
            xyz = axis * np.sin(angle / 2)
            self.imu_state.orientation = np.array([w, xyz[0], xyz[1], xyz[2]])
        
        return True

    def process_imu(self, imu_msg):
        """
        Process a single IMU measurement.
        
        This implements IMU state propagation using the continuous-discrete
        extended Kalman filter.
        
        Args:
            imu_msg: IMU measurement with angular_velocity and linear_acceleration
        """
        dt = imu_msg.timestamp - self.imu_state.timestamp
        if dt <= 0:
            return
            
        # Get current IMU state
        gyro = imu_msg.angular_velocity - self.imu_state.gyro_bias
        acc = imu_msg.linear_acceleration - self.imu_state.acc_bias
        
        # State propagation using midpoint integration
        self._predict_state(gyro, acc, dt)
        
        # Covariance propagation
        self._predict_covariance(gyro, acc, dt)
        
        # Update timestamp
        self.imu_state.timestamp = imu_msg.timestamp

    def _predict_state(self, gyro, acc, dt):
        """
        Propagate IMU state using midpoint integration.
        """
        # Get rotation from quaternion
        R_w_imu = to_rotation(self.imu_state.orientation).T
        
        # Compute rotation increment
        dtheta = gyro * dt
        dR = rodrigues(gyro, dt)
        
        # Update orientation
        dq = small_angle_quaternion(dtheta)
        self.imu_state.orientation = quaternion_normalize(
            quaternion_multiply(self.imu_state.orientation, dq))
        
        # Get average rotation for velocity/position update
        R_w_imu_new = to_rotation(self.imu_state.orientation).T
        R_avg = (R_w_imu + R_w_imu_new) / 2
        
        # Compute acceleration in world frame
        acc_world = R_avg @ acc + IMUState.gravity
        
        # Update velocity
        self.imu_state.velocity += acc_world * dt
        
        # Update position
        self.imu_state.position += self.imu_state.velocity * dt + \
                                   0.5 * acc_world * dt * dt

    def _predict_covariance(self, gyro, acc, dt):
        """
        Propagate state covariance matrix.
        """
        R_w_imu = to_rotation(self.imu_state.orientation).T
        
        # Continuous-time state transition matrix (linearized)
        Phi = np.zeros((IMUStateIndex.DIM, IMUStateIndex.DIM))
        
        # d(theta)/d(theta)
        Phi[0:3, 0:3] = -skew(gyro)
        
        # d(theta)/d(bg)
        Phi[0:3, 9:12] = -np.eye(3)
        
        # d(v)/d(theta)
        Phi[6:9, 0:3] = -R_w_imu @ skew(acc)
        
        # d(v)/d(ba)
        Phi[6:9, 12:15] = -R_w_imu
        
        # d(p)/d(v)
        Phi[3:6, 6:9] = np.eye(3)
        
        # Discretize
        Phi = np.eye(IMUStateIndex.DIM) + Phi * dt
        
        # Process noise covariance
        G = np.zeros((IMUStateIndex.DIM, 12))
        G[0:3, 0:3] = -np.eye(3)
        G[6:9, 3:6] = -R_w_imu
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)
        
        # Continuous noise covariance
        Qc = np.zeros((12, 12))
        Qc[0:3, 0:3] = np.eye(3) * self.imu_state.gyro_noise
        Qc[3:6, 3:6] = np.eye(3) * self.imu_state.acc_noise
        Qc[6:9, 6:9] = np.eye(3) * self.imu_state.gyro_bias_noise
        Qc[9:12, 9:12] = np.eye(3) * self.imu_state.acc_bias_noise
        
        # Discrete process noise
        Qd = Phi @ G @ Qc @ G.T @ Phi.T * dt
        
        # Propagate covariance
        state_dim = self.state_cov.shape[0]
        if state_dim > IMUStateIndex.DIM:
            # There are camera states
            P_ii = self.state_cov[:IMUStateIndex.DIM, :IMUStateIndex.DIM]
            P_ic = self.state_cov[:IMUStateIndex.DIM, IMUStateIndex.DIM:]
            
            self.state_cov[:IMUStateIndex.DIM, :IMUStateIndex.DIM] = \
                Phi @ P_ii @ Phi.T + Qd
            self.state_cov[:IMUStateIndex.DIM, IMUStateIndex.DIM:] = \
                Phi @ P_ic
            self.state_cov[IMUStateIndex.DIM:, :IMUStateIndex.DIM] = \
                self.state_cov[:IMUStateIndex.DIM, IMUStateIndex.DIM:].T
        else:
            self.state_cov = Phi @ self.state_cov @ Phi.T + Qd
        
        # Ensure symmetry
        self.state_cov = (self.state_cov + self.state_cov.T) / 2

    def state_augmentation(self, timestamp):
        """
        Add a new camera state to the filter.
        
        This is called when a new image frame is received.
        
        Args:
            timestamp: timestamp of the new camera state
        """
        R_w_imu = to_rotation(self.imu_state.orientation).T
        R_imu_cam0 = self.imu_state.R_imu_cam0
        t_cam0_imu = self.imu_state.t_cam0_imu
        
        # Compute camera orientation and position in world frame
        R_w_cam0 = R_w_imu @ R_imu_cam0.T
        t_cam0_w = self.imu_state.position + R_w_imu @ t_cam0_imu
        
        # Create new camera state
        cam_state = CamState(self.state_server_id)
        cam_state.timestamp = timestamp
        cam_state.orientation = to_quaternion(R_w_cam0.T)
        cam_state.position = t_cam0_w
        cam_state.R_imu_cam0 = R_imu_cam0.copy()
        cam_state.t_cam0_imu = t_cam0_imu.copy()
        
        self.cam_states[self.state_server_id] = cam_state
        self.state_server_id += 1
        
        # Augment covariance matrix
        state_dim = self.state_cov.shape[0]
        new_dim = state_dim + CamStateIndex.DIM
        
        # Jacobian for augmentation
        J = np.zeros((CamStateIndex.DIM, state_dim))
        
        # d(cam_ori)/d(imu_ori)
        J[0:3, 0:3] = R_imu_cam0
        
        # d(cam_ori)/d(ext_rot)
        J[0:3, 15:18] = np.eye(3)
        
        # d(cam_pos)/d(imu_pos)
        J[3:6, 3:6] = np.eye(3)
        
        # d(cam_pos)/d(imu_ori)
        J[3:6, 0:3] = skew(R_w_imu @ t_cam0_imu)
        
        # d(cam_pos)/d(ext_trans)
        J[3:6, 18:21] = R_w_imu
        
        # Augment covariance
        new_cov = np.zeros((new_dim, new_dim))
        new_cov[:state_dim, :state_dim] = self.state_cov
        new_cov[state_dim:, :state_dim] = J @ self.state_cov
        new_cov[:state_dim, state_dim:] = new_cov[state_dim:, :state_dim].T
        new_cov[state_dim:, state_dim:] = J @ self.state_cov @ J.T
        
        self.state_cov = new_cov
        
        # Ensure symmetry
        self.state_cov = (self.state_cov + self.state_cov.T) / 2

    def add_feature_observations(self, feature_msg):
        """
        Add feature observations from a new image.
        
        Args:
            feature_msg: Feature message containing tracked features
        """
        curr_state_id = self.state_server_id - 1
        
        for feature in feature_msg.features:
            if feature.id not in self.map_server:
                # New feature
                new_feature = Feature(feature.id, self.config.optimization_config)
                self.map_server[feature.id] = new_feature
            
            # Add observation
            self.map_server[feature.id].observations[curr_state_id] = np.array([
                feature.u0, feature.v0, feature.u1, feature.v1
            ])

    def measurement_update(self, feature_msg):
        """
        Perform MSCKF measurement update.
        
        This is the main update function that processes feature observations
        and updates the state estimate.
        
        Args:
            feature_msg: Feature message containing tracked features
            
        Returns:
            H: Jacobian matrix (for training data collection)
            r: Residual vector (for training data collection)
            state_pred: Predicted state before update
        """
        # Add new observations
        self.add_feature_observations(feature_msg)
        
        # Find features that have been lost
        curr_feature_ids = set([f.id for f in feature_msg.features])
        features_to_remove = []
        
        for feature_id in self.map_server:
            if feature_id not in curr_feature_ids:
                features_to_remove.append(feature_id)
        
        # Also remove features observed in removed camera states
        oldest_cam_id = min(self.cam_states.keys()) if self.cam_states else 0
        
        # Process lost features
        jacobian_rows = []
        residual_rows = []
        
        for feature_id in features_to_remove:
            feature = self.map_server[feature_id]
            
            # Skip features with too few observations
            if len(feature.observations) < 3:
                del self.map_server[feature_id]
                continue
            
            # Check if feature has enough motion
            if not feature.check_motion(self.cam_states):
                del self.map_server[feature_id]
                continue
            
            # Initialize feature position
            if not feature.initialize_position(self.cam_states):
                del self.map_server[feature_id]
                continue
            
            # Compute Jacobian and residual
            H_j, r_j = self._compute_feature_jacobian(feature)
            
            if H_j is not None and H_j.shape[0] > 0:
                jacobian_rows.append(H_j)
                residual_rows.append(r_j)
            
            del self.map_server[feature_id]
        
        # Return early if no valid measurements
        if len(jacobian_rows) == 0:
            return None, None, None
        
        # Stack all Jacobians and residuals
        H = np.vstack(jacobian_rows)
        r = np.hstack(residual_rows)
        
        # Store predicted state for training
        state_pred = self._get_state_vector()
        
        # Perform update based on mode
        if self.mode == 'baseline':
            self._ekf_update(H, r)
        elif self.mode == 'sknet':
            self._sknet_update(H, r, state_pred)
        
        # Record trajectory
        self.position_history.append(self.imu_state.position.copy())
        self.orientation_history.append(self.imu_state.orientation.copy())
        self.timestamp_history.append(self.imu_state.timestamp)
        
        # Collect training data if enabled
        if self.collect_training_data:
            self._collect_training_sample(H, r, state_pred)
        
        return H, r, state_pred

    def _compute_feature_jacobian(self, feature):
        """
        Compute the Jacobian and residual for a single feature.
        
        Args:
            feature: Feature object with observations and 3D position
            
        Returns:
            H_j: Jacobian matrix for this feature
            r_j: Residual vector for this feature
        """
        # Get camera state IDs that observed this feature
        cam_state_ids = list(feature.observations.keys())
        
        # Compute Jacobians
        H_xj = []
        H_fj = []
        r_j = []
        
        for cam_id in cam_state_ids:
            if cam_id not in self.cam_states:
                continue
                
            cam_state = self.cam_states[cam_id]
            
            # Measurement function: project 3D point to camera
            p_w = feature.position  # 3D position in world frame
            
            R_w_c = to_rotation(cam_state.orientation)
            t_c_w = cam_state.position
            
            # Transform to camera frame
            p_c = R_w_c @ (p_w - t_c_w)
            
            if p_c[2] <= 0:
                continue
            
            # Predicted measurement
            z_hat = p_c[:2] / p_c[2]
            
            # Actual measurement (mono)
            z = feature.observations[cam_id][:2]
            
            # Residual
            r_j.append(z - z_hat)
            
            # Jacobian w.r.t. feature position
            J_p = np.zeros((2, 3))
            J_p[0, 0] = 1 / p_c[2]
            J_p[0, 2] = -p_c[0] / (p_c[2] ** 2)
            J_p[1, 1] = 1 / p_c[2]
            J_p[1, 2] = -p_c[1] / (p_c[2] ** 2)
            
            J_f = J_p @ R_w_c
            H_fj.append(J_f)
            
            # Jacobian w.r.t. camera state
            J_c_ori = J_p @ skew(p_c)
            J_c_pos = -J_p @ R_w_c
            
            H_x = np.zeros((2, self.state_cov.shape[0]))
            
            # Find index of this camera state
            cam_idx = list(self.cam_states.keys()).index(cam_id)
            state_idx = IMUStateIndex.DIM + cam_idx * CamStateIndex.DIM
            
            H_x[:, state_idx:state_idx+3] = J_c_ori
            H_x[:, state_idx+3:state_idx+6] = J_c_pos
            
            H_xj.append(H_x)
        
        if len(r_j) == 0:
            return None, None
        
        H_xj = np.vstack(H_xj)
        H_fj = np.vstack(H_fj)
        r_j = np.hstack(r_j)
        
        # Project to null space of H_fj to eliminate feature position
        A = null_space(H_fj.T)
        
        if A.shape[1] == 0:
            return None, None
        
        H_o = A.T @ H_xj
        r_o = A.T @ r_j
        
        return H_o, r_o

    def _ekf_update(self, H, r):
        """
        Standard EKF measurement update.
        
        Args:
            H: Jacobian matrix
            r: Residual vector
        """
        # Observation noise
        R = np.eye(len(r)) * self.observation_noise
        
        # Innovation covariance
        S = H @ self.state_cov @ H.T + R
        
        # Kalman gain
        K = self.state_cov @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        
        # State correction
        delta_x = K @ r
        
        # Apply correction
        self._apply_state_correction(delta_x)
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.state_cov.shape[0]) - K @ H
        self.state_cov = I_KH @ self.state_cov @ I_KH.T + K @ R @ K.T
        
        # Ensure symmetry
        self.state_cov = (self.state_cov + self.state_cov.T) / 2

    def _sknet_update(self, H, r, state_pred):
        """
        SKNet-based measurement update.
        
        Uses SKNet to predict Pk and Sk, then computes:
        K = Pk @ H^T @ Sk
        delta_x = K @ r
        
        Args:
            H: Jacobian matrix
            r: Residual vector
            state_pred: Predicted state before update
        """
        # Get Pk and Sk from SKNet
        Pk, Sk = self.sknet_adapter.forward(
            self.state_cov,  # Current covariance as context
            H,
            r,
            state_pred,
            self._get_previous_state_info()
        )
        
        # Compute Kalman gain: K = Pk @ H^T @ Sk
        K = Pk @ H.T @ Sk
        
        # State correction
        delta_x = K @ r
        
        # Apply correction
        self._apply_state_correction(delta_x)
        
        # Update covariance using Pk (SKNet's estimate)
        # P_post = (I - K @ H) @ P_pred
        I_KH = np.eye(self.state_cov.shape[0]) - K @ H
        self.state_cov = I_KH @ Pk
        
        # Ensure symmetry and positive semi-definiteness
        self.state_cov = (self.state_cov + self.state_cov.T) / 2
        
        # Fix any negative eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(self.state_cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        self.state_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def _apply_state_correction(self, delta_x):
        """
        Apply state correction to IMU and camera states.
        
        Args:
            delta_x: State correction vector
        """
        # IMU orientation correction
        dtheta = delta_x[IMUStateIndex.ORI:IMUStateIndex.ORI+3]
        dq = small_angle_quaternion(dtheta)
        self.imu_state.orientation = quaternion_normalize(
            quaternion_multiply(dq, self.imu_state.orientation))
        
        # IMU position correction
        self.imu_state.position += delta_x[IMUStateIndex.POS:IMUStateIndex.POS+3]
        
        # IMU velocity correction
        self.imu_state.velocity += delta_x[IMUStateIndex.VEL:IMUStateIndex.VEL+3]
        
        # Gyro bias correction
        self.imu_state.gyro_bias += delta_x[IMUStateIndex.BG:IMUStateIndex.BG+3]
        
        # Acc bias correction
        self.imu_state.acc_bias += delta_x[IMUStateIndex.BA:IMUStateIndex.BA+3]
        
        # Extrinsic rotation correction
        dtheta_ext = delta_x[IMUStateIndex.EXT_ROT:IMUStateIndex.EXT_ROT+3]
        R_ext_correction = rodrigues(dtheta_ext / np.linalg.norm(dtheta_ext) 
                                     if np.linalg.norm(dtheta_ext) > 1e-10 else np.array([0, 0, 1]),
                                     np.linalg.norm(dtheta_ext))
        self.imu_state.R_imu_cam0 = R_ext_correction @ self.imu_state.R_imu_cam0
        
        # Extrinsic translation correction
        self.imu_state.t_cam0_imu += delta_x[IMUStateIndex.EXT_TRANS:IMUStateIndex.EXT_TRANS+3]
        
        # Camera state corrections
        cam_ids = list(self.cam_states.keys())
        for i, cam_id in enumerate(cam_ids):
            cam_state = self.cam_states[cam_id]
            state_idx = IMUStateIndex.DIM + i * CamStateIndex.DIM
            
            # Orientation correction
            dtheta_cam = delta_x[state_idx:state_idx+3]
            dq_cam = small_angle_quaternion(dtheta_cam)
            cam_state.orientation = quaternion_normalize(
                quaternion_multiply(dq_cam, cam_state.orientation))
            
            # Position correction
            cam_state.position += delta_x[state_idx+3:state_idx+6]

    def _get_state_vector(self):
        """
        Get current state as a vector for training data collection.
        
        Returns:
            state: State vector (position + velocity + orientation)
        """
        return np.concatenate([
            self.imu_state.position,
            self.imu_state.velocity,
            self.imu_state.orientation
        ])

    def _get_previous_state_info(self):
        """
        Get information about previous state for SKNet input.
        
        Returns:
            Dictionary with previous state information
        """
        return {
            'timestamp': self.imu_state.timestamp,
            'position_history': self.position_history[-10:] if len(self.position_history) >= 10 else self.position_history,
            'num_cam_states': len(self.cam_states)
        }

    def _collect_training_sample(self, H, r, state_pred):
        """
        Collect training data sample.
        
        Args:
            H: Jacobian matrix
            r: Residual vector
            state_pred: Predicted state
        """
        # Compute traditional EKF values as teacher signal
        R = np.eye(len(r)) * self.observation_noise
        S = H @ self.state_cov @ H.T + R
        P_active = self.state_cov.copy()
        
        # Compute linearization error
        lin_error = np.zeros(len(r))  # Simplified
        
        # Store training sample
        sample = {
            'H': H.copy(),
            'r': r.copy(),
            'state_pred': state_pred.copy(),
            'P_active': P_active.copy(),
            'S': S.copy(),
            'H_flat': H.flatten(),
            'lin_error': lin_error,
            'timestamp': self.imu_state.timestamp
        }
        self.training_data.append(sample)

    def prune_cam_states(self):
        """
        Remove old camera states when the buffer is full.
        """
        if len(self.cam_states) <= self.config.max_cam_state_size:
            return
        
        # Find camera states to remove
        cam_ids = list(self.cam_states.keys())
        
        # Keep keyframes and remove non-keyframes
        rm_cam_ids = []
        
        # Simple strategy: remove oldest non-keyframe states
        for i in range(len(cam_ids) - self.config.max_cam_state_size):
            rm_cam_ids.append(cam_ids[i])
        
        # Remove camera states and update covariance
        for cam_id in rm_cam_ids:
            # Find index
            cam_idx = list(self.cam_states.keys()).index(cam_id)
            state_idx = IMUStateIndex.DIM + cam_idx * CamStateIndex.DIM
            
            # Remove from covariance
            keep_indices = np.concatenate([
                np.arange(state_idx),
                np.arange(state_idx + CamStateIndex.DIM, self.state_cov.shape[0])
            ])
            self.state_cov = self.state_cov[np.ix_(keep_indices, keep_indices)]
            
            # Remove observations from features
            for feature in self.map_server.values():
                if cam_id in feature.observations:
                    del feature.observations[cam_id]
            
            # Remove camera state
            del self.cam_states[cam_id]

    def reset(self):
        """Reset the filter to initial state."""
        self.imu_state = IMUState()
        self._init_imu_state()
        self.cam_states = {}
        self.state_cov = np.zeros((IMUStateIndex.DIM, IMUStateIndex.DIM))
        self._init_state_covariance()
        self.map_server = {}
        self.position_history = []
        self.orientation_history = []
        self.timestamp_history = []
        self.training_data = []
        self.is_first_frame = True

    def get_position(self):
        """Get current position estimate."""
        return self.imu_state.position.copy()

    def get_orientation(self):
        """Get current orientation estimate (quaternion)."""
        return self.imu_state.orientation.copy()

    def get_trajectory(self):
        """Get full trajectory history."""
        return {
            'positions': np.array(self.position_history),
            'orientations': np.array(self.orientation_history),
            'timestamps': np.array(self.timestamp_history)
        }
