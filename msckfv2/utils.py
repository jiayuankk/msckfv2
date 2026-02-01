"""
Utility functions for MSCKF-VIO.
Contains quaternion operations, rotation conversions, and geometric utilities.
"""
import numpy as np
from scipy.spatial.transform import Rotation


class Isometry3d:
    """
    Represents a rigid body transformation in 3D (rotation + translation).
    """
    def __init__(self, R=np.identity(3), t=np.zeros(3)):
        """
        Args:
            R: rotation matrix (3x3)
            t: translation vector (3,)
        """
        self.R = R
        self.t = t

    def __mul__(self, other):
        """
        Compose two Isometry3d transformations.
        """
        if isinstance(other, Isometry3d):
            return Isometry3d(
                self.R @ other.R,
                self.R @ other.t + self.t)
        elif isinstance(other, np.ndarray):
            # Transform a point
            return self.R @ other + self.t
        else:
            raise TypeError(f"Cannot multiply Isometry3d with {type(other)}")

    def inverse(self):
        """
        Return the inverse of this transformation.
        """
        R_inv = self.R.T
        return Isometry3d(R_inv, -R_inv @ self.t)

    def matrix(self):
        """
        Return the 4x4 homogeneous transformation matrix.
        """
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T


def to_rotation(q):
    """
    Convert a unit quaternion to a rotation matrix.
    
    Args:
        q: quaternion in [w, x, y, z] order (Hamilton convention)
        
    Returns:
        R: 3x3 rotation matrix
    """
    q = np.array(q).flatten()
    if len(q) != 4:
        raise ValueError(f"Expected quaternion with 4 elements, got {len(q)}")
    
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    
    # Extract components
    w, x, y, z = q
    
    # Compute rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R


def to_quaternion(R):
    """
    Convert a rotation matrix to a unit quaternion.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        q: quaternion in [w, x, y, z] order (Hamilton convention)
    """
    # Use scipy for numerical stability
    rot = Rotation.from_matrix(R)
    q = rot.as_quat()  # Returns [x, y, z, w]
    # Convert to [w, x, y, z]
    return np.array([q[3], q[0], q[1], q[2]])


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    
    Args:
        q1, q2: quaternions in [w, x, y, z] order
        
    Returns:
        result: quaternion product q1 * q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_normalize(q):
    """
    Normalize a quaternion to unit length.
    
    Args:
        q: quaternion
        
    Returns:
        normalized quaternion
    """
    return q / np.linalg.norm(q)


def small_angle_quaternion(dtheta):
    """
    Convert a small angle rotation vector to a quaternion.
    
    Args:
        dtheta: rotation vector (3,)
        
    Returns:
        q: quaternion [w, x, y, z]
    """
    dq = dtheta / 2.0
    norm_sq = np.dot(dq, dq)
    
    if norm_sq < 1.0:
        q = np.array([np.sqrt(1.0 - norm_sq), dq[0], dq[1], dq[2]])
    else:
        q = np.array([1.0, dq[0], dq[1], dq[2]])
        q = q / np.linalg.norm(q)
    
    return q


def skew(v):
    """
    Create a skew-symmetric matrix from a vector.
    
    Args:
        v: 3D vector
        
    Returns:
        3x3 skew-symmetric matrix
    """
    v = np.array(v).flatten()
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def rodrigues(r, dt):
    """
    Compute the rotation matrix from a rotation vector using Rodrigues' formula.
    
    Args:
        r: angular velocity vector (3,)
        dt: time interval
        
    Returns:
        R: rotation matrix
    """
    theta = np.linalg.norm(r) * dt
    if theta < 1e-10:
        return np.eye(3)
    
    k = r / np.linalg.norm(r)
    K = skew(k)
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def chi_square_test(r, S, dof, confidence=0.95):
    """
    Perform chi-square test for outlier detection.
    
    Args:
        r: residual vector
        S: innovation covariance matrix
        dof: degrees of freedom
        confidence: confidence level
        
    Returns:
        True if the residual passes the test (is an inlier)
    """
    # Chi-square critical values at 95% confidence
    chi_square_table = {
        1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.07,
        6: 12.59, 7: 14.07, 8: 15.51, 9: 16.92, 10: 18.31
    }
    
    # For dof > 10, use approximation
    if dof > 10:
        threshold = dof + 2 * np.sqrt(2 * dof)
    else:
        threshold = chi_square_table.get(dof, dof + 2 * np.sqrt(2 * dof))
    
    try:
        mahalanobis_sq = r.T @ np.linalg.solve(S, r)
    except np.linalg.LinAlgError:
        return False
    
    return mahalanobis_sq < threshold


def compute_state_dim(num_cam_states):
    """
    Compute the total state dimension for MSCKF.
    
    IMU state: 21 dimensions
        - orientation (4) stored as quaternion but error state is 3
        - position (3)
        - velocity (3)
        - gyro bias (3)
        - acc bias (3)
        - extrinsic rotation (4->3)
        - extrinsic translation (3)
    
    Error state: 21 dimensions (quaternion errors are 3D)
    Camera state: 6 dimensions each (orientation error 3 + position 3)
    
    Args:
        num_cam_states: number of camera states
        
    Returns:
        state_dim: total error state dimension
    """
    IMU_STATE_DIM = 21  # Error state dimension for IMU
    CAM_STATE_DIM = 6   # Error state dimension per camera
    return IMU_STATE_DIM + num_cam_states * CAM_STATE_DIM


# IMU state indices (for error state)
class IMUStateIndex:
    # Orientation error (3)
    ORI = 0
    # Position (3)
    POS = 3
    # Velocity (3)
    VEL = 6
    # Gyroscope bias (3)
    BG = 9
    # Accelerometer bias (3)
    BA = 12
    # Extrinsic rotation error (3)
    EXT_ROT = 15
    # Extrinsic translation (3)
    EXT_TRANS = 18
    # Total IMU error state dimension
    DIM = 21


class CamStateIndex:
    # Orientation error (3)
    ORI = 0
    # Position (3)
    POS = 3
    # Dimension per camera state
    DIM = 6
