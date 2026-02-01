"""
Visual-Inertial Odometry (VIO) main entry point.

This module provides the main interface for running MSCKF-VIO with:
1. Baseline mode: Traditional MSCKF using EKF update
2. Fusion mode: MSCKF with SKNet for measurement updates

Both modes can run simultaneously for comparison.
"""
import numpy as np
import time
import os
from collections import namedtuple
from threading import Thread
from queue import Queue

from config import ConfigEuRoC
from dataset import EuRoCDataset, DataPublisher
from image import ImageProcessor
from msckf import MSCKF
from sknet_adapter import SKNetAdapter, SKNetTrainer, prepare_training_data


class VIO:
    """
    Visual-Inertial Odometry system.
    
    Supports running both baseline MSCKF and MSCKF+SKNet fusion simultaneously
    for comparison. Both produce real closed-loop trajectories (not shadow).
    """
    
    def __init__(self, config, mode='both', sknet_model_path=None):
        """
        Initialize VIO system.
        
        Args:
            config: Configuration object
            mode: 'baseline', 'sknet', or 'both' for comparison
            sknet_model_path: Path to pretrained SKNet model (for 'sknet' or 'both')
        """
        self.config = config
        self.mode = mode
        
        # Image processor (shared between both modes)
        self.image_processor = ImageProcessor(config)
        
        # Initialize baseline MSCKF if needed
        self.msckf_baseline = None
        if mode in ['baseline', 'both']:
            self.msckf_baseline = MSCKF(config, mode='baseline')
            self.msckf_baseline.collect_training_data = True  # Collect for potential training
        
        # Initialize SKNet-fused MSCKF if needed
        self.msckf_sknet = None
        self.sknet_adapter = None
        if mode in ['sknet', 'both']:
            # Determine state and observation dimensions
            # For MSCKF: 21 (IMU error state) + 6 * num_cam_states
            # We use a fixed size for SKNet (can be adjusted)
            state_dim = 147  # 21 + 6*21 (max 21 camera states)
            obs_dim = 50     # Maximum observation dimension
            
            self.sknet_adapter = SKNetAdapter(
                state_dim=state_dim,
                obs_dim=obs_dim,
                model_path=sknet_model_path
            )
            self.msckf_sknet = MSCKF(config, mode='sknet', sknet_adapter=self.sknet_adapter)
        
        # IMU buffer for initialization
        self.imu_buffer = []
        self.is_initialized = False
        
        # Trajectory storage
        self.trajectory_baseline = {'positions': [], 'orientations': [], 'timestamps': []}
        self.trajectory_sknet = {'positions': [], 'orientations': [], 'timestamps': []}
        
    def process_imu(self, imu_msg):
        """
        Process IMU measurement.
        
        Args:
            imu_msg: IMU message with angular_velocity and linear_acceleration
        """
        # Store for initialization
        if not self.is_initialized:
            self.imu_buffer.append(imu_msg)
            return
        
        # Process in baseline MSCKF
        if self.msckf_baseline is not None:
            self.msckf_baseline.process_imu(imu_msg)
        
        # Process in SKNet-fused MSCKF
        if self.msckf_sknet is not None:
            self.msckf_sknet.process_imu(imu_msg)
        
        # Also feed to image processor for tracking prediction
        self.image_processor.imu_callback(imu_msg)
    
    def process_image(self, stereo_msg):
        """
        Process stereo image pair.
        
        Args:
            stereo_msg: Stereo image message with cam0_msg and cam1_msg
            
        Returns:
            results: Dictionary with results from both modes (if applicable)
        """
        results = {}
        
        # Try to initialize if not done yet
        if not self.is_initialized:
            if len(self.imu_buffer) >= 200:
                self._initialize()
        
        if not self.is_initialized:
            return results
        
        # Process image through image processor
        # Note: The original ImageProcessor uses 'stareo_callback' (with typo)
        feature_msg = self.image_processor.stareo_callback(stereo_msg)
        
        if feature_msg is None or len(feature_msg.features) == 0:
            return results
        
        # Process in baseline MSCKF
        if self.msckf_baseline is not None:
            # State augmentation
            self.msckf_baseline.state_augmentation(feature_msg.timestamp)
            
            # Measurement update
            H, r, state_pred = self.msckf_baseline.measurement_update(feature_msg)
            
            # Prune camera states
            self.msckf_baseline.prune_cam_states()
            
            # Store trajectory
            pos = self.msckf_baseline.get_position()
            ori = self.msckf_baseline.get_orientation()
            self.trajectory_baseline['positions'].append(pos.copy())
            self.trajectory_baseline['orientations'].append(ori.copy())
            self.trajectory_baseline['timestamps'].append(feature_msg.timestamp)
            
            results['baseline'] = {
                'position': pos,
                'orientation': ori,
                'timestamp': feature_msg.timestamp
            }
        
        # Process in SKNet-fused MSCKF
        if self.msckf_sknet is not None:
            # State augmentation
            self.msckf_sknet.state_augmentation(feature_msg.timestamp)
            
            # Measurement update (uses SKNet)
            H, r, state_pred = self.msckf_sknet.measurement_update(feature_msg)
            
            # Prune camera states
            self.msckf_sknet.prune_cam_states()
            
            # Store trajectory
            pos = self.msckf_sknet.get_position()
            ori = self.msckf_sknet.get_orientation()
            self.trajectory_sknet['positions'].append(pos.copy())
            self.trajectory_sknet['orientations'].append(ori.copy())
            self.trajectory_sknet['timestamps'].append(feature_msg.timestamp)
            
            results['sknet'] = {
                'position': pos,
                'orientation': ori,
                'timestamp': feature_msg.timestamp
            }
        
        return results
    
    def _initialize(self):
        """Initialize filters using stationary IMU data."""
        print("Initializing VIO with", len(self.imu_buffer), "IMU samples...")
        
        if self.msckf_baseline is not None:
            success = self.msckf_baseline.initialize_gravity_and_bias(self.imu_buffer)
            if not success:
                print("Failed to initialize baseline MSCKF")
                return
            # Set timestamp
            self.msckf_baseline.imu_state.timestamp = self.imu_buffer[-1].timestamp
        
        if self.msckf_sknet is not None:
            success = self.msckf_sknet.initialize_gravity_and_bias(self.imu_buffer)
            if not success:
                print("Failed to initialize SKNet MSCKF")
                return
            # Set timestamp
            self.msckf_sknet.imu_state.timestamp = self.imu_buffer[-1].timestamp
            # Reset SKNet adapter
            self.sknet_adapter.reset()
        
        self.is_initialized = True
        print("VIO initialized successfully!")
        
        # Clear buffer
        self.imu_buffer = []
    
    def get_trajectories(self):
        """
        Get trajectories from both modes.
        
        Returns:
            Dictionary with 'baseline' and/or 'sknet' trajectories
        """
        trajectories = {}
        
        if self.msckf_baseline is not None:
            trajectories['baseline'] = {
                'positions': np.array(self.trajectory_baseline['positions']),
                'orientations': np.array(self.trajectory_baseline['orientations']),
                'timestamps': np.array(self.trajectory_baseline['timestamps'])
            }
        
        if self.msckf_sknet is not None:
            trajectories['sknet'] = {
                'positions': np.array(self.trajectory_sknet['positions']),
                'orientations': np.array(self.trajectory_sknet['orientations']),
                'timestamps': np.array(self.trajectory_sknet['timestamps'])
            }
        
        return trajectories
    
    def get_training_data(self):
        """
        Get training data collected from baseline MSCKF.
        
        Returns:
            List of training samples
        """
        if self.msckf_baseline is not None:
            return self.msckf_baseline.training_data
        return []
    
    def reset(self):
        """Reset VIO system."""
        if self.msckf_baseline is not None:
            self.msckf_baseline.reset()
        if self.msckf_sknet is not None:
            self.msckf_sknet.reset()
        if self.sknet_adapter is not None:
            self.sknet_adapter.reset()
        
        self.imu_buffer = []
        self.is_initialized = False
        self.trajectory_baseline = {'positions': [], 'orientations': [], 'timestamps': []}
        self.trajectory_sknet = {'positions': [], 'orientations': [], 'timestamps': []}


def compute_ape(estimated, ground_truth):
    """
    Compute Absolute Pose Error (APE) between estimated and ground truth trajectories.
    
    Args:
        estimated: Dictionary with 'positions' and 'timestamps'
        ground_truth: Dictionary with 'positions' and 'timestamps'
        
    Returns:
        Dictionary with APE metrics
    """
    # Align timestamps
    est_times = estimated['timestamps']
    gt_times = ground_truth['timestamps']
    
    # Find corresponding poses
    errors = []
    for i, t in enumerate(est_times):
        # Find closest ground truth timestamp
        idx = np.argmin(np.abs(gt_times - t))
        if np.abs(gt_times[idx] - t) > 0.05:  # 50ms threshold
            continue
        
        # Compute position error
        error = np.linalg.norm(estimated['positions'][i] - ground_truth['positions'][idx])
        errors.append(error)
    
    if len(errors) == 0:
        return {'rmse': float('inf'), 'mean': float('inf'), 'std': 0, 'max': float('inf')}
    
    errors = np.array(errors)
    return {
        'rmse': np.sqrt(np.mean(errors**2)),
        'mean': np.mean(errors),
        'std': np.std(errors),
        'max': np.max(errors)
    }


def visualize_trajectories(trajectories, ground_truth=None, save_path=None):
    """
    Visualize trajectories for comparison.
    
    Args:
        trajectories: Dictionary with 'baseline' and/or 'sknet' trajectories
        ground_truth: Optional ground truth trajectory
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not available for visualization")
        return
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(131, projection='3d')
    
    colors = {'baseline': 'blue', 'sknet': 'red', 'ground_truth': 'green'}
    
    if ground_truth is not None:
        pos = ground_truth['positions']
        ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'g-', label='Ground Truth', alpha=0.7)
    
    if 'baseline' in trajectories:
        pos = trajectories['baseline']['positions']
        if len(pos) > 0:
            ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', label='Baseline MSCKF', alpha=0.7)
    
    if 'sknet' in trajectories:
        pos = trajectories['sknet']['positions']
        if len(pos) > 0:
            ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'r-', label='MSCKF+SKNet', alpha=0.7)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    ax1.set_title('3D Trajectory')
    
    # XY plot
    ax2 = fig.add_subplot(132)
    
    if ground_truth is not None:
        pos = ground_truth['positions']
        ax2.plot(pos[:, 0], pos[:, 1], 'g-', label='Ground Truth', alpha=0.7)
    
    if 'baseline' in trajectories:
        pos = trajectories['baseline']['positions']
        if len(pos) > 0:
            ax2.plot(pos[:, 0], pos[:, 1], 'b-', label='Baseline MSCKF', alpha=0.7)
    
    if 'sknet' in trajectories:
        pos = trajectories['sknet']['positions']
        if len(pos) > 0:
            ax2.plot(pos[:, 0], pos[:, 1], 'r-', label='MSCKF+SKNet', alpha=0.7)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.set_title('XY Trajectory')
    ax2.axis('equal')
    
    # APE over time
    ax3 = fig.add_subplot(133)
    
    if ground_truth is not None:
        if 'baseline' in trajectories:
            ape_baseline = compute_ape(trajectories['baseline'], ground_truth)
            # Plot position error over time (simplified)
            if len(trajectories['baseline']['positions']) > 0:
                errors = []
                for i, t in enumerate(trajectories['baseline']['timestamps']):
                    idx = np.argmin(np.abs(ground_truth['timestamps'] - t))
                    if np.abs(ground_truth['timestamps'][idx] - t) < 0.05:
                        error = np.linalg.norm(trajectories['baseline']['positions'][i] - 
                                              ground_truth['positions'][idx])
                        errors.append(error)
                if len(errors) > 0:
                    ax3.plot(range(len(errors)), errors, 'b-', 
                            label=f'Baseline (RMSE: {ape_baseline["rmse"]:.3f}m)', alpha=0.7)
        
        if 'sknet' in trajectories:
            ape_sknet = compute_ape(trajectories['sknet'], ground_truth)
            if len(trajectories['sknet']['positions']) > 0:
                errors = []
                for i, t in enumerate(trajectories['sknet']['timestamps']):
                    idx = np.argmin(np.abs(ground_truth['timestamps'] - t))
                    if np.abs(ground_truth['timestamps'][idx] - t) < 0.05:
                        error = np.linalg.norm(trajectories['sknet']['positions'][i] - 
                                              ground_truth['positions'][idx])
                        errors.append(error)
                if len(errors) > 0:
                    ax3.plot(range(len(errors)), errors, 'r-', 
                            label=f'SKNet (RMSE: {ape_sknet["rmse"]:.3f}m)', alpha=0.7)
    
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Position Error (m)')
    ax3.legend()
    ax3.set_title('Absolute Position Error')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved trajectory visualization to {save_path}")
    
    plt.show()


def run_vio_comparison(dataset_path, duration=None, sknet_model_path=None, 
                       visualize=True, save_results=True):
    """
    Run VIO in comparison mode with both baseline and SKNet fusion.
    
    Args:
        dataset_path: Path to EuRoC dataset
        duration: Duration in seconds (None for full sequence)
        sknet_model_path: Path to pretrained SKNet model
        visualize: Whether to visualize results
        save_results: Whether to save results to file
        
    Returns:
        results: Dictionary with trajectories and metrics
    """
    # Load dataset
    config = ConfigEuRoC()
    dataset = EuRoCDataset(dataset_path)
    
    # Initialize VIO in comparison mode
    vio = VIO(config, mode='both', sknet_model_path=sknet_model_path)
    
    # Setup queues and publishers
    img_queue = Queue()
    imu_queue = Queue()
    
    duration = duration or float('inf')
    imu_publisher = DataPublisher(dataset.imu, imu_queue, duration)
    img_publisher = DataPublisher(dataset.stereo, img_queue, duration)
    
    # Start publishers
    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)
    
    # Process IMU in separate thread
    def process_imu_thread():
        while True:
            msg = imu_queue.get()
            if msg is None:
                return
            vio.process_imu(msg)
    
    imu_thread = Thread(target=process_imu_thread)
    imu_thread.start()
    
    # Process images in main thread
    print("Running VIO comparison...")
    frame_count = 0
    
    while True:
        msg = img_queue.get()
        if msg is None:
            break
        
        results = vio.process_image(msg)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames")
            if 'baseline' in results:
                print(f"  Baseline position: {results['baseline']['position']}")
            if 'sknet' in results:
                print(f"  SKNet position: {results['sknet']['position']}")
    
    # Stop publishers
    imu_publisher.stop()
    img_publisher.stop()
    imu_thread.join()
    
    print(f"\nProcessed total {frame_count} frames")
    
    # Get trajectories
    trajectories = vio.get_trajectories()
    
    # Get ground truth
    gt_positions = []
    gt_timestamps = []
    for gt_msg in dataset.groundtruth:
        gt_positions.append(gt_msg.p)
        gt_timestamps.append(gt_msg.timestamp)
    
    ground_truth = {
        'positions': np.array(gt_positions),
        'timestamps': np.array(gt_timestamps)
    }
    
    # Compute metrics
    results = {'trajectories': trajectories, 'ground_truth': ground_truth}
    
    if 'baseline' in trajectories and len(trajectories['baseline']['positions']) > 0:
        ape_baseline = compute_ape(trajectories['baseline'], ground_truth)
        results['ape_baseline'] = ape_baseline
        print(f"\nBaseline MSCKF APE:")
        print(f"  RMSE: {ape_baseline['rmse']:.4f} m")
        print(f"  Mean: {ape_baseline['mean']:.4f} m")
        print(f"  Max:  {ape_baseline['max']:.4f} m")
    
    if 'sknet' in trajectories and len(trajectories['sknet']['positions']) > 0:
        ape_sknet = compute_ape(trajectories['sknet'], ground_truth)
        results['ape_sknet'] = ape_sknet
        print(f"\nMSCKF+SKNet APE:")
        print(f"  RMSE: {ape_sknet['rmse']:.4f} m")
        print(f"  Mean: {ape_sknet['mean']:.4f} m")
        print(f"  Max:  {ape_sknet['max']:.4f} m")
    
    # Save results
    if save_results:
        save_dir = './results'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save trajectories
        if 'baseline' in trajectories and len(trajectories['baseline']['positions']) > 0:
            np.savetxt(
                os.path.join(save_dir, 'trajectory_baseline.txt'),
                np.hstack([
                    trajectories['baseline']['timestamps'].reshape(-1, 1),
                    trajectories['baseline']['positions']
                ]),
                header='timestamp x y z'
            )
        
        if 'sknet' in trajectories and len(trajectories['sknet']['positions']) > 0:
            np.savetxt(
                os.path.join(save_dir, 'trajectory_sknet.txt'),
                np.hstack([
                    trajectories['sknet']['timestamps'].reshape(-1, 1),
                    trajectories['sknet']['positions']
                ]),
                header='timestamp x y z'
            )
        
        print(f"\nSaved results to {save_dir}")
    
    # Visualize
    if visualize:
        visualize_trajectories(trajectories, ground_truth, 
                              save_path='./results/trajectory_comparison.png' if save_results else None)
    
    # Get training data
    training_data = vio.get_training_data()
    if len(training_data) > 0:
        print(f"\nCollected {len(training_data)} training samples")
        results['training_data'] = training_data
    
    return results


def train_sknet_from_msckf_data(training_data, state_dim=147, obs_dim=50,
                                 epochs=100, learning_rate=1e-3, save_path=None):
    """
    Train SKNet using data collected from baseline MSCKF.
    
    Args:
        training_data: List of training samples from MSCKF
        state_dim: State dimension for SKNet
        obs_dim: Observation dimension for SKNet
        epochs: Number of training epochs
        learning_rate: Learning rate
        save_path: Path to save trained model
        
    Returns:
        trained_adapter: Trained SKNetAdapter
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    print(f"Preparing training data from {len(training_data)} samples...")
    
    # Prepare training batches
    batches = prepare_training_data(training_data, state_dim, obs_dim)
    
    if len(batches) == 0:
        print("No valid training data")
        return None
    
    # Create adapter and trainer
    adapter = SKNetAdapter(state_dim, obs_dim)
    trainer = SKNetTrainer(adapter, learning_rate=learning_rate)
    
    # Collate batches
    from sknet_adapter import collate_batches
    full_batch = collate_batches(batches)
    
    # Create dataset
    dataset = TensorDataset(
        full_batch['state_inno'],
        full_batch['residual'],
        full_batch['diff_state'],
        full_batch['diff_obs'],
        full_batch['lin_error'],
        full_batch['H_flat'],
        full_batch['P_target'],
        full_batch['S_inv_target']
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    print(f"Training SKNet for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            batch_dict = {
                'state_inno': batch[0],
                'residual': batch[1],
                'diff_state': batch[2],
                'diff_obs': batch[3],
                'lin_error': batch[4],
                'H_flat': batch[5],
                'P_target': batch[6],
                'S_inv_target': batch[7]
            }
            
            loss = trainer.train_step(batch_dict)
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Save model
    if save_path is not None:
        adapter.save_model(save_path)
    
    return adapter


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MSCKF-VIO with SKNet fusion')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to EuRoC dataset')
    parser.add_argument('--duration', type=float, default=None,
                       help='Duration in seconds (default: full sequence)')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['baseline', 'sknet', 'both'],
                       help='VIO mode')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to pretrained SKNet model')
    parser.add_argument('--train', action='store_true',
                       help='Train SKNet after running baseline')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Run VIO
    results = run_vio_comparison(
        args.dataset,
        duration=args.duration,
        sknet_model_path=args.model,
        visualize=not args.no_visualize
    )
    
    # Train SKNet if requested
    if args.train and 'training_data' in results:
        print("\n" + "="*50)
        print("Training SKNet from collected data...")
        print("="*50)
        
        adapter = train_sknet_from_msckf_data(
            results['training_data'],
            epochs=100,
            save_path='./models/sknet_trained.pth'
        )
        
        if adapter is not None:
            print("SKNet training complete!")
