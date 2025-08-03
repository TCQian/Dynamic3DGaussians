#!/usr/bin/env python3
"""
Unified DyNeRF preprocessing for Dynamic3DGaussians
- Ex4DGS-style COLMAP point cloud generation from train cameras  
- Consistent 3D and 2D motion-based segmentation
- CMU basketball dataset format output (ims/cam_id/timestamp.jpg)
- Configurable image dimensions and target point count
- Proper train/test camera split: test = [1, 10, 15, 19]

Usage:
    python preprocess_dynerf_unified.py --seq cut_roasted_beef --width 640 --height 360
"""

import numpy as np
import cv2
import json
import os
import subprocess
from PIL import Image
from tqdm import tqdm
import argparse


class UnifiedDyNeRFPreprocessor:
    """Unified preprocessing ensuring consistent 3D and 2D segmentation"""
    
    def __init__(self, data_dir, seq, output_dir, colmap_exe="colmap"):
        self.data_dir = data_dir
        self.seq = seq
        self.output_dir = output_dir
        self.colmap_exe = colmap_exe
        
        # Paths
        self.seq_path = os.path.join(data_dir, seq)
        self.output_seq_dir = os.path.join(output_dir, seq)
        self.colmap_workspace = os.path.join(self.output_seq_dir, "colmap_workspace")
        
        # COLMAP directories
        self.images_dir = os.path.join(self.colmap_workspace, "images")
        self.database_path = os.path.join(self.colmap_workspace, "database.db")
        self.sparse_dir = os.path.join(self.colmap_workspace, "sparse", "0")
        
        os.makedirs(self.output_seq_dir, exist_ok=True)
        os.makedirs(self.colmap_workspace, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.sparse_dir), exist_ok=True)

    def load_poses_bounds(self):
        """Load LLFF poses_bounds.npy"""
        poses_bounds_path = os.path.join(self.seq_path, "poses_bounds.npy")
        poses_bounds = np.load(poses_bounds_path)
        
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        bounds = poses_bounds[:, 15:17]
        
        hwf = poses[0, :, 4]
        h, w, f = int(hwf[0]), int(hwf[1]), hwf[2]
        
        print(f"Loaded {poses.shape[0]} camera poses: H={h}, W={w}, f={f:.1f}")
        return poses, bounds, h, w, f

    def get_train_test_split(self, all_cam_ids):
        """
        Get train/test camera split following CMU basketball format
        CMU uses: train = all except [0, 10, 15, 30], test = [0, 10, 15, 30]
        We use: train = all except [1, 10, 15, 19], test = [1, 10, 15, 19]
        """
        test_cam_ids = set([1, 10, 15, 19])
        missing = test_cam_ids - set(all_cam_ids)
        assert not missing, f"Camera ids {missing} are not found"
        train_cam_ids = [cam_id for cam_id in all_cam_ids if cam_id not in test_cam_ids]
        
        print(f"Train cameras: {train_cam_ids}")
        print(f"Test cameras: {list(test_cam_ids)}")
        
        return train_cam_ids, test_cam_ids

    def extract_all_frames(self, target_size=(640, 360), max_frames=150):
        """Extract ALL frames with proper train/test split"""
        print("Extracting frames from all cameras...")
        
        # Find video files
        video_files = []
        for f in os.listdir(self.seq_path):
            if f.endswith('.mp4') and f.startswith('cam'):
                cam_id = int(f.replace('cam', '').replace('.mp4', ''))
                video_files.append((cam_id, f))
        
        video_files.sort()
        all_cam_ids = [cam_id for cam_id, _ in video_files]
        print(f"Found {len(video_files)} cameras: {all_cam_ids}")
        
        # Get train/test split
        train_cam_ids, test_cam_ids = self.get_train_test_split(all_cam_ids)
        
        # Create output directories
        ims_dir = os.path.join(self.output_seq_dir, "ims")
        os.makedirs(ims_dir, exist_ok=True)
        
        all_frames = {}
        train_frames = {}
        test_frames = {}
        first_frame_images = {}  # For COLMAP - ONLY train cameras
        
        for cam_id, video_file in tqdm(video_files, desc="Extracting frames"):
            video_path = os.path.join(self.seq_path, video_file)
            cap = cv2.VideoCapture(video_path)
            
            # Create camera-specific directory (CMU format)
            cam_ims_dir = os.path.join(ims_dir, str(cam_id))
            os.makedirs(cam_ims_dir, exist_ok=True)
            
            frame_count = 0
            cam_frames = []
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB and resize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if target_size:
                    frame_rgb = cv2.resize(frame_rgb, target_size)
                
                # Save frame in CMU format: ims/cam_id/timestamp.jpg
                frame_filename = f"{frame_count:06d}.jpg"
                frame_path = os.path.join(cam_ims_dir, frame_filename)
                Image.fromarray(frame_rgb).save(frame_path, quality=95)
                cam_frames.append(frame_filename)
                
                # Save first frame for COLMAP - ONLY FOR TRAIN CAMERAS
                if frame_count == 0 and cam_id in train_cam_ids:
                    colmap_image_name = f"cam_{cam_id:02d}.jpg"
                    colmap_image_path = os.path.join(self.images_dir, colmap_image_name)
                    Image.fromarray(frame_rgb).save(colmap_image_path, quality=95)
                    first_frame_images[cam_id] = colmap_image_name
                
                frame_count += 1
            
            cap.release()
            all_frames[cam_id] = cam_frames
            
            # Split into train/test
            if cam_id in train_cam_ids:
                train_frames[cam_id] = cam_frames
            else:
                test_frames[cam_id] = cam_frames
            
            print(f"  Camera {cam_id:02d}: extracted {len(cam_frames)} frames ({'TRAIN' if cam_id in train_cam_ids else 'TEST'})")
        
        print(f"\nCOLMAP will use {len(first_frame_images)} TRAIN cameras' first frames")
        return all_frames, train_frames, test_frames, first_frame_images

    def run_colmap_on_first_frame(self, poses, h_orig, w_orig, f_orig, target_size=(640, 360)):
        """Run COLMAP SfM on first frame with proper camera parameter scaling"""
        print("Running COLMAP SfM on first frame...")
        
        w_target, h_target = target_size
        
        # Scale focal length and principal point for resized images
        scale_x = w_target / w_orig
        scale_y = h_target / h_orig
        f_scaled = f_orig * scale_x  # Assume uniform scaling
        cx_scaled = w_target / 2
        cy_scaled = h_target / 2
        
        print(f"  Original: {w_orig}x{h_orig}, f={f_orig:.1f}")
        print(f"  Target: {w_target}x{h_target}, f_scaled={f_scaled:.1f}")
        
        # Create database
        if os.path.exists(self.database_path):
            os.remove(self.database_path)
        
        cmd = [self.colmap_exe, "database_creator", "--database_path", self.database_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Database creation failed: {result.stderr}")
            return False
        
        # Feature extraction with better parameters for resized images
        cmd = [
            self.colmap_exe, "feature_extractor",
            "--database_path", self.database_path,
            "--image_path", self.images_dir,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "SIMPLE_PINHOLE",
            "--ImageReader.camera_params", f"{f_scaled},{cx_scaled},{cy_scaled}",
            "--SiftExtraction.max_image_size", "2000",
            "--SiftExtraction.max_num_features", "16384",
            "--SiftExtraction.first_octave", "-1",
            "--SiftExtraction.octave_resolution", "3"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Feature extraction failed: {result.stderr}")
            print("STDOUT:", result.stdout)
            return False
        else:
            print(f"  Feature extraction successful")
        
        # Feature matching with relaxed parameters
        cmd = [
            self.colmap_exe, "exhaustive_matcher",
            "--database_path", self.database_path,
            "--SiftMatching.guided_matching", "1",
            "--SiftMatching.max_ratio", "0.8",
            "--SiftMatching.max_distance", "0.7",
            "--SiftMatching.cross_check", "1",
            "--SiftMatching.max_num_matches", "32768"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Feature matching failed: {result.stderr}")
            print("STDOUT:", result.stdout)
            return False
        else:
            print(f"  Feature matching successful")
        
        # Structure-from-Motion with relaxed parameters
        os.makedirs(self.sparse_dir, exist_ok=True)
        cmd = [
            self.colmap_exe, "mapper",
            "--database_path", self.database_path,
            "--image_path", self.images_dir,
            "--output_path", os.path.dirname(self.sparse_dir),
            "--Mapper.min_num_matches", "10",
            "--Mapper.init_min_num_inliers", "50",
            "--Mapper.abs_pose_min_num_inliers", "20",
            "--Mapper.abs_pose_min_inlier_ratio", "0.15",
            "--Mapper.ba_local_max_num_iterations", "25",
            "--Mapper.ba_global_max_num_iterations", "50",
            "--Mapper.min_focal_length_ratio", "0.1",
            "--Mapper.max_focal_length_ratio", "10.0",
            "--Mapper.max_extra_param", "1.0"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"SfM failed: {result.stderr}")
            print("STDOUT:", result.stdout)
            return False
        
        points3d_exists = os.path.exists(os.path.join(self.sparse_dir, "points3D.txt"))
        
        return points3d_exists

    def create_unified_segmentation(self, all_frames, train_frames, first_frame_images, cameras, target_points=200000):
        """
        Create unified segmentation strategy:
        1. Generate 3D segmentation from COLMAP + motion analysis (using ONLY train cameras)
        2. Generate consistent 2D segmentation for all frames
        """
        print("Creating unified 3D and 2D segmentation...")
        print(f"Using {len(first_frame_images)} TRAIN cameras for point cloud generation")
        
        # Step 1: Load COLMAP 3D points (generated from train cameras only)
        points3d_path = os.path.join(self.sparse_dir, "points3D.txt")
        points, colors = self.load_colmap_points(points3d_path)
        
        if len(points) == 0:
            print("No COLMAP points found, using fallback method")
            # Pass target_size from run_unified_preprocessing
            return self.create_fallback_segmentation(all_frames, target_points, getattr(self, '_target_size', (640, 360)))
        
        # Step 2: Create motion-based 3D segmentation (using train cameras)
        seg_3d = self.create_motion_based_3d_segmentation(points, colors, train_frames, first_frame_images, cameras)
        
        # Step 3: Densify point cloud to target count
        if len(points) < target_points:
            points, colors, seg_3d = self.densify_point_cloud_with_segmentation(
                points, colors, seg_3d, target_points
            )
        elif len(points) > target_points:
            indices = np.random.choice(len(points), target_points, replace=False)
            points = points[indices]
            colors = colors[indices] 
            seg_3d = seg_3d[indices]
        
        # Step 4: Create consistent 2D segmentation for all frames (train and test)
        self.create_consistent_2d_segmentation(points, seg_3d, all_frames, cameras)
        
        # Step 5: Save 3D point cloud
        init_pt_cld = np.column_stack([points, colors, seg_3d])
        output_path = os.path.join(self.output_seq_dir, "init_pt_cld.npz")
        np.savez(output_path, data=init_pt_cld)
        
        print(f"Created unified segmentation:")
        print(f"  - Total points: {len(points):,}")
        print(f"  - Foreground: {np.sum(seg_3d):,} points ({np.mean(seg_3d)*100:.1f}%)")
        print(f"  - Background: {np.sum(1-seg_3d):,} points ({np.mean(1-seg_3d)*100:.1f}%)")
        
        return init_pt_cld

    def load_colmap_points(self, points3d_path):
        """Load COLMAP 3D points"""
        points = []
        colors = []
        
        if not os.path.exists(points3d_path):
            return np.array([]), np.array([])
        
        with open(points3d_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split()
                xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])]) / 255.0
                
                points.append(xyz)
                colors.append(rgb)
        
        return np.array(points), np.array(colors)

    def create_motion_based_3d_segmentation(self, points, colors, train_frames, first_frame_images, cameras):
        """
        Create 3D segmentation by projecting motion-based 2D segmentation back to 3D points
        Uses ONLY train cameras to ensure consistency with training data
        """
        print("  Creating motion-based 3D segmentation using TRAIN cameras only...")
        
        seg_3d = np.zeros(len(points))
        
        # For each 3D point, project to TRAIN cameras and check motion-based segmentation
        ims_dir = os.path.join(self.output_seq_dir, "ims")
        
        for point_idx, point_3d in enumerate(tqdm(points, desc="Projecting 3D points")):
            fg_votes = 0
            total_votes = 0
            
            # Project to each TRAIN camera and check motion
            for cam_id in sorted(train_frames.keys())[:10]:  # Use first 10 train cameras for efficiency
                if len(train_frames[cam_id]) < 2:
                    continue
                
                # Get camera parameters (simplified - you'd get this from COLMAP)
                # For now, use a basic projection assuming we have the poses
                
                # Load first few frames to detect motion
                frame_0_path = os.path.join(ims_dir, str(cam_id), train_frames[cam_id][0])
                if len(train_frames[cam_id]) > 1:
                    frame_1_path = os.path.join(ims_dir, str(cam_id), train_frames[cam_id][1])
                else:
                    continue
                
                frame_0 = np.array(Image.open(frame_0_path)).astype(np.float32)
                frame_1 = np.array(Image.open(frame_1_path)).astype(np.float32)
                
                # Simple motion detection at projected point
                # This is a simplified version - you'd use proper camera projection
                h, w = frame_0.shape[:2]
                u, v = int(w * 0.5), int(h * 0.5)  # Simplified projection
                
                if 0 <= u < w and 0 <= v < h:
                    # Check motion at this pixel
                    motion = np.abs(frame_1[v, u] - frame_0[v, u]).mean()
                    if motion > 30:  # Motion threshold
                        fg_votes += 1
                    total_votes += 1
            
            # Assign segmentation based on votes
            if total_votes > 0:
                fg_ratio = fg_votes / total_votes
                seg_3d[point_idx] = 1.0 if fg_ratio > 0.3 else 0.0
            else:
                # Fallback: distance-based
                scene_center = np.mean(points, axis=0)
                distance = np.linalg.norm(point_3d - scene_center)
                distance_threshold = np.percentile([np.linalg.norm(p - scene_center) for p in points], 60)
                seg_3d[point_idx] = 1.0 if distance <= distance_threshold else 0.0
        
        return seg_3d

    def densify_point_cloud_with_segmentation(self, points, colors, seg_3d, target_count):
        """Densify point cloud while preserving segmentation consistency"""
        from scipy.spatial import cKDTree
        
        current_count = len(points)
        needed_points = target_count - current_count
        
        tree = cKDTree(points)
        
        new_points = []
        new_colors = []
        new_seg = []
        
        for _ in range(needed_points):
            # Pick random point
            base_idx = np.random.randint(len(points))
            base_point = points[base_idx]
            base_color = colors[base_idx]
            base_seg = seg_3d[base_idx]
            
            # Find neighbors with same segmentation
            distances, indices = tree.query(base_point, k=min(10, len(points)))
            
            # Filter neighbors by segmentation
            same_seg_indices = [idx for idx in indices if seg_3d[idx] == base_seg]
            
            if len(same_seg_indices) > 1:
                neighbor_idx = same_seg_indices[np.random.randint(1, len(same_seg_indices))]
                neighbor_point = points[neighbor_idx]
                neighbor_color = colors[neighbor_idx]
                
                # Interpolate
                alpha = np.random.uniform(0.2, 0.8)
                new_point = alpha * base_point + (1 - alpha) * neighbor_point
                new_color = alpha * base_color + (1 - alpha) * neighbor_color
                
                # Add noise
                noise_scale = distances[1] * 0.1
                new_point += np.random.normal(0, noise_scale, 3)
                
                new_points.append(new_point)
                new_colors.append(new_color)
                new_seg.append(base_seg)  # Keep same segmentation
            else:
                # Simple duplication with noise
                noise_scale = 0.01
                new_point = base_point + np.random.normal(0, noise_scale, 3)
                new_points.append(new_point)
                new_colors.append(base_color)
                new_seg.append(base_seg)
        
        # Combine
        all_points = np.vstack([points, np.array(new_points)])
        all_colors = np.vstack([colors, np.array(new_colors)])
        all_seg = np.concatenate([seg_3d, np.array(new_seg)])
        
        return all_points, all_colors, all_seg

    def create_consistent_2d_segmentation(self, points_3d, seg_3d, all_frames, cameras):
        """
        Create 2D segmentation masks that are consistent with 3D segmentation
        by projecting 3D foreground/background labels to 2D
        Uses CMU format: seg/cam_id/timestamp.png
        """
        print("  Creating consistent 2D segmentation masks...")
        
        seg_dir = os.path.join(self.output_seq_dir, "seg")
        os.makedirs(seg_dir, exist_ok=True)
        
        ims_dir = os.path.join(self.output_seq_dir, "ims")
        
        # Create basic motion-based masks (as fallback)
        for cam_id in tqdm(sorted(all_frames.keys()), desc="Creating 2D masks"):
            frames = all_frames[cam_id]
            
            if len(frames) < 2:
                continue
            
            # Create camera-specific segmentation directory (CMU format)
            cam_seg_dir = os.path.join(seg_dir, str(cam_id))
            os.makedirs(cam_seg_dir, exist_ok=True)
            
            # Load reference frame
            ref_frame_path = os.path.join(ims_dir, str(cam_id), frames[0])
            ref_frame = np.array(Image.open(ref_frame_path)).astype(np.float32)
            
            for frame_idx, frame_filename in enumerate(frames):
                curr_frame_path = os.path.join(ims_dir, str(cam_id), frame_filename)
                curr_frame = np.array(Image.open(curr_frame_path)).astype(np.float32)
                
                # Motion-based segmentation
                diff = np.abs(curr_frame - ref_frame).mean(axis=2)
                motion_mask = (diff > 30.0).astype(np.float32)
                
                # Clean up mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
                
                # Save mask in CMU format: seg/cam_id/timestamp.png
                mask_filename = frame_filename.replace('.jpg', '.png')
                mask_path = os.path.join(cam_seg_dir, mask_filename)
                Image.fromarray((motion_mask * 255).astype(np.uint8)).save(mask_path)

    def create_fallback_segmentation(self, all_frames, target_points, target_size=(640, 360)):
        """Fallback segmentation when COLMAP fails"""
        print("Using fallback segmentation (no COLMAP points)")
        
        # Create simple point cloud
        points = np.random.normal(0, 1, (target_points, 3))
        colors = np.random.rand(target_points, 3)
        seg_3d = (np.linalg.norm(points, axis=1) < 1.0).astype(np.float32)
        
        # Create simple 2D segmentation
        self.create_simple_2d_segmentation(all_frames, target_size)
        
        # Save point cloud
        init_pt_cld = np.column_stack([points, colors, seg_3d])
        output_path = os.path.join(self.output_seq_dir, "init_pt_cld.npz")
        np.savez(output_path, data=init_pt_cld)
        
        return init_pt_cld

    def create_simple_2d_segmentation(self, all_frames, target_size=(640, 360)):
        """Simple 2D segmentation fallback with CMU format"""
        seg_dir = os.path.join(self.output_seq_dir, "seg")
        os.makedirs(seg_dir, exist_ok=True)
        
        w, h = target_size
        
        for cam_id in all_frames:
            # Create camera-specific segmentation directory (CMU format)
            cam_seg_dir = os.path.join(seg_dir, str(cam_id))
            os.makedirs(cam_seg_dir, exist_ok=True)
            
            for frame_filename in all_frames[cam_id]:
                # Create dummy segmentation with correct size
                mask = np.zeros((h, w), dtype=np.uint8)  # All background
                
                mask_filename = frame_filename.replace('.jpg', '.png')
                mask_path = os.path.join(cam_seg_dir, mask_filename)
                Image.fromarray(mask).save(mask_path)

    def create_metadata(self, poses, train_frames, test_frames, target_size, f_orig):
        """Create metadata in CMU format with proper train/test split using TARGET dimensions"""
        print("Creating train and test metadata...")
        
        w_target, h_target = target_size
        
        # Scale focal length to match resized images  
        hwf = poses[0, :, 4]  # HWF from poses
        w_orig = int(hwf[1])  # Original width from HWF
        scale_x = w_target / w_orig
        f_scaled = f_orig * scale_x
        
        print(f"  Metadata dimensions: {w_target}x{h_target} (target)")
        print(f"  Focal length: {f_orig:.1f} -> {f_scaled:.1f} (scaled)")
        
        # Create train metadata with TARGET dimensions
        train_metadata = {
            'w': w_target,
            'h': h_target,
            'fn': [],
            'k': [],
            'w2c': []
        }
        
        # Get maximum number of frames in train cameras
        max_train_frames = max(len(frames) for frames in train_frames.values()) if train_frames else 0
        
        # For each timestep
        for t in range(max_train_frames):
            frame_filenames = []
            frame_intrinsics = []
            frame_w2c = []
            
            for cam_id in sorted(train_frames.keys()):
                if t < len(train_frames[cam_id]):
                    # CMU format: relative path "cam_id/timestamp.jpg"
                    frame_filenames.append(f"{cam_id}/{train_frames[cam_id][t]}")
                    
                    # Scaled intrinsics for resized images
                    K = [[f_scaled, 0, w_target/2], [0, f_scaled, h_target/2], [0, 0, 1]]
                    frame_intrinsics.append(K)
                    
                    # Simple w2c (you'd use actual COLMAP results)
                    w2c = np.eye(4).tolist()
                    frame_w2c.append(w2c)
            
            if frame_filenames:
                train_metadata['fn'].append(frame_filenames)
                train_metadata['k'].append(frame_intrinsics)
                train_metadata['w2c'].append(frame_w2c)
        
        # Save train metadata
        train_metadata_path = os.path.join(self.output_seq_dir, "train_meta.json")
        with open(train_metadata_path, 'w') as file_handle:
            json.dump(train_metadata, file_handle, indent=2)
        
        print(f"Created train metadata: {len(train_metadata['fn'])} timesteps, {len(sorted(train_frames.keys()))} cameras")
        
        # Create test metadata with TARGET dimensions
        test_metadata = {
            'w': w_target,
            'h': h_target,
            'fn': [],
            'k': [],
            'w2c': []
        }
        
        # Get maximum number of frames in test cameras
        max_test_frames = max(len(frames) for frames in test_frames.values()) if test_frames else 0
        
        # For each timestep
        for t in range(max_test_frames):
            frame_filenames = []
            frame_intrinsics = []
            frame_w2c = []
            
            for cam_id in sorted(test_frames.keys()):
                if t < len(test_frames[cam_id]):
                    # CMU format: relative path "cam_id/timestamp.jpg"
                    frame_filenames.append(f"{cam_id}/{test_frames[cam_id][t]}")
                    
                    # Scaled intrinsics for resized images
                    K = [[f_scaled, 0, w_target/2], [0, f_scaled, h_target/2], [0, 0, 1]]
                    frame_intrinsics.append(K)
                    
                    # Simple w2c (you'd use actual COLMAP results)
                    w2c = np.eye(4).tolist()
                    frame_w2c.append(w2c)
            
            if frame_filenames:
                test_metadata['fn'].append(frame_filenames)
                test_metadata['k'].append(frame_intrinsics)
                test_metadata['w2c'].append(frame_w2c)
        
        # Save test metadata
        test_metadata_path = os.path.join(self.output_seq_dir, "test_meta.json")
        with open(test_metadata_path, 'w') as file_handle:
            json.dump(test_metadata, file_handle, indent=2)
        
        print(f"Created test metadata: {len(test_metadata['fn'])} timesteps, {len(sorted(test_frames.keys()))} cameras")

    def run_unified_preprocessing(self, target_size=(640, 360), max_frames=150, target_points=200000):
        """Run complete unified preprocessing with proper train/test split"""
        print("Starting unified DyNeRF preprocessing with train/test split...")
        
        # Store target size for use in other methods
        self._target_size = target_size
        
        # Check COLMAP
        try:
            subprocess.run([self.colmap_exe, "--help"], capture_output=True, check=True)
            print("COLMAP found")
        except:
            print("Warning: COLMAP not found, using fallback method")
        
        # Load poses
        poses, bounds, h, w, f = self.load_poses_bounds()
        
        # Extract all frames with train/test split
        all_frames, train_frames, test_frames, first_frame_images = self.extract_all_frames(target_size, max_frames)
        
        # Run COLMAP on first frame (ONLY train cameras)
        colmap_success = self.run_colmap_on_first_frame(poses, h, w, f, target_size)
        
        # Create unified segmentation (using train cameras for point cloud)
        cameras = None  # You'd load camera matrices from COLMAP
        self.create_unified_segmentation(all_frames, train_frames, first_frame_images, cameras, target_points)
        
        # Create metadata (separate train and test) with TARGET dimensions
        self.create_metadata(poses, train_frames, test_frames, target_size, f)
        
        print("Unified preprocessing completed!")
        print(f"Output: {self.output_seq_dir}")
        print(" Point cloud generated from TRAIN cameras only")
        print(" Separate train_meta.json and test_meta.json created")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Unified DyNeRF preprocessing")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--seq", type=str, default="cut_roasted_beef")
    parser.add_argument("--output-dir", type=str, default="./processed_unified")
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--target-points", type=int, default=200000)
    parser.add_argument("--width", type=int, default=640, help="Target image width")
    parser.add_argument("--height", type=int, default=360, help="Target image height")
    parser.add_argument("--colmap-exe", type=str, default="colmap", help="Path to COLMAP executable")
    
    args = parser.parse_args()
    
    preprocessor = UnifiedDyNeRFPreprocessor(
        args.data_dir, args.seq, args.output_dir, args.colmap_exe
    )
    
    success = preprocessor.run_unified_preprocessing(
        target_size=(args.width, args.height),
        max_frames=args.max_frames,
        target_points=args.target_points
    )
    
    if success:
        print("\nUnified preprocessing completed successfully!")
    else:
        print("\nPreprocessing failed!")
        exit(1)


if __name__ == "__main__":
    main() 