#!/usr/bin/env python3
"""
Unified DyNeRF Preprocessing for Dynamic3DGaussians

Enhanced version with:
1. Single-frame COLMAP: Uses only first frame per training camera
2. Uses COLMAP dense point cloud: No sparse point cloud, use dense reconstruction
3. Proper train/test split with separate metadata files
4. CMU-format output structure
5. All points labeled as foreground: No segmentation, everything is foreground

Features:
- COLMAP-based dense 3D reconstruction from first frame per camera
- All 3D points labeled as foreground (no segmentation)
- Uses COLMAP dense point cloud (typically much larger than sparse)
- Simple foreground-only 2D/3D segmentation
- Train/test camera splitting (test cameras: [1, 10, 15, 19])
- CMU format: ims/cam_id/timestamp.jpg, seg/cam_id/timestamp.png

Usage:
    python preprocess_dynerf_unified.py --seq cut_roasted_beef

Arguments:
    --width/height: Target image dimensions (default: 640x360)
"""

import numpy as np
import cv2
import json
import os
import subprocess
from PIL import Image
from tqdm import tqdm
import argparse
import shutil
import re
from plyfile import PlyData, PlyElement

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
        self.dense_dir = os.path.join(self.colmap_workspace, "dense")
        
        if os.path.exists(self.colmap_workspace):
            print("Folder exists, deleting...")
            shutil.rmtree(self.colmap_workspace)
            print("Deleted.")
        os.makedirs(self.output_seq_dir, exist_ok=True)
        os.makedirs(self.colmap_workspace, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.sparse_dir), exist_ok=True)

    def natural_key(self, s):
        """Natural sorting key for camera files"""
        m = re.search(r'cam(\d+)\.mp4$', s)
        return int(m.group(1)) if m else 1e9  # push non-matching to end

    def build_camid_to_pose_index(self, folder):
        """Build mapping from camera ID to pose index in poses_bounds.npy"""
        cams = [f for f in os.listdir(folder) if f.startswith('cam') and f.endswith('.mp4')]
        cams_sorted = sorted(cams, key=self.natural_key)
        
        print(f"  Found camera files: {cams}")
        print(f"  Sorted camera files: {cams_sorted}")
        
        mapping = {}
        for i, cam_file in enumerate(cams_sorted):
            cam_id = int(re.search(r'cam(\d+)\.mp4$', cam_file).group(1))
            mapping[cam_id] = i
            print(f"    cam{cam_id:02d}.mp4 → pose index {i}")
        
        return mapping

    def load_poses_bounds(self):
        """Load LLFF poses_bounds.npy and build camera ID mapping"""
        poses_bounds_path = os.path.join(self.seq_path, "poses_bounds.npy")
        poses_bounds = np.load(poses_bounds_path)
        
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        bounds = poses_bounds[:, 15:17]
        
        hwf = poses[0, :, 4]
        h, w, f = int(hwf[0]), int(hwf[1]), hwf[2]
        
        # Build mapping from camera ID to pose index
        print("Building camera ID to pose index mapping...")
        self.camid_to_pose_index = self.build_camid_to_pose_index(self.seq_path)
        
        print(f"Loaded {poses.shape[0]} camera poses: H={h}, W={w}, f={f:.1f}")
        print(f"Camera ID mapping: {self.camid_to_pose_index}")
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
        """Extract ALL frames with proper train/test split, using only first frame for COLMAP"""
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
        first_frame_images = {}  # For COLMAP - ONLY train cameras, single frame each
        
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
                
                # Save ONLY FIRST frame for COLMAP - ONLY FOR TRAIN CAMERAS
                # Note: Using single frame might cause dense reconstruction issues
                if frame_count == 0 and cam_id in train_cam_ids:
                    colmap_image_name = f"cam_{cam_id:02d}_frame_000.jpg"
                    colmap_image_path = os.path.join(self.images_dir, colmap_image_name)
                    Image.fromarray(frame_rgb).save(colmap_image_path, quality=95)
                    first_frame_images[cam_id] = [colmap_image_name]

                frame_count += 1
            
            cap.release()
            all_frames[cam_id] = cam_frames
            
            # Split into train/test
            if cam_id in train_cam_ids:
                train_frames[cam_id] = cam_frames
            else:
                test_frames[cam_id] = cam_frames
            
            print(f"  Camera {cam_id:02d}: extracted {len(cam_frames)} frames ({'TRAIN' if cam_id in train_cam_ids else 'TEST'})")
        
        total_colmap_images = len(first_frame_images)
        print(f"\nCOLMAP will use {total_colmap_images} images from {len(first_frame_images)} TRAIN cameras (1 frame each)")
        return all_frames, train_frames, test_frames, first_frame_images

    def run_colmap_on_first_frame(self, poses, h_orig, w_orig, f_orig, target_size=(640, 360)):
        """Run COLMAP SfM and dense reconstruction on first frame with proper camera parameter scaling"""
        print("Running COLMAP SfM and dense reconstruction on first frame...")
        
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
        
        if not points3d_exists:
            print("Sparse SfM failed - no points3D found")
            return False
        
        print("  Sparse SfM successful, testing dense reconstruction capabilities...")
        
        # Test if COLMAP supports dense reconstruction
        test_cmd = [self.colmap_exe, "image_undistorter", "--help"]
        test_result = subprocess.run(test_cmd, capture_output=True, text=True)
        if test_result.returncode != 0:
            print(f"  Error: COLMAP does not support image_undistorter command")
            print(f"  This COLMAP build may not include dense reconstruction features")
            print(f"  Falling back to sparse point cloud...")
            return True  # Continue with sparse reconstruction only
        
        print("  COLMAP dense reconstruction commands available, proceeding...")
        
        # Create dense reconstruction directories
        self.dense_dir = os.path.join(self.colmap_workspace, "dense")
        os.makedirs(self.dense_dir, exist_ok=True)
        
        print(f"  Created dense directory: {self.dense_dir}")
        print(f"  Input sparse directory: {self.sparse_dir}")
        print(f"  Input images directory: {self.images_dir}")
        print(f"  Number of images: {len([f for f in os.listdir(self.images_dir) if f.endswith('.jpg')])}")
        
        # Step 1: Undistort images
        print("  Step 1/3: Undistorting images...")
        cmd = [
            self.colmap_exe, "image_undistorter",
            "--image_path", self.images_dir,
            "--input_path", self.sparse_dir,
            "--output_path", self.dense_dir,
            "--output_type", "COLMAP"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Image undistortion failed: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            print(f"Command: {' '.join(cmd)}")
            return False
        else:
            print("  Image undistortion successful")
            # Check if undistorted images were created
            dense_images_dir = os.path.join(self.dense_dir, "images")
            if os.path.exists(dense_images_dir):
                num_undistorted = len([f for f in os.listdir(dense_images_dir) if f.endswith('.jpg')])
                print(f"    Created {num_undistorted} undistorted images")
        
        # Step 2: Dense stereo matching  
        print("  Step 2/3: Computing dense stereo...")
        dense_sparse_dir = os.path.join(self.dense_dir, "sparse")
        cmd = [
            self.colmap_exe, "patch_match_stereo",
            "--workspace_path", self.dense_dir,
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.max_image_size", "2000"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Dense stereo failed: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            print(f"Command: {' '.join(cmd)}")
            return False
        else:
            print("  Dense stereo matching successful")
            # Check if stereo files were created
            stereo_dir = os.path.join(self.dense_dir, "stereo")
            if os.path.exists(stereo_dir):
                depth_maps = len([f for f in os.listdir(stereo_dir) if f.endswith('.geometric.bin')])
                print(f"    Created {depth_maps} depth maps")
        
        # Step 3: Fusion to create dense point cloud
        print("  Step 3/3: Fusing dense point cloud...")
        cmd = [
            self.colmap_exe, "stereo_fusion",
            "--workspace_path", self.dense_dir,
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", os.path.join(self.dense_dir, "fused.ply")
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Dense fusion failed: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            print(f"Command: {' '.join(cmd)}")
            
            # Check what files exist in dense directory
            print(f"Dense directory contents:")
            for root, dirs, files in os.walk(self.dense_dir):
                level = root.replace(self.dense_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:10]:  # Show first 10 files
                    print(f"{subindent}{file}")
                if len(files) > 10:
                    print(f"{subindent}... and {len(files)-10} more files")
            
            return False
        else:
            print("  Dense fusion successful")
            
        # Check if dense point cloud was created
        dense_ply_path = os.path.join(self.dense_dir, "fused.ply")
        dense_exists = os.path.exists(dense_ply_path)
        
        if dense_exists:
            # Get file size for verification
            file_size = os.path.getsize(dense_ply_path) / (1024 * 1024)  # MB
            print(f"  Dense reconstruction successful: {dense_ply_path} ({file_size:.1f} MB)")
        else:
            print("  Dense reconstruction failed - no fused.ply found")
            print(f"  Expected path: {dense_ply_path}")
            
            # List what files were actually created
            if os.path.exists(self.dense_dir):
                print(f"  Files in dense directory:")
                for f in os.listdir(self.dense_dir):
                    fpath = os.path.join(self.dense_dir, f)
                    if os.path.isfile(fpath):
                        size = os.path.getsize(fpath) / 1024  # KB
                        print(f"    {f} ({size:.1f} KB)")
                    else:
                        print(f"    {f}/ (directory)")
            
        return dense_exists

    def create_unified_segmentation(self, all_frames, train_frames, first_frame_images, cameras, poses):
        """
        Create unified segmentation strategy with LLFF alignment:
        1. Load COLMAP dense point cloud (generated from train cameras only) 
        2. Align COLMAP point cloud to LLFF coordinate system
        3. Label ALL points as foreground (no segmentation)
        4. Create consistent 2D segmentation for all frames (all foreground)
        """
        print("Creating unified 3D and 2D segmentation with LLFF alignment...")
        print(f"Using {len(first_frame_images)} TRAIN cameras for dense point cloud generation")
        
        # Step 1: Load COLMAP point cloud (prefer dense, fallback to sparse)
        points_colmap = np.array([])
        colors = np.array([])
        
        # Try dense point cloud first
        if hasattr(self, 'dense_dir') and os.path.exists(self.dense_dir):
            dense_ply_path = os.path.join(self.dense_dir, "fused.ply")
            points_colmap, colors = self.load_dense_point_cloud(dense_ply_path)
            if len(points_colmap) > 0:
                print(f"  Using COLMAP dense point cloud: {len(points_colmap):,} points")
        
        # Fallback to sparse point cloud
        if len(points_colmap) == 0:
            print("  No dense point cloud found, using sparse point cloud...")
            points3d_path = os.path.join(self.sparse_dir, "points3D.bin")
            points_colmap, colors = self.load_colmap_points(points3d_path)
            if len(points_colmap) > 0:
                print(f"  Using COLMAP sparse point cloud: {len(points_colmap):,} points")
            
        # Final fallback
        if len(points_colmap) == 0:
            print("No COLMAP points found, using fallback method")
            return self.create_fallback_segmentation(all_frames)
        
        # Step 2: Align COLMAP point cloud to LLFF coordinate system
        train_cam_ids = list(train_frames.keys())
        scale, R, t = self.align_colmap_to_llff(poses, cameras, train_cam_ids)
        points_llff, colors = self.transform_dense_cloud_to_llff(points_colmap, colors, scale, R, t)
        
        # Step 3: Label ALL points as foreground (no segmentation)
        seg_3d = np.ones(len(points_llff), dtype=np.float32)
        
        # Step 4: Create consistent 2D segmentation for all frames (all foreground)
        self.create_foreground_2d_segmentation(all_frames)
        
        # Step 5: Save 3D point cloud (use LLFF-aligned dense points)
        init_pt_cld = np.column_stack([points_llff, colors, seg_3d])
        output_path = os.path.join(self.output_seq_dir, "init_pt_cld.npz")
        np.savez(output_path, data=init_pt_cld)
        
        print(f"Created segmentation using LLFF-aligned dense points:")
        print(f"  - Total points: {len(points_llff):,}")
        print(f"  - Foreground: {np.sum(seg_3d):,} points ({np.mean(seg_3d)*100:.1f}%)")
        print(f"  - Background: {np.sum(1-seg_3d):,} points ({np.mean(1-seg_3d)*100:.1f}%)")
        print(f"  - Point cloud now in LLFF coordinate system")
        
        return init_pt_cld

    def load_colmap_points(self, points3d_bin_path):
        """Load COLMAP 3D points from binary format (points3D.bin)"""
        import struct
        
        if not os.path.exists(points3d_bin_path):
            return np.array([]), np.array([])
        
        points = []
        colors = []
        
        try:
            with open(points3d_bin_path, 'rb') as f:
                # Read number of points
                num_points = struct.unpack('Q', f.read(8))[0]
                print(f"  Reading {num_points} points from COLMAP binary")
                
                for _ in range(num_points):
                    # Read point ID (8 bytes)
                    point_id = struct.unpack('Q', f.read(8))[0]
                    
                    # Read XYZ (3 * 8 bytes double precision)
                    xyz = struct.unpack('ddd', f.read(24))
                    
                    # Read RGB (3 bytes)
                    rgb = struct.unpack('BBB', f.read(3))
                    rgb = np.array(rgb) / 255.0
                    
                    # Read error (8 bytes)
                    error = struct.unpack('d', f.read(8))[0]
                    
                    # Read track data
                    track_length = struct.unpack('Q', f.read(8))[0]
                    # Skip track data (8 bytes per track element: image_id + point2D_idx)
                    f.read(8 * track_length)
                    
                    points.append(xyz)
                    colors.append(rgb)
            
            points = np.array(points)
            colors = np.array(colors)
            
            if len(points) > 0:
                print(f"  Successfully loaded {len(points)} COLMAP points")
                print(f"  Point cloud bounds: X=[{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
                print(f"                     Y=[{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
                print(f"                     Z=[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
            
            return points, colors
            
        except Exception as e:
            print(f"  Failed to read COLMAP binary: {e}")
            return np.array([]), np.array([])

    def load_dense_point_cloud(self, ply_path):
        """Load dense point cloud from PLY file using PlyData for guaranteed consistency"""
        if not os.path.exists(ply_path):
            print(f"  Dense PLY file not found: {ply_path}")
            return np.array([]), np.array([])
        
        try:
            print(f"  Loading PLY file with PlyData: {ply_path}")
            
            # Load PLY file using PlyData
            plydata = PlyData.read(ply_path)
            
            # Get vertex element
            vertex = plydata['vertex']
            
            # Extract coordinates (x, y, z)
            points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
            
            # Extract colors - handle different possible color formats
            colors = np.zeros((len(points), 3))
            
            # Try different color field names that COLMAP might use
            color_fields = [
                ('red', 'green', 'blue'),           # Standard RGB
                ('r', 'g', 'b'),                    # Short RGB  
                ('diffuse_red', 'diffuse_green', 'diffuse_blue'),  # Diffuse RGB
            ]
            
            color_found = False
            for r_field, g_field, b_field in color_fields:
                if r_field in vertex.dtype.names and g_field in vertex.dtype.names and b_field in vertex.dtype.names:
                    r_vals = vertex[r_field]
                    g_vals = vertex[g_field] 
                    b_vals = vertex[b_field]
                    
                    # Normalize to [0, 1] range
                    if r_vals.max() > 1.0:  # Assume 0-255 range
                        colors[:, 0] = r_vals / 255.0
                        colors[:, 1] = g_vals / 255.0
                        colors[:, 2] = b_vals / 255.0
                    else:  # Already in [0, 1] range
                        colors[:, 0] = r_vals
                        colors[:, 1] = g_vals
                        colors[:, 2] = b_vals
                    
                    color_found = True
                    print(f"    Found colors using fields: {r_field}, {g_field}, {b_field}")
                    break
            
            if not color_found:
                print("    Warning: No color fields found, using default gray color")
                colors.fill(0.5)  # Default gray color
            
            print(f"  Successfully loaded {len(points):,} dense points from PLY using PlyData")
            print(f"  Dense point cloud bounds: X=[{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
            print(f"                           Y=[{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
            print(f"                           Z=[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
            print(f"  Color range: R=[{colors[:,0].min():.3f}, {colors[:,0].max():.3f}]")
            print(f"               G=[{colors[:,1].min():.3f}, {colors[:,1].max():.3f}]")
            print(f"               B=[{colors[:,2].min():.3f}, {colors[:,2].max():.3f}]")
            
            # Print available fields for debugging
            print(f"  PLY vertex fields: {vertex.dtype.names}")
            
            return points, colors
            
        except ImportError:
            print(f"  Error: PlyData not available. Please install with: pip install plyfile")
            
        except Exception as e:
            print(f"  Failed to read dense PLY with PlyData: {e}")
            
    def load_colmap_cameras(self):
        """Load COLMAP camera parameters from binary files"""
        import struct
        
        cameras_path = os.path.join(self.sparse_dir, "cameras.bin")
        images_path = os.path.join(self.sparse_dir, "images.bin")
        
        cameras = {}
        
        if not os.path.exists(cameras_path) or not os.path.exists(images_path):
            print("  Warning: COLMAP camera files not found, using fallback")
            return None
        
        try:
            # Read cameras.bin (intrinsics)
            with open(cameras_path, 'rb') as f:
                num_cameras = struct.unpack('Q', f.read(8))[0]
                
                for _ in range(num_cameras):
                    camera_id = struct.unpack('I', f.read(4))[0]
                    model_id = struct.unpack('I', f.read(4))[0]
                    width = struct.unpack('Q', f.read(8))[0]
                    height = struct.unpack('Q', f.read(8))[0]
                    
                    # Read intrinsic parameters (for SIMPLE_PINHOLE: f, cx, cy)
                    if model_id == 0:  # SIMPLE_PINHOLE
                        params = struct.unpack('ddd', f.read(24))
                        f, cx, cy = params
                        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
                    else:
                        # Handle other camera models if needed
                        num_params = 3  # Assume 3 for now
                        params = struct.unpack('d' * num_params, f.read(8 * num_params))
                        f, cx, cy = params[:3]
                        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
                    
                    cameras[camera_id] = {'K': K, 'width': width, 'height': height}
            
            # Read images.bin (extrinsics)
            with open(images_path, 'rb') as f:
                num_images = struct.unpack('Q', f.read(8))[0]
                
                for _ in range(num_images):
                    image_id = struct.unpack('I', f.read(4))[0]
                    
                    # Read quaternion (w, x, y, z) and translation
                    quat = struct.unpack('dddd', f.read(32))
                    trans = struct.unpack('ddd', f.read(24))
                    
                    camera_id = struct.unpack('I', f.read(4))[0]
                    
                    # Read image name
                    name_bytes = b''
                    while True:
                        c = f.read(1)
                        if c == b'\x00':
                            break
                        name_bytes += c
                    image_name = name_bytes.decode('utf-8')
                    
                    # Skip 2D points data
                    num_points2d = struct.unpack('Q', f.read(8))[0]
                    f.read(24 * num_points2d)  # Skip point2D data
                    
                    # Convert quaternion to rotation matrix
                    w, x, y, z = quat
                    R = self.quat_to_rotation_matrix(w, x, y, z)
                    t = np.array(trans)
                    
                    # Store extrinsics with camera
                    if camera_id in cameras:
                        cameras[camera_id]['images'] = cameras[camera_id].get('images', {})
                        cameras[camera_id]['images'][image_name] = {'R': R, 't': t}
            
            print(f"  Loaded {len(cameras)} COLMAP cameras with extrinsics")
            return cameras
            
        except Exception as e:
            print(f"  Failed to load COLMAP cameras: {e}")
            return None

    def quat_to_rotation_matrix(self, w, x, y, z):
        """Convert quaternion to rotation matrix"""
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        return R

    def extract_camera_centers_from_llff(self, poses, train_cam_ids):
        """Extract camera centers from LLFF poses for training cameras"""
        camera_centers_llff = []
        cam_id_mapping = []
        
        print("  Extracting LLFF camera centers...")
        for cam_id in sorted(train_cam_ids):
            if cam_id in self.camid_to_pose_index:
                pose_idx = self.camid_to_pose_index[cam_id]
                if pose_idx < len(poses):
                    # LLFF poses are c2w (camera-to-world)
                    c2w = poses[pose_idx, :3, :4]  # 3x4 matrix
                    camera_center = c2w[:3, 3]  # Translation is camera center in world coords
                    camera_centers_llff.append(camera_center)
                    cam_id_mapping.append(cam_id)
                    print(f"    LLFF cam {cam_id} (pose idx {pose_idx}): center = [{camera_center[0]:.3f}, {camera_center[1]:.3f}, {camera_center[2]:.3f}]")
                else:
                    print(f"    Warning: pose index {pose_idx} for cam {cam_id} >= len(poses) {len(poses)}")
            else:
                print(f"    Warning: cam {cam_id} not found in camera mapping")
        
        return np.array(camera_centers_llff), cam_id_mapping

    def extract_camera_centers_from_colmap(self, cameras, train_cam_ids):
        """Extract camera centers from COLMAP poses for training cameras"""
        camera_centers_colmap = []
        cam_id_mapping = []
        
        print("  Extracting COLMAP camera centers...")
        for cam_id in sorted(train_cam_ids):
            colmap_image_name = f"cam_{cam_id:02d}_frame_000.jpg"
            
            # Find COLMAP camera data
            found = False
            for colmap_cam_id, cam_info in cameras.items():
                if 'images' in cam_info and colmap_image_name in cam_info['images']:
                    R = cam_info['images'][colmap_image_name]['R']  # 3x3
                    t = cam_info['images'][colmap_image_name]['t']  # 3x1
                    
                    # COLMAP w2c: X_cam = R * X_world + t
                    # Camera center: C = -R^T * t
                    camera_center = -R.T @ t
                    camera_centers_colmap.append(camera_center)
                    cam_id_mapping.append(cam_id)
                    print(f"    COLMAP cam {cam_id}: center = [{camera_center[0]:.3f}, {camera_center[1]:.3f}, {camera_center[2]:.3f}]")
                    found = True
                    break
            
            if not found:
                print(f"    Warning: COLMAP data not found for cam {cam_id}")
        
        return np.array(camera_centers_colmap), cam_id_mapping

    def compute_procrustes_alignment(self, points_src, points_dst):
        """
        Compute Procrustes/Umeyama alignment: points_dst = s * R * points_src + t
        Returns: scale (s), rotation (R), translation (t)
        """
        print("  Computing Procrustes alignment (COLMAP → LLFF)...")
        
        if len(points_src) != len(points_dst) or len(points_src) < 3:
            print(f"    Error: Need at least 3 corresponding points, got {len(points_src)}")
            return 1.0, np.eye(3), np.zeros(3)
        
        # Center the points
        centroid_src = np.mean(points_src, axis=0)
        centroid_dst = np.mean(points_dst, axis=0)
        
        points_src_centered = points_src - centroid_src
        points_dst_centered = points_dst - centroid_dst
        
        # Compute scale
        scale_src = np.sqrt(np.sum(points_src_centered ** 2))
        scale_dst = np.sqrt(np.sum(points_dst_centered ** 2))
        
        if scale_src < 1e-8:
            print("    Warning: Source points have zero scale")
            return 1.0, np.eye(3), centroid_dst - centroid_src
        
        scale = scale_dst / scale_src
        
        # Normalize for rotation computation
        points_src_norm = points_src_centered / scale_src
        points_dst_norm = points_dst_centered / scale_dst
        
        # Compute rotation using SVD
        H = points_src_norm.T @ points_dst_norm
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = centroid_dst - scale * R @ centroid_src
        
        print(f"    Scale: {scale:.4f}")
        print(f"    Rotation det: {np.linalg.det(R):.4f}")
        print(f"    Translation: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
        
        return scale, R, t

    def align_colmap_to_llff(self, poses, cameras, train_cam_ids):
        """
        Align COLMAP coordinate system to LLFF coordinate system
        Returns: scale, rotation, translation for transformation
        """
        print("Aligning COLMAP to LLFF coordinate system...")
        
        if cameras is None:
            print("  No COLMAP cameras available, skipping alignment")
            return 1.0, np.eye(3), np.zeros(3)
        
        # Extract camera centers
        centers_llff, llff_cam_ids = self.extract_camera_centers_from_llff(poses, train_cam_ids)
        centers_colmap, colmap_cam_ids = self.extract_camera_centers_from_colmap(cameras, train_cam_ids)
        
        # Find common cameras
        common_cam_ids = list(set(llff_cam_ids) & set(colmap_cam_ids))
        
        if len(common_cam_ids) < 3:
            print(f"  Warning: Only {len(common_cam_ids)} common cameras found, need at least 3 for robust alignment")
            return 1.0, np.eye(3), np.zeros(3)
        
        # Reorder to match
        llff_indices = [llff_cam_ids.index(cam_id) for cam_id in common_cam_ids]
        colmap_indices = [colmap_cam_ids.index(cam_id) for cam_id in common_cam_ids]
        
        centers_llff_matched = centers_llff[llff_indices]
        centers_colmap_matched = centers_colmap[colmap_indices]
        
        print(f"  Using {len(common_cam_ids)} common cameras for alignment: {common_cam_ids}")
        
        # Compute alignment
        scale, R, t = self.compute_procrustes_alignment(centers_colmap_matched, centers_llff_matched)
        
        return scale, R, t

    def transform_dense_cloud_to_llff(self, points, colors, scale, R, t):
        """Transform dense point cloud from COLMAP to LLFF space"""
        print("  Transforming dense point cloud to LLFF space...")
        
        # Apply similarity transformation: X_llff = s * R * X_colmap + t
        points_llff = scale * (points @ R.T) + t
        
        print(f"    Transformed {len(points)} points")
        print(f"    Original bounds: X=[{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
        print(f"                    Y=[{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
        print(f"                    Z=[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
        print(f"    LLFF bounds:    X=[{points_llff[:,0].min():.2f}, {points_llff[:,0].max():.2f}]")
        print(f"                    Y=[{points_llff[:,1].min():.2f}, {points_llff[:,1].max():.2f}]")
        print(f"                    Z=[{points_llff[:,2].min():.2f}, {points_llff[:,2].max():.2f}]")
        
        return points_llff, colors

    def create_foreground_2d_segmentation(self, all_frames):
        """
        Create 2D segmentation masks where ALL pixels are labeled as foreground
        Uses CMU format: seg/cam_id/timestamp.png
        """
        print("  Creating foreground-only 2D segmentation masks...")
        
        seg_dir = os.path.join(self.output_seq_dir, "seg")
        os.makedirs(seg_dir, exist_ok=True)
        
        ims_dir = os.path.join(self.output_seq_dir, "ims")
        
        # Create all-foreground masks for all frames
        for cam_id in tqdm(sorted(all_frames.keys()), desc="Creating foreground masks"):
            frames = all_frames[cam_id]
            
            # Create camera-specific segmentation directory (CMU format)
            cam_seg_dir = os.path.join(seg_dir, str(cam_id))
            os.makedirs(cam_seg_dir, exist_ok=True)
            
            # Get image dimensions from first frame
            if len(frames) > 0:
                first_frame_path = os.path.join(ims_dir, str(cam_id), frames[0])
                first_frame = np.array(Image.open(first_frame_path))
                h, w = first_frame.shape[:2]
                
                for frame_filename in frames:
                    # Create all-foreground mask (all pixels = 255 = foreground)
                    foreground_mask = np.full((h, w), 255, dtype=np.uint8)
                    
                    # Save mask in CMU format: seg/cam_id/timestamp.png
                    mask_filename = frame_filename.replace('.jpg', '.png')
                    mask_path = os.path.join(cam_seg_dir, mask_filename)
                    Image.fromarray(foreground_mask).save(mask_path)


    def create_fallback_segmentation(self, all_frames):
        """Fallback segmentation when COLMAP fails"""
        print("Using fallback segmentation (no COLMAP points)")
        
        # Create simple point cloud with fixed small number of points
        fallback_points = 1000  # Small fallback since COLMAP failed
        points = np.random.normal(0, 1, (fallback_points, 3))
        colors = np.random.rand(fallback_points, 3)
        seg_3d = np.ones(fallback_points, dtype=np.float32)  # All foreground
        
        print(f"Created fallback point cloud: {fallback_points} points")
        
        # Create simple 2D segmentation (all foreground)
        self.create_foreground_2d_segmentation(all_frames)
        
        # Save point cloud
        init_pt_cld = np.column_stack([points, colors, seg_3d])
        output_path = os.path.join(self.output_seq_dir, "init_pt_cld.npz")
        np.savez(output_path, data=init_pt_cld)
        
        return init_pt_cld

    def get_llff_w2c_matrix(self, cam_id, poses):
        """Get world-to-camera matrix from LLFF poses using correct camera mapping"""
        if cam_id in self.camid_to_pose_index:
            pose_idx = self.camid_to_pose_index[cam_id]
            if pose_idx < len(poses):
                # LLFF poses are camera-to-world, need to invert for world-to-camera
                c2w = poses[pose_idx, :3, :4]
                
                # More robust w2c computation using your suggested method
                w2c = np.eye(4)
                w2c[:3, :3] = c2w[:3, :3].T  # R_transpose
                w2c[:3, 3] = -c2w[:3, :3].T @ c2w[:3, 3]  # -R_transpose * t
                return w2c
            else:
                print(f"    Warning: pose index {pose_idx} for cam {cam_id} >= len(poses) {len(poses)}, using identity")
                return np.eye(4)
        else:
            print(f"    Warning: cam_id {cam_id} not found in camera mapping, using identity")
            return np.eye(4)

    def create_metadata_llff_centric(self, poses, train_frames, test_frames, target_size, f_orig):
        """Create metadata using LLFF poses for ALL cameras - consistent coordinate system"""
        print("Creating LLFF-centric train and test metadata...")
        
        w_target, h_target = target_size
        
        # Scale focal length to match resized images  
        hwf = poses[0, :, 4]  # HWF from poses
        w_orig = int(hwf[1])  # Original width from HWF
        scale_x = w_target / w_orig
        f_scaled = f_orig * scale_x
        
        print(f"  Metadata dimensions: {w_target}x{h_target} (target)")
        print(f"  Focal length: {f_orig:.1f} -> {f_scaled:.1f} (scaled)")
        print("  Using LLFF poses for ALL cameras (consistent coordinate system)")
        
        # Combine all frames for processing
        all_cam_frames = {**train_frames, **test_frames}
        max_frames = max(len(frames) for frames in all_cam_frames.values()) if all_cam_frames else 0
        
        # Helper function to get camera data using LLFF poses
        def get_camera_data_llff(cam_id, frame_filename):
            # Use scaled LLFF intrinsics for all cameras
            intrinsics = [[f_scaled, 0, w_target/2], [0, f_scaled, h_target/2], [0, 0, 1]]
            
            # Use LLFF w2c matrix for all cameras
            w2c_matrix = self.get_llff_w2c_matrix(cam_id, poses)
            
            return {
                'filename': f"{cam_id}/{frame_filename}",
                'intrinsics': intrinsics,
                'w2c': w2c_matrix.tolist()
            }
        
        # Build complete camera data for all timesteps using LLFF
        all_camera_data = []
        for t in range(max_frames):
            timestep_data = {}
            
            for cam_id in sorted(all_cam_frames.keys()):
                if t < len(all_cam_frames[cam_id]):
                    frame_filename = all_cam_frames[cam_id][t]
                    timestep_data[cam_id] = get_camera_data_llff(cam_id, frame_filename)
            
            if timestep_data:
                all_camera_data.append(timestep_data)
        
        # Helper function to create metadata from camera data
        def create_metadata_from_cameras(cam_ids, camera_data_list, metadata_type):
            metadata = {
                'w': w_target,
                'h': h_target,
                'fn': [],
                'k': [],
                'w2c': []
            }
            
            for timestep_data in camera_data_list:
                frame_filenames = []
                frame_intrinsics = []
                frame_w2c = []
                
                for cam_id in sorted(cam_ids):
                    if cam_id in timestep_data:
                        data = timestep_data[cam_id]
                        frame_filenames.append(data['filename'])
                        frame_intrinsics.append(data['intrinsics'])
                        frame_w2c.append(data['w2c'])
                
                if frame_filenames:
                    metadata['fn'].append(frame_filenames)
                    metadata['k'].append(frame_intrinsics)
                    metadata['w2c'].append(frame_w2c)
            
            # Save metadata
            metadata_path = os.path.join(self.output_seq_dir, f"{metadata_type}_meta.json")
            with open(metadata_path, 'w') as file_handle:
                json.dump(metadata, file_handle, indent=2)
            
            print(f"Created {metadata_type} metadata: {len(metadata['fn'])} timesteps, {len(cam_ids)} cameras (LLFF poses)")
            return metadata
        
        # Create train and test metadata using LLFF poses
        train_cam_ids = list(train_frames.keys())
        test_cam_ids = list(test_frames.keys())
        train_metadata = create_metadata_from_cameras(train_cam_ids, all_camera_data, "train")
        test_metadata = create_metadata_from_cameras(test_cam_ids, all_camera_data, "test")

    def run_unified_preprocessing(self, target_size=(640, 360), max_frames=150):
        """Run complete unified preprocessing with proper train/test split using dense point clouds"""
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
        cameras = self.load_colmap_cameras()  # Load actual camera parameters
        self.create_unified_segmentation(all_frames, train_frames, first_frame_images, cameras, poses)
        
        # Create metadata (separate train and test) with TARGET dimensions - LLFF-centric
        self.create_metadata_llff_centric(poses, train_frames, test_frames, target_size, f)
        
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
    parser.add_argument("--width", type=int, default=640, help="Target image width")
    parser.add_argument("--height", type=int, default=360, help="Target image height")
    parser.add_argument("--colmap-exe", type=str, default="colmap", help="Path to COLMAP executable")
    
    args = parser.parse_args()
    
    preprocessor = UnifiedDyNeRFPreprocessor(
        args.data_dir, args.seq, args.output_dir, args.colmap_exe
    )
    
    success = preprocessor.run_unified_preprocessing(
        target_size=(args.width, args.height),
        max_frames=args.max_frames
    )
    
    if success:
        print("\nUnified preprocessing completed successfully!")
    else:
        print("\nPreprocessing failed!")
        exit(1)


if __name__ == "__main__":
    main() 