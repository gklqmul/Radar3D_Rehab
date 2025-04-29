import cv2
import numpy as np
import pykinect_azure as pykinect
import os

class KinectSkeletonProcessor:
    def __init__(self, video_filename, save_dir="dataset/env1/subjects/"):
        self.video_filename = video_filename
        self.save_dir = save_dir
        self.device = None
        self.body_tracker = None
        # Initialize latest_joints to store the latest valid joint data
        self.latest_joints = {
            'body': np.zeros((32, 3)),
            'color': np.zeros((32, 3)),
            'depth': np.zeros((32, 3)),
            'depth_converted': np.zeros((32, 3))
        }
        self.all_skeletons = {"body": [], "color": [], "depth": [], "depth_converted": []}
        os.makedirs(self.save_dir, exist_ok=True)

    def initialize_kinect(self):
        """Initialize Kinect playback and body tracker."""
        pykinect.initialize_libraries(track_body=True)
        self.device = pykinect.start_playback(self.video_filename)
        playback_calibration = self.device.get_calibration()
        self.body_tracker = pykinect.start_body_tracker(calibration=playback_calibration)

    def get_skeleton_data(self, body_frame, depth_image, transformed_points_map, points_map):
        """Extract 3D skeleton data from Kinect frame."""
        skeletons = {"body": [], "color": [], "depth": [], "depth_converted": []}

        if body_frame.get_num_bodies() == 0:
            return None
        body_id = 0
        color_skeleton_2d = body_frame.get_body2d(body_id, pykinect.K4A_CALIBRATION_TYPE_COLOR).numpy()
        depth_skeleton_2d = body_frame.get_body2d(body_id, pykinect.K4A_CALIBRATION_TYPE_DEPTH).numpy()
        skeleton_3d = body_frame.get_body(body_id).numpy()  # (32, 4)

        body_joints, color_joints, depth_joints, depth_converted_joints = [], [], [], []
        for joint in range(pykinect.K4ABT_JOINT_COUNT):
            color_joint_2d = color_skeleton_2d[joint, :]
            depth_joint_2d = depth_skeleton_2d[joint, :]

            dh, dw = depth_image.shape  # Convert depth 2D to 3D

            dx, dy = int(depth_joint_2d[0]), int(depth_joint_2d[1])
            # limit the x, y coordinates to the image size
            dx = np.clip(dx, 0, dw - 1)
            dy = np.clip(dy, 0, dh - 1)
            depth = depth_image[dy, dx]
            depth_joint_float2 = pykinect.k4a_float2_t(depth_joint_2d)
            depth_joint_float3 = self.device.calibration.convert_2d_to_3d(
                depth_joint_float2, depth, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH
            )
            depth_converted_joint_3d = [depth_joint_float3.xyz.x, depth_joint_float3.xyz.y, depth_joint_float3.xyz.z]

            # Extract 3D coordinates from different spaces
            h, w, _ = transformed_points_map.shape
            x, y = int(color_joint_2d[0]), int(color_joint_2d[1])

            # limit the x, y coordinates to the image size
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)

            color_joint_3d = transformed_points_map[y, x, :]
            depth_joint_3d = points_map[dy, dx, :]
            body_joint_3d = skeleton_3d[joint, :3]

            # Add the joint coordinates to the respective lists
            body_joints.append(body_joint_3d)
            color_joints.append(color_joint_3d)
            depth_joints.append(depth_joint_3d)
            depth_converted_joints.append(depth_converted_joint_3d)

        skeletons["body"] = body_joints
        skeletons["color"] = color_joints
        skeletons["depth"] = depth_joints
        skeletons["depth_converted"] = depth_converted_joints

        return skeletons

    def interpolate_missing_joints(self, current_joints):
        """Fill missing joints using the latest available data for all four coordinate systems."""
        interpolated_joints = current_joints.copy()  # Copy current frame data

        for key in ['body', 'color', 'depth', 'depth_converted']:
            for i in range(32):  # Iterate over 32 joints
                if np.all(current_joints[key][i] == 0):
                    # If current joint data is missing (all zeros), fill with latest_joints
                    interpolated_joints[key][i] = self.latest_joints[key][i].copy()
                else:
                    # If current joint data is valid, update latest_joints
                    self.latest_joints[key][i] = current_joints[key][i].copy()

        return interpolated_joints

    def save_skeleton_npy(self):
        """Save all frames' skeleton data to NPY files for each coordinate system."""
        # Convert to numpy array of shape (num_frames, 32, 3)
        body_data = np.array(self.all_skeletons["body"])
        color_data = np.array(self.all_skeletons["color"])
        depth_data = np.array(self.all_skeletons["depth"])
        depth_converted_data = np.array(self.all_skeletons["depth_converted"])

        # Save each coordinate system data to separate NPY files
        np.save(os.path.join(self.save_dir, "body_skeleton.npy"), body_data)
        np.save(os.path.join(self.save_dir, "color_skeleton.npy"), color_data)
        np.save(os.path.join(self.save_dir, "depth_skeleton.npy"), depth_data)
        np.save(os.path.join(self.save_dir, "depth_converted_skeleton.npy"), depth_converted_data)

    def process_video(self):
        """Process the video and extract skeleton data."""
        self.initialize_kinect()

        while True:
            ret, capture = self.device.update()
            if not ret or capture is None:
                print("Invalid capture.")
                break

            body_frame = self.body_tracker.update(capture=capture)
            ret_color, color_image = capture.get_color_image()
            ret_depth, depth_image = capture.get_depth_image()

            if not ret_color or not ret_depth:
                continue

            ret_transformed_color, transformed_color_image = capture.get_transformed_color_image()
            ret_point, points = capture.get_pointcloud()
            ret_transformed_point, transformed_points = capture.get_transformed_pointcloud()

            if not ret_transformed_color or not ret_transformed_point or not ret_point:
                continue

            points_map = points.reshape((transformed_color_image.shape[0], transformed_color_image.shape[1], 3))
            transformed_points_map = transformed_points.reshape((color_image.shape[0], color_image.shape[1], 3))

            skeletons = self.get_skeleton_data(body_frame, depth_image, transformed_points_map, points_map)

            if skeletons is not None:
                # Interpolate missing joints using latest_joints
                skeletons = self.interpolate_missing_joints(skeletons)

                # Append skeletons for this frame
                for key in self.all_skeletons:
                    self.all_skeletons[key].append(skeletons[key])

        # Save all frames into separate NPY files for each coordinate system
        self.save_skeleton_npy()
        print("All frames saved as NPY files for each coordinate system.")
        