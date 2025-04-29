import os
import numpy as np
import scipy.io

from process.utils.transferCoordinate import transform_large_point_set

class DataAligner:
    def __init__(self, skeleton_folder, radar_file_path, save_folder):
        """
        Initialize the data aligner and load file paths.
        :param skeleton_folder: Directory containing Kinect data
        :param radar_file_path: Path to the radar .mat file
        :param save_folder: Path to the saving data
        """
        self.skeleton_folder = skeleton_folder
        self.radar_file_path = radar_file_path
        self.save_folder = save_folder + "/aligned"

        # Load skeleton and timestamps
        self.action_segment_file = os.path.join(skeleton_folder, 'action_segments.txt')
        self.skeleton_data = np.load(os.path.join(skeleton_folder, 'body_skeleton.npy'))
        self.skeleton_timestamps = np.load(os.path.join(skeleton_folder, 'timestamps.npy'))

    def load_radar_data(self):
        """
        Load radar data and timestamps from the .mat file.
        """
        mat_data = scipy.io.loadmat(self.radar_file_path)
        if 'pc' not in mat_data:
            raise ValueError(f"'{self.radar_file_path}' does not contain the 'pc' variable!")
        radar_struct = mat_data['pc']

        num_frames = radar_struct.shape[1]
        radar_timestamps = np.zeros(num_frames)
        for i in range(num_frames):
            radar_timestamps[i] = float(radar_struct[0, i]['timestamp'][0])

        return radar_timestamps, radar_struct

    def load_action_segments(self):
        """
        Load action segments from the text file.
        """
        action_segments = []
        with open(self.action_segment_file, 'r') as f:
            action_segments = [list(map(int, line.strip().split(','))) for line in f]
        return action_segments

    def align_and_segment_data(self):
        """
        Segment and align skeleton and radar data based on action segments.
        """
        radar_timestamps, radar_struct = self.load_radar_data()
        action_segments = self.load_action_segments()

        for segment_id, (start_frame, end_frame) in enumerate(action_segments, start=1):
            # Extract skeleton segment
            skeleton_segment = self.skeleton_data[start_frame:end_frame + 1]
            skeleton_segment_timestamps = self.skeleton_timestamps[start_frame:end_frame + 1]

            # Find radar segment indices
            start_idx = np.searchsorted(radar_timestamps, skeleton_segment_timestamps[0], side='left')
            end_idx = np.searchsorted(radar_timestamps, skeleton_segment_timestamps[-1], side='right') - 1

            if start_idx > end_idx:
                print(f"No radar data overlap for segment {segment_id}, skipping...")
                continue

            radar_segment = radar_struct[:, start_idx:end_idx + 1]
            radar_segment_timestamps = radar_timestamps[start_idx:end_idx + 1]

            # Align skeleton to radar by matching frame rates (30Hz -> 18Hz)
            matched_indices = np.searchsorted(skeleton_segment_timestamps, radar_segment_timestamps, side='left')
            matched_indices = np.clip(matched_indices, 0, len(skeleton_segment) - 1)
            aligned_skeleton_segment = skeleton_segment[matched_indices]
            if not os.path.exists(self.save_folder):
                os.mkdir(self.save_folder)
            # transform the skeleton data to radar coordinate
            aligned_skeleton_segment = transform_large_point_set(aligned_skeleton_segment)
            # Save aligned skeleton data
            aligned_skeleton_path = os.path.join(self.save_folder,f"aligned_skeleton_segment{segment_id:02d}.npy")
            np.save(aligned_skeleton_path, aligned_skeleton_segment)
            print(f"Saved aligned skeleton segment: {aligned_skeleton_path}")

            # Save radar segment
            radar_save_path = os.path.join(self.save_folder, f"aligned_radar_segment{segment_id:02d}.mat")
            scipy.io.savemat(radar_save_path, {'radar_data': radar_segment})
            print(f"Saved aligned radar segment: {radar_save_path}")


