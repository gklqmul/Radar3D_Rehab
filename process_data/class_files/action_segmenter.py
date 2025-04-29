import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os

class ActionSegmenter:
    def __init__(self, max_frame_diff=100, frame_count=500, min_segment_length=5, long_stationary_threshold=60, smooth_sigma=1.0):
        """
        init ActionSegmenter
        :param threshold_T: threshold for motion energy
        :param min_segment_length: minimum length of action segment
        :param long_stationary_threshold: threshold for long stationary segment
        :param smooth_sigma: parameter for gaussian smoothing
        """
        self.threshold_T = None
        self.min_segment_length = min_segment_length
        self.long_stationary_threshold = long_stationary_threshold
        self.smooth_sigma = smooth_sigma
        self.frame_count = frame_count
        self.max_frame_diff = max_frame_diff

    def segment_actions(self, skeleton_frames):
        """
        Split the action sequence
        :param skeleton_frames: Sequence of skeleton joint coordinates, shape (N, M, 3), where N is the number of frames, M is the number of joints, and 3 represents (x, y, z) coordinates
        :return: List of segmented action sequences, each action segment is a list of frame indices
        """
        # Data preprocessing: smoothing
        skeleton_frames = self._smooth_frames(skeleton_frames)

        # Compute overall motion energy for each frame
        motion_energy = self._calculate_motion_energy(skeleton_frames)

        # Dynamically compute the threshold threshold_T
        self.threshold_T = self._calculate_threshold_T(motion_energy)
        
        # Detect stationary frames
        is_stationary = motion_energy < self.threshold_T

        # Split action sequences
        action_segments = self._split_actions(is_stationary)

        # Detect long stationary segments
        long_stationary_segments = self._find_long_stationary_segments(is_stationary)

        # Merge action segments until encountering long stationary segments
        action_segments = self._merge_until_long_stationary(long_stationary_segments)

        return motion_energy, action_segments
    
    def _calculate_threshold_T(self, motion_energy):
        
        """
        Dynamically compute the threshold threshold_T
        :param motion_energy: Motion energy of each frame
        :return: Dynamically computed threshold
        """
        threshold_T_values = []

        # Sliding window processing of motion energy sequence
        for i in range(len(motion_energy) - self.long_stationary_threshold + 1):
            window = motion_energy[i:i + self.long_stationary_threshold]
            # Compute the difference between the maximum and minimum values in the window
            frame_diff = np.max(window) - np.min(window)

            # If the difference is less than max_frame_diff, consider it a valid window
            if frame_diff < self.max_frame_diff:
                # Compute a higher value within the window (e.g., 90th percentile)
                threshold_T_values.append(np.percentile(window, 100))

        # If no valid window is found, use the global 90th percentile as the threshold
        if len(threshold_T_values) == 0:
            threshold_T = np.percentile(motion_energy, 90)
        else:
            # Use the maximum of all valid window values as the global threshold
            threshold_T = np.max(threshold_T_values)

        return threshold_T
    
    def _smooth_frames(self, skeleton_frames):
        """
        Apply smoothing to skeleton joint coordinates
        :param skeleton_frames: Sequence of skeleton joint coordinates
        :return: Smoothed sequence of skeleton joint coordinates
        """
        smoothed_frames = np.zeros_like(skeleton_frames)
        for joint in range(skeleton_frames.shape[1]):
            for dim in range(3):  # x, y, z dimensions
                smoothed_frames[:, joint, dim] = gaussian_filter1d(
                    skeleton_frames[:, joint, dim], sigma=self.smooth_sigma
                )
        return smoothed_frames

    def _calculate_motion_energy(self, skeleton_frames):
        """
        Compute overall motion energy for each frame
        :param skeleton_frames: Sequence of skeleton joint coordinates
        :return: Motion energy of each frame, shape (N,)
        """
        # Compute frame-to-frame differences, combining x, y, and z coordinates
        velocity = np.linalg.norm(np.diff(skeleton_frames, axis=0), axis=2)  # Shape (N-1, M)
        # Compute overall motion energy for each frame
        motion_energy = np.sum(velocity, axis=1)  # Shape (N-1,)
        # Insert 0 at the first frame to maintain frame count consistency
        motion_energy = np.insert(motion_energy, 0, 0)
        return motion_energy

    def _split_actions(self, is_stationary):
        """
        Split action sequences based on stationary frames
        :param is_stationary: Boolean array indicating whether each frame is stationary
        :return: List of segmented action sequences
        """
        action_segments = []
        current_segment = []

        for frame_idx, stationary in enumerate(is_stationary):
            if stationary:
                # Detect a stationary frame, end the current action segment
                if current_segment:
                    action_segments.append(current_segment)
                    current_segment = []
            else:
                # Continue the current action segment
                current_segment.append(frame_idx)

        # Add the last action segment
        if current_segment:
            action_segments.append(current_segment)

        return action_segments

    def _find_long_stationary_segments(self, is_stationary):
        """
        Detect long stationary segments (longer than long_stationary_threshold frames)
        :param is_stationary: Boolean array indicating whether each frame is stationary
        :return: List of start and end frame indices of long stationary segments
        """
        long_stationary_segments = []
        start = None

        for frame_idx, stationary in enumerate(is_stationary):
            if stationary and start is None:
                start = frame_idx  # Record start frame of long stationary segment
            elif not stationary and start is not None:
                if frame_idx - start >= self.long_stationary_threshold:
                    long_stationary_segments.append((start, frame_idx - 1))
                start = None  # Reset start frame

        # Check the last stationary segment
        if start is not None and len(is_stationary) - start >= self.long_stationary_threshold:
            long_stationary_segments.append((start, len(is_stationary) - 1))

        return long_stationary_segments

    def _merge_until_long_stationary(self, long_stationary_segments):
        """
        Split action segments based on long stationary frames and exclude long stationary frames
        :param long_stationary_segments: List of start and end frame indices of long stationary segments
        :return: List of segmented action sequences
        """
        # Extract start and end frames of long stationary segments
        long_stationary_frames = []
        for start, end in long_stationary_segments:
            long_stationary_frames.append((start, end))

        # Add boundary frames
        long_stationary_frames.append((-1, -1))  # Frame before the first frame
        long_stationary_frames.append((self.frame_count, self.frame_count))  # Frame after the last frame

        # Sort by start frame
        long_stationary_frames.sort()

        # Split action segments
        processed_segments = []
        for i in range(len(long_stationary_frames) - 1):
            # End frame of the current long stationary segment
            current_end = long_stationary_frames[i][1]
            # Start frame of the next long stationary segment
            next_start = long_stationary_frames[i + 1][0]

            # The start frame of the action segment is the end frame of the current long stationary segment + 1
            # The end frame of the action segment is the start frame of the next long stationary segment - 1
            start = current_end + 1
            end = next_start - 1

            # Ensure the action segment is valid
            if start <= end:
                processed_segments.append((start, end))

        return processed_segments


