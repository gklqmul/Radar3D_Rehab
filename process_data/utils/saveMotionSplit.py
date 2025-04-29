import os
import numpy as np
from process.class_files.action_segmenter import ActionSegmenter
from process.utils.FindAndSave import plot_motion_energy, save_action_segments, save_skeleton_segments


# Set root directory to find multiple `1/body_skeleton.npy`, `2/body_skeleton.npy` ... `6/body_skeleton.npy`
BASE_DIR = "dataset/env2/subjects/subject26/origal"

def find_skeleton_files(base_dir, num_folders=6):
    """
    Find `1/body_skeleton.npy` to `6/body_skeleton.npy` in `base_dir` and return a list of full file paths.
    """
    skeleton_files = []
    
    # Iterate over folders `1` to `6`
    for i in range(7, num_folders + 7):  
        folder_path = os.path.join(base_dir, str(i))
        skeleton_file = os.path.join(folder_path, "body_skeleton.npy")
        
        if os.path.exists(skeleton_file):
            skeleton_files.append((i, folder_path, skeleton_file))  # Store (folder index, folder path, file path)
        else:
            print(f"File not found: {skeleton_file}")

    return skeleton_files

def process_all_npy(base_dir, num_folders=6):
    """
    Process all .npy files in the folder and extract corresponding skeletal joint data.
    """
    timestamp_files = []

    # Iterate over folders `1` to `6`
    for i in range(7, num_folders + 7):  
        folder_path = os.path.join(base_dir, str(i))
        timestamp_file = os.path.join(folder_path, str(i), ".npy")
        
        if os.path.exists(timestamp_file):
            timestamp_files.append((i, folder_path, timestamp_file))  # Store (folder index, folder path, file path)
        else:
            print(f"File not found: {timestamp_file}")

    return timestamp_files
    

def process_skeleton_file(folder_path, skeleton_file):
    """
    Process a single `.npy` file, segment actions, and save results in the corresponding folder.
    """
    print(f"Processing: {skeleton_file}")

    # Load skeleton data
    skeleton_frames = np.load(skeleton_file, allow_pickle=True)
    
    # Create action segmenter
    segmenter = ActionSegmenter(frame_count=len(skeleton_frames), min_segment_length=60, long_stationary_threshold=40, smooth_sigma=1.0)

    # Segment actions
    motion_energy, action_segments = segmenter.segment_actions(skeleton_frames)

    # Detect long stationary segments
    is_stationary = motion_energy < segmenter.threshold_T
    long_stationary_segments = segmenter._find_long_stationary_segments(is_stationary)

    # Save results in the current folder `1/`, `2/` ... `6/`
    action_segments_path = os.path.join(folder_path, "action_segments.txt")
    save_action_segments(action_segments, filename=action_segments_path)
    # save_skeleton_segments(skeleton_frames, action_segments, folder_path)

    # Save `motion_energy.png` in the current folder
    energy_plot_path = os.path.join(folder_path, "motion_energy.png")
    plot_motion_energy(motion_energy, action_segments, segmenter.threshold_T, long_stationary_segments, save_path=energy_plot_path)

    print(f"Processing complete: {skeleton_file}, results saved to {folder_path}/\n")

# if __name__ == "__main__":
#     # Get all `.npy` files to process
#     skeleton_files = find_skeleton_files(BASE_DIR)

#     if not skeleton_files:
#         print("No valid `body_skeleton.npy` files found!")
#     else:
#         print(f"Found {len(skeleton_files)} files, starting processing...")
#         # Process each `.npy` file
#         for folder_index, folder_path, skeleton_file in skeleton_files:
#             process_skeleton_file(folder_index, folder_path, skeleton_file)
