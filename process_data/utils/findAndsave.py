import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
import re


def get_aligned_mat_files(base_dir):
    """
    Recursively finds all .mat files in the `aligned` folders under the specified base directory.

    Args:
        base_dir (str): The base directory path, e.g., "dataset/env1/subjects/subject02/origal".

    Returns:
        list: A list of paths to all found .mat files.
    """
    # Convert base_dir to a Path object
    base_dir = Path(base_dir)

    # Find all .mat files in the `aligned` folders
    mat_files = list(base_dir.glob("**/aligned/*.mat"))

    # Convert Path objects to string paths
    mat_files = [str(file) for file in mat_files]

    return mat_files

def move_files_to_aligned(base_path):
    """
    Create action folders under the 'aligned' directory and move files from 
    the 'original' folders directly into these action folders.
    
    Args: 
        base_path: The base directory path, e.g., "dataset/env1/subjects/subject02".
    """
    source_base = os.path.join(base_path, "original")
    target_folder = os.path.join(base_path, "aligned")
    
    # Create the aligned folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    new_segment_number = 1
    
    for folder_num in range(1, 7):
        folder_path = os.path.join(source_base, str(folder_num), "aligned")
        
        if not os.path.exists(folder_path):
            print(f"folder path: {folder_path} doesn't exist, skipping...")
            continue
        
        files = [f for f in os.listdir(folder_path) if f.endswith(('.h5', '.npy'))]
        
        # Group files by segment number
        segment_groups = {}
        for file in files:
            match = re.search(r'segment(\d+)', file)
            if match:
                segment_num = int(match.group(1))
                if segment_num not in segment_groups:
                    segment_groups[segment_num] = []
                segment_groups[segment_num].append((os.path.join(folder_path, file), file))
        
        for segment_num in sorted(segment_groups.keys()):
            # Create action folder for this segment
            action_folder = os.path.join(target_folder, f"action{new_segment_number:02d}")
            if not os.path.exists(action_folder):
                os.makedirs(action_folder)
                
            for source_path, original_name in segment_groups[segment_num]:
                # Create new filename with updated segment number
                file_base = original_name.split('segment')[0]
                ext = os.path.splitext(original_name)[1]
                new_name = f"{file_base}segment{new_segment_number:02d}{ext}"
                
                # Copy file directly to the action folder
                action_file_path = os.path.join(action_folder, new_name)
                shutil.copy2(source_path, action_file_path)
                print(f"copied to action folder: {source_path} -> {action_file_path}")
            
            new_segment_number += 1
    
    print(f"Processed {new_segment_number-1} segments and created {new_segment_number-1} action folders")

def find_files(base_dir, extension):
    """
    Search for files with a specific extension in a given directory.

    Args:
        base_dir (str): The root directory to search.
        extension (str): The file extension to filter (e.g., '.npy', '.mkv').

    Returns:
        list: A list of full file paths.
    """
    found_files = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(extension):
                found_files.append(os.path.join(root, file))
    found_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    return found_files

def find_files_fromsubfolder(base_dir, file_name_template, num_folders=6, start_index=1):
    """
    Find files with a configurable name pattern in a series of indexed folders.

    Args:
        base_dir (str): The root directory containing the indexed subfolders.
        file_name_template (str): The file name pattern, with `{index}` as a placeholder for folder index.
        num_folders (int): The number of indexed folders to search.
        start_index (int): The starting index for the folders.

    Returns:
        list: A list of tuples (folder index, folder path, file path).
    """

    ''' Example usage
    if __name__ == "__main__":
        base_dir = "dataset/env2/subjects/subject26/origal"
        file_name_template = "{index}/body_skeleton.npy"  # Configurable pattern
        files = find_files(base_dir, file_name_template)

        if not files:
            print("No files found!")
        else:
            print(f"Found {len(files)} files:")
            for folder_index, folder_path, file_path in files:
                print(f"{file_path}")
    '''

    found_files = []
    # Iterate over the indexed folders
    for i in range(start_index, num_folders + start_index):  
        folder_path = os.path.join(base_dir, str(i))
        file_name = file_name_template.format(index=i)
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            found_files.append(( folder_path, file_path))  # Store (folder path, file path)
        else:
            print(f"File not found: {file_path}")

    return found_files

def save_action_segments(segments, filename="action_segments.txt"):
    """
    Save the action segments to a text file in the format `start_frame,end_frame`.
    Args:
        segments (list): A list of tuples containing the start and end frame indices of each action segment.
        filename (str): The name of the output text file.
    """
    with open(filename, "w") as f:
        for start, end in segments:
            f.write(f"{start},{end}\n")
    print(f"action splite information has saved in {filename}")

def save_skeleton_segments(skeleton_frames, action_segments, folder_path="dataset"):
    """
    Save the skeleton segments to individual .npy files in the `skeleton_segments` directory.
    Args:
        skeleton_frames (np.ndarray): The full skeleton data, ndarray.
        action_segments (list): A list of tuples containing the start and end frame indices of each action segment.
        folder_path (str): The base directory path to save the skeleton segments.
    """

    skeleton_segments_dir = os.path.join(folder_path, "skeleton_segments")
    os.makedirs(skeleton_segments_dir, exist_ok=True)

    # save each action segment to a separate .npy file like action01.npy, action02.npy...
    for idx, (start, end) in enumerate(action_segments, start=1):
        segment_data = skeleton_frames[start:end + 1]
        segment_filename = os.path.join(skeleton_segments_dir, f"action{idx:02d}.npy")  # action01.npy, action02.npy...
        np.save(segment_filename, segment_data)
        
def save_time_segments(timestamps, action_segments, folder_path="dataset"):
    """
    Save the timestamps of the action segments to individual .npy files in the `skeleton_segments` directory.
    Args:
        timestamps (np.ndarray): The full timestamps data, ndarray.
        action_segments (list): A list of tuples containing the start and end frame indices of each action segment.
        folder_path (str): The base directory path to save the time segments.
    """
    skeleton_segments_dir = os.path.join(folder_path, "skeleton_segments")
    os.makedirs(skeleton_segments_dir, exist_ok=True)

    # save each action segment to a separate .npy file like timestamp01.npy, timestamp02.npy...
    for idx, (start, end) in enumerate(action_segments, start=1):
        segment_data = timestamps[start:end + 1]
        time_filename = os.path.join(skeleton_segments_dir, f"timestamp{idx:02d}.npy")  # action01.npy, action02.npy...
        np.save(time_filename, segment_data)

def plot_motion_energy(motion_energy, action_segments, threshold_T, long_stationary_segments, save_path="plots/motion_energy_plot.png"):
    """
    Plot the motion energy curve with action segments and long stationary segments.
    Args:
        motion_energy (np.ndarray): The motion energy values.
        action_segments (list): A list of tuples containing the start and end frame indices of each action segment.
        threshold_T (float): The threshold value for detecting motion.
        long_stationary_segments (list): A list of tuples containing the start and end frame indices of each long stationary segment.
        save_path (str): The path to save the plot.
    """
    plt.figure(figsize=(12, 6))

    plt.plot(motion_energy, label="Motion Energy", color="blue")

    plt.axhline(y=threshold_T, color="red", linestyle="--", label="Threshold (T)")

    for segment in action_segments:
        start_frame = segment[0]
        end_frame = segment[-1]
        plt.axvline(x=start_frame, color="green", linestyle="--", alpha=0.5)
        plt.axvline(x=end_frame, color="green", linestyle="--", alpha=0.5)
        plt.fill_betweenx(
            y=[0, max(motion_energy)],
            x1=start_frame,
            x2=end_frame,
            color="green",
            alpha=0.1,
            label="Action Segment" if segment == action_segments[0] else "",
        )

    for start, end in long_stationary_segments:
        plt.axvspan(start, end, color="gray", alpha=0.3, label="Long Stationary Segment" if start == long_stationary_segments[0][0] else "")

    plt.xlabel("Frame Index")
    plt.ylabel("Motion Energy")
    plt.title("Motion Energy and Action Segmentation")
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300)
    print(f"motion energy plot is saved in: {save_path}")



