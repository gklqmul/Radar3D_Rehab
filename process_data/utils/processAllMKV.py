import os
import shutil
from process.class_files.kinect_skeleton_processor import KinectSkeletonProcessor

def process_all_mkv(mkv_files, time_files, subjectNum="subject26"):
    """
    Process all MKV files in the given list.
    Args:
        mkv_files (list): List of MKV files to process
        time_files (list): List of corresponding time files
        subjectNum (str): Subject number
    """
   
    for index, mkv_file in enumerate(mkv_files, start=1):
        if index > 6:
            save_dir = os.path.join("dataset", "env2", "subjects", subjectNum, "origal", str(index-6))
        else:
            save_dir = os.path.join("dataset", "env1", "subjects", subjectNum, "origal", str(index))
        
        os.makedirs(save_dir, exist_ok=True)  # Create the save directory if it doesn't exist

        print(f"Processing: {mkv_file} Save directory: {save_dir}")
        processor = KinectSkeletonProcessor(mkv_file, save_dir)  # Pass save_dir to the class
        processor.process_video()

        if time_files and index-1 < len(time_files):
            time_file = time_files[index-1]  # Get the corresponding time file (0-based index)
            timestamp_save_path = os.path.join(save_dir, "timestamps.npy")
            shutil.copy2(time_file, timestamp_save_path)
            print(f"Saved timestamp file: {time_file} as {timestamp_save_path}")
            
        print(f"Processing complete: {mkv_file}\n")


