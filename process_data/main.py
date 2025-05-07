
from class_files.data_aligner import DataAligner
from utils.processAllMKV import process_all_mkv
from utils.saveMotionSplit import process_skeleton_file
from utils.findAndsave import find_files, find_files_fromsubfolder, get_aligned_mat_files, move_files_to_aligned
from utils.reformatRadardata import convert_mat_to_hdf5


if __name__ == "__main__":

    for participant_num in range(1, 27):
        currentParticipant = f"{participant_num:02d}"
        for currentEnv in ["env1", "env2"]:
            mkv_folder = "YOUR MKV SAVED PATH/".format(currentParticipant=currentParticipant)
            mkv_flies = find_files(mkv_folder, ".mkv")
            time_files = find_files(mkv_folder, ".npy") 
            if not mkv_flies:
                print("No valid `.mkv` files found!")
            else:
                process_all_mkv(mkv_flies, time_files,subjectNum=f"subject{currentParticipant}")

            skeletonpointfolder = f"dataset/{currentEnv}/subjects/subject{currentParticipant}/original"
            # # Get all `.npy` files to process
            skeleton_files = find_files_fromsubfolder(skeletonpointfolder, "body_skeleton.npy")
            if not skeleton_files:
                print("No valid `body_skeleton.npy` files found!")
            else:
                print(f"Found {len(skeleton_files)} files, starting processing...")
                # Process each `.npy` file
                for folder_path, skeleton_file in skeleton_files:
                    process_skeleton_file(folder_path, skeleton_file)

    # before aligning data, make sure action_segments.txt has correct timestamps split
    # and the radar data is in the correct format, has x,y,z,snr..., like our orginal .mat files


    # Data alignment and segmentation
    for participant_num in range(1, 27):
      
        currentParticipant = f"{participant_num:02d}"
        for currentEnv in ["env1", "env2"]:
            skeletonpointfolder = f"dataset/{currentEnv}/subjects/subject{currentParticipant}/original"
            # Get all `.npy` files to process
            skeleton_files = find_files_fromsubfolder(skeletonpointfolder, "body_skeleton.npy")
            radar_file_path = find_files(f"YOUR RADAR DATA PATH", ".mat") #find all .mat files for same participant
            if not radar_file_path or not skeleton_files:
                print("No valid `.mat` or `action_segments.txt` files found!")
            else:
                for val1, val2 in zip(skeleton_files, radar_file_path):
                    aligner = DataAligner(val1[0], val2, val1[0])
                    aligner.align_and_segment_data()
                print(f"Data alignment and segmentation complete for {len(skeleton_files)} files.")
            print("data align done!")
            radar_files = get_aligned_mat_files(f"dataset/{currentEnv}/subjects/subject{currentParticipant}/original")
            convert_mat_to_hdf5(radar_files)
            base_path = f"dataset/{currentEnv}/subjects/subject{currentParticipant}"
            move_files_to_aligned(base_path)  
        print(f"\n============== process participant: {currentParticipant} ==============")



        
       


   
    