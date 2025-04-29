import h5py
import scipy.io as sio
import numpy as np
import json
from pathlib import Path


def convert_mat_to_hdf5(mat_file_list):
    """
    Converts multiple .mat files with radar_data into individual HDF5 files,
    saving each HDF5 file in the same location as the corresponding .mat file.

    Args:
        mat_file_list (list): List of paths to .mat files.
    """
    for mat_file_path in mat_file_list:
        try:
            # Load the .mat file
            mat_data = sio.loadmat(mat_file_path)
            pc = mat_data["radar_data"]  # Extract radar_data struct
            timestamps = pc["timestamp"].flatten()  # Extract timestamps

            # Get attribute names
            all_attribute_names = np.array(pc[0, 0].dtype.names, dtype="S")  # Convert to byte strings
            frame_attributes = all_attribute_names[:2]  # Frame-level attributes
            point_attributes = all_attribute_names[2:]  # Point-level attributes

            # Generate the output HDF5 file path
            mat_file_path = Path(mat_file_path)
            hdf5_output_path = mat_file_path.with_suffix(".h5")

            # Create the HDF5 file
            with h5py.File(hdf5_output_path, "w") as h5f:
                # Store attribute names as file-level attributes
                h5f.attrs["frame_attributes"] = [attr.decode() for attr in frame_attributes]
                h5f.attrs["point_attributes"] = [attr.decode() for attr in point_attributes]

                # Create a group for frames
                frames_group = h5f.create_group("frames")

                # Iterate through each frame
                for i in range(pc.shape[1]):
                    frame_data = pc[0, i]  # Extract frame i (struct)

                    # Extract frame-level attributes
                    frame_metadata = {attr.decode(): frame_data[attr.decode()].item() for attr in frame_attributes}

                    # Determine the number of points in this frame
                    num_points = len(frame_data[point_attributes[0].decode()].flatten())

                    # Create an array for point-level attributes
                    frame_array = np.zeros((num_points, len(point_attributes)), dtype=np.float32)
                    for j, attr in enumerate(point_attributes):
                        frame_array[:, j] = frame_data[attr.decode()].flatten()

                    # Store the frame data
                    frame_ds = frames_group.create_dataset(f"frame_{i:03d}", data=frame_array)

                    # Add metadata attributes
                    timestamp_str = str(timestamps[i])
                    frame_ds.attrs["timestamp"] = timestamp_str  # Store timestamp
                    frame_ds.attrs["frame_metadata"] = json.dumps(frame_metadata)  # Store frame-level metadata

            print(f"Converted {mat_file_path} to {hdf5_output_path}")

        except Exception as e:
            print(f"Error processing file {mat_file_path}: {e}")
            continue

    print("Conversion complete.")

