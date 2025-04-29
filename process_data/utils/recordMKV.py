import cv2
import time
import numpy as np
import pykinect_azure as pykinect

if __name__ == "__main__":
	# Initialize the library, if the library is not found, adqd the library path as argument
	pykinect.initialize_libraries()

	# Modify camera configuration
	device_config = pykinect.default_configuration
	device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
	device_config.synchronized_images_only = True
	
	#print(device_config)
	all_timestamp = []
	# Start device
	timestamp = int(time.time())  
	video_filename = f"{timestamp}.mkv"  

	device = pykinect.start_device(config=device_config, record=True, record_filepath=video_filename)

	cv2.namedWindow('color Image',cv2.WINDOW_NORMAL)
	    # Record the start time
	start_time = time.time()
	while True:

		# Get capture
		capture = device.update()
		all_timestamp.append(time.time())
		# Get color image
		ret, color_image = capture.get_color_image()
		if not ret:
			continue
		# Plot the image
		cv2.imshow('color Image',color_image)
		
		# Press q key to stop
		if cv2.waitKey(1) == ord('q'): 
			break

	filename = f"{timestamp}.npy"  
	np.save(filename, np.array(all_timestamp))