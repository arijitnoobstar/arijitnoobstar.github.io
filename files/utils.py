"""
This file automatically generates the depth and instance segmented frames
Edit the settings below to your needs. Instance IDs are found in scene_data.json per scene
"""

import cv2
import numpy as np
import os

#------------ SETTINGS -------------#
generate_depth_frames = False
generate_depth_videos = True
generate_instance_frames = False
generate_instance_videos = True
# PATH TO DATASET (where the 5 directories of AVoE are present)
root_path = 'PATH TO DATA FOLDER'
#-----------------------------------#

event_categories = ['A_support', 'B_occlusion', 'C_container', 'D_collision', 'E_barrier']
trial_num_indices = [range(1,376), range(3751,3826), range(3751,3826), range(4501,4551), range(4501,4551)]
segments = ['train/expected/', 'validation/expected/', 'validation/surprising/', 'test/expected/', 'test/surprising/']
depth_paths = []
instance_paths = []
trial_paths = []

# save paths of all videos
for ec in event_categories:
	ec_path = root_path + ec + '/'
	for i, segment in enumerate(segments):
		segment_path = ec_path + segment
		for num in trial_num_indices[i]:
			trial_paths.append(segment_path + 'trial_{}/'.format(num))
			if generate_depth_frames or generate_depth_videos:
				depth_paths.append(segment_path + 'trial_{}/depth_raw.npz'.format(num))
			if generate_instance_frames or generate_instance_videos:
				instance_paths.append(segment_path + 'trial_{}/inst_raw.npz'.format(num))

def frames_to_media(data, type, trial_path, generate_frame = False, generate_video = False):
	"""
	converts numpy data (num_frames, height, width) into avi video
	"""
	# gather all frames

	height, width = data[0].shape
	size = (width, height)
	
	if generate_video:
		video = cv2.VideoWriter('{}/{}.avi'.format(trial_path, type) ,cv2.VideoWriter_fourcc(*'DIVX'), 24, size)
	
	for idx in range(len(data)):
		cv2.imwrite(trial_path + "{}_{}.jpg".format(type, idx+1), np.uint16(data[idx]))
		if generate_video:
			video.write(cv2.imread(trial_path + "{}_{}.jpg".format(type, idx+1)))
		if not generate_frame:
			os.remove(trial_path + "{}_{}.jpg".format(type, idx+1))
	if generate_video:
		video.release()

# generate data for depth raw
if generate_depth_frames or generate_depth_videos:
	num_depth = len(depth_paths)
	for i, x in enumerate(depth_paths):
		data = np.load(x)['arr_0']
		data = np.transpose(data,(2,0,1)) # transpose axes into (num_frames, height, width)
		frames_to_media(data, 'depth_raw', trial_paths[i], generate_depth_frames, generate_depth_videos)
		if i < num_depth - 1:
			print("Completed (Depth): {}/{}".format(i+1, num_depth), end = "\r")
		else:
			print("Completed (Depth): {}/{}".format(i+1, num_depth))

# generate data for instance raw
if generate_instance_frames or generate_instance_videos:
	num_instance = len(instance_paths)
	for i, x in enumerate(instance_paths):
		data = np.load(x)['arr_0']
		data = np.transpose(data,(2,0,1)) # transpose axes into (num_frames, height, width)
		frames_to_media(data, 'inst_raw', trial_paths[i], generate_instance_frames, generate_instance_videos)
		if i < num_instance - 1:
			print("Completed (Instance): {}/{}".format(i+1, num_instance), end = "\r")
		else:
			print("Completed (Instance): {}/{}".format(i+1, num_instance))