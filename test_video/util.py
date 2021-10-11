import cv2
import numpy as np
from configLoader import ConfigLoader
import DataProcessing as dp
from PIL import Image

def raw_image_sensor(image_resize, img_real, offset=[0, 0]):
	config = ConfigLoader()
	image_resize[185:661, 0:848] = img_real[1:477, :]
	# image_resize[372:848, 0:848] = img_real[1:477, :]
	image_raw = cv2.resize(image_resize, (320, 320))
	image, mask = dp.background_subtraction(image_raw)
	# image_black_background = image.copy()
	# image_black_background = cv2.resize(image_black_background, (512, 512))
	image[np.where(mask == 0)] = config.get_green_background
	# target_pos, target_angle = find_convexhull(image, offset, image_black_background)
	# return image, target_pos, target_angle
	return image, image_raw


def raw_image_depth(image, img_real):
	# image[184:664, 0:848] = img_real
	# image = cv2.resize(image, (480, 480))
	image = img_real[0:480, 184:664]
	image = cv2.resize(image, (320, 320))
	return image


def ar_blending(raw_img ,image, img_render):
	config = ConfigLoader()
	object_color = config.get_object_mask

	mask = render_object_background_removal(img_render)
	image_with_object = raw_img.copy()
	image_with_object_mask = image.copy()
	image_with_object_mask[np.where(mask == 0)] = object_color
	image_with_object[np.where(mask == 0)] = img_render[np.where(mask == 0)]
	mask = 255 - mask

	return image_with_object, image_with_object_mask, mask


def render_object_background_removal(img_render):
	mask = cv2.inRange(img_render, (254, 0, 0), (256, 0, 0))
	# mask[np.where((img_render == (0, 0, 0)).all(axis=2))] = 255
	return mask


def opencv_to_pil(opencv_img):
	return Image.fromarray(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))


def pil_to_opencv(pil_img, channel=3):
	opencv_image = np.array(pil_img)
	if channel == 3:
		# Convert RGB to BGR
		opencv_image = opencv_image[:, :, ::-1].copy()
	
	return opencv_image


def mesh_normalization(verts):
	min_x = np.min(verts[:, 0])
	min_y = np.min(verts[:, 1])
	min_z = np.min(verts[:, 2])
	
	max_x = np.max(verts[:, 0])
	max_y = np.max(verts[:, 1])
	max_z = np.max(verts[:, 2])
	
	max_axis = np.max([max_x - min_x, max_y - min_y, max_z - min_z])
	
	def normalize(p):
		return ((p[0] - (max_x + min_x) / 2) / max_axis * 2,
		        (p[1] - (max_y + min_y) / 2) / max_axis * 2,
		        (p[2] - (max_z + min_z) / 2) / max_axis * 2)
	
	vfunc = np.vectorize(normalize)
	verts = np.apply_along_axis(normalize, 1, verts)
	
	min_x = np.min(verts[:, 0])
	min_y = np.min(verts[:, 1])
	min_z = np.min(verts[:, 2])
	
	max_x = np.max(verts[:, 0])
	max_y = np.max(verts[:, 1])
	max_z = np.max(verts[:, 2])
	print(str(min_x) + ' ' + str(max_x) + '  ' + str(min_y) + ' ' + str(max_y) + '   ' + str(min_z) + ' ' + str(max_z))
	return verts


def pixel_to_pos_reprojection(canvas, depth, pixel_pos):
	translation = 1 / canvas.projection[0][0]
	return [translation * -depth * pixel_pos[0], translation * -depth * pixel_pos[1], depth]


def find_convexhull(frame, offset, frame_black_background):
	config = ConfigLoader()
	object_color = config.get_object_mask
	
	# frame, _ = dp.background_subtraction_syn(frame)
	
	gray = cv2.cvtColor(frame_black_background, cv2.COLOR_BGR2GRAY)  # convert to grayscale
	blur = cv2.medianBlur(gray, 9)  # blur the image
	ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
	
	############################################
	# find plam position with distance transform
	dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
	cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
	index = np.unravel_index(np.argmax(dist), np.shape(dist))[::-1]
	palm = np.add([(index[0] - 256) / 256.0, (index[1] - 256) / -256.0], offset)
	
	############################################
	# find two wrist points with convex defects
	up = (0, 0)
	down = (0, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	hull = []
	for i in range(len(contours)):
		hull.append(cv2.convexHull(contours[i], False))
	if len(hull) >= 0 and len(contours) >= 1:
		hand_hull = hull[0]
		hull_ = cv2.convexHull(contours[0], returnPoints=False)
		# defects = cv2.convexityDefects(contours[0], hull_)
		# if not isinstance(defects, type(None)):
		# 	for i in range(defects.shape[0]):
		# 		s, e, f, d = defects[i, 0]
		# 		up = tuple(contours[0][f][0])
		# 		if up[0] < 450:
		# 			break
		# 	for i in reversed(range(defects.shape[0])):
		# 		s, e, f, d = defects[i, 0]
		# 		down = tuple(contours[0][f][0])
		# 		if down[0] < 450:
		# 			break
		# 	cv2.circle(frame, up, 5, (0, 0, 255), -1)
		# 	cv2.circle(frame, down, 5, (0, 0, 255), -1)
		leftmost_index = np.argmin(hand_hull[:, 0, 0])
		# frame = cv2.circle(frame, (hull[0][leftmost_index][0][0], hull[0][leftmost_index][0][1]), 20, object_color, -1)
		leftmost = np.add([(hull[0][leftmost_index][0][0] - 256) / 256.0, (hull[0][leftmost_index][0][1] - 256) / -256.0],
		                  offset)
		dist = np.linalg.norm(leftmost - palm)
		# print(np.arcsin((palm[1] - leftmost[1]) / dist) / np.pi * 180)
		return palm, np.arcsin((palm[1] - leftmost[1]) / dist) / np.pi * 180
	else:
		return [0, 0], 0


def depth_map_to_RGB_image(depth_map):
	h, w = np.shape(depth_map)
	RGB_image = np.zeros((h, w, 3))
	RGB_image = RGB_image.astype(np.uint8)
	depth1 = depth_map // 255
	depth2 = depth_map % 255
	RGB_image[:, :, 0] = depth1
	RGB_image[:, :, 1] = depth2
	return RGB_image

def RGB_map_to_depth_map(RGB_image):
	h, w, _ = np.shape(RGB_image)
	depth_map = np.zeros((h, w))
	RGB_image = RGB_image.astype(np.int)
	depth_map[:, :] = RGB_image[:, :, 1] + RGB_image[:, :, 0] * 255
	return depth_map


def depth_buffer_to_absolute_depth(depth_buffer, near=1, far=100):
	depth = np.divide(depth_buffer, 255.0)
	z_ndc = np.subtract(np.multiply(depth, 2), 1)
	z_eye = np.divide(2*near*far, np.subtract(near+far, np.multiply(z_ndc, far-near)))
	return z_eye


import math
def rotationMatrixToEulerAngles(R):
	sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
	
	singular = sy < 1e-6
	
	if not singular:
		x = math.atan2(R[2, 1], R[2, 2])
		y = math.atan2(-R[2, 0], sy)
		z = math.atan2(R[1, 0], R[0, 0])
	else:
		x = math.atan2(-R[1, 2], R[1, 1])
		y = math.atan2(-R[2, 0], sy)
		z = 0
	
	return np.array([x, y, z])
