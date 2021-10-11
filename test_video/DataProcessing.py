import os
import cv2
import re
import numpy as np
import random
import math

# green background subtraction
def green_screen_subtraction(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_green = np.array([25, 0, 0])
	upper_green = np.array([100, 255, 255])
	mask = cv2.inRange(hsv, lower_green, upper_green)
	mask = 255 - mask
	res = cv2.bitwise_and(image, image, mask=mask)
	brightness_range = 10
	contrast_range = 0.2
	augmented_frame = (1 - random.uniform(-contrast_range,
	                                      contrast_range)) * res
	augmented_frame = augmented_frame + random.randint(-brightness_range, brightness_range)
	res = np.uint8(np.clip(augmented_frame, 0, 255))
	res[mask == 0] = (119, 178, 89)
	return res, mask


def blue_object_detection(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_blue = np.array([90, 0, 0])
	upper_blue = np.array([150, 255, 255])
	mask_1 = cv2.inRange(hsv, lower_blue, upper_blue)
	mask = 255 - mask_1
	res = cv2.bitwise_and(image, image, mask=mask)
	return res, mask


def blue_object_detection_target(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_blue = np.array([100, 50, 50])
	upper_blue = np.array([150, 255, 255])
	mask_1 = cv2.inRange(hsv, lower_blue, upper_blue)
	mask = 255 - mask_1
	res = cv2.bitwise_and(image, image, mask=mask)
	return res, mask

def red_button_detection(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_red = np.array([0, 200, 0])
	upper_red = np.array([20, 255, 255])
	mask = cv2.inRange(hsv, lower_red, upper_red)
	res = cv2.bitwise_and(image, image, mask=mask)
	return res, mask

def yellow_button_detection(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_red = np.array([30, 255, 54])
	upper_red = np.array([30, 255, 255])
	mask = cv2.inRange(hsv, lower_red, upper_red)
	res = cv2.bitwise_and(image, image, mask=mask)
	return res, mask

def blue_saber_detection(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_red = np.array([120, 255, 54])
	upper_red = np.array([120, 255, 255])
	mask = cv2.inRange(hsv, lower_red, upper_red)
	res = cv2.bitwise_and(image, image, mask=mask)
	return res, mask


def background_subtraction(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_green = np.array([30, 25, 25])
	upper_green = np.array([120, 255, 255])
	lower_blue = np.array([100, 50, 50])
	upper_blue = np.array([150, 255, 255])
	mask = cv2.inRange(hsv, lower_green, upper_green)
	mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
	mask[np.where(mask2==255)] = 255
	mask = 255 - mask
	res = cv2.bitwise_and(image, image, mask=mask)
	return res, mask

# def background_subtraction(image):
# 	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# 	lower_green = np.array([140, 20, 0])
# 	upper_green = np.array([180, 200, 255])
# 	mask_1 = cv2.inRange(hsv, lower_green, upper_green)
# 	lower_green = np.array([0, 20, 0])
# 	upper_green = np.array([20, 200, 255])
# 	mask_2 = cv2.inRange(hsv, lower_green, upper_green)
# 	mask_1[np.where(mask_2 == 255)] = 255
# 	image[np.where(np.logical_and(mask_1 != 255, mask_2 != 255))] = (119, 178, 89)
# 	return image, mask_1


def background_subtraction_syn(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_green = np.array([40, 0, 0])
	upper_green = np.array([100, 255, 255])
	mask = cv2.inRange(hsv, lower_green, upper_green)
	mask = 255 - mask
	res = cv2.bitwise_and(image, image, mask=mask)
	return res, mask


def real_background_subtraction(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_green = np.array([30, 10, 10])
	upper_green = np.array([100, 255, 255])
	mask = cv2.inRange(hsv, lower_green, upper_green)
	mask = 255 - mask
	res = cv2.bitwise_and(image, image, mask=mask)
	return res, mask

def create_bimask(img_origin):
	res2, object_mask = blue_object_detection(img_origin)
	object_mask = 255 - object_mask
	return object_mask

def create_trimask(img_origin):
	res1, background_mask = background_subtraction_syn(img_origin)
	res2, object_mask = blue_object_detection(img_origin)
	object_mask[np.where(background_mask == 0)] = 1
	object_mask[np.where(object_mask == 0)] = 128
	object_mask[np.where(object_mask == 1)] = 0
	return object_mask


def create_trimask_target(img_origin):
	res1, background_mask = real_background_subtraction(img_origin)
	res2, object_mask = blue_object_detection_target(img_origin)
	object_mask[np.where(background_mask == 0)] = 1
	object_mask[np.where(object_mask == 0)] = 128
	object_mask[np.where(object_mask == 1)] = 0
	return object_mask


# overlapping object to hand
def image_overlap(hand, target_object):
	_, object_mask = background_subtraction(target_object)
	hand_bg = cv2.bitwise_and(hand, hand, mask=255 - object_mask)
	hand_fg = cv2.bitwise_and(target_object, target_object, mask=object_mask)
	result = cv2.add(hand_bg, hand_fg)
	return result


#
def image_overlap_with_prediction(hand, target_object, prediction_mask):
	hand_bg = cv2.bitwise_and(hand, hand, mask=prediction_mask)
	hand_fg = cv2.bitwise_and(target_object, target_object, mask=255 - prediction_mask)
	result = cv2.add(hand_bg, hand_fg)
	return result


def image_overlap_with_prediction_confidence(hand, target_object, prediction_mask, confidence):
	# hand_bg = cv2.bitwise_and(hand, hand, mask=prediction_mask)
	# hand_fg = cv2.bitwise_and(target_object, target_object, mask=255 - prediction_mask)
	# result = cv2.add(hand_bg, hand_fg)
	confidence = np.tile(np.expand_dims(confidence, axis=-1), (1, 1, 3))
	confidence = 1 + confidence
	target_object_region, mask = blue_object_detection_target(target_object)
	hand_trans = np.where(np.logical_and(prediction_mask == 255, mask == 0))
	object_trans = np.where(np.logical_and(prediction_mask == 0, mask == 0))
	# confidence = cv2.medianBlur(confidence, 3)
	# confidence = confidence / 255
	result = target_object.copy()
	# result[hand_trans] = hand[hand_trans]
	result[hand_trans] = hand[hand_trans] * confidence[hand_trans] + target_object[hand_trans] * (1 - confidence[hand_trans])
	result[object_trans] = hand[object_trans] * (1 - confidence[object_trans]) + target_object[object_trans] * confidence[object_trans]
	return result


def create_input_image(img_hand, img_obj):
	res, mask = background_subtraction(img_obj)
	result = image_overlap(img_hand, img_obj, mask)
	return result


def create_input():
	path = "Data/"
	files = os.listdir(path)
	for file in files:
		if file.find('origin') != -1:
			num = re.findall("\d+", file)[0]
			img_hand_filename = str(num) + '_hand.png'
			img_object_filename = str(num) + '_object.png'
			img_input_filename = "Input/" + str(num) + '_input.png'
			
			img_hand = cv2.imread(path + img_hand_filename)
			img_object = cv2.imread(path + img_object_filename)
			img_input = cv2.imwrite(img_input_filename, create_input_image(img_hand, img_object))


def detect_object_bounding_box(img_object):
	res, mask = green_screen_subtraction(img_object)
	idx = cv2.findNonZero(mask)
	rec = cv2.boundingRect(idx)
	cv2.rectangle(res, (rec[0], rec[1]), (rec[2] + rec[0], rec[3] + rec[1]), (0, 255, 0), 2)
	return res, rec


def sigmoid(x):
	return 1 / (1 + math.exp(x))


def check_mkdir(dir_name):
	if not os.path.exists(dir_name):
		os.mkdir(dir_name)
