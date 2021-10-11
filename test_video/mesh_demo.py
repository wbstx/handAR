import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

import PIL
from PIL import Image

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from utils.preprocessing import process_bbox, generate_patch_image
from utils.transforms import pixel2cam, cam2pixel
from utils.mano import MANO

# sys.path.insert(0, cfg.smpl_path)
sys.path.insert(0, cfg.mano_path)
# from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from utils.manopth.manopth.manolayer import ManoLayer
from utils.vis import vis_mesh, save_obj, vis_keypoints_with_skeleton
from canvas import Canvas
import vispy
from vispy import app, io, gloo, scene, visuals
from vispy.util.transforms import perspective, translate, rotate, ortho, scale
import matplotlib.pyplot as plt
import math

cfg.set_args('0', 'lixel')
cudnn.benchmark = True

origin = False

joint_num = 21

# MANO mesh
vertex_num = 778
mano_layer = ManoLayer(ncomps=45, mano_root=cfg.mano_path + '/mano/models')
face = mano_layer.th_faces.numpy()
joint_regressor = mano_layer.th_J_regressor.numpy()
root_joint_idx = 0

model_path = '../weights/snapshot_%d.pth.tar' % 24

assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))

from model_no_render import get_model
model = get_model(vertex_num, joint_num, 'test')

model = DataParallel(model).cuda()
ckpt = torch.load(model_path)


model.module.pose_backbone.load_state_dict(ckpt['pose_backbone'])
model.module.pose_net.load_state_dict(ckpt['posenet'])
model.module.pose2feat.load_state_dict(ckpt['pose2feat'])
model.module.mesh_backbone.load_state_dict(ckpt['mesh_backbone'])
model.module.mesh_net.load_state_dict(ckpt['mesh_net'])
model.module.gcn.load_state_dict(ckpt['gcn'])
model.module.global_img_feat.load_state_dict(ckpt['global_img_feat'])
model.module.segmentation_net.load_state_dict(ckpt['segmentation_net'])

model.eval()

# Set the cuda device
if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
else:
	device = torch.device("cpu")

transform = transforms.ToTensor()


def pil2opencv(img):
	open_cv_image = np.array(img)
	open_cv_image = open_cv_image[:, :, ::-1].copy()
	return open_cv_image


def opencv2pil(img):
	pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	return pil_img


def depth_buffer_to_absolute_depth(depth_buffer, near=1, far=100):
	depth = np.divide(depth_buffer, 255.0)
	z_ndc = np.subtract(np.multiply(depth, 2), 1)
	z_eye = np.divide(2 * near * far, np.subtract(near + far, np.multiply(z_ndc, far - near)))
	return z_eye


brightness = contrast = 1.0

start_recording = False
video = []
mesh = []

c = Canvas(mesh_name='hand', has_texture=False)
faces = c.mesh.get_faces()

old2new_matching = np.load('matching.npy').astype(np.int)
pos_window_tvec = []
pos_window_rvec = []

def tvec_smoothing(tvec):
	alpha = 0.7
	if np.isnan(tvec).any():
		return pos_window_tvec[-1]
	if len(pos_window_tvec) < 4:
		pos_window_tvec.append(np.array(tvec))
		return np.array(tvec)
	else:
		curr_tvec = np.array([0, 0, 0])
		para = 0
		for i in range(0, 4):
			curr_tvec = np.add(curr_tvec, np.multiply(pos_window_tvec[3 - i], math.pow((1 - alpha), i + 1)))
			para += math.pow((1 - alpha), i + 1)
		curr_tvec = np.add(np.multiply(tvec, alpha), np.multiply(curr_tvec, alpha))
		curr_tvec /= (para * alpha + alpha)
		pos_window_tvec.pop(0)
		pos_window_tvec.append(curr_tvec)
		return curr_tvec

def rvec_smoothing(rvec):
	alpha = 0.7
	if len(pos_window_rvec) != 0:
		pass
	if len(pos_window_rvec) < 8:
		pos_window_rvec.append(np.array(rvec))
		return np.array(rvec)
	else:
		curr_rvec = [0, 0, 0]
		para = 0
		for i in range(0, 8):
			curr_rvec = np.add(curr_rvec, np.multiply(pos_window_rvec[7 - i], math.pow((1 - alpha), i + 1)))
			para += math.pow((1 - alpha), i + 1)
		curr_rvec = np.add(np.multiply(rvec, alpha), np.multiply(curr_rvec, alpha))
		curr_rvec /= (para * alpha + alpha)
		# curr_rvec = np.divide(curr_rvec, alpha + alpha * para)
		pos_window_rvec.pop(0)
		pos_window_rvec.append(curr_rvec)
		return curr_rvec

front_triangle_index = 761
back_triangle_index = 755
middle_finder_major = [395, 364]

hand_layer = MANO()

links = ( (0,1), (0,5), (0,9), (0,13), (0,17), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16), (17,18), (18,19), (19,20) )

import scipy.io as scio
import numpy as np
import cv2


V = c.mesh.V.copy()

hand_mesh = np.zeros((778, 3))
for i in range(778):
	hand_mesh[i] = V[:]['a_position'][old2new_matching[i], :3]
origin_joint = np.dot(hand_layer.joint_regressor, hand_mesh)

# Palm
origin_x = V[:]['a_position'][144, :3] - V[:]['a_position'][145, :3]
origin_x = origin_x / np.linalg.norm(origin_x)
origin_y = V[:]['a_position'][144, :3] - V[:]['a_position'][146, :3]
origin_y = origin_y / np.linalg.norm(origin_y)
origin_z = np.cross(origin_x, origin_y)
origin_z = origin_z / np.linalg.norm(origin_z)
origin_y = np.cross(origin_x, origin_z)
origin_y = origin_y / np.linalg.norm(origin_y)

M_o = np.array([[origin_x[0], origin_x[1], origin_x[2]],
                [origin_y[0], origin_y[1], origin_y[2]],
                [origin_z[0], origin_z[1], origin_z[2]]])

for kkk in range(0, 100):
	img_path = '../imgs/' + str(kkk).zfill(4) + '.png'

	original_img = cv2.imread('imgs/' + img_path)
	original_img = original_img[:320, :320]


	original_img = cv2.resize(original_img, (320, 320))
	frame = original_img
	original_img_height, original_img_width = original_img.shape[:2]

	h, w = frame.shape[0], frame.shape[1]
	if h < w:
		frame = frame[:, int((w - h) / 2):int((w + h) / 2)]
	else:
		frame = frame[int((h - w) / 2):int((h + w) / 2), :]
	frame = cv2.resize(frame, (320, 320))

	pil_hand_frame = opencv2pil(frame)

	pil_hand_frame = PIL.ImageEnhance.Brightness(pil_hand_frame).enhance(brightness)
	pil_hand_frame = PIL.ImageEnhance.Color(pil_hand_frame).enhance(contrast)

	frame = pil2opencv(pil_hand_frame)

	original_img = frame
	original_img_height, original_img_width = original_img.shape[:2]

	bbox = [0, 0, 320, 320]  # xmin, ymin, width, height
	bbox = process_bbox(bbox, original_img_width, original_img_height)

	img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
	bbox_img = img.copy()
	img = transform(img.astype(np.float32)) / 255
	img = img.cuda()[None, :, :, :]
	# forward
	inputs = {'img': img}
	targets = {}
	meta_info = {'bb2img_trans': bb2img_trans}

	with torch.no_grad():
		out = model(inputs, targets, meta_info, 'test')



	img = img[0].cpu().numpy().transpose(1, 2, 0)  # cfg.input_img_shape[1], cfg.input_img_shape[0], 3
	if origin:
		mesh_lixel_img = out['mesh_coord_img'][0].cpu().numpy()
	else:
		# mesh_lixel_img = out['mesh_coord_img'][0].cpu().numpy()
		mesh_lixel_img = out['gcn'][0].cpu().numpy()
	test = mesh_lixel_img.copy()
	#
	if not origin:
		joint = out['pose'][0].cpu().numpy()
	else:
		joint = out['joint_coord_img'][0].cpu().numpy()

	pred_joint = joint.copy()
	# print(joint)
	# np.save(str(kkk) + '.npy', joint[:, :2])

	# restore mesh_lixel_img to original image space and continuous depth space
	mesh_lixel_img[:, 0] = mesh_lixel_img[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
	mesh_lixel_img[:, 1] = mesh_lixel_img[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
	mesh_lixel_img[:, :2] = np.dot(bb2img_trans,
	                               np.concatenate((mesh_lixel_img[:, :2], np.ones_like(mesh_lixel_img[:, :1])),
	                                              1).transpose(1, 0)).transpose(1, 0)

	mesh_lixel_img[:, 2] = (mesh_lixel_img[:, 2] / cfg.output_hm_shape[0] * 2. - 1) * (cfg.bbox_3d_size / 2)

	joint[:, 0] = joint[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
	joint[:, 1] = joint[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
	joint[:, :2] = np.dot(bb2img_trans,
	                               np.concatenate((joint[:, :2], np.ones_like(joint[:, :1])),
	                                              1).transpose(1, 0)).transpose(1, 0)

	# visualize lixel mesh in 2D space
	vis_img = original_img.copy().astype(np.uint8)
	# res[np.where((res==(255, 255, 255)).all(axis=2))] = vis_img[np.where((res==(255, 255, 255)).all(axis=2))]
	# vis_img = cv2.addWeighted(vis_img, 0.5, res, 0.5, 0)
	vis_img = vis_mesh(vis_img, mesh_lixel_img)

	joint_img = original_img.copy().astype(np.uint8)
	joint_img = vis_keypoints_with_skeleton(joint_img, joint, links)

	c.view = c.default_view
	c.program['u_view'] = c.view
	c.program['u_mat_rendering'] = 0.0

	homo_coord = np.append(mesh_lixel_img, np.ones((mesh_lixel_img.shape[0], 1)), axis=1)
	old2new_coord = np.zeros((778, 4))
	for i in range(778):
		old2new_coord[old2new_matching[i]] = homo_coord[i]

	# mapped_coord = np.zeros((c.mapping.shape[0], 4))
	# mapped_coord = np.zeros((958, 4))
	mapped_coord = np.zeros((c.mapping.shape[0], 4))
	# mapped_coord = np.zeros((778, 4))

	for i in range(mapped_coord.shape[0]):
		mapped_coord[i] = old2new_coord[int(c.mapping[i]) - 1]

	mapped_coord[:, :2] = mapped_coord[:, :2] / 320 * 2 - 1
	mapped_coord[:, 1] *= -1

	# mapped_coord[:, :2] *= 1.04
	mapped_coord[:, 2] *= 2.5  # Thickness Hacking

	V = c.mesh.V.copy()

	######################
	# Old Hand Mesh
	######################

	scale = 0.4 # paddle
	V[:]['a_position'][mapped_coord.shape[0]:, :3] -= V[:]['a_position'][145, :3]


	# Scale
	V[:]['a_position'][mapped_coord.shape[0]:, :3] *= scale

	######################
	# New Hand Mesh
	######################
	view_pos = np.dot(np.linalg.inv(c.projection.T), mapped_coord.T)
	model_pos = np.dot(np.linalg.inv(c.view.T), view_pos)
	world_pos = np.dot(np.linalg.inv(c.model.T), model_pos)
	world_pos = world_pos / world_pos[3, :]

	V[:]['a_position'][:mapped_coord.shape[0], :] = world_pos.transpose(1, 0)

	hand_mesh = np.zeros((778, 3))
	for i in range(778):
		hand_mesh[i] = V[:]['a_position'][old2new_matching[i], :3]
	joint = np.dot(hand_layer.joint_regressor, hand_mesh)

	######################
	# Object Transformation
	######################

	object_verts = V[:]['a_position'][mapped_coord.shape[0]:, :3]
	object_center = np.average(object_verts, axis=0)

	# Rotation
	new_x = V[:]['a_position'][144, :3] - V[:]['a_position'][145, :3]
	new_x = new_x / np.linalg.norm(new_x)
	new_y = V[:]['a_position'][144, :3] - V[:]['a_position'][146, :3]
	new_y = new_y / np.linalg.norm(new_y)
	new_z = np.cross(new_x, new_y)
	new_z = new_z / np.linalg.norm(new_z)
	new_y = np.cross(new_x, new_z)
	new_y = new_y / np.linalg.norm(new_y)

	M_n = np.array([[new_x[0], new_x[1], new_x[2]],
	                [new_y[0], new_y[1], new_y[2]],
	                [new_z[0], new_z[1], new_z[2]]])

	M_n = rvec_smoothing(M_n)

	adjust = np.zeros((3, 3))

	V[:]['a_position'][mapped_coord.shape[0]:, :3] = np.dot(M_n.T, np.dot(np.linalg.inv(M_o.T),
	                                                                      V[:]['a_position'][mapped_coord.shape[0]:,
	                                                                      :3].T)).T

	pos = tvec_smoothing(V[:]['a_position'][145, :3])
	V[:]['a_position'][mapped_coord.shape[0]:, :3] += pos

	c.vertices_buff.set_data(V)
	light_mat = np.zeros((4, 4)).astype(np.float)
	light_mat[:3, :3] = np.dot(M_n.T, np.linalg.inv(M_o.T))
	light_mat[-1, -1] = 1
	c.program['u_light_mat'] = light_mat

	c.update()

	frame_render = c.render()
	frame_render = np.array(frame_render[:, :, 0:3])
	frame_render = cv2.cvtColor(frame_render, cv2.COLOR_RGB2BGR)

	hsv = cv2.cvtColor(frame_render, cv2.COLOR_BGR2HSV)
	hand_mask = cv2.inRange(hsv, (60, 0, 0), (80, 256, 256))

	######################
	# Hand Mesh
	######################

	result = frame.copy()
	result[np.where(hand_mask == 0)] = frame_render[np.where(hand_mask == 0)]

	cv2.imshow('frame', result)

	key = cv2.waitKey(1)
	if key == ord('q'):
		break