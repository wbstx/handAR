import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module_no_render import PoseNet, Pose2Feat, MeshNet, ParamRegressor, HeatmapNet, GCN, UpdateBlock, SegmentationNet
from nets.loss import CoordLoss, ParamLoss, NormalVectorLoss, EdgeLengthLoss, SequenceLoss, SequenceNormalLoss, SequenceEdgeLoss
from utils.smpl import SMPL
from utils.mano import MANO
from utils.transforms import pixel2cam, cam2pixel
from config import cfg
import math

from utils.manopth.manopth.manolayer import ManoLayer

import cv2
import numpy as np

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.transforms import FaceToEdge
from torch_geometric.utils import dropout_adj

from nets.layer import make_conv1d_layers

img_size = 128

class Model(nn.Module):
	def __init__(self, pose_backbone, pose_net, pose2feat, mesh_backbone, mesh_net):
		super(Model, self).__init__()
		self.pose_backbone = pose_backbone
		self.pose_net = pose_net
		self.pose2feat = pose2feat

		self.mesh_backbone = mesh_backbone
		self.mesh_net = mesh_net
		self.gcn = GCN(input_dim=3 + 64 + 64 + 256 + 512 + 3, hidden_dim=64, output_dim=3, drop_edge=True)

		self.segmentation_net = SegmentationNet()
		self.segmentation_net.apply(init_weights)

		self.global_img_feat = nn.Sequential(
			make_conv1d_layers([2048, 512], kernel=1, stride=1, padding=0, bnrelu_final=False)
		)

		self.param_regressor = ParamRegressor(21)

		self.batch_size = cfg.train_batch_size

		self.human_model = MANO()
		self.human_model_layer = self.human_model.layer.to(torch.cuda.current_device())

		self.root_joint_idx = self.human_model.root_joint_idx

		self.joint_regressor = self.human_model.joint_regressor

		self.coord_loss = CoordLoss()
		self.param_loss = ParamLoss()
		self.normal_loss = NormalVectorLoss(self.human_model.face)
		self.edge_loss = EdgeLengthLoss(self.human_model.face)
		self.mask_loss = nn.NLLLoss()

		self.pose_skeleton = torch.from_numpy(np.array(self.human_model.skeleton)).permute(1, 0).to(torch.cuda.current_device())
		self.pose_skeleton = self.pose_skeleton.long()
		self.face = self.human_model_layer.th_faces.permute(1, 0).to(torch.cuda.current_device())
		self.face2edge = FaceToEdge()

	def project(self, img_feat, vertices, img_size):
		v = vertices[:, :, :2]
		v = v.unsqueeze(2)
		v = v / 32. - 1.0
		output = F.grid_sample(img_feat, v, align_corners=False)
		return output.squeeze(-1).permute(0, 2, 1)

	def forward(self, inputs, targets, meta_info, mode):
		#############################
		# Stage: Pose
		#############################
		shared_img_feat, pose_img_feat, feats, feats128 = self.pose_backbone(inputs['img'])

		joint_coord_img, joint_feat = self.pose_net(pose_img_feat, feats)  # BatchSize x 21 x 3

		#############################
		# Stage: mesh
		#############################
		
		_, mesh_img_feat, mesh_feats = self.mesh_backbone(joint_feat+shared_img_feat, skip_early=True)
		global_img_feat = self.global_img_feat(mesh_img_feat.mean((2,3))[:,:,None])
		global_img_feat = global_img_feat.permute(0, 2, 1).repeat(1, 778, 1)

		mesh_coord_img = self.mesh_net(mesh_img_feat, mesh_feats)

		joint_img_from_mesh = torch.bmm(
			torch.from_numpy(self.joint_regressor).cuda()[None, :, :].repeat(mesh_coord_img.shape[0], 1, 1),
			mesh_coord_img)

		rough_mesh = mesh_coord_img.clone()

		#############################
		# Stage: refine
		#############################

		B = Batch.from_data_list([Data(x=x, face=self.face).to(torch.cuda.current_device()) for x in rough_mesh])
		B = self.face2edge(B)  # -> B is the rough output

		proj_img = self.project(inputs['img'], B.x.view(-1, 778, 3), 64)
		proj_feat0 = self.project(feats128, B.x.view(-1, 778, 3), 64)
		proj_feat1 = self.project(shared_img_feat, B.x.view(-1, 778, 3), 64)
		proj_feat2 = self.project(mesh_feats[0], B.x.view(-1, 778, 3), 64)

		cat_feat = torch.cat((proj_img, proj_feat0, proj_feat1, proj_feat2, global_img_feat), dim=2).view(-1, 3 + 64 + 64 + 256 + 512)  # ((N x 778) x 320)

		B.x = torch.cat((B.x, cat_feat), dim=1)
		x = self.gcn(B).view(-1, 778, 3)
		x = x + rough_mesh

		# test output
		out = {}
		out['joint_coord_img'] = joint_img_from_mesh
		out['mesh_coord_img'] = mesh_coord_img
		out['bb2img_trans'] = meta_info['bb2img_trans']
		out['gcn'] = x.view(-1, 778, 3)
		out['pose'] = joint_img_from_mesh

		if 'fit_mesh_coord_cam' in targets:
			out['mesh_coord_cam_target'] = targets['fit_mesh_coord_cam']
		return out

def init_weights(m):
	if type(m) == nn.ConvTranspose2d:
		nn.init.kaiming_normal_(m.weight)
	elif type(m) == nn.Conv2d:
		nn.init.kaiming_normal_(m.weight)
		nn.init.constant_(m.bias, 0)
	elif type(m) == nn.BatchNorm2d:
		nn.init.constant_(m.weight,1)
		nn.init.constant_(m.bias,0)
	elif type(m) == nn.Linear:
		nn.init.kaiming_normal_(m.weight)
		nn.init.constant_(m.bias,0)

def get_model(vertex_num, joint_num, mode):
	pose_backbone = ResNetBackbone(cfg.resnet_type)
	pose_net = PoseNet(joint_num)
	pose2feat = Pose2Feat(joint_num)
	mesh_backbone = ResNetBackbone(cfg.resnet_type)
	mesh_net = MeshNet(vertex_num)

	if mode == 'train':
		pose_backbone.init_weights()
		pose_net.apply(init_weights)
		pose2feat.apply(init_weights)
		mesh_backbone.init_weights()
		mesh_net.apply(init_weights)
		# param_regressor.apply(init_weights)

	model = Model(pose_backbone, pose_net, pose2feat, mesh_backbone, mesh_net)
	return model

