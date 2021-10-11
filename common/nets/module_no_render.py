import torch
import torch.nn as nn
from torch.nn import functional as F
from config import cfg
from nets.layer import make_conv_layers, make_deconv_layers, make_conv1d_layers, make_linear_layers

import torchgeometry as tgm
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv, JumpingKnowledge
from torch_geometric.transforms import FaceToEdge
from torch_geometric.utils import dropout_adj


heatmap_tensor = torch.tensor([0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63], requires_grad=False)

class PoseNet(nn.Module):
    def __init__(self, joint_num):
        super(PoseNet, self).__init__()
        self.joint_num = joint_num
        self.deconv = make_deconv_layers([2048,256,256,256])

        self.conv_x = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([2048,256*cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0)
        self.conv_z_2 = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

        self.conv = make_conv_layers([256, 64], kernel=3, stride=1, padding=1)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        # coord = heatmap1d * torch.cuda.comm.broadcast(torch.arange(heatmap_size).type(torch.cuda.FloatTensor), devices=[heatmap1d.device.index])[0]
        coord = heatmap1d * torch.arange(heatmap_size).type(torch.cuda.FloatTensor).to(heatmap1d.device)
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat, feats=None):
        # Batchsize x 2048 x 8 x 8
        img_feat_xy = self.deconv(img_feat)
        # Batchsize x 256 x 64 x 64

        # # x axis
        # img_feat_x = img_feat_xy.mean((2))
        # heatmap_x = self.conv_x(img_feat_x)
        # coord_x = self.soft_argmax_1d(heatmap_x)
        #
        # # y axis
        # img_feat_y = img_feat_xy.mean((3))
        # heatmap_y = self.conv_y(img_feat_y)
        # # (BatchSize x 21 x 64)
        # coord_y = self.soft_argmax_1d(heatmap_y)
        #
        # # z axis
        # img_feat_z = img_feat.mean((2,3))[:,:,None]
        # # (BatchSize x 2048 x 1)
        # img_feat_z = self.conv_z_1(img_feat_z)
        # img_feat_z = img_feat_z.view(-1,256,cfg.output_hm_shape[0])
        # heatmap_z = self.conv_z_2(img_feat_z)
        # # (BatchSize x 21 x 64)
        # coord_z = self.soft_argmax_1d(heatmap_z)
        #
        # joint_coord = torch.cat((coord_x, coord_y, coord_z),2)
        # # joint_coord_feat = torch.cat((heatmap_x, heatmap_y, heatmap_z),2)

        joint_feat = self.conv(img_feat_xy)
        # joint_feat = img_feat_xy

        return joint_feat, joint_feat


class SegmentationNet(nn.Module):
    def __init__(self):
        super(SegmentationNet, self).__init__()
        self.conv1 = nn.Conv2d(2048, 1024, 1, stride=1, padding=0)
        # self.conv1 = nn.Conv2d(2048, 1024, 3, stride=1, padding=1)
        self.fusion1 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)

        self.conv2 = nn.Conv2d(1024, 512, 1, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(1024, 512, 3, stride=1, padding=1)
        self.fusion2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        self.conv3 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
        # self.conv3 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.fusion3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 64, 1, stride=1, padding=0)
        # self.conv3 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.fusion4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.predict_head = nn.Sequential(nn.Conv2d(64, 16, 3, stride=1, padding=1),
                                          nn.Conv2d(16, 2, 1))

    def forward(self, img_feat, feats, feats128):  # 8 x 8
        x = nn.functional.interpolate(img_feat, scale_factor=2, mode='bilinear')
        x = self.conv1(x) # 16 x 16
        x = self.fusion1(x + feats[-1])
        x = F.relu(self.bn1(x))

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv2(x) # 32 x 32
        x = self.fusion2(x + feats[-2])
        x = F.relu(self.bn2(x))

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv3(x) # 64 x 64
        x = self.fusion3(x + feats[-3])
        x = F.relu(self.bn3(x))

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv4(x)
        x = self.fusion4(x + feats128)

        x = self.predict_head(x) # 128 x 128
        return F.log_softmax(x, dim=1)


class HeatmapNet(nn.Module):
    def __init__(self, joint_num):
        super(HeatmapNet, self).__init__()
        self.joint_num = joint_num
        self.deconv = make_deconv_layers([2048,256,256,256])

        self.conv = make_conv_layers([256, self.joint_num], kernel=3, stride=1, padding=1)

        self.conv_x = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([2048, 256*cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0)
        self.conv_z_2 = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax(self, x, beta=1):
        # heatmap1d = F.softmax(heatmap1d, 2)
        x_exp = torch.exp(x * beta)
        x_exp_sum = torch.sum(x_exp, 2, keepdim=True)
        x_size = x.shape[2]
        # coord = x_exp / x_exp_sum * torch.cuda.comm.broadcast(torch.arange(x_size).type(torch.cuda.FloatTensor), devices=[x.device.index])[0]
        coord = x_exp / x_exp_sum * torch.arange(x_size).type(torch.cuda.FloatTensor).to(x.device)[0]
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat):
        # Batchsize x 2048 x 8 x 8
        img_feat_xy = self.deconv(img_feat)
        # Batchsize x 256 x 64 x 64
        heatmap = self.conv(img_feat_xy)

        # x axis
        img_feat_x = heatmap.mean((2))
        # heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax(img_feat_x)

        # y axis
        img_feat_y = heatmap.mean((3))
        # heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax(img_feat_y)

        # z axis
        img_feat_z = img_feat.mean((2, 3))[:, :, None]
        # (BatchSize x 2048 x 1)
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1,256,cfg.output_hm_shape[0])
        heatmap_z = self.conv_z_2(img_feat_z)
        # (BatchSize x 21 x 64)
        coord_z = self.soft_argmax(heatmap_z)

        joint_coord = torch.cat((coord_x, coord_y, coord_z), 2)

        return heatmap, joint_coord

# Stage Render
# class Pose2Feat(nn.Module):
#     def __init__(self, joint_num):
#         super(Pose2Feat, self).__init__()
#         self.joint_num = joint_num
#         self.conv = make_conv_layers([64+joint_num, 64], kernel=1, stride=1, padding=0, bnrelu_final=True)
#
#     def forward(self, pos_feat):
#         feat = self.conv(pos_feat)
#         return feat

# Stage Lixel
class Pose2Feat(nn.Module):
    def __init__(self, joint_num):
        super(Pose2Feat, self).__init__()
        self.joint_num = joint_num
        self.conv = make_conv_layers([64 + joint_num * cfg.output_hm_shape[0], 64])
        # self.conv = make_conv_layers([320+joint_num*cfg.output_hm_shape[0],64])

    def forward(self, img_feat, joint_heatmap_3d):
        # Batchsize x 21 x 64 x 64 x 64
        joint_heatmap_3d = joint_heatmap_3d.view(-1,self.joint_num*cfg.output_hm_shape[0],cfg.output_hm_shape[1],cfg.output_hm_shape[2])
        # img_feat # Batchsize x 64 x 64 x 64
        feat = torch.cat((img_feat, joint_heatmap_3d),1)
        feat = self.conv(feat)
        return feat

class MeshNet(nn.Module):
    def __init__(self, vertex_num):
        super(MeshNet, self).__init__()
        self.vertex_num = vertex_num
        self.deconv = make_deconv_layers([2048,256,256,256])

        # self.conv1 = nn.Conv2d(2048, 1024, 1, stride=1, padding=0)
        # self.fusion1 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(1024)
        #
        # self.conv2 = nn.Conv2d(1024, 512, 1, stride=1, padding=0)
        # self.fusion2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(512)
        #
        # self.conv3 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
        # self.fusion3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(256)
        #
        # self.conv4 = nn.Conv2d(256, 512, 1, stride=1, padding=0)
        # self.se_block = SEBlock(512)

        self.conv_x = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([2048,256*cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0)
        self.conv_z_2 = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax(self, x, beta=1):
        x_exp = torch.exp(x * beta)
        x_exp_sum = torch.sum(x_exp, 2, keepdim=True)
        x_size = x.shape[2]
        # coord = x_exp / x_exp_sum * torch.cuda.comm.broadcast(torch.arange(x_size).type(torch.cuda.FloatTensor), devices=[x.device.index])[0]
        # coord = x_exp / x_exp_sum * torch.arange(x_size).type(torch.cuda.FloatTensor).to(x.device)
        coord = x_exp / x_exp_sum * heatmap_tensor.type(torch.cuda.FloatTensor).to(x.device)
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        # coord = heatmap1d * torch.cuda.comm.broadcast(torch.arange(heatmap_size).type(torch.cuda.FloatTensor), devices=[heatmap1d.device.index])[0]
        # coord = heatmap1d * torch.arange(heatmap_size).type(torch.cuda.FloatTensor).to(heatmap1d.device)
        coord = heatmap1d * heatmap_tensor.type(torch.cuda.FloatTensor).to(heatmap1d.device)
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat, feats=None):
        img_feat_xy = self.deconv(img_feat)

        # x = nn.functional.interpolate(img_feat, scale_factor=2, mode='bilinear')
        # x = self.conv1(x) # 16 x 16
        # x = self.fusion1(x + feats[-1])
        # x = F.relu(self.bn1(x))
        #
        # x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        # x = self.conv2(x) # 32 x 32
        # x = self.fusion2(x + feats[-2])
        # x = F.relu(self.bn2(x))
        #
        # x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        # x = self.conv3(x) # 64 x 64
        # x = self.fusion3(x + feats[-3])
        # x = F.relu(self.bn3(x))
        #
        # x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        # x = self.conv4(x) # 128 x 128
        # x = self.se_block(x)
        #
        # img_feat_xy = x

        # x axis
        img_feat_x = img_feat_xy.mean((2))
        heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax_1d(heatmap_x)

        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y)

        # z axis
        img_feat_z = img_feat.mean((2,3))[:,:,None]
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1,256,cfg.output_hm_shape[0])
        heatmap_z = self.conv_z_2(img_feat_z)
        coord_z = self.soft_argmax_1d(heatmap_z)

        mesh_coord = torch.cat((coord_x, coord_y, coord_z),2)
        return mesh_coord

class ParamRegressor(nn.Module):
    def __init__(self, joint_num):
        super(ParamRegressor, self).__init__()
        self.joint_num = joint_num
        # self.fc = make_linear_layers([self.joint_num*3, 1024, 512], use_bn=True)
        self.fc = make_linear_layers([778*3, 1024, 512], use_bn=True)
        # if 'FreiHAND' in cfg.trainset_3d:
        self.fc_pose = make_linear_layers([512, 16*6], relu_final=False) # hand joint orientation
        # else:
        #     self.fc_pose = make_linear_layers([512, 24*6], relu_final=False) # body joint orientation
        self.fc_shape = make_linear_layers([512, 10], relu_final=False) # shape parameter

    def rot6d_to_rotmat(self,x):
        x = x.view(-1,3,2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2)
        return torch.stack((b1, b2, b3), dim=-1)

    def forward(self, mesh_3d):
        # pose_3d = pose_3d.view(-1,self.joint_num*3)
        mesh_3d = mesh_3d.view(-1, 778*3)
        feat = self.fc(mesh_3d)

        pose = self.fc_pose(feat)
        pose = self.rot6d_to_rotmat(pose)
        pose = torch.cat([pose, torch.zeros((pose.shape[0],3,1)).cuda().float()],2)
        # Bug fixed here
        # if 'FreiHAND' in cfg.trainset_3d:
        pose = tgm.rotation_matrix_to_angle_axis(pose).reshape(-1, 16*3)
        # else:
        #     pose = tgm.rotation_matrix_to_angle_axis(pose).reshape(-1, 24*3)

        shape = self.fc_shape(feat)

        return pose, shape

class GraphConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, relu_bn=True):
        super(GraphConv, self).__init__()
        nn1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.conv = GINConv(nn1)
        self.bn = torch.nn.BatchNorm1d(output_dim)

        # self.conv = GCNConv(input_dim, output_dim)
        # self.bn = nn.GroupNorm(output_dim // 8, output_dim)
        self.relu_bn = relu_bn
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        if self.relu_bn:
            out = self.bn(out)
            out = F.relu(out)
        return out

class GraphResConv(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(GraphResConv, self).__init__()
        nn1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        # self.bn1 = nn.GroupNorm(hidden_dim // 8, hidden_dim)

        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        # self.bn2 = nn.GroupNorm(hidden_dim // 8, hidden_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        self.conv2.reset_parameters()
        self.bn2.reset_parameters()

    def forward(self, x, edge_index):
        identity = x

        out = self.conv1(x, edge_index)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out, edge_index)
        out = self.bn2(out)
        out = F.relu(out)
        out = out + identity

        return out

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, drop_edge=False):
        super(GCN, self).__init__()
        # self.conv1 = SAGEConv(input_dim, 32)
        # self.conv2 = SAGEConv(32, 32)
        # self.conv3 = SAGEConv(32, output_dim)

        # self.conv1 = GraphConv(input_dim, 256, hidden_dim)
        # self.conv2 = GraphConv(hidden_dim, hidden_dim, hidden_dim)
        # self.conv3 = GraphConv(hidden_dim, hidden_dim, output_dim, relu_bn=False)
        # self.lin = nn.Linear(hidden_dim, 64)
        # self.bn = nn.BatchNorm1d(64)
        # self.tail = nn.Linear(64, output_dim)

        # self.jk = JumpingKnowledge(mode='lstm', channels=128, num_layers=3)
        # nn2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
        # self.conv2 = GINConv(nn2)
        #
        # nn3 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, output_dim))
        # self.conv3 = GINConv(nn3)

        self.head = GraphConv(input_dim, 256, hidden_dim)
        self.res1 = GraphResConv(hidden_dim)
        self.res2 = GraphResConv(hidden_dim)
        self.res3 = GraphResConv(hidden_dim)
        # self.res4 = GraphResConv(hidden_dim)
        self.tail = GraphConv(hidden_dim, 32, output_dim, relu_bn=False)

        self.drop_edge = drop_edge

        self.reset_parameters()

    def reset_parameters(self):
        # self.conv1.reset_parameters()
        # self.conv2.reset_parameters()
        # self.conv3.reset_parameters()
        # self.lin.reset_parameters()
        # self.bn.reset_parameters()
        # self.tail.reset_parameters()
        # self.jk.reset_parameters()
        self.head.reset_parameters()
        self.res1.reset_parameters()
        self.res2.reset_parameters()
        self.res3.reset_parameters()
        # self.res4.reset_parameters()
        self.tail.reset_parameters()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # if self.drop_edge:
        #     edge_index, _ = dropout_adj(edge_index, p=0.5)

        # out = self.conv1(x, edge_index)
        # out = self.conv2(out, edge_index)
        # out = self.conv3(out, edge_index)

        out = self.head(x, edge_index)
        out = self.res1(out, edge_index)
        out = self.res2(out, edge_index)
        out = self.res3(out, edge_index)
        # out = self.res4(out, edge_index)
        out = self.tail(out, edge_index)

        # x1 = F.relu(self.conv1(x, edge_index))
        # x2 = F.relu(self.conv2(x1, edge_index))
        # x_gcn = self.conv3(x2, edge_index)

        return out

class GCNGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(GCNGRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # nn1 = nn.Sequential(nn.Linear(hidden_dim+input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        # self.convz = GINConv(nn1)
        self.convz = nn.Conv1d(hidden_dim + input_dim, hidden_dim, kernel_size=1)

        # nn2 = nn.Sequential(nn.Linear(hidden_dim+input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        # self.convr = GINConv(nn2)
        self.convr = nn.Conv1d(hidden_dim + input_dim, hidden_dim, kernel_size=1)

        # nn3 = nn.Sequential(nn.Linear(hidden_dim+input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        # self.convq = GINConv(nn3)
        self.convq = nn.Conv1d(hidden_dim + input_dim, hidden_dim, kernel_size=1)

        self.reset_parameter()

    def reset_parameter(self):
        self.convz.reset_parameters()
        self.convr.reset_parameters()
        self.convq.reset_parameters()

    def forward(self, h, B):
        x, edge_index = B.x, B.edge_index
        x = x.reshape(-1, 778, self.input_dim).permute(0, 2, 1)
        h = h.reshape(-1, 778, self.hidden_dim).permute(0, 2, 1)
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        # z = torch.sigmoid(self.convz(hx, edge_index))
        # r = torch.sigmoid(self.convr(hx, edge_index))
        # q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1), edge_index))

        h = (1-z) * h + z * q
        h = h.permute(0, 2, 1)
        h = h.reshape(-1, self.hidden_dim)
        return h

class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(UpdateBlock, self).__init__()
        self.gru = GCNGRU(hidden_dim=hidden_dim, input_dim=input_dim)
        self.head = GCN(input_dim=hidden_dim, output_dim=3, drop_edge=False)

        # self.mask = nn.Sequential(
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 16*9, 1, padding=0))

    def forward(self, B, hidden):

        h = self.gru(hidden, B)
        data = Data(x=h, edge_index=B.edge_index)
        offset = self.head(data)

        return offset, h

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)