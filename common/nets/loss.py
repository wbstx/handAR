import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from config import cfg

class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(self, coord_out, coord_gt, valid, is_3D=None):
        if cfg.loss_type == 'l1':
            loss = torch.abs(coord_out - coord_gt) * valid
        elif cfg.loss_type == 'l2':
            loss = torch.square(coord_out - coord_gt) * valid

        if is_3D is not None:
            loss_z = loss[:,:,2:] * is_3D[:,None,None].float()
            loss = torch.cat((loss[:,:,:2], loss_z),2)
        return loss


class SequenceLoss(nn.Module):
    def __init__(self):
        super(SequenceLoss, self).__init__()

    def forward(self, preds, gt, valid, gamma=0.8):
        n_predictions = len(preds)
        loss = 0.0
        weight = 0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            if cfg.loss_type == 'l1':
                i_loss = torch.abs(preds[i] - gt) * valid
            elif cfg.loss_type == 'l2':
                i_loss = torch.square(preds[i] - gt) * valid
            loss += i_weight * i_loss
            weight += i_weight
        return loss / weight


class SequenceNormalLoss(nn.Module):
    def __init__(self, face):
        super(SequenceNormalLoss, self).__init__()
        self.normal_loss = NormalVectorLoss(face)

    def forward(self, preds, gt, valid, gamma=0.8):
        n_predictions = len(preds)
        loss = 0.0
        weight = 0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            i_loss = self.normal_loss(preds[i], gt, valid)
            loss += i_weight * i_loss
            weight += i_weight
        return loss / weight


class SequenceEdgeLoss(nn.Module):
    def __init__(self, face):
        super(SequenceEdgeLoss, self).__init__()
        self.edge_loss = EdgeLengthLoss(face)

    def forward(self, preds, gt, valid, gamma=0.8):
        n_predictions = len(preds)
        loss = 0.0
        weight = 0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            i_loss = self.edge_loss(preds[i], gt, valid)
            loss += i_weight * i_loss
            weight += i_weight
        return loss / weight


class ParamLoss(nn.Module):
    def __init__(self):
        super(ParamLoss, self).__init__()

    def forward(self, param_out, param_gt, valid):
        loss = torch.abs(param_out - param_gt) * valid
        return loss

class NormalVectorLoss(nn.Module):
    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt, valid):
        face = torch.LongTensor(self.face).cuda()

        v1_out = coord_out[:,face[:,1],:] - coord_out[:,face[:,0],:]
        v1_out = F.normalize(v1_out, p=2, dim=2) # L2 normalize to make unit vector
        v2_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,0],:]
        v2_out = F.normalize(v2_out, p=2, dim=2) # L2 normalize to make unit vector
        v3_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,1],:]
        v3_out = F.normalize(v3_out, p=2, dim=2) # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:,face[:,1],:] - coord_gt[:,face[:,0],:]
        v1_gt = F.normalize(v1_gt, p=2, dim=2) # L2 normalize to make unit vector
        v2_gt = coord_gt[:,face[:,2],:] - coord_gt[:,face[:,0],:]
        v2_gt = F.normalize(v2_gt, p=2, dim=2) # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2) # L2 normalize to make unit vector

        valid_mask = valid[:,face[:,0],:] * valid[:,face[:,1],:] * valid[:,face[:,2],:]
        
        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True)) * valid_mask
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True)) * valid_mask
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True)) * valid_mask
        loss = torch.cat((cos1, cos2, cos3),1)
        return loss

class EdgeLengthLoss(nn.Module):
    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt, valid):
        face = torch.LongTensor(self.face).cuda()

        d1_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,1],:])**2,2,keepdim=True))
        d2_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True))
        d3_out = torch.sqrt(torch.sum((coord_out[:,face[:,1],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,1],:])**2,2,keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,1],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True))

        valid_mask_1 = valid[:,face[:,0],:] * valid[:,face[:,1],:]
        valid_mask_2 = valid[:,face[:,0],:] * valid[:,face[:,2],:]
        valid_mask_3 = valid[:,face[:,1],:] * valid[:,face[:,2],:]
        
        diff1 = torch.abs(d1_out - d1_gt) * valid_mask_1
        diff2 = torch.abs(d2_out - d2_gt) * valid_mask_2
        diff3 = torch.abs(d3_out - d3_gt) * valid_mask_3
        loss = torch.cat((diff1, diff2, diff3),1)
        return loss

