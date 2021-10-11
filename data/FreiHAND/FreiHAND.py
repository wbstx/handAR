import os
import os.path as osp
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import cv2
import random
import json
import math
import copy
from pycocotools.coco import COCO

import sys
sys.path.insert(0, osp.join('../..', 'main'))
sys.path.insert(0, osp.join('../..', 'data'))
sys.path.insert(0, osp.join('../..', 'common'))

from config import cfg
from utils.mano import MANO
from utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation
from utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db
from utils.vis import vis_keypoints, vis_mesh, save_obj


class FreiHAND(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('../../dataset/FreiHAND/data')
        self.human_bbox_root_dir = osp.join('../../dataset/FreiHAND/rootnet_output/bbox_root_freihand_output.json')
        
        # MANO joint set
        self.mano = MANO()
        self.face = self.mano.face
        self.joint_regressor = self.mano.joint_regressor
        self.vertex_num = self.mano.vertex_num
        self.joint_num = self.mano.joint_num
        self.joints_name = self.mano.joints_name
        self.skeleton = self.mano.skeleton
        self.root_joint_idx = self.mano.root_joint_idx

        self.datalist = self.load_data()
        # self.datalist = self.datalist[:256]

    def load_data(self):
        if self.data_split == 'train':
            db = COCO(osp.join(self.data_path, 'freihand_train_coco.json'))
            with open(osp.join(self.data_path, 'freihand_train_data.json')) as f:
                data = json.load(f)
            
        else:
            db = COCO(osp.join(self.data_path, 'freihand_eval_coco.json'))
            with open(osp.join(self.data_path, 'freihand_eval_data.json')) as f:
                data = json.load(f)
            print("Get bounding box and root from " + self.human_bbox_root_dir)
            bbox_root_result = {}
            with open(self.human_bbox_root_dir) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                bbox_root_result[str(annot[i]['image_id'])] = {'bbox': np.array(annot[i]['bbox']), 'root': np.array(annot[i]['root_cam'])}

        with open(osp.join(self.data_path, 'ensemble.json')) as f:
            test_gt = json.load(f)
            # 0 -> joint
            # 1 -> mesh

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.data_path, img['file_name'])
            img_shape = (img['height'], img['width'])
            db_idx = str(img['db_idx'])

            if self.data_split == 'train':
                cam_param, mano_param, joint_cam = data[db_idx]['cam_param'], data[db_idx]['mano_param'], data[db_idx]['joint_3d']
                joint_cam = np.array(joint_cam).reshape(-1,3)
                bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
                if bbox is None: continue
                root_joint_depth = joint_cam[self.root_joint_idx][2]

            else:
                cam_param, scale = data[db_idx]['cam_param'], data[db_idx]['scale']
                cam_param['R'] = np.eye(3).astype(np.float32).tolist(); cam_param['t'] = np.zeros((3), dtype=np.float32) # dummy
                joint_cam = np.ones((self.joint_num,3), dtype=np.float32) # dummy
                mano_param = {'pose': np.ones((48), dtype=np.float32), 'shape': np.ones((10), dtype=np.float32)}
                bbox = bbox_root_result[str(image_id)]['bbox'] # bbox should be aspect ratio preserved-extended. It is done in RootNet.
                root_joint_depth = bbox_root_result[str(image_id)]['root'][2]

                mesh_cam_out = test_gt[1][int(db_idx)]
                joint_cam_out = test_gt[0][int(db_idx)]


            if self.data_split == 'train':
                datalist.append({
                    'img_path': img_path,
                    'img_shape': img_shape,
                    'bbox': bbox,
                    'joint_cam': joint_cam,
                    'cam_param': cam_param,
                    'mano_param': mano_param,
                    'root_joint_depth': root_joint_depth})
            else:
                datalist.append({
                    'img_path': img_path,
                    'img_shape': img_shape,
                    'bbox': bbox,
                    'joint_cam': joint_cam,
                    'cam_param': cam_param,
                    'mano_param': mano_param,
                    'root_joint_depth': root_joint_depth,
                    'mesh_cam_out': mesh_cam_out,
                    'joint_cam_out': joint_cam_out})

        return datalist

    def get_mano_coord(self, mano_param, cam_param):
        pose, shape = mano_param['pose'], mano_param['shape']
        mano_pose = torch.FloatTensor(pose).view(-1,3); mano_shape = torch.FloatTensor(shape).view(1,-1); # mano parameters (pose: 48 dimension, shape: 10 dimension)
        R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3,3), np.array(cam_param['t'], dtype=np.float32).reshape(3) # camera rotation and translation
        mano_trans = torch.from_numpy(t).view(-1,3)

        # merge root pose and camera rotation
        root_pose = mano_pose[self.root_joint_idx,:].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
        mano_pose[self.root_joint_idx] = torch.from_numpy(root_pose).view(3)
        mano_pose = mano_pose.view(1,-1)

        # get mesh and joint coordinates
        mano_mesh_coord, mano_joint_coord = self.mano.layer(mano_pose, mano_shape, mano_trans)
        mano_mesh_coord = mano_mesh_coord.numpy().reshape(self.vertex_num,3); mano_joint_coord = mano_joint_coord.numpy().reshape(self.joint_num,3)

        # milimeter -> meter
        mano_mesh_coord /= 1000; mano_joint_coord /= 1000;
        return mano_mesh_coord, mano_joint_coord, mano_pose[0].numpy(), mano_shape[0].numpy()

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, joint_cam, cam_param, mano_param = data['img_path'], data['img_shape'], data['bbox'], data['joint_cam'], data['cam_param'], data['mano_param']

        num = int(img_path[-12:-4])
        num = num % 32560
        mask_path = img_path
        mask_path = mask_path[:-12] + str(num).zfill(8) + mask_path[-4:]
        mask_path = mask_path.replace('rgb', 'mask')

        tex_mask_path = img_path
        tex_mask_path = tex_mask_path[:-12] + str(num).zfill(8) + tex_mask_path[-4:]
        tex_mask_path = tex_mask_path.replace('rgb', 'tex_mask')

        # img
        img = load_img(img_path)
        if self.data_split == 'train':
            mask = load_img(mask_path)
            tex_mask = load_img(tex_mask_path)
            # tex_mask = cv2.cvtColor(tex_mask, cv2.COLOR_BGR2RGB)

        origin_img = img.copy()
        # bbox = [0, 0, 224, 224]

        if self.data_split == 'train':
            img, mask, tex_mask, img2bb_trans, bb2img_trans, rot, _ = augmentation(img, mask, tex_mask, bbox, self.data_split, exclude_flip=True) # FreiHAND dataset only contains right hands. do not perform flip aug.
            mask = cv2.resize(mask, (128, 128), cv2.INTER_NEAREST)
            tex_mask = cv2.resize(tex_mask, (128, 128), cv2.INTER_NEAREST)
            origin_img = self.transform(origin_img.astype(np.float32))/255.
            img = self.transform(img.astype(np.float32))/255.
            mask = self.transform(mask.astype(np.float32))/255.
            tex_mask = self.transform(tex_mask.astype(np.float32)) / 255.

            mask = mask[0:1, :, :]
            mask = 1. - mask
            mask = np.round(mask)

        else:
            img, mask, tex_mask, img2bb_trans, bb2img_trans, rot, _ = augmentation(img, img, img, bbox, self.data_split, exclude_flip=True) # FreiHAND dataset only contains right hands. do not perform flip aug.

            mask = cv2.resize(mask, (128, 128), cv2.INTER_NEAREST)
            tex_mask = cv2.resize(tex_mask, (128, 128), cv2.INTER_NEAREST)
            origin_img = self.transform(origin_img.astype(np.float32))/255.
            img = self.transform(img.astype(np.float32))/255.
            mask = self.transform(mask.astype(np.float32))/255.
            tex_mask = self.transform(tex_mask.astype(np.float32)) / 255.

            mask = mask[0:1, :, :]
            mask = 1. - mask
            mask = np.round(mask)

        if self.data_split == 'train':
            # mano coordinates
            mano_mesh_cam, mano_joint_cam, mano_pose, mano_shape = self.get_mano_coord(mano_param, cam_param)

            mano_coord_cam = np.concatenate((mano_mesh_cam, mano_joint_cam))
            focal, princpt = cam_param['focal'], cam_param['princpt']
            mano_coord_img = cam2pixel(mano_coord_cam, focal, princpt)
            origin_mesh_img = mano_coord_img[:self.vertex_num].copy()


            # affine transform x,y coordinates. root-relative depth
            mano_coord_img_xy1 = np.concatenate((mano_coord_img[:,:2], np.ones_like(mano_coord_img[:,:1])),1)
            mano_coord_img[:,:2] = np.dot(img2bb_trans, mano_coord_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]

            root_joint_depth = mano_coord_cam[self.vertex_num + self.root_joint_idx][2]
            mano_coord_img[:,2] = mano_coord_img[:,2] - root_joint_depth

            mano_coord_img[:,0] = mano_coord_img[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            mano_coord_img[:,1] = mano_coord_img[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
            mano_coord_img[:,2] = (mano_coord_img[:,2] / (cfg.bbox_3d_size / 2) + 1)/2. * cfg.output_hm_shape[0]

            # check truncation
            mano_trunc = ((mano_coord_img[:,0] >= 0) * (mano_coord_img[:,0] < cfg.output_hm_shape[2]) * \
                        (mano_coord_img[:,1] >= 0) * (mano_coord_img[:,1] < cfg.output_hm_shape[1]) * \
                        (mano_coord_img[:,2] >= 0) * (mano_coord_img[:,2] < cfg.output_hm_shape[0])).reshape(-1,1).astype(np.float32)

            # split mesh and joint coordinates
            mano_mesh_img = mano_coord_img[:self.vertex_num]; mano_joint_img = mano_coord_img[self.vertex_num:];
            mano_mesh_trunc = mano_trunc[:self.vertex_num]; mano_joint_trunc = mano_trunc[self.vertex_num:];

            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1]], dtype=np.float32)
            # parameter

            mano_pose = mano_pose.reshape(-1,3)
            root_pose = mano_pose[self.root_joint_idx,:]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
            mano_pose[self.root_joint_idx] = root_pose.reshape(3)
            mano_pose = mano_pose.reshape(-1)

            # mano coordinate
            mano_joint_cam = mano_joint_cam - mano_joint_cam[self.root_joint_idx,None] # root-relative
            mano_joint_cam = np.dot(rot_aug_mat, mano_joint_cam.transpose(1,0)).transpose(1,0)

            mano_mesh_cam[:,2] = mano_mesh_cam[:,2] - root_joint_depth

            # R, t = torch.from_numpy(np.array(cam_param['R'], dtype=np.float32).reshape(3,3)), torch.from_numpy(np.array(cam_param['t'], dtype=np.float32).reshape(3)) # camera rotation and translation
            # inputs = {'img': img, 'origin_img': origin_img, 'cam_param': cam_param, 'R': R, 't': t, 'img_path': img_path}
            inputs = {'img': img, 'origin_img': origin_img, # 'cam_param': cam_param,
                      'img_path': img_path}

            # targets = {'orig_joint_img': orig_joint_img, 'fit_joint_img': mano_joint_img, 'fit_mesh_img': mano_mesh_img, 'orig_joint_cam': orig_joint_cam, 'fit_joint_cam': mano_joint_cam, 'pose_param': mano_pose, 'shape_param': mano_shape,
            #     'fit_mesh_cam': mano_mesh_cam, 'root_joint_depth': root_joint_depth, 'mask': mask, 'tex_mask': tex_mask, 'origin_mesh_img': origin_mesh_img}
            targets = {'fit_joint_img': mano_joint_img, 'fit_mesh_img': mano_mesh_img, 'fit_joint_cam': mano_joint_cam,
                'fit_mesh_cam': mano_mesh_cam, 'root_joint_depth': root_joint_depth, 'mask': mask, 'tex_mask': tex_mask,
                       'mano_param': mano_pose, 'mano_shape': mano_shape}

            # meta_info = {'orig_joint_valid': orig_joint_valid, 'orig_joint_trunc': orig_joint_trunc, 'fit_joint_trunc': mano_joint_trunc, 'fit_mesh_trunc': mano_mesh_trunc, 'is_valid_fit': float(True), 'is_3D': float(True),
            #     'bb2img_trans': bb2img_trans, 'img2bb_trans': img2bb_trans}

            meta_info = {'fit_joint_trunc': mano_joint_trunc, 'fit_mesh_trunc': mano_mesh_trunc, 'is_valid_fit': float(True), 'is_3D': float(True),
                'bb2img_trans': bb2img_trans, 'img2bb_trans': img2bb_trans, 'param': float(True), 'mask': float(True)}
        else:
            inputs = {'img': img}
            targets = {}
            meta_info = {'bb2img_trans': bb2img_trans}

        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):

        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'joint_out': [], 'mesh_out': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]

            # x,y: resize to input image space and perform bbox to image affine transform
            mesh_out_img = out['mesh_coord_img']
            # mesh_out_img = out['gcn']
            mesh_out_img[:,0] = mesh_out_img[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            mesh_out_img[:,1] = mesh_out_img[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            mesh_out_img_xy1 = np.concatenate((mesh_out_img[:,:2], np.ones_like(mesh_out_img[:,:1])),1)
            mesh_out_img[:,:2] = np.dot(out['bb2img_trans'], mesh_out_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]

            # z: devoxelize and translate to absolute depth
            root_joint_depth = annot['root_joint_depth']
            mesh_out_img[:,2] = (mesh_out_img[:,2] / cfg.output_hm_shape[0] * 2. - 1) * (cfg.bbox_3d_size / 2)
            mesh_out_img[:,2] = mesh_out_img[:,2] + root_joint_depth

            # camera back-projection
            cam_param = annot['cam_param']
            focal, princpt = cam_param['focal'], cam_param['princpt']
            mesh_out_cam = pixel2cam(mesh_out_img, focal, princpt)
            
            if cfg.stage == 'param':
                mesh_out_cam = out['mesh_coord_cam']
            joint_out_cam = np.dot(self.joint_regressor, mesh_out_cam)

            eval_result['mesh_out'].append(mesh_out_cam.tolist())
            eval_result['joint_out'].append(joint_out_cam.tolist())
 
            vis = False
            if vis:
                filename = annot['img_path'].split('/')[-1][:-4]

                img = load_img(annot['img_path'])[:,:,::-1]
                img = vis_mesh(img, mesh_out_img, 0.5)
                cv2.imwrite('objs/' + filename + '.jpg', img)

                filename = 'objs/' + annot['img_path'].split('/')[-1][:-4]
                save_obj(mesh_out_cam, self.mano.face, filename + '.obj')
        return eval_result

    def print_eval_result(self, eval_result):
        output_save_path = osp.join(cfg.result_dir, 'pred.json')
        with open(output_save_path, 'w') as f:
            json.dump([eval_result['joint_out'], eval_result['mesh_out']], f)
        print('Saved at ' + output_save_path)


if __name__ == '__main__':
    dataset = FreiHAND(transforms.ToTensor(), "train")
    batch_generator = DataLoader(dataset=dataset, batch_size=1,
                                 shuffle=False,
                                 num_workers=30, pin_memory=True)

    for itr, (inputs, targets, meta_info) in enumerate(batch_generator):
        if itr < 10:
            target = (inputs['img'][0] * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            # joints = (targets['fit_joint_img'])
            # print(targets['fit_joint_cam'][0].cpu().numpy())

            # print(joints)
            #
            # mask = (targets['mask'][0] * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            # mask = cv2.resize(mask, (256, 256))
            # mask = cv2.addWeighted(target, 0.5, mask, 0.5, 0)
            #
            target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
            # # from utils.vis import vis_mesh, save_obj, render_mesh, vis_normal
            target1 = vis_mesh(target, targets['fit_mesh_img'][0].cpu().numpy() * 4)
            # target2 = vis_keypoints(target, targets['fit_joint_img'][0].cpu().numpy() * 4)
            # # target2 = vis_normal(target, targets['fit_mesh_img'][0].cpu().numpy() * 4, verts_normals[0, :, :].cpu().numpy())

            save_obj(targets['fit_mesh_cam'][0].cpu().numpy(), dataset.mano.face, str(itr) + '.obj')

            cv2.imwrite('t' + str(itr) + '.png', target1)
            # cv2.imwrite('tt' + str(itr) + '.png', target2)
            # cv2.imwrite('ttt' + str(itr) + '.png', mask)
        else:
            break
