import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from config import cfg
from dataset import MultipleDatasets
# from realDataset import RealDataset
import numpy as np
import random

# dynamic dataset import
for i in range(len(cfg.trainset_3d)):
	exec('from ' + cfg.trainset_3d[i] + ' import ' + cfg.trainset_3d[i])
for i in range(len(cfg.trainset_2d)):
	exec('from ' + cfg.trainset_2d[i] + ' import ' + cfg.trainset_2d[i])
exec('from ' + cfg.testset + ' import ' + cfg.testset)

class Base(object):
	__metaclass__ = abc.ABCMeta

	def __init__(self, log_name='logs.txt'):
		self.cur_epoch = 0

		# timer
		self.tot_timer = Timer()
		self.gpu_timer = Timer()
		self.read_timer = Timer()

		# logger
		self.logger = colorlogger(cfg.log_dir, log_name=log_name)

	@abc.abstractmethod
	def _make_batch_generator(self, args):
		return

	@abc.abstractmethod
	def _make_model(self, args):
		return

class Tester(Base):
	def __init__(self, test_epoch):
		self.test_epoch = int(test_epoch)
		super(Tester, self).__init__(log_name='test_logs.txt')

	def _make_batch_generator(self, args):
		# data load and construct batch generator
		self.logger.info("Creating dataset...")
		testset_loader = eval(cfg.testset)(transforms.ToTensor(), "test")

		sampler = torch.utils.data.distributed.DistributedSampler(
			testset_loader,
			num_replicas=cfg.num_gpus,
			rank=args.local_rank,
			shuffle=False
		)

		batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
		                             sampler=sampler, # shuffle=False,
		                             num_workers=cfg.num_thread, pin_memory=True, drop_last=False)

		self.testset = testset_loader
		self.vertex_num = testset_loader.vertex_num
		self.joint_num = testset_loader.joint_num
		self.batch_generator = batch_generator

	def _make_model(self, args):
		model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
		if not cfg.rnn:
			assert os.path.exists(model_path), 'Cannot find model at ' + model_path
		self.logger.info('Load checkpoint from {}'.format(model_path))

		# prepare network
		self.logger.info("Creating graph...")
		from model_no_render import get_model

		torch.cuda.set_device(args.local_rank)

		world_size = cfg.num_gpus
		torch.distributed.init_process_group(
			'nccl',
			init_method='env://',
			world_size=world_size,
			rank=args.local_rank,
		)

		model = get_model(self.vertex_num, self.joint_num, 'test')

		model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
		device = torch.device('cuda:{}'.format(args.local_rank))
		model = model.to(device)
		model = torch.nn.parallel.DistributedDataParallel(
			model,
			device_ids=[args.local_rank],
			output_device=args.local_rank,
			find_unused_parameters=True
		)
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

		self.model = model

	def _evaluate(self, outs, cur_sample_idx):
		eval_result = self.testset.evaluate(outs, cur_sample_idx)
		return eval_result

	def _print_eval_result(self, eval_result):
		self.testset.print_eval_result(eval_result)
