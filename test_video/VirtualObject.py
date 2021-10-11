import numpy as np
from abc import ABCMeta, abstractmethod


class VirtualObject(metaclass=ABCMeta):
	def __init__(self, verts, groups):
		self.verts = verts
		self.groups = groups
		self.num_verts = verts.shape[0]
		self.num_groups = groups[-1] + 1

		self.hypergroups = {}

	@abstractmethod
	def define_hypergroups(self):
		pass

	@abstractmethod
	def map_hypergroups(self):
		pass

	def construct_active_groups(self, hypergroup):
		active_parts = np.zeros((self.num_groups, 1)).astype(np.float32)
		active_groups = np.zeros(self.num_verts)
		for group in self.hypergroups[hypergroup]:
			active_groups[np.where(self.groups==group)] = 1
			active_parts[group, 0] = 1.0
		return active_parts, active_groups

class RubiksCube(VirtualObject):
	def __init__(self, verts, groups):
		super(RubiksCube, self).__init__(verts, groups)
		self.topology = self.construct_cube_topology()
		self.define_hypergroups()

	def define_hypergroups(self):
		self.hypergroups['r'] = list(self.topology[2].flatten())
		self.hypergroups['rd'] = list(self.topology[2].flatten()) + list(self.topology[1].flatten())

	def map_hypergroups(self, touching_group):
		if touching_group in [22, 23, 24]:
			return 'rd'
		elif touching_group in [5, 6, 7]:
			return 'r'
		else:
			return 'invalid'

	def construct_cube_topology(self):
		topology = np.zeros((3, 3, 3)).astype(np.int)
		topology[2] = np.array([
			[6, 5, 4],
			[7, 2, 3],
			[8, 9, 10]
		])
		topology[1] = np.array([
			[23, 22, 21],
			[24, 1, 20],
			[25, 26, 27]
		])
		topology[0] = np.array([
			[15, 14, 13],
			[16, 11, 12],
			[17, 18, 19]
		])
		topology -= 1
		return topology


class Cards(VirtualObject):
	def __init__(self, verts, groups):
		super(Cards, self).__init__(verts, groups)
		self.topology = self.construct_cards_topology()
		self.define_hypergroups()

	def define_hypergroups(self):
		self.hypergroups['l'] = self.topology[0]
		self.hypergroups['m'] = self.topology[1]
		self.hypergroups['r'] = self.topology[2]

	def map_hypergroups(self, touching_group):
		if touching_group in self.topology[0]:
			return 'l'
		elif touching_group in self.topology[1]:
			return 'm'
		elif touching_group in self.topology[2]:
			return 'r'
		else:
			return 'invalid'

	def construct_cards_topology(self):
		topology = []
		topology.append([0, 1, 2, 3, 4, 5, 6])
		topology.append([7, 8, 9, 10, 11, 12, 13, 14])
		topology.append([15, 16, 17, 18, 19, 20, 21, 22])
		return topology
