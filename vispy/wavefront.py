# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# This module was taken from visvis
"""
This module produces functionality to read and write wavefront (.OBJ) files.

http://en.wikipedia.org/wiki/Wavefront_.obj_file

The wavefront format is quite powerful and allows a wide variety of surfaces
to be described.

This implementation does only supports mesh stuff, so no nurbs etc. Further,
material properties are ignored, although this might be implemented later,

The classes are written with compatibility of Python3 in mind.

"""

import numpy as np
import time
from os import path as op
import collections

from ..ext.gzip_open import gzip_open
from ..geometry import _calculate_normals
from ..util import logger


class Material(object):
	def __init__(self):
		self.ka = [0., 0., 0.]
		self.kd = [0., 0., 0.]
		self.ks = [0., 0., 0.]
		self.ke = [0., 0., 0.]

		self.set_ka = False
		self.set_kd = False
		self.set_ks = False
		self.set_ke = False

		self.name = None

	def set_value(self, str_target, line):
		if str_target != 'name':
			numbers = [num for num in line.split(' ') if num]
			value = [float(num) for num in numbers[1:3 + 1]]
			if str_target == 'ka':
				self.ka = value
				self.set_ka = True
			if str_target == 'kd':
				self.kd = value
				self.set_kd = True
			if str_target == 'ks':
				self.ks = value
				self.set_ks = True
			if str_target == 'ke':
				self.ke = value
				self.set_ke = True
		else:
			self.name = line.split(' ')[1]

	def check_complete(self):
		if self.set_ka or self.set_kd or self.set_ks:
		# if self.set_ka and self.set_kd and self.set_ks:
			return True
		return False

	def log(self):
		print(self.name + ' ' + str(self.ka) + ' ' + str(self.kd) + ' ' + str(self.ks) + ' ' + str(self.ke))


def findmtl(mtl_name, mtls):
	m = filter(lambda mtl: mtl.name == mtl_name, mtls)
	return mtls.index(list(m)[0])


class WavefrontReader(object):

	def __init__(self, f):
		self._f = f

		# Original vertices, normals and texture coords.
		# These are not necessarily of the same length.
		self._v = []
		self._vn = []
		self._vt = []

		# Final vertices, normals and texture coords.
		# All three lists are of the same length, as opengl wants it.
		self._vertices = []
		self._normals = []
		self._texcords = []

		self._usingmtls = collections.OrderedDict()

		self._group = []
		self._group_num = 0
		# The faces, indices to vertex/normal/texcords arrays.
		self._faces = []

		# Dictionary to keep track of processed face data, so we can
		# convert the original v/vn/vn to the final vertices/normals/texcords.
		self._facemap = {}

		self._mtl = []
		self.materials = []

	@classmethod
	def read(cls, fname, has_texture):
		""" read(fname, fmt)

		This classmethod is the entry point for reading OBJ files.

		Parameters
		----------
		fname : str
				The name of the file to read.
		fmt : str
				Can be "obj" or "gz" to specify the file format.
		"""
		# Open file
		cls.fname= fname
		fmt = op.splitext(fname)[1].lower()
		assert fmt in ('.obj', '.gz')
		opener = open if fmt == '.obj' else gzip_open
		with opener(fname, 'rb') as f:
			try:
				reader = WavefrontReader(f)
				while True:
					reader.readLine()
			except EOFError:
				pass

		# Done
		t0 = time.time()
		mesh = reader.finish(has_texture)
		logger.debug('reading mesh took ' +
		             str(time.time() - t0) +
		             ' seconds')
		return mesh

	def readLine(self):
		""" The method that reads a line and processes it.
		"""

		# Read line
		line = self._f.readline().decode('ascii', 'ignore')
		if not line:
			raise EOFError()
		line = line.strip()

		if line.startswith('v '):
			# self._vertices.append( *self.readTuple(line) )
			self._v.append(self.readTuple(line))
			self._group.append(self._group_num)
		elif line.startswith('vt '):
			self._vt.append(self.readTuple(line, 3))
		elif line.startswith('vn '):
			self._vn.append(self.readTuple(line))
		elif line.startswith('f '):
			self._faces.append(self.readFace(line))
		elif line.startswith('usemtl '):
			mat = line.split(' ')[1]
			mat = self.materials.index(list(filter(lambda m: m.name == mat, self.materials))[0])
			self._usingmtls[len(self._faces)] = mat
		elif line.startswith('#'):
			pass  # Comment
		elif line.startswith('mtllib '):
			folder = self.fname.split('/')[0]
			self.readMtlFile(folder + '/' + line.split(' ')[1])
		# logger.warning('Notice reading .OBJ: material properties are '
		#                'ignored.')
		elif line.startswith('g '):
			# pass
			self._group_num += 1
		##### Current using mat
		elif line.startswith('usemtl '):
			self.mat = findmtl(line.split(' ')[1], self.materials)
		elif any(line.startswith(x) for x in ('s ', 'o ')):
			pass  # Ignore groups and smoothing groups, obj names, material
		elif not line.strip():
			pass
		else:
			logger.warning('Notice reading .OBJ: ignoring %s command.'
			               % line.strip())

	def readMtlFile(self, fname):
		with open(fname, 'rb') as f:
			try:
				current_material = Material()
				while True:
					line = f.readline().decode('ascii', 'ignore')
					if not line:
						raise EOFError()
					line = line.strip()
					if line.startswith('newmtl '):
						current_material.set_value('name', line)
					elif line.startswith('Ka '):
						current_material.set_value('ka', line)
						if current_material.check_complete():
							self.materials.append(current_material)
							current_material = Material()
					elif line.startswith('Kd '):
						current_material.set_value('kd', line)
						if current_material.check_complete():
							self.materials.append(current_material)
							current_material = Material()
					elif line.startswith('Ks '):
						current_material.set_value('ks', line)
						if current_material.check_complete():
							self.materials.append(current_material)
							current_material = Material()
					elif line.startswith('Ke '):
						current_material.set_value('ke', line)
						if current_material.check_complete():
							self.materials.append(current_material)
							current_material = Material()
					elif line.startswith('#'):
						pass
			except EOFError:
				# for material in self.materials:
				# 	material.log()
				pass

	def readTuple(self, line, n=3):
		""" Reads a tuple of numbers. e.g. vertices, normals or teture coords.
		"""
		numbers = [num for num in line.split(' ') if num]
		return [float(num) for num in numbers[1:n + 1]]

	def readFace(self, line):
		""" Each face consists of three or more sets of indices. Each set
		consists of 1, 2 or 3 indices to vertices/normals/texcords.
		"""

		# Get parts (skip first)
		indexSets = [num for num in line.split(' ') if num][1:]

		final_face = []
		for indexSet in indexSets:

			# Did we see this exact index earlier? If so, it's easy
			final_index = self._facemap.get(indexSet)
			if final_index is not None:
				final_face.append(final_index)
				continue

			# If not, we need to sync the vertices/normals/texcords ...

			# Get and store final index
			final_index = len(self._vertices)
			final_face.append(final_index)
			self._facemap[indexSet] = final_index

			# What indices were given?
			indices = [i for i in indexSet.split('/')]

			# Store new set of vertex/normal/texcords.
			# If there is a single face that does not specify the texcord
			# index, the texcords are ignored. Likewise for the normals.
			if True:
				vertex_index = self._absint(indices[0], len(self._v))
				self._vertices.append(self._v[vertex_index])
			if self._texcords is not None:
				if len(indices) > 1 and indices[1]:
					texcord_index = self._absint(indices[1], len(self._vt))
					self._texcords.append(self._vt[texcord_index])
				else:
					if self._texcords:
						logger.warning('Ignoring texture coordinates because '
						               'it is not specified for all faces.')
					self._texcords = None
			if self._normals is not None:
				if len(indices) > 2 and indices[2]:
					normal_index = self._absint(indices[2], len(self._vn))
					self._normals.append(self._vn[normal_index])
				else:
					if self._normals:
						logger.warning('Ignoring normals because it is not '
						               'specified for all faces.')
					self._normals = None

		# Check face
		if self._faces and len(self._faces[0]) != len(final_face):
			raise RuntimeError(
				'Vispy requires that all faces are either triangles or quads.')

		# Done
		return final_face

	def _absint(self, i, ref):
		i = int(i)
		if i > 0:
			return i - 1
		else:
			return ref + i

	def _calculate_normals(self):
		vertices, faces = self._vertices, self._faces
		if faces is None:
			# ensure it's always 2D so we can use our methods
			faces = np.arange(0, vertices.size, dtype=np.uint32)[:, np.newaxis]
		print(faces.shape)
		normals = _calculate_normals(vertices, faces)
		return normals

	def rearrange(self):
		_v = []
		_f = []
		_mtl = []
		_group = []
		print(self._usingmtls)
		for i in range(1, len(self._usingmtls) + 1):
			shown_vertices = {}
			corr = len(_v)

			if i != len(self._usingmtls):
				end = list(self._usingmtls.keys())[i]
			else:
				end = len(self._faces)

			for j in range(list(self._usingmtls.keys())[i - 1], end):
				face = self._faces[j]

				### Check if v has appeared
				for v in face:
					if not v in shown_vertices:
						shown_vertices[v] = len(_v)
						_v.append(self._vertices[v])
						_group.append(self._group[v])
						_mtl.append(list(self._usingmtls.values())[i - 1])

				_f.append([shown_vertices[face[0]], shown_vertices[face[1]], shown_vertices[face[2]]])
		return np.array(_v), np.array(_f), np.array(_mtl).astype(np.int8), np.array(_group)

	def finish(self, has_texture):
		""" Converts gathere lists to numpy arrays and creates
		BaseMesh instance.
		"""
		self._vertices = np.array(self._vertices, 'float32')
		if self._faces:
			self._faces = np.array(self._faces, 'uint32')
		else:
			# Use vertices only
			self._vertices = np.array(self._v, 'float32')
			self._faces = None
		if self._normals:
			self._normals = np.array(self._normals, 'float32')
		else:
			self._normals = self._calculate_normals()
		if self._texcords:
			self._texcords = np.array(self._texcords, 'float32')
		else:
			self._texcords = None

		# print("Before: ", self._vertices.shape)
		if not has_texture:
			self._vertices, self._faces, self._mtl, self._group = self.rearrange()
		# print("After: ", self._vertices.shape)

		self._normals = self._calculate_normals()

		mapping = np.zeros((len(self._facemap)))
		for k, v in self._facemap.items():
			mapping[v] = int(k.split('/')[0])

		return self._vertices, self._faces, self._normals, \
		       self._texcords, self._mtl, self._group, self.materials, mapping


class WavefrontWriter(object):

	def __init__(self, f):
		self._f = f

	@classmethod
	def write(cls, fname, vertices, faces, normals,
	          texcoords, name='', reshape_faces=True):
		""" This classmethod is the entry point for writing mesh data to OBJ.

		Parameters
		----------
		fname : string
				The filename to write to. Must end with ".obj" or ".gz".
		vertices : numpy array
				The vertex data
		faces : numpy array
				The face data
		texcoords : numpy array
				The texture coordinate per vertex
		name : str
				The name of the object (e.g. 'teapot')
		reshape_faces : bool
				Reshape the `faces` array to (Nf, 3). Set to `False`
				if you need to write a mesh with non triangular faces.
		"""
		# Open file
		fmt = op.splitext(fname)[1].lower()
		if fmt not in ('.obj', '.gz'):
			raise ValueError('Filename must end with .obj or .gz, not "%s"'
			                 % (fmt,))
		opener = open if fmt == '.obj' else gzip_open
		f = opener(fname, 'wb')
		try:
			writer = WavefrontWriter(f)
			writer.writeMesh(vertices, faces, normals,
			                 texcoords, name, reshape_faces=reshape_faces)
		except EOFError:
			pass
		finally:
			f.close()

	def writeLine(self, text):
		""" Simple writeLine function to write a line of code to the file.
		The encoding is done here, and a newline character is added.
		"""
		text += '\n'
		self._f.write(text.encode('ascii'))

	def writeTuple(self, val, what):
		""" Writes a tuple of numbers (on one line).
		"""
		# Limit to three values. so RGBA data drops the alpha channel
		# Format can handle up to 3 texcords
		val = val[:3]
		# Make string
		val = ' '.join([str(v) for v in val])
		# Write line
		self.writeLine('%s %s' % (what, val))

	def writeFace(self, val, what='f'):
		""" Write the face info to the net line.
		"""
		# OBJ counts from 1
		val = [v + 1 for v in val]
		# Make string
		if self._hasValues and self._hasNormals:
			val = ' '.join(['%i/%i/%i' % (v, v, v) for v in val])
		elif self._hasNormals:
			val = ' '.join(['%i//%i' % (v, v) for v in val])
		elif self._hasValues:
			val = ' '.join(['%i/%i' % (v, v) for v in val])
		else:
			val = ' '.join(['%i' % v for v in val])
		# Write line
		self.writeLine('%s %s' % (what, val))

	def writeMesh(self, vertices, faces, normals, values,
	              name='', reshape_faces=True):
		""" Write the given mesh instance.
		"""

		# Store properties
		self._hasNormals = normals is not None
		self._hasValues = values is not None
		self._hasFaces = faces is not None

		# Get faces and number of vertices
		if faces is None:
			faces = np.arange(len(vertices))
			reshape_faces = True

		if reshape_faces:
			Nfaces = faces.size // 3
			faces = faces.reshape((Nfaces, 3))
		else:
			is_triangular = np.array([len(f) == 3
			                          for f in faces])
			if not (np.all(is_triangular)):
				logger.warning('''Faces doesn't appear to be triangular,
                be advised the file cannot be read back in vispy''')
		# Number of vertices
		N = vertices.shape[0]

		# Get string with stats
		stats = []
		stats.append('%i vertices' % N)
		if self._hasValues:
			stats.append('%i texcords' % N)
		else:
			stats.append('no texcords')
		if self._hasNormals:
			stats.append('%i normals' % N)
		else:
			stats.append('no normals')
		stats.append('%i faces' % faces.shape[0])

		# Write header
		self.writeLine('# Wavefront OBJ file')
		self.writeLine('# Created by vispy.')
		self.writeLine('#')
		if name:
			self.writeLine('# object %s' % name)
		else:
			self.writeLine('# unnamed object')
		self.writeLine('# %s' % ', '.join(stats))
		self.writeLine('')

		# Write data
		if True:
			for i in range(N):
				self.writeTuple(vertices[i], 'v')
		if self._hasNormals:
			for i in range(N):
				self.writeTuple(normals[i], 'vn')
		if self._hasValues:
			for i in range(N):
				self.writeTuple(values[i], 'vt')
		if True:
			for i in range(faces.shape[0]):
				self.writeFace(faces[i])
