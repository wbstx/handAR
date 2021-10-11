import numpy as np
import vispy
from vispy import app, io, gloo, scene, visuals
from vispy.util.transforms import perspective, translate, rotate, ortho, scale
from vispy.visuals.transforms import STTransform, NullTransform
from vispy.geometry import meshdata as md
from vispy import keys
import GLSLOperator
import os
from vispy.gloo.util import _screenshot
import scipy.misc
import time
import pywavefront

import util
import VirtualObject as vo


class MyMeshData(md.MeshData):
	""" Add to Meshdata class the capability to export good data for gloo """

	def __init__(self, vertices=None, faces=None, edges=None,
	             vertex_colors=None, face_colors=None):
		md.MeshData.__init__(self, vertices=None, faces=None, edges=None,
		                     vertex_colors=None, face_colors=None)

	def set_group(self, groups):
		assert groups.shape[0] == self.V.shape[0]
		self.groups = groups

	def get_glTriangles(self):
		"""
		Build vertices for a colored mesh.
				V  is the vertices
				I1 is the indices for a filled mesh (use with GL_TRIANGLES)
				I2 is the indices for an outline mesh (use with GL_LINES)
		"""
		vtype = [('a_position', np.float32, 4),
		         ('a_normal', np.float32, 3),
		         ('a_color', np.float32, 3)]
		vertices = self.get_vertices()
		vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
		normals = self.get_vertex_normals()
		faces = np.uint32(self.get_faces())

		edges = np.uint32(self.get_edges().reshape((-1)))
		# colors = self.get_vertex_colors()

		nbrVerts = vertices.shape[0]
		self.V = np.zeros(nbrVerts, dtype=vtype)
		self.V[:]['a_position'] = vertices
		self.V[:]['a_normal'] = normals
		self.initial_V = self.V.copy()
		# V[:]['a_color'] = colors[:,:-1]
		return self.V, faces.reshape((-1)), edges.reshape((-1))


class Canvas(app.Canvas):

	def __init__(self, title='MeshViewer', size=(320, 320), mesh_name='model', has_texture=False):
		app.Canvas.__init__(self, keys='interactive')
		self.size = size
		self.has_texture = has_texture

		self._rendertex = gloo.Texture2D((size + (3,)))
		# Create FBO, attach the color buffer and depth buffer
		self._fbo = gloo.FrameBuffer(self._rendertex, gloo.RenderBuffer(size, format='depth'))

		self.title = title
		self.real_mesh_name = mesh_name
		self.mesh_name = ''.join([i for i in mesh_name if not i.isdigit()])
		self.program = GLSLOperator.create_program('glsl/vert_mesh.glsl', 'glsl/frag_mesh.glsl')

		self.default_model = np.dot(rotate(0, axis=[0, 0, 1]), rotate(0, axis=[0, 0, 1]))
		if self.mesh_name == 'radio':
			self.default_model = np.dot(rotate(-90, axis=[1, 0, 0]), translate([0, 0.7, 0]))

		self.model = self.default_model
		GLSLOperator.set_default_MVP(self.program)
		self.view_scale = 4
		self.view = translate((0, 0, -self.view_scale))
		self.default_view = self.view.copy()
		self.program['u_view'] = self.view
		self.program['u_model'] = self.model

		self.model_name = '../model/' + self.mesh_name + '.obj'
		if has_texture:
			if os.path.exists('../model/' + self.mesh_name + '.png'):
				tex = io.imread('../model/' + self.mesh_name + '.png')
			else:
				tex = io.imread('../model/' + self.mesh_name + '.jpg')
		else:
			tex = np.zeros((512, 512, 3)).astype(np.uint8)
		self.program['has_texture'] = has_texture
		self.program['u_texture'] = gloo.Texture2D(tex, interpolation='linear')
		self._button = None
		self.visible = True

		self.timer = app.Timer('auto', connect=self.on_timer, start=True)

		self.img_id = 0
		self.isStart = False

		self.time_decay = 1.0
		self.start_time = np.Infinity
		self.init_data()

		gloo.set_state('opaque')
		gloo.set_polygon_offset(1, 1)
		gloo.set_state(depth_test=True)
		gloo.set_depth_range(0.0, 1.0)

		self.show()

	# ---------------------------------

	# ---------------------------------

	def on_timer(self, event):

		def cross(a, b):
			c = [a[1] * b[2] - a[2] * b[1],
			     a[2] * b[0] - a[0] * b[2],
			     a[0] * b[1] - a[1] * b[0]]

			return c

		if self.isStart:
			# ori = np.array(cross([0, 0, 1], self.sample_loc[self.img_id, :]))
			# ori = ori / np.sqrt(np.sum(np.square(ori)))
			# angle = np.dot([0, 0, 1], self.sample_loc[self.img_id, :]) * 180 / np.pi
			#
			# ori = (np.array([0, 0, 1]) + np.array(self.sample_loc[self.img_id, :])) / 2.0
			angle = 180
			# print(ori, angle)
			# self.set_model(rotate(angle, ori))

			if (time.time() - self.start_time > self.time_decay):
				self.start_time = time.time()
				self.im = _screenshot((0, 0, self.size[0], self.size[1]))
				scipy.misc.imsave("imgs/{}.png".format(self.img_id), self.im)
				self.img_id = (self.img_id + 1) % self.total_len
				if self.img_id == 0:
					self.isStart = False

	# ---------------------------------
	def on_resize(self, event):
		self.apply_zoom()

	def set_model(self, model):
		self.model = model
		self.program['u_model'] = self.model
		self.update()

	def apply_zoom(self):
		gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
		self.projection = perspective(60.0, self.physical_size[0] /
		                              float(self.physical_size[1]), 1.0, 100.0)
		self.program['u_projection'] = self.projection

	def draw_view(self):
		gloo.set_state(blend=False, depth_test=True,
		               polygon_offset_fill=True)
		self.program['u_color'] = 1, 1, 1, 1
		self.program.draw('triangles', self.filled_buf)

	def on_draw(self, event):
		gloo.set_clear_color('green')
		gloo.clear(color=True, depth=True)
		vp = (0, 0, self.physical_size[0], self.physical_size[1])
		gloo.set_viewport(*vp)

		# self.program['u_view'] = np.dot(rotate(-90*((i+1)//2),axis=[i//2,1-i//2,0]),self.view)
		self.draw_view()

	def init_data(self, normalization=True):
		verts, faces, normals, texcoords, mtls, groups, materials, mapping = io.read_mesh(self.model_name, self.has_texture)
		# verts, faces, normals, texcoords, usingmtls, materials = io.read_mesh(self.model_name)
		# io.write_mesh('newgun.obj',verts, faces, normals, texcoords)
		self.texcoords = texcoords.copy()
		self.mapping = mapping
		mats = np.zeros((66, 3)).astype(np.float32)
		for index, material in enumerate(materials):
			mats[index * 3, :] = material.ka
			mats[index * 3 + 1, :] = material.kd
			mats[index * 3 + 2, :] = material.ks
		mats[1, :] = [0.588, 0.588, 0.588]
		if normalization:
			# centroid = np.mean(verts, axis=0, keepdims=True)
			# furthest_distance = np.amax(np.sqrt(np.sum((verts - centroid) ** 2, axis=-1)), keepdims=True)
			#
			# verts = (verts - centroid) / furthest_distance
			verts = util.mesh_normalization(verts)
		# self.virtual_object = vo.RubiksCube(verts, groups)
		# self.virtual_object = vo.Cards(verts, groups)

		self.mesh = MyMeshData()
		self.mesh.set_vertices(verts)
		self.mesh.set_faces(faces)
		# mesh.set_vertex_colors(colors)
		vertices, filled, outline = self.mesh.get_glTriangles()
		self.faces = filled
		self.set_data(vertices, filled, outline)
		self.program['u_mat_rendering'] = 0.0
		groups = np.zeros_like(verts.astype(np.float32)[:, 0:1])
		groups[778:] = 1
		if self.has_texture:
			texcoords[:, 1] = 1 - texcoords[:, 1]
			self.program['a_texcoord'] = gloo.VertexBuffer(texcoords[:, 0:2])
			self.program['a_mtl'] = gloo.VertexBuffer(verts.astype(np.float32)[:, 0:1])
			# self.program['a_group'] = gloo.VertexBuffer(verts.astype(np.float32)[:, 0:1])
			self.program['a_group'] = gloo.VertexBuffer(groups)
		else:
			self.program['a_mtl'] = gloo.VertexBuffer(np.array(mtls).astype(np.float32))
			# self.program['a_group'] = gloo.VertexBuffer(np.array(groups).astype(np.float32))
			self.program['a_group'] = gloo.VertexBuffer(verts.astype(np.float32)[:, 0:1])
			self.program['a_texcoord'] = gloo.VertexBuffer(verts.astype(np.float32)[:, 0:2])
			self.program['u_materials'] = mats.astype(np.float32)
			self.update()

	# ---------------------------------
	def set_data(self, vertices, filled, outline):
		self.filled_buf = gloo.IndexBuffer(filled)
		self.outline_buf = gloo.IndexBuffer(outline)
		self.vertices_buff = gloo.VertexBuffer(vertices)
		self.program.bind(self.vertices_buff)
		self.update()

	def on_mouse_press(self, event):
		if event.button == 1:
			self._button = event
		else:
			self._button = None

	def on_mouse_release(self, event):
		self._button = None

	def on_mouse_move(self, event):
		import math
		if event.button == 1:
			dx, dy = self._button.pos - event.pos
			nx = -dy
			ny = -dx
			scale = max(math.sqrt(nx * nx + ny * ny), 1e-6)
			nx = nx / scale
			ny = ny / scale
			angle = scale * 0.01 * 80 / 90.0
			self.model = np.dot(rotate(angle, (nx, ny, 0)), self.model)
			self.program['u_model'] = self.model
			self.update()

	def on_mouse_wheel(self, event):
		if event.delta[1] > 0:
			self.model = np.dot(scale([1.5, 1.5, 1.5]), self.model)
		else:
			self.model = np.dot(scale([0.9, 0.9, 0.9]), self.model)
		self.program['u_model'] = self.model
		self.update()

	def on_key_press(self, event):
		if event.key == keys.SPACE:
			print('start to snap shoot')
			self.isStart = True
			self.start_time = time.time()
		if event.key == keys.ESCAPE:
			exit(0)