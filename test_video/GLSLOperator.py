#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

from vispy import gloo
from vispy.util.transforms import perspective, translate, rotate
import numpy as np
'''
Loading GLSL File with filename
'''
def load_glsl(filename):
    with open(filename) as file_object:
        file_context = file_object.read()
        return file_context


def create_program(vert, frag):
    vert_glsl = load_glsl(vert)
    frag_glsl = load_glsl(frag)

    return gloo.Program(vert_glsl, frag_glsl, count=4)

def set_default_MVP(program):
    model = np.eye(4, dtype=np.float32)
    #model = rotate(180,axis=[0,0,1])
    print(model)
    projection = np.eye(4, dtype=np.float32)
    view = np.dot(translate((0, 0, -6.0)),rotate(180,axis=[0,0,1]))

    program['u_model'] = model
    program['u_view'] = view
    program['u_projection'] = projection

    return program





