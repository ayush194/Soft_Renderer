
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

from lighting import Lighting
from transform import Transform
from rasterizer import SoftRasterizer
from mesh import Mesh

class SoftRenderer(nn.Module):
    def __init__(self, image_size=100, background_color=[0,0,0], near=1, far=100, anti_aliasing=True, 
                fill_back=True, eps=1e-3, sigma_val=1e-5, dist_eps=1e-4, gamma_val=1e-4,
                P=None, dist_coeffs=None, orig_size=512, perspective=True, viewing_angle=30, 
                viewing_scale=1.0, eye=None, camera_direction=[0,0,1],
                light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                light_directions=[0,1,0]):
        super(SoftRenderer, self).__init__()

        # light
        self.lighting = Lighting(light_intensity_ambient, light_color_ambient, light_intensity_directionals, 
                                light_color_directionals, light_directions)

        # camera
        self.transform = Transform(P, dist_coeffs, orig_size, perspective, viewing_angle, viewing_scale, 
                                    eye, camera_direction)

        # rasterization
        self.rasterizer = SoftRasterizer(image_size, background_color, near, far, anti_aliasing, 
                                        fill_back, eps, sigma_val, dist_eps, gamma_val)

    def set_sigma(self, sigma):
        self.rasterizer.sigma_val = sigma

    def set_gamma(self, gamma):
        self.rasterizer.gamma_val = gamma

    def render_mesh(self, mesh, mode=None):
        mesh = self.lighting(mesh)
        mesh = self.transform(mesh)
        return self.rasterizer(mesh, mode)

    def forward(self, vertices, faces, textures=None, mode=None):
        mesh = Mesh(vertices, faces, textures=textures)
        return self.render_mesh(mesh, mode)