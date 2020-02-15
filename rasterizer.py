
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from forwardsoftrasterize import forwardSoftRasterize

class SoftRasterizeFunction(Function):

    @staticmethod
    def forward(ctx, face_vertices, textures, image_size=50, background_color=[0, 0, 0], near=1, far=100, 
                fill_back=True, eps=1e-3, sigma_val=1e-5, dist_eps=1e-4, gamma_val=1e-4):

        # face_vertices: [nb, nf, 3, 3]
        # textures: [nb, nf, R * R, 3]
        # faces_info: [nb, nf, 27]
        # aggrs_info: [nb, 2, is, is]
        # soft_colors: [nb, 4, is, is]

        batch_size, num_faces = face_vertices.shape[:2]
        # faces_info: [inv*9, sym*9, obt*3, 0*6]
        faces_info = torch.FloatTensor(batch_size, num_faces, 9*3).fill_(0.0)
        aggrs_info = torch.FloatTensor(batch_size, 2, image_size, image_size).fill_(0.0) 
        soft_colors = torch.FloatTensor(batch_size, 4, image_size, image_size).fill_(1.0)
        for i in range(3):
            soft_colors[:, i, :, :] *= background_color[i]
        forwardSoftRasterize(face_vertices, textures, faces_info, aggrs_info, soft_colors,
            image_size, near, far, eps, sigma_val, dist_eps, gamma_val, fill_back)
        # forwardSoftRasterize fills faces_info and aggrs_info which will be used during backward
        return soft_colors

class SoftRasterizer(nn.Module):
    def __init__(self, image_size=50, background_color=[0, 0, 0], near=1, far=100, anti_aliasing=False, 
                fill_back=False, eps=1e-3, sigma_val=1e-5, dist_eps=1e-4, gamma_val=1e-4):
        super(SoftRasterizer, self).__init__()

        self.image_size = image_size
        self.background_color = background_color
        self.near = near
        self.far = far
        self.anti_aliasing = anti_aliasing
        self.eps = eps
        self.fill_back = fill_back
        self.sigma_val = sigma_val
        self.dist_eps = dist_eps
        self.gamma_val = gamma_val

    def forward(self, mesh, mode=None):
        image_size = self.image_size * (2 if self.anti_aliasing else 1)
        images = SoftRasterizeFunction.apply(mesh.face_vertices, mesh.face_textures, image_size, self.background_color, 
                                self.near, self.far, self.fill_back, self.eps, self.sigma_val, self.dist_eps, self.gamma_val)

        if self.anti_aliasing:
            images = F.avg_pool2d(images, kernel_size=2, stride=2)

        return images