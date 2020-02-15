
import math
import numpy as np
import torch
import torch.nn as nn

from extras import lookAt, perspectiveTransform, orthogonalTransform, get_points_from_angles

class LookAt(nn.Module):
    def __init__(self, perspective=True, viewing_angle=30, viewing_scale=1.0, eye=None):
        super(LookAt, self).__init__()

        self.perspective = perspective
        self.viewing_angle = viewing_angle
        self.viewing_scale = viewing_scale
        self._eye = eye

        if self._eye is None:
            self._eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]

    def forward(self, vertices):
        vertices = lookAt(vertices, list(self._eye))
        # perspective transformation
        if self.perspective:
            vertices = perspectiveTransform(vertices, angle=self.viewing_angle)
        else:
            vertices = orthogonalTransform(vertices, scale=self.viewing_scale)
        return vertices

class Transform(nn.Module):
    def __init__(self, P=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1]):
        super(Transform, self).__init__()

        # camera mode is LookAt
        self.transformer = LookAt(perspective, viewing_angle, viewing_scale, eye)

    def forward(self, mesh):
        mesh.vertices = self.transformer(mesh.vertices)
        return mesh

    def set_eyes_from_angles(self, distances, elevations, azimuths):
        self.transformer._eye = get_points_from_angles(distances, elevations, azimuths)

    def set_eyes(self, eyes):
        self.transformer._eye = eyes

    @property
    def eyes(self):
        return self.transformer._eyes
    