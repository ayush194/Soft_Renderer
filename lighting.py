import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from extras import ambientLighting, directionalLighting

class AmbientLighting(nn.Module):
    def __init__(self, light_intensity=0.5, light_color=(1,1,1)):
        super(AmbientLighting, self).__init__()

        self.light_intensity = light_intensity
        self.light_color = light_color

    def forward(self, light):
        return ambientLighting(light, self.light_intensity, self.light_color)


class DirectionalLighting(nn.Module):
    def __init__(self, light_intensity=0.5, light_color=(1,1,1), light_direction=(0,1,0)):
        super(DirectionalLighting, self).__init__()

        self.light_intensity = light_intensity
        self.light_color = light_color
        self.light_direction = light_direction

    def forward(self, light, normals):
        return directionalLighting(light, normals, self.light_intensity, self.light_color, self.light_direction)


class Lighting(nn.Module):
    def __init__(self, intensity_ambient=0.5, color_ambient=[1,1,1],
                intensity_directionals=0.5, color_directionals=[1,1,1],
                directions=[0,1,0]):
        super(Lighting, self).__init__()

        self.ambient = AmbientLighting(intensity_ambient, color_ambient)
        self.directionals = nn.ModuleList([DirectionalLighting(intensity_directionals,
                                                               color_directionals,
                                                               directions)])

    def forward(self, mesh):
        light = torch.zeros_like(mesh.faces, dtype=torch.float32)
        light = light.contiguous()
        light = self.ambient(light)
        for directional in self.directionals:
            light = directional(light, mesh.surface_normals)
        mesh.textures = mesh.textures * light[:, :, None, :]

        return mesh