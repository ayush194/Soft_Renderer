import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def lookAt(vertices, eye, at=[0, 0, 0], up=[0, 1, 0]):
    """
    "Look at" transformation of vertices.
    """
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')

    at = torch.tensor(at, dtype=torch.float32)
    up = torch.tensor(up, dtype=torch.float32)
    eye = torch.tensor(eye, dtype=torch.float32)

    batch_size = vertices.shape[0]
    at, up, eye = [x[None, :].repeat(batch_size, 1) if x.ndimension() == 1 else x for x in [at, up, eye]]

    # create new axes
    # eps is chosen as 1e-5 to match the chainer version
    z_axis = F.normalize(at - eye, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis), eps=1e-5)

    # create rotation matrix: [bs, 3, 3]
    r = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)

    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != eye.shape:
        eye = eye[:, None, :]
    vertices = vertices - eye
    vertices = torch.matmul(vertices, r.transpose(1,2))

    return vertices

def perspectiveTransform(vertices, angle=30.):
    '''
    Compute perspective distortion from a given angle
    '''
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')
    device = vertices.device
    angle = torch.tensor(angle / 180 * math.pi, dtype=torch.float32, device=device)
    angle = angle[None]
    width = torch.tan(angle)
    width = width[:, None] 
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] / z / width
    y = vertices[:, :, 1] / z / width
    vertices = torch.stack((x,y,z), dim=2)
    return vertices

def orthogonalTransform(vertices, scale):
    '''
    Compute orthogonal projection from a given angle
    To find equivalent scale to perspective projection
    set scale = focal_pixel / object_depth  -- to 0~H/W pixel range
              = 1 / ( object_depth * tan(half_fov_angle) ) -- to -1~1 pixel range
    '''
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] * scale
    y = vertices[:, :, 1] * scale
    vertices = torch.stack((x,y,z), dim=2)
    return vertices

def get_points_from_angles(distance, elevation, azimuth, degrees=True):
    if isinstance(distance, float) or isinstance(distance, int):
        if degrees:
            elevation = math.radians(elevation)
            azimuth = math.radians(azimuth)
        return (
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
            -distance * math.cos(elevation) * math.cos(azimuth))
    else:
        if degrees:
            elevation = math.pi / 180. * elevation
            azimuth = math.pi / 180. * azimuth
    #
        return torch.stack([
            distance * torch.cos(elevation) * torch.sin(azimuth),
            distance * torch.sin(elevation),
            -distance * torch.cos(elevation) * torch.cos(azimuth)
            ]).transpose(1,0)

def ambientLighting(light, light_intensity=0.5, light_color=(1,1,1)):

    light_color = torch.tensor(light_color, dtype=torch.float32)
    if light_color.ndimension() == 1:
        light_color = light_color[None, :]
    light += light_intensity * light_color[:, None, :]
    return light #[nb, :, 3]
    # returns a batch_size x #faces x 3 matrix where all the 3rd dim vectors are [0.5, 0.5, 0.5]
    # apparently light[i, j] representing the ambient light vector at jth face of ith batch

def directionalLighting(light, normals, light_intensity=0.5, light_color=(1,1,1), 
                         light_direction=(0,1,0)):
    # normals: [nb, :, 3]

    light_color = torch.tensor(light_color, dtype=torch.float32)
    light_direction = torch.tensor(light_direction, dtype=torch.float32)
    light_color, light_direction = [x[None, :] if x.ndimension() == 1 else x for x in [light_color, light_direction]]
    
    # light_color, light_direnction: [nb, 3]

    cosine = F.relu(torch.sum(normals * light_direction, dim=2)) #[]
    light += light_intensity * (light_color[:, None, :] * cosine[:, :, None])
    return light #[nb, :, 3]

def getFacesFromVertexIndices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    # replaces vertex indices in each face by the coordinates of the vertex

    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    faces = faces + (torch.arange(bs, dtype=torch.int32) * nv)[:, None, None]
    # since we flatten out the vertices in the statement below, we need to
    # added the required offset in order to correctly index the vertices
    vertices = vertices.reshape((bs * nv, 3))
    # the above statement flattens out the vertices in all the batches
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]
    # this replaces the vertex indices with the actual 3d coordinates of the vertices in faces matrix

def getVertexNormals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    # compute and returns normals for each face
    # for triangle ABC, normal is BC x BA

    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    normals = torch.zeros(bs * nv, 3)

    faces = faces + (torch.arange(bs, dtype=torch.int32) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(), 
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(), 
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals
