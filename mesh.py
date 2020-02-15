import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from extras import getFacesFromVertexIndices, getVertexNormals
from objloader import loadObject
#from save_obj import save_obj


class Mesh(object):
    '''
    A simple class for creating and manipulating trimesh objects
    '''
    def __init__(self, vertices, faces, textures=None, texture_res=1):
        '''
        vertices, faces and textures(if not None) are expected to be Tensor objects
        '''
        self._vertices = vertices
        self._faces = faces

        if isinstance(self._vertices, np.ndarray):
            self._vertices = torch.from_numpy(self._vertices).float()
        if isinstance(self._faces, np.ndarray):
            self._faces = torch.from_numpy(self._faces).int()
        if self._vertices.ndimension() == 2:
            self._vertices = self._vertices[None, :, :]
        if self._faces.ndimension() == 2:
            self._faces = self._faces[None, :, :]

        self.batch_size = self._vertices.shape[0]
        self.num_vertices = self._vertices.shape[1]
        self.num_faces = self._faces.shape[1]

        self._face_vertices = None
        self._face_vertices_update = True
        self._surface_normals = None
        self._surface_normals_update = True
        self._vertex_normals = None
        self._vertex_normals_update = True

        self._fill_back = False

        # create textures
        if textures is None:
            self._textures = torch.ones(self.batch_size, self.num_faces, texture_res**2, 3, 
                                        dtype=torch.float32)
            self.texture_res = texture_res
        else:
            if isinstance(textures, np.ndarray):
                textures = torch.from_numpy(textures).float()
            if textures.ndimension() == 3:
                textures = textures[None, :, :, :]
            if textures.ndimension() == 2:
                textures = textures[None, :, :]
            self._textures = textures
            self.texture_res = int(np.sqrt(self._textures.shape[2]))

        self._origin_vertices = self._vertices
        self._origin_faces = self._faces
        self._origin_textures = self._textures

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, faces):
        # need check tensor
        self._faces = faces
        self.num_faces = self._faces.shape[1]
        self._face_vertices_update = True
        self._surface_normals_update = True
        self._vertex_normals_update = True

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        # need check tensor
        self._vertices = vertices
        self.num_vertices = self._vertices.shape[1]
        self._face_vertices_update = True
        self._surface_normals_update = True
        self._vertex_normals_update = True

    @property
    def textures(self):
        return self._textures

    @textures.setter
    def textures(self, textures):
        # need check tensor
        self._textures = textures

    @property
    def face_vertices(self):
        if self._face_vertices_update:
            self._face_vertices = getFacesFromVertexIndices(self.vertices, self.faces)
            self._face_vertices_update = False
        return self._face_vertices
        # batch_size x #faces x 3 x 3 matrix where the vertex indices have been replaced
        # by the actual vertex positions in the NDC coordinates

    @property
    def surface_normals(self):
        if self._surface_normals_update:
            v10 = self.face_vertices[:, :, 0] - self.face_vertices[:, :, 1]
            v12 = self.face_vertices[:, :, 2] - self.face_vertices[:, :, 1]
            # v10, v12 are batch_size x #faces x 3
            self._surface_normals = F.normalize(torch.cross(v12, v10), p=2, dim=2, eps=1e-6)
            self._surface_normals_update = False
        # if a face consists of the vertices v0 v1 v2 (anticlockwise), then
        # this calculates the normal as norm((v2 - v1) x (v0 - v1))
        # the return value is a 3D matrix with batch_size x #faces x normal
        return self._surface_normals

    @property
    def vertex_normals(self):
        if self._vertex_normals_update:
            self._vertex_normals = getVertexNormals(self.vertices, self.faces)
            self._vertex_normals_update = False
        return self._vertex_normals

    @property
    def face_textures(self):
        return self.textures

    def fill_back_(self):
        if not self._fill_back:
            self.faces = torch.cat((self.faces, self.faces[:, :, [2, 1, 0]]), dim=1)
            self.textures = torch.cat((self.textures, self.textures), dim=1)
            self._fill_back = True

    def reset_(self):
        self.vertices = self._origin_vertices
        self.faces = self._origin_faces
        self.textures = self._origin_textures
        self._fill_back = False
    
    # classmethods are equivalent of static methods in C++
    @classmethod
    def from_obj(cls, filename_obj, normalization=False, load_texture=False, texture_res=1):
        '''
        Create a Mesh object from a .obj file
        '''
        if load_texture:
            vertices, faces, textures = loadObject(filename_obj, normalization=normalization, texture_res=texture_res, load_texture=True)
        else:
            vertices, faces = loadObject(filename_obj, normalization=normalization, texture_res=texture_res, load_texture=False)
            textures = None
        return cls(vertices, faces, textures, texture_res)