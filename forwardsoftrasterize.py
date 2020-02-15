import torch, math, tqdm
import numpy as np

def getBarycentricCoords(x, y, face_attrib):
    return np.matmul(face_attrib.numpy()[:9].reshape((3, 3)), np.array([x, y, 1.]).reshape((3,)))
    
def isInBoundingBox(x, y, face, threshold):
    return ((x > max(max(face[0, 0].item(), face[1, 0].item()), face[2, 0].item()) + threshold) or
            (x < min(min(face[0, 0].item(), face[1, 0].item()), face[2, 0].item()) - threshold) or
            (y > max(max(face[0, 1].item(), face[1, 1].item()), face[2, 1].item()) + threshold) or
            (y < min(min(face[0, 1].item(), face[1, 1].item()), face[2, 1].item()) - threshold))

def isFaceFrontside(face):
    return (face[2, 1].item() - face[0, 1].item()) * (face[1, 0].item() - face[0, 0].item()) < \
        (face[1, 1].item() - face[0, 1].item()) * (face[2, 0].item() - face[0, 0].item())

def clipBarycentricCoords(w):
    w_clip = np.clip(w, 0., 1.)
    # clips each coordinate in [0, 1]
    w_sum = max(w[0] + w[1] + w[2], 1e-5)
    w_clip /= w_sum
    # above step is normalization of the new coordinates
    # since the new coordinates are all positive, the new point
    # will be somewhere inside or on the traingle
    return w_clip

def distanceFromTriangle(w, face, face_attrib, xp, yp):
    face_sym = face_attrib[9:18]
    face_obt = face_attrib[18:]

    if (w.max() < 1.0 and w.min() > 0.0):
        # pixel inside the triangle, w[0] + w[1] + w[2] = 0
        # now calculate the distance of point x given by barycentric coordinates w
        # from each of the sides of the triangle face and store the minimum distance
        min_dist, min_dist_x, min_dist_y = 100000000., 0., 0.

        for k in range(3):
            v0, v1 = k, (k + 1) % 3
            a0 = face_sym.numpy().reshape(3, 3)[v0] - face_sym.numpy().reshape(3, 3)[v1]
            tmp1 = (np.dot(w, a0) - a0[v1]) / (a0[v0] - a0[v1])
            tmp2 = [tmp1, 1-tmp1, 0]
            t0 = np.array(tmp2[-v0:] + tmp2[:-v0]) - w

            # calculate distance
            dist_x = np.dot(t0, face[:, 0].numpy())
            dist_y = np.dot(t0, face[:, 1].numpy())
            dist = dist_x * dist_x + dist_y * dist_y

            if (dist < min_dist):
                min_dist, min_dist_x, min_dist_y = dist, dist_x, dist_y
                t = t0

        dist_x = min_dist_x
        dist_y = min_dist_y
        sign = 1
    else:
        # pixel is outside the triangle
        v0 = -1
        pix_pos = np.array([xp, yp])
        if (w[1] <= 0 and w[2] <= 0):
            v0 = 0
            if (face_obt[0] == 1 and np.dot(pix_pos - face[0, :-1], face[2:, -1] - face[0, :-1]) > 0):
                v0 = 2
        elif (w[2] <= 0 and w[0] <= 0):
            v0 = 1
            if (face_obt[1] == 1 and np.dot(pix_pos - face[1, :-1], face[0:, -1] - face[1, :-1]) > 0):
                v0 = 0
        elif (w[0] <= 0 and w[1] <= 0):
            v0 = 2
            if (face_obt[2] == 1 and np.dot(pix_pos - face[2, :-1], face[1:, -1] - face[2, :-1]) > 0):
                v0 = 1
        elif (w[0] <= 0):
            v0 = 1
        elif (w[1] <= 0):
            v0 = 2
        elif (w[2] <= 0):
            v0 = 0

        v1 = (v0 + 1) % 3
        a0 = face_sym.numpy().reshape(3, 3)[v0] - face_sym.numpy().reshape(3, 3)[v1]

        tmp1 = (np.dot(w, a0) - a0[v1]) / (a0[v0] - a0[v1])
        tmp2 = [tmp1, 1 - tmp1, 0]
        t = np.array(tmp2[-v0:] + tmp2[:-v0])
        # clamp to [0, 1]
        np.clip(t, 0., 1.)
        t -= w

        # calculate distance
        dist_x = np.dot(t, face[:, 0].numpy())
        dist_y = np.dot(t, face[:, 1].numpy())
        sign = -1
    return t, sign, dist_x, dist_y

def forwardSampleTexture(texture, w, R, k):
    # sample surface color with resolution as R
    w_x = int(w[0] * R - 1e-6)
    w_y = int(w[1] * R - 1e-6)
    # w_x, w_y is used for indexing into the texture matrix (texture_res x texture_res)
    # which contains the colors sampled for discrete barycentric coordinates
    if ((w[0] + w[1]) * R - w_x - w_y <= 1):
        # frac(w[0] * R) + frac(w[1] * R) <= 1
        texture_k = texture[w_y * R + w_x, k].item()
    else:
        texture_k = texture[(R - 1 - w_y) * R + R - 1 - w_x, k].item()
    return texture_k

def forwardComputeFaceAttributes(faces, faces_attrib, image_size):

    batch_size = faces.shape[0]
    nf = faces.shape[1]
    # nf is number_of_faces while fn is face_number
    for bn in range(batch_size):
        for fn in range(nf):
            face = faces[bn, fn]
            face_inv = faces_attrib[bn, fn, :9]
            face_sym = faces_attrib[bn, fn, 9:18]
            face_obt = faces_attrib[bn, fn, 18:]

            p = np.ones((3, 3))
            p[:, :-1] = face[:, :-1]
            p = p.T
            # p is storing the x,y coordinates of the 3 vertices of a face

            # compute face_inv
            # the image plane is at z = 1
            # so, to compute the barycentric coordinates of a point (x,y,1), we have the matrix equation
            # [v0.x v1.x v2.x] [w0]   [x]
            # [v0.y v1.y v2.y] [w1] = [y]
            # [1    1    1   ] [w2]   [1]
            # So, to get (w0, w1, w2), we calculate the inverse of the first matrix
            #         [v0.x v1.x v2.x]
            # p = M = [v0.y v1.y v2.y]
            #         [1    1    1   ]
            # calculating inverse(M)
            face_inv[:] = torch.from_numpy(np.linalg.inv(p).flatten())
            face_sym[:] = torch.from_numpy(np.matmul(p.T, p)).flatten()
            # F * F.T where,
            #     [v0.x v0.y 1]         [v0.x v1.x v2.x]
            # F = [v1.x v1.y 1]   F.T = [v0.y v1.y v2.y]
            #     [v2.x v2.y 1]         [1    1    1   ]
            # this is basically taking the dot product of the position vectors among themselves
            # this will be needed when calculating the distance of a point from the sides of the triangle
            for k in range(3):
                k0, k1, k2 = k, (k + 1) % 3, (k + 2) % 3
                # check if this arc formed by vk0, vk1, vk2 is obtuse, by checking dot product
                if np.dot(p[:-1, k1] - p[:-1, k0], p[:-1, k2] - p[:-1, k0]) < 0:
                    break


def forwardSoftRasterize(faces, textures, faces_attrib, aggrs_info, pixel_colors,
                            image_size, near, far, eps, sigma, dist_eps, gamma, double_side):

    # geometry information is passed through faces, textures
    # the forward pass will now compute faces_attrib, aggrs_info, pixel_colors
    # precompute some attributes for faces which will be stored in faces_attrib
    # these attributes are needed for future computations and backwards
    batch_size = faces.shape[0]
    nf = faces.shape[1]
    texture_res = int(math.sqrt(textures.shape[2]))
    forwardComputeFaceAttributes(faces, faces_attrib, image_size)
    print("Performing Soft Rasterization...")
    loop = tqdm.tqdm(list(range(0, image_size * image_size)))
    for bn in range(batch_size):
        # print("batch no: " + str(bn), end=' ')
        for pn in loop:
            # print(str(pn), end=' ')
            # pn is pixel_number
            yi = int(image_size - 1 - (pn / image_size))
            # row_number of the pixel from bottom
            xi = int(pn % image_size)
            # column_number of pixel
            # (yi, xi) = (0,0) pixel is at the bottom left of the image
            yp = (2. * yi + 1. - image_size) / image_size
            xp = (2. * xi + 1. - image_size) / image_size
            # range of (yp, xp) is [1/is - 1, 1 - 1/is]
            # maps the pixel space [0, is-1][0, is-1] -> (-1, 1)(-1, 1)
            # basically repositining the image pixels, so that the centre of image is (0,0)
            threshold = dist_eps * sigma
            # Initialize pixel color with white color
            softmax_sum = math.exp(eps / gamma)
            # the const denominator term
            softmax_max = eps
            pixel_color = (pixel_colors[bn, :, yi, xi].numpy() * softmax_sum)
            pixel_color[3] = 1.

            for fn in range(nf):
                # computation for fn'th face (or the fn'th triangle)
                face = faces[bn, fn]
                texture = textures[bn, fn]
                face_attrib = faces_attrib[bn, fn]

                if (isInBoundingBox(xp, yp, face, math.sqrt(threshold))):
                    continue
                # checks if pixel is sqrt(threshold) distance away from the bounding box of triangle fn
                # triangle not too far away from pixel, will have a significant influence D^i

                # dis, dis_x, dis_y, t[3], w[3], w_clip[3], sign, soft_fragment
                # make t, w, w_clip tensors so that you can vectorize future code
                # soft_fragment is basically the D corresponding to this pixel and triangle (i.e. D_fn^i)

                # compute barycentric coordinate w
                w = getBarycentricCoords(xp, yp, face_attrib)

                # compute probability map based on distance functions euclidean distance
                t, sign, dis_x, dis_y = distanceFromTriangle(w, face, face_attrib, xp, yp)
                dis = dis_x * dis_x + dis_y * dis_y
                if (sign < 0 and dis >= threshold):
                    # triangle far away from the pixel, ignore
                    continue
                soft_fragment = 1. / (1. + math.exp(-sign * dis / sigma))

                # aggragate for alpha channel
                pixel_color[3] *= (1. - soft_fragment)

                w_clip = clipBarycentricCoords(w)
                # clips the coordinates in the range [0, 1]
                zp = 1. / np.sum(w_clip / face[:, 2].numpy())
                # zp is the projection of the pixel on the actual triangle in 3D NDC
                if (zp < near or zp > far):
                    # triangle out of screen, hence pass
                    continue

                # aggregate for rgb channels
                # D * Softmax (Z)
                if (isFaceFrontside(face) or double_side):
                    zp_norm =  (far - zp) / (far - near)
                    # basically inverting zp which means that points on triangles lying closer to the screen
                    # will have greater weights in the aggregator function
                    exp_delta_zp = 1.
                    if (zp_norm > softmax_max):
                        exp_delta_zp = math.exp((softmax_max - zp_norm) / gamma)
                        softmax_max = zp_norm
                    exp_z = math.exp((zp_norm - softmax_max) / gamma)
                    softmax_sum = exp_delta_zp * softmax_sum + exp_z * soft_fragment
                    for k in range(3):
                        color_k = forwardSampleTexture(texture, w_clip, texture_res, k)
                        # soft_fragment
                        pixel_color[k] = exp_delta_zp * pixel_color[k] + exp_z * soft_fragment * color_k

            # finalize aggregation
            pixel_colors[bn, 3, yi, xi] =  1. - pixel_color[3]

            # normalize colors
            for k in range(3):
                pixel_colors[bn, k, yi, xi] = pixel_color[k] / softmax_sum
            aggrs_info[bn, 0, yi, xi] = softmax_sum
            aggrs_info[bn, 1, yi, xi] = softmax_max
    
    return faces_attrib, aggrs_info, pixel_colors
