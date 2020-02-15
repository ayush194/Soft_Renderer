import numpy as np
import math, tqdm

def loadTextures(image, faces, textures, is_update):

    print("Loading Textures...")
    loop = tqdm.tqdm(list(range(0, textures.shape[0] * textures.shape[1])))
    for i in loop:
        R = int(math.sqrt(textures.shape[1]))
        # R is the texture resolution (4 by default)
        fn = int(i / (R * R))
        # fn is the face number
        w_y = int((i % (R * R)) / R)
        # w_y is the row number of the R x R texture for face fn
        w_x = i % R
        # w_x is the column number of the R x R texture for face fn

        if (is_update[fn] == 0):
            continue
        
        # compute barycentric coordinates
        w0, w1, w2 = 0, 0, 0
        if (w_x + w_y < R):
            w0 = (w_x + 1. / 3.) / R
            w1 = (w_y + 1. / 3.) / R
            w2 = 1. - w0 - w1
        else:
            w0 = ((R - 1. - w_x) + 2. / 3.) / R
            w1 = ((R - 1. - w_y) + 2. / 3.) / R
            w2 = 1. - w0 - w1
        # basically this associates each cell of an R x R texture
        # with a barycentric coordinate of the traingle

        face = faces[fn]
        texture = textures[fn, i % (R*R)]

        pos_x = (face[0, 0] * w0 + face[1, 0] * w1 + face[2, 0] * w2) * (image.shape[1] - 1)
        pos_y = (face[0, 1] * w0 + face[1, 1] * w1 + face[2, 1] * w2) * (image.shape[0] - 1)

        # bilinear sampling
        weight_x1 = pos_x - int(pos_x)
        weight_x0 = 1 - weight_x1
        weight_y1 = pos_y - int(pos_y)
        weight_y0 = 1 - weight_y1
        for k in range(3):
            c = 0
            c += image[int(pos_y), int(pos_x), k] * (weight_x0 * weight_y0)
            c += image[int(pos_y + 1), int(pos_x), k] * (weight_x0 * weight_y1)
            c += image[int(pos_y), int(pos_x + 1), k] * (weight_x1 * weight_y0)
            c += image[int(pos_y + 1), int(pos_x + 1), k] * (weight_x1 * weight_y1)
            texture[k] = c
            
    return textures