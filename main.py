import matplotlib.pyplot as plt
import os, imageio, argparse
import numpy as np
from renderer import SoftRenderer
from mesh import Mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, 
        default='meshes/icosphere/icosphere.obj')
    parser.add_argument('-o', '--output_dir', type=str, 
        default='rendered/')
    args = parser.parse_args()

    # other settings
    camera_distance = 2.732
    elevation = 30
    azimuth = 0

    # load from Wavefront .obj file
    mesh = Mesh.from_obj(args.input_file, load_texture=False)

    os.makedirs(args.output_dir, exist_ok=True)

    # create renderer with SoftRas
    renderer = SoftRenderer()
    renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
    image = renderer.render_mesh(mesh).numpy()[0].transpose((1, 2, 0))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'render.png'), mode='i')
    writer.append_data((255*image).astype(np.uint8))
    writer.close()

if __name__ == '__main__':
    main()
