# Soft_Renderer
This is my implementation of the differentiable renderer described in the paper below. The rasterization step is performed on the CPU.

Link to the paper: https://arxiv.org/pdf/1904.01786.pdf

The renderer takes a Wafefront file (.obj) as input and gives a render in png/jpg format which can be changed in the main.py file. To build the code with a sample icosphere.obj inside the meshes/icosphere directory, simply call the main.py script the input file specified with the -i flag,

```
$ python3 main.py -i meshes/icosphere/icosphere.obj
```

The rendered output will be written in the "rendered" directory.
