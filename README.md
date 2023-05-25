# Visualization of thread block scheduling in CUDA

This repository provides a simple application for the visualization of the order
in which thread blocks are scheduled onto multiprocessors on CUDA enabled GPUs.
The results for GeForce GTX 1070 are in the following gifs.

![1D grid](figures/1D_grid.gif "1D grid")
![2D grid](figures/2D_grid.gif "2D grid")

The first one was generated for a 1D grid of size 4096. For better presentation
the blocks in the gif are displayed as a 2D array. Each row corresponds to
a single chunk of the 1D grid and consecutive rows contain consecutive chunks.

The second gif was generated for a 2D grid of size 64 x 64. Each block in
the gif has the same coordinates as in the grid.

The timings were collected using a sample kernel for the sparse matrix-vector
product with the matrix in the ELL format.
