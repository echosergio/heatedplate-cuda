# Steady state heat equation calculation with CUDA

This code solves the steady state heat equation on a rectangular region. The sequential version of this program needs approximately
18/epsilon iterations to complete.

The physical region, and the boundary conditions, are suggested
by this diagram;

```
               W = 0
         +------------------+
         |                  |
W = 100  |                  | W = 100
         |                  |
         +------------------+
               W = 100
```

The region is covered with a grid of M by N nodes, and an N by N
array W is used to record the temperature.  The correspondence between
array indices and locations in the region is suggested by giving the
indices of the four corners:

```
              I = 0
      [0][0]-------------[0][N-1]
         |                  |
  J = 0  |                  |  J = N-1
         |                  |
    [M-1][0]-----------[M-1][N-1]
              I = M-1
```
The steady state solution to the discrete heat equation satisfies the
following condition at an interior grid point:

`W[Central] = (1/4) * ( W[North] + W[South] + W[East] + W[West] )`

where "Central" is the index of the grid point, "North" is the index
of its immediate neighbor to the "north", and so on.

Given an approximate solution of the steady state heat equation, a
"better" solution is given by replacing each interior point by the
average of its 4 neighbors - in other words, by using the condition
as an ASSIGNMENT statement:

`W[Central]  <=  (1/4) * ( W[North] + W[South] + W[East] + W[West] )`

If this process is repeated often enough, the difference between successive
estimates of the solution will go to zero.

This program carries out such an iteration, using a tolerance specified by
the user, and writes the final estimate of the solution to a file that can
be used for graphic processing.

## CUDA solution

The solution is limited to a 2D matrix with max dimension size of 512

gridDim: Dimensions of the grid  
blockIdx: The location of a block within the grid  
blockDim: Dimensions of the block  
threadIdx: Location of a thread within it's own block  

Blocks in a grid: gridDim.x * gridDim.y  
Threads in a block: blockDim.x * blockDim.y * blockDim.z

```
dim3 dimGrid(16, 16);  // 256 blocks
dim3 dimBlock(32, 32); // 1024 threads
```

The different parts than can be parallelized are divided into the following kernels:

```
copy_grid<<<dimGrid, dimBlock>>>  
calculate_solution<<<dimGrid, dimBlock>>>  
epsilon_reduction<<<dimGrid, dimBlock>>>  
```

## Grid results

<table style="width:100%;">
  <tr>
    <th style="text-align: center;">Epsilon</th>
    <th style="text-align: center;">Image bmp</th>
  </tr>
  <tr>
    <th style="text-align: center;">0.1</td>
    <th style="text-align: center;"><img src="https://raw.githubusercontent.com/sergiovhe/heatedplate-cuda/master/img/0.1.bmp" alt="0.1.bmp" style="width: 120px;"/></td>
  </tr>
  <tr>
    <th style="text-align: center;">0.01</td>
    <th style="text-align: center;"><img src="https://raw.githubusercontent.com/sergiovhe/heatedplate-cuda/master/img/0.01.bmp" alt="0.01.bmp" style="width: 120px;"/></td>
  </tr>
  <tr>
    <th style="text-align: center;">0.001</td>
    <th style="text-align: center;"><img src="https://raw.githubusercontent.com/sergiovhe/heatedplate-cuda/master/img/0.001.bmp" alt="0.001.bmp" style="width: 120px;"/></td>
  </tr>
  <tr>
    <th style="text-align: center;">0.0001</td>
    <th style="text-align: center;"><img src="https://raw.githubusercontent.com/sergiovhe/heatedplate-cuda/master/img/0.0001.bmp" alt="0.0001.bmp" style="width: 120px;"/></td>
  </tr>
  <tr>
    <th style="text-align: center;">0.00001</td>
    <th style="text-align: center;"><img src="https://raw.githubusercontent.com/sergiovhe/heatedplate-cuda/master/img/0.00001.bmp" alt="0.00001.bmp" style="width: 120px;"/></td>
  </tr>
</table>