# Parallel-Programming
Singular Value Decompositon (SVD) using OpenMP &amp; CUDA 

At first the task was to implement the SVD in OpenMP, so there was experimentation <br>
1) With the available methods to calculate the needed matrices to achieve SVD <br>
2) The parallelisation that could be achieved with each method. <br>

First, QR algorithm was implemented, but the signs of some eigenvectors' values that were needed <br>
for the decomposition, were opposite of their respective signs. Nevertheless it is kept in the OpenMP folder <br>
for the sake of time it took to write it. <br>
So finally, the method used was Jacobi-Rotation to calculate the eigenvectors and eigenvalues of the decomposed matrix. <br>
<br>
Having the serial and parallel code in OpenMP there was only the stage of <br>
converting the code to CUDA
