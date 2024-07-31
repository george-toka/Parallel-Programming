# Parallel-Programming
Singular Value Decompositon (SVD) using OpenMP &amp; CUDA 

At first the task was to implement the SVD in OpenMP, so there was experimentation <br>
1) With the available methods to calculate the needed matrices to achieve SVD <br>
2) The parallelisation that could be achieved with each method. <br>

First, QR algorithm was implemented, but the signs of some eigenvectors' values that were needed <br>
for the decomposition, were opposite of their respective signs. So the final method used was <br>
Jacobi-Rotation to calculate the eigenvectors and eigenvalues of the decomposed matrix. <br>
<br>
Finally, having the serial and parallel code in OpenMP there was the stage only the stage of <br>
converting the code to CUDA
