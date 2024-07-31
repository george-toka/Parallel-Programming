#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

const int N = 4096;// You can adjust the size as needed
const double epsilon = 1e-6;

template <typename T>
__device__ T sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

// Device functions to replace max and abs
__device__ double device_max(double a, double b) {
    return a > b ? a : b;
}

__device__ double device_abs(double a) {
    return a < 0 ? -a : a;
}

__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void transpose(double* A, double* AT) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        AT[j * N + i] = A[i * N + j];
    }
}
void transposeMatrix(double* A, double* AT) {

    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++)
			AT[j * N + i] = A[i * N + j];
    }
}
__global__ void svdIteration(double* U_t, double* V_t, int* I1, int* I2, double* C, int l, int r1, int r2, int N) {
    extern __shared__ double sharedMem[];

    int p = blockIdx.x * blockDim.x + threadIdx.x;
    double threadMaxC = 0.0;

    if (p < r1) {
        // Process I1
        int i = I1[p];
        int j = i + l;

        double alpha = 0, beta = 0, gamma = 0;
        for (int k = 0; k < N; k++) {
            double ui = U_t[i * N + k];
            double uj = U_t[j * N + k];
            alpha += ui * ui;
            beta += uj * uj;
            gamma += ui * uj;
        }

        double zeta = (beta - alpha) / (2.0 * gamma);
        double t = sgn(zeta) / (abs(zeta) + sqrt(1.0 + (zeta * zeta)));
        double c = 1.0 / (sqrt(1.0 + (t * t)));
        double s = c * t;

        for (int k = 0; k < N; k++) {
            double tempU = U_t[i * N + k];
            U_t[i * N + k] = c * tempU - s * U_t[j * N + k];
            U_t[j * N + k] = s * tempU + c * U_t[j * N + k];

            double tempV = V_t[i * N + k];
            V_t[i * N + k] = c * tempV - s * V_t[j * N + k];
            V_t[j * N + k] = s * tempV + c * V_t[j * N + k];
        }

        threadMaxC = max(threadMaxC, abs(gamma) / sqrt(alpha * beta));
    }

    if (p < r2) {
        // Process I2
        int i = I2[p];
        int j = i + l;

        double alpha = 0, beta = 0, gamma = 0;
        for (int k = 0; k < N; k++) {
            double ui = U_t[i * N + k];
            double uj = U_t[j * N + k];
            alpha += ui * ui;
            beta += uj * uj;
            gamma += ui * uj;
        }

        double zeta = (beta - alpha) / (2.0 * gamma);
        double t = sgn(zeta) / (abs(zeta) + sqrt(1.0 + (zeta * zeta)));
        double c = 1.0 / (sqrt(1.0 + (t * t)));
        double s = c * t;

        for (int k = 0; k < N; k++) {
            double tempU = U_t[i * N + k];
            U_t[i * N + k] = c * tempU - s * U_t[j * N + k];
            U_t[j * N + k] = s * tempU + c * U_t[j * N + k];

            double tempV = V_t[i * N + k];
            V_t[i * N + k] = c * tempV - s * V_t[j * N + k];
            V_t[j * N + k] = s * tempV + c * V_t[j * N + k];
        }

        threadMaxC = max(threadMaxC, abs(gamma) / sqrt(alpha * beta));
    }

    // Use shared memory to reduce max values within the block
    sharedMem[threadIdx.x] = threadMaxC;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedMem[threadIdx.x] = max(sharedMem[threadIdx.x], sharedMem[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (threadIdx.x == 0) {
        atomicMaxDouble(&C[blockIdx.x], sharedMem[0]);
    }
}





__global__ void matMul(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void fillMatrixWithRandomValues(double* A) {
    for (int i = 0; i < N * N; ++i) {
        A[i] = rand() % 99 - 49; // generates numbers from -49 to 49
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << msg << ": " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

void printMatrix(const double* A, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << A[i * cols + j] << " ";
        }
        cout << endl;
    }
}

int main() {
    double *U, *V, *S, *S_full, *U_t, *V_t, *A, *reconstructedA;
    int *I1, *I2;
    double *C;
    double it_end, it_st;
    double converge = 1.0;

    // Host pointers
    double *h_U, *h_V, *h_U_t, *h_V_t, *h_A, *h_S, *h_S_full, *h_reconstructedA;
    int *h_I1, *h_I2;
    double *h_C;

    // Allocate host memory
    h_U = (double*)malloc(N * N * sizeof(double));
    h_V = (double*)malloc(N * N * sizeof(double));
    h_U_t = (double*)malloc(N * N * sizeof(double));
    h_V_t = (double*)malloc(N * N * sizeof(double));
    h_A = (double*)malloc(N * N * sizeof(double));
    h_S = (double*)malloc(N * sizeof(double));
    h_S_full = (double*)malloc(N * N * sizeof(double));
    h_I1 = (int*)malloc(N * sizeof(int));
    h_I2 = (int*)malloc(N * sizeof(int));
    h_C = (double*)malloc(N * sizeof(double));
    h_reconstructedA = (double*)malloc(N * N * sizeof(double));

    // Allocate device memory
    checkCudaError(cudaMalloc(&U, N * N * sizeof(double)), "Failed to allocate U");
    checkCudaError(cudaMalloc(&V, N * N * sizeof(double)), "Failed to allocate V");
    checkCudaError(cudaMalloc(&U_t, N * N * sizeof(double)), "Failed to allocate U_t");
    checkCudaError(cudaMalloc(&V_t, N * N * sizeof(double)), "Failed to allocate V_t");
    checkCudaError(cudaMalloc(&A, N * N * sizeof(double)), "Failed to allocate A");
    checkCudaError(cudaMalloc(&S, N * sizeof(double)), "Failed to allocate S");
    checkCudaError(cudaMalloc(&S_full, N * N * sizeof(double)), "Failed to allocate S_full");
    checkCudaError(cudaMalloc(&I1, N * sizeof(int)), "Failed to allocate I1");
    checkCudaError(cudaMalloc(&I2, N * sizeof(int)), "Failed to allocate I2");
    checkCudaError(cudaMalloc(&C, N * sizeof(double)), "Failed to allocate C");
    checkCudaError(cudaMalloc(&reconstructedA, N * N * sizeof(double)), "Failed to allocate reconstructed A");

    fillMatrixWithRandomValues(h_A);
    cout << "Original matrix A:" << endl;
    printMatrix(h_A, 5, 5);
    cout << endl;

    // Copy A from host to device
    checkCudaError(cudaMemcpy(A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice), "Failed to copy A to device");

    double start = clock();

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transpose<<<numBlocks,dim3(32,32)>>>(A, U_t);
    double trans = clock();
    printf("For transpose it took %f seconds\n",(trans-start) / CLOCKS_PER_SEC);
 checkCudaError(cudaGetLastError(), "Failed to launch transpose kernel");
    checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize after transpose kernel");

    // Initialize V_t on host
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_V_t[i * N + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Copy V_t from host to device
    checkCudaError(cudaMemcpy(V_t, h_V_t, N * N * sizeof(double), cudaMemcpyHostToDevice), "Failed to copy V_t to device");
   double while_start = clock();
while (converge > epsilon) {
    converge = 0.0;

    for (int l = 1; l < N; l++) {
        int r1 = 0, r2 = 0;
        for (int i = 0; i + l < N; i++) {
            if (i % (2 * l) < l)
                h_I1[r1++] = i;
            else
                h_I2[r2++] = i;
        }

        // Copy I1 and I2 from host to device
        checkCudaError(cudaMemcpy(I1, h_I1, r1 * sizeof(int), cudaMemcpyHostToDevice), "Failed to copy I1 to device");
        checkCudaError(cudaMemcpy(I2, h_I2, r2 * sizeof(int), cudaMemcpyHostToDevice), "Failed to copy I2 to device");

        // Initialize C on host
        for (int z = 0; z < N; z++) h_C[z] = converge;

        // Copy C from host to device
        checkCudaError(cudaMemcpy(C, h_C, N * sizeof(double), cudaMemcpyHostToDevice), "Failed to copy C to device");

        it_st = clock();

        // Launch svdIteration kernel for I1 and I2 concurrently
        int blockSize = 4;
        int gridSize1 = (N + blockSize - 1) / blockSize;
        int sharedMemSize = blockSize * sizeof(double);  // Shared memory size

        svdIteration<<<gridSize1, blockSize, sharedMemSize>>>(U_t, V_t, I1, I2, C, l, r1, r2, N);

        checkCudaError(cudaGetLastError(), "Failed to launch svdIteration kernel");
        checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize after svdIteration kernel");

        it_end = clock();

        // Copy C from device to host
        checkCudaError(cudaMemcpy(h_C, C, N * sizeof(double), cudaMemcpyDeviceToHost), "Failed to copy C to host");

        for (int z = 0; z < N; z++) converge = max(converge, h_C[z]);
    }
}
double while_end = clock();
cout << "While loop took " << (while_end - while_start) / CLOCKS_PER_SEC << " seconds" << endl;

    // Copy U_t and V_t from device to host
    checkCudaError(cudaMemcpy(h_U_t, U_t, N * N * sizeof(double), cudaMemcpyDeviceToHost), "Failed to copy U_t to host");
    checkCudaError(cudaMemcpy(h_V_t, V_t, N * N * sizeof(double), cudaMemcpyDeviceToHost), "Failed to copy V_t to host");

    for (int i = 0; i < N; i++) {
        double t = 0;
        for (int j = 0; j < N; j++) {
            t += pow(h_U_t[i * N + j], 2);
        }
        t = sqrt(t);
        for (int j = 0; j < N; j++) {
            h_U_t[i * N + j] /= t;
            if (i == j) {
                h_S[i] = t;
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_U[i * N + j] = h_U_t[j * N + i];
            h_V[i * N + j] = h_V_t[j * N + i];
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_S_full[i * N + j] = (i == j) ? h_S[i] : 0.0;
        }
    }

    double end = clock();

    // Allocate US on device
    double* US;
    checkCudaError(cudaMalloc(&US, N * N * sizeof(double)), "Failed to allocate US");

    // Copy U and S_full from host to device
    checkCudaError(cudaMemcpy(U, h_U, N * N * sizeof(double), cudaMemcpyHostToDevice), "Failed to copy U to device");
    checkCudaError(cudaMemcpy(S_full, h_S_full, N * N * sizeof(double), cudaMemcpyHostToDevice), "Failed to copy S_full to device");

    matMul<<<numBlocks, threadsPerBlock>>>(U, S_full, US, N);
    checkCudaError(cudaGetLastError(), "Failed to launch matMul kernel for U * S");
    checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize after matMul kernel for U * S");

    double* V_t_transposed;
    checkCudaError(cudaMalloc(&V_t_transposed, N * N * sizeof(double)), "Failed to allocate V_t_transposed");

    // Transpose V on host
    transposeMatrix(h_V, h_V_t);

    // Copy V_t_transposed from host to device
    checkCudaError(cudaMemcpy(V_t_transposed, h_V_t, N * N * sizeof(double), cudaMemcpyHostToDevice), "Failed to copy V_t_transposed to device");

    matMul<<<numBlocks, threadsPerBlock>>>(US, V_t_transposed, reconstructedA, N);
    checkCudaError(cudaGetLastError(), "Failed to launch matMul kernel for US * V^T");
    checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize after matMul kernel for US * V^T");

    // Copy reconstructedA from device to host
    checkCudaError(cudaMemcpy(h_reconstructedA, reconstructedA, N * N * sizeof(double), cudaMemcpyDeviceToHost), "Failed to copy reconstructedA to host");

    cout << "SVD Finished after: " << (end - start) / CLOCKS_PER_SEC << " seconds" << endl << endl;
    cout << "Iteration after: " << (it_end - it_st) / CLOCKS_PER_SEC << " seconds" << endl << endl;

    cout << "\nReconstructed matrix A (U * S * V^T):" << endl;
    printMatrix(h_reconstructedA, 5, 5);

    // Free device memory
    cudaFree(U);
    cudaFree(V);
    cudaFree(U_t);
    cudaFree(V_t);
    cudaFree(A);
    cudaFree(S);
    cudaFree(S_full);
    cudaFree(I1);
    cudaFree(I2);
    cudaFree(C);
    cudaFree(reconstructedA);
    cudaFree(US);
    cudaFree(V_t_transposed);

    // Free host memory
    free(h_U);
    free(h_V);
    free(h_U_t);
    free(h_V_t);
    free(h_A);
    free(h_S);
    free(h_S_full);
    free(h_I1);
    free(h_I2);
    free(h_C);
    free(h_reconstructedA);

    return 0;
}



