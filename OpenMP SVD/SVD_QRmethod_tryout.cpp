#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>

#define N 4

using namespace std;

void printMatrix(double** A) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << A[i][j] << " ";
        }
        cout << endl;
    }
}

void printVector(double* v) {
    for (int i = 0; i < N; ++i) {
        cout << v[i] << " ";
    }
    cout << "\n" << endl;
}
 
void matrixMultiply(double** A, double** B, double** result) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result[i][j] = 0;
            for (size_t k = 0; k < N; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void transpose(double** A, double** AT) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            AT[j][i] = A[i][j];
        }
    }
}

void qrDecomposition(double** A, double** Q, double** R) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            Q[i][j] = A[i][j];
            R[i][j] = 0;
        }
    }

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < i; ++j) {
            double dot = 0;
            for (size_t k = 0; k < N; ++k) {
                dot += Q[k][j] * A[k][i];
            }
            for (size_t k = 0; k < N; ++k) {
                Q[k][i] -= dot * Q[k][j];
            }
            R[j][i] = dot;
        }

        double norm = 0;
        for (size_t k = 0; k < N; ++k) {
            norm += Q[k][i] * Q[k][i];
        }
        norm = sqrt(norm);

        for (size_t k = 0; k < N; ++k) {
            Q[k][i] /= norm;
        }
        R[i][i] = norm;
    }
}

void qrAlgorithm(double** A, double* eigenvalues, double** eigenvectors) {
    double** Q = (double**)malloc(N * sizeof(double*));
    double** R = (double**)malloc(N * sizeof(double*));
    double** A_copy = (double**)malloc(N * sizeof(double*));
    double** temp = (double**)malloc(N * sizeof(double*));

    for (size_t i = 0; i < N; ++i) {
        Q[i] = (double*)malloc(N * sizeof(double));
        R[i] = (double*)malloc(N * sizeof(double));
        A_copy[i] = (double*)malloc(N * sizeof(double));
        temp[i] = (double*)malloc(N * sizeof(double));
    }

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A_copy[i][j] = A[i][j];
            eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    const int maxIterations = 1000;
    const double tolerance = 1e-10;

    for (int iter = 0; iter < maxIterations; ++iter) {
        qrDecomposition(A_copy, Q, R);
        double** A_new = (double**)malloc(N * sizeof(double*));
        for (size_t i = 0; i < N; ++i) {
            A_new[i] = (double*)malloc(N * sizeof(double));
        }
        matrixMultiply(R, Q, A_new);

        matrixMultiply(eigenvectors, Q, temp);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                eigenvectors[i][j] = temp[i][j];
            }
        }

        bool converged = true;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (i != j && fabs(A_new[i][j]) > tolerance) {
                    converged = false;
                    break;
                }
            }
            if (!converged) break;
        }

        if (converged) {
            for (size_t i = 0; i < N; ++i) {
                free(A_new[i]);
            }
            free(A_new);
            break;
        }

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                A_copy[i][j] = A_new[i][j];
            }
        }

        for (size_t i = 0; i < N; ++i) {
            free(A_new[i]);
        }
        free(A_new);
    }

    for (size_t i = 0; i < N; ++i) {
        eigenvalues[i] = A_copy[i][i];
    }

    for (size_t i = 0; i < N; ++i) {
        free(Q[i]);
        free(R[i]);
        free(A_copy[i]);
        free(temp[i]);
    }
    free(Q);
    free(R);
    free(A_copy);
    free(temp);
}

void calculateEigenvectors(double** A, double** eigenvectors) {
    double** Q = (double**)malloc(N * sizeof(double*));
    double** R = (double**)malloc(N * sizeof(double*));
    double** A_copy = (double**)malloc(N * sizeof(double*));
    double** temp = (double**)malloc(N * sizeof(double*));

    for (size_t i = 0; i < N; ++i) {
        Q[i] = (double*)malloc(N * sizeof(double));
        R[i] = (double*)malloc(N * sizeof(double));
        A_copy[i] = (double*)malloc(N * sizeof(double));
        temp[i] = (double*)malloc(N * sizeof(double));
    }

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A_copy[i][j] = A[i][j];
        }
    }

    const int maxIterations = 1000;
    const double tolerance = 1e-10;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int iter = 0; iter < maxIterations; ++iter) {
        qrDecomposition(A_copy, Q, R);
        double** A_new = (double**)malloc(N * sizeof(double*));
        for (size_t i = 0; i < N; ++i) {
            A_new[i] = (double*)malloc(N * sizeof(double));
        }
        matrixMultiply(R, Q, A_new);

        matrixMultiply(eigenvectors, Q, temp);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                eigenvectors[i][j] = temp[i][j];
            }
        }

        bool converged = true;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (i != j && fabs(A_new[i][j]) > tolerance) {
                    converged = false;
                    break;
                }
            }
            if (!converged) break;
        }

        if (converged) {
            for (size_t i = 0; i < N; ++i) {
                free(A_new[i]);
            }
            free(A_new);
            break;
        }

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                A_copy[i][j] = A_new[i][j];
            }
        }

        for (size_t i = 0; i < N; ++i) {
            free(A_new[i]);
        }
        free(A_new);
    }

    for (size_t i = 0; i < N; ++i) {
        free(Q[i]);
        free(R[i]);
        free(A_copy[i]);
        free(temp[i]);
    }
    free(Q);
    free(R);
    free(A_copy);
    free(temp);
}

void svd(double** A, double** U, double* S, double** V) {
    double** AT = (double**)malloc(N * sizeof(double*));
    double** ATA = (double**)malloc(N * sizeof(double*));
    double** AAT = (double**)malloc(N * sizeof(double*));

    for (size_t i = 0; i < N; ++i) {
        AT[i] = (double*)malloc(N * sizeof(double));
        ATA[i] = (double*)malloc(N * sizeof(double));
        AAT[i] = (double*)malloc(N * sizeof(double));
    }
    
    transpose(A, AT);
    matrixMultiply(AT, A, ATA);
    matrixMultiply(A, AT, AAT);
    
    double eigenvalues_ATA[N];
    qrAlgorithm(ATA, eigenvalues_ATA, V);
    
    for (int i = 0; i < N; ++i) {
        S[i] = sqrt(fabs(eigenvalues_ATA[i]));
    }

    calculateEigenvectors(AAT, U);

    for (size_t i = 0; i < N; ++i) {
        free(AT[i]);
        free(ATA[i]);
        free(AAT[i]);
    }
    free(AT);
    free(ATA);
    free(AAT);
}

void fillMatrixWithRandomValues(double** A) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = rand() % 99 - 49; // generates numbers from -49 to 49
        }
    }
}


int main() {
    double** A = (double**)malloc(N * sizeof(double*));
    for (size_t i = 0; i < N; ++i) {
        A[i] = (double*)malloc(N * sizeof(double));
    }
    
    A[0][0] = 10; A[0][1] = 3;  A[0][2] = 7;  A[0][3] = 1;
    A[1][0] = 4;  A[1][1] = 9;  A[1][2] = -6; A[1][3] = 15;
    A[2][0] = 47; A[2][1] = -13; A[2][2] = -2; A[2][3] = 1;
    A[3][0] = -1; A[3][1] = 35; A[3][2] = 4;  A[3][3] = 6;

//    // Open the file in read mode
//    ifstream inFile("A_mat.data");
//    if (!inFile.is_open()) {
//        cerr << "Error opening file for reading" << endl;
//        return 1;
//    }
//
//    // Read the matrix from the file
//    for (int i = 0; i < N; ++i) {
//        for (int j = 0; j < N; ++j) {
//            inFile >> A[i][j];
//        }
//    }
//
//    // Close the file
//    inFile.close();
	
//	//random A matrix using srand
//	// Initialize random seed
//    srand(static_cast<unsigned int>(time(0)));
//    // Fill the matrix with random values
//    fillMatrixWithRandomValues(A);

    double** U = (double**)malloc(N * sizeof(double*));
    double* S = (double*)malloc(N * sizeof(double));
    double** V = (double**)malloc(N * sizeof(double*));

    for (size_t i = 0; i < N; ++i) {
        U[i] = (double*)malloc(N * sizeof(double));
        V[i] = (double*)malloc(N * sizeof(double));
    }

    svd(A, U, S, V);

    cout << "Original Matrix A:" << endl;
    printMatrix(A);

    cout << "\nMatrix U:" << endl;
    printMatrix(U);

    cout << "\nSingular Values S (as a vector):" << endl;
    printVector(S);

    cout << "\nSingular Values S (as a matrix):" << endl;
    double** S_matrix = (double**)malloc(N * sizeof(double*));
    for (int i = 0; i < N; ++i) {
        S_matrix[i] = (double*)malloc(N * sizeof(double));
        for (int j = 0; j < N; ++j) {
            S_matrix[i][j] = 0;
        }
        S_matrix[i][i] = S[i];
    }
    printMatrix(S_matrix);

    cout << "\nMatrix V:" << endl;
    printMatrix(V);

    double** temp = (double**)malloc(N * sizeof(double*));
    double** reconstructed_A = (double**)malloc(N * sizeof(double*));
    for (size_t i = 0; i < N; ++i) {
        temp[i] = (double*)malloc(N * sizeof(double));
        reconstructed_A[i] = (double*)malloc(N * sizeof(double));
    }
//    matrixMultiply(U, S_matrix, temp);
//    transpose(V, V);
//    matrixMultiply(temp, V, reconstructed_A);

//    cout << "\nReconstructed Matrix A:" << endl;
//    printMatrix(reconstructed_A);
    
//    //export to check with matlab
//    // Open a file in write mode
//    ofstream outFile("S_mat.data");
//    ofstream outA("A_mat.data");
//    if (!outFile || !outA) {
//        cerr << "Error opening file for writing" << endl;
//        return 1;
//    }
//
//    // Get the number of rows and columns
////    int rows = sizeof(S_matrix) / sizeof(S_matrix[0]);
////    int cols = sizeof(S_matrix[0]) / sizeof(S_matrix[0][0]);
//	int rows = N , cols = N;
//
//    // Write the matrix to the file
//    for (int i = 0; i < rows; ++i) {
//        for (int j = 0; j < cols; ++j) {
//            outFile << S_matrix[i][j] << " ";
//            outA << A[i][j] << " ";
//        }
//        outFile << endl;  // Newline at the end of each row
//    	outA << endl;
//	}
//
//    // Close the file
//    outFile.close();
//    outA.close();

//    cout << "Matrix successfully written to matrix.data" << endl;

    for (size_t i = 0; i < N; ++i) {
        free(A[i]);
        free(U[i]);
        free(V[i]);
        free(S_matrix[i]);
        free(temp[i]);
        free(reconstructed_A[i]);
    }
    free(A);
    free(U);
    free(V);
    free(S_matrix);
    free(S);
    free(temp);
    free(reconstructed_A);

    return 0;
}


