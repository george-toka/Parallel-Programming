//standard matrix creation
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <cmath>

#define N 4
using namespace std;
void fillMatrixWithRandomValues(double** A) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = rand() % 99 - 49; // generates numbers from -49 to 49
        }
    }
}

void printMatrix(double** A) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << A[i][j] << " ";
        }
        cout << endl;
    }
}

int main(){
	double **S_matrix;
	S_matrix = new double*[N];
	for(int i=0; i<N; i++){
		S_matrix[i] = new double [N]; 
	}
	
	    // Open the file in read mode
    ifstream inFile("A_4.data");
    if (!inFile.is_open()) {
        cerr << "Error opening file for reading" << endl;
        return 1;
    }

    // Read the matrix from the file
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            inFile >> S_matrix[i][j];
        }
    }

    // Close the file
    inFile.close();
	
	printMatrix(S_matrix);
	
	fillMatrixWithRandomValues(S_matrix);	
	  //export to check with matlab
     // Open a file in write mode
	    ofstream outFile("A_4_new.data");
	    
	    if (!outFile) {
	        cerr << "Error opening file for writing" << endl;
	        return 1;
	    }
	
	    // Get the number of rows and columns
//		int rows = sizeof(S_matrix) / sizeof(S_matrix[0]);
//	    int cols = sizeof(S_matrix[0]) / sizeof(S_matrix[0][0]);
		int rows = N , cols = N;
	
	    // Write the matrix to the file
	    for (int i = 0; i < rows; ++i) {
	        for (int j = 0; j < cols; ++j) {
	            outFile << S_matrix[i][j] << " ";
	            
	        }
	        outFile << endl;  // Newline at the end of each row
	    	
		}
	
    // Close the file
	    outFile.close();
		

    cout << "Matrix successfully written to matrix.data" << endl;
	
	for(int i=0; i<N; i++){
		delete []S_matrix[i]; 
	}
	delete [] S_matrix;
	return 0;
}
