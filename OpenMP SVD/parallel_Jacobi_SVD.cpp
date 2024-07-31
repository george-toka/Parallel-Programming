#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <ctime>
#include <sys/time.h>
#include <omp.h>

#define epsilon 1.e-8
#define N 256
#define M 256
#define num 4

using namespace std;



template <typename T> double sgn(T val)
{
    return (val > T(0)) - (val < T(0));
}

void transpose(double** A, double** AT) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            AT[j][i] = A[i][j];
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

void fillMatrixWithRandomValues(double** A) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = rand() % 99 - 49; // generates numbers from -49 to 49
        }
    }
}

void printVec(double *I){
	cout<<endl;
	for(int i=0; i<N; ++i)
		cout<<I[i]<<" ";
	
	cout<<endl;
}

int readMatrixFromMemory(double** A, char *filename){
	
	// Open the file in read mode
    ifstream inFile(filename);
    if (!inFile.is_open()) {
        cerr << "Error opening file for reading" << endl;
        return 1;
    }

    // Read the matrix from the file
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            inFile >> A[i][j];
        }
    }

    // Close the file
    inFile.close();
}


int main(){
	double **U,**V, *S,**U_t, **V_t, **A, **temp, **S_mat, **reconA;
    double alpha, beta, gamma, c, zeta, t,s,sub_zeta, converge;
	int *I1, *I2;

	double *C;
	C = new double[num];

  	int acum = 0;
    int temp1, temp2;
	converge = 1.0;

	S_mat = new double*[N];
	reconA = new double*[N];
	temp = new double*[N];
	U = new double*[N];
	V = new double*[N];
	U_t = new double*[N];
	V_t = new double*[N];
	A = new double*[N];
	S = new double[N];
	I1= new int[N];
	I2= new int[N];

	for(int i =0; i<N; i++){
		S_mat[i]= new double [N];
		temp[i]= new double[N];
		reconA[i]= new double[N];
		U[i] = new double[N];
 		V[i] = new double[N];
		U_t[i] = new double[N];
		V_t[i] = new double[N];
		A[i] = new double[N];
	}
	
    

    
	//Use matrix file to get input matrix
     readMatrixFromMemory(A, "A_256.data");
     
	cout<<"SVD NxN with N: "<<N<<endl;
	cout<<"Computing SVD for "<<num<<" Threads"<<endl<<endl;
	double start = omp_get_wtime();

	//U_t = AT V_t arxika einai identity
	
	transpose(A,U_t);
	
	for(int i=0; i<M;i++){
    	for(int j=0; j<N;j++){

      	if(i==j){
        	V_t[i][j] = 1.0;
      	}
      	else{
        	V_t[i][j] = 0.0;
    	}
	 }
	}
	
	/* SVD using Jacobi algorithm (Parallel)*/

   

   double conv;
   while(converge > epsilon){ 		//convergence
    converge = 0.0;	
   		
    acum++;				//counter of loops
	
	   for (int l = 1; l < M; l ++) {
		   
		   int r1 = 0, r2 = 0;
		   for (int i = 0; i + l < M; i++) {
			   if (i % (2 * l) < l)
				   I1[++r1] = i;
			   else
				   I2[++r2] = i;
		   }
		   for (int k = 0; k < num; k++) {
			   C[k] = converge;
		   }
		   
			
		
		   #pragma omp parallel for num_threads(num)
		   for (int p = 1; p <= r1; p++){
			   int k = omp_get_thread_num();
			   int i = I1[p], j = i + l;
			   double alpha = 0, beta = 0, gamma = 0;
			   double zeta, t, c, s;
			   for (int k = 0; k < N; k++) {
				   alpha = alpha + (U_t[i][k] * U_t[i][k]);
				   beta = beta + (U_t[j][k] * U_t[j][k]);
				   gamma = gamma + (U_t[i][k] * U_t[j][k]);
			   }
			   C[k] = max(C[k], abs(gamma)/sqrt(alpha*beta));
			   //converge = max(converge, abs(gamma)/sqrt(alpha*beta));	//compute convergence
			   //basicaly is the angle
			   //between column i and j
			   
			   
			   zeta = (beta - alpha) / (2.0 * gamma);
			   t = sgn(zeta) / (abs(zeta) + sqrt(1.0 + (zeta*zeta)));        //compute tan of angle
			   c = 1.0 / (sqrt (1.0 + (t*t)));				//extract cos
			   s = c*t;							//extrac sin
			   for(int k=0; k<N; k++){
				   t = U_t[i][k];
				   U_t[i][k] = c*t - s*U_t[j][k];
				   U_t[j][k] = s*t + c*U_t[j][k];
				   
				   t = V_t[i][k];
				   V_t[i][k] = c*t - s*V_t[j][k];
				   V_t[j][k] = s*t + c*V_t[j][k];
				   
			   }
		   }
		   #pragma omp parallel for num_threads(num)
		   for (int p = 1; p <= r2; p++){
			   int k = omp_get_thread_num();
			   int i = I2[p], j = i + l;
			   double alpha = 0, beta = 0, gamma = 0;
			   double zeta, t, c, s;
			   for (int k = 0; k < N; k++) {
				   alpha = alpha + (U_t[i][k] * U_t[i][k]);
				   beta = beta + (U_t[j][k] * U_t[j][k]);
				   gamma = gamma + (U_t[i][k] * U_t[j][k]);
			   }
			   C[k] = max(C[k], abs(gamma)/sqrt(alpha*beta));
			   //converge = max(converge, abs(gamma)/sqrt(alpha*beta));	//compute convergence
			   //basicaly is the angle
			   //between column i and j
			   
			   
			   zeta = (beta - alpha) / (2.0 * gamma);
			   t = sgn(zeta) / (abs(zeta) + sqrt(1.0 + (zeta*zeta)));        //compute tan of angle
			   c = 1.0 / (sqrt (1.0 + (t*t)));				//extract cos
			   s = c*t;							//extrac sin
			   for(int k=0; k<N; k++){
				   t = U_t[i][k];
				   U_t[i][k] = c*t - s*U_t[j][k];
				   U_t[j][k] = s*t + c*U_t[j][k];
				   
				   t = V_t[i][k];
				   V_t[i][k] = c*t - s*V_t[j][k];
				   V_t[j][k] = s*t + c*V_t[j][k];
				   
			   }
		   }
		   for (int k = 0; k < num; k++)
			   converge = max(converge, C[k]);
	   }
 }
	
	//Create matrix S

  for(int i =0; i<M; i++){

    t=0;
    for(int j=0; j<N;j++){
      t=t + pow(U_t[i][j],2);
    }
    t = sqrt(t);

    for(int j=0; j<N;j++){
      U_t[i][j] = U_t[i][j] / t;
      if(i == j){
        S[i] = t;
      }
    }
  }
  
  
  // fix final result

  for(int i =0; i<M; i++){
    
    for(int j =0; j<N; j++){

      U[i][j] = U_t[j][i];
      V[i][j] = V_t[j][i];
      
    }
    
  }
  
  double end = omp_get_wtime();
  cout<<"SVD Finished after: "<<end-start<<" seconds"<<endl<<endl;
  

  //free memory
  delete [] S;
   for(int i = 0; i<N;i++){
   		delete[] S_mat[i];
	   delete [] A[i];
	   delete [] U[i];
	   delete [] V[i];
	   delete [] U_t[i];
	   delete [] V_t[i];
	   delete [] reconA[i];
	   delete [] temp[i];
   }
   delete [] S_mat;
   delete [] A;
   delete [] U;
   delete [] U_t;
   delete [] V_t;
   delete [] V;
   delete [] reconA;
   delete [] temp;   
   delete [] I1;
   delete [] I2;
   delete [] C;
	
	return 0;
}
