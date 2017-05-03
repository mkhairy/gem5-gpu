#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>
using namespace std;
#include <cuda_runtime.h>
//#include <sdkHelper.h> 
#define TIMES 1

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////HELP FUNCTIONS/////////////////////////////////////////////////
void RandomInit(float* data, int n)
{
    for (int i=0; i<n; i++)
	{
        data[i] = rand() / (float)RAND_MAX;
	}
}

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);        
    }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
        file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////__NAIVE_MATRIX_MULTIPLICATION_///////////////////////////////////////////////
// Device code
// Compute C = A * B

#define TILEWIDTH_X 32
#define TILEWIDTH_Y 32
#define TILEWIDTH 32

__global__ void matrixMultiply(float * A, float * B, float * C,
			        int numAColumns,
			       int numBColumns,
			       int numCRows, int numCColumns) {
      float Cvalue = 0;
  	  int row = blockIdx.y*blockDim.y + threadIdx.y;
  	  int col = blockIdx.x * blockDim.x + threadIdx.x;
     if(row < numCRows && col < numCColumns)
        {
          for(int e=0; e<numAColumns; ++e)
            Cvalue += A[row*numAColumns + e] * B[e*numBColumns + col];
          C[row*numCColumns + col] = Cvalue;
        }
}
void MatrixMulOnHost(float * A, float * B, float * C,
			       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns)
{
	for (int i = 0; i < numARows; ++i)
	for (int j = 0; j < numBColumns; ++j) {
	float sum = 0;
	for (int k = 0; k < numAColumns; ++k) {
	float a = A[i * numAColumns + k];
	float b = B[k * numBColumns + j];
	sum += a * b;
	}
	C[i * numCColumns + j] = sum;
	}
}

int MatrixMulti(int numARows, int numAColumns, int numBRows, int numBColumns, int blockx, int blocky, bool optimzed, bool define=false) 
{
	if(!optimzed)
		printf("NAIVE MATRIX MULTIPLICATION\n");
	else if(define)
		printf("Optimzed MATRIX MULTIPLICATION with static shared memory allocation\n");
	else
		printf("Optimzed MATRIX MULTIPLICATION with static dynamic memory allocation\n");

    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numCRows = numARows;; // number of rows in the matrix C (you have to set this)
    int numCColumns = numBColumns;; // number of columns in the matrix C (you have to set this)
	double total_time=0;
	//StopWatchInterface* timer;
    int sizeA = numARows*numAColumns*sizeof(float);
	int sizeB = numBRows*numBColumns*sizeof(float);
	int sizeC = numCRows*numCColumns*sizeof(float);

	if(numAColumns != numBRows)
	{
		cout<<"Error in inputs dimension! A columns != B rows"<<endl;
		exit(-1);
	}
    // Allocate input vectors h_A and h_B in host memory
    hostA = (float*)malloc(sizeA);
    hostB = (float*)malloc(sizeB);
    hostC = (float*)malloc(sizeC);
    
    // Initialize input vectors
    RandomInit(hostA, numARows*numAColumns);
    RandomInit(hostB, numBRows*numBColumns);

    cout<<"The dimensions of A are "<<numARows<<" x "<<numAColumns<<endl;
    cout<<"The dimensions of B are "<<numBRows<<" x "<<numBColumns<<endl;

    //Allocate GPU memory here
   // checkCudaErrors(cudaMalloc(&deviceA, sizeA));
  //	checkCudaErrors(cudaMalloc(&deviceB, sizeB));
  //	checkCudaErrors(cudaMalloc(&deviceC, sizeC));

    //@@ Copy memory to the GPU here
	//checkCudaErrors(cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice));
  //	checkCudaErrors(cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice));
    
#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

	dim3 dimBlock, dimGrid;

	dimBlock = dim3(blockx, blocky);
	dimGrid = dim3((numCColumns+blockx-1)/blockx, (numCRows+blocky-1)/blocky);

	matrixMultiply<<<dimGrid, dimBlock>>>(hostA, hostB, hostC, numAColumns, numBColumns, numCRows, numCColumns);
	getLastCudaError("kernel launch failure");

	checkCudaErrors(cudaThreadSynchronize());


#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

	double dSeconds = total_time/((double)TIMES * 1000);
	double dNumOps = 2.0 * (double)numARows * (double)numAColumns * (double)numBColumns;
	double gflops = 1.0e-9 * dNumOps/dSeconds;
	cout<<"Time = "<<dSeconds*1.0e3<< "msec"<<endl<<"gflops = "<<gflops<<endl;

    //@@ Copy the GPU memory back to the CPU here
//	checkCudaErrors(cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost));

	// Verify result
	//float* hostcpu = (float*)malloc(sizeC);
	/*MatrixMulOnHost(hostA,hostB,hostcpu,numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns);
    int i;
	int j;
    for (i = 0; i < numCRows; ++i) 
	for(j=0; j<numCColumns; j++)
	{
        if (fabs(hostC[i*numCColumns + j] - hostcpu[i*numCColumns + j]) > 1e-3)
		{
            break;
		}
    }*/

    //@@ Free the GPU memory here
	//checkCudaErrors(cudaFree(deviceA));
  //	checkCudaErrors(cudaFree(deviceB));
  	//checkCudaErrors(cudaFree(deviceC));
	//cudaDeviceReset();

    free(hostA);
    free(hostB);
    free(hostC);
	//free(hostcpu);
	/*if(i == numCRows && j == numCColumns)
		cout<<"SUCCSESS"<<endl;
	else 
		cout<<"FAILED"<<endl; */

    return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc,char *argv[])
{ 

if(argc < 6)
     printf("Unsuffcient number of arguments!\n");
else
	{
		MatrixMulti(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), false);
	}
}
