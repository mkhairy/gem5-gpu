#include <cassert>
#include <cstring>
#include <iostream>

#include <sys/time.h>
double gettime() {
    struct timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec+t.tv_usec*1e-6;
}

using namespace std;

#define MAX_COUNT 1000000

__global__ void gpu_atomics_address_fun(unsigned *counter_array, unsigned iters, unsigned fun) {
    unsigned index = threadIdx.x / fun;
    atomicMax(&counter_array[index], threadIdx.x);
    for (unsigned i = 0; i < iters; i++) {
        // atomicInc(&counter_array[index], MAX_COUNT);
        // atomicAdd(&counter_array[index], 3);
        // atomicMax(&counter_array[index], threadIdx.x);
        // atomicExch(&counter_array[index], threadIdx.x);
        // atomicCAS(&counter_array[index], 4, threadIdx.x);
        // atomicAnd(&counter_array[index], threadIdx.x);
        // index = atomicInc(&counter_array[index], blockDim.x) / fun;
        index = atomicMax(&counter_array[index], threadIdx.x) / fun;
        // index = atomicMax(&counter_array[index], threadIdx.x);
    }
    counter_array[threadIdx.x] = index;
}

__global__ void gpu_atomics_shared_address_fun(unsigned *counter_array, unsigned iters, unsigned fun) {
    extern __shared__ unsigned our_counter[];
    our_counter[threadIdx.x] = 0;
    __syncthreads();
    for (unsigned i = 0; i < iters; i++) {
        atomicInc(&our_counter[threadIdx.x/fun], MAX_COUNT);
        // atomicAdd(&our_counter[threadIdx.x/fun], 3);
        // atomicMax(&our_counter[threadIdx.x/fun], threadIdx.x);
    }
    __syncthreads();
    atomicAdd(&counter_array[threadIdx.x], our_counter[threadIdx.x]);
}

int main(int argc, char** argv) {
    unsigned threads_per_block = 32;
    unsigned num_blocks = 1;
    unsigned num_iterations = 1;
    bool use_shared_memory = false;
    unsigned address_fun = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-b")) {
            if (i < argc) {
                num_blocks = atoi(argv[++i]);
            } else {
                cout << "Need to specify number of blocks to '-b'\n";
                exit(-1);
            }
        } else if (!strcmp(argv[i], "-f")) {
            if (i < argc) {
                address_fun = atoi(argv[++i]);
            } else {
                cout << "Need to specify number of writers per address to '-f'\n";
                exit(-1);
            }
        } else if (!strcmp(argv[i], "-h")) {
            use_shared_memory = true;
        } else if (!strcmp(argv[i], "-i")) {
            if (i < argc) {
                num_iterations = atoi(argv[++i]);
            } else {
                cout << "Need to specify number of iterations to '-i'\n";
                exit(-1);
            }
        } else if (!strcmp(argv[i], "-t")) {
            if (i < argc) {
                threads_per_block = atoi(argv[++i]);
            } else {
                cout << "Need to specify number of threads to '-t'\n";
                exit(-1);
            }
        } else {
            cerr << "ERROR: Invalid argument: " << argv[i] << endl;
        }
    }

    // Check inputs
    assert(address_fun > 0);

    // Device data and pointers
    unsigned *d_counter_array;
    cudaMalloc(&d_counter_array, threads_per_block * sizeof(unsigned));
    cudaMemset(d_counter_array, 0, threads_per_block * sizeof(unsigned));

    double start_time, end_time;
    start_time = gettime();

    if (!use_shared_memory) {
        gpu_atomics_address_fun<<<num_blocks, threads_per_block>>>(d_counter_array, num_iterations, address_fun);
    } else {
        size_t shared_size = threads_per_block * sizeof(unsigned);
        gpu_atomics_shared_address_fun<<<num_blocks, threads_per_block, shared_size>>>(d_counter_array, num_iterations, address_fun);
    }

    cudaThreadSynchronize();

    end_time = gettime();

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "ERROR: Kernel execution failed with code: " << err
             << ", message: " << cudaGetErrorString(err) << endl;
        exit(-1);
    }

    unsigned *counter_array = new unsigned[threads_per_block];
    cudaMemcpy(counter_array, d_counter_array, threads_per_block * sizeof(unsigned), cudaMemcpyDeviceToHost);

    cout << "Threads per block: " << threads_per_block << endl;
    cout << "Number of blocks: " << num_blocks << endl;
    cout << "Final vals:" << endl;
    for (int i = 0; i < threads_per_block; i++) {
        cout << counter_array[i] << endl;
    }
    cout << "Time: " << (end_time - start_time) << endl;

    delete [] counter_array;
    return 0;
}
