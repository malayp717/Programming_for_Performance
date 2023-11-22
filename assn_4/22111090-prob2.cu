// nvcc -lineinfo -res-usage -arch=sm_75 -std=c++14 22111090-prob2.cu -o 22111090-prob2

#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <ctime>
#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <chrono>
#include <sys/time.h>
using namespace std;

std::random_device rd;
std::mt19937 rng(rd());
std::uniform_int_distribution<int> uni(0, 5);

// template <typename T>
void print(int *A, int N) {
    for (int i = 0; i < N; i++) cout << A[i] << ' ';
    cout << endl;
}

template <typename T>
void check(T &A, T &B, int N) {
    long long numDiffs = 0;

    for (int i = 0; i < N; i++) if (A[i] != B[i]) numDiffs++;

    cout << numDiffs << " differences found between Sequential and CUDA / Thrust version" << endl;
}

void seqSum(int* input, int* output, int N) {

    output[0] = 0;
    for(int i = 0; i < N-1 ; i++) {
        output[i+1] = output[i] + input[i];
    }
}

__global__ void exclusiveScanGPU(int *d_array, int *d_result, int N, int *d_aux) {

    extern __shared__ int temp[]; 

    int rIdx = 2 * threadIdx.x + blockDim.x * 2 * blockIdx.x;
    int tIdx = threadIdx.x;  
    int i = 2 * tIdx;   
    int offset = 1;

    temp[i] = d_array[rIdx];
    temp[i+1] = d_array[rIdx+1];  

    for (int d = blockDim.x; d > 0; d = d/2) {   
        __syncthreads();  

        if (tIdx < d) {
            int curr = offset*(i+1)-1;
            int next = offset*(i+2)-1;
            temp[next] += temp[curr];  
        }

        offset *= 2; 
    }

    if (tIdx == 0) {
        if(d_aux) {
            d_aux[blockIdx.x] = temp[N-1];
        }
        temp[N - 1] = 0; 
    } 

    for (int d = 1; d < blockDim.x*2; d *= 2) {  
        offset = offset / 2;
        __syncthreads();  

        if (tIdx < d)                       
        {  
            int curr = offset*(i+1)-1;  
            int next = offset*(i+2)-1;

            // Swap
            int tempCurrent = temp[curr];  
            temp[curr] = temp[next]; 
            temp[next] += tempCurrent;   
        }  
    }  

    __syncthreads(); 

    d_result[rIdx] = temp[i];  
    d_result[rIdx+1] = temp[i+1];      
}

__global__ void sum(int *d_incr, int *d_result, int N) {
    int addThis = d_incr[blockIdx.x];
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    d_result[tid] += addThis;
}

double calculate_time(clock_t &begin, clock_t &end) {
    return double(end - begin)/CLOCKS_PER_SEC;
}

int main() {
    const long long N = (1 << 20) ;
    int threads = (1 << 10);

    struct timeval start, stop;

    int blocks = (N + threads - 1) / threads;
    long long sz = blocks * threads;

    int *h_input = new int[sz];
    int *h_seq = new int[sz];
    int *h_cuda = new int[sz];
    int *h_aux = new int[sz];
    int *h_incr = new int[sz];
    int *h_thrust = new int[sz];

    memset(h_input, 0, sizeof h_input);
    memset(h_seq, 0, sizeof h_seq);
    memset(h_cuda, 0, sizeof h_cuda);
    memset(h_thrust, 0, sizeof h_thrust);
    memset(h_aux, 0, sizeof h_aux);
    memset(h_incr, 0, sizeof h_incr);

    for (int i = 0; i < N; i++) {
        h_input[i] = h_thrust[i] = uni(rng);
    }

    /* Sequential Code begins */

    gettimeofday(&start, NULL);
    seqSum(h_input, h_seq, N);
    gettimeofday(&stop, NULL);
    printf("Sequential elapsed time (us): %lu \n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

    /* CUDA Code begins */

    int *d_input, *d_aux, *d_output, *d_incr;

    cudaMalloc((void**)&d_input, sz * sizeof(int));
    cudaMalloc((void**)&d_aux, sz * sizeof(int));
    cudaMalloc((void**)&d_output, sz * sizeof(int));
    cudaMalloc((void**)&d_incr, sz * sizeof(int));

    cudaMemcpy(d_input, h_input, sz * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(threads / 2, 1, 1);
    dim3 gridSize(blocks, 1, 1);

    cudaEvent_t begin, end;
    float cudaDuration = 0.0f, t = 0.0f;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    cudaEventRecord(begin);
    exclusiveScanGPU<<<gridSize, blockSize, threads * sizeof(int)>>>(d_input, d_output, threads, d_aux);
    cudaDeviceSynchronize();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&t, begin, end);
    cudaDuration += t;

    cudaEventRecord(begin);
    exclusiveScanGPU<<<dim3(1, 1, 1), blockSize, threads * sizeof(int)>>>(d_aux, d_incr, threads, NULL);
    cudaDeviceSynchronize();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&t, begin, end);
    cudaDuration += t;

    cudaEventRecord(begin);
    sum<<<gridSize, dim3(threads, 1, 1)>>>(d_incr, d_output, N);
    cudaDeviceSynchronize();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&t, begin, end);
    cudaDuration += t;

    cudaMemcpy(h_cuda, d_output, sz * sizeof(int), cudaMemcpyDeviceToHost);
    check(h_seq, h_cuda, N);
    cout << "CUDA elapsed time(us): " << cudaDuration * 1000 << endl;

    /* Thrust code begins */

    gettimeofday(&start, NULL);
    thrust::exclusive_scan(h_thrust, h_thrust + N, h_thrust);
    gettimeofday(&stop, NULL);
    check(h_seq, h_thrust, N);
    printf("Thrust elapsed time (us): %lu \n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

    // print(h_input, N);
    // print(h_seq, N);
    // print(h_cuda, N);
    // print(h_thrust, N);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_aux);
    cudaFree(d_incr);
    delete [] h_input;
    delete [] h_seq;
    delete [] h_cuda;
    delete [] h_aux;
    delete [] h_incr;
    delete [] h_thrust;

    return 0;
}
