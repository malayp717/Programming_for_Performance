// nvcc -lineinfo -res-usage -arch=sm_75 -std=c++11 22111090-prob1-v1.cu -o 22111090-prob1-v1

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>
#include <random>

const uint64_t N = (64);
uint64_t SIZE = N * N * N;
#define THRESHOLD (0.000001)
using std::cerr;
using std::cout;
using std::endl;

std::default_random_engine generator;
std::normal_distribution<float> distribution(1.0f,1.0f);

#define cudaCheckError(ans)                                                                        \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

// TODO: Edit the function definition as required
__global__ void kernel1(float* d_input, float* d_output) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  if (i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
    d_output[i*N*N + j*N + k] = 0.8f *
        (d_input[(i-1)*N*N + j*N + k] + d_input[(i+1)*N*N + j*N + k] +
         d_input[i*N*N + (j-1)*N + k] + d_input[i*N*N + (j+1)*N + k] +
         d_input[i*N*N + j*N + (k-1)] + d_input[i*N*N + j*N + (k+1)]);
  }
}

__global__ void kernel2(const float* d_input, float* d_output) {

  extern __shared__ float tile[];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;

  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int tidz = threadIdx.z;

  tile[blockDim.y*blockDim.z*tidx + blockDim.z*tidy + tidz] = d_input[i*N*N + j*N + k];
  
  __syncthreads();

  if (i > 0 && i < N-1 && j > 0 && j < N-1 && k > 0 && k < N-1) {
    if (tidx > 0 && tidx < (blockDim.x-1) && tidy > 0 && tidy < (blockDim.y-1) && tidz > 0 && tidz < (blockDim.z-1)) {
      d_output[i*N*N + j*N + k] = 0.8f * (
        tile[blockDim.y*blockDim.z*(tidx-1) + blockDim.z*tidy + tidz] + tile[blockDim.y*blockDim.z*(tidx+1) + blockDim.z*tidy + tidz] +
        tile[blockDim.y*blockDim.z*tidx + blockDim.z*(tidy-1) + tidz] + tile[blockDim.y*blockDim.z*tidx + blockDim.z*(tidy+1) + tidz] +
        tile[blockDim.y*blockDim.z*tidx + blockDim.z*tidy + (tidz-1)] + tile[blockDim.y*blockDim.z*tidx + blockDim.z*tidy + (tidz+1)]);
    }
    else if (tidx > 0 && tidx < (blockDim.x-1) && tidy > 0 && tidy < (blockDim.y-1)) {
      d_output[i*N*N + j*N + k] = 0.8f * (
        tile[blockDim.y*blockDim.z*(tidx-1) + blockDim.z*tidy + tidz] + tile[blockDim.y*blockDim.z*(tidx+1) + blockDim.z*tidy + tidz] +
        tile[blockDim.y*blockDim.z*tidx + blockDim.z*(tidy-1) + tidz] + tile[blockDim.y*blockDim.z*tidx + blockDim.z*(tidy+1) + tidz] +
        d_input[i*N*N + j*N + (k-1)] + d_input[i*N*N + j*N + (k+1)]);
    }
    else if (tidx > 0 && tidx < (blockDim.x-1) && tidz > 0 && tidz < (blockDim.z-1)) {
      d_output[i*N*N + j*N + k] = 0.8f * (
        tile[blockDim.y*blockDim.z*(tidx-1) + blockDim.z*tidy + tidz] + tile[blockDim.y*blockDim.z*(tidx+1) + blockDim.z*tidy + tidz] +
        d_input[i*N*N + (j-1)*N + k] + d_input[i*N*N + (j+1)*N + k] +
        tile[blockDim.y*blockDim.z*tidx + blockDim.z*tidy + (tidz-1)] + tile[blockDim.y*blockDim.z*tidx + blockDim.z*tidy + (tidz+1)]);
    }
    else if (tidy > 0 && tidy < (blockDim.y-1) && tidz > 0 && tidz < (blockDim.z-1)) {
      d_output[i*N*N + j*N + k] = 0.8f * (
        d_input[(i-1)*N*N + j*N + k] + d_input[(i+1)*N*N + j*N + k] +
        tile[blockDim.y*blockDim.z*tidx + blockDim.z*(tidy-1) + tidz] + tile[blockDim.y*blockDim.z*tidx + blockDim.z*(tidy+1) + tidz] +
        tile[blockDim.y*blockDim.z*tidx + blockDim.z*tidy + (tidz-1)] + tile[blockDim.y*blockDim.z*tidx + blockDim.z*tidy + (tidz+1)]);
    }
    else if (tidx > 0 && tidx < (blockDim.x-1)) {
      d_output[i*N*N + j*N + k] = 0.8f * (
        tile[blockDim.y*blockDim.z*(tidx-1) + blockDim.z*tidy + tidz] + tile[blockDim.y*blockDim.z*(tidx+1) + blockDim.z*tidy + tidz] +
        d_input[i*N*N + (j-1)*N + k] + d_input[i*N*N + (j+1)*N + k] +
        d_input[i*N*N + j*N + (k-1)] + d_input[i*N*N + j*N + (k+1)]);
    }
    else if (tidy > 0 && tidy < (blockDim.y-1)) {
      d_output[i*N*N + j*N + k] = 0.8f * (
        d_input[(i-1)*N*N + j*N + k] + d_input[(i+1)*N*N + j*N + k] +
        tile[blockDim.y*blockDim.z*tidx + blockDim.z*(tidy-1) + tidz] + tile[blockDim.y*blockDim.z*tidx + blockDim.z*(tidy+1) + tidz] +
        d_input[i*N*N + j*N + (k-1)] + d_input[i*N*N + j*N + (k+1)]);
    }
    else if (tidz > 0 && tidz < (blockDim.z-1)) {
      d_output[i*N*N + j*N + k] = 0.8f * (
        d_input[(i-1)*N*N + j*N + k] + d_input[(i+1)*N*N + j*N + k] +
        d_input[i*N*N + (j-1)*N + k] + d_input[i*N*N + (j+1)*N + k] +
        tile[blockDim.y*blockDim.z*tidx + blockDim.z*tidy + (tidz-1)] + tile[blockDim.y*blockDim.z*tidx + blockDim.z*tidy + (tidz+1)]);
    }
    else {
      d_output[i*N*N + j*N + k] = 0.8f *
        (d_input[(i-1)*N*N + j*N + k] + d_input[(i+1)*N*N + j*N + k] +
         d_input[i*N*N + (j-1)*N + k] + d_input[i*N*N + (j+1)*N + k] +
         d_input[i*N*N + j*N + (k-1)] + d_input[i*N*N + j*N + (k+1)]);
    }
  }
}

// TODO: Edit the function definition as required
__host__ void stencil(float* h_input, float* h_output) {
  for (int i = 1; i < N-1; i++) {
    for (int j = 1; j < N-1; j++) {
      for (int k = 1; k < N-1; k++) {
        h_output[(i * N * N) + (j * N) + k] = 0.8f *
                (h_input[((i-1) * N * N) + (j * N) + k] + h_input[((i+1) * N * N) + (j * N) + k] +
                 h_input[(i * N * N) + ((j-1) * N) + k] + h_input[(i * N * N) + ((j+1) * N) + k] +
                 h_input[(i * N * N) + (j * N) + (k-1)] + h_input[(i * N * N) + (j * N) + (k+1)]);
      }
    }
  }
}

__host__ void check_result(const float* w_ref, const float* w_opt, const uint64_t size) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        this_diff = w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

void print_mat(float* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        printf("%lf,", A[i * N * N + j * N + k]);
      }
      printf("      ");
    }
    printf("\n");
  }
}

double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main() {

  float *h_input = new float[SIZE];
  float *h_serial = new float[SIZE];
  float *h_k1 = new float[SIZE];
  float *h_k2 = new float[SIZE];

  memset(h_serial, 0.0f, sizeof(h_serial));

  for (int i = 0; i < SIZE; i++) h_input[i] = distribution(generator);

  double clkbegin = rtclock();
  stencil(h_input, h_serial);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU (ms): " << cpu_time * 1000 << endl;

  float *d_input, *d_k1, *d_k2;

  /*  Naive Kernel implementation starts */

  size_t T_ROWS = 8, T_COLS = 8, T_DEPTH = 8;
  size_t G_ROWS = (N + T_ROWS - 1)/ T_ROWS, G_COLS = (N + T_COLS - 1)/ T_COLS, G_DEPTH = (N + T_DEPTH - 1)/ T_DEPTH;

  cudaCheckError(cudaMalloc((void **)&d_input, SIZE * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&d_k1, SIZE * sizeof(float)));
  cudaCheckError(cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice));

  dim3 gridSize_1(G_ROWS, G_COLS, G_DEPTH);
  dim3 blockSize_1(T_ROWS, T_COLS, T_DEPTH);

  // cudaError_t status;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float kernel_time;
  cudaEventRecord(start);
  kernel1<<<gridSize_1, blockSize_1>>>(d_input, d_k1);
  cudaCheckError(cudaPeekAtLastError());
  cudaCheckError(cudaDeviceSynchronize());
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernel_time, start, end);
  std::cout << "Naive Kernel time (ms): " << kernel_time << "\n";

  cudaMemcpy(h_k1, d_k1, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  check_result(h_serial, h_k1, N);
  // print_mat(h_k1);

  /* Shared Memory Kernel implementation starts */

  cudaCheckError(cudaMalloc((void **)&d_k2, SIZE * sizeof(float)));
  cudaCheckError(cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice));

  int block_sizes[] = {1, 2, 4, 8};

  for (int block_size : block_sizes) {

    dim3 blockSize_2(block_size, block_size, block_size);
    dim3 gridSize_2(((N + blockSize_2.x - 1) / blockSize_2.x), ((N + blockSize_2.y - 1) / blockSize_2.y), ((N + blockSize_2.z - 1) / blockSize_2.z));

    int sz = blockSize_2.y*blockSize_2.z*blockSize_2.x + blockSize_2.z*blockSize_2.y + blockSize_2.z;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    kernel2<<<gridSize_2, blockSize_2, sz * sizeof(float)>>>(d_input, d_k2);
    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize());
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&kernel_time, start, end);
    std::cout << "Shared Memory Kernel block size: "<< block_size << ", time (ms): " << kernel_time << "\n";

    cudaMemcpy(h_k2, d_k2, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    check_result(h_serial, h_k2, N);
  }

  // TODO: Free memory

  delete [] h_input;
  delete [] h_serial;
  delete [] h_k1;
  delete [] h_k2;
  cudaFree(d_input);
  cudaFree(d_k1);
  cudaFree(d_k2);

  return EXIT_SUCCESS;
}
