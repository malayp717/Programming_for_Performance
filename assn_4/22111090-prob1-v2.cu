// nvcc -lineinfo -res-usage -arch=sm_75 -std=c++11 22111090-prob1-v2.cu -o 22111090-prob1-v2

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

__global__ void kernel(const float* d_input, float* d_output) {

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

  float *h_input, *h_serial, *h_kernel;

  cudaCheckError(cudaMallocHost((void**)&h_input, SIZE * sizeof(float)));
  cudaCheckError(cudaMallocHost((void**)&h_serial, SIZE * sizeof(float)));
  cudaCheckError(cudaMallocHost((void**)&h_kernel, SIZE * sizeof(float)));

  for (int i = 0; i < SIZE; i++) h_input[i] = distribution(generator);

  double clkbegin = rtclock();
  stencil(h_input, h_serial);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU (ms): " << cpu_time * 1000 << endl;

  /*  Pinned Memory Kernel implementation starts */

  cudaEvent_t start, end;
  float kernel_time;

  float *d_input, *d_kernel;

  cudaCheckError(cudaMalloc((void**)&d_input, SIZE * sizeof(float)));
  cudaCheckError(cudaMalloc((void**)&d_kernel, SIZE * sizeof(float)));
  cudaCheckError(cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice));

  int block_sizes[] = {1, 2, 4, 8};
  for (int block_size : block_sizes) {

    dim3 blockSize(block_size, block_size, block_size);
    dim3 gridSize(((N + blockSize.x - 1) / blockSize.x), ((N + blockSize.y - 1) / blockSize.y), ((N + blockSize.z - 1) / blockSize.z));

    int sz = blockSize.y*blockSize.z*blockSize.x + blockSize.z*blockSize.y + blockSize.z;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    kernel<<<gridSize, blockSize, sz * sizeof(float)>>>(d_input, d_kernel);
    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize());
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&kernel_time, start, end);
    std::cout << "Pinned Memory Kernel block size: "<< block_size << ", time (ms): " << kernel_time << "\n";

    cudaCheckError(cudaMemcpy(h_kernel, d_kernel, SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    check_result(h_serial, h_kernel, N);
  }

  // TODO: Free memory
  cudaFree(d_input);
  cudaFree(d_kernel);
  cudaFreeHost(h_input);
  cudaFreeHost(h_serial);
  cudaFreeHost(h_kernel);

  return EXIT_SUCCESS;
}