// nvcc --extended-lambda -lineinfo -res-usage -arch=sm_75 -std=c++14 22111090-prob3-v4.cu -o 22111090-prob3-v4

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
using namespace std;

#define NSEC_SEC_MUL (1.0e9)

struct Result {
  double x[10];
};

void gridloopsearch(double *a, double *b, double kk);

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct timespec begin_grid, end_main;

// to store values of disp.txt
double a[120];

// to store values of grid.txt
double b[30];

int main() {
  int i, j;

  i = 0;
  FILE* fp = fopen("./disp.txt", "r");
  if (fp == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  while (!feof(fp)) {
    if (!fscanf(fp, "%lf", &a[i])) {
      printf("Error: fscanf failed while reading disp.txt\n");
      exit(EXIT_FAILURE);
    }
    i++;
  }
  fclose(fp);

  // read grid file
  j = 0;
  FILE* fpq = fopen("./grid.txt", "r");
  if (fpq == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  while (!feof(fpq)) {
    if (!fscanf(fpq, "%lf", &b[j])) {
      printf("Error: fscanf failed while reading grid.txt\n");
      exit(EXIT_FAILURE);
    }
    j++;
  }
  fclose(fpq);

  // grid value initialize
  // initialize value of kk;
  double kk = 0.3;

  clock_gettime(CLOCK_MONOTONIC_RAW, &begin_grid);
  gridloopsearch(a, b, kk);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_main);
  printf("Total time = %f seconds\n", (end_main.tv_nsec - begin_grid.tv_nsec) / NSEC_SEC_MUL +
                                          (end_main.tv_sec - begin_grid.tv_sec));

  return EXIT_SUCCESS;
}

__global__ void gridloopsearch_kernel(int* pnts, Result* results, double *a, double *b, double *kk) {

  extern __shared__ double shmem[][3];

  double dd1=b[0],dd2=b[1],dd3=b[2],dd4=b[3],dd5=b[4],dd6=b[5],dd7=b[6],dd8=b[7],dd9=b[8],dd10=b[9];
  double dd11=b[10],dd12=b[11],dd13=b[12],dd14=b[13],dd15=b[14],dd16=b[15],dd17=b[16],dd18=b[17],dd19=b[18],dd20=b[19];
  double dd21=b[20],dd22=b[21],dd23=b[22],dd24=b[23],dd25=b[24],dd26=b[25],dd27=b[26],dd28=b[27],dd29=b[28],dd30=b[29];
  double c11=a[0],c12=a[1],c13=a[2],c14=a[3],c15=a[4],c16=a[5],c17=a[6],c18=a[7],c19=a[8],c110=a[9],d1=a[10],ey1=a[11];
  double c21=a[12],c22=a[13],c23=a[14],c24=a[15],c25=a[16],c26=a[17],c27=a[18],c28=a[19],c29=a[20],c210=a[21],d2=a[22];
  double ey2=a[23],c31=a[24],c32=a[25],c33=a[26],c34=a[27],c35=a[28],c36=a[29],c37=a[30],c38=a[31],c39=a[32],c310=a[33];
  double d3=a[34],ey3=a[35],c41=a[36],c42=a[37],c43=a[38],c44=a[39],c45=a[40],c46=a[41],c47=a[42],c48=a[43],c49=a[44];
  double c410=a[45],d4=a[46],ey4=a[47],c51=a[48],c52=a[49],c53=a[50],c54=a[51],c55=a[52],c56=a[53],c57=a[54],c58=a[55];
  double c59=a[56],c510=a[57],d5=a[58],ey5=a[59],c61=a[60],c62=a[61],c63=a[62],c64=a[63],c65=a[64],c66=a[65],c67=a[66];
  double c68=a[67],c69=a[68],c610=a[69],d6=a[70],ey6=a[71],c71=a[72],c72=a[73],c73=a[74],c74=a[75],c75=a[76],c76=a[77];
  double c77=a[78],c78=a[79],c79=a[80],c710=a[81],d7=a[82],ey7=a[83],c81=a[84],c82=a[85],c83=a[86],c84=a[87],c85=a[88];
  double c86=a[89],c87=a[90],c88=a[91],c89=a[92],c810=a[93],d8=a[94],ey8=a[95],c91=a[96],c92=a[97],c93=a[98],c94=a[99];
  double c95=a[100],c96=a[101],c97=a[102],c98=a[103],c99=a[104],c910=a[105],d9=a[106],ey9=a[107],c101=a[108],c102=a[109];
  double c103=a[110],c104=a[111],c105=a[112],c106=a[113],c107=a[114],c108=a[115],c109=a[116],c1010=a[117],d10=a[118],ey10=a[119];

  int r1 = blockIdx.x * blockDim.x + threadIdx.x;
  int r2 = blockIdx.y * blockDim.y + threadIdx.y;
  int r3 = blockIdx.z * blockDim.z + threadIdx.z;
  int tidx = threadIdx.x, tidy = threadIdx.y, tidz = threadIdx.z;

  int s1, s2, s3, s4, s5, s6, s7, s8, s9, s10;
  s1 = static_cast<int>(floor((dd2 - dd1) / dd3));
  s2 = static_cast<int>(floor((dd5 - dd4) / dd6));
  s3 = static_cast<int>(floor((dd8 - dd7) / dd9));
  s4 = static_cast<int>(floor((dd11 - dd10) / dd12));
  s5 = static_cast<int>(floor((dd14 - dd13) / dd15));
  s6 = static_cast<int>(floor((dd17 - dd16) / dd18));
  s7 = static_cast<int>(floor((dd20 - dd19) / dd21));
  s8 = static_cast<int>(floor((dd23 - dd22) / dd24));
  s9 = static_cast<int>(floor((dd26 - dd25) / dd27));
  s10 = static_cast<int>(floor((dd29 - dd28) / dd30));

  double e1, e2, e3, e4, e5, e6, e7, e8, e9, e10;
  e1 = (*kk) * ey1;
  e2 = (*kk) * ey2;
  e3 = (*kk) * ey3;
  e4 = (*kk) * ey4;
  e5 = (*kk) * ey5;
  e6 = (*kk) * ey6;
  e7 = (*kk) * ey7;
  e8 = (*kk) * ey8;
  e9 = (*kk) * ey9;
  e10 = (*kk) * ey10;

  double x1, x2, x3, x4, x5, x6, x7, x8, x9, x10;
  double q1, q2, q3, q4, q5, q6, q7, q8, q9, q10;

  int shmem_idx = tidx + blockDim.x*(tidy + blockDim.y*tidz);

  if (r1 < s1 && r2 < s2 && r3 < s3) {
    shmem[shmem_idx][0] = dd1 + r1*dd3;
    shmem[shmem_idx][1] = dd4 + r2*dd6;
    shmem[shmem_idx][2] = dd7 + r3*dd9;
  }

  __syncthreads();

  if (r1 < s1 && r2 < s2 && r3 < s3) {

    for (int r4 = 0; r4 < s4; r4++) {
      x4 = dd10 + r4 * dd12;

      for (int r5 = 0; r5 < s5; r5++) {
        x5 = dd13 + r5 * dd15;

        for (int r6 = 0; r6 < s6; r6++) {
          x6 = dd16 + r6 * dd18;

          for (int r7 = 0; r7 < s7; r7++) {
            x7 = dd19 + r7 * dd21;

            for (int r8 = 0; r8 < s8; r8++) {
              x8 = dd22 + r8 * dd24;

              for (int r9 = 0; r9 < s9; r9++) {
                x9 = dd25 + r9 * dd27;

                for (int r10 = 0; r10 < s10; r10++) {
                  x10 = dd28 + r10 * dd30;
                  x1 = shmem[shmem_idx][0];
                  x2 = shmem[shmem_idx][1];
                  x3 = shmem[shmem_idx][2];

                  // constraints

                  q1 = fabs(c11 * x1 + c12 * x2 + c13 * x3 + c14 * x4 + c15 * x5 + c16 * x6 +
                            c17 * x7 + c18 * x8 + c19 * x9 + c110 * x10 - d1);

                  q2 = fabs(c21 * x1 + c22 * x2 + c23 * x3 + c24 * x4 + c25 * x5 + c26 * x6 +
                          c27 * x7 + c28 * x8 + c29 * x9 + c210 * x10 - d2);

                  q3 = fabs(c31 * x1 + c32 * x2 + c33 * x3 + c34 * x4 + c35 * x5 + c36 * x6 +
                          c37 * x7 + c38 * x8 + c39 * x9 + c310 * x10 - d3);

                  q4 = fabs(c41 * x1 + c42 * x2 + c43 * x3 + c44 * x4 + c45 * x5 + c46 * x6 +
                          c47 * x7 + c48 * x8 + c49 * x9 + c410 * x10 - d4);

                  q5 = fabs(c51 * x1 + c52 * x2 + c53 * x3 + c54 * x4 + c55 * x5 + c56 * x6 +
                          c57 * x7 + c58 * x8 + c59 * x9 + c510 * x10 - d5);

                  q6 = fabs(c61 * x1 + c62 * x2 + c63 * x3 + c64 * x4 + c65 * x5 + c66 * x6 +
                          c67 * x7 + c68 * x8 + c69 * x9 + c610 * x10 - d6);

                  q7 = fabs(c71 * x1 + c72 * x2 + c73 * x3 + c74 * x4 + c75 * x5 + c76 * x6 +
                          c77 * x7 + c78 * x8 + c79 * x9 + c710 * x10 - d7);

                  q8 = fabs(c81 * x1 + c82 * x2 + c83 * x3 + c84 * x4 + c85 * x5 + c86 * x6 +
                          c87 * x7 + c88 * x8 + c89 * x9 + c810 * x10 - d8);

                  q9 = fabs(c91 * x1 + c92 * x2 + c93 * x3 + c94 * x4 + c95 * x5 + c96 * x6 +
                          c97 * x7 + c98 * x8 + c99 * x9 + c910 * x10 - d9);

                  q10 = fabs(c101 * x1 + c102 * x2 + c103 * x3 + c104 * x4 + c105 * x5 +
                           c106 * x6 + c107 * x7 + c108 * x8 + c109 * x9 + c1010 * x10 - d10);

                  if ((q1 <= e1) && (q2 <= e2) && (q3 <= e3) && (q4 <= e4) && (q5 <= e5) &&
                    (q6 <= e6) && (q7 <= e7) && (q8 <= e8) && (q9 <= e9) && (q10 <= e10)) {
                      int i = atomicAdd(pnts, 1);
                      Result res;
                      res.x[0] = x1;
                      res.x[1] = x2;
                      res.x[2] = x3;
                      res.x[3] = x4;
                      res.x[4] = x5;
                      res.x[5] = x6;
                      res.x[6] = x7;
                      res.x[7] = x8;
                      res.x[8] = x9;
                      res.x[9] = x10;
                      results[i] = res;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void gridloopsearch(double *a, double *b, double kk) {
    // opening the "results-v1.txt" for writing he results in append mode
    FILE* fptr = fopen("./results-v4.txt", "w");
    if (fptr == NULL) {
      printf("Error in creating file !");
      exit(1);
    }

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int sharedMemoryPerBlock = prop.sharedMemPerBlock;
    // cout << "Shared Memory Size (in KB): " << sharedMemoryPerBlock / 1024 << endl;

    // double *h_results;
    const int max_results = 20000;
    // h_results = new double[max_results*10];

    // int *d_pnts;
    thrust::device_vector<int> d_pnts(1);
    thrust::device_vector<Result> d_results(max_results);
    // int h_pnts = 0;

    double *d_a, *d_b, *d_kk;

    cudaCheckError(cudaMalloc((void**)&d_a, 120 * sizeof(double)));
    cudaCheckError(cudaMalloc((void**)&d_b, 30 * sizeof(double)));
    cudaCheckError(cudaMalloc((void**)&d_kk, sizeof(double)));
    // cudaCheckError(cudaMalloc((void**)&d_results, max_results * 10 * sizeof(double)));
    // cudaCheckError(cudaMalloc((void**)&d_pnts, sizeof(int)));

    cudaCheckError(cudaMemcpy(d_kk, &kk, sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_a, a, 120 * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_b, b, 30 * sizeof(double), cudaMemcpyHostToDevice));
    // cudaCheckError(cudaMemcpy(d_pnts, &h_pnts, sizeof(int), cudaMemcpyHostToDevice));

    // Launch the kernel

    int s1, s2, s3;
    s1 = static_cast<int>(floor((b[1] - b[0]) / b[2]));
    s2 = static_cast<int>(floor((b[4] - b[3]) / b[5]));
    s3 = static_cast<int>(floor((b[7] - b[6]) / b[8]));

    int x = 4, y = 4, z = 4;
    dim3 gridSize((s1+x-1)/x, (s2+y-1)/y, (s3+z-1)/z);
    dim3 blockSize(x, y, z);

    // cout << gridSize.x << ' ' << gridSize.y << ' ' << gridSize.z << endl;
    // cout << blockSize.x << ' ' << blockSize.y << ' ' << blockSize.z << endl;

    int sz = blockSize.x + blockSize.x*(blockSize.y + blockSize.y*blockSize.z);
    // cout << "SHMEM size (in KB): " << (sz * 3 * sizeof(double)) / 1024 << endl;

    assert((sz * 3 * sizeof(double)) <= sharedMemoryPerBlock);

    cudaEvent_t start, end;
    float time_elapsed;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    gridloopsearch_kernel<<<gridSize, blockSize, sz * 3 * sizeof(double)>>>(raw_pointer_cast(d_pnts.data()), raw_pointer_cast(d_results.data()), d_a, d_b, d_kk);
    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize());
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_elapsed, start, end);

    cout << "CUDA time elapsed: " << time_elapsed / 1000.0 << " seconds" << endl;

    thrust::host_vector<int> h_pnts(d_pnts);
    thrust::sort(thrust::device, d_results.begin(), d_results.begin() + h_pnts[0],
      [] __device__ (const Result &a, const Result &b) {
        if (a.x[0] == b.x[0] && a.x[1] == b.x[1] && a.x[2] == b.x[2] && a.x[3] == b.x[3] && a.x[4] == b.x[4] && a.x[5] == b.x[5]
          && a.x[6] == b.x[6] && a.x[7] == b.x[7] && a.x[8] == b.x[8]) return a.x[9] < b.x[9];
        if (a.x[0] == b.x[0] && a.x[1] == b.x[1] && a.x[2] == b.x[2] && a.x[3] == b.x[3] && a.x[4] == b.x[4] && a.x[5] == b.x[5]
          && a.x[6] == b.x[6] && a.x[7] == b.x[7]) return a.x[8] < b.x[8];
        if (a.x[0] == b.x[0] && a.x[1] == b.x[1] && a.x[2] == b.x[2] && a.x[3] == b.x[3] && a.x[4] == b.x[4] && a.x[5] == b.x[5]
          && a.x[6] == b.x[6]) return a.x[7] < b.x[7];
        if (a.x[0] == b.x[0] && a.x[1] == b.x[1] && a.x[2] == b.x[2] && a.x[3] == b.x[3] && a.x[4] == b.x[4] && a.x[5] == b.x[5])
          return a.x[6] < b.x[6];
        if (a.x[0] == b.x[0] && a.x[1] == b.x[1] && a.x[2] == b.x[2] && a.x[3] == b.x[3] && a.x[4] == b.x[4]) return a.x[5] < b.x[5];
        if (a.x[0] == b.x[0] && a.x[1] == b.x[1] && a.x[2] == b.x[2] && a.x[3] == b.x[3]) return a.x[4] < b.x[4];
        if (a.x[0] == b.x[0] && a.x[1] == b.x[1] && a.x[2] == b.x[2]) return a.x[3] < b.x[3];
        if (a.x[0] == b.x[0] && a.x[1] == b.x[1]) return a.x[2] < b.x[2];
        if (a.x[0] == b.x[0]) return a.x[1] < b.x[1];
        return a.x[0] < b.x[0];
      });

    thrust::host_vector<Result> h_results(d_results.begin(), d_results.begin() + h_pnts[0]);

    for (int i = 0; i < h_pnts[0]; i++) {
        fprintf(fptr, "%lf\t", h_results[i].x[0]);
        fprintf(fptr, "%lf\t", h_results[i].x[1]);
        fprintf(fptr, "%lf\t", h_results[i].x[2]);
        fprintf(fptr, "%lf\t", h_results[i].x[3]);
        fprintf(fptr, "%lf\t", h_results[i].x[4]);
        fprintf(fptr, "%lf\t", h_results[i].x[5]);
        fprintf(fptr, "%lf\t", h_results[i].x[6]);
        fprintf(fptr, "%lf\t", h_results[i].x[7]);
        fprintf(fptr, "%lf\t", h_results[i].x[8]);
        fprintf(fptr, "%lf\n", h_results[i].x[9]);
    }

    fclose(fptr);
    printf("result pnts: %d\n", h_pnts[0]);

    d_pnts.clear();
    d_results.clear();
    h_results.clear();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_kk);
}