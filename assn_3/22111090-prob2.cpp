// Compile: g++ -O2 -fopenmp -o prob2 22111090-prob2.cpp

#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <cstring>

using std::cout;
using std::endl;

#define N (1 << 24)

// Number of array elements a task will process
#define GRANULARITY (1 << 10)

uint64_t reference_sum(uint32_t *A)
{
  uint64_t seq_sum = 0;
  for (int i = 0; i < N; i++)
  {
    seq_sum += A[i];
  }
  return seq_sum;
}

uint64_t par_sum_omp_nored_v1(uint32_t *A)
{
  // SB: Write your OpenMP code here

  uint64_t seq_sum = 0;

  #pragma omp parallel for
  for (int i = 0; i < N; i++)
  { 
    #pragma omp atomic
    seq_sum += A[i];
  }

  return seq_sum;
}

uint64_t par_sum_omp_nored_v2(uint32_t *A)
{
  // SB: Write your OpenMP code here

  uint64_t seq_sum = 0;
  uint64_t sum[4][8];
  memset(sum, 0, sizeof(sum));

  #pragma omp parallel num_threads(4)
  {
    int tid = omp_get_thread_num();

    #pragma omp for
    for (int i = 0; i < N; i++)
    {
      sum[tid][0] += A[i];
    }
  }

  for (int i = 0; i < 4; i++)
    seq_sum += sum[i][0];

  return seq_sum;
}

uint64_t par_sum_omp_red(uint32_t *A)
{
  // SB: Write your OpenMP code here
  uint64_t seq_sum = 0;

  #pragma omp parallel for num_threads(4) reduction(+ : seq_sum)
  for (int i = 0; i < N; i++)
  {
    seq_sum += A[i];
  }

  return seq_sum;
}

uint64_t sum_array(uint32_t *A, int start, int end) {
  uint64_t loc_sum = 0;
  for (int i = start; i < end; i++)
    loc_sum += A[i];

  return loc_sum;
}

uint64_t par_sum_omp_tasks(uint32_t *A)
{
  uint64_t seq_sum = 0;
  uint64_t loc_sums[4][8];
  memset(loc_sums, 0, sizeof(loc_sums));

  #pragma omp parallel num_threads(4)
  {
    #pragma omp for schedule(static)
    for (int i = 0; i < N; i += GRANULARITY)
    {

      int start = i, end = (i + GRANULARITY < N)? i + GRANULARITY : N;

      #pragma omp task shared(loc_sums)
      {
        loc_sums[omp_get_thread_num()][0] += sum_array(A, start, end);
      }
    }
  }

  for (int i = 0; i < 4; i++)
    seq_sum += loc_sums[i][0];

  return seq_sum;
}

int main()
{
  uint32_t *x = new uint32_t[N];
  for (int i = 0; i < N; i++)
  {
    x[i] = i;
  }

  double start_time, end_time, pi;

  start_time = omp_get_wtime();
  uint64_t seq_sum = reference_sum(x);
  end_time = omp_get_wtime();
  cout << "Sequential sum: " << seq_sum << " in " << (end_time - start_time) << " seconds\n";

  start_time = omp_get_wtime();
  uint64_t par_sum_1 = par_sum_omp_nored_v1(x);
  end_time = omp_get_wtime();
  assert(seq_sum == par_sum_1);
  cout << "Parallel sum (thread-local, atomic): " << par_sum_1 << " in " << (end_time - start_time)
       << " seconds\n";

  start_time = omp_get_wtime();
  uint64_t par_sum_2 = par_sum_omp_nored_v2(x);
  end_time = omp_get_wtime();
  assert(seq_sum == par_sum_2);
  cout << "Parallel sum (thread-local, atomic): " << par_sum_2 << " in " << (end_time - start_time)
       << " seconds\n";

  start_time = omp_get_wtime();
  uint64_t ws_sum = par_sum_omp_red(x);
  end_time = omp_get_wtime();
  assert(seq_sum == ws_sum);
  cout << "Parallel sum (worksharing construct): " << ws_sum << " in " << (end_time - start_time)
       << " seconds\n";

  start_time = omp_get_wtime();
  uint64_t task_sum = par_sum_omp_tasks(x);
  end_time = omp_get_wtime();
  if (seq_sum != task_sum)
  {
    cout << "Seq sum: " << seq_sum << " Task sum: " << task_sum << "\n";
  }
  assert(seq_sum == task_sum);
  cout << "Parallel sum (OpenMP tasks): " << task_sum << " in " << (end_time - start_time)
       << " seconds\n";

  return EXIT_SUCCESS;
}
