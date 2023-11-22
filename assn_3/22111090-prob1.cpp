// Compile: g++ -O2 -mavx -mavx2 -fopenmp -march=native -o prob1 22111090-prob1.cpp

#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <immintrin.h>

using std::cout;
using std::endl;
using std::ios;

const int N = (1 << 13);
const int Niter = 10;
const double THRESHOLD = 0.000001;
const int B = 16;

double rtclock()
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0)
  {
    cout << "Error return from gettimeofday: " << stat << endl;
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void reference(double **A, const double *x, double *y_ref, double *z_ref)
{
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      y_ref[j] = y_ref[j] + A[i][j] * x[i];
      z_ref[j] = z_ref[j] + A[j][i] * x[i];
    }
  }
}

void check_result(const double *w_ref, const double *w_opt)
{
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++)
  {
    this_diff = w_ref[i] - w_opt[i];
    if (fabs(this_diff) > THRESHOLD)
    {
      numdiffs++;
      if (this_diff > maxdiff)
        maxdiff = this_diff;
    }
  }

  if (numdiffs > 0)
  {
    cout << numdiffs << " Diffs found over threshold " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  }
  else
  {
    cout << "No differences found between base and test versions\n";
  }
}

// TODO: INITIALLY IDENTICAL TO REFERENCE; MAKE YOUR CHANGES TO OPTIMIZE THE CODE
// You can create multiple versions of the optimized() function to test your changes
void optimized(double **__restrict__ A, const double *__restrict__ x, double *__restrict__ y_opt, double *__restrict__ z_opt)
{

  for (int i_T = 0; i_T < N; i_T += B) {
    for (int j_T = 0; j_T < N; j_T += B) {
      for (int i = 0; i < B; i++) {
        #pragma ivdep
        for (int j = 0; j < B; j++) {
          y_opt[j_T+j] = y_opt[j_T+j] + A[i_T+i][j_T+j] * x[i_T+i];
        }
      }
    }
  }

  for (int j_T = 0; j_T < N; j_T += B) {
    for (int i_T = 0; i_T < N; i_T += B) {
      for (int j = 0; j < B; j++) {
        #pragma ivdep
        for (int i = 0; i < B; i++) {
          z_opt[j_T+j] = z_opt[j_T+j] + A[j_T+j][i_T+i] * x[i_T+i];
        }
      }
    }
  }
}

void avx_version(double **__restrict__ A, double *__restrict__ x, double *__restrict__ y_opt, double *__restrict__ z_opt)
{
  __m256d rA, rB, rC;

  for (int i_T = 0; i_T < N; i_T += B) {
    for (int j_T = 0; j_T < N; j_T += B) {
      for (int i = 0; i < B; i++) {
        for (int j = 0; j < B; j += 4) {
          rA = _mm256_loadu_pd(&y_opt[j+j_T]);
          rB = _mm256_loadu_pd(&A[i+i_T][j+j_T]);
          rC = _mm256_broadcast_sd(&x[i+i_T]);

          rC = _mm256_add_pd(rA, _mm256_mul_pd(rB, rC));

          _mm256_storeu_pd(&y_opt[j+j_T], rC);
        }
      }
    }
  }

  for (int j_T = 0; j_T < N; j_T += B) {
    for (int i_T = 0; i_T < N; i_T += B) {
      for (int j = 0; j < B; j++) {
        for (int i = 0; i < B; i += 4) {
          rA = _mm256_loadu_pd(&z_opt[j+j_T]);
          rB = _mm256_loadu_pd(&A[j+j_T][i+i_T]);
          rC = _mm256_broadcast_sd(&x[i+i_T]);

          rC = _mm256_add_pd(rA, _mm256_mul_pd(rB, rC));

          _mm256_storeu_pd(&z_opt[j+j_T], rC);
        }
      }
    }
  }
}

void avx_version_2(double **__restrict__ A, double *__restrict__ x, double *__restrict__ y_opt, double *__restrict__ z_opt)
{
  __m256d rA, rB, rC;

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j += 4) {
        rA = _mm256_loadu_pd(&y_opt[j]);
        rB = _mm256_loadu_pd(&A[i][j]);
        rC = _mm256_broadcast_sd(&x[i]);

        rC = _mm256_add_pd(rA, _mm256_mul_pd(rB, rC));

        _mm256_storeu_pd(&y_opt[j], rC);
      }
    }

    for (int j = 0; j < N; j++) {
      for (int i = 0; i < N; i += 4) {
        rA = _mm256_loadu_pd(&z_opt[j]);
        rB = _mm256_loadu_pd(&A[j][i]);
        rC = _mm256_broadcast_sd(&x[i]);

        rC = _mm256_add_pd(rA, _mm256_mul_pd(rB, rC));

        _mm256_storeu_pd(&z_opt[j], rC);
      }
    }
}

int main()
{
  double clkbegin, clkend;
  double t;

  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(5);

  double **A;
  A = new double *[N];
  for (int i = 0; i < N; i++)
  {
    A[i] = new double[N];
  }

  double *x, *y_ref, *z_ref, *y_opt, *z_opt;
  x = new double[N];
  y_ref = new double[N];
  z_ref = new double[N];
  y_opt = new double[N];
  z_opt = new double[N];

  for (int i = 0; i < N; i++)
  {
    x[i] = i;
    y_ref[i] = 1.0;
    y_opt[i] = 1.0;
    z_ref[i] = 2.0;
    z_opt[i] = 2.0;
    for (int j = 0; j < N; j++)
    {
      A[i][j] = (i + 2.0 * j) / (2.0 * N);
    }
  }

  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++)
  {
    reference(A, x, y_ref, z_ref);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Reference Version: Matrix Size = " << N << ", " << 4.0 * 1e-9 * N * N * Niter / t
       << " GFLOPS; Time = " << t / Niter << " sec\n";

  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++)
  {
    optimized(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);

  // Reset
  for (int i = 0; i < N; i++)
  {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // Another optimized version possibly

  // Version with intinsics

  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++)
  {
    avx_version(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Intrinsics Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);

  // Reset
  for (int i = 0; i < N; i++)
  {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++)
  {
    avx_version_2(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Intrinsics Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);

  return EXIT_SUCCESS;
}
