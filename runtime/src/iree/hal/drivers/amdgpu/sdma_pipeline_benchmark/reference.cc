// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define H(x)                        \
  do {                              \
    if ((x) != hipSuccess) abort(); \
  } while (0)
#define R(x)                                    \
  do {                                          \
    if ((x) != rocblas_status_success) abort(); \
  } while (0)
int main() {
  int m = 4096, n = 4096, k = 4096;
  void *a, *b, *c;
  H(hipMalloc(&a, (size_t)m * k * 2));
  H(hipMalloc(&b, (size_t)n * k * 2));
  H(hipMalloc(&c, (size_t)m * n * 4));
  H(hipMemset(a, 0, (size_t)m * k * 2));
  H(hipMemset(b, 0, (size_t)n * k * 2));
  rocblas_handle handle;
  R(rocblas_create_handle(&handle));
  float alpha = 1, beta = 0;
  hipEvent_t start, end;
  H(hipEventCreate(&start));
  H(hipEventCreate(&end));
  std::vector<float> times;
  for (int i = 0; i < 15; i++) {
    H(hipEventRecord(start));
    R(rocblas_gemm_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none, n, m, k,
        &alpha, b, rocblas_datatype_f16_r, k, a, rocblas_datatype_f16_r, k,
        &beta, c, rocblas_datatype_f32_r, n, c, rocblas_datatype_f32_r, n,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
    H(hipEventRecord(end));
    H(hipEventSynchronize(end));
    float ms;
    H(hipEventElapsedTime(&ms, start, end));
    if (i >= 5) times.push_back(ms);
  }
  std::sort(times.begin(), times.end());
  printf("rocblas_4096_median_ms=%g tflops=%g\n", times[5],
         2. * m * n * k / (times[5] * 1e9));
  R(rocblas_destroy_handle(handle));
  H(hipEventDestroy(start));
  H(hipEventDestroy(end));
  H(hipFree(a));
  H(hipFree(b));
  H(hipFree(c));
}
