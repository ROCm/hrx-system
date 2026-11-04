// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "runtime.h"

#include "driver.h"
#include "iree/testing/gtest.h"

namespace {

CUresult context_synchronize_result = CUDA_SUCCESS;
int context_synchronize_count = 0;

TEST(CudaRuntimeTest, DeviceSynchronizeDelegatesAndPublishesLastError) {
  context_synchronize_count = 0;
  EXPECT_EQ(cudaSuccess, cudaGetLastError());

  context_synchronize_result = CUDA_SUCCESS;
  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
  EXPECT_EQ(1, context_synchronize_count);
  EXPECT_EQ(cudaSuccess, cudaPeekAtLastError());

  context_synchronize_result = CUDA_ERROR_INVALID_CONTEXT;
  EXPECT_EQ(cudaErrorIncompatibleDriverContext, cudaDeviceSynchronize());
  EXPECT_EQ(2, context_synchronize_count);
  EXPECT_EQ(cudaErrorIncompatibleDriverContext, cudaPeekAtLastError());
  EXPECT_EQ(cudaErrorIncompatibleDriverContext, cudaGetLastError());
  EXPECT_EQ(cudaSuccess, cudaPeekAtLastError());
}

}  // namespace

extern "C" CUresult CUDAAPI cuCtxSynchronize(void) {
  ++context_synchronize_count;
  return context_synchronize_result;
}

extern "C" CUresult CUDAAPI cuMemGetInfo(size_t* free, size_t* total) {
  (void)free;
  (void)total;
  return CUDA_ERROR_NOT_SUPPORTED;
}
