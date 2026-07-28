// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <dlfcn.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"
#include "libhrx/cts/core/amdgpu_hip_printf_test_data.hpp"

namespace {

const char* CandidateLibPath() {
  if (const char* environment_path = std::getenv("HRX_TEST_LIBAMDHIP64");
      environment_path && *environment_path != '\0') {
    return environment_path;
  }
  return "libamdhip64.so";
}

template <typename T>
T ResolveHipSymbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipGetDevicePropertiesFn = hipError_t (*)(hipDeviceProp_t* properties,
                                                int device);
using HipModuleLoadDataFn = hipError_t (*)(hipModule_t* module,
                                           const void* image);
using HipModuleUnloadFn = hipError_t (*)(hipModule_t module);
using HipModuleGetFunctionFn = hipError_t (*)(hipFunction_t* function,
                                              hipModule_t module,
                                              const char* name);
using HipModuleLaunchKernelFn = hipError_t (*)(
    hipFunction_t function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_memory_bytes,
    hipStream_t stream, void** arguments, void** extra);
using HipDeviceSynchronizeFn = hipError_t (*)(void);
using HipMallocFn = hipError_t (*)(hipDeviceptr_t* pointer, size_t size);
using HipFreeFn = hipError_t (*)(hipDeviceptr_t pointer);
using HipMemcpyFn = hipError_t (*)(void* target, const void* source,
                                   size_t size, hipMemcpyKind kind);

class HipBlockingPrintfStressTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (hrx_cts_amdgpu_hip_printf_test_kernels_size() == 0) {
      GTEST_SKIP() << "ROCm device libraries are unavailable";
    }

    library_ = dlopen(CandidateLibPath(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) {
      GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": "
                   << dlerror();
    }

    const auto init = ResolveHipSymbol<HipInitFn>(library_, "hipInit");
    const auto get_device =
        ResolveHipSymbol<HipGetDeviceFn>(library_, "hipGetDevice");
    const auto get_device_properties =
        ResolveHipSymbol<HipGetDevicePropertiesFn>(library_,
                                                   "hipGetDeviceProperties");
    module_load_data_ =
        ResolveHipSymbol<HipModuleLoadDataFn>(library_, "hipModuleLoadData");
    module_unload_ =
        ResolveHipSymbol<HipModuleUnloadFn>(library_, "hipModuleUnload");
    module_get_function_ = ResolveHipSymbol<HipModuleGetFunctionFn>(
        library_, "hipModuleGetFunction");
    module_launch_kernel_ = ResolveHipSymbol<HipModuleLaunchKernelFn>(
        library_, "hipModuleLaunchKernel");
    device_synchronize_ = ResolveHipSymbol<HipDeviceSynchronizeFn>(
        library_, "hipDeviceSynchronize");
    hip_malloc_ = ResolveHipSymbol<HipMallocFn>(library_, "hipMalloc");
    hip_free_ = ResolveHipSymbol<HipFreeFn>(library_, "hipFree");
    hip_memcpy_ = ResolveHipSymbol<HipMemcpyFn>(library_, "hipMemcpy");

    ASSERT_NE(nullptr, init);
    ASSERT_NE(nullptr, get_device);
    ASSERT_NE(nullptr, get_device_properties);
    ASSERT_NE(nullptr, module_load_data_);
    ASSERT_NE(nullptr, module_unload_);
    ASSERT_NE(nullptr, module_get_function_);
    ASSERT_NE(nullptr, module_launch_kernel_);
    ASSERT_NE(nullptr, device_synchronize_);
    ASSERT_NE(nullptr, hip_malloc_);
    ASSERT_NE(nullptr, hip_free_);
    ASSERT_NE(nullptr, hip_memcpy_);

    const hipError_t init_result = init(/*flags=*/0);
    if (init_result != hipSuccess) {
      GTEST_SKIP() << "hipInit failed: " << init_result;
    }
    int device = 0;
    ASSERT_EQ(hipSuccess, get_device(&device));
    hipDeviceProp_t properties = {};
    ASSERT_EQ(hipSuccess, get_device_properties(&properties, device));

    const hrx_cts::AmdgpuHipPrintfTestImage test_image =
        hrx_cts::FindAmdgpuHipPrintfTestImage(properties.gcnArchName);
    ASSERT_NE(nullptr, test_image.file)
        << "no embedded HIP printf HSACO for " << properties.gcnArchName;

    std::vector<uint8_t> image(test_image.file->data,
                               test_image.file->data + test_image.file->size);
    ASSERT_EQ(hipSuccess, module_load_data_(&module_, image.data()));
    std::fill(image.begin(), image.end(), uint8_t{0xA5});
    std::vector<uint8_t>().swap(image);
  }

  void TearDown() override {
    if (output_) {
      EXPECT_EQ(hipSuccess, hip_free_(output_));
    }
    if (module_) {
      EXPECT_EQ(hipSuccess, module_unload_(module_));
    }
  }

  // Dynamically loaded HIP compatibility library.
  void* library_ = nullptr;
  // Module loader entry point.
  HipModuleLoadDataFn module_load_data_ = nullptr;
  // Module release entry point.
  HipModuleUnloadFn module_unload_ = nullptr;
  // Kernel lookup entry point.
  HipModuleGetFunctionFn module_get_function_ = nullptr;
  // Kernel launch entry point.
  HipModuleLaunchKernelFn module_launch_kernel_ = nullptr;
  // Device synchronization entry point.
  HipDeviceSynchronizeFn device_synchronize_ = nullptr;
  // Device allocation entry point.
  HipMallocFn hip_malloc_ = nullptr;
  // Device allocation release entry point.
  HipFreeFn hip_free_ = nullptr;
  // Host-device transfer entry point.
  HipMemcpyFn hip_memcpy_ = nullptr;
  // Loaded HIP printf fixture module.
  hipModule_t module_ = nullptr;
  // Device result allocation owned by the active test.
  hipDeviceptr_t output_ = nullptr;
};

TEST_F(HipBlockingPrintfStressTest, FormatsOutputLargerThanOneMiB) {
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess,
            module_get_function_(&function, module_,
                                 "hrx_blocking_printf_large_output"));
  ASSERT_EQ(hipSuccess, hip_malloc_(&output_, sizeof(int)));

  hipDeviceptr_t result = output_;
  void* arguments[] = {&result};
  ::testing::internal::CaptureStdout();
  const hipError_t launch_result = module_launch_kernel_(
      function, 1, 1, 1, 1, 1, 1,
      /*shared_memory_bytes=*/0, /*stream=*/nullptr, arguments,
      /*extra=*/nullptr);
  const hipError_t synchronize_result =
      launch_result == hipSuccess ? device_synchronize_() : launch_result;
  std::fflush(stdout);
  const std::string output = ::testing::internal::GetCapturedStdout();
  ASSERT_EQ(hipSuccess, launch_result);
  ASSERT_EQ(hipSuccess, synchronize_result);

  constexpr int kExpectedOutputLength = 1024 * 1024 + 1;
  ASSERT_EQ(kExpectedOutputLength, output.size());
  EXPECT_TRUE(std::all_of(output.begin(), output.end() - 1,
                          [](char value) { return value == ' '; }));
  EXPECT_EQ('7', output.back());

  int printf_result = 0;
  ASSERT_EQ(hipSuccess,
            hip_memcpy_(&printf_result, output_, sizeof(printf_result),
                        hipMemcpyDeviceToHost));
  EXPECT_EQ(kExpectedOutputLength, printf_result);
}

TEST_F(HipBlockingPrintfStressTest, RecyclesPacketsAcrossManyWorkitems) {
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess,
            module_get_function_(&function, module_,
                                 "hrx_blocking_printf_many_workitems"));

  constexpr uint32_t kWorkitemCount = 128 * 1024;
  constexpr uint32_t kBlockSize = 256;
  static_assert(kWorkitemCount % kBlockSize == 0);
  ASSERT_EQ(hipSuccess, hip_malloc_(&output_, kWorkitemCount * sizeof(int)));
  uint32_t workgroup_size = kBlockSize;
  uint32_t workitem_count = kWorkitemCount;
  hipDeviceptr_t result = output_;
  void* arguments[] = {&workgroup_size, &workitem_count, &result};

  ::testing::internal::CaptureStdout();
  const hipError_t launch_result = module_launch_kernel_(
      function, kWorkitemCount / kBlockSize, 1, 1, kBlockSize, 1, 1,
      /*shared_memory_bytes=*/0, /*stream=*/nullptr, arguments,
      /*extra=*/nullptr);
  const hipError_t synchronize_result =
      launch_result == hipSuccess ? device_synchronize_() : launch_result;
  std::fflush(stdout);
  const std::string output = ::testing::internal::GetCapturedStdout();
  ASSERT_EQ(hipSuccess, launch_result);
  ASSERT_EQ(hipSuccess, synchronize_result);
  EXPECT_EQ(kWorkitemCount, output.size());
  EXPECT_TRUE(std::all_of(output.begin(), output.end(),
                          [](char value) { return value == '7'; }));

  std::vector<int> printf_results(kWorkitemCount);
  ASSERT_EQ(hipSuccess, hip_memcpy_(printf_results.data(), output_,
                                    printf_results.size() * sizeof(int),
                                    hipMemcpyDeviceToHost));
  EXPECT_TRUE(std::all_of(printf_results.begin(), printf_results.end(),
                          [](int value) { return value == 1; }));
}

}  // namespace
