// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <dlfcn.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

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
using HipGetDevicePropertiesR0600Fn =
    hipError_t (*)(hipDeviceProp_t* properties, int device);
using HipModuleLoadDataFn = hipError_t (*)(hipModule_t* module,
                                           const void* image);
using HipModuleUnloadFn = hipError_t (*)(hipModule_t module);
using HipModuleGetFunctionFn = hipError_t (*)(hipFunction_t* function,
                                              hipModule_t module,
                                              const char* name);
using HipModuleOccupancyMaxActiveBlocksFn =
    hipError_t (*)(int* block_count, hipFunction_t function, int block_size,
                   size_t dynamic_shared_memory_size);
using HipModuleOccupancyMaxActiveBlocksWithFlagsFn =
    hipError_t (*)(int* block_count, hipFunction_t function, int block_size,
                   size_t dynamic_shared_memory_size, unsigned int flags);
using HipModuleOccupancyMaxPotentialBlockSizeFn =
    hipError_t (*)(int* grid_size, int* block_size, hipFunction_t function,
                   size_t dynamic_shared_memory_size, int block_size_limit);
using HipOccupancyMaxActiveBlocksFn =
    hipError_t (*)(int* block_count, const void* function, int block_size,
                   size_t dynamic_shared_memory_size);
using HipOccupancyMaxPotentialBlockSizeFn =
    hipError_t (*)(int* grid_size, int* block_size, const void* function,
                   size_t dynamic_shared_memory_size, int block_size_limit);
using HipOccupancyAvailableDynamicMemoryFn =
    hipError_t (*)(size_t* dynamic_shared_memory_size, const void* function,
                   int block_count, int block_size);
using HipRegisterFatBinaryFn = void** (*)(const void* image);
using HipUnregisterFatBinaryFn = void (*)(void** registration);
// The compiler ABI names these pointer-only metadata slots `uint3`. The local
// binding's `dim3` has the same three-unsigned-component representation and the
// CTS always passes null for both slots.
using HipCompilerIndex = dim3;
using HipRegisterFunctionFn = void (*)(
    void** registration, const void* host_function, char* device_function,
    const char* device_name, unsigned int thread_limit,
    HipCompilerIndex* thread_index, HipCompilerIndex* block_index,
    dim3* block_dimensions, dim3* grid_dimensions, int* shared_memory_size);

struct ScopedModule {
  ~ScopedModule() {
    if (value) {
      const hipError_t result = unload(value);
      EXPECT_EQ(hipSuccess, result);
    }
  }

  // Loaded module released at scope exit.
  hipModule_t value = nullptr;
  // Module release entry point from the binding under test.
  HipModuleUnloadFn unload = nullptr;
};

struct ScopedRegistration {
  ~ScopedRegistration() {
    if (value) {
      unregister(value);
    }
  }

  // Compiler-style fat-binary registration released at scope exit.
  void** value = nullptr;
  // Registration release entry point from the binding under test.
  HipUnregisterFatBinaryFn unregister = nullptr;
};

void OccupancyHostStub() {}

TEST(HipOccupancyTest, LoadedAndRegisteredFunctionsUseExactQueueOccupancy) {
  void* library = dlopen(CandidateLibPath(), RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": " << dlerror();
  }

  const auto init = ResolveHipSymbol<HipInitFn>(library, "hipInit");
  const auto get_device =
      ResolveHipSymbol<HipGetDeviceFn>(library, "hipGetDevice");
  const auto get_device_properties =
      ResolveHipSymbol<HipGetDevicePropertiesR0600Fn>(
          library, "hipGetDevicePropertiesR0600");
  const auto module_load_data =
      ResolveHipSymbol<HipModuleLoadDataFn>(library, "hipModuleLoadData");
  const auto module_unload =
      ResolveHipSymbol<HipModuleUnloadFn>(library, "hipModuleUnload");
  const auto module_get_function =
      ResolveHipSymbol<HipModuleGetFunctionFn>(library, "hipModuleGetFunction");
  const auto module_max_active =
      ResolveHipSymbol<HipModuleOccupancyMaxActiveBlocksFn>(
          library, "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
  const auto module_max_active_with_flags =
      ResolveHipSymbol<HipModuleOccupancyMaxActiveBlocksWithFlagsFn>(
          library,
          "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
  const auto module_max_potential =
      ResolveHipSymbol<HipModuleOccupancyMaxPotentialBlockSizeFn>(
          library, "hipModuleOccupancyMaxPotentialBlockSize");
  const auto runtime_max_active =
      ResolveHipSymbol<HipOccupancyMaxActiveBlocksFn>(
          library, "hipOccupancyMaxActiveBlocksPerMultiprocessor");
  const auto runtime_max_potential =
      ResolveHipSymbol<HipOccupancyMaxPotentialBlockSizeFn>(
          library, "hipOccupancyMaxPotentialBlockSize");
  const auto available_dynamic_memory =
      ResolveHipSymbol<HipOccupancyAvailableDynamicMemoryFn>(
          library, "hipOccupancyAvailableDynamicSMemPerBlock");
  const auto register_fat_binary = ResolveHipSymbol<HipRegisterFatBinaryFn>(
      library, "__hipRegisterFatBinary");
  const auto unregister_fat_binary = ResolveHipSymbol<HipUnregisterFatBinaryFn>(
      library, "__hipUnregisterFatBinary");
  const auto register_function =
      ResolveHipSymbol<HipRegisterFunctionFn>(library, "__hipRegisterFunction");

  ASSERT_NE(nullptr, init);
  ASSERT_NE(nullptr, get_device);
  ASSERT_NE(nullptr, get_device_properties);
  ASSERT_NE(nullptr, module_load_data);
  ASSERT_NE(nullptr, module_unload);
  ASSERT_NE(nullptr, module_get_function);
  ASSERT_NE(nullptr, module_max_active);
  ASSERT_NE(nullptr, module_max_active_with_flags);
  ASSERT_NE(nullptr, module_max_potential);
  ASSERT_NE(nullptr, runtime_max_active);
  ASSERT_NE(nullptr, runtime_max_potential);
  ASSERT_NE(nullptr, available_dynamic_memory);
  ASSERT_NE(nullptr, register_fat_binary);
  ASSERT_NE(nullptr, unregister_fat_binary);
  ASSERT_NE(nullptr, register_function);

  const hipError_t init_result = init(/*flags=*/0);
  if (init_result != hipSuccess) {
    GTEST_SKIP() << "hipInit failed: " << init_result;
  }
  int device = 0;
  ASSERT_EQ(hipSuccess, get_device(&device));
  hipDeviceProp_t properties = {};
  ASSERT_EQ(hipSuccess, get_device_properties(&properties, device));
  ASSERT_GT(properties.maxThreadsPerBlock, 0);

  const hrx_cts::AmdgpuExecutableTestImage test_image =
      hrx_cts::FindAmdgpuExecutableTestImage(properties.gcnArchName);
  ASSERT_NE(nullptr, test_image.file)
      << "no embedded HSACO for " << properties.gcnArchName;

  ScopedModule module = {/*.value=*/nullptr, /*.unload=*/module_unload};
  ASSERT_EQ(hipSuccess, module_load_data(&module.value, test_image.file->data));
  hipFunction_t module_function = nullptr;
  ASSERT_EQ(hipSuccess,
            module_get_function(&module_function, module.value, "hrx_noop"));

  const int probe_block_size = std::min(64, properties.maxThreadsPerBlock);
  int module_active_blocks = 0;
  ASSERT_EQ(hipSuccess, module_max_active(&module_active_blocks,
                                          module_function, probe_block_size,
                                          /*dynamic_shared_memory_size=*/0));
  ASSERT_GT(module_active_blocks, 0);
  int flagged_active_blocks = 0;
  ASSERT_EQ(hipSuccess,
            module_max_active_with_flags(&flagged_active_blocks,
                                         module_function, probe_block_size,
                                         /*dynamic_shared_memory_size=*/0,
                                         hipOccupancyDisableCachingOverride));
  EXPECT_EQ(module_active_blocks, flagged_active_blocks);

  int module_grid_size = 0;
  int module_block_size = 0;
  ASSERT_EQ(hipSuccess,
            module_max_potential(&module_grid_size, &module_block_size,
                                 module_function,
                                 /*dynamic_shared_memory_size=*/0,
                                 /*block_size_limit=*/0));
  ASSERT_GT(module_grid_size, 0);
  ASSERT_GT(module_block_size, 0);
  ASSERT_LE(module_block_size, properties.maxThreadsPerBlock);
  int optimal_active_blocks = 0;
  ASSERT_EQ(hipSuccess, module_max_active(&optimal_active_blocks,
                                          module_function, module_block_size,
                                          /*dynamic_shared_memory_size=*/0));
  ASSERT_GT(optimal_active_blocks, 0);
  EXPECT_EQ(0, module_grid_size % optimal_active_blocks);
  EXPECT_GT(module_grid_size / optimal_active_blocks, 0);

  int unchanged_grid_size = 17;
  int unchanged_block_size = 19;
  EXPECT_EQ(hipErrorInvalidValue,
            module_max_potential(&unchanged_grid_size, &unchanged_block_size,
                                 module_function,
                                 /*dynamic_shared_memory_size=*/0,
                                 /*block_size_limit=*/-1));
  EXPECT_EQ(17, unchanged_grid_size);
  EXPECT_EQ(19, unchanged_block_size);

  ScopedRegistration registration = {
      /*.value=*/register_fat_binary(test_image.file->data),
      /*.unregister=*/unregister_fat_binary,
  };
  ASSERT_NE(nullptr, registration.value);
  char device_function_name[] = "hrx_noop";
  register_function(registration.value,
                    reinterpret_cast<const void*>(&OccupancyHostStub),
                    device_function_name, "hrx_noop", /*thread_limit=*/0,
                    /*thread_index=*/nullptr, /*block_index=*/nullptr,
                    /*block_dimensions=*/nullptr, /*grid_dimensions=*/nullptr,
                    /*shared_memory_size=*/nullptr);
  const void* registered_function =
      reinterpret_cast<const void*>(&OccupancyHostStub);

  int runtime_active_blocks = 0;
  ASSERT_EQ(hipSuccess,
            runtime_max_active(&runtime_active_blocks, registered_function,
                               probe_block_size,
                               /*dynamic_shared_memory_size=*/0));
  EXPECT_EQ(module_active_blocks, runtime_active_blocks);

  int runtime_grid_size = 0;
  int runtime_block_size = 0;
  ASSERT_EQ(hipSuccess,
            runtime_max_potential(&runtime_grid_size, &runtime_block_size,
                                  registered_function,
                                  /*dynamic_shared_memory_size=*/0,
                                  /*block_size_limit=*/0));
  EXPECT_EQ(module_grid_size, runtime_grid_size);
  EXPECT_EQ(module_block_size, runtime_block_size);

  size_t dynamic_shared_memory_size = 0;
  ASSERT_EQ(hipSuccess, available_dynamic_memory(
                            &dynamic_shared_memory_size, registered_function,
                            runtime_active_blocks, probe_block_size));
  int active_blocks_at_boundary = 0;
  ASSERT_EQ(hipSuccess,
            runtime_max_active(&active_blocks_at_boundary, registered_function,
                               probe_block_size, dynamic_shared_memory_size));
  EXPECT_GE(active_blocks_at_boundary, runtime_active_blocks);
  if (dynamic_shared_memory_size < UINT32_MAX) {
    int active_blocks_beyond_boundary = 0;
    ASSERT_EQ(
        hipSuccess,
        runtime_max_active(&active_blocks_beyond_boundary, registered_function,
                           probe_block_size, dynamic_shared_memory_size + 1));
    EXPECT_LT(active_blocks_beyond_boundary, runtime_active_blocks);
  }
}

}  // namespace
