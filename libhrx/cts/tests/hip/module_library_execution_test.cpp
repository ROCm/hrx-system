// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/temp_file.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"
#include "libhrx/cts/core/amdgpu_hip_module_library_test_data.hpp"

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
using HipHalDeinitFn = hipError_t (*)(void);
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipSetDeviceFn = hipError_t (*)(int device);
using HipGetDeviceCountFn = hipError_t (*)(int* device_count);
using HipGetDevicePropertiesFn = hipError_t (*)(hipDeviceProp_t* properties,
                                                int device);
using HipDeviceSynchronizeFn = hipError_t (*)(void);
using HipMallocFn = hipError_t (*)(hipDeviceptr_t* pointer, size_t size);
using HipFreeFn = hipError_t (*)(hipDeviceptr_t pointer);
using HipMemcpyFn = hipError_t (*)(void* target, const void* source,
                                   size_t size, hipMemcpyKind kind);
using HipStreamCreateFn = hipError_t (*)(hipStream_t* stream);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t stream);
using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t stream);
using HipEventCreateFn = hipError_t (*)(hipEvent_t* event);
using HipEventDestroyFn = hipError_t (*)(hipEvent_t event);
using HipStreamBeginCaptureFn = hipError_t (*)(hipStream_t stream,
                                               hipStreamCaptureMode mode);
using HipStreamEndCaptureFn = hipError_t (*)(hipStream_t stream,
                                             hipGraph_t* graph);
using HipGraphGetNodesFn = hipError_t (*)(hipGraph_t graph,
                                          hipGraphNode_t* nodes,
                                          size_t* node_count);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t graph);

using HipModuleLoadDataExFn = hipError_t (*)(hipModule_t* module,
                                             const void* image,
                                             unsigned int option_count,
                                             hipJitOption* options,
                                             void** option_values);
using HipModuleLoadFatBinaryFn = hipError_t (*)(hipModule_t* module,
                                                const void* image);
using HipModuleUnloadFn = hipError_t (*)(hipModule_t module);
using HipModuleGetFunctionFn = hipError_t (*)(hipFunction_t* function,
                                              hipModule_t module,
                                              const char* name);
using HipModuleGetGlobalFn = hipError_t (*)(hipDeviceptr_t* pointer,
                                            size_t* size, hipModule_t module,
                                            const char* name);
using HipModuleGetTexRefFn = hipError_t (*)(void** texture_reference,
                                            hipModule_t module,
                                            const char* name);
using HipModuleGetFunctionCountFn = hipError_t (*)(unsigned int* count,
                                                   hipModule_t module);
using HipModuleLaunchKernelFn = hipError_t (*)(
    hipFunction_t function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_memory_bytes,
    hipStream_t stream, void** arguments, void** extra);
using HipExtModuleLaunchKernelFn = hipError_t (*)(
    hipFunction_t function, unsigned int global_size_x,
    unsigned int global_size_y, unsigned int global_size_z,
    unsigned int local_size_x, unsigned int local_size_y,
    unsigned int local_size_z, size_t shared_memory_bytes, hipStream_t stream,
    void** arguments, void** extra, hipEvent_t start_event,
    hipEvent_t stop_event, uint32_t flags);

using HipLibraryLoadDataFn = hipError_t (*)(
    hipLibrary_t* library, const void* code, hipJitOption* jit_options,
    void** jit_option_values, unsigned int jit_option_count,
    hipLibraryOption* library_options, void** library_option_values,
    unsigned int library_option_count);
using HipLibraryLoadFromFileFn = hipError_t (*)(
    hipLibrary_t* library, const char* file_name, hipJitOption* jit_options,
    void** jit_option_values, unsigned int jit_option_count,
    hipLibraryOption* library_options, void** library_option_values,
    unsigned int library_option_count);
using HipLibraryUnloadFn = hipError_t (*)(hipLibrary_t library);
using HipLibraryGetKernelFn = hipError_t (*)(hipKernel_t* kernel,
                                             hipLibrary_t library,
                                             const char* name);
using HipLibraryGetKernelCountFn = hipError_t (*)(unsigned int* count,
                                                  hipLibrary_t library);
using HipLibraryEnumerateKernelsFn = hipError_t (*)(hipKernel_t* kernels,
                                                    unsigned int kernel_count,
                                                    hipLibrary_t library);
using HipLibraryGetGlobalFn = hipError_t (*)(void** pointer, size_t* size,
                                             hipLibrary_t library,
                                             const char* name);
using HipKernelGetFunctionFn = hipError_t (*)(hipFunction_t* function,
                                              hipKernel_t kernel);
using HipKernelGetLibraryFn = hipError_t (*)(hipLibrary_t* library,
                                             hipKernel_t kernel);
using HipKernelGetNameFn = hipError_t (*)(const char** name,
                                          hipKernel_t kernel);
using HipKernelGetParamInfoFn = hipError_t (*)(hipKernel_t kernel,
                                               size_t parameter_index,
                                               size_t* parameter_offset,
                                               size_t* parameter_size);
using HipKernelGetAttributeFn = hipError_t (*)(int* value,
                                               hipFunction_attribute attribute,
                                               hipKernel_t kernel,
                                               hipDevice_t device);

using HipLinkCreateFn = hipError_t (*)(unsigned int option_count,
                                       hipJitOption* options,
                                       void** option_values,
                                       hipLinkState_t* state);
using HipLinkAddDataFn = hipError_t (*)(hipLinkState_t state,
                                        hipJitInputType type, void* data,
                                        size_t size, const char* name,
                                        unsigned int option_count,
                                        hipJitOption* options,
                                        void** option_values);
using HipLinkAddFileFn = hipError_t (*)(hipLinkState_t state,
                                        hipJitInputType type, const char* path,
                                        unsigned int option_count,
                                        hipJitOption* options,
                                        void** option_values);
using HipLinkCompleteFn = hipError_t (*)(hipLinkState_t state, void** binary,
                                         size_t* size);
using HipLinkDestroyFn = hipError_t (*)(hipLinkState_t state);

using HipGetSymbolAddressFn = hipError_t (*)(void** pointer,
                                             const void* symbol);
using HipGetSymbolSizeFn = hipError_t (*)(size_t* size, const void* symbol);
using HipMemcpyToSymbolFn = hipError_t (*)(const void* symbol,
                                           const void* source, size_t size,
                                           size_t offset, hipMemcpyKind kind);
using HipMemcpyFromSymbolFn = hipError_t (*)(void* target, const void* symbol,
                                             size_t size, size_t offset,
                                             hipMemcpyKind kind);
using HipGetFuncBySymbolFn = hipError_t (*)(hipFunction_t* function,
                                            const void* symbol);
using HipRegisterFatBinaryFn = void** (*)(const void* image);
using HipUnregisterFatBinaryFn = void (*)(void** registration);
using HipCompilerIndex = dim3;
using HipRegisterFunctionFn = void (*)(
    void** registration, const void* host_function, char* device_function,
    const char* device_name, unsigned int thread_limit,
    HipCompilerIndex* thread_index, HipCompilerIndex* block_index,
    dim3* block_dimensions, dim3* grid_dimensions, int* shared_memory_size);
using HipRegisterManagedVarFn = void (*)(void* registration,
                                         void** publication_slot,
                                         void* initial_value,
                                         const char* device_name, size_t size,
                                         unsigned int alignment);

struct HipApi {
  bool Resolve(void* library) {
#define HRX_RESOLVE_HIP_API(field, type, symbol) \
  field = ResolveHipSymbol<type>(library, symbol)
    HRX_RESOLVE_HIP_API(init, HipInitFn, "hipInit");
    HRX_RESOLVE_HIP_API(deinit, HipHalDeinitFn, "hipHALDeinit");
    HRX_RESOLVE_HIP_API(get_device, HipGetDeviceFn, "hipGetDevice");
    HRX_RESOLVE_HIP_API(set_device, HipSetDeviceFn, "hipSetDevice");
    HRX_RESOLVE_HIP_API(get_device_count, HipGetDeviceCountFn,
                        "hipGetDeviceCount");
    HRX_RESOLVE_HIP_API(get_device_properties, HipGetDevicePropertiesFn,
                        "hipGetDeviceProperties");
    HRX_RESOLVE_HIP_API(device_synchronize, HipDeviceSynchronizeFn,
                        "hipDeviceSynchronize");
    HRX_RESOLVE_HIP_API(malloc, HipMallocFn, "hipMalloc");
    HRX_RESOLVE_HIP_API(free, HipFreeFn, "hipFree");
    HRX_RESOLVE_HIP_API(memcpy, HipMemcpyFn, "hipMemcpy");
    HRX_RESOLVE_HIP_API(stream_create, HipStreamCreateFn, "hipStreamCreate");
    HRX_RESOLVE_HIP_API(stream_destroy, HipStreamDestroyFn, "hipStreamDestroy");
    HRX_RESOLVE_HIP_API(stream_synchronize, HipStreamSynchronizeFn,
                        "hipStreamSynchronize");
    HRX_RESOLVE_HIP_API(event_create, HipEventCreateFn, "hipEventCreate");
    HRX_RESOLVE_HIP_API(event_destroy, HipEventDestroyFn, "hipEventDestroy");
    HRX_RESOLVE_HIP_API(stream_begin_capture, HipStreamBeginCaptureFn,
                        "hipStreamBeginCapture");
    HRX_RESOLVE_HIP_API(stream_end_capture, HipStreamEndCaptureFn,
                        "hipStreamEndCapture");
    HRX_RESOLVE_HIP_API(graph_get_nodes, HipGraphGetNodesFn,
                        "hipGraphGetNodes");
    HRX_RESOLVE_HIP_API(graph_destroy, HipGraphDestroyFn, "hipGraphDestroy");
    HRX_RESOLVE_HIP_API(module_load_data_ex, HipModuleLoadDataExFn,
                        "hipModuleLoadDataEx");
    HRX_RESOLVE_HIP_API(module_load_fat_binary, HipModuleLoadFatBinaryFn,
                        "hipModuleLoadFatBinary");
    HRX_RESOLVE_HIP_API(module_unload, HipModuleUnloadFn, "hipModuleUnload");
    HRX_RESOLVE_HIP_API(module_get_function, HipModuleGetFunctionFn,
                        "hipModuleGetFunction");
    HRX_RESOLVE_HIP_API(module_get_global, HipModuleGetGlobalFn,
                        "hipModuleGetGlobal");
    HRX_RESOLVE_HIP_API(module_get_tex_ref, HipModuleGetTexRefFn,
                        "hipModuleGetTexRef");
    HRX_RESOLVE_HIP_API(module_get_function_count, HipModuleGetFunctionCountFn,
                        "hipModuleGetFunctionCount");
    HRX_RESOLVE_HIP_API(module_launch_kernel, HipModuleLaunchKernelFn,
                        "hipModuleLaunchKernel");
    HRX_RESOLVE_HIP_API(ext_module_launch_kernel, HipExtModuleLaunchKernelFn,
                        "hipExtModuleLaunchKernel");
    HRX_RESOLVE_HIP_API(library_load_data, HipLibraryLoadDataFn,
                        "hipLibraryLoadData");
    HRX_RESOLVE_HIP_API(library_load_from_file, HipLibraryLoadFromFileFn,
                        "hipLibraryLoadFromFile");
    HRX_RESOLVE_HIP_API(library_unload, HipLibraryUnloadFn, "hipLibraryUnload");
    HRX_RESOLVE_HIP_API(library_get_kernel, HipLibraryGetKernelFn,
                        "hipLibraryGetKernel");
    HRX_RESOLVE_HIP_API(library_get_kernel_count, HipLibraryGetKernelCountFn,
                        "hipLibraryGetKernelCount");
    HRX_RESOLVE_HIP_API(library_enumerate_kernels, HipLibraryEnumerateKernelsFn,
                        "hipLibraryEnumerateKernels");
    HRX_RESOLVE_HIP_API(library_get_global, HipLibraryGetGlobalFn,
                        "hipLibraryGetGlobal");
    HRX_RESOLVE_HIP_API(kernel_get_function, HipKernelGetFunctionFn,
                        "hipKernelGetFunction");
    HRX_RESOLVE_HIP_API(kernel_get_library, HipKernelGetLibraryFn,
                        "hipKernelGetLibrary");
    HRX_RESOLVE_HIP_API(kernel_get_name, HipKernelGetNameFn,
                        "hipKernelGetName");
    HRX_RESOLVE_HIP_API(kernel_get_param_info, HipKernelGetParamInfoFn,
                        "hipKernelGetParamInfo");
    HRX_RESOLVE_HIP_API(kernel_get_attribute, HipKernelGetAttributeFn,
                        "hipKernelGetAttribute");
    HRX_RESOLVE_HIP_API(link_create, HipLinkCreateFn, "hipLinkCreate");
    HRX_RESOLVE_HIP_API(link_add_data, HipLinkAddDataFn, "hipLinkAddData");
    HRX_RESOLVE_HIP_API(link_add_file, HipLinkAddFileFn, "hipLinkAddFile");
    HRX_RESOLVE_HIP_API(link_complete, HipLinkCompleteFn, "hipLinkComplete");
    HRX_RESOLVE_HIP_API(link_destroy, HipLinkDestroyFn, "hipLinkDestroy");
    HRX_RESOLVE_HIP_API(get_symbol_address, HipGetSymbolAddressFn,
                        "hipGetSymbolAddress");
    HRX_RESOLVE_HIP_API(get_symbol_size, HipGetSymbolSizeFn,
                        "hipGetSymbolSize");
    HRX_RESOLVE_HIP_API(memcpy_to_symbol, HipMemcpyToSymbolFn,
                        "hipMemcpyToSymbol");
    HRX_RESOLVE_HIP_API(memcpy_from_symbol, HipMemcpyFromSymbolFn,
                        "hipMemcpyFromSymbol");
    HRX_RESOLVE_HIP_API(get_func_by_symbol, HipGetFuncBySymbolFn,
                        "hipGetFuncBySymbol");
    HRX_RESOLVE_HIP_API(register_fat_binary, HipRegisterFatBinaryFn,
                        "__hipRegisterFatBinary");
    HRX_RESOLVE_HIP_API(unregister_fat_binary, HipUnregisterFatBinaryFn,
                        "__hipUnregisterFatBinary");
    HRX_RESOLVE_HIP_API(register_function, HipRegisterFunctionFn,
                        "__hipRegisterFunction");
    HRX_RESOLVE_HIP_API(register_managed_variable, HipRegisterManagedVarFn,
                        "__hipRegisterManagedVar");
#undef HRX_RESOLVE_HIP_API
    return init && deinit && get_device && set_device && get_device_count &&
           get_device_properties && device_synchronize && malloc && free &&
           memcpy && stream_create && stream_destroy && stream_synchronize &&
           event_create && event_destroy && stream_begin_capture &&
           stream_end_capture && graph_get_nodes && graph_destroy &&
           module_load_data_ex && module_load_fat_binary && module_unload &&
           module_get_function && module_get_global && module_get_tex_ref &&
           module_get_function_count && module_launch_kernel &&
           ext_module_launch_kernel && library_load_data &&
           library_load_from_file && library_unload && library_get_kernel &&
           library_get_kernel_count && library_enumerate_kernels &&
           library_get_global && kernel_get_function && kernel_get_library &&
           kernel_get_name && kernel_get_param_info && kernel_get_attribute &&
           link_create && link_add_data && link_add_file && link_complete &&
           link_destroy && get_symbol_address && get_symbol_size &&
           memcpy_to_symbol && memcpy_from_symbol && get_func_by_symbol &&
           register_fat_binary && unregister_fat_binary && register_function &&
           register_managed_variable;
  }

  HipInitFn init = nullptr;
  HipHalDeinitFn deinit = nullptr;
  HipGetDeviceFn get_device = nullptr;
  HipSetDeviceFn set_device = nullptr;
  HipGetDeviceCountFn get_device_count = nullptr;
  HipGetDevicePropertiesFn get_device_properties = nullptr;
  HipDeviceSynchronizeFn device_synchronize = nullptr;
  HipMallocFn malloc = nullptr;
  HipFreeFn free = nullptr;
  HipMemcpyFn memcpy = nullptr;
  HipStreamCreateFn stream_create = nullptr;
  HipStreamDestroyFn stream_destroy = nullptr;
  HipStreamSynchronizeFn stream_synchronize = nullptr;
  HipEventCreateFn event_create = nullptr;
  HipEventDestroyFn event_destroy = nullptr;
  HipStreamBeginCaptureFn stream_begin_capture = nullptr;
  HipStreamEndCaptureFn stream_end_capture = nullptr;
  HipGraphGetNodesFn graph_get_nodes = nullptr;
  HipGraphDestroyFn graph_destroy = nullptr;
  HipModuleLoadDataExFn module_load_data_ex = nullptr;
  HipModuleLoadFatBinaryFn module_load_fat_binary = nullptr;
  HipModuleUnloadFn module_unload = nullptr;
  HipModuleGetFunctionFn module_get_function = nullptr;
  HipModuleGetGlobalFn module_get_global = nullptr;
  HipModuleGetTexRefFn module_get_tex_ref = nullptr;
  HipModuleGetFunctionCountFn module_get_function_count = nullptr;
  HipModuleLaunchKernelFn module_launch_kernel = nullptr;
  HipExtModuleLaunchKernelFn ext_module_launch_kernel = nullptr;
  HipLibraryLoadDataFn library_load_data = nullptr;
  HipLibraryLoadFromFileFn library_load_from_file = nullptr;
  HipLibraryUnloadFn library_unload = nullptr;
  HipLibraryGetKernelFn library_get_kernel = nullptr;
  HipLibraryGetKernelCountFn library_get_kernel_count = nullptr;
  HipLibraryEnumerateKernelsFn library_enumerate_kernels = nullptr;
  HipLibraryGetGlobalFn library_get_global = nullptr;
  HipKernelGetFunctionFn kernel_get_function = nullptr;
  HipKernelGetLibraryFn kernel_get_library = nullptr;
  HipKernelGetNameFn kernel_get_name = nullptr;
  HipKernelGetParamInfoFn kernel_get_param_info = nullptr;
  HipKernelGetAttributeFn kernel_get_attribute = nullptr;
  HipLinkCreateFn link_create = nullptr;
  HipLinkAddDataFn link_add_data = nullptr;
  HipLinkAddFileFn link_add_file = nullptr;
  HipLinkCompleteFn link_complete = nullptr;
  HipLinkDestroyFn link_destroy = nullptr;
  HipGetSymbolAddressFn get_symbol_address = nullptr;
  HipGetSymbolSizeFn get_symbol_size = nullptr;
  HipMemcpyToSymbolFn memcpy_to_symbol = nullptr;
  HipMemcpyFromSymbolFn memcpy_from_symbol = nullptr;
  HipGetFuncBySymbolFn get_func_by_symbol = nullptr;
  HipRegisterFatBinaryFn register_fat_binary = nullptr;
  HipUnregisterFatBinaryFn unregister_fat_binary = nullptr;
  HipRegisterFunctionFn register_function = nullptr;
  HipRegisterManagedVarFn register_managed_variable = nullptr;
};

class HipModuleLibraryExecutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    library_ = dlopen(CandidateLibPath(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) {
      GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": "
                   << dlerror();
    }
    ASSERT_TRUE(api_.Resolve(library_));
    const hipError_t init_result = api_.init(/*flags=*/0);
    if (init_result != hipSuccess) {
      GTEST_SKIP() << "hipInit failed: " << init_result;
    }
    ASSERT_EQ(hipSuccess, api_.get_device(&device_));
    ASSERT_EQ(hipSuccess, api_.get_device_properties(&properties_, device_));
    executable_image_ =
        hrx_cts::FindAmdgpuExecutableTestImage(properties_.gcnArchName);
    hip_image_ =
        hrx_cts::FindAmdgpuHipModuleLibraryTestImage(properties_.gcnArchName);
    ASSERT_NE(nullptr, executable_image_.file)
        << "no embedded executable HSACO for " << properties_.gcnArchName;
    ASSERT_NE(nullptr, hip_image_.file)
        << "no embedded HIP HSACO for " << properties_.gcnArchName;
  }

  void TearDown() override {
    if (!library_) return;
    if (api_.deinit) {
      EXPECT_EQ(hipSuccess, api_.deinit());
    }
    dlclose(library_);
  }

  void* library_ = nullptr;
  HipApi api_;
  int device_ = 0;
  hipDeviceProp_t properties_ = {};
  hrx_cts::AmdgpuExecutableTestImage executable_image_;
  hrx_cts::AmdgpuHipModuleLibraryTestImage hip_image_;
};

struct BundleEntry {
  uint64_t offset;
  uint64_t size;
  uint64_t triple_size;
};

void AppendBytes(std::vector<uint8_t>& buffer, const void* data,
                 size_t length) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  buffer.insert(buffer.end(), bytes, bytes + length);
}

std::vector<uint8_t> MakeBundle(
    const std::vector<std::pair<std::string, std::vector<uint8_t>>>& entries) {
  constexpr char kMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
  static_assert(sizeof(kMagic) - 1 == 24, "bundle magic length changed");

  uint64_t payload_offset = sizeof(kMagic) - 1 + sizeof(uint64_t);
  for (const auto& entry : entries) {
    payload_offset += sizeof(BundleEntry) + entry.first.size();
  }

  std::vector<uint8_t> bundle;
  AppendBytes(bundle, kMagic, sizeof(kMagic) - 1);
  const uint64_t entry_count = entries.size();
  AppendBytes(bundle, &entry_count, sizeof(entry_count));
  uint64_t next_payload_offset = payload_offset;
  for (const auto& entry : entries) {
    const BundleEntry bundle_entry = {
        /*.offset=*/next_payload_offset,
        /*.size=*/entry.second.size(),
        /*.triple_size=*/entry.first.size(),
    };
    AppendBytes(bundle, &bundle_entry, sizeof(bundle_entry));
    AppendBytes(bundle, entry.first.data(), entry.first.size());
    next_payload_offset += entry.second.size();
  }
  for (const auto& entry : entries) {
    AppendBytes(bundle, entry.second.data(), entry.second.size());
  }
  return bundle;
}

void ManagedIncrementHostStub() {}

TEST_F(HipModuleLibraryExecutionTest,
       ModuleAndLazyLibraryMetadataUseNativeArgumentLayout) {
  hipModule_t module = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.module_load_data_ex(&module, executable_image_.file->data,
                                     /*option_count=*/0, /*options=*/nullptr,
                                     /*option_values=*/nullptr));

  unsigned int module_function_count = 0;
  EXPECT_EQ(hipSuccess,
            api_.module_get_function_count(&module_function_count, module));
  EXPECT_GT(module_function_count, 0u);
  hipDeviceptr_t global_pointer = nullptr;
  size_t global_size = 0;
  ASSERT_EQ(hipSuccess, api_.module_get_global(&global_pointer, &global_size,
                                               module, "hrx_device_global"));
  EXPECT_EQ(sizeof(uint32_t), global_size);
  uint32_t global_value = 0;
  ASSERT_EQ(hipSuccess,
            api_.memcpy(&global_value, global_pointer, sizeof(global_value),
                        hipMemcpyDeviceToHost));
  EXPECT_EQ(17u, global_value);
  void* texture_reference = nullptr;
  EXPECT_EQ(
      hipErrorNotSupported,
      api_.module_get_tex_ref(&texture_reference, module, "hrx_device_global"));
  ASSERT_EQ(hipSuccess, api_.module_unload(module));

  module = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.module_load_fat_binary(&module, executable_image_.file->data));
  ASSERT_EQ(hipSuccess, api_.module_unload(module));

  std::vector<uint8_t> transient_image(
      executable_image_.file->data,
      executable_image_.file->data + executable_image_.file->size);
  hipLibrary_t library = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.library_load_data(
                &library, transient_image.data(), /*jit_options=*/nullptr,
                /*jit_option_values=*/nullptr, /*jit_option_count=*/0,
                /*library_options=*/nullptr,
                /*library_option_values=*/nullptr,
                /*library_option_count=*/0));
  std::fill(transient_image.begin(), transient_image.end(), uint8_t{0xA5});
  std::vector<uint8_t>().swap(transient_image);

  std::array<hipError_t, 2> query_results = {};
  std::array<unsigned int, 2> query_counts = {};
  std::thread first_query([&] {
    query_results[0] = api_.set_device(device_);
    if (query_results[0] == hipSuccess) {
      query_results[0] =
          api_.library_get_kernel_count(&query_counts[0], library);
    }
  });
  std::thread second_query([&] {
    query_results[1] = api_.set_device(device_);
    if (query_results[1] == hipSuccess) {
      query_results[1] =
          api_.library_get_kernel_count(&query_counts[1], library);
    }
  });
  first_query.join();
  second_query.join();
  EXPECT_EQ(hipSuccess, query_results[0]);
  EXPECT_EQ(hipSuccess, query_results[1]);
  ASSERT_EQ(query_counts[0], query_counts[1]);
  ASSERT_GT(query_counts[0], 0u);

  std::vector<hipKernel_t> kernels(query_counts[0]);
  ASSERT_EQ(hipSuccess, api_.library_enumerate_kernels(
                            kernels.data(), kernels.size(), library));
  EXPECT_TRUE(
      std::all_of(kernels.begin(), kernels.end(),
                  [](hipKernel_t kernel) { return kernel != nullptr; }));

  hipKernel_t kernel = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.library_get_kernel(&kernel, library,
                                    "hrx_transform_padded_parameters"));
  const char* kernel_name = nullptr;
  ASSERT_EQ(hipSuccess, api_.kernel_get_name(&kernel_name, kernel));
  EXPECT_STREQ("hrx_transform_padded_parameters", kernel_name);
  hipLibrary_t kernel_library = nullptr;
  ASSERT_EQ(hipSuccess, api_.kernel_get_library(&kernel_library, kernel));
  EXPECT_EQ(library, kernel_library);

  const std::array<size_t, 4> expected_offsets = {0, 8, 16, 24};
  const std::array<size_t, 4> expected_sizes = {
      sizeof(uint16_t), sizeof(void*), sizeof(uint32_t), sizeof(void*)};
  for (size_t i = 0; i < expected_offsets.size(); ++i) {
    size_t offset = SIZE_MAX;
    size_t size = SIZE_MAX;
    ASSERT_EQ(hipSuccess,
              api_.kernel_get_param_info(kernel, i, &offset, &size));
    EXPECT_EQ(expected_offsets[i], offset);
    EXPECT_EQ(expected_sizes[i], size);
  }

  int constant_bytes = -1;
  EXPECT_EQ(
      hipErrorNotSupported,
      api_.kernel_get_attribute(&constant_bytes, hipFuncAttributeConstSizeBytes,
                                kernel, device_));

  void* library_global = nullptr;
  size_t library_global_size = 0;
  ASSERT_EQ(hipSuccess,
            api_.library_get_global(&library_global, &library_global_size,
                                    library, "hrx_device_global"));
  EXPECT_EQ(global_size, library_global_size);

  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess, api_.kernel_get_function(&function, kernel));
  hipDeviceptr_t input = nullptr;
  hipDeviceptr_t output = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&input, sizeof(uint32_t)));
  ASSERT_EQ(hipSuccess, api_.malloc(&output, sizeof(uint32_t)));
  const uint32_t input_value = 7;
  ASSERT_EQ(hipSuccess, api_.memcpy(input, &input_value, sizeof(input_value),
                                    hipMemcpyHostToDevice));
  uint16_t bias = 3;
  uint32_t scale = 5;
  hipDeviceptr_t input_argument = input;
  hipDeviceptr_t output_argument = output;
  void* arguments[] = {&bias, &input_argument, &scale, &output_argument};
  ASSERT_EQ(hipSuccess, api_.module_launch_kernel(function, 1, 1, 1, 1, 1, 1, 0,
                                                  /*stream=*/nullptr, arguments,
                                                  /*extra=*/nullptr));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  uint32_t output_value = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&output_value, output, sizeof(output_value),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(input_value * scale + bias, output_value);
  EXPECT_EQ(hipSuccess, api_.free(output));
  EXPECT_EQ(hipSuccess, api_.free(input));
  EXPECT_EQ(hipSuccess, api_.library_unload(library));
}

TEST_F(HipModuleLibraryExecutionTest,
       LazyLibraryOwnsInputsCachesFailuresAndSelectsOnFirstQuery) {
  std::array<uint8_t, 64> malformed_image = {};
  hipLibrary_t malformed_library = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.library_load_data(
                &malformed_library, malformed_image.data(),
                /*jit_options=*/nullptr, /*jit_option_values=*/nullptr,
                /*jit_option_count=*/0, /*library_options=*/nullptr,
                /*library_option_values=*/nullptr,
                /*library_option_count=*/0));
  std::fill(malformed_image.begin(), malformed_image.end(), uint8_t{0xA5});
  unsigned int count = 0;
  EXPECT_EQ(hipErrorInvalidImage,
            api_.library_get_kernel_count(&count, malformed_library));
  EXPECT_EQ(hipErrorInvalidImage,
            api_.library_get_kernel_count(&count, malformed_library));
  EXPECT_EQ(hipSuccess, api_.library_unload(malformed_library));

  hipJitOption jit_option = hipJitOptionTarget;
  unsigned int jit_target = 0;
  void* jit_option_value = &jit_target;
  hipLibrary_t invalid_library = nullptr;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.library_load_data(&invalid_library,
                                   executable_image_.file->data, &jit_option,
                                   &jit_option_value, /*jit_option_count=*/1,
                                   /*library_options=*/nullptr,
                                   /*library_option_values=*/nullptr,
                                   /*library_option_count=*/0));

  hipLibraryOption unknown_option = static_cast<hipLibraryOption>(UINT_MAX);
  void* present_option_value = reinterpret_cast<void*>(uintptr_t{1});
  EXPECT_EQ(hipErrorInvalidValue,
            api_.library_load_data(
                &invalid_library, executable_image_.file->data,
                /*jit_options=*/nullptr, /*jit_option_values=*/nullptr,
                /*jit_option_count=*/0, &unknown_option, &present_option_value,
                /*library_option_count=*/1));

  hipLibraryOption preserved_option = hipLibraryBinaryIsPreserved;
  hipLibrary_t preserved_library = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.library_load_data(
                &preserved_library, executable_image_.file->data,
                /*jit_options=*/nullptr, /*jit_option_values=*/nullptr,
                /*jit_option_count=*/0, &preserved_option,
                &present_option_value, /*library_option_count=*/1));
  ASSERT_EQ(hipSuccess,
            api_.library_get_kernel_count(&count, preserved_library));
  EXPECT_GT(count, 0u);
  EXPECT_EQ(hipSuccess, api_.library_unload(preserved_library));

  hipLibraryOption host_table_option =
      hipLibraryHostUniversalFunctionAndDataTable;
  hipLibrary_t host_table_library = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.library_load_data(
                &host_table_library, executable_image_.file->data,
                /*jit_options=*/nullptr, /*jit_option_values=*/nullptr,
                /*jit_option_count=*/0, &host_table_option,
                &present_option_value, /*library_option_count=*/1));
  ASSERT_EQ(hipSuccess,
            api_.library_get_kernel_count(&count, host_table_library));
  EXPECT_EQ(hipSuccess, api_.library_unload(host_table_library));

  iree::testing::TempFilePath image_path("hip_lazy_library", ".hsaco");
  {
    std::ofstream output(image_path.path(), std::ios::binary);
    ASSERT_TRUE(output.good());
    output.write(reinterpret_cast<const char*>(executable_image_.file->data),
                 executable_image_.file->size);
    ASSERT_TRUE(output.good());
  }
  std::vector<char> transient_path(image_path.path().begin(),
                                   image_path.path().end());
  transient_path.push_back('\0');
  hipLibrary_t file_library = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.library_load_from_file(
                &file_library, transient_path.data(), /*jit_options=*/nullptr,
                /*jit_option_values=*/nullptr, /*jit_option_count=*/0,
                /*library_options=*/nullptr,
                /*library_option_values=*/nullptr,
                /*library_option_count=*/0));
  std::fill(transient_path.begin(), transient_path.end(), '\0');
  ASSERT_EQ(hipSuccess, api_.library_get_kernel_count(&count, file_library));
  EXPECT_GT(count, 0u);
  EXPECT_EQ(hipSuccess, api_.library_unload(file_library));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.library_load_from_file(
                &invalid_library, image_path.path().c_str(), &jit_option,
                &jit_option_value, /*jit_option_count=*/1,
                /*library_options=*/nullptr,
                /*library_option_values=*/nullptr,
                /*library_option_count=*/0));
  EXPECT_TRUE(image_path.Remove());

  int device_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count));
  int other_device = -1;
  hipDeviceProp_t other_properties = {};
  hrx_cts::AmdgpuExecutableTestImage other_image;
  for (int candidate = 0; candidate < device_count; ++candidate) {
    if (candidate == device_) continue;
    ASSERT_EQ(hipSuccess,
              api_.get_device_properties(&other_properties, candidate));
    other_image =
        hrx_cts::FindAmdgpuExecutableTestImage(other_properties.gcnArchName);
    if (other_image.file &&
        other_image.target_key != executable_image_.target_key) {
      other_device = candidate;
      break;
    }
  }
  if (other_device >= 0) {
    const std::string current_triple =
        "hipv4-amdgcn-amd-amdhsa--" + executable_image_.target_key;
    const std::string other_triple =
        "hipv4-amdgcn-amd-amdhsa--" + other_image.target_key;
    std::vector<uint8_t> bundle = MakeBundle({
        {current_triple,
         std::vector<uint8_t>(
             executable_image_.file->data,
             executable_image_.file->data + executable_image_.file->size)},
        {other_triple,
         std::vector<uint8_t>(other_image.file->data,
                              other_image.file->data + other_image.file->size)},
    });
    hipLibrary_t selected_library = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.library_load_data(
                  &selected_library, bundle.data(), /*jit_options=*/nullptr,
                  /*jit_option_values=*/nullptr, /*jit_option_count=*/0,
                  /*library_options=*/nullptr,
                  /*library_option_values=*/nullptr,
                  /*library_option_count=*/0));
    std::fill(bundle.begin(), bundle.end(), uint8_t{0xA5});
    ASSERT_EQ(hipSuccess, api_.set_device(other_device));
    ASSERT_EQ(hipSuccess,
              api_.library_get_kernel_count(&count, selected_library));
    EXPECT_GT(count, 0u);
    EXPECT_EQ(hipSuccess, api_.library_unload(selected_library));
    ASSERT_EQ(hipSuccess, api_.set_device(device_));
  }
}

TEST_F(HipModuleLibraryExecutionTest,
       ExtendedLaunchExecutesExactTailAndValidatesEventsAtomically) {
  hipModule_t module = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.module_load_data_ex(&module, executable_image_.file->data,
                                     /*option_count=*/0, /*options=*/nullptr,
                                     /*option_values=*/nullptr));
  hipFunction_t exact_function = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.module_get_function(&exact_function, module,
                                     "hrx_store_exact_workitem_indices"));

  constexpr size_t kOutputCount = 128;
  hipDeviceptr_t output = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&output, kOutputCount * sizeof(uint32_t)));
  const std::array<uint32_t, kOutputCount> zeros = {};
  ASSERT_EQ(hipSuccess, api_.memcpy(output, zeros.data(), sizeof(zeros),
                                    hipMemcpyHostToDevice));
  uint32_t local_size = 64;
  hipDeviceptr_t output_argument = output;
  void* exact_arguments[] = {&local_size, &output_argument};
  ASSERT_EQ(hipSuccess,
            api_.ext_module_launch_kernel(
                exact_function, /*global_size_x=*/100, /*global_size_y=*/1,
                /*global_size_z=*/1, /*local_size_x=*/local_size,
                /*local_size_y=*/1, /*local_size_z=*/1,
                /*shared_memory_bytes=*/0, /*stream=*/nullptr, exact_arguments,
                /*extra=*/nullptr, /*start_event=*/nullptr,
                /*stop_event=*/nullptr, /*flags=*/0));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  std::array<uint32_t, kOutputCount> values = {};
  ASSERT_EQ(hipSuccess, api_.memcpy(values.data(), output, sizeof(values),
                                    hipMemcpyDeviceToHost));
  for (size_t i = 0; i < 100; ++i) EXPECT_EQ(i + 1, values[i]);
  for (size_t i = 100; i < values.size(); ++i) EXPECT_EQ(0u, values[i]);

  hipFunction_t size_function = nullptr;
  ASSERT_EQ(hipSuccess, api_.module_get_function(&size_function, module,
                                                 "hrx_report_dispatch_size"));
  hipDeviceptr_t reported_size = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&reported_size, 2 * sizeof(uint32_t)));
  hipDeviceptr_t reported_size_argument = reported_size;
  void* size_arguments[] = {&reported_size_argument};
  ASSERT_EQ(hipSuccess,
            api_.ext_module_launch_kernel(
                size_function, /*global_size_x=*/17, /*global_size_y=*/1,
                /*global_size_z=*/1, /*local_size_x=*/64,
                /*local_size_y=*/1, /*local_size_z=*/1,
                /*shared_memory_bytes=*/0, /*stream=*/nullptr, size_arguments,
                /*extra=*/nullptr, /*start_event=*/nullptr,
                /*stop_event=*/nullptr, /*flags=*/0));
  ASSERT_EQ(hipSuccess, api_.device_synchronize());
  std::array<uint32_t, 2> reported_geometry = {};
  ASSERT_EQ(hipSuccess,
            api_.memcpy(reported_geometry.data(), reported_size,
                        sizeof(reported_geometry), hipMemcpyDeviceToHost));
  EXPECT_EQ(17u, reported_geometry[0]);
  EXPECT_EQ(64u, reported_geometry[1]);
  EXPECT_EQ(hipSuccess, api_.free(reported_size));

  hipModule_t hip_module = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.module_load_data_ex(&hip_module, hip_image_.file->data,
                                     /*option_count=*/0, /*options=*/nullptr,
                                     /*option_values=*/nullptr));
  hipFunction_t uniform_function = nullptr;
  ASSERT_EQ(hipSuccess, api_.module_get_function(&uniform_function, hip_module,
                                                 "hrx_uniform_store_output"));
  uint32_t uniform_value = 91;
  void* uniform_arguments[] = {&output_argument, &uniform_value};
  EXPECT_EQ(hipErrorInvalidValue,
            api_.ext_module_launch_kernel(
                uniform_function, /*global_size_x=*/100, /*global_size_y=*/1,
                /*global_size_z=*/1, /*local_size_x=*/64,
                /*local_size_y=*/1, /*local_size_z=*/1,
                /*shared_memory_bytes=*/0, /*stream=*/nullptr,
                uniform_arguments, /*extra=*/nullptr,
                /*start_event=*/nullptr, /*stop_event=*/nullptr, /*flags=*/0));

  hipStream_t stream = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_create(&stream));
  hipEvent_t stale_event = nullptr;
  ASSERT_EQ(hipSuccess, api_.event_create(&stale_event));
  ASSERT_EQ(hipSuccess, api_.event_destroy(stale_event));

  ASSERT_EQ(hipSuccess, api_.memcpy(output, zeros.data(), sizeof(zeros),
                                    hipMemcpyHostToDevice));
  EXPECT_EQ(hipErrorInvalidHandle,
            api_.ext_module_launch_kernel(
                exact_function, /*global_size_x=*/64, /*global_size_y=*/1,
                /*global_size_z=*/1, /*local_size_x=*/64,
                /*local_size_y=*/1, /*local_size_z=*/1,
                /*shared_memory_bytes=*/0, stream, exact_arguments,
                /*extra=*/nullptr, /*start_event=*/nullptr, stale_event,
                /*flags=*/0));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream));
  ASSERT_EQ(hipSuccess, api_.memcpy(values.data(), output, sizeof(values),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(zeros, values);

  ASSERT_EQ(hipSuccess,
            api_.stream_begin_capture(stream, hipStreamCaptureModeGlobal));
  EXPECT_EQ(hipErrorInvalidHandle,
            api_.ext_module_launch_kernel(
                exact_function, /*global_size_x=*/64, /*global_size_y=*/1,
                /*global_size_z=*/1, /*local_size_x=*/64,
                /*local_size_y=*/1, /*local_size_z=*/1,
                /*shared_memory_bytes=*/0, stream, exact_arguments,
                /*extra=*/nullptr, /*start_event=*/nullptr, stale_event,
                /*flags=*/0));
  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_end_capture(stream, &graph));
  size_t node_count = SIZE_MAX;
  ASSERT_EQ(hipSuccess,
            api_.graph_get_nodes(graph, /*nodes=*/nullptr, &node_count));
  EXPECT_EQ(0u, node_count);
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));

  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(
        hipErrorInvalidValue,
        api_.module_launch_kernel(exact_function, 1, 1, 1, 1, 1, 1, 0, stream,
                                  /*arguments=*/nullptr, /*extra=*/nullptr));
  }
  EXPECT_EQ(hipSuccess, api_.stream_destroy(stream));
  EXPECT_EQ(hipSuccess, api_.module_unload(hip_module));
  EXPECT_EQ(hipSuccess, api_.free(output));
  EXPECT_EQ(hipSuccess, api_.module_unload(module));
}

TEST_F(HipModuleLibraryExecutionTest,
       LibraryKernelRejectsForeignExplicitAndDefaultStreams) {
  int device_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count));
  if (device_count < 2) GTEST_SKIP() << "requires two AMDGPU devices";

  hipLibrary_t library = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.library_load_data(
                &library, executable_image_.file->data,
                /*jit_options=*/nullptr, /*jit_option_values=*/nullptr,
                /*jit_option_count=*/0, /*library_options=*/nullptr,
                /*library_option_values=*/nullptr,
                /*library_option_count=*/0));
  hipKernel_t kernel = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.library_get_kernel(&kernel, library,
                                    "hrx_transform_padded_parameters"));
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess, api_.kernel_get_function(&function, kernel));

  const int foreign_device = device_ == 0 ? 1 : 0;
  ASSERT_EQ(hipSuccess, api_.set_device(foreign_device));
  hipStream_t foreign_stream = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_create(&foreign_stream));
  EXPECT_EQ(
      hipErrorInvalidDevice,
      api_.module_launch_kernel(function, 1, 1, 1, 1, 1, 1, 0, foreign_stream,
                                /*arguments=*/nullptr, /*extra=*/nullptr));
  EXPECT_EQ(hipErrorInvalidDevice,
            api_.module_launch_kernel(
                function, 1, 1, 1, 1, 1, 1, 0, /*stream=*/nullptr,
                /*arguments=*/nullptr, /*extra=*/nullptr));
  EXPECT_EQ(hipSuccess, api_.stream_destroy(foreign_stream));
  ASSERT_EQ(hipSuccess, api_.set_device(device_));
  EXPECT_EQ(hipSuccess, api_.library_unload(library));
}

TEST_F(HipModuleLibraryExecutionTest,
       ManagedRegistrationPublishesSharedStorageAndReleasesEachImage) {
  for (int iteration = 0; iteration < 3; ++iteration) {
    void** registration = api_.register_fat_binary(hip_image_.file->data);
    ASSERT_NE(nullptr, registration);

    constexpr char kKernelName[] = "hrx_increment_managed_value";
    api_.register_function(
        registration, reinterpret_cast<const void*>(&ManagedIncrementHostStub),
        const_cast<char*>(kKernelName), kKernelName, /*thread_limit=*/0,
        /*thread_index=*/nullptr, /*block_index=*/nullptr,
        /*block_dimensions=*/nullptr, /*grid_dimensions=*/nullptr,
        /*shared_memory_size=*/nullptr);

    uint32_t initial_value = 19;
    void* publication = &initial_value;
    api_.register_managed_variable(registration, &publication, &initial_value,
                                   "hrx_managed_value", sizeof(initial_value),
                                   alignof(uint32_t));
    ASSERT_NE(&initial_value, publication);
    EXPECT_EQ(initial_value, *static_cast<uint32_t*>(publication));

    size_t symbol_size = 0;
    ASSERT_EQ(hipSuccess, api_.get_symbol_size(&symbol_size, publication));
    EXPECT_EQ(sizeof(uint32_t), symbol_size);
    void* symbol_address = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.get_symbol_address(&symbol_address, publication));
    ASSERT_NE(nullptr, symbol_address);

    const uint32_t copied_value = 41 + iteration;
    ASSERT_EQ(hipSuccess, api_.memcpy_to_symbol(
                              publication, &copied_value, sizeof(copied_value),
                              /*offset=*/0, hipMemcpyHostToDevice));
    EXPECT_EQ(copied_value, *static_cast<uint32_t*>(publication));
    EXPECT_EQ(
        hipErrorInvalidValue,
        api_.memcpy_to_symbol(publication, &copied_value, sizeof(copied_value),
                              /*offset=*/1, hipMemcpyHostToDevice));

    hipFunction_t function = nullptr;
    ASSERT_EQ(hipSuccess, api_.get_func_by_symbol(
                              &function, reinterpret_cast<const void*>(
                                             &ManagedIncrementHostStub)));
    hipDeviceptr_t observed = nullptr;
    ASSERT_EQ(hipSuccess, api_.malloc(&observed, sizeof(uint32_t)));
    uint32_t increment = 7;
    hipDeviceptr_t observed_argument = observed;
    void* arguments[] = {&increment, &observed_argument};
    ASSERT_EQ(hipSuccess,
              api_.module_launch_kernel(function, 1, 1, 1, 1, 1, 1, 0,
                                        /*stream=*/nullptr, arguments,
                                        /*extra=*/nullptr));
    ASSERT_EQ(hipSuccess, api_.device_synchronize());

    const uint32_t expected_value = copied_value + increment;
    uint32_t observed_value = 0;
    ASSERT_EQ(hipSuccess,
              api_.memcpy(&observed_value, observed, sizeof(observed_value),
                          hipMemcpyDeviceToHost));
    EXPECT_EQ(expected_value, observed_value);
    EXPECT_EQ(expected_value, *static_cast<uint32_t*>(publication));
    uint32_t copied_back_value = 0;
    ASSERT_EQ(hipSuccess,
              api_.memcpy_from_symbol(&copied_back_value, publication,
                                      sizeof(copied_back_value), /*offset=*/0,
                                      hipMemcpyDeviceToHost));
    EXPECT_EQ(expected_value, copied_back_value);
    EXPECT_EQ(hipErrorInvalidValue,
              api_.memcpy_from_symbol(&copied_back_value, publication,
                                      sizeof(copied_back_value), /*offset=*/1,
                                      hipMemcpyDeviceToHost));

    // Captured nodes retain their module independently of the compiler
    // registration. Keep a captured launch alive while unregistering to verify
    // that its imported managed backing remains valid until graph destruction.
    hipStream_t capture_stream = nullptr;
    ASSERT_EQ(hipSuccess, api_.stream_create(&capture_stream));
    ASSERT_EQ(hipSuccess, api_.stream_begin_capture(
                              capture_stream, hipStreamCaptureModeGlobal));
    ASSERT_EQ(hipSuccess,
              api_.module_launch_kernel(function, 1, 1, 1, 1, 1, 1, 0,
                                        capture_stream, arguments,
                                        /*extra=*/nullptr));
    hipGraph_t retained_graph = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.stream_end_capture(capture_stream, &retained_graph));

    api_.unregister_fat_binary(registration);
    EXPECT_EQ(&initial_value, publication);
    EXPECT_EQ(hipSuccess, api_.graph_destroy(retained_graph));
    EXPECT_EQ(hipSuccess, api_.stream_destroy(capture_stream));
    EXPECT_EQ(hipSuccess, api_.free(observed));
  }
}

TEST_F(HipModuleLibraryExecutionTest,
       LinkerOwnsDataAndFileInputsThroughExecutableLaunch) {
  static constexpr uint32_t kNoopSpirv[] = {
      0x07230203u, 0x00010000u, 0x00000000u, 0x00000005u, 0x00000000u,
      0x00020011u, 0x00000004u, 0x00020011u, 0x00000006u, 0x0003000eu,
      0x00000002u, 0x00000002u, 0x0007000fu, 0x00000006u, 0x00000003u,
      0x5f787268u, 0x6b6e696cu, 0x6e5f6465u, 0x00706f6fu, 0x00060010u,
      0x00000003u, 0x00000011u, 0x00000001u, 0x00000001u, 0x00000001u,
      0x00020013u, 0x00000001u, 0x00030021u, 0x00000002u, 0x00000001u,
      0x00050036u, 0x00000001u, 0x00000003u, 0x00000000u, 0x00000002u,
      0x000200f8u, 0x00000004u, 0x000100fdu, 0x00010038u,
  };

  hipLinkState_t state = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.link_create(/*option_count=*/0, /*options=*/nullptr,
                             /*option_values=*/nullptr, &state));
  std::vector<uint32_t> transient_spirv(std::begin(kNoopSpirv),
                                        std::end(kNoopSpirv));
  ASSERT_EQ(hipSuccess,
            api_.link_add_data(state, hipJitInputSpirv, transient_spirv.data(),
                               transient_spirv.size() * sizeof(uint32_t),
                               "linked_noop.spv",
                               /*option_count=*/0, /*options=*/nullptr,
                               /*option_values=*/nullptr));
  std::fill(transient_spirv.begin(), transient_spirv.end(), UINT32_MAX);
  std::vector<uint32_t>().swap(transient_spirv);

  void* linked_binary = nullptr;
  size_t linked_binary_size = 0;
  ASSERT_EQ(hipSuccess,
            api_.link_complete(state, &linked_binary, &linked_binary_size));
  ASSERT_NE(nullptr, linked_binary);
  ASSERT_GT(linked_binary_size, 0u);
  hipModule_t module = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.module_load_data_ex(&module, linked_binary,
                                     /*option_count=*/0, /*options=*/nullptr,
                                     /*option_values=*/nullptr));
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.module_get_function(&function, module, "hrx_linked_noop"));
  ASSERT_EQ(hipSuccess, api_.module_launch_kernel(
                            function, 1, 1, 1, 1, 1, 1, 0, /*stream=*/nullptr,
                            /*arguments=*/nullptr, /*extra=*/nullptr));
  EXPECT_EQ(hipSuccess, api_.device_synchronize());
  EXPECT_EQ(hipSuccess, api_.module_unload(module));
  EXPECT_EQ(hipSuccess, api_.link_destroy(state));

  iree::testing::TempFilePath spirv_path("hip_linker_input", ".spv");
  {
    std::ofstream output(spirv_path.path(), std::ios::binary);
    ASSERT_TRUE(output.good());
    output.write(reinterpret_cast<const char*>(kNoopSpirv), sizeof(kNoopSpirv));
    ASSERT_TRUE(output.good());
  }
  state = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.link_create(/*option_count=*/0, /*options=*/nullptr,
                             /*option_values=*/nullptr, &state));
  ASSERT_EQ(hipSuccess,
            api_.link_add_file(state, hipJitInputSpirv,
                               spirv_path.path().c_str(), /*option_count=*/0,
                               /*options=*/nullptr,
                               /*option_values=*/nullptr));
  ASSERT_TRUE(spirv_path.Remove());
  linked_binary = nullptr;
  linked_binary_size = 0;
  ASSERT_EQ(hipSuccess,
            api_.link_complete(state, &linked_binary, &linked_binary_size));
  ASSERT_NE(nullptr, linked_binary);
  EXPECT_GT(linked_binary_size, 0u);
  EXPECT_EQ(hipSuccess, api_.link_destroy(state));
}

}  // namespace
