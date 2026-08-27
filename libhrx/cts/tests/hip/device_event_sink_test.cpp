// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <dlfcn.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
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
using HipHALDeinitFn = hipError_t (*)(void);
using HipHRXSetDeviceEventSinkFn = hipError_t (*)(hrx_device_event_sink_t sink);
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

struct CapturedPrintfEvent {
  // Event envelope with all borrowed spans cleared.
  hrx_device_event_t envelope = {};
  // Copied device identifier.
  std::string device_id;
  // Copied driver identifier.
  std::string driver_id;
  // Original event payload length.
  size_t payload_length = 0;
  // Original implementation payload length.
  size_t implementation_payload_length = 0;
  // Printf payload with all borrowed spans cleared.
  hrx_device_printf_event_t printf_payload = {};
  // Copied formatted text.
  std::string text;
  // Copied encoded arguments.
  std::vector<uint8_t> arguments;
  // Whether the event held a complete printf payload.
  bool has_printf_payload = false;
};

struct EventCapture {
  // Serializes callback writes with test-thread inspection.
  std::mutex mutex;
  // Complete events copied during their callback lifetime.
  std::vector<CapturedPrintfEvent> events;
};

std::string CopyStringView(hrx_string_view_t view) {
  return view.size == 0 ? std::string() : std::string(view.data, view.size);
}

std::vector<uint8_t> CopyByteSpan(hrx_const_byte_span_t span) {
  if (span.data_length == 0) return {};
  const auto* data = static_cast<const uint8_t*>(span.data);
  return std::vector<uint8_t>(data, data + span.data_length);
}

void CaptureDeviceEvent(void* user_data, const hrx_device_event_t* event) {
  auto* capture = static_cast<EventCapture*>(user_data);
  CapturedPrintfEvent captured;
  captured.envelope = *event;
  captured.device_id = CopyStringView(event->source.device_id);
  captured.driver_id = CopyStringView(event->source.driver_id);
  captured.payload_length = event->payload.data_length;
  captured.implementation_payload_length =
      event->implementation_payload.data_length;
  captured.envelope.source.device_id = {};
  captured.envelope.source.driver_id = {};
  captured.envelope.payload = {};
  captured.envelope.implementation_payload = {};

  if (event->payload.data_length >= sizeof(hrx_device_printf_event_t)) {
    std::memcpy(&captured.printf_payload, event->payload.data,
                sizeof(captured.printf_payload));
    captured.text = CopyStringView(captured.printf_payload.text);
    captured.arguments = CopyByteSpan(captured.printf_payload.arguments);
    captured.printf_payload.text = {};
    captured.printf_payload.arguments = {};
    captured.has_printf_payload = true;
  }

  std::lock_guard<std::mutex> lock(capture->mutex);
  capture->events.push_back(std::move(captured));
}

TEST(HipDeviceEventSinkTest, RedirectsBlockingPrintfAcrossRuntimeLifetimes) {
  if (hrx_cts_amdgpu_hip_printf_test_kernels_size() == 0) {
    GTEST_SKIP() << "ROCm device libraries are unavailable";
  }

  void* library = dlopen(CandidateLibPath(), RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": " << dlerror();
  }

  const auto init = ResolveHipSymbol<HipInitFn>(library, "hipInit");
  const auto deinit = ResolveHipSymbol<HipHALDeinitFn>(library, "hipHALDeinit");
  const auto set_device_event_sink =
      ResolveHipSymbol<HipHRXSetDeviceEventSinkFn>(library,
                                                   "hipHRXSetDeviceEventSink");
  const auto get_device =
      ResolveHipSymbol<HipGetDeviceFn>(library, "hipGetDevice");
  const auto get_device_properties = ResolveHipSymbol<HipGetDevicePropertiesFn>(
      library, "hipGetDeviceProperties");
  const auto module_load_data =
      ResolveHipSymbol<HipModuleLoadDataFn>(library, "hipModuleLoadData");
  const auto module_unload =
      ResolveHipSymbol<HipModuleUnloadFn>(library, "hipModuleUnload");
  const auto module_get_function =
      ResolveHipSymbol<HipModuleGetFunctionFn>(library, "hipModuleGetFunction");
  const auto module_launch_kernel = ResolveHipSymbol<HipModuleLaunchKernelFn>(
      library, "hipModuleLaunchKernel");
  const auto device_synchronize =
      ResolveHipSymbol<HipDeviceSynchronizeFn>(library, "hipDeviceSynchronize");
  const auto hip_malloc = ResolveHipSymbol<HipMallocFn>(library, "hipMalloc");
  const auto hip_free = ResolveHipSymbol<HipFreeFn>(library, "hipFree");
  const auto hip_memcpy = ResolveHipSymbol<HipMemcpyFn>(library, "hipMemcpy");

  ASSERT_NE(nullptr, init);
  ASSERT_NE(nullptr, deinit);
  ASSERT_NE(nullptr, set_device_event_sink);
  ASSERT_NE(nullptr, get_device);
  ASSERT_NE(nullptr, get_device_properties);
  ASSERT_NE(nullptr, module_load_data);
  ASSERT_NE(nullptr, module_unload);
  ASSERT_NE(nullptr, module_get_function);
  ASSERT_NE(nullptr, module_launch_kernel);
  ASSERT_NE(nullptr, device_synchronize);
  ASSERT_NE(nullptr, hip_malloc);
  ASSERT_NE(nullptr, hip_free);
  ASSERT_NE(nullptr, hip_memcpy);

  const hipError_t first_init_result = init(/*flags=*/0);
  if (first_init_result != hipSuccess) {
    dlclose(library);
    GTEST_SKIP() << "hipInit failed: " << first_init_result;
  }

  EventCapture capture;
  const hrx_device_event_sink_t sink = {
      /*.fn=*/CaptureDeviceEvent,
      /*.user_data=*/&capture,
  };
  EXPECT_EQ(hipErrorSetOnActiveProcess, set_device_event_sink(sink));
  ASSERT_EQ(hipSuccess, deinit());
  ASSERT_EQ(hipSuccess, set_device_event_sink(sink));
  ASSERT_EQ(hipSuccess, init(/*flags=*/0));

  int device = 0;
  ASSERT_EQ(hipSuccess, get_device(&device));
  hipDeviceProp_t properties = {};
  ASSERT_EQ(hipSuccess, get_device_properties(&properties, device));

  const hrx_cts::AmdgpuHipPrintfTestImage test_image =
      hrx_cts::FindAmdgpuHipPrintfTestImage(properties.gcnArchName);
  ASSERT_NE(nullptr, test_image.file)
      << "no embedded HIP printf HSACO for " << properties.gcnArchName;

  hipModule_t module = nullptr;
  ASSERT_EQ(hipSuccess, module_load_data(&module, test_image.file->data));
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess,
            module_get_function(&function, module, "hrx_blocking_printf"));

  hipDeviceptr_t output = nullptr;
  ASSERT_EQ(hipSuccess, hip_malloc(&output, 2 * sizeof(int)));
  const std::array<int, 2> initial_result = {-1, -1};
  ASSERT_EQ(hipSuccess,
            hip_memcpy(output, initial_result.data(), sizeof(initial_result),
                       hipMemcpyHostToDevice));

  uint32_t value = 42;
  hipDeviceptr_t result = output;
  void* arguments[] = {&value, &result};
  ::testing::internal::CaptureStdout();
  const hipError_t launch_result =
      module_launch_kernel(function, 1, 1, 1, 1, 1, 1,
                           /*shared_memory_bytes=*/0, /*stream=*/nullptr,
                           arguments, /*extra=*/nullptr);
  const hipError_t synchronize_result =
      launch_result == hipSuccess ? device_synchronize() : launch_result;
  std::fflush(stdout);
  const std::string stdout_text = ::testing::internal::GetCapturedStdout();
  ASSERT_EQ(hipSuccess, launch_result);
  ASSERT_EQ(hipSuccess, synchronize_result);
  EXPECT_TRUE(stdout_text.empty());

  constexpr int kExpectedPrintfLength = sizeof("hrx cts printf value=42\n") - 1;
  constexpr int kPostPrintfMarker = 0xC0FFEE;
  std::array<int, 2> actual_result = {};
  ASSERT_EQ(hipSuccess,
            hip_memcpy(actual_result.data(), output, sizeof(actual_result),
                       hipMemcpyDeviceToHost));
  EXPECT_EQ(kExpectedPrintfLength, actual_result[0]);
  EXPECT_EQ(kPostPrintfMarker, actual_result[1]);

  std::vector<CapturedPrintfEvent> events;
  {
    std::lock_guard<std::mutex> lock(capture.mutex);
    events = capture.events;
  }
  ASSERT_EQ(1u, events.size());
  const CapturedPrintfEvent& event = events.front();
  EXPECT_EQ(sizeof(hrx_device_event_t), event.envelope.record_length);
  EXPECT_EQ(HRX_DEVICE_EVENT_ABI_VERSION_0, event.envelope.abi_version);
  EXPECT_EQ(HRX_DEVICE_EVENT_TYPE_PRINTF, event.envelope.type);
  EXPECT_EQ(HRX_DEVICE_EVENT_SEVERITY_INFO, event.envelope.severity);
  EXPECT_EQ(HRX_DEVICE_EVENT_FLAG_NONE, event.envelope.flags);
  EXPECT_TRUE(event.device_id.empty());
  EXPECT_EQ("hip", event.driver_id);
  EXPECT_EQ(static_cast<uint32_t>(device),
            event.envelope.source.physical_device_ordinal);
  EXPECT_EQ(UINT32_MAX, event.envelope.source.queue_ordinal);
  EXPECT_EQ(0u, event.envelope.source.executable_id);
  EXPECT_EQ(UINT32_MAX, event.envelope.source.export_ordinal);
  EXPECT_EQ(sizeof(hrx_device_printf_event_t), event.payload_length);
  EXPECT_EQ(0u, event.implementation_payload_length);
  ASSERT_TRUE(event.has_printf_payload);
  EXPECT_EQ(sizeof(hrx_device_printf_event_t),
            event.printf_payload.record_length);
  EXPECT_EQ(HRX_DEVICE_PRINTF_EVENT_ABI_VERSION_0,
            event.printf_payload.abi_version);
  EXPECT_EQ(HRX_DEVICE_PRINTF_STREAM_STDOUT, event.printf_payload.stream);
  EXPECT_EQ(HRX_DEVICE_PRINTF_FLAG_NONE, event.printf_payload.flags);
  EXPECT_EQ(0u, event.printf_payload.format_id);
  EXPECT_EQ("hrx cts printf value=42\n", event.text);
  EXPECT_TRUE(event.arguments.empty());

  EXPECT_EQ(hipSuccess, hip_free(output));
  EXPECT_EQ(hipSuccess, module_unload(module));
}

}  // namespace
