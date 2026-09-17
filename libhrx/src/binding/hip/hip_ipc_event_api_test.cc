// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

#include "binding/hip/api.h"
#include "iree/testing/coordinated_test.h"
#include "iree/testing/gtest.h"

namespace {

template <typename Cleanup>
class ScopeExit {
 public:
  explicit ScopeExit(Cleanup cleanup) : cleanup_(std::move(cleanup)) {}
  ~ScopeExit() { cleanup_(); }
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;

 private:
  Cleanup cleanup_;
};

template <typename Cleanup>
ScopeExit(Cleanup) -> ScopeExit<Cleanup>;

const char* CandidateLibPath() {
  if (const char* env = std::getenv("HRX_TEST_LIBAMDHIP64");
      env && *env != '\0') {
    return env;
  }
#ifdef HRX_TEST_LIBAMDHIP64_PATH
  return HRX_TEST_LIBAMDHIP64_PATH;
#else
  return nullptr;
#endif
}

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipHALDeinitFn = hipError_t (*)(void);
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipGetDevicePropertiesFn = hipError_t (*)(hipDeviceProp_t* properties,
                                                int device);
using HipCtxCreateFn = hipError_t (*)(hipCtx_t* context, unsigned int flags,
                                      hipDevice_t device);
using HipCtxDestroyFn = hipError_t (*)(hipCtx_t context);
using HipStreamCreateWithFlagsFn = hipError_t (*)(hipStream_t* stream,
                                                  unsigned int flags);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t stream);
using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t stream);
using HipStreamWaitEventFn = hipError_t (*)(hipStream_t stream,
                                            hipEvent_t event,
                                            unsigned int flags);
using HipStreamBeginCaptureFn = hipError_t (*)(hipStream_t stream,
                                               hipStreamCaptureMode mode);
using HipStreamBeginCaptureToGraphFn = hipError_t (*)(
    hipStream_t stream, hipGraph_t graph, const hipGraphNode_t* dependencies,
    const void* dependency_data, size_t dependency_count,
    hipStreamCaptureMode mode);
using HipStreamEndCaptureFn = hipError_t (*)(hipStream_t stream,
                                             hipGraph_t* graph);
using HipStreamIsCapturingFn =
    hipError_t (*)(hipStream_t stream, hipStreamCaptureStatus* capture_status);
using HipLaunchHostFuncFn = hipError_t (*)(hipStream_t stream, hipHostFn_t fn,
                                           void* user_data);
using HipEventCreateWithFlagsFn = hipError_t (*)(hipEvent_t* event,
                                                 unsigned int flags);
using HipEventDestroyFn = hipError_t (*)(hipEvent_t event);
using HipEventRecordFn = hipError_t (*)(hipEvent_t event, hipStream_t stream);
using HipEventQueryFn = hipError_t (*)(hipEvent_t event);
using HipEventSynchronizeFn = hipError_t (*)(hipEvent_t event);
using HipEventElapsedTimeFn = hipError_t (*)(float* milliseconds,
                                             hipEvent_t start, hipEvent_t stop);
using HipIpcGetEventHandleFn = hipError_t (*)(hipIpcEventHandle_t* handle,
                                              hipEvent_t event);
using HipIpcOpenEventHandleFn = hipError_t (*)(hipEvent_t* event,
                                               hipIpcEventHandle_t handle);
using HipExtLaunchKernelFn = hipError_t (*)(
    const void* function_address, dim3 num_blocks, dim3 dim_blocks, void** args,
    size_t shared_memory_bytes, hipStream_t stream, hipEvent_t start_event,
    hipEvent_t stop_event, int flags);
using HipExtModuleLaunchKernelFn = hipError_t (*)(
    hipFunction_t function, unsigned int global_x, unsigned int global_y,
    unsigned int global_z, unsigned int local_x, unsigned int local_y,
    unsigned int local_z, unsigned int shared_memory_bytes, hipStream_t stream,
    void** kernel_params, void** extra, hipEvent_t start_event,
    hipEvent_t stop_event, int flags);
using HipGraphCreateFn = hipError_t (*)(hipGraph_t* graph, unsigned int flags);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t graph);
using HipGraphGetNodesFn = hipError_t (*)(hipGraph_t graph,
                                          hipGraphNode_t* nodes,
                                          size_t* node_count);
using HipGraphInstantiateFn = hipError_t (*)(hipGraphExec_t* graph_exec,
                                             hipGraph_t graph,
                                             hipGraphNode_t* error_node,
                                             char* log_buffer,
                                             size_t buffer_size);
using HipGraphExecDestroyFn = hipError_t (*)(hipGraphExec_t graph_exec);
using HipGraphAddNodeFn = hipError_t (*)(hipGraphNode_t* node, hipGraph_t graph,
                                         const hipGraphNode_t* dependencies,
                                         size_t dependency_count,
                                         const void* node_params);
using HipGraphAddEventNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, hipEvent_t event);
using HipGraphEventNodeSetEventFn = hipError_t (*)(hipGraphNode_t node,
                                                   hipEvent_t event);
using HipGraphEventNodeGetEventFn = hipError_t (*)(hipGraphNode_t node,
                                                   hipEvent_t* event);
using HipGraphExecEventNodeSetEventFn = hipError_t (*)(
    hipGraphExec_t graph_exec, hipGraphNode_t node, hipEvent_t event);

template <typename T>
T ResolveHipSymbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

struct HipRuntimeApi {
  bool Load() {
    const char* library_path = CandidateLibPath();
    if (!library_path) {
      std::fprintf(stderr, "libamdhip64 path was not provided\n");
      return false;
    }
    library = dlopen(library_path, RTLD_LAZY | RTLD_LOCAL);
    if (!library) {
      std::fprintf(stderr, "cannot dlopen %s: %s\n", library_path, dlerror());
      return false;
    }

#define HRX_RESOLVE_HIP(field, symbol)                          \
  do {                                                          \
    field = ResolveHipSymbol<decltype(field)>(library, symbol); \
    if (!field) {                                               \
      std::fprintf(stderr, "cannot resolve %s\n", symbol);      \
      return false;                                             \
    }                                                           \
  } while (false)
    HRX_RESOLVE_HIP(init, "hipInit");
    HRX_RESOLVE_HIP(hal_deinit, "hipHALDeinit");
    HRX_RESOLVE_HIP(get_device, "hipGetDevice");
    HRX_RESOLVE_HIP(get_device_properties, "hipGetDeviceProperties");
    HRX_RESOLVE_HIP(context_create, "hipCtxCreate");
    HRX_RESOLVE_HIP(context_destroy, "hipCtxDestroy");
    HRX_RESOLVE_HIP(stream_create_with_flags, "hipStreamCreateWithFlags");
    HRX_RESOLVE_HIP(stream_destroy, "hipStreamDestroy");
    HRX_RESOLVE_HIP(stream_synchronize, "hipStreamSynchronize");
    HRX_RESOLVE_HIP(stream_wait_event, "hipStreamWaitEvent");
    HRX_RESOLVE_HIP(stream_begin_capture, "hipStreamBeginCapture");
    HRX_RESOLVE_HIP(stream_begin_capture_to_graph,
                    "hipStreamBeginCaptureToGraph");
    HRX_RESOLVE_HIP(stream_end_capture, "hipStreamEndCapture");
    HRX_RESOLVE_HIP(stream_is_capturing, "hipStreamIsCapturing");
    HRX_RESOLVE_HIP(launch_host_function, "hipLaunchHostFunc");
    HRX_RESOLVE_HIP(event_create_with_flags, "hipEventCreateWithFlags");
    HRX_RESOLVE_HIP(event_destroy, "hipEventDestroy");
    HRX_RESOLVE_HIP(event_record, "hipEventRecord");
    HRX_RESOLVE_HIP(event_query, "hipEventQuery");
    HRX_RESOLVE_HIP(event_synchronize, "hipEventSynchronize");
    HRX_RESOLVE_HIP(event_elapsed_time, "hipEventElapsedTime");
    HRX_RESOLVE_HIP(ipc_get_event_handle, "hipIpcGetEventHandle");
    HRX_RESOLVE_HIP(ipc_open_event_handle, "hipIpcOpenEventHandle");
    HRX_RESOLVE_HIP(ext_launch_kernel, "hipExtLaunchKernel");
    HRX_RESOLVE_HIP(ext_module_launch_kernel, "hipExtModuleLaunchKernel");
    HRX_RESOLVE_HIP(graph_create, "hipGraphCreate");
    HRX_RESOLVE_HIP(graph_destroy, "hipGraphDestroy");
    HRX_RESOLVE_HIP(graph_get_nodes, "hipGraphGetNodes");
    HRX_RESOLVE_HIP(graph_instantiate, "hipGraphInstantiate");
    HRX_RESOLVE_HIP(graph_exec_destroy, "hipGraphExecDestroy");
    HRX_RESOLVE_HIP(graph_add_node, "hipGraphAddNode");
    HRX_RESOLVE_HIP(graph_add_event_record_node, "hipGraphAddEventRecordNode");
    HRX_RESOLVE_HIP(graph_add_event_wait_node, "hipGraphAddEventWaitNode");
    HRX_RESOLVE_HIP(graph_event_record_node_get_event,
                    "hipGraphEventRecordNodeGetEvent");
    HRX_RESOLVE_HIP(graph_event_wait_node_get_event,
                    "hipGraphEventWaitNodeGetEvent");
    HRX_RESOLVE_HIP(graph_event_record_node_set_event,
                    "hipGraphEventRecordNodeSetEvent");
    HRX_RESOLVE_HIP(graph_event_wait_node_set_event,
                    "hipGraphEventWaitNodeSetEvent");
    HRX_RESOLVE_HIP(graph_exec_event_record_node_set_event,
                    "hipGraphExecEventRecordNodeSetEvent");
    HRX_RESOLVE_HIP(graph_exec_event_wait_node_set_event,
                    "hipGraphExecEventWaitNodeSetEvent");
#undef HRX_RESOLVE_HIP
    return true;
  }

  hipError_t Initialize() {
    const hipError_t result = init(/*flags=*/0);
    initialized = result == hipSuccess;
    return result;
  }

  hipError_t Shutdown() {
    if (!initialized) return hipSuccess;
    initialized = false;
    return hal_deinit();
  }

  ~HipRuntimeApi() {
    if (initialized) {
      const hipError_t result = Shutdown();
      if (result != hipSuccess) {
        std::fprintf(stderr, "hipHALDeinit failed: %d\n",
                     static_cast<int>(result));
      }
    }
  }

  // Loaded runtime shared object. It remains loaded until process exit so a
  // test failure cannot unload code while an asynchronous callback is live.
  void* library = nullptr;
  bool initialized = false;
  HipInitFn init = nullptr;
  HipHALDeinitFn hal_deinit = nullptr;
  HipGetDeviceFn get_device = nullptr;
  HipGetDevicePropertiesFn get_device_properties = nullptr;
  HipCtxCreateFn context_create = nullptr;
  HipCtxDestroyFn context_destroy = nullptr;
  HipStreamCreateWithFlagsFn stream_create_with_flags = nullptr;
  HipStreamDestroyFn stream_destroy = nullptr;
  HipStreamSynchronizeFn stream_synchronize = nullptr;
  HipStreamWaitEventFn stream_wait_event = nullptr;
  HipStreamBeginCaptureFn stream_begin_capture = nullptr;
  HipStreamBeginCaptureToGraphFn stream_begin_capture_to_graph = nullptr;
  HipStreamEndCaptureFn stream_end_capture = nullptr;
  HipStreamIsCapturingFn stream_is_capturing = nullptr;
  HipLaunchHostFuncFn launch_host_function = nullptr;
  HipEventCreateWithFlagsFn event_create_with_flags = nullptr;
  HipEventDestroyFn event_destroy = nullptr;
  HipEventRecordFn event_record = nullptr;
  HipEventQueryFn event_query = nullptr;
  HipEventSynchronizeFn event_synchronize = nullptr;
  HipEventElapsedTimeFn event_elapsed_time = nullptr;
  HipIpcGetEventHandleFn ipc_get_event_handle = nullptr;
  HipIpcOpenEventHandleFn ipc_open_event_handle = nullptr;
  HipExtLaunchKernelFn ext_launch_kernel = nullptr;
  HipExtModuleLaunchKernelFn ext_module_launch_kernel = nullptr;
  HipGraphCreateFn graph_create = nullptr;
  HipGraphDestroyFn graph_destroy = nullptr;
  HipGraphGetNodesFn graph_get_nodes = nullptr;
  HipGraphInstantiateFn graph_instantiate = nullptr;
  HipGraphExecDestroyFn graph_exec_destroy = nullptr;
  HipGraphAddNodeFn graph_add_node = nullptr;
  HipGraphAddEventNodeFn graph_add_event_record_node = nullptr;
  HipGraphAddEventNodeFn graph_add_event_wait_node = nullptr;
  HipGraphEventNodeGetEventFn graph_event_record_node_get_event = nullptr;
  HipGraphEventNodeGetEventFn graph_event_wait_node_get_event = nullptr;
  HipGraphEventNodeSetEventFn graph_event_record_node_set_event = nullptr;
  HipGraphEventNodeSetEventFn graph_event_wait_node_set_event = nullptr;
  HipGraphExecEventNodeSetEventFn graph_exec_event_record_node_set_event =
      nullptr;
  HipGraphExecEventNodeSetEventFn graph_exec_event_wait_node_set_event =
      nullptr;
};

template <typename T>
T ReadWireValue(const hipIpcEventHandle_t& handle, size_t byte_offset) {
  T value = {};
  std::memcpy(&value, handle.reserved + byte_offset, sizeof(value));
  return value;
}

bool IsZeroed(const void* data, size_t data_length) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < data_length; ++i) {
    if (bytes[i] != 0) return false;
  }
  return true;
}

class HostGate {
 public:
  ~HostGate() { Open(); }

  static void Callback(void* user_data) {
    static_cast<HostGate*>(user_data)->Wait();
  }

  void WaitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return entered_; });
  }

  void Open() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      is_open_ = true;
    }
    condition_.notify_all();
  }

 private:
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    entered_ = true;
    condition_.notify_all();
    condition_.wait(lock, [this] { return is_open_; });
  }

  std::mutex mutex_;
  std::condition_variable condition_;
  bool entered_ = false;
  bool is_open_ = false;
};

TEST(HipIpcEventApiTest, ValidatesCallsAndMatchesStockWire) {
  HipRuntimeApi api;
  ASSERT_TRUE(api.Load());
  ASSERT_EQ(hipSuccess, api.Initialize());

  int device = -1;
  ASSERT_EQ(hipSuccess, api.get_device(&device));
  hipDeviceProp_t properties = {};
  ASSERT_EQ(hipSuccess, api.get_device_properties(&properties, device));
  EXPECT_EQ(1, properties.ipcEventSupported);

  EXPECT_EQ(hipErrorInvalidValue,
            api.ipc_get_event_handle(/*handle=*/nullptr, /*event=*/nullptr));
  hipIpcEventHandle_t handle;
  std::memset(&handle, 0xA5, sizeof(handle));
  EXPECT_EQ(hipErrorInvalidValue,
            api.ipc_get_event_handle(&handle, /*event=*/nullptr));
  EXPECT_TRUE(IsZeroed(&handle, sizeof(handle)));
  std::memset(&handle, 0xA5, sizeof(handle));
  EXPECT_EQ(hipErrorInvalidValue,
            api.ipc_get_event_handle(
                &handle, reinterpret_cast<hipEvent_t>(uintptr_t{1})));
  EXPECT_TRUE(IsZeroed(&handle, sizeof(handle)));

  hipEvent_t ordinary_event = nullptr;
  ASSERT_EQ(hipSuccess, api.event_create_with_flags(&ordinary_event,
                                                    hipEventDisableTiming));
  std::memset(&handle, 0xA5, sizeof(handle));
  EXPECT_EQ(hipErrorInvalidConfiguration,
            api.ipc_get_event_handle(&handle, ordinary_event));
  EXPECT_TRUE(IsZeroed(&handle, sizeof(handle)));
  EXPECT_EQ(hipSuccess, api.event_destroy(ordinary_event));

  hipEvent_t invalid_event = reinterpret_cast<hipEvent_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorInvalidValue,
            api.event_create_with_flags(&invalid_event, hipEventInterprocess));
  EXPECT_EQ(nullptr, invalid_event);

  hipEvent_t event = nullptr;
  ASSERT_EQ(hipSuccess,
            api.event_create_with_flags(
                &event, hipEventInterprocess | hipEventDisableTiming));
  ASSERT_NE(nullptr, event);
  EXPECT_EQ(hipSuccess, api.event_query(event));

  ASSERT_EQ(hipSuccess, api.ipc_get_event_handle(&handle, event));
  EXPECT_EQ(1u, ReadWireValue<uint32_t>(handle, 0));
  EXPECT_EQ(static_cast<int32_t>(getpid()), ReadWireValue<int32_t>(handle, 4));
  EXPECT_TRUE(IsZeroed(handle.reserved + 40, 24));
  EXPECT_EQ(hipErrorNotReady, api.event_query(event));

  hipEvent_t imported_event = reinterpret_cast<hipEvent_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorInvalidContext,
            api.ipc_open_event_handle(&imported_event, handle));
  EXPECT_EQ(nullptr, imported_event);

  hipIpcEventHandle_t malformed = handle;
  const uint32_t unknown_type = 7;
  std::memcpy(malformed.reserved, &unknown_type, sizeof(unknown_type));
  imported_event = reinterpret_cast<hipEvent_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorInvalidValue,
            api.ipc_open_event_handle(&imported_event, malformed));
  EXPECT_EQ(nullptr, imported_event);

  EXPECT_EQ(hipErrorInvalidValue,
            api.ipc_open_event_handle(/*event=*/nullptr, handle));

  EXPECT_EQ(hipSuccess, api.event_record(event, /*stream=*/nullptr));
  EXPECT_EQ(hipSuccess, api.event_synchronize(event));
  EXPECT_EQ(hipSuccess, api.event_query(event));

  EXPECT_EQ(hipSuccess, api.event_destroy(event));
  EXPECT_EQ(hipSuccess, api.Shutdown());
}

TEST(HipIpcEventApiTest, SerializesConcurrentFirstRecordAndExports) {
  HipRuntimeApi api;
  ASSERT_TRUE(api.Load());
  ASSERT_EQ(hipSuccess, api.Initialize());

  hipEvent_t event = nullptr;
  ASSERT_EQ(hipSuccess,
            api.event_create_with_flags(
                &event, hipEventInterprocess | hipEventDisableTiming));

  std::mutex start_mutex;
  std::condition_variable start_condition;
  int ready_count = 0;
  bool start = false;
  auto wait_for_start = [&] {
    std::unique_lock<std::mutex> lock(start_mutex);
    ++ready_count;
    start_condition.notify_all();
    start_condition.wait(lock, [&] { return start; });
  };

  hipIpcEventHandle_t first_handle = {};
  hipIpcEventHandle_t second_handle = {};
  hipError_t first_export_result = hipErrorUnknown;
  hipError_t second_export_result = hipErrorUnknown;
  hipError_t record_result = hipErrorUnknown;
  std::thread first_export([&] {
    wait_for_start();
    first_export_result = api.ipc_get_event_handle(&first_handle, event);
  });
  std::thread second_export([&] {
    wait_for_start();
    second_export_result = api.ipc_get_event_handle(&second_handle, event);
  });
  std::thread record([&] {
    wait_for_start();
    record_result = api.event_record(event, /*stream=*/nullptr);
  });
  {
    std::unique_lock<std::mutex> lock(start_mutex);
    start_condition.wait(lock, [&] { return ready_count == 3; });
    start = true;
  }
  start_condition.notify_all();
  first_export.join();
  second_export.join();
  record.join();

  EXPECT_EQ(hipSuccess, first_export_result);
  EXPECT_EQ(hipSuccess, second_export_result);
  EXPECT_EQ(hipSuccess, record_result);
  EXPECT_EQ(0,
            std::memcmp(&first_handle, &second_handle, sizeof(first_handle)));
  EXPECT_EQ(hipSuccess, api.event_synchronize(event));
  EXPECT_EQ(hipSuccess, api.event_destroy(event));
  EXPECT_EQ(hipSuccess, api.Shutdown());
}

TEST(HipIpcEventApiTest, RecordsOnlyIpcEventAcrossSameDeviceContexts) {
  HipRuntimeApi api;
  ASSERT_TRUE(api.Load());
  ASSERT_EQ(hipSuccess, api.Initialize());

  hipEvent_t ordinary_event = nullptr;
  ASSERT_EQ(hipSuccess, api.event_create_with_flags(&ordinary_event,
                                                    hipEventDisableTiming));
  hipEvent_t ipc_event = nullptr;
  ASSERT_EQ(hipSuccess,
            api.event_create_with_flags(
                &ipc_event, hipEventInterprocess | hipEventDisableTiming));

  int device = -1;
  ASSERT_EQ(hipSuccess, api.get_device(&device));
  hipCtx_t recording_context = nullptr;
  ASSERT_EQ(hipSuccess,
            api.context_create(&recording_context, /*flags=*/0, device));
  hipStream_t recording_stream = nullptr;
  ASSERT_EQ(hipSuccess, api.stream_create_with_flags(&recording_stream,
                                                     hipStreamNonBlocking));

  EXPECT_EQ(hipErrorInvalidHandle,
            api.event_record(ordinary_event, recording_stream));
  EXPECT_EQ(hipSuccess, api.event_query(ordinary_event));
  EXPECT_EQ(hipSuccess, api.event_record(ipc_event, recording_stream));
  EXPECT_EQ(hipSuccess, api.event_synchronize(ipc_event));

  EXPECT_EQ(hipSuccess, api.stream_destroy(recording_stream));
  EXPECT_EQ(hipSuccess, api.context_destroy(recording_context));
  EXPECT_EQ(hipSuccess, api.event_destroy(ipc_event));
  EXPECT_EQ(hipSuccess, api.event_destroy(ordinary_event));
  EXPECT_EQ(hipSuccess, api.Shutdown());
}

TEST(HipIpcEventApiTest, PendingDestroyDoesNotWaitForCompletion) {
  HipRuntimeApi api;
  ASSERT_TRUE(api.Load());
  ASSERT_EQ(hipSuccess, api.Initialize());

  hipStream_t stream = nullptr;
  ASSERT_EQ(hipSuccess,
            api.stream_create_with_flags(&stream, hipStreamNonBlocking));
  hipEvent_t event = nullptr;
  ASSERT_EQ(hipSuccess,
            api.event_create_with_flags(
                &event, hipEventInterprocess | hipEventDisableTiming));

  HostGate gate;
  ASSERT_EQ(hipSuccess,
            api.launch_host_function(stream, HostGate::Callback, &gate));
  ASSERT_EQ(hipSuccess, api.event_record(event, stream));
  gate.WaitUntilEntered();
  EXPECT_EQ(hipErrorNotReady, api.event_query(event));

  // The submitted callback still owns the pending record. Handle removal must
  // return while the gate remains closed.
  EXPECT_EQ(hipSuccess, api.event_destroy(event));
  gate.Open();

  EXPECT_EQ(hipSuccess, api.stream_synchronize(stream));
  EXPECT_EQ(hipSuccess, api.stream_destroy(stream));
  EXPECT_EQ(hipSuccess, api.Shutdown());
}

TEST(HipIpcEventApiTest, RejectsCaptureGraphAndExtensionMutationPaths) {
  HipRuntimeApi api;
  ASSERT_TRUE(api.Load());
  ASSERT_EQ(hipSuccess, api.Initialize());

  hipStream_t stream = nullptr;
  ASSERT_EQ(hipSuccess,
            api.stream_create_with_flags(&stream, hipStreamNonBlocking));
  hipEvent_t ipc_event = nullptr;
  ASSERT_EQ(hipSuccess,
            api.event_create_with_flags(
                &ipc_event, hipEventInterprocess | hipEventDisableTiming));
  hipEvent_t ordinary_event = nullptr;
  ASSERT_EQ(hipSuccess, api.event_create_with_flags(&ordinary_event,
                                                    hipEventDisableTiming));

  ASSERT_EQ(hipSuccess,
            api.stream_begin_capture(stream, hipStreamCaptureModeGlobal));
  EXPECT_EQ(hipErrorStreamCaptureUnsupported,
            api.event_record(ipc_event, stream));
  EXPECT_EQ(hipErrorStreamCaptureInvalidated,
            api.stream_end_capture(stream, /*graph=*/nullptr));

  ASSERT_EQ(hipSuccess,
            api.stream_begin_capture(stream, hipStreamCaptureModeGlobal));
  EXPECT_EQ(hipErrorStreamCaptureUnsupported,
            api.stream_wait_event(stream, ipc_event, /*flags=*/0));
  EXPECT_EQ(hipErrorStreamCaptureInvalidated,
            api.stream_end_capture(stream, /*graph=*/nullptr));

  const void* unregistered_function =
      reinterpret_cast<const void*>(uintptr_t{1});
  ASSERT_EQ(hipSuccess,
            api.stream_begin_capture(stream, hipStreamCaptureModeGlobal));
  EXPECT_EQ(hipErrorStreamCaptureUnsupported,
            api.ext_launch_kernel(
                unregistered_function, dim3{1, 1, 1}, dim3{1, 1, 1},
                /*args=*/nullptr, /*shared_memory_bytes=*/0, stream, ipc_event,
                /*stop_event=*/nullptr, /*flags=*/0));
  EXPECT_EQ(hipErrorStreamCaptureInvalidated,
            api.stream_end_capture(stream, /*graph=*/nullptr));
  EXPECT_EQ(
      hipErrorNotSupported,
      api.ext_launch_kernel(unregistered_function, dim3{1, 1, 1}, dim3{1, 1, 1},
                            /*args=*/nullptr, /*shared_memory_bytes=*/0, stream,
                            /*start_event=*/nullptr, ipc_event, /*flags=*/0));
  EXPECT_EQ(
      hipErrorNotSupported,
      api.ext_module_launch_kernel(
          reinterpret_cast<hipFunction_t>(uintptr_t{1}),
          /*global_x=*/1, /*global_y=*/1, /*global_z=*/1,
          /*local_x=*/1, /*local_y=*/1, /*local_z=*/1,
          /*shared_memory_bytes=*/0, stream, /*kernel_params=*/nullptr,
          /*extra=*/nullptr, ipc_event, /*stop_event=*/nullptr, /*flags=*/0));
  ASSERT_EQ(hipSuccess,
            api.stream_begin_capture(stream, hipStreamCaptureModeGlobal));
  EXPECT_EQ(
      hipErrorStreamCaptureUnsupported,
      api.ext_module_launch_kernel(
          reinterpret_cast<hipFunction_t>(uintptr_t{1}),
          /*global_x=*/1, /*global_y=*/1, /*global_z=*/1,
          /*local_x=*/1, /*local_y=*/1, /*local_z=*/1,
          /*shared_memory_bytes=*/0, stream, /*kernel_params=*/nullptr,
          /*extra=*/nullptr, /*start_event=*/nullptr, ipc_event, /*flags=*/0));
  EXPECT_EQ(hipErrorStreamCaptureInvalidated,
            api.stream_end_capture(stream, /*graph=*/nullptr));

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api.graph_create(&graph, /*flags=*/0));
  size_t node_count = 0;
  ASSERT_EQ(hipSuccess,
            api.graph_get_nodes(graph, /*nodes=*/nullptr, &node_count));
  ASSERT_EQ(0u, node_count);

  hipGraphNode_t rejected_node = reinterpret_cast<hipGraphNode_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorNotSupported,
            api.graph_add_event_record_node(&rejected_node, graph,
                                            /*dependencies=*/nullptr,
                                            /*dependency_count=*/0, ipc_event));
  EXPECT_EQ(reinterpret_cast<hipGraphNode_t>(uintptr_t{1}), rejected_node);
  EXPECT_EQ(hipSuccess,
            api.graph_get_nodes(graph, /*nodes=*/nullptr, &node_count));
  EXPECT_EQ(0u, node_count);

  rejected_node = reinterpret_cast<hipGraphNode_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorNotSupported,
            api.graph_add_event_wait_node(&rejected_node, graph,
                                          /*dependencies=*/nullptr,
                                          /*dependency_count=*/0, ipc_event));
  EXPECT_EQ(reinterpret_cast<hipGraphNode_t>(uintptr_t{1}), rejected_node);
  EXPECT_EQ(hipSuccess,
            api.graph_get_nodes(graph, /*nodes=*/nullptr, &node_count));
  EXPECT_EQ(0u, node_count);

  hipGraphNodeParams generic_params = {};
  generic_params.type = hipGraphNodeTypeEventRecord;
  generic_params.eventRecord.event = ipc_event;
  rejected_node = reinterpret_cast<hipGraphNode_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorNotSupported,
            api.graph_add_node(&rejected_node, graph,
                               /*dependencies=*/nullptr,
                               /*dependency_count=*/0, &generic_params));
  EXPECT_EQ(reinterpret_cast<hipGraphNode_t>(uintptr_t{1}), rejected_node);
  EXPECT_EQ(hipSuccess,
            api.graph_get_nodes(graph, /*nodes=*/nullptr, &node_count));
  EXPECT_EQ(0u, node_count);

  generic_params = {};
  generic_params.type = hipGraphNodeTypeWaitEvent;
  generic_params.eventWait.event = ipc_event;
  rejected_node = reinterpret_cast<hipGraphNode_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorNotSupported,
            api.graph_add_node(&rejected_node, graph,
                               /*dependencies=*/nullptr,
                               /*dependency_count=*/0, &generic_params));
  EXPECT_EQ(reinterpret_cast<hipGraphNode_t>(uintptr_t{1}), rejected_node);
  EXPECT_EQ(hipSuccess,
            api.graph_get_nodes(graph, /*nodes=*/nullptr, &node_count));
  EXPECT_EQ(0u, node_count);

  hipGraphNode_t record_node = nullptr;
  ASSERT_EQ(hipSuccess, api.graph_add_event_record_node(
                            &record_node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, ordinary_event));
  hipGraphNode_t wait_node = nullptr;
  ASSERT_EQ(hipSuccess, api.graph_add_event_wait_node(
                            &wait_node, graph, &record_node,
                            /*dependency_count=*/1, ordinary_event));
  EXPECT_EQ(hipErrorNotSupported,
            api.graph_event_record_node_set_event(record_node, ipc_event));
  EXPECT_EQ(hipErrorNotSupported,
            api.graph_event_wait_node_set_event(wait_node, ipc_event));
  hipEvent_t configured_event = nullptr;
  EXPECT_EQ(hipSuccess, api.graph_event_record_node_get_event(
                            record_node, &configured_event));
  EXPECT_EQ(ordinary_event, configured_event);
  EXPECT_EQ(hipSuccess,
            api.graph_event_wait_node_get_event(wait_node, &configured_event));
  EXPECT_EQ(ordinary_event, configured_event);

  hipGraphExec_t graph_exec = nullptr;
  ASSERT_EQ(hipSuccess,
            api.graph_instantiate(&graph_exec, graph, /*error_node=*/nullptr,
                                  /*log_buffer=*/nullptr, /*buffer_size=*/0));
  EXPECT_EQ(hipErrorNotSupported, api.graph_exec_event_record_node_set_event(
                                      graph_exec, record_node, ipc_event));
  EXPECT_EQ(hipErrorNotSupported, api.graph_exec_event_wait_node_set_event(
                                      graph_exec, wait_node, ipc_event));

  EXPECT_EQ(hipSuccess, api.graph_exec_destroy(graph_exec));
  EXPECT_EQ(hipSuccess, api.graph_destroy(graph));
  EXPECT_EQ(hipSuccess, api.event_destroy(ordinary_event));
  EXPECT_EQ(hipSuccess, api.event_destroy(ipc_event));
  EXPECT_EQ(hipSuccess, api.stream_destroy(stream));
  EXPECT_EQ(hipSuccess, api.Shutdown());
}

void ExpectCaptureStatus(HipRuntimeApi& api, hipStream_t stream,
                         hipStreamCaptureStatus expected_status) {
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  const hipError_t result = api.stream_is_capturing(stream, &capture_status);
  EXPECT_EQ(hipSuccess, result);
  if (result == hipSuccess) {
    EXPECT_EQ(expected_status, capture_status);
  }
}

TEST(HipIpcEventApiTest,
     StaleCapturedEventOperationsLeaveSameContextSuccessorActive) {
  HipRuntimeApi api;
  ASSERT_TRUE(api.Load());
  ASSERT_EQ(hipSuccess, api.Initialize());

  hipStream_t source_stream = nullptr;
  hipStream_t wait_stream = nullptr;
  hipEvent_t start_event = nullptr;
  hipEvent_t stop_event = nullptr;
  hipGraph_t graph = nullptr;
  hipGraph_t successor_graph = nullptr;
  ScopeExit cleanup([&] {
    if (source_stream) {
      hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
      if (api.stream_is_capturing(source_stream, &capture_status) ==
              hipSuccess &&
          capture_status != hipStreamCaptureStatusNone) {
        hipGraph_t cleanup_graph = nullptr;
        (void)api.stream_end_capture(source_stream, &cleanup_graph);
        if (cleanup_graph && cleanup_graph != graph &&
            cleanup_graph != successor_graph) {
          EXPECT_EQ(hipSuccess, api.graph_destroy(cleanup_graph));
        }
      }
    }
    if (stop_event) {
      EXPECT_EQ(hipSuccess, api.event_destroy(stop_event));
    }
    if (start_event) {
      EXPECT_EQ(hipSuccess, api.event_destroy(start_event));
    }
    if (wait_stream) {
      EXPECT_EQ(hipSuccess, api.stream_destroy(wait_stream));
    }
    if (source_stream) {
      EXPECT_EQ(hipSuccess, api.stream_destroy(source_stream));
    }
    if (successor_graph && successor_graph != graph) {
      EXPECT_EQ(hipSuccess, api.graph_destroy(successor_graph));
    }
    if (graph) {
      EXPECT_EQ(hipSuccess, api.graph_destroy(graph));
    }
    EXPECT_EQ(hipSuccess, api.Shutdown());
  });

  ASSERT_EQ(hipSuccess,
            api.stream_create_with_flags(&source_stream, hipStreamNonBlocking));
  ASSERT_EQ(hipSuccess,
            api.stream_create_with_flags(&wait_stream, hipStreamNonBlocking));
  ASSERT_EQ(hipSuccess,
            api.event_create_with_flags(&start_event, hipEventDefault));
  ASSERT_EQ(hipSuccess,
            api.event_create_with_flags(&stop_event, hipEventDefault));

  ASSERT_EQ(hipSuccess, api.stream_begin_capture(source_stream,
                                                 hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess, api.event_record(start_event, source_stream));
  ASSERT_EQ(hipSuccess, api.event_record(stop_event, source_stream));
  ASSERT_EQ(hipSuccess, api.stream_end_capture(source_stream, &graph));
  ASSERT_NE(nullptr, graph);

  ASSERT_EQ(hipSuccess, api.stream_begin_capture_to_graph(
                            source_stream, graph, /*dependencies=*/nullptr,
                            /*dependency_data=*/nullptr, /*dependency_count=*/0,
                            hipStreamCaptureModeGlobal));
  EXPECT_EQ(hipErrorCapturedEvent, api.event_query(start_event));
  ExpectCaptureStatus(api, source_stream, hipStreamCaptureStatusActive);
  EXPECT_EQ(hipErrorCapturedEvent, api.event_synchronize(start_event));
  ExpectCaptureStatus(api, source_stream, hipStreamCaptureStatusActive);
  float milliseconds = -1.0f;
  EXPECT_EQ(hipErrorCapturedEvent,
            api.event_elapsed_time(&milliseconds, start_event, stop_event));
  ExpectCaptureStatus(api, source_stream, hipStreamCaptureStatusActive);

  EXPECT_EQ(hipErrorInvalidValue,
            api.stream_wait_event(wait_stream, start_event, /*flags=*/0));
  ExpectCaptureStatus(api, wait_stream, hipStreamCaptureStatusNone);
  ExpectCaptureStatus(api, source_stream, hipStreamCaptureStatusActive);

  EXPECT_EQ(hipSuccess,
            api.stream_end_capture(source_stream, &successor_graph));
  EXPECT_EQ(graph, successor_graph);
}

TEST(HipIpcEventApiTest,
     StaleCapturedEventOperationsLeaveDifferentContextSuccessorActive) {
  HipRuntimeApi api;
  ASSERT_TRUE(api.Load());
  ASSERT_EQ(hipSuccess, api.Initialize());

  hipCtx_t origin_context = nullptr;
  hipCtx_t successor_context = nullptr;
  hipStream_t origin_stream = nullptr;
  hipStream_t successor_stream = nullptr;
  hipEvent_t start_event = nullptr;
  hipEvent_t stop_event = nullptr;
  hipGraph_t graph = nullptr;
  hipGraph_t successor_graph = nullptr;
  ScopeExit cleanup([&] {
    if (successor_stream) {
      hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
      if (api.stream_is_capturing(successor_stream, &capture_status) ==
              hipSuccess &&
          capture_status != hipStreamCaptureStatusNone) {
        hipGraph_t cleanup_graph = nullptr;
        (void)api.stream_end_capture(successor_stream, &cleanup_graph);
        if (cleanup_graph && cleanup_graph != graph &&
            cleanup_graph != successor_graph) {
          EXPECT_EQ(hipSuccess, api.graph_destroy(cleanup_graph));
        }
      }
    }
    if (origin_stream) {
      hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
      if (api.stream_is_capturing(origin_stream, &capture_status) ==
              hipSuccess &&
          capture_status != hipStreamCaptureStatusNone) {
        hipGraph_t cleanup_graph = nullptr;
        (void)api.stream_end_capture(origin_stream, &cleanup_graph);
        if (cleanup_graph && cleanup_graph != graph &&
            cleanup_graph != successor_graph) {
          EXPECT_EQ(hipSuccess, api.graph_destroy(cleanup_graph));
        }
      }
    }
    if (stop_event) {
      EXPECT_EQ(hipSuccess, api.event_destroy(stop_event));
    }
    if (start_event) {
      EXPECT_EQ(hipSuccess, api.event_destroy(start_event));
    }
    if (successor_stream) {
      EXPECT_EQ(hipSuccess, api.stream_destroy(successor_stream));
    }
    if (origin_stream) {
      EXPECT_EQ(hipSuccess, api.stream_destroy(origin_stream));
    }
    if (successor_graph && successor_graph != graph) {
      EXPECT_EQ(hipSuccess, api.graph_destroy(successor_graph));
    }
    if (graph) {
      EXPECT_EQ(hipSuccess, api.graph_destroy(graph));
    }
    if (successor_context) {
      EXPECT_EQ(hipSuccess, api.context_destroy(successor_context));
    }
    if (origin_context) {
      EXPECT_EQ(hipSuccess, api.context_destroy(origin_context));
    }
    EXPECT_EQ(hipSuccess, api.Shutdown());
  });

  int device = -1;
  ASSERT_EQ(hipSuccess, api.get_device(&device));
  ASSERT_EQ(hipSuccess,
            api.context_create(&origin_context, /*flags=*/0, device));
  ASSERT_EQ(hipSuccess,
            api.stream_create_with_flags(&origin_stream, hipStreamNonBlocking));
  ASSERT_EQ(hipSuccess,
            api.event_create_with_flags(&start_event, hipEventDefault));
  ASSERT_EQ(hipSuccess,
            api.event_create_with_flags(&stop_event, hipEventDefault));

  ASSERT_EQ(hipSuccess, api.stream_begin_capture(origin_stream,
                                                 hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess, api.event_record(start_event, origin_stream));
  ASSERT_EQ(hipSuccess, api.event_record(stop_event, origin_stream));
  ASSERT_EQ(hipSuccess, api.stream_end_capture(origin_stream, &graph));
  ASSERT_NE(nullptr, graph);

  ASSERT_EQ(hipSuccess,
            api.context_create(&successor_context, /*flags=*/0, device));
  ASSERT_EQ(hipSuccess, api.stream_create_with_flags(&successor_stream,
                                                     hipStreamNonBlocking));
  ASSERT_EQ(hipSuccess, api.stream_begin_capture_to_graph(
                            successor_stream, graph, /*dependencies=*/nullptr,
                            /*dependency_data=*/nullptr, /*dependency_count=*/0,
                            hipStreamCaptureModeGlobal));

  EXPECT_EQ(hipErrorCapturedEvent, api.event_query(start_event));
  ExpectCaptureStatus(api, successor_stream, hipStreamCaptureStatusActive);
  EXPECT_EQ(hipErrorCapturedEvent, api.event_synchronize(start_event));
  ExpectCaptureStatus(api, successor_stream, hipStreamCaptureStatusActive);
  float milliseconds = -1.0f;
  EXPECT_EQ(hipErrorCapturedEvent,
            api.event_elapsed_time(&milliseconds, start_event, stop_event));
  ExpectCaptureStatus(api, successor_stream, hipStreamCaptureStatusActive);

  EXPECT_EQ(hipSuccess,
            api.stream_end_capture(successor_stream, &successor_graph));
  EXPECT_EQ(graph, successor_graph);
}

bool RunCompletedEventCycle(HipRuntimeApi& api) {
  hipEvent_t event = nullptr;
  if (api.event_create_with_flags(
          &event, hipEventInterprocess | hipEventDisableTiming) != hipSuccess) {
    return false;
  }
  const bool succeeded =
      api.event_record(event, /*stream=*/nullptr) == hipSuccess &&
      api.event_synchronize(event) == hipSuccess;
  return api.event_destroy(event) == hipSuccess && succeeded;
}

TEST(HipIpcEventApiTest, RestartsAfterRuntimeShutdown) {
  HipRuntimeApi api;
  ASSERT_TRUE(api.Load());
  ASSERT_EQ(hipSuccess, api.Initialize());
  EXPECT_TRUE(RunCompletedEventCycle(api));
  ASSERT_EQ(hipSuccess, api.Shutdown());

  ASSERT_EQ(hipSuccess, api.Initialize());
  EXPECT_TRUE(RunCompletedEventCycle(api));
  EXPECT_EQ(hipSuccess, api.Shutdown());
}

bool MakePath(const char* temp_directory, const char* filename, char* out_path,
              size_t path_capacity) {
  const int length =
      std::snprintf(out_path, path_capacity, "%s/%s", temp_directory, filename);
  return length >= 0 && static_cast<size_t>(length) < path_capacity;
}

bool WriteFile(const char* temp_directory, const char* filename,
               const void* data, size_t data_length) {
  char path[1024];
  if (!MakePath(temp_directory, filename, path, sizeof(path))) return false;
  FILE* file = std::fopen(path, "wb");
  if (!file) return false;
  const bool wrote = data_length == 0 ||
                     std::fwrite(data, 1, data_length, file) == data_length;
  return std::fclose(file) == 0 && wrote;
}

bool ReadFile(const char* temp_directory, const char* filename, void* data,
              size_t data_length) {
  char path[1024];
  if (!MakePath(temp_directory, filename, path, sizeof(path))) return false;
  FILE* file = std::fopen(path, "rb");
  if (!file) return false;
  const bool read = std::fread(data, 1, data_length, file) == data_length;
  return std::fclose(file) == 0 && read;
}

bool MakeFifo(const char* temp_directory, const char* filename) {
  char path[1024];
  if (!MakePath(temp_directory, filename, path, sizeof(path))) return false;
  return mkfifo(path, 0600) == 0;
}

int OpenFifo(const char* path, int flags) {
  int file_descriptor = -1;
  do {
    file_descriptor = open(path, flags);
  } while (file_descriptor < 0 && errno == EINTR);
  return file_descriptor;
}

bool WaitFifo(const char* temp_directory, const char* filename) {
  char path[1024];
  if (!MakePath(temp_directory, filename, path, sizeof(path))) return false;
  // A blocking FIFO open pairs directly with the consumer's writer open. The
  // byte transfer is the readiness contract; no scheduling delay is assumed.
  const int file_descriptor = OpenFifo(path, O_RDONLY);
  if (file_descriptor < 0) return false;

  char byte = 0;
  ssize_t read_count = -1;
  do {
    read_count = read(file_descriptor, &byte, 1);
  } while (read_count < 0 && errno == EINTR);
  const int close_result = close(file_descriptor);
  return read_count == 1 && byte == 'R' && close_result == 0;
}

bool SignalFifo(const char* temp_directory, const char* filename) {
  char path[1024];
  if (!MakePath(temp_directory, filename, path, sizeof(path))) return false;
  const int file_descriptor = OpenFifo(path, O_WRONLY);
  if (file_descriptor < 0) return false;
  const char byte = 'R';
  ssize_t write_count = -1;
  do {
    write_count = write(file_descriptor, &byte, 1);
  } while (write_count < 0 && errno == EINTR);
  const int close_result = close(file_descriptor);
  return write_count == 1 && close_result == 0;
}

bool RoleHipSuccess(hipError_t result, const char* operation) {
  if (result == hipSuccess) return true;
  std::fprintf(stderr, "%s failed: %d\n", operation, static_cast<int>(result));
  return false;
}

bool DestroyImportedEventsConcurrently(HipRuntimeApi& api, hipEvent_t first,
                                       hipEvent_t second) {
  std::mutex mutex;
  std::condition_variable condition;
  int ready_count = 0;
  bool start = false;
  auto wait_for_start = [&] {
    std::unique_lock<std::mutex> lock(mutex);
    ++ready_count;
    condition.notify_all();
    condition.wait(lock, [&] { return start; });
  };

  hipError_t first_result = hipErrorUnknown;
  hipError_t second_result = hipErrorUnknown;
  std::thread first_destroy([&] {
    wait_for_start();
    first_result = api.event_destroy(first);
  });
  std::thread second_destroy([&] {
    wait_for_start();
    second_result = api.event_destroy(second);
  });
  {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return ready_count == 2; });
    start = true;
  }
  condition.notify_all();
  first_destroy.join();
  second_destroy.join();
  return first_result == hipSuccess && second_result == hipSuccess;
}

int ProducerRole(int argc, char** argv, const char* temp_directory) {
  (void)argc;
  (void)argv;
  HipRuntimeApi api;
  if (!api.Load() || !RoleHipSuccess(api.Initialize(), "hipInit")) return 1;

  hipEvent_t event = nullptr;
  if (!RoleHipSuccess(api.event_create_with_flags(
                          &event, hipEventInterprocess | hipEventDisableTiming),
                      "hipEventCreateWithFlags")) {
    return 1;
  }
  if (!RoleHipSuccess(api.event_query(event), "pre-export hipEventQuery")) {
    return 1;
  }

  hipIpcEventHandle_t handle = {};
  if (!RoleHipSuccess(api.ipc_get_event_handle(&handle, event),
                      "hipIpcGetEventHandle") ||
      ReadWireValue<uint32_t>(handle, 0) != 1 ||
      ReadWireValue<int32_t>(handle, 4) != static_cast<int32_t>(getpid()) ||
      !IsZeroed(handle.reserved + 40, 24) ||
      api.event_query(event) != hipErrorNotReady ||
      !WriteFile(temp_directory, "event.handle", &handle, sizeof(handle)) ||
      !MakeFifo(temp_directory, "consumer.armed")) {
    std::fprintf(stderr, "producer failed stock handle validation\n");
    return 1;
  }

  iree_coordinated_test_signal_ready(temp_directory);
  if (!WaitFifo(temp_directory, "consumer.armed")) {
    std::fprintf(stderr, "consumer did not arm its event wait\n");
    return 1;
  }
  if (!RoleHipSuccess(api.event_record(event, /*stream=*/nullptr),
                      "hipEventRecord") ||
      !RoleHipSuccess(api.event_synchronize(event), "hipEventSynchronize") ||
      !RoleHipSuccess(api.event_destroy(event), "hipEventDestroy") ||
      !RoleHipSuccess(api.Shutdown(), "hipHALDeinit")) {
    return 1;
  }
  return 0;
}

int ConsumerRole(int argc, char** argv, const char* temp_directory) {
  (void)argc;
  (void)argv;
  HipRuntimeApi api;
  if (!api.Load() || !RoleHipSuccess(api.Initialize(), "hipInit")) return 1;

  hipIpcEventHandle_t handle = {};
  if (!ReadFile(temp_directory, "event.handle", &handle, sizeof(handle))) {
    std::fprintf(stderr, "consumer could not read event handle\n");
    return 1;
  }
  hipEvent_t first_event = nullptr;
  hipEvent_t second_event = nullptr;
  if (!RoleHipSuccess(api.ipc_open_event_handle(&first_event, handle),
                      "first hipIpcOpenEventHandle") ||
      !RoleHipSuccess(api.ipc_open_event_handle(&second_event, handle),
                      "second hipIpcOpenEventHandle") ||
      first_event == second_event ||
      api.event_query(first_event) != hipErrorNotReady ||
      api.event_query(second_event) != hipErrorNotReady) {
    std::fprintf(stderr, "consumer failed duplicate import validation\n");
    return 1;
  }

  hipIpcEventHandle_t reexported_handle = {};
  if (!RoleHipSuccess(api.ipc_get_event_handle(&reexported_handle, first_event),
                      "re-export hipIpcGetEventHandle") ||
      ReadWireValue<uint32_t>(reexported_handle, 0) != 1 ||
      ReadWireValue<int32_t>(reexported_handle, 4) !=
          static_cast<int32_t>(getpid()) ||
      std::memcmp(reexported_handle.reserved + 8, handle.reserved + 8, 32) !=
          0 ||
      !IsZeroed(reexported_handle.reserved + 40, 24)) {
    std::fprintf(stderr, "consumer failed re-export validation\n");
    return 1;
  }
  hipEvent_t same_process_event = reinterpret_cast<hipEvent_t>(uintptr_t{1});
  if (api.ipc_open_event_handle(&same_process_event, reexported_handle) !=
          hipErrorInvalidContext ||
      same_process_event != nullptr) {
    std::fprintf(stderr, "consumer accepted a same-process event handle\n");
    return 1;
  }

  int device = -1;
  hipCtx_t wait_context = nullptr;
  hipStream_t wait_stream = nullptr;
  if (!RoleHipSuccess(api.get_device(&device), "hipGetDevice") ||
      !RoleHipSuccess(api.context_create(&wait_context, /*flags=*/0, device),
                      "hipCtxCreate") ||
      !RoleHipSuccess(
          api.stream_create_with_flags(&wait_stream, hipStreamNonBlocking),
          "hipStreamCreateWithFlags") ||
      !RoleHipSuccess(api.stream_wait_event(wait_stream, first_event,
                                            /*flags=*/0),
                      "hipStreamWaitEvent")) {
    return 1;
  }

  if (!SignalFifo(temp_directory, "consumer.armed")) {
    std::fprintf(stderr, "consumer could not publish readiness\n");
    return 1;
  }
  if (!RoleHipSuccess(api.stream_synchronize(wait_stream),
                      "hipStreamSynchronize") ||
      !RoleHipSuccess(api.event_query(first_event), "first hipEventQuery") ||
      !RoleHipSuccess(api.event_query(second_event), "second hipEventQuery") ||
      !RoleHipSuccess(api.stream_destroy(wait_stream), "hipStreamDestroy") ||
      !RoleHipSuccess(api.context_destroy(wait_context), "hipCtxDestroy") ||
      !DestroyImportedEventsConcurrently(api, first_event, second_event) ||
      !RoleHipSuccess(api.Shutdown(), "hipHALDeinit")) {
    return 1;
  }

  return 0;
}

static const iree_test_role_t kIpcEventRoles[] = {
    {"producer", ProducerRole, /*signals_ready=*/true},
    {"consumer", ConsumerRole, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kIpcEventConfig = {
    /*.roles=*/kIpcEventRoles,
    /*.role_count=*/2,
};
IREE_COORDINATED_TEST_REGISTER(kIpcEventConfig);

TEST(HipIpcEventApiTest, SharesPreRecordEventAcrossProcessesAndContexts) {
  EXPECT_EQ(0, iree_coordinated_test_run(iree_coordinated_test_argc(),
                                         iree_coordinated_test_argv(),
                                         &kIpcEventConfig));
}

enum class IpcCaptureOperation {
  kRecord,
  kWait,
};

void ExpectIpcOperationInvalidatesJoinedCapture(IpcCaptureOperation operation) {
  HipRuntimeApi api;
  ASSERT_TRUE(api.Load());
  ASSERT_EQ(hipSuccess, api.Initialize());

  hipCtx_t origin_context = nullptr;
  hipCtx_t participant_context = nullptr;
  hipStream_t origin_stream = nullptr;
  hipStream_t participant_stream = nullptr;
  hipEvent_t ordinary_event = nullptr;
  hipEvent_t ipc_event = nullptr;
  hipGraph_t unexpected_graph = nullptr;
  ScopeExit cleanup([&] {
    if (origin_stream) {
      hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
      if (api.stream_is_capturing(origin_stream, &capture_status) ==
              hipSuccess &&
          capture_status != hipStreamCaptureStatusNone) {
        hipGraph_t cleanup_graph = nullptr;
        (void)api.stream_end_capture(origin_stream, &cleanup_graph);
        if (cleanup_graph && cleanup_graph != unexpected_graph) {
          EXPECT_EQ(hipSuccess, api.graph_destroy(cleanup_graph));
        }
      }
    }
    if (ipc_event) {
      EXPECT_EQ(hipSuccess, api.event_destroy(ipc_event));
    }
    if (ordinary_event) {
      EXPECT_EQ(hipSuccess, api.event_destroy(ordinary_event));
    }
    if (participant_stream) {
      EXPECT_EQ(hipSuccess, api.stream_destroy(participant_stream));
    }
    if (origin_stream) {
      EXPECT_EQ(hipSuccess, api.stream_destroy(origin_stream));
    }
    if (unexpected_graph) {
      EXPECT_EQ(hipSuccess, api.graph_destroy(unexpected_graph));
    }
    if (participant_context) {
      EXPECT_EQ(hipSuccess, api.context_destroy(participant_context));
    }
    if (origin_context) {
      EXPECT_EQ(hipSuccess, api.context_destroy(origin_context));
    }
    EXPECT_EQ(hipSuccess, api.Shutdown());
  });

  int device = -1;
  ASSERT_EQ(hipSuccess, api.get_device(&device));
  ASSERT_EQ(hipSuccess,
            api.context_create(&origin_context, /*flags=*/0, device));
  ASSERT_EQ(hipSuccess,
            api.stream_create_with_flags(&origin_stream, hipStreamNonBlocking));
  ASSERT_EQ(hipSuccess, api.event_create_with_flags(&ordinary_event,
                                                    hipEventDisableTiming));

  ASSERT_EQ(hipSuccess,
            api.context_create(&participant_context, /*flags=*/0, device));
  ASSERT_EQ(hipSuccess, api.stream_create_with_flags(&participant_stream,
                                                     hipStreamNonBlocking));
  ASSERT_EQ(hipSuccess,
            api.event_create_with_flags(
                &ipc_event, hipEventInterprocess | hipEventDisableTiming));

  ASSERT_EQ(hipSuccess, api.stream_begin_capture(origin_stream,
                                                 hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess, api.event_record(ordinary_event, origin_stream));
  ASSERT_EQ(hipSuccess, api.stream_wait_event(participant_stream,
                                              ordinary_event, /*flags=*/0));

  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  ASSERT_EQ(hipSuccess,
            api.stream_is_capturing(origin_stream, &capture_status));
  EXPECT_EQ(hipStreamCaptureStatusActive, capture_status);
  ASSERT_EQ(hipSuccess,
            api.stream_is_capturing(participant_stream, &capture_status));
  EXPECT_EQ(hipStreamCaptureStatusActive, capture_status);

  const hipError_t rejection_result =
      operation == IpcCaptureOperation::kRecord
          ? api.event_record(ipc_event, participant_stream)
          : api.stream_wait_event(participant_stream, ipc_event, /*flags=*/0);
  EXPECT_EQ(hipErrorStreamCaptureUnsupported, rejection_result);

  ASSERT_EQ(hipSuccess,
            api.stream_is_capturing(participant_stream, &capture_status));
  EXPECT_EQ(hipStreamCaptureStatusInvalidated, capture_status);
  ASSERT_EQ(hipSuccess,
            api.stream_is_capturing(origin_stream, &capture_status));
  EXPECT_EQ(hipStreamCaptureStatusInvalidated, capture_status);
  EXPECT_EQ(hipErrorStreamCaptureInvalidated,
            api.stream_end_capture(origin_stream, &unexpected_graph));
}

TEST(HipIpcEventApiTest, IpcRecordInvalidatesCrossContextCaptureParticipants) {
  ExpectIpcOperationInvalidatesJoinedCapture(IpcCaptureOperation::kRecord);
}

TEST(HipIpcEventApiTest, IpcWaitInvalidatesCrossContextCaptureParticipants) {
  ExpectIpcOperationInvalidatesJoinedCapture(IpcCaptureOperation::kWait);
}

}  // namespace
