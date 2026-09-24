// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>

#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"

namespace {

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
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipGetDevicePropertiesR0600Fn =
    hipError_t (*)(hipDeviceProp_t* properties, int device);
using HipDeviceGetAttributeFn = hipError_t (*)(int* value,
                                               hipDeviceAttribute_t attribute,
                                               int device);
using HipDeviceGetDevResourceFn = hipError_t (*)(hipDevice_t device,
                                                 hipDevResource* resource,
                                                 hipDevResourceType type);
using HipDeviceGetExecutionCtxFn = hipError_t (*)(hipExecutionCtx_t* context,
                                                  int device);
using HipDevSmResourceSplitByCountFn =
    hipError_t (*)(hipDevResource* result, unsigned int* group_count,
                   const hipDevResource* input, hipDevResource* remainder,
                   unsigned int flags, unsigned int minimum_count);
using HipDevSmResourceSplitFn = hipError_t (*)(
    hipDevResource* result, unsigned int group_count,
    const hipDevResource* input, hipDevResource* remainder, unsigned int flags,
    hipDevSmResourceGroupParams* group_parameters);
using HipDevResourceGenerateDescFn =
    hipError_t (*)(hipDevResourceDesc_t* descriptor, hipDevResource* resources,
                   unsigned int resource_count);
using HipGreenCtxCreateFn = hipError_t (*)(hipExecutionCtx_t* context,
                                           hipDevResourceDesc_t descriptor,
                                           int device, unsigned int flags);
using HipExecutionCtxDestroyFn = hipError_t (*)(hipExecutionCtx_t context);
using HipExecutionCtxGetDevResourceFn =
    hipError_t (*)(hipExecutionCtx_t context, hipDevResource* resource,
                   hipDevResourceType type);
using HipExecutionCtxGetDeviceFn = hipError_t (*)(hipDevice_t* device,
                                                  hipExecutionCtx_t context);
using HipExecutionCtxGetIdFn = hipError_t (*)(hipExecutionCtx_t context,
                                              unsigned long long* context_id);
using HipExecutionCtxStreamCreateFn = hipError_t (*)(hipStream_t* stream,
                                                     hipExecutionCtx_t context,
                                                     unsigned int flags,
                                                     int priority);
using HipExecutionCtxRecordEventFn = hipError_t (*)(hipExecutionCtx_t context,
                                                    hipEvent_t event);
using HipExecutionCtxWaitEventFn = hipError_t (*)(hipExecutionCtx_t context,
                                                  hipEvent_t event);
using HipExecutionCtxSynchronizeFn = hipError_t (*)(hipExecutionCtx_t context);
using HipDeviceGetStreamPriorityRangeFn =
    hipError_t (*)(int* least_priority, int* greatest_priority);
using HipStreamCreateWithPriorityFn = hipError_t (*)(hipStream_t* stream,
                                                     unsigned int flags,
                                                     int priority);
using HipExtStreamCreateWithCUMaskFn = hipError_t (*)(hipStream_t* stream,
                                                      uint32_t mask_count,
                                                      const uint32_t* mask);
using HipExtStreamGetCUMaskFn = hipError_t (*)(hipStream_t stream,
                                               uint32_t mask_count,
                                               uint32_t* mask);
using HipStreamGetDevResourceFn = hipError_t (*)(hipStream_t stream,
                                                 hipDevResource* resource,
                                                 hipDevResourceType type);
using HipStreamGetFlagsFn = hipError_t (*)(hipStream_t stream,
                                           unsigned int* flags);
using HipStreamGetPriorityFn = hipError_t (*)(hipStream_t stream,
                                              int* priority);
using HipStreamBeginCaptureFn = hipError_t (*)(hipStream_t stream,
                                               hipStreamCaptureMode mode);
using HipStreamEndCaptureFn = hipError_t (*)(hipStream_t stream,
                                             hipGraph_t* graph);
using HipStreamIsCapturingFn =
    hipError_t (*)(hipStream_t stream, hipStreamCaptureStatus* capture_status);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t stream);
using HipStreamWriteValue32Fn = hipError_t (*)(hipStream_t stream,
                                               void* pointer, uint32_t value,
                                               unsigned int flags);
using HipStreamWriteValue64Fn = hipError_t (*)(hipStream_t stream,
                                               void* pointer, uint64_t value,
                                               unsigned int flags);
using HipStreamWaitValue32Fn = hipError_t (*)(hipStream_t stream, void* pointer,
                                              uint32_t value,
                                              unsigned int flags,
                                              uint32_t mask);
using HipStreamWaitValue64Fn = hipError_t (*)(hipStream_t stream, void* pointer,
                                              uint64_t value,
                                              unsigned int flags,
                                              uint64_t mask);
using HipStreamBatchMemOpFn =
    hipError_t (*)(hipStream_t stream, unsigned int count,
                   hipStreamBatchMemOpParams* parameters, unsigned int flags);
using HipLaunchHostFuncFn = hipError_t (*)(hipStream_t stream, hipHostFn_t fn,
                                           void* user_data);
using HipLaunchKernelExCFn = hipError_t (*)(const hipLaunchConfig_t* config,
                                            const void* function,
                                            void** arguments);
using HipDrvLaunchKernelExFn = hipError_t (*)(const HIP_LAUNCH_CONFIG* config,
                                              hipFunction_t function,
                                              void** kernel_parameters,
                                              void** extra);
using HipModuleLaunchCooperativeKernelFn = hipError_t (*)(
    hipFunction_t function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_memory_bytes,
    hipStream_t stream, void** kernel_parameters);
using HipExtLaunchMultiKernelMultiDeviceFn = hipError_t (*)(
    hipLaunchParams* launch_parameters, int device_count, unsigned int flags);
using HipGraphCreateFn = hipError_t (*)(hipGraph_t* graph, unsigned int flags);
using HipGraphAddEmptyNodeFn =
    hipError_t (*)(hipGraphNode_t* node, hipGraph_t graph,
                   const hipGraphNode_t* dependencies, size_t dependency_count);
using HipGraphInstantiateFn = hipError_t (*)(hipGraphExec_t* executable,
                                             hipGraph_t graph,
                                             hipGraphNode_t* error_node,
                                             char* log_buffer,
                                             size_t buffer_size);
using HipGraphLaunchFn = hipError_t (*)(hipGraphExec_t executable,
                                        hipStream_t stream);
using HipGraphExecDestroyFn = hipError_t (*)(hipGraphExec_t executable);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t graph);
using HipEventCreateWithFlagsFn = hipError_t (*)(hipEvent_t* event,
                                                 unsigned int flags);
using HipEventDestroyFn = hipError_t (*)(hipEvent_t event);
using HipEventRecordFn = hipError_t (*)(hipEvent_t event, hipStream_t stream);
using HipEventQueryFn = hipError_t (*)(hipEvent_t event);
using HipEventSynchronizeFn = hipError_t (*)(hipEvent_t event);

struct ExecutionContextDeleter {
  // Runtime entry point used to destroy a live execution context.
  HipExecutionCtxDestroyFn destroy = nullptr;

  void operator()(ihipExecutionCtx_t* context) const {
    if (context) {
      const hipError_t result = destroy(context);
      EXPECT_EQ(hipSuccess, result);
    }
  }
};

using ScopedExecutionContext =
    std::unique_ptr<ihipExecutionCtx_t, ExecutionContextDeleter>;

struct StreamDeleter {
  // Runtime entry point used to destroy a live stream.
  HipStreamDestroyFn destroy = nullptr;

  void operator()(hipStream_st* stream) const {
    if (stream) {
      const hipError_t result = destroy(stream);
      EXPECT_EQ(hipSuccess, result);
    }
  }
};

using ScopedStream = std::unique_ptr<hipStream_st, StreamDeleter>;

struct EventDeleter {
  // Runtime entry point used to destroy a live event.
  HipEventDestroyFn destroy = nullptr;

  void operator()(hipEvent_st* event) const {
    if (event) {
      const hipError_t result = destroy(event);
      EXPECT_EQ(hipSuccess, result);
    }
  }
};

using ScopedEvent = std::unique_ptr<hipEvent_st, EventDeleter>;

// Host callback that blocks stream progress on an explicit test-controlled
// condition. Destruction releases the gate so a fatal assertion cannot strand
// asynchronous work during fixture cleanup.
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

  // Serializes gate state accessed by the test and callback threads.
  std::mutex mutex_;

  // Notifies entry and release state transitions.
  std::condition_variable condition_;

  // True once the stream callback has begun waiting.
  bool entered_ = false;

  // True once the callback is allowed to return.
  bool is_open_ = false;
};

// Owns the process-scoped HIP runtime under test and its resource entry points.
// Calls cross the shared-library ABI instead of linking the implementation.
struct HipRuntimeApi {
  // Handle returned by dlopen for the HIP runtime instance.
  void* library = nullptr;

  // Initializes the HIP runtime instance.
  HipInitFn init = nullptr;

  // Returns the calling thread's current device ordinal.
  HipGetDeviceFn get_device = nullptr;

  // Queries the aggregate properties of one device.
  HipGetDevicePropertiesR0600Fn get_device_properties = nullptr;

  // Queries one property of one device.
  HipDeviceGetAttributeFn device_get_attribute = nullptr;

  // Queries the process-visible execution resource of one device.
  HipDeviceGetDevResourceFn device_get_resource = nullptr;

  // Returns the process-managed primary execution context for one device.
  HipDeviceGetExecutionCtxFn device_execution_context = nullptr;

  // Splits one exact SM resource into equal-size partitions.
  HipDevSmResourceSplitByCountFn split_sm_by_count = nullptr;

  // Splits one exact SM resource into caller-shaped partitions.
  HipDevSmResourceSplitFn split_sm = nullptr;

  // Generates a one-shot descriptor from exact execution resources.
  HipDevResourceGenerateDescFn generate_descriptor = nullptr;

  // Creates a resource-partitioned execution context.
  HipGreenCtxCreateFn create_context = nullptr;

  // Destroys a resource-partitioned execution context.
  HipExecutionCtxDestroyFn destroy_context = nullptr;

  // Queries the canonical resource owned by an execution context.
  HipExecutionCtxGetDevResourceFn context_get_resource = nullptr;

  // Queries the device ordinal associated with an execution context.
  HipExecutionCtxGetDeviceFn context_get_device = nullptr;

  // Queries the process-unique execution-context identifier.
  HipExecutionCtxGetIdFn context_get_id = nullptr;

  // Creates a stream on an exact execution context.
  HipExecutionCtxStreamCreateFn context_stream_create = nullptr;

  // Records an event after all current execution-context work.
  HipExecutionCtxRecordEventFn context_record_event = nullptr;

  // Orders current and future execution-context work after an event.
  HipExecutionCtxWaitEventFn context_wait_event = nullptr;

  // Synchronizes all current execution-context work.
  HipExecutionCtxSynchronizeFn context_synchronize = nullptr;

  // Queries the binding's supported stream priority range.
  HipDeviceGetStreamPriorityRangeFn device_get_stream_priority_range = nullptr;

  // Creates a stream with a scheduling priority hint.
  HipStreamCreateWithPriorityFn stream_create_with_priority = nullptr;

  // Creates a stream confined to an exact HIP CU mask.
  HipExtStreamCreateWithCUMaskFn stream_create_with_cu_mask = nullptr;

  // Queries the exact HIP CU mask assigned to a stream.
  HipExtStreamGetCUMaskFn stream_get_cu_mask = nullptr;

  // Queries the exact SM resource assigned to a stream.
  HipStreamGetDevResourceFn stream_get_resource = nullptr;

  // Queries stream creation flags.
  HipStreamGetFlagsFn stream_get_flags = nullptr;

  // Queries the clamped scheduling priority assigned to a stream.
  HipStreamGetPriorityFn stream_get_priority = nullptr;

  // Begins graph capture on one stream.
  HipStreamBeginCaptureFn stream_begin_capture = nullptr;

  // Ends graph capture on one stream.
  HipStreamEndCaptureFn stream_end_capture = nullptr;

  // Queries the graph capture state of one stream.
  HipStreamIsCapturingFn stream_is_capturing = nullptr;

  // Destroys a live or execution-context-detached stream.
  HipStreamDestroyFn stream_destroy = nullptr;

  // Enqueues a 32-bit stream-ordered write.
  HipStreamWriteValue32Fn write_value_32 = nullptr;

  // Enqueues a 64-bit stream-ordered write.
  HipStreamWriteValue64Fn write_value_64 = nullptr;

  // Enqueues a 32-bit stream-ordered wait.
  HipStreamWaitValue32Fn wait_value_32 = nullptr;

  // Enqueues a 64-bit stream-ordered wait.
  HipStreamWaitValue64Fn wait_value_64 = nullptr;

  // Enqueues one transaction of stream memory operations.
  HipStreamBatchMemOpFn batch_mem_op = nullptr;

  // Enqueues a host callback in a stream.
  HipLaunchHostFuncFn launch_host_function = nullptr;

  // Launches a compiler-registered kernel with extended configuration.
  HipLaunchKernelExCFn launch_kernel_ex = nullptr;

  // Launches a module kernel with extended configuration.
  HipDrvLaunchKernelExFn driver_launch_kernel_ex = nullptr;

  // Launches a module kernel cooperatively.
  HipModuleLaunchCooperativeKernelFn module_launch_cooperative_kernel = nullptr;

  // Launches one matching kernel on each explicit device stream.
  HipExtLaunchMultiKernelMultiDeviceFn launch_multi_device = nullptr;

  // Creates a graph template.
  HipGraphCreateFn graph_create = nullptr;

  // Adds a dependency-only node to a graph template.
  HipGraphAddEmptyNodeFn graph_add_empty_node = nullptr;

  // Instantiates a graph template.
  HipGraphInstantiateFn graph_instantiate = nullptr;

  // Launches a graph executable.
  HipGraphLaunchFn graph_launch = nullptr;

  // Destroys a graph executable.
  HipGraphExecDestroyFn graph_exec_destroy = nullptr;

  // Destroys a graph template.
  HipGraphDestroyFn graph_destroy = nullptr;

  // Creates an event with explicit flags.
  HipEventCreateWithFlagsFn event_create_with_flags = nullptr;

  // Destroys a live event.
  HipEventDestroyFn event_destroy = nullptr;

  // Records an event on one stream.
  HipEventRecordFn event_record = nullptr;

  // Queries event completion without blocking.
  HipEventQueryFn event_query = nullptr;

  // Waits for event completion.
  HipEventSynchronizeFn event_synchronize = nullptr;
};

template <typename T>
T ResolveHipSymbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

class HipExecutionResourceApiTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!api_.library) {
      const char* library_path = CandidateLibPath();
      ASSERT_NE(library_path, nullptr)
          << "the build must provide the libamdhip64 artifact under test";
      api_.library = dlopen(library_path, RTLD_LAZY | RTLD_LOCAL);
      ASSERT_NE(api_.library, nullptr)
          << "cannot dlopen " << library_path << ": " << dlerror();

      api_.init = ResolveHipSymbol<HipInitFn>(api_.library, "hipInit");
      api_.get_device =
          ResolveHipSymbol<HipGetDeviceFn>(api_.library, "hipGetDevice");
      api_.get_device_properties =
          ResolveHipSymbol<HipGetDevicePropertiesR0600Fn>(
              api_.library, "hipGetDevicePropertiesR0600");
      api_.device_get_attribute = ResolveHipSymbol<HipDeviceGetAttributeFn>(
          api_.library, "hipDeviceGetAttribute");
      api_.device_get_resource = ResolveHipSymbol<HipDeviceGetDevResourceFn>(
          api_.library, "hipDeviceGetDevResource");
      api_.device_execution_context =
          ResolveHipSymbol<HipDeviceGetExecutionCtxFn>(
              api_.library, "hipDeviceGetExecutionCtx");
      api_.split_sm_by_count = ResolveHipSymbol<HipDevSmResourceSplitByCountFn>(
          api_.library, "hipDevSmResourceSplitByCount");
      api_.split_sm = ResolveHipSymbol<HipDevSmResourceSplitFn>(
          api_.library, "hipDevSmResourceSplit");
      api_.generate_descriptor = ResolveHipSymbol<HipDevResourceGenerateDescFn>(
          api_.library, "hipDevResourceGenerateDesc");
      api_.create_context = ResolveHipSymbol<HipGreenCtxCreateFn>(
          api_.library, "hipGreenCtxCreate");
      api_.destroy_context = ResolveHipSymbol<HipExecutionCtxDestroyFn>(
          api_.library, "hipExecutionCtxDestroy");
      api_.context_get_resource =
          ResolveHipSymbol<HipExecutionCtxGetDevResourceFn>(
              api_.library, "hipExecutionCtxGetDevResource");
      api_.context_get_device = ResolveHipSymbol<HipExecutionCtxGetDeviceFn>(
          api_.library, "hipExecutionCtxGetDevice");
      api_.context_get_id = ResolveHipSymbol<HipExecutionCtxGetIdFn>(
          api_.library, "hipExecutionCtxGetId");
      api_.context_stream_create =
          ResolveHipSymbol<HipExecutionCtxStreamCreateFn>(
              api_.library, "hipExecutionCtxStreamCreate");
      api_.context_record_event =
          ResolveHipSymbol<HipExecutionCtxRecordEventFn>(
              api_.library, "hipExecutionCtxRecordEvent");
      api_.context_wait_event = ResolveHipSymbol<HipExecutionCtxWaitEventFn>(
          api_.library, "hipExecutionCtxWaitEvent");
      api_.context_synchronize = ResolveHipSymbol<HipExecutionCtxSynchronizeFn>(
          api_.library, "hipExecutionCtxSynchronize");
      api_.device_get_stream_priority_range =
          ResolveHipSymbol<HipDeviceGetStreamPriorityRangeFn>(
              api_.library, "hipDeviceGetStreamPriorityRange");
      api_.stream_create_with_priority =
          ResolveHipSymbol<HipStreamCreateWithPriorityFn>(
              api_.library, "hipStreamCreateWithPriority");
      api_.stream_create_with_cu_mask =
          ResolveHipSymbol<HipExtStreamCreateWithCUMaskFn>(
              api_.library, "hipExtStreamCreateWithCUMask");
      api_.stream_get_cu_mask = ResolveHipSymbol<HipExtStreamGetCUMaskFn>(
          api_.library, "hipExtStreamGetCUMask");
      api_.stream_get_resource = ResolveHipSymbol<HipStreamGetDevResourceFn>(
          api_.library, "hipStreamGetDevResource");
      api_.stream_get_flags = ResolveHipSymbol<HipStreamGetFlagsFn>(
          api_.library, "hipStreamGetFlags");
      api_.stream_get_priority = ResolveHipSymbol<HipStreamGetPriorityFn>(
          api_.library, "hipStreamGetPriority");
      api_.stream_begin_capture = ResolveHipSymbol<HipStreamBeginCaptureFn>(
          api_.library, "hipStreamBeginCapture");
      api_.stream_end_capture = ResolveHipSymbol<HipStreamEndCaptureFn>(
          api_.library, "hipStreamEndCapture");
      api_.stream_is_capturing = ResolveHipSymbol<HipStreamIsCapturingFn>(
          api_.library, "hipStreamIsCapturing");
      api_.stream_destroy = ResolveHipSymbol<HipStreamDestroyFn>(
          api_.library, "hipStreamDestroy");
      api_.write_value_32 = ResolveHipSymbol<HipStreamWriteValue32Fn>(
          api_.library, "hipStreamWriteValue32");
      api_.write_value_64 = ResolveHipSymbol<HipStreamWriteValue64Fn>(
          api_.library, "hipStreamWriteValue64");
      api_.wait_value_32 = ResolveHipSymbol<HipStreamWaitValue32Fn>(
          api_.library, "hipStreamWaitValue32");
      api_.wait_value_64 = ResolveHipSymbol<HipStreamWaitValue64Fn>(
          api_.library, "hipStreamWaitValue64");
      api_.batch_mem_op = ResolveHipSymbol<HipStreamBatchMemOpFn>(
          api_.library, "hipStreamBatchMemOp");
      api_.launch_host_function = ResolveHipSymbol<HipLaunchHostFuncFn>(
          api_.library, "hipLaunchHostFunc");
      api_.launch_kernel_ex = ResolveHipSymbol<HipLaunchKernelExCFn>(
          api_.library, "hipLaunchKernelExC");
      api_.driver_launch_kernel_ex = ResolveHipSymbol<HipDrvLaunchKernelExFn>(
          api_.library, "hipDrvLaunchKernelEx");
      api_.module_launch_cooperative_kernel =
          ResolveHipSymbol<HipModuleLaunchCooperativeKernelFn>(
              api_.library, "hipModuleLaunchCooperativeKernel");
      api_.launch_multi_device =
          ResolveHipSymbol<HipExtLaunchMultiKernelMultiDeviceFn>(
              api_.library, "hipExtLaunchMultiKernelMultiDevice");
      api_.graph_create =
          ResolveHipSymbol<HipGraphCreateFn>(api_.library, "hipGraphCreate");
      api_.graph_add_empty_node = ResolveHipSymbol<HipGraphAddEmptyNodeFn>(
          api_.library, "hipGraphAddEmptyNode");
      api_.graph_instantiate = ResolveHipSymbol<HipGraphInstantiateFn>(
          api_.library, "hipGraphInstantiate");
      api_.graph_launch =
          ResolveHipSymbol<HipGraphLaunchFn>(api_.library, "hipGraphLaunch");
      api_.graph_exec_destroy = ResolveHipSymbol<HipGraphExecDestroyFn>(
          api_.library, "hipGraphExecDestroy");
      api_.graph_destroy =
          ResolveHipSymbol<HipGraphDestroyFn>(api_.library, "hipGraphDestroy");
      api_.event_create_with_flags =
          ResolveHipSymbol<HipEventCreateWithFlagsFn>(
              api_.library, "hipEventCreateWithFlags");
      api_.event_destroy =
          ResolveHipSymbol<HipEventDestroyFn>(api_.library, "hipEventDestroy");
      api_.event_record =
          ResolveHipSymbol<HipEventRecordFn>(api_.library, "hipEventRecord");
      api_.event_query =
          ResolveHipSymbol<HipEventQueryFn>(api_.library, "hipEventQuery");
      api_.event_synchronize = ResolveHipSymbol<HipEventSynchronizeFn>(
          api_.library, "hipEventSynchronize");
    }

    ASSERT_NE(api_.init, nullptr);
    ASSERT_NE(api_.get_device, nullptr);
    ASSERT_NE(api_.get_device_properties, nullptr);
    ASSERT_NE(api_.device_get_attribute, nullptr);
    ASSERT_NE(api_.device_get_resource, nullptr);
    ASSERT_NE(api_.device_execution_context, nullptr);
    ASSERT_NE(api_.split_sm_by_count, nullptr);
    ASSERT_NE(api_.split_sm, nullptr);
    ASSERT_NE(api_.generate_descriptor, nullptr);
    ASSERT_NE(api_.create_context, nullptr);
    ASSERT_NE(api_.destroy_context, nullptr);
    ASSERT_NE(api_.context_get_resource, nullptr);
    ASSERT_NE(api_.context_get_device, nullptr);
    ASSERT_NE(api_.context_get_id, nullptr);
    ASSERT_NE(api_.context_stream_create, nullptr);
    ASSERT_NE(api_.context_record_event, nullptr);
    ASSERT_NE(api_.context_wait_event, nullptr);
    ASSERT_NE(api_.context_synchronize, nullptr);
    ASSERT_NE(api_.device_get_stream_priority_range, nullptr);
    ASSERT_NE(api_.stream_create_with_priority, nullptr);
    ASSERT_NE(api_.stream_create_with_cu_mask, nullptr);
    ASSERT_NE(api_.stream_get_cu_mask, nullptr);
    ASSERT_NE(api_.stream_get_resource, nullptr);
    ASSERT_NE(api_.stream_get_flags, nullptr);
    ASSERT_NE(api_.stream_get_priority, nullptr);
    ASSERT_NE(api_.stream_begin_capture, nullptr);
    ASSERT_NE(api_.stream_end_capture, nullptr);
    ASSERT_NE(api_.stream_is_capturing, nullptr);
    ASSERT_NE(api_.stream_destroy, nullptr);
    ASSERT_NE(api_.write_value_32, nullptr);
    ASSERT_NE(api_.write_value_64, nullptr);
    ASSERT_NE(api_.wait_value_32, nullptr);
    ASSERT_NE(api_.wait_value_64, nullptr);
    ASSERT_NE(api_.batch_mem_op, nullptr);
    ASSERT_NE(api_.launch_host_function, nullptr);
    ASSERT_NE(api_.launch_kernel_ex, nullptr);
    ASSERT_NE(api_.driver_launch_kernel_ex, nullptr);
    ASSERT_NE(api_.module_launch_cooperative_kernel, nullptr);
    ASSERT_NE(api_.launch_multi_device, nullptr);
    ASSERT_NE(api_.graph_create, nullptr);
    ASSERT_NE(api_.graph_add_empty_node, nullptr);
    ASSERT_NE(api_.graph_instantiate, nullptr);
    ASSERT_NE(api_.graph_launch, nullptr);
    ASSERT_NE(api_.graph_exec_destroy, nullptr);
    ASSERT_NE(api_.graph_destroy, nullptr);
    ASSERT_NE(api_.event_create_with_flags, nullptr);
    ASSERT_NE(api_.event_destroy, nullptr);
    ASSERT_NE(api_.event_record, nullptr);
    ASSERT_NE(api_.event_query, nullptr);
    ASSERT_NE(api_.event_synchronize, nullptr);

    ASSERT_EQ(hipSuccess, api_.init(/*flags=*/0));
    ASSERT_EQ(hipSuccess, api_.get_device(&device_));
  }

  // Runtime entry points loaded once from the HIP shared object under test.
  static HipRuntimeApi api_;

  // Live device ordinal used by resource queries.
  int device_ = -1;
};

HipRuntimeApi HipExecutionResourceApiTest::api_;

TEST_F(HipExecutionResourceApiTest, ReturnsFullVisibleSmResource) {
  hipDevResource resource;
  ASSERT_EQ(hipSuccess,
            api_.device_get_resource(device_, &resource, hipDevResourceTypeSm));
  EXPECT_EQ(resource.type, hipDevResourceTypeSm);
  ASSERT_GT(resource.sm.smCount, 0u);
  ASSERT_GT(resource.sm.minSmPartitionSize, 0u);
  ASSERT_GT(resource.sm.smCoscheduledAlignment, 0u);
  EXPECT_LE(resource.sm.minSmPartitionSize, resource.sm.smCount);
  EXPECT_EQ(resource.sm.minSmPartitionSize % resource.sm.smCoscheduledAlignment,
            0u);
  EXPECT_EQ(resource.sm.smCount % resource.sm.smCoscheduledAlignment, 0u);
  EXPECT_EQ(resource.sm.flags, hipDevSmResourceGroupDefault);
  EXPECT_EQ(resource.nextResource, nullptr);
}

TEST_F(HipExecutionResourceApiTest, RejectsInvalidQueriesWithoutPublishing) {
  EXPECT_EQ(hipErrorInvalidValue,
            api_.device_get_resource(device_, nullptr, hipDevResourceTypeSm));

  hipDevResource resource;
  std::memset(&resource, 0xA5, sizeof(resource));
  const hipDevResource expected_resource = resource;
  EXPECT_EQ(hipErrorInvalidResourceType,
            api_.device_get_resource(device_, &resource,
                                     hipDevResourceTypeWorkqueueConfig));
  EXPECT_EQ(std::memcmp(&resource, &expected_resource, sizeof(resource)), 0);

  EXPECT_EQ(
      hipErrorInvalidDevice,
      api_.device_get_resource(/*device=*/-1, &resource, hipDevResourceTypeSm));
  EXPECT_EQ(std::memcmp(&resource, &expected_resource, sizeof(resource)), 0);
}

TEST_F(HipExecutionResourceApiTest,
       ReturnsStableDeviceManagedExecutionContext) {
  hipExecutionCtx_t first_context = nullptr;
  ASSERT_EQ(hipSuccess, api_.device_execution_context(&first_context, device_));
  ASSERT_NE(first_context, nullptr);
  hipExecutionCtx_t second_context = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.device_execution_context(&second_context, device_));
  EXPECT_EQ(second_context, first_context);

  hipDevResource full_resource;
  ASSERT_EQ(hipSuccess, api_.device_get_resource(device_, &full_resource,
                                                 hipDevResourceTypeSm));
  hipDevResource context_resource;
  ASSERT_EQ(hipSuccess,
            api_.context_get_resource(first_context, &context_resource,
                                      hipDevResourceTypeSm));
  EXPECT_EQ(
      std::memcmp(&context_resource, &full_resource, sizeof(full_resource)), 0);

  hipDevice_t context_device = -1;
  EXPECT_EQ(hipSuccess,
            api_.context_get_device(&context_device, first_context));
  EXPECT_EQ(context_device, device_);
  unsigned long long context_id = 0;
  EXPECT_EQ(hipSuccess, api_.context_get_id(first_context, &context_id));
  EXPECT_NE(context_id, 0u);

  EXPECT_EQ(hipErrorInvalidValue, api_.destroy_context(first_context));
  EXPECT_EQ(hipSuccess, api_.context_get_id(first_context, &context_id));

  hipExecutionCtx_t untouched_context =
      reinterpret_cast<hipExecutionCtx_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorInvalidDevice,
            api_.device_execution_context(&untouched_context, -1));
  EXPECT_EQ(untouched_context,
            reinterpret_cast<hipExecutionCtx_t>(uintptr_t{1}));
}

TEST_F(HipExecutionResourceApiTest, SplitsFullResourceThroughPublicAbi) {
  hipDevResource full_resource;
  ASSERT_EQ(hipSuccess, api_.device_get_resource(device_, &full_resource,
                                                 hipDevResourceTypeSm));

  hipDevResource untouched_remainder;
  std::memset(&untouched_remainder, 0xA5, sizeof(untouched_remainder));
  const hipDevResource expected_untouched_remainder = untouched_remainder;
  unsigned int possible_partition_count = 1;
  ASSERT_EQ(hipSuccess, api_.split_sm_by_count(
                            /*result=*/nullptr, &possible_partition_count,
                            &full_resource, &untouched_remainder, /*flags=*/0,
                            full_resource.sm.minSmPartitionSize));
  ASSERT_GT(possible_partition_count, 0u);
  EXPECT_EQ(std::memcmp(&untouched_remainder, &expected_untouched_remainder,
                        sizeof(untouched_remainder)),
            0);

  const unsigned int requested_partition_count =
      possible_partition_count < 2 ? possible_partition_count : 2;
  std::vector<hipDevResource> partitions(requested_partition_count);
  unsigned int actual_partition_count = requested_partition_count;
  hipDevResource remainder;
  ASSERT_EQ(hipSuccess,
            api_.split_sm_by_count(partitions.data(), &actual_partition_count,
                                   &full_resource, &remainder, /*flags=*/0,
                                   full_resource.sm.minSmPartitionSize));
  ASSERT_EQ(actual_partition_count, requested_partition_count);

  uint64_t returned_sm_count = 0;
  for (const hipDevResource& partition : partitions) {
    EXPECT_EQ(partition.type, hipDevResourceTypeSm);
    ASSERT_GT(partition.sm.smCoscheduledAlignment, 0u);
    EXPECT_GE(partition.sm.smCount, full_resource.sm.minSmPartitionSize);
    EXPECT_EQ(partition.sm.smCount % partition.sm.smCoscheduledAlignment, 0u);
    EXPECT_EQ(partition.sm.flags, hipDevSmResourceGroupDefault);
    returned_sm_count += partition.sm.smCount;
  }
  if (remainder.type == hipDevResourceTypeSm) {
    ASSERT_GT(remainder.sm.smCoscheduledAlignment, 0u);
    EXPECT_GT(remainder.sm.smCount, 0u);
    EXPECT_EQ(remainder.sm.smCount % remainder.sm.smCoscheduledAlignment, 0u);
    returned_sm_count += remainder.sm.smCount;
  } else {
    EXPECT_EQ(remainder.type, hipDevResourceTypeInvalid);
  }
  EXPECT_EQ(returned_sm_count, full_resource.sm.smCount);
}

TEST_F(HipExecutionResourceApiTest, SplitsStructuredResourcesThroughPublicAbi) {
  hipDevResource full_resource;
  ASSERT_EQ(hipSuccess, api_.device_get_resource(device_, &full_resource,
                                                 hipDevResourceTypeSm));
  ASSERT_GT(full_resource.sm.minSmPartitionSize, 0u);
  ASSERT_GE(full_resource.sm.smCount, 3u * full_resource.sm.minSmPartitionSize);

  hipDevSmResourceGroupParams group_parameters[2] = {};
  group_parameters[0].smCount = 2u * full_resource.sm.minSmPartitionSize;
  group_parameters[1].smCount = full_resource.sm.minSmPartitionSize;
  hipDevResource partitions[2];
  hipDevResource remainder;
  ASSERT_EQ(hipSuccess,
            api_.split_sm(partitions, /*group_count=*/2, &full_resource,
                          &remainder, /*flags=*/0, group_parameters));

  EXPECT_EQ(partitions[0].sm.smCount, group_parameters[0].smCount);
  EXPECT_EQ(partitions[1].sm.smCount, group_parameters[1].smCount);
  for (const auto& parameter : group_parameters) {
    EXPECT_EQ(parameter.coscheduledSmCount,
              full_resource.sm.smCoscheduledAlignment);
    EXPECT_EQ(parameter.preferredCoscheduledSmCount,
              parameter.coscheduledSmCount);
  }
  uint64_t returned_sm_count =
      partitions[0].sm.smCount + partitions[1].sm.smCount;
  if (remainder.type == hipDevResourceTypeSm) {
    returned_sm_count += remainder.sm.smCount;
  } else {
    EXPECT_EQ(remainder.type, hipDevResourceTypeInvalid);
  }
  EXPECT_EQ(returned_sm_count, full_resource.sm.smCount);
}

TEST_F(HipExecutionResourceApiTest, RejectsUnsupportedSplitWithoutPublishing) {
  hipDevResource full_resource;
  ASSERT_EQ(hipSuccess, api_.device_get_resource(device_, &full_resource,
                                                 hipDevResourceTypeSm));

  auto expect_failure_without_publication = [&](unsigned int flags,
                                                unsigned int minimum_count,
                                                hipError_t expected_result) {
    hipDevResource partition;
    std::memset(&partition, 0xA5, sizeof(partition));
    const hipDevResource expected_partition = partition;
    hipDevResource remainder;
    std::memset(&remainder, 0x5A, sizeof(remainder));
    const hipDevResource expected_remainder = remainder;
    unsigned int partition_count = 1;

    EXPECT_EQ(
        api_.split_sm_by_count(&partition, &partition_count, &full_resource,
                               &remainder, flags, minimum_count),
        expected_result);
    EXPECT_EQ(partition_count, 1u);
    EXPECT_EQ(std::memcmp(&partition, &expected_partition, sizeof(partition)),
              0);
    EXPECT_EQ(std::memcmp(&remainder, &expected_remainder, sizeof(remainder)),
              0);
  };

  expect_failure_without_publication(hipDevSmResourceSplitIgnoreSmCoscheduling,
                                     full_resource.sm.minSmPartitionSize,
                                     hipErrorNotSupported);
  expect_failure_without_publication(
      hipDevSmResourceSplitMaxPotentialClusterSize,
      full_resource.sm.minSmPartitionSize, hipErrorNotSupported);
  expect_failure_without_publication(
      /*flags=*/4, full_resource.sm.minSmPartitionSize, hipErrorInvalidValue);
  expect_failure_without_publication(/*flags=*/0, full_resource.sm.smCount + 1,
                                     hipErrorInvalidValue);
}

TEST_F(HipExecutionResourceApiTest,
       CreatesOverlappingContextsWithCanonicalResources) {
  hipDevResource full_resource;
  ASSERT_EQ(hipSuccess, api_.device_get_resource(device_, &full_resource,
                                                 hipDevResourceTypeSm));

  hipDevResourceDesc_t first_descriptor = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.generate_descriptor(&first_descriptor, &full_resource, 1));
  hipDevResourceDesc_t second_descriptor = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.generate_descriptor(&second_descriptor, &full_resource, 1));

  hipExecutionCtx_t first_context = nullptr;
  ASSERT_EQ(hipSuccess, api_.create_context(&first_context, first_descriptor,
                                            device_, /*flags=*/0));
  ScopedExecutionContext first_context_guard(
      first_context, ExecutionContextDeleter{api_.destroy_context});
  hipExecutionCtx_t second_context = nullptr;
  ASSERT_EQ(hipSuccess, api_.create_context(&second_context, second_descriptor,
                                            device_, /*flags=*/0));
  ScopedExecutionContext second_context_guard(
      second_context, ExecutionContextDeleter{api_.destroy_context});

  hipDevResource first_resource;
  ASSERT_EQ(hipSuccess,
            api_.context_get_resource(first_context, &first_resource,
                                      hipDevResourceTypeSm));
  EXPECT_EQ(std::memcmp(&first_resource, &full_resource, sizeof(full_resource)),
            0);
  hipDevResource second_resource;
  ASSERT_EQ(hipSuccess,
            api_.context_get_resource(second_context, &second_resource,
                                      hipDevResourceTypeSm));
  EXPECT_EQ(
      std::memcmp(&second_resource, &full_resource, sizeof(full_resource)), 0);

  hipDevice_t first_device = -1;
  EXPECT_EQ(hipSuccess, api_.context_get_device(&first_device, first_context));
  EXPECT_EQ(first_device, device_);
  unsigned long long first_id = 0;
  unsigned long long second_id = 0;
  EXPECT_EQ(hipSuccess, api_.context_get_id(first_context, &first_id));
  EXPECT_EQ(hipSuccess, api_.context_get_id(second_context, &second_id));
  EXPECT_NE(first_id, 0u);
  EXPECT_NE(second_id, 0u);
  EXPECT_NE(first_id, second_id);
}

TEST_F(HipExecutionResourceApiTest,
       CreatesExactStreamsAndOrphansThemWithTheirContext) {
  hipDevResource full_resource;
  ASSERT_EQ(hipSuccess, api_.device_get_resource(device_, &full_resource,
                                                 hipDevResourceTypeSm));

  hipDevResource partition;
  unsigned int partition_count = 1;
  hipDevResource remainder;
  ASSERT_EQ(hipSuccess,
            api_.split_sm_by_count(&partition, &partition_count, &full_resource,
                                   &remainder, /*flags=*/0,
                                   full_resource.sm.minSmPartitionSize));
  ASSERT_EQ(partition_count, 1u);

  hipDevResourceDesc_t descriptor = nullptr;
  ASSERT_EQ(hipSuccess, api_.generate_descriptor(&descriptor, &partition, 1));
  hipExecutionCtx_t context = nullptr;
  ASSERT_EQ(hipSuccess, api_.create_context(&context, descriptor, device_, 0));
  ScopedExecutionContext context_guard(
      context, ExecutionContextDeleter{api_.destroy_context});

  hipStream_t untouched_stream =
      reinterpret_cast<hipStream_t>(uintptr_t{0x1234});
  EXPECT_EQ(hipErrorInvalidValue,
            api_.context_stream_create(&untouched_stream, context,
                                       /*flags=*/2, /*priority=*/0));
  EXPECT_EQ(untouched_stream, reinterpret_cast<hipStream_t>(uintptr_t{0x1234}));

  hipStream_t stream = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.context_stream_create(&stream, context, hipStreamDefault,
                                       /*priority=*/0));
  ScopedStream stream_guard(stream, StreamDeleter{api_.stream_destroy});

  unsigned int stream_flags = 0;
  ASSERT_EQ(hipSuccess, api_.stream_get_flags(stream, &stream_flags));
  EXPECT_EQ(stream_flags, hipStreamNonBlocking);

  hipDevResource context_resource;
  ASSERT_EQ(hipSuccess, api_.context_get_resource(context, &context_resource,
                                                  hipDevResourceTypeSm));
  hipDevResource stream_resource;
  ASSERT_EQ(hipSuccess, api_.stream_get_resource(stream, &stream_resource,
                                                 hipDevResourceTypeSm));
  EXPECT_EQ(std::memcmp(&stream_resource, &context_resource,
                        sizeof(context_resource)),
            0);

  // CU-mask streams use the same canonical partition as resource-context
  // streams instead of projecting a second mask model into the HAL.
  const uint32_t mask_count = (full_resource.sm.smCount + 31u) / 32u;
  std::vector<uint32_t> mask(mask_count, 0u);
  ASSERT_EQ(hipSuccess,
            api_.stream_get_cu_mask(stream, mask_count, mask.data()));
  hipStream_t masked_stream = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_create_with_cu_mask(
                            &masked_stream, mask_count, mask.data()));
  ScopedStream masked_stream_guard(masked_stream,
                                   StreamDeleter{api_.stream_destroy});
  hipDevResource masked_stream_resource;
  ASSERT_EQ(hipSuccess,
            api_.stream_get_resource(masked_stream, &masked_stream_resource,
                                     hipDevResourceTypeSm));
  EXPECT_EQ(std::memcmp(&masked_stream_resource, &context_resource,
                        sizeof(context_resource)),
            0);

  ASSERT_EQ(hipSuccess, api_.destroy_context(context_guard.release()));

  std::memset(&stream_resource, 0xA5, sizeof(stream_resource));
  const hipDevResource expected_resource = stream_resource;
  EXPECT_EQ(
      hipErrorContextIsDestroyed,
      api_.stream_get_resource(stream, &stream_resource, hipDevResourceTypeSm));
  EXPECT_EQ(std::memcmp(&stream_resource, &expected_resource,
                        sizeof(stream_resource)),
            0);
  EXPECT_EQ(hipSuccess, api_.stream_destroy(stream_guard.release()));
}

TEST_F(HipExecutionResourceApiTest,
       OrdersCurrentAndFutureExecutionContextStreamsWithEvents) {
  hipDevResource full_resource;
  ASSERT_EQ(hipSuccess, api_.device_get_resource(device_, &full_resource,
                                                 hipDevResourceTypeSm));
  hipDevResourceDesc_t descriptor = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.generate_descriptor(&descriptor, &full_resource, 1));
  hipExecutionCtx_t context = nullptr;
  ASSERT_EQ(hipSuccess, api_.create_context(&context, descriptor, device_, 0));
  ScopedExecutionContext context_guard(
      context, ExecutionContextDeleter{api_.destroy_context});

  ScopedStream first_stream_guard(nullptr, StreamDeleter{api_.stream_destroy});
  ScopedStream second_stream_guard(nullptr, StreamDeleter{api_.stream_destroy});
  ScopedStream source_stream_guard(nullptr, StreamDeleter{api_.stream_destroy});
  ScopedStream later_stream_guard(nullptr, StreamDeleter{api_.stream_destroy});
  ScopedEvent record_event_guard(nullptr, EventDeleter{api_.event_destroy});
  ScopedEvent source_event_guard(nullptr, EventDeleter{api_.event_destroy});
  ScopedEvent current_event_guard(nullptr, EventDeleter{api_.event_destroy});
  ScopedEvent later_event_guard(nullptr, EventDeleter{api_.event_destroy});

  hipStream_t first_stream = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.context_stream_create(&first_stream, context, hipStreamDefault,
                                       /*priority=*/0));
  first_stream_guard.reset(first_stream);
  hipStream_t second_stream = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.context_stream_create(&second_stream, context,
                                       hipStreamDefault, /*priority=*/0));
  second_stream_guard.reset(second_stream);

  HostGate record_gate;
  HostGate source_gate;

  hipEvent_t record_event = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.event_create_with_flags(&record_event, hipEventDisableTiming));
  record_event_guard.reset(record_event);
  ASSERT_EQ(hipSuccess, api_.launch_host_function(
                            second_stream, HostGate::Callback, &record_gate));
  ASSERT_EQ(hipSuccess, api_.context_record_event(context, record_event));
  record_gate.WaitUntilEntered();
  EXPECT_EQ(hipErrorNotReady, api_.event_query(record_event));
  record_gate.Open();
  ASSERT_EQ(hipSuccess, api_.context_synchronize(context));
  EXPECT_EQ(hipSuccess, api_.event_query(record_event));

  hipStream_t source_stream = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_create_with_priority(&source_stream,
                                                         hipStreamNonBlocking,
                                                         /*priority=*/0));
  source_stream_guard.reset(source_stream);
  hipEvent_t source_event = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.event_create_with_flags(&source_event, hipEventDisableTiming));
  source_event_guard.reset(source_event);
  ASSERT_EQ(hipSuccess, api_.launch_host_function(
                            source_stream, HostGate::Callback, &source_gate));
  ASSERT_EQ(hipSuccess, api_.event_record(source_event, source_stream));
  source_gate.WaitUntilEntered();

  ASSERT_EQ(hipSuccess, api_.context_wait_event(context, source_event));

  hipStream_t later_stream = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.context_stream_create(&later_stream, context, hipStreamDefault,
                                       /*priority=*/0));
  later_stream_guard.reset(later_stream);
  hipEvent_t current_event = nullptr;
  ASSERT_EQ(hipSuccess, api_.event_create_with_flags(&current_event,
                                                     hipEventDisableTiming));
  current_event_guard.reset(current_event);
  hipEvent_t later_event = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.event_create_with_flags(&later_event, hipEventDisableTiming));
  later_event_guard.reset(later_event);
  ASSERT_EQ(hipSuccess, api_.event_record(current_event, first_stream));
  ASSERT_EQ(hipSuccess, api_.event_record(later_event, later_stream));
  EXPECT_EQ(hipErrorNotReady, api_.event_query(current_event));
  EXPECT_EQ(hipErrorNotReady, api_.event_query(later_event));

  source_gate.Open();
  ASSERT_EQ(hipSuccess, api_.context_synchronize(context));
  EXPECT_EQ(hipSuccess, api_.event_query(current_event));
  EXPECT_EQ(hipSuccess, api_.event_query(later_event));
}

TEST_F(HipExecutionResourceApiTest,
       PrimaryExecutionContextIncludesPartitionedStreams) {
  hipExecutionCtx_t primary_context = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.device_execution_context(&primary_context, device_));

  hipDevResource full_resource;
  ASSERT_EQ(hipSuccess, api_.device_get_resource(device_, &full_resource,
                                                 hipDevResourceTypeSm));
  hipDevResourceDesc_t descriptor = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.generate_descriptor(&descriptor, &full_resource, 1));
  hipExecutionCtx_t partitioned_context = nullptr;
  ASSERT_EQ(hipSuccess, api_.create_context(&partitioned_context, descriptor,
                                            device_, /*flags=*/0));
  ScopedExecutionContext context_guard(
      partitioned_context, ExecutionContextDeleter{api_.destroy_context});

  hipStream_t stream = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.context_stream_create(&stream, partitioned_context,
                                       hipStreamDefault, /*priority=*/0));
  ScopedStream stream_guard(stream, StreamDeleter{api_.stream_destroy});
  hipEvent_t event = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.event_create_with_flags(&event, hipEventDisableTiming));
  ScopedEvent event_guard(event, EventDeleter{api_.event_destroy});
  HostGate gate;

  ASSERT_EQ(hipSuccess,
            api_.launch_host_function(stream, HostGate::Callback, &gate));
  ASSERT_EQ(hipSuccess, api_.context_record_event(primary_context, event));
  gate.WaitUntilEntered();
  EXPECT_EQ(hipErrorNotReady, api_.event_query(event));
  gate.Open();
  ASSERT_EQ(hipSuccess, api_.context_synchronize(primary_context));
  EXPECT_EQ(hipSuccess, api_.event_query(event));
}

TEST_F(HipExecutionResourceApiTest,
       StreamOperationsReportDetachedExecutionContext) {
  hipDevResource full_resource;
  ASSERT_EQ(hipSuccess, api_.device_get_resource(device_, &full_resource,
                                                 hipDevResourceTypeSm));
  hipDevResourceDesc_t descriptor = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.generate_descriptor(&descriptor, &full_resource, 1));
  hipExecutionCtx_t context = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.create_context(&context, descriptor, device_, /*flags=*/0));

  hipStream_t stream = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.context_stream_create(&stream, context, hipStreamDefault,
                                       /*priority=*/0));
  ASSERT_NE(stream, nullptr);

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  hipGraphNode_t empty_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_empty_node(&empty_node, graph,
                                                  /*dependencies=*/nullptr,
                                                  /*dependency_count=*/0));
  hipGraphExec_t graph_exec = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&graph_exec, graph,
                                               /*error_node=*/nullptr,
                                               /*log_buffer=*/nullptr,
                                               /*buffer_size=*/0));

  ASSERT_EQ(hipSuccess, api_.destroy_context(context));

  alignas(uint64_t) uint64_t target = 0;
  EXPECT_EQ(
      hipErrorStreamDetached,
      api_.write_value_32(stream, &target, 1, hipStreamWriteValueDefault));
  EXPECT_EQ(
      hipErrorStreamDetached,
      api_.write_value_64(stream, &target, 1, hipStreamWriteValueDefault));
  EXPECT_EQ(
      hipErrorStreamDetached,
      api_.wait_value_32(stream, &target, 1, hipStreamWaitValueEq, UINT32_MAX));
  EXPECT_EQ(
      hipErrorStreamDetached,
      api_.wait_value_64(stream, &target, 1, hipStreamWaitValueEq, UINT64_MAX));

  hipStreamBatchMemOpParams parameter = {};
  parameter.writeValue.operation = hipStreamMemOpWriteValue32;
  parameter.writeValue.address = reinterpret_cast<hipDeviceptr_t>(&target);
  parameter.writeValue.value = 1;
  parameter.writeValue.flags = hipStreamWriteValueDefault;
  EXPECT_EQ(hipErrorStreamDetached,
            api_.batch_mem_op(stream, 1, &parameter, /*flags=*/0));

  const dim3 one = {1, 1, 1};
  hipLaunchConfig_t runtime_config = {};
  runtime_config.gridDim = one;
  runtime_config.blockDim = one;
  runtime_config.stream = stream;
  const void* function = reinterpret_cast<const void*>(uintptr_t{1});
  EXPECT_EQ(hipErrorStreamDetached,
            api_.launch_kernel_ex(&runtime_config, function,
                                  /*arguments=*/nullptr));

  HIP_LAUNCH_CONFIG driver_config = {};
  driver_config.gridDimX = 1;
  driver_config.gridDimY = 1;
  driver_config.gridDimZ = 1;
  driver_config.blockDimX = 1;
  driver_config.blockDimY = 1;
  driver_config.blockDimZ = 1;
  driver_config.hStream = stream;
  EXPECT_EQ(hipErrorStreamDetached,
            api_.driver_launch_kernel_ex(
                &driver_config, (hipFunction_t)function,
                /*kernel_parameters=*/nullptr, /*extra=*/nullptr));

  hipLaunchAttribute cooperative_attribute = {};
  cooperative_attribute.id = hipLaunchAttributeCooperative;
  cooperative_attribute.val.cooperative = 1;
  runtime_config.attrs = &cooperative_attribute;
  runtime_config.numAttrs = 1;
  driver_config.attrs = &cooperative_attribute;
  driver_config.numAttrs = 1;
  EXPECT_EQ(hipErrorStreamDetached,
            api_.launch_kernel_ex(&runtime_config, function,
                                  /*arguments=*/nullptr));
  EXPECT_EQ(hipErrorStreamDetached,
            api_.driver_launch_kernel_ex(
                &driver_config, (hipFunction_t)function,
                /*kernel_parameters=*/nullptr, /*extra=*/nullptr));

  hipLaunchAttribute prefetch_attribute = {};
  prefetch_attribute.id = hipLaunchAttributeExtDynDataPrefetch;
  runtime_config.attrs = &prefetch_attribute;
  runtime_config.numAttrs = 1;
  driver_config.attrs = &prefetch_attribute;
  driver_config.numAttrs = 1;
  EXPECT_EQ(hipErrorStreamDetached,
            api_.launch_kernel_ex(&runtime_config, function,
                                  /*arguments=*/nullptr));
  EXPECT_EQ(hipErrorStreamDetached,
            api_.driver_launch_kernel_ex(
                &driver_config, (hipFunction_t)function,
                /*kernel_parameters=*/nullptr, /*extra=*/nullptr));

  prefetch_attribute.val.dynDataPrefetch =
      reinterpret_cast<const hipExtDynDataPrefetchConfig*>(uintptr_t{1});
  EXPECT_EQ(hipErrorStreamDetached,
            api_.launch_kernel_ex(&runtime_config, function,
                                  /*arguments=*/nullptr));
  EXPECT_EQ(hipErrorStreamDetached,
            api_.driver_launch_kernel_ex(
                &driver_config, (hipFunction_t)function,
                /*kernel_parameters=*/nullptr, /*extra=*/nullptr));

  EXPECT_EQ(hipErrorStreamDetached,
            api_.module_launch_cooperative_kernel(
                (hipFunction_t)function, /*grid_dim_x=*/1, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1, /*block_dim_y=*/1,
                /*block_dim_z=*/1, /*shared_memory_bytes=*/0, stream,
                /*kernel_parameters=*/nullptr));
  hipLaunchParams multi_device_launch = {
      /*.func=*/const_cast<void*>(function),
      /*.gridDim=*/one,
      /*.blockDim=*/one,
      /*.args=*/nullptr,
      /*.sharedMem=*/0,
      /*.stream=*/stream,
  };
  EXPECT_EQ(hipErrorStreamDetached,
            api_.launch_multi_device(&multi_device_launch,
                                     /*device_count=*/1, /*flags=*/0));
  EXPECT_EQ(hipErrorStreamDetached, api_.graph_launch(graph_exec, stream));

  EXPECT_EQ(hipSuccess, api_.stream_destroy(stream));
  EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(graph_exec));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

TEST_F(HipExecutionResourceApiTest,
       ExecutionContextEventsInvalidateStreamCapture) {
  hipDevResource full_resource;
  ASSERT_EQ(hipSuccess, api_.device_get_resource(device_, &full_resource,
                                                 hipDevResourceTypeSm));
  hipDevResourceDesc_t descriptor = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.generate_descriptor(&descriptor, &full_resource, 1));
  hipExecutionCtx_t context = nullptr;
  ASSERT_EQ(hipSuccess, api_.create_context(&context, descriptor, device_, 0));
  ScopedExecutionContext context_guard(
      context, ExecutionContextDeleter{api_.destroy_context});

  hipStream_t stream = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.context_stream_create(&stream, context, hipStreamDefault,
                                       /*priority=*/0));
  ScopedStream stream_guard(stream, StreamDeleter{api_.stream_destroy});
  hipEvent_t event = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.event_create_with_flags(&event, hipEventDisableTiming));
  ScopedEvent event_guard(event, EventDeleter{api_.event_destroy});

  ASSERT_EQ(hipSuccess,
            api_.stream_begin_capture(stream, hipStreamCaptureModeGlobal));
  EXPECT_EQ(hipErrorStreamCaptureUnsupported,
            api_.context_record_event(context, event));
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  ASSERT_EQ(hipSuccess, api_.stream_is_capturing(stream, &capture_status));
  EXPECT_EQ(hipStreamCaptureStatusInvalidated, capture_status);
  EXPECT_EQ(hipErrorStreamCaptureInvalidated,
            api_.stream_end_capture(stream, nullptr));

  ASSERT_EQ(hipSuccess,
            api_.stream_begin_capture(stream, hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess, api_.event_record(event, stream));
  EXPECT_EQ(hipErrorStreamCaptureUnsupported,
            api_.context_wait_event(context, event));
  capture_status = hipStreamCaptureStatusNone;
  ASSERT_EQ(hipSuccess, api_.stream_is_capturing(stream, &capture_status));
  EXPECT_EQ(hipStreamCaptureStatusInvalidated, capture_status);
  EXPECT_EQ(hipErrorStreamCaptureInvalidated,
            api_.stream_end_capture(stream, nullptr));
}

TEST_F(HipExecutionResourceApiTest, CreatesStreamsAtClampedHardwarePriorities) {
  int least_priority = 7;
  int greatest_priority = 7;
  ASSERT_EQ(hipSuccess, api_.device_get_stream_priority_range(
                            &least_priority, &greatest_priority));
  ASSERT_LT(greatest_priority, least_priority);

  hipDeviceProp_t properties;
  std::memset(&properties, 0, sizeof(properties));
  ASSERT_EQ(hipSuccess, api_.get_device_properties(&properties, device_));
  EXPECT_TRUE(properties.streamPrioritiesSupported);

  int priorities_supported = 0;
  ASSERT_EQ(hipSuccess,
            api_.device_get_attribute(
                &priorities_supported,
                hipDeviceAttributeStreamPrioritiesSupported, device_));
  EXPECT_EQ(priorities_supported, 1);

  hipStream_t high_priority_stream = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_create_with_priority(
                            &high_priority_stream, hipStreamNonBlocking,
                            greatest_priority - 1));
  ScopedStream high_priority_stream_guard(high_priority_stream,
                                          StreamDeleter{api_.stream_destroy});

  int actual_priority = 7;
  ASSERT_EQ(hipSuccess,
            api_.stream_get_priority(high_priority_stream, &actual_priority));
  EXPECT_EQ(actual_priority, greatest_priority);

  hipStream_t low_priority_stream = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_create_with_priority(&low_priority_stream,
                                                         hipStreamNonBlocking,
                                                         least_priority + 1));
  ScopedStream low_priority_stream_guard(low_priority_stream,
                                         StreamDeleter{api_.stream_destroy});
  actual_priority = 7;
  ASSERT_EQ(hipSuccess,
            api_.stream_get_priority(low_priority_stream, &actual_priority));
  EXPECT_EQ(actual_priority, least_priority);

  hipDevResource full_resource;
  ASSERT_EQ(hipSuccess, api_.device_get_resource(device_, &full_resource,
                                                 hipDevResourceTypeSm));
  hipDevResource stream_resource;
  ASSERT_EQ(hipSuccess,
            api_.stream_get_resource(high_priority_stream, &stream_resource,
                                     hipDevResourceTypeSm));
  EXPECT_EQ(
      std::memcmp(&stream_resource, &full_resource, sizeof(full_resource)), 0);
}

TEST_F(HipExecutionResourceApiTest,
       ConsumesDescriptorsOnlyForSuccessfulCreation) {
  hipDevResource full_resource;
  ASSERT_EQ(hipSuccess, api_.device_get_resource(device_, &full_resource,
                                                 hipDevResourceTypeSm));
  hipDevResourceDesc_t descriptor = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.generate_descriptor(&descriptor, &full_resource, 1));

  hipExecutionCtx_t untouched_context =
      reinterpret_cast<hipExecutionCtx_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorInvalidValue,
            api_.create_context(&untouched_context, descriptor, device_,
                                /*flags=*/1));
  EXPECT_EQ(untouched_context,
            reinterpret_cast<hipExecutionCtx_t>(uintptr_t{1}));

  hipExecutionCtx_t context = nullptr;
  ASSERT_EQ(hipSuccess, api_.create_context(&context, descriptor, device_,
                                            /*flags=*/0));
  ScopedExecutionContext context_guard(
      context, ExecutionContextDeleter{api_.destroy_context});

  untouched_context = reinterpret_cast<hipExecutionCtx_t>(uintptr_t{1});
  EXPECT_EQ(hipErrorInvalidValue,
            api_.create_context(&untouched_context, descriptor, device_,
                                /*flags=*/0));
  EXPECT_EQ(untouched_context,
            reinterpret_cast<hipExecutionCtx_t>(uintptr_t{1}));

  hipDevResource untouched_resource;
  std::memset(&untouched_resource, 0xA5, sizeof(untouched_resource));
  const hipDevResource expected_resource = untouched_resource;
  EXPECT_EQ(hipErrorInvalidResourceType,
            api_.context_get_resource(context, &untouched_resource,
                                      hipDevResourceTypeWorkqueueConfig));
  EXPECT_EQ(std::memcmp(&untouched_resource, &expected_resource,
                        sizeof(untouched_resource)),
            0);

  ASSERT_EQ(hipSuccess, api_.destroy_context(context_guard.release()));
  hipDevice_t untouched_device = -7;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.context_get_device(&untouched_device, context));
  EXPECT_EQ(untouched_device, -7);
  EXPECT_EQ(hipErrorInvalidValue, api_.destroy_context(context));
}

}  // namespace
