// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

// Acceptance tests for hipEventRecord / hipEventQuery / hipEventElapsedTime
// against real device work.
//
// A kernel publishes entry through coherent host memory and waits for release.
// Its event interval contains the measured hold and is contained by the full
// host enqueue/wait window, independent of host scheduling and GPU throughput.

#include <dlfcn.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <string>
#include <thread>
#include <vector>

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
using HipModuleLaunchKernelFn = hipError_t (*)(
    hipFunction_t function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_memory_bytes,
    hipStream_t stream, void** arguments, void** extra);
using HipDeviceSynchronizeFn = hipError_t (*)(void);
using HipHostMallocFn = hipError_t (*)(void** pointer, size_t size,
                                       unsigned int flags);
using HipHostFreeFn = hipError_t (*)(void* pointer);
using HipHostGetDevicePointerFn = hipError_t (*)(hipDeviceptr_t* device_pointer,
                                                 void* host_pointer,
                                                 unsigned int flags);
using HipEventCreateWithFlagsFn = hipError_t (*)(hipEvent_t* event,
                                                 unsigned flags);
using HipEventRecordFn = hipError_t (*)(hipEvent_t event, hipStream_t stream);
using HipEventQueryFn = hipError_t (*)(hipEvent_t event);
using HipEventSynchronizeFn = hipError_t (*)(hipEvent_t event);
using HipEventDestroyFn = hipError_t (*)(hipEvent_t event);
using HipEventElapsedTimeFn = hipError_t (*)(float* milliseconds,
                                             hipEvent_t start, hipEvent_t stop);
using HipGraphCreateFn = hipError_t (*)(hipGraph_t* graph, unsigned int flags);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t graph);
using HipGraphAddKernelNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, const void* params);
using HipGraphAddEventRecordNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, hipEvent_t event);
using HipGraphInstantiateFn = hipError_t (*)(hipGraphExec_t* graph_executable,
                                             hipGraph_t graph,
                                             hipGraphNode_t* error_node,
                                             char* log_buffer,
                                             size_t buffer_size);
using HipGraphLaunchFn = hipError_t (*)(hipGraphExec_t graph_executable,
                                        hipStream_t stream);
using HipGraphExecDestroyFn = hipError_t (*)(hipGraphExec_t graph_executable);

// Requested hold after kernel entry. Assertions use the measured duration.
constexpr double kKernelHoldMs = 100.0;

// Fraction of the observed hold the device interval must exceed, allowing for
// differences between the host and device clock rates.
constexpr double kMinimumHoldFraction = 1.0 / 4.0;

// Ceiling on device-reported elapsed time as a multiple of the host-observed
// duration; the host window contains the device work.
constexpr float kMaximumDeviceFactor = 2.0f;

// Mapped coherent state for hrx_gated_store_output. Its first two words match
// hrx_launch_gate_t in executable_kernels.c; the third is the kernel's output.
struct GatedKernelState {
  void Reset() {
    __atomic_store_n(&entered, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&released, 0, __ATOMIC_RELEASE);
    output = 0;
  }

  void WaitUntilEntered() const {
    while (!__atomic_load_n(&entered, __ATOMIC_ACQUIRE)) {
      std::this_thread::yield();
    }
  }

  void Release() { __atomic_store_n(&released, 1, __ATOMIC_RELEASE); }

  // Set by the kernel once it has begun executing.
  uint32_t entered = 0;
  // Set by the host to allow the kernel to finish.
  uint32_t released = 0;
  // Written by the kernel after release; read by the host after completion.
  uint32_t output = 0;
};
static_assert(offsetof(GatedKernelState, released) == sizeof(uint32_t));
static_assert(offsetof(GatedKernelState, output) == 2 * sizeof(uint32_t));

// The HIP runtime is a process-global singleton, so the library handle, module,
// and device allocation are per-suite rather than per-test.
class HipEventTimingTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { InitializeSuite(); }

  static void TearDownTestSuite() {
    if (kernel_state_) {
      kernel_state_->~GatedKernelState();
      EXPECT_EQ(hipSuccess, host_free_(kernel_state_));
      kernel_state_ = nullptr;
    }
    if (module_) {
      EXPECT_EQ(hipSuccess, module_unload_(module_));
      module_ = nullptr;
    }
    // Not dlclosed: the runtime spawns background threads that outlive any
    // teardown reachable from here and would execute unmapped text.
  }

  void SetUp() override {
    if (!skip_reason_.empty()) {
      GTEST_SKIP() << skip_reason_;
    }
    ASSERT_NE(nullptr, gated_function_);
    ASSERT_NE(nullptr, kernel_state_);
    kernel_state_->Reset();
  }

  void TearDown() override {
    // An assertion may have exited while the kernel was still parked. Release
    // and drain before retiring graph/event handles or reusing mapped state.
    if (kernel_state_) {
      kernel_state_->Release();
      EXPECT_EQ(hipSuccess, device_synchronize_());
    }
    for (hipGraphExec_t executable : graph_executables_) {
      EXPECT_EQ(hipSuccess, graph_exec_destroy_(executable));
    }
    for (hipGraph_t graph : graphs_) {
      EXPECT_EQ(hipSuccess, graph_destroy_(graph));
    }
    for (hipEvent_t event : events_) {
      EXPECT_EQ(hipSuccess, event_destroy_(event));
    }
  }

  // Brings up the HIP runtime, loads the test kernel, and allocates its mapped
  // state. Sets |skip_reason_| only when the machine has no device; every
  // other failure is a defect and fails the suite.
  static void InitializeSuite() {
    library_ = dlopen(CandidateLibPath(), RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(nullptr, library_)
        << "cannot dlopen " << CandidateLibPath() << ": " << dlerror()
        << "; the shim is a data dependency of this target, so it is present "
           "and failing to load it is a defect rather than a missing "
           "environment";

    init_ = ResolveHipSymbol<HipInitFn>(library_, "hipInit");
    get_device_ = ResolveHipSymbol<HipGetDeviceFn>(library_, "hipGetDevice");
    get_device_properties_ = ResolveHipSymbol<HipGetDevicePropertiesR0600Fn>(
        library_, "hipGetDevicePropertiesR0600");
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
    host_malloc_ = ResolveHipSymbol<HipHostMallocFn>(library_, "hipHostMalloc");
    host_free_ = ResolveHipSymbol<HipHostFreeFn>(library_, "hipHostFree");
    host_get_device_pointer_ = ResolveHipSymbol<HipHostGetDevicePointerFn>(
        library_, "hipHostGetDevicePointer");
    event_create_with_flags_ = ResolveHipSymbol<HipEventCreateWithFlagsFn>(
        library_, "hipEventCreateWithFlags");
    event_record_ =
        ResolveHipSymbol<HipEventRecordFn>(library_, "hipEventRecord");
    event_query_ = ResolveHipSymbol<HipEventQueryFn>(library_, "hipEventQuery");
    event_synchronize_ = ResolveHipSymbol<HipEventSynchronizeFn>(
        library_, "hipEventSynchronize");
    event_destroy_ =
        ResolveHipSymbol<HipEventDestroyFn>(library_, "hipEventDestroy");
    event_elapsed_time_ = ResolveHipSymbol<HipEventElapsedTimeFn>(
        library_, "hipEventElapsedTime");
    graph_create_ =
        ResolveHipSymbol<HipGraphCreateFn>(library_, "hipGraphCreate");
    graph_destroy_ =
        ResolveHipSymbol<HipGraphDestroyFn>(library_, "hipGraphDestroy");
    graph_add_kernel_node_ = ResolveHipSymbol<HipGraphAddKernelNodeFn>(
        library_, "hipGraphAddKernelNode");
    graph_add_event_record_node_ =
        ResolveHipSymbol<HipGraphAddEventRecordNodeFn>(
            library_, "hipGraphAddEventRecordNode");
    graph_instantiate_ = ResolveHipSymbol<HipGraphInstantiateFn>(
        library_, "hipGraphInstantiate");
    graph_launch_ =
        ResolveHipSymbol<HipGraphLaunchFn>(library_, "hipGraphLaunch");
    graph_exec_destroy_ = ResolveHipSymbol<HipGraphExecDestroyFn>(
        library_, "hipGraphExecDestroy");

    ASSERT_NE(nullptr, init_);
    ASSERT_NE(nullptr, get_device_);
    ASSERT_NE(nullptr, get_device_properties_);
    ASSERT_NE(nullptr, module_load_data_);
    ASSERT_NE(nullptr, module_unload_);
    ASSERT_NE(nullptr, module_get_function_);
    ASSERT_NE(nullptr, module_launch_kernel_);
    ASSERT_NE(nullptr, device_synchronize_);
    ASSERT_NE(nullptr, host_malloc_);
    ASSERT_NE(nullptr, host_free_);
    ASSERT_NE(nullptr, host_get_device_pointer_);
    ASSERT_NE(nullptr, event_create_with_flags_);
    ASSERT_NE(nullptr, event_record_);
    ASSERT_NE(nullptr, event_query_);
    ASSERT_NE(nullptr, event_synchronize_);
    ASSERT_NE(nullptr, event_destroy_);
    ASSERT_NE(nullptr, event_elapsed_time_);
    ASSERT_NE(nullptr, graph_create_);
    ASSERT_NE(nullptr, graph_destroy_);
    ASSERT_NE(nullptr, graph_add_kernel_node_);
    ASSERT_NE(nullptr, graph_add_event_record_node_);
    ASSERT_NE(nullptr, graph_instantiate_);
    ASSERT_NE(nullptr, graph_launch_);
    ASSERT_NE(nullptr, graph_exec_destroy_);

    // hipErrorNoDevice is the one answer that means "this machine has nothing
    // to run on"; every other failure means the runtime is broken.
    const hipError_t init_result = init_(/*flags=*/0);
    if (init_result == hipErrorNoDevice) {
      skip_reason_ = "no HIP device is visible on this machine";
      return;
    }
    ASSERT_EQ(hipSuccess, init_result) << "hipInit failed for a reason other "
                                          "than the machine having no device";
    int device = 0;
    ASSERT_EQ(hipSuccess, get_device_(&device));
    hipDeviceProp_t properties = {};
    ASSERT_EQ(hipSuccess, get_device_properties_(&properties, device));

    const hrx_cts::AmdgpuExecutableTestImage test_image =
        hrx_cts::FindAmdgpuExecutableTestImage(properties.gcnArchName);
    ASSERT_NE(nullptr, test_image.file)
        << "no embedded HSACO for " << properties.gcnArchName;

    std::vector<uint8_t> image(test_image.file->data,
                               test_image.file->data + test_image.file->size);
    ASSERT_EQ(hipSuccess, module_load_data_(&module_, image.data()));
    ASSERT_EQ(hipSuccess, module_get_function_(&gated_function_, module_,
                                               "hrx_gated_store_output"));

    void* host_state = nullptr;
    ASSERT_EQ(hipSuccess,
              host_malloc_(&host_state, sizeof(GatedKernelState),
                           hipHostMallocMapped | hipHostMallocCoherent));
    kernel_state_ = new (host_state) GatedKernelState{};
    ASSERT_EQ(hipSuccess,
              host_get_device_pointer_(&device_state_, kernel_state_,
                                       /*flags=*/0));
  }

  static hipError_t LaunchGatedKernel(uint32_t value) {
    auto* output = reinterpret_cast<uint8_t*>(device_state_) +
                   offsetof(GatedKernelState, output);
    void* arguments[] = {&device_state_, &output, &value};
    return module_launch_kernel_(gated_function_, 1, 1, 1, 1, 1, 1,
                                 /*shared_memory_bytes=*/0,
                                 /*stream=*/nullptr, arguments,
                                 /*extra=*/nullptr);
  }

  // Holds a kernel that has reported entry. The device event interval contains
  // this measured duration even if the host is descheduled before release.
  double HoldKernelAndRelease() {
    kernel_state_->WaitUntilEntered();
    const auto held_from = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(
        std::chrono::duration<double, std::milli>(kKernelHoldMs));
    const double held_ms = MillisecondsSince(held_from);
    kernel_state_->Release();
    return held_ms;
  }

  // Checks the nested intervals without assumptions about enqueue speed or
  // kernel throughput. The host window encloses submission and completion.
  void ExpectDeviceTimed(float device_ms, double held_ms, double host_ms) {
    EXPECT_GT(device_ms, held_ms * kMinimumHoldFraction)
        << "the device interval must contain the " << held_ms
        << " ms observed while the kernel was held";
    EXPECT_LT(device_ms, host_ms * kMaximumDeviceFactor)
        << "the device interval must fit in the " << host_ms
        << " ms host window that contains the device work";
  }

  // Creates an event owned by the fixture, including on assertion exits.
  hipEvent_t CreateEvent(unsigned int flags = 0) {
    hipEvent_t event = nullptr;
    EXPECT_EQ(hipSuccess, event_create_with_flags_(&event, flags));
    if (event) {
      events_.push_back(event);
    }
    return event;
  }

  // Builds a record/kernel/record graph. The node captures its argument values;
  // the fixture owns the graph, executable, and referenced device allocation.
  void BuildTimedGraph(uint32_t value, hipEvent_t start, hipEvent_t stop,
                       hipGraphExec_t* out_executable) {
    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipSuccess, graph_create_(&graph, /*flags=*/0));
    graphs_.push_back(graph);

    hipGraphNode_t start_node = nullptr;
    ASSERT_EQ(hipSuccess,
              graph_add_event_record_node_(&start_node, graph,
                                           /*dependencies=*/nullptr,
                                           /*dependency_count=*/0, start));

    auto* output = reinterpret_cast<uint8_t*>(device_state_) +
                   offsetof(GatedKernelState, output);
    void* arguments[] = {&device_state_, &output, &value};
    const hipKernelNodeParams params = {
        /*.blockDim=*/{1, 1, 1},
        /*.extra=*/nullptr,
        /*.func=*/gated_function_,
        /*.gridDim=*/{1, 1, 1},
        /*.kernelParams=*/arguments,
        /*.sharedMemBytes=*/0,
    };
    hipGraphNode_t kernel_node = nullptr;
    ASSERT_EQ(hipSuccess,
              graph_add_kernel_node_(&kernel_node, graph, &start_node,
                                     /*dependency_count=*/1, &params));

    hipGraphNode_t stop_node = nullptr;
    ASSERT_EQ(hipSuccess,
              graph_add_event_record_node_(&stop_node, graph, &kernel_node,
                                           /*dependency_count=*/1, stop));

    ASSERT_EQ(hipSuccess,
              graph_instantiate_(out_executable, graph, nullptr, nullptr, 0));
    graph_executables_.push_back(*out_executable);
  }

  // Reuses the mapped state only after the preceding launch has completed.
  void ReplayAndMeasure(hipGraphExec_t executable, hipEvent_t start,
                        hipEvent_t stop, uint32_t expected_value) {
    kernel_state_->Reset();
    const auto host_start = std::chrono::steady_clock::now();
    ASSERT_EQ(hipSuccess, graph_launch_(executable, /*stream=*/nullptr));
    const double held_ms = HoldKernelAndRelease();
    ASSERT_EQ(hipSuccess, event_synchronize_(stop));
    const double host_total_ms = MillisecondsSince(host_start);
    EXPECT_EQ(expected_value, kernel_state_->output);
    float device_ms = -1.0f;
    ASSERT_EQ(hipSuccess, event_elapsed_time_(&device_ms, start, stop));
    ExpectDeviceTimed(device_ms, held_ms, host_total_ms);
  }

  static double MillisecondsSince(
      std::chrono::steady_clock::time_point start_time) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start_time)
        .count();
  }

  // Reason the whole suite cannot run, or empty when it can. Set only when the
  // machine has no HIP device, never for a failure of the runtime itself.
  inline static std::string skip_reason_;

  inline static void* library_ = nullptr;
  inline static hipModule_t module_ = nullptr;
  // Gated store kernel loaded from the suite-owned module.
  inline static hipFunction_t gated_function_ = nullptr;
  // Coherent host allocation retained until every test has drained its work.
  inline static GatedKernelState* kernel_state_ = nullptr;
  // Device mapping of kernel_state_, including the output word.
  inline static hipDeviceptr_t device_state_ = nullptr;

  // Events created by this test, retired after draining accepted kernel work.
  std::vector<hipEvent_t> events_;
  // Graphs created by this test, retained until their executables are retired.
  std::vector<hipGraph_t> graphs_;
  // Graph executables created by this test, retired after draining launches.
  std::vector<hipGraphExec_t> graph_executables_;

  inline static HipInitFn init_ = nullptr;
  inline static HipGetDeviceFn get_device_ = nullptr;
  inline static HipGetDevicePropertiesR0600Fn get_device_properties_ = nullptr;
  inline static HipModuleLoadDataFn module_load_data_ = nullptr;
  inline static HipModuleUnloadFn module_unload_ = nullptr;
  inline static HipModuleGetFunctionFn module_get_function_ = nullptr;
  inline static HipModuleLaunchKernelFn module_launch_kernel_ = nullptr;
  inline static HipDeviceSynchronizeFn device_synchronize_ = nullptr;
  // Allocates the coherent host state shared with the kernel.
  inline static HipHostMallocFn host_malloc_ = nullptr;
  // Frees the suite-owned coherent host state after all tests have drained.
  inline static HipHostFreeFn host_free_ = nullptr;
  // Resolves the device mapping of the suite-owned coherent host allocation.
  inline static HipHostGetDevicePointerFn host_get_device_pointer_ = nullptr;
  inline static HipEventCreateWithFlagsFn event_create_with_flags_ = nullptr;
  inline static HipEventRecordFn event_record_ = nullptr;
  inline static HipEventQueryFn event_query_ = nullptr;
  inline static HipEventSynchronizeFn event_synchronize_ = nullptr;
  inline static HipEventDestroyFn event_destroy_ = nullptr;
  inline static HipEventElapsedTimeFn event_elapsed_time_ = nullptr;
  inline static HipGraphCreateFn graph_create_ = nullptr;
  inline static HipGraphDestroyFn graph_destroy_ = nullptr;
  inline static HipGraphAddKernelNodeFn graph_add_kernel_node_ = nullptr;
  inline static HipGraphAddEventRecordNodeFn graph_add_event_record_node_ =
      nullptr;
  inline static HipGraphInstantiateFn graph_instantiate_ = nullptr;
  inline static HipGraphLaunchFn graph_launch_ = nullptr;
  inline static HipGraphExecDestroyFn graph_exec_destroy_ = nullptr;
};

TEST_F(HipEventTimingTest, DirectRecordMeasuresDeviceWork) {
  hipEvent_t start = CreateEvent();
  ASSERT_NE(nullptr, start);
  hipEvent_t stop = CreateEvent();
  ASSERT_NE(nullptr, stop);

  const auto host_start = std::chrono::steady_clock::now();
  ASSERT_EQ(hipSuccess, event_record_(start, /*stream=*/nullptr));
  ASSERT_EQ(hipSuccess, LaunchGatedKernel(/*value=*/0xC001));
  ASSERT_EQ(hipSuccess, event_record_(stop, /*stream=*/nullptr));

  const double held_ms = HoldKernelAndRelease();
  ASSERT_EQ(hipSuccess, event_synchronize_(stop));
  const double host_total_ms = MillisecondsSince(host_start);
  EXPECT_EQ(0xC001u, kernel_state_->output);

  ASSERT_EQ(hipSuccess, event_query_(start));
  ASSERT_EQ(hipSuccess, event_query_(stop));

  float elapsed_ms = 0.0f;
  ASSERT_EQ(hipSuccess, event_elapsed_time_(&elapsed_ms, start, stop));
  ExpectDeviceTimed(elapsed_ms, held_ms, host_total_ms);
}

// The same property through graph replay, which reaches the device by a
// different path than a direct record.
TEST_F(HipEventTimingTest, GraphReplayedRecordMeasuresDeviceWork) {
  hipEvent_t start = CreateEvent();
  ASSERT_NE(nullptr, start);
  hipEvent_t stop = CreateEvent();
  ASSERT_NE(nullptr, stop);

  hipGraphExec_t executable = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      BuildTimedGraph(/*value=*/0xC001, start, stop, &executable));
  ASSERT_NO_FATAL_FAILURE(
      ReplayAndMeasure(executable, start, stop, /*expected_value=*/0xC001));
}

// Alternate two executables sharing one event pair. Each replay must capture
// new ticks, ordered after the previous replay against a retained reference.
// No duration ratio is implied by the relative work or scheduling of launches.
TEST_F(HipEventTimingTest, GraphReplayRetimesEachLaunch) {
  hipEvent_t reference = CreateEvent();
  ASSERT_NE(nullptr, reference);
  hipEvent_t start = CreateEvent();
  ASSERT_NE(nullptr, start);
  hipEvent_t stop = CreateEvent();
  ASSERT_NE(nullptr, stop);

  hipGraphExec_t first = nullptr;
  hipGraphExec_t second = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      BuildTimedGraph(/*value=*/0xC001, start, stop, &first));
  ASSERT_NO_FATAL_FAILURE(
      BuildTimedGraph(/*value=*/0xC002, start, stop, &second));

  ASSERT_EQ(hipSuccess, event_record_(reference, /*stream=*/nullptr));
  ASSERT_EQ(hipSuccess, event_synchronize_(reference));
  float previous_start_ms = -1.0f;
  float previous_stop_ms = 0.0f;
  for (hipGraphExec_t executable : {first, second, first}) {
    SCOPED_TRACE(executable == first ? "first executable"
                                     : "second executable");
    ASSERT_NO_FATAL_FAILURE(ReplayAndMeasure(
        executable, start, stop, executable == first ? 0xC001 : 0xC002));
    float start_ms = -1.0f;
    float stop_ms = -1.0f;
    ASSERT_EQ(hipSuccess, event_elapsed_time_(&start_ms, reference, start));
    ASSERT_EQ(hipSuccess, event_elapsed_time_(&stop_ms, reference, stop));
    EXPECT_GE(start_ms, previous_stop_ms);
    EXPECT_GT(start_ms, previous_start_ms);
    EXPECT_GT(stop_ms, previous_stop_ms);
    previous_start_ms = start_ms;
    previous_stop_ms = stop_ms;
  }
}

// A timing-disabled event still has to cover preceding kernel work.
TEST_F(HipEventTimingTest, DisableTimingEventsSynchronizeButDoNotTime) {
  hipEvent_t start = CreateEvent(hipEventDisableTiming);
  ASSERT_NE(nullptr, start);
  hipEvent_t stop = CreateEvent(hipEventDisableTiming);
  ASSERT_NE(nullptr, stop);

  ASSERT_EQ(hipSuccess, event_record_(start, /*stream=*/nullptr));
  ASSERT_EQ(hipSuccess, LaunchGatedKernel(/*value=*/0xC001));
  ASSERT_EQ(hipSuccess, event_record_(stop, /*stream=*/nullptr));
  kernel_state_->WaitUntilEntered();
  // Device progress can precede host completion publication. Observe the
  // preceding event through its wait contract while the kernel remains held.
  ASSERT_EQ(hipSuccess, event_synchronize_(start));
  EXPECT_EQ(hipSuccess, event_query_(start));
  EXPECT_EQ(hipErrorNotReady, event_query_(stop));
  kernel_state_->Release();
  ASSERT_EQ(hipSuccess, event_synchronize_(stop));
  EXPECT_EQ(0xC001u, kernel_state_->output);
  EXPECT_EQ(hipSuccess, event_query_(start));
  EXPECT_EQ(hipSuccess, event_query_(stop));

  float elapsed_ms = -1.0f;
  EXPECT_EQ(hipErrorInvalidHandle,
            event_elapsed_time_(&elapsed_ms, start, stop));
  EXPECT_FLOAT_EQ(-1.0f, elapsed_ms) << "a refused call wrote a duration";
}

}  // namespace
