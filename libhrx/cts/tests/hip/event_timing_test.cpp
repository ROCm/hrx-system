// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

// Acceptance tests for hipEventRecord / hipEventQuery / hipEventElapsedTime
// against real device work.
//
// Every timing assertion is a bound scaled to the host-observed duration of the
// same work rather than to an assumed clock rate. A device-timed pair
// bracketing a long kernel reports roughly the kernel duration while a
// host-timed pair reports the microseconds spent enqueuing it, and only a lower
// bound derived from the host window separates the two.

#include <dlfcn.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
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
using HipEventCreateFn = hipError_t (*)(hipEvent_t* event);
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

// Dependent-chain steps the timed kernel runs. Every assertion scales itself to
// the duration the host observes, so this count sets how long the test takes
// and not what it accepts; it needs only to be large enough that the long and
// short kernels below report durations a device clock tells apart.
constexpr uint64_t kSpinIterations = 4000000ull;

// Ratio between the long and short timed kernels.
constexpr uint64_t kWorkRatio = 10ull;

// Steps the short kernel runs, for tests that compare two device durations.
constexpr uint64_t kShortSpinIterations = kSpinIterations / kWorkRatio;

// Smallest factor by which the long kernel's reported duration must exceed the
// short kernel's. Below kWorkRatio for noise, above 1 to catch stale ticks.
constexpr float kMinimumMeasuredWorkRatio = 3.0f;

// Fraction of the host-observed duration that device-reported elapsed time must
// exceed. Loose enough to hold if the advertised tick frequency is off by an
// order of magnitude, still far above what host-enqueue spacing reports.
constexpr float kMinimumDeviceFraction = 1.0f / 50.0f;

// Ceiling on device-reported elapsed time as a multiple of the host-observed
// duration; the host window contains the device work.
constexpr float kMaximumDeviceFactor = 2.0f;

// A graph that records |start|, runs the spin kernel, and records |stop|,
// together with its instantiated executable.
struct TimedGraph {
  // Graph holding the record/kernel/record nodes, owned.
  hipGraph_t graph = nullptr;
  // Executable instantiated from |graph|, owned.
  hipGraphExec_t executable = nullptr;
  // Kernel output pointer the kernel node's argument list points at, so it has
  // to outlive every launch of |executable|.
  uint64_t* output = nullptr;
  // Trip count the kernel node's argument list points at, same lifetime.
  uint64_t iterations = 0;
};

// The HIP runtime is a process-global singleton, so the library handle, module,
// and device allocation are per-suite rather than per-test.
class HipEventTimingTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { InitializeSuite(); }

  static void TearDownTestSuite() {
    if (spin_output_) {
      EXPECT_EQ(hipSuccess, hip_free_(spin_output_));
      spin_output_ = nullptr;
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
    ASSERT_NE(nullptr, spin_function_);
  }

  // Brings up the HIP runtime, loads the test kernels, and allocates the spin
  // output. Sets |skip_reason_| only when the machine has no device; every
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
    get_device_properties_ = ResolveHipSymbol<HipGetDevicePropertiesFn>(
        library_, "hipGetDeviceProperties");
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
    event_create_ =
        ResolveHipSymbol<HipEventCreateFn>(library_, "hipEventCreate");
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
    ASSERT_NE(nullptr, hip_malloc_);
    ASSERT_NE(nullptr, hip_free_);
    ASSERT_NE(nullptr, event_create_);
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

    ASSERT_EQ(hipSuccess, hip_malloc_(&spin_output_, sizeof(uint64_t)));

    std::vector<uint8_t> image(test_image.file->data,
                               test_image.file->data + test_image.file->size);
    ASSERT_EQ(hipSuccess, module_load_data_(&module_, image.data()));
    ASSERT_EQ(hipSuccess, module_get_function_(&spin_function_, module_,
                                               "hrx_spin_dependent_chain"));

    // Warm the module and queue so first-launch costs land outside the measured
    // window.
    ASSERT_EQ(hipSuccess, LaunchSpin(/*iterations=*/1, /*stream=*/nullptr));
    ASSERT_EQ(hipSuccess, device_synchronize_());
  }

  static hipError_t LaunchSpin(uint64_t iterations, hipStream_t stream) {
    uint64_t* output = static_cast<uint64_t*>(spin_output_);
    void* arguments[] = {&output, &iterations};
    return module_launch_kernel_(spin_function_, 1, 1, 1, 1, 1, 1,
                                 /*shared_memory_bytes=*/0, stream, arguments,
                                 /*extra=*/nullptr);
  }

  // Asserts that |device_ms| is a device-timeline measurement of work that took
  // |host_ms| as observed from the host, rather than host-enqueue spacing.
  void ExpectDeviceTimed(float device_ms, double host_ms, const char* what) {
    // The host window must itself be long enough for the bounds below to
    // separate device timing from enqueue timing.
    ASSERT_GT(host_ms, 1.0)
        << what
        << ": timed work was too short to distinguish device timing "
           "from host-enqueue timing";
    EXPECT_GT(device_ms, host_ms * kMinimumDeviceFraction)
        << what << ": elapsed time " << device_ms << " ms is far below the "
        << host_ms
        << " ms the same work took as observed from the host, which is what "
           "host-enqueue timing rather than device timing reports";
    EXPECT_LT(device_ms, host_ms * kMaximumDeviceFactor)
        << what << ": elapsed time " << device_ms << " ms exceeds the "
        << host_ms << " ms host window that contains the device work";
  }

  // Builds and instantiates a graph that records |start|, spins for
  // |iterations|, and records |stop|.
  void BuildTimedGraph(uint64_t iterations, hipEvent_t start, hipEvent_t stop,
                       TimedGraph* out_timed) {
    out_timed->output = static_cast<uint64_t*>(spin_output_);
    out_timed->iterations = iterations;
    ASSERT_EQ(hipSuccess, graph_create_(&out_timed->graph, /*flags=*/0));

    hipGraphNode_t start_node = nullptr;
    ASSERT_EQ(hipSuccess,
              graph_add_event_record_node_(&start_node, out_timed->graph,
                                           /*dependencies=*/nullptr,
                                           /*dependency_count=*/0, start));

    void* arguments[] = {&out_timed->output, &out_timed->iterations};
    const hipKernelNodeParams params = {
        /*.blockDim=*/{1, 1, 1},
        /*.extra=*/nullptr,
        /*.func=*/spin_function_,
        /*.gridDim=*/{1, 1, 1},
        /*.kernelParams=*/arguments,
        /*.sharedMemBytes=*/0,
    };
    hipGraphNode_t kernel_node = nullptr;
    ASSERT_EQ(hipSuccess, graph_add_kernel_node_(
                              &kernel_node, out_timed->graph, &start_node,
                              /*dependency_count=*/1, &params));

    hipGraphNode_t stop_node = nullptr;
    ASSERT_EQ(hipSuccess, graph_add_event_record_node_(
                              &stop_node, out_timed->graph, &kernel_node,
                              /*dependency_count=*/1, stop));

    ASSERT_EQ(hipSuccess,
              graph_instantiate_(&out_timed->executable, out_timed->graph,
                                 nullptr, nullptr, 0));
  }

  void DestroyTimedGraph(TimedGraph* timed) {
    EXPECT_EQ(hipSuccess, graph_exec_destroy_(timed->executable));
    EXPECT_EQ(hipSuccess, graph_destroy_(timed->graph));
    timed->executable = nullptr;
    timed->graph = nullptr;
  }

  // Replays |timed| and reports what the event pair measured, checking on the
  // way that the measurement is of device work rather than of the launch.
  void ReplayAndMeasure(const TimedGraph& timed, hipEvent_t start,
                        hipEvent_t stop, const char* what, float* out_ms) {
    *out_ms = 0.0f;
    const auto host_start = std::chrono::steady_clock::now();
    ASSERT_EQ(hipSuccess, graph_launch_(timed.executable, /*stream=*/nullptr));
    ASSERT_EQ(hipSuccess, event_synchronize_(stop));
    const double host_total_ms = MillisecondsSince(host_start);
    ASSERT_EQ(hipSuccess, event_elapsed_time_(out_ms, start, stop));
    ExpectDeviceTimed(*out_ms, host_total_ms, what);
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
  inline static hipFunction_t spin_function_ = nullptr;
  inline static hipDeviceptr_t spin_output_ = nullptr;

  inline static HipInitFn init_ = nullptr;
  inline static HipGetDeviceFn get_device_ = nullptr;
  inline static HipGetDevicePropertiesFn get_device_properties_ = nullptr;
  inline static HipModuleLoadDataFn module_load_data_ = nullptr;
  inline static HipModuleUnloadFn module_unload_ = nullptr;
  inline static HipModuleGetFunctionFn module_get_function_ = nullptr;
  inline static HipModuleLaunchKernelFn module_launch_kernel_ = nullptr;
  inline static HipDeviceSynchronizeFn device_synchronize_ = nullptr;
  inline static HipMallocFn hip_malloc_ = nullptr;
  inline static HipFreeFn hip_free_ = nullptr;
  inline static HipEventCreateFn event_create_ = nullptr;
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
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  ASSERT_EQ(hipSuccess, event_create_(&start));
  ASSERT_EQ(hipSuccess, event_create_(&stop));

  const auto host_start = std::chrono::steady_clock::now();
  ASSERT_EQ(hipSuccess, event_record_(start, /*stream=*/nullptr));
  ASSERT_EQ(hipSuccess, LaunchSpin(kSpinIterations, /*stream=*/nullptr));
  ASSERT_EQ(hipSuccess, event_record_(stop, /*stream=*/nullptr));

  // Captured before synchronizing so it is the enqueue cost alone.
  const double host_enqueue_ms = MillisecondsSince(host_start);

  ASSERT_EQ(hipSuccess, event_synchronize_(stop));
  const double host_total_ms = MillisecondsSince(host_start);

  ASSERT_EQ(hipSuccess, event_query_(start));
  ASSERT_EQ(hipSuccess, event_query_(stop));

  float elapsed_ms = 0.0f;
  ASSERT_EQ(hipSuccess, event_elapsed_time_(&elapsed_ms, start, stop));
  ExpectDeviceTimed(elapsed_ms, host_total_ms, "direct record");

  EXPECT_LT(host_enqueue_ms, host_total_ms * kMinimumDeviceFraction)
      << "enqueue cost " << host_enqueue_ms
      << " ms is not negligible against the " << host_total_ms
      << " ms of device work, so this test cannot distinguish device timing "
         "from host-enqueue timing";

  EXPECT_EQ(hipSuccess, event_destroy_(start));
  EXPECT_EQ(hipSuccess, event_destroy_(stop));
}

// The same property through graph replay, which reaches the device by a
// different path than a direct record.
TEST_F(HipEventTimingTest, GraphReplayedRecordMeasuresDeviceWork) {
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  ASSERT_EQ(hipSuccess, event_create_(&start));
  ASSERT_EQ(hipSuccess, event_create_(&stop));

  TimedGraph timed;
  ASSERT_NO_FATAL_FAILURE(
      BuildTimedGraph(kSpinIterations, start, stop, &timed));

  const auto host_start = std::chrono::steady_clock::now();
  ASSERT_EQ(hipSuccess, graph_launch_(timed.executable, /*stream=*/nullptr));
  const double host_enqueue_ms = MillisecondsSince(host_start);
  ASSERT_EQ(hipSuccess, event_synchronize_(stop));
  const double host_total_ms = MillisecondsSince(host_start);

  float elapsed_ms = 0.0f;
  ASSERT_EQ(hipSuccess, event_elapsed_time_(&elapsed_ms, start, stop));
  ExpectDeviceTimed(elapsed_ms, host_total_ms, "graph-replayed record");

  EXPECT_LT(host_enqueue_ms, host_total_ms * kMinimumDeviceFraction)
      << "graph launch cost " << host_enqueue_ms
      << " ms is not negligible against the " << host_total_ms
      << " ms of device work, so this test cannot distinguish device timing "
         "from host-enqueue timing";

  DestroyTimedGraph(&timed);
  EXPECT_EQ(hipSuccess, event_destroy_(start));
  EXPECT_EQ(hipSuccess, event_destroy_(stop));
}

// Each replay must re-time the events. Alternating a long graph and a short one
// over one event pair catches an implementation that wrote the ticks once: the
// two reported durations are on the same clock, so comparing them against each
// other cancels the advertised frequency.
TEST_F(HipEventTimingTest, GraphReplayRetimesEachLaunch) {
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  ASSERT_EQ(hipSuccess, event_create_(&start));
  ASSERT_EQ(hipSuccess, event_create_(&stop));

  TimedGraph long_graph;
  TimedGraph short_graph;
  ASSERT_NO_FATAL_FAILURE(
      BuildTimedGraph(kSpinIterations, start, stop, &long_graph));
  ASSERT_NO_FATAL_FAILURE(
      BuildTimedGraph(kShortSpinIterations, start, stop, &short_graph));

  float first_long_ms = 0.0f;
  ASSERT_NO_FATAL_FAILURE(ReplayAndMeasure(
      long_graph, start, stop, "first long replay", &first_long_ms));
  float short_ms = 0.0f;
  ASSERT_NO_FATAL_FAILURE(
      ReplayAndMeasure(short_graph, start, stop, "short replay", &short_ms));
  float second_long_ms = 0.0f;
  ASSERT_NO_FATAL_FAILURE(ReplayAndMeasure(
      long_graph, start, stop, "second long replay", &second_long_ms));

  EXPECT_LT(short_ms * kMinimumMeasuredWorkRatio, first_long_ms)
      << "the short replay reported " << short_ms
      << " ms against the long replay's " << first_long_ms
      << " ms for a tenth of the work, so the replay did not re-time the "
         "events";
  EXPECT_LT(short_ms * kMinimumMeasuredWorkRatio, second_long_ms)
      << "the second long replay reported " << second_long_ms
      << " ms, close to the preceding short replay's " << short_ms
      << " ms, so the replay reported stale ticks";

  DestroyTimedGraph(&short_graph);
  DestroyTimedGraph(&long_graph);
  EXPECT_EQ(hipSuccess, event_destroy_(start));
  EXPECT_EQ(hipSuccess, event_destroy_(stop));
}

// A timing-disabled event still has to order streams; only the timing is gone.
TEST_F(HipEventTimingTest, DisableTimingEventsSynchronizeButDoNotTime) {
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  ASSERT_EQ(hipSuccess,
            event_create_with_flags_(&start, hipEventDisableTiming));
  ASSERT_EQ(hipSuccess, event_create_with_flags_(&stop, hipEventDisableTiming));

  ASSERT_EQ(hipSuccess, event_record_(start, /*stream=*/nullptr));
  ASSERT_EQ(hipSuccess, LaunchSpin(/*iterations=*/1024, /*stream=*/nullptr));
  ASSERT_EQ(hipSuccess, event_record_(stop, /*stream=*/nullptr));
  ASSERT_EQ(hipSuccess, event_synchronize_(stop));
  EXPECT_EQ(hipSuccess, event_query_(start));
  EXPECT_EQ(hipSuccess, event_query_(stop));

  float elapsed_ms = -1.0f;
  EXPECT_EQ(hipErrorInvalidHandle,
            event_elapsed_time_(&elapsed_ms, start, stop));
  EXPECT_FLOAT_EQ(-1.0f, elapsed_ms) << "a refused call wrote a duration";

  EXPECT_EQ(hipSuccess, event_destroy_(start));
  EXPECT_EQ(hipSuccess, event_destroy_(stop));
}

}  // namespace
