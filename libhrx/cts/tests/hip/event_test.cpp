// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <dlfcn.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"

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
using HipStreamCreateFn = hipError_t (*)(hipStream_t* stream);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t stream);
using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t stream);
using HipStreamWaitEventFn = hipError_t (*)(hipStream_t stream,
                                            hipEvent_t event,
                                            unsigned int flags);
using HipStreamBeginCaptureFn = hipError_t (*)(hipStream_t stream,
                                               hipStreamCaptureMode mode);
using HipStreamEndCaptureFn = hipError_t (*)(hipStream_t stream,
                                             hipGraph_t* graph);
using HipStreamIsCapturingFn =
    hipError_t (*)(hipStream_t stream, hipStreamCaptureStatus* capture_status);
using HipEventCreateFn = hipError_t (*)(hipEvent_t* event);
using HipEventCreateWithFlagsFn = hipError_t (*)(hipEvent_t* event,
                                                 unsigned int flags);
using HipEventDestroyFn = hipError_t (*)(hipEvent_t event);
using HipEventRecordFn = hipError_t (*)(hipEvent_t event, hipStream_t stream);
using HipEventQueryFn = hipError_t (*)(hipEvent_t event);
using HipEventSynchronizeFn = hipError_t (*)(hipEvent_t event);
using HipEventElapsedTimeFn = hipError_t (*)(float* ms, hipEvent_t start,
                                             hipEvent_t stop);
using HipLaunchHostFuncFn = hipError_t (*)(hipStream_t stream, hipHostFn_t fn,
                                           void* user_data);
using HipMallocAsyncFn = hipError_t (*)(void** device_ptr, size_t size,
                                        hipStream_t stream);
using HipMemcpyAsyncFn = hipError_t (*)(void* destination, const void* source,
                                        size_t size_bytes, hipMemcpyKind kind,
                                        hipStream_t stream);
using HipFreeAsyncFn = hipError_t (*)(void* device_ptr, hipStream_t stream);
using HipGraphCreateFn = hipError_t (*)(hipGraph_t* graph, unsigned int flags);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t graph);
using HipGraphAddEventRecordNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, hipEvent_t event);
using HipGraphAddEventWaitNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, hipEvent_t event);
using HipGraphEventRecordNodeSetEventFn = hipError_t (*)(hipGraphNode_t node,
                                                         hipEvent_t event);
using HipGraphEventWaitNodeSetEventFn = hipError_t (*)(hipGraphNode_t node,
                                                       hipEvent_t event);
using HipGraphAddHostNodeFn = hipError_t (*)(hipGraphNode_t* node,
                                             hipGraph_t graph,
                                             const hipGraphNode_t* dependencies,
                                             size_t dependency_count,
                                             const void* node_params);
using HipGraphAddChildGraphNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, hipGraph_t child_graph);
using HipGraphExecChildGraphNodeSetParamsFn = hipError_t (*)(
    hipGraphExec_t graph_exec, hipGraphNode_t node, hipGraph_t child_graph);
using HipGraphInstantiateFn = hipError_t (*)(hipGraphExec_t* graph_exec,
                                             hipGraph_t graph,
                                             hipGraphNode_t* error_node,
                                             char* log_buffer,
                                             size_t buffer_size);
using HipGraphLaunchFn = hipError_t (*)(hipGraphExec_t graph_exec,
                                        hipStream_t stream);
using HipGraphExecDestroyFn = hipError_t (*)(hipGraphExec_t graph_exec);
using HipGraphExecEventRecordNodeSetEventFn = hipError_t (*)(
    hipGraphExec_t graph_exec, hipGraphNode_t node, hipEvent_t event);
using HipGraphExecEventWaitNodeSetEventFn = hipError_t (*)(
    hipGraphExec_t graph_exec, hipGraphNode_t node, hipEvent_t event);
using HipCtxCreateFn = hipError_t (*)(hipCtx_t* ctx, unsigned int flags,
                                      hipDevice_t device);
using HipCtxDestroyFn = hipError_t (*)(hipCtx_t ctx);
using HipCtxGetCurrentFn = hipError_t (*)(hipCtx_t* ctx);
using HipCtxSetCurrentFn = hipError_t (*)(hipCtx_t ctx);

// Host callback that parks a stream timeline until the test thread releases it.
struct StreamGate {
  // Set by the callback once it has started running.
  std::atomic<bool> entered{false};
  // Set by the test thread to let the callback return.
  std::atomic<bool> released{false};
  // Set by the callback as its final store. The callback writes through a
  // pointer into the frame that enqueued it, so that frame must outlive this.
  std::atomic<bool> finished{false};
};

void GateHostFunction(void* user_data) {
  auto* gate = static_cast<StreamGate*>(user_data);
  gate->entered.store(true, std::memory_order_release);
  while (!gate->released.load(std::memory_order_acquire)) {
    sched_yield();
  }
  gate->finished.store(true, std::memory_order_release);
}

// Host callback that samples whether a gate callback had already finished when
// it ran.
struct StreamMarker {
  // Gate sampled by the callback, borrowed from the enqueuing frame.
  const StreamGate* gate = nullptr;
  // Whether |gate| had finished when the callback sampled it. Only meaningful
  // once |finished| is set.
  std::atomic<bool> saw_gate_finished{false};
  // Set by the callback as its final store. See StreamGate.
  std::atomic<bool> finished{false};
};

void MarkerHostFunction(void* user_data) {
  auto* marker = static_cast<StreamMarker*>(user_data);
  marker->saw_gate_finished.store(
      marker->gate->finished.load(std::memory_order_acquire),
      std::memory_order_release);
  marker->finished.store(true, std::memory_order_release);
}

// Host callback whose only action is to store true through |user_data|.
void RanHostFunction(void* user_data) {
  static_cast<std::atomic<bool>*>(user_data)->store(true,
                                                    std::memory_order_release);
}

// Keeps the enclosing frame alive until every host callback registered with it
// has returned, releasing the registered gate first so a parked callback can
// get there. Callbacks are registered only once their enqueue has succeeded.
class ScopedHostCallbacks {
 public:
  ScopedHostCallbacks() = default;
  ScopedHostCallbacks(const ScopedHostCallbacks&) = delete;
  ScopedHostCallbacks& operator=(const ScopedHostCallbacks&) = delete;
  ~ScopedHostCallbacks() {
    ReleaseGate();
    for (const std::atomic<bool>* finished : pending_) {
      while (!finished->load(std::memory_order_acquire)) {
        sched_yield();
      }
    }
  }

  // Registers a parked gate callback whose enqueue has succeeded. Only one gate
  // may be registered; nothing could release a displaced one.
  void AddGate(StreamGate* gate) {
    ASSERT_EQ(nullptr, gate_) << "a gate is already registered";
    gate_ = gate;
    Add(gate->finished);
  }

  // Registers a callback whose enqueue has succeeded and that stores
  // |finished| as its final action.
  void Add(const std::atomic<bool>& finished) { pending_.push_back(&finished); }

  // Lets the registered gate callback return. Idempotent: the destructor
  // repeats it for the exit paths that never get here.
  void ReleaseGate() {
    if (gate_) {
      gate_->released.store(true, std::memory_order_release);
    }
  }

 private:
  // Gate released before the destructor waits, or null when none is registered.
  // Borrowed from the enclosing frame.
  StreamGate* gate_ = nullptr;
  // Completion flags of the registered callbacks, borrowed from that frame.
  std::vector<const std::atomic<bool>*> pending_;
};

// Host callback that parks every launch reaching it until the test thread
// releases that launch, so one graph can be replayed with a different hold each
// time.
struct ReplayGate {
  // Launches whose callback has started, and so the generation of the launch
  // parked right now.
  std::atomic<int> entered{0};
  // Highest generation the test thread has let return.
  std::atomic<int> released{0};
  // Launches whose callback has returned. The callback writes through a
  // pointer into the frame that instantiated the graph, so that frame must
  // outlive every launch of it.
  std::atomic<int> finished{0};
};

void ReplayGateHostFunction(void* user_data) {
  auto* gate = static_cast<ReplayGate*>(user_data);
  const int generation =
      gate->entered.fetch_add(1, std::memory_order_acq_rel) + 1;
  while (gate->released.load(std::memory_order_acquire) < generation) {
    sched_yield();
  }
  gate->finished.fetch_add(1, std::memory_order_acq_rel);
}

// Keeps the enclosing frame alive until every launch registered with it has run
// its gate callback to completion, releasing any still parked first. Launches
// are registered only once their enqueue has succeeded.
class ScopedReplayGate {
 public:
  explicit ScopedReplayGate(ReplayGate* gate) : gate_(gate) {}
  ScopedReplayGate(const ScopedReplayGate&) = delete;
  ScopedReplayGate& operator=(const ScopedReplayGate&) = delete;
  ~ScopedReplayGate() {
    gate_->released.store(launch_count_, std::memory_order_release);
    while (gate_->finished.load(std::memory_order_acquire) < launch_count_) {
      sched_yield();
    }
  }

  // Registers a launch whose enqueue has succeeded and returns its generation.
  int AddLaunch() { return ++launch_count_; }

  // Returns once |generation|'s callback is parked.
  void WaitUntilParked(int generation) const {
    while (gate_->entered.load(std::memory_order_acquire) < generation) {
      sched_yield();
    }
  }

  // Lets |generation|'s parked callback return. Generations are released in
  // order, so this never lowers what an earlier call published.
  void Release(int generation) {
    gate_->released.store(generation, std::memory_order_release);
  }

 private:
  // Gate the registered launches park on, borrowed from the enclosing frame.
  ReplayGate* gate_;
  // Launches registered so far, which is the generation of the last one.
  int launch_count_ = 0;
};

double MillisecondsSince(std::chrono::steady_clock::time_point from) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - from)
      .count();
}

// Milliseconds the test thread parks the queue for between two records. Long
// enough that the interval it opens is orders of magnitude above the time two
// hipEventRecord calls take, short enough to keep the suite quick.
constexpr double kGateHoldMs = 100.0;

// The short hold of the replay test, a tenth of the long one.
constexpr double kShortGateHoldMs = kGateHoldMs / 10.0;

// Fraction of the observed hold a device-timed interval must exceed. Correct
// behavior reports the hold plus microseconds; timing the host's enqueue
// reports the fraction of a millisecond two records take, which is more than
// two orders of magnitude below this.
constexpr double kMinimumHoldFraction = 1.0 / 4.0;

// Multiple of the observed hold a device-timed interval must stay under, which
// is what catches ticks converted on a rate nothing advertised.
constexpr double kMaximumHoldFactor = 4.0;

// Smallest factor by which a long replay's reported interval must exceed a
// short replay's. Below the ratio of the holds for noise, well above 1 to catch
// a replay reporting the ticks of an earlier one.
constexpr float kMinimumReplayRatio = 3.0f;

class HipEventTest : public ::testing::Test {
 protected:
  void SetUp() override {
    library_ = dlopen(CandidateLibPath(), RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(nullptr, library_)
        << "cannot dlopen " << CandidateLibPath() << ": " << dlerror();

    hip_.init = ResolveHipSymbol<HipInitFn>(library_, "hipInit");
    hip_.stream_create =
        ResolveHipSymbol<HipStreamCreateFn>(library_, "hipStreamCreate");
    hip_.stream_destroy =
        ResolveHipSymbol<HipStreamDestroyFn>(library_, "hipStreamDestroy");
    hip_.stream_synchronize = ResolveHipSymbol<HipStreamSynchronizeFn>(
        library_, "hipStreamSynchronize");
    hip_.stream_wait_event =
        ResolveHipSymbol<HipStreamWaitEventFn>(library_, "hipStreamWaitEvent");
    hip_.stream_begin_capture = ResolveHipSymbol<HipStreamBeginCaptureFn>(
        library_, "hipStreamBeginCapture");
    hip_.stream_end_capture = ResolveHipSymbol<HipStreamEndCaptureFn>(
        library_, "hipStreamEndCapture");
    hip_.stream_is_capturing = ResolveHipSymbol<HipStreamIsCapturingFn>(
        library_, "hipStreamIsCapturing");
    hip_.event_create =
        ResolveHipSymbol<HipEventCreateFn>(library_, "hipEventCreate");
    hip_.event_create_with_flags = ResolveHipSymbol<HipEventCreateWithFlagsFn>(
        library_, "hipEventCreateWithFlags");
    hip_.event_destroy =
        ResolveHipSymbol<HipEventDestroyFn>(library_, "hipEventDestroy");
    hip_.event_record =
        ResolveHipSymbol<HipEventRecordFn>(library_, "hipEventRecord");
    hip_.event_query =
        ResolveHipSymbol<HipEventQueryFn>(library_, "hipEventQuery");
    hip_.event_synchronize = ResolveHipSymbol<HipEventSynchronizeFn>(
        library_, "hipEventSynchronize");
    hip_.event_elapsed_time = ResolveHipSymbol<HipEventElapsedTimeFn>(
        library_, "hipEventElapsedTime");
    hip_.launch_host_func =
        ResolveHipSymbol<HipLaunchHostFuncFn>(library_, "hipLaunchHostFunc");
    hip_.malloc_async =
        ResolveHipSymbol<HipMallocAsyncFn>(library_, "hipMallocAsync");
    hip_.free_async =
        ResolveHipSymbol<HipFreeAsyncFn>(library_, "hipFreeAsync");
    hip_.memcpy_async =
        ResolveHipSymbol<HipMemcpyAsyncFn>(library_, "hipMemcpyAsync");
    hip_.graph_create =
        ResolveHipSymbol<HipGraphCreateFn>(library_, "hipGraphCreate");
    hip_.graph_destroy =
        ResolveHipSymbol<HipGraphDestroyFn>(library_, "hipGraphDestroy");
    hip_.graph_add_event_record_node =
        ResolveHipSymbol<HipGraphAddEventRecordNodeFn>(
            library_, "hipGraphAddEventRecordNode");
    hip_.graph_add_event_wait_node =
        ResolveHipSymbol<HipGraphAddEventWaitNodeFn>(
            library_, "hipGraphAddEventWaitNode");
    hip_.graph_event_record_node_set_event =
        ResolveHipSymbol<HipGraphEventRecordNodeSetEventFn>(
            library_, "hipGraphEventRecordNodeSetEvent");
    hip_.graph_event_wait_node_set_event =
        ResolveHipSymbol<HipGraphEventWaitNodeSetEventFn>(
            library_, "hipGraphEventWaitNodeSetEvent");
    hip_.graph_add_host_node = ResolveHipSymbol<HipGraphAddHostNodeFn>(
        library_, "hipGraphAddHostNode");
    hip_.graph_add_child_graph_node =
        ResolveHipSymbol<HipGraphAddChildGraphNodeFn>(
            library_, "hipGraphAddChildGraphNode");
    hip_.graph_exec_child_graph_node_set_params =
        ResolveHipSymbol<HipGraphExecChildGraphNodeSetParamsFn>(
            library_, "hipGraphExecChildGraphNodeSetParams");
    hip_.graph_instantiate = ResolveHipSymbol<HipGraphInstantiateFn>(
        library_, "hipGraphInstantiate");
    hip_.graph_launch =
        ResolveHipSymbol<HipGraphLaunchFn>(library_, "hipGraphLaunch");
    hip_.graph_exec_destroy = ResolveHipSymbol<HipGraphExecDestroyFn>(
        library_, "hipGraphExecDestroy");
    hip_.graph_exec_event_record_node_set_event =
        ResolveHipSymbol<HipGraphExecEventRecordNodeSetEventFn>(
            library_, "hipGraphExecEventRecordNodeSetEvent");
    hip_.graph_exec_event_wait_node_set_event =
        ResolveHipSymbol<HipGraphExecEventWaitNodeSetEventFn>(
            library_, "hipGraphExecEventWaitNodeSetEvent");
    hip_.ctx_create =
        ResolveHipSymbol<HipCtxCreateFn>(library_, "hipCtxCreate");
    hip_.ctx_destroy =
        ResolveHipSymbol<HipCtxDestroyFn>(library_, "hipCtxDestroy");
    hip_.ctx_get_current =
        ResolveHipSymbol<HipCtxGetCurrentFn>(library_, "hipCtxGetCurrent");
    hip_.ctx_set_current =
        ResolveHipSymbol<HipCtxSetCurrentFn>(library_, "hipCtxSetCurrent");

    ASSERT_NE(nullptr, hip_.init);
    ASSERT_NE(nullptr, hip_.stream_create);
    ASSERT_NE(nullptr, hip_.stream_destroy);
    ASSERT_NE(nullptr, hip_.stream_synchronize);
    ASSERT_NE(nullptr, hip_.stream_wait_event);
    ASSERT_NE(nullptr, hip_.stream_begin_capture);
    ASSERT_NE(nullptr, hip_.stream_end_capture);
    ASSERT_NE(nullptr, hip_.stream_is_capturing);
    ASSERT_NE(nullptr, hip_.event_create);
    ASSERT_NE(nullptr, hip_.event_create_with_flags);
    ASSERT_NE(nullptr, hip_.event_destroy);
    ASSERT_NE(nullptr, hip_.event_record);
    ASSERT_NE(nullptr, hip_.event_query);
    ASSERT_NE(nullptr, hip_.event_synchronize);
    ASSERT_NE(nullptr, hip_.event_elapsed_time);
    ASSERT_NE(nullptr, hip_.launch_host_func);
    ASSERT_NE(nullptr, hip_.malloc_async);
    ASSERT_NE(nullptr, hip_.free_async);
    ASSERT_NE(nullptr, hip_.memcpy_async);
    ASSERT_NE(nullptr, hip_.graph_create);
    ASSERT_NE(nullptr, hip_.graph_destroy);
    ASSERT_NE(nullptr, hip_.graph_add_event_record_node);
    ASSERT_NE(nullptr, hip_.graph_add_event_wait_node);
    ASSERT_NE(nullptr, hip_.graph_event_record_node_set_event);
    ASSERT_NE(nullptr, hip_.graph_event_wait_node_set_event);
    ASSERT_NE(nullptr, hip_.graph_add_host_node);
    ASSERT_NE(nullptr, hip_.graph_add_child_graph_node);
    ASSERT_NE(nullptr, hip_.graph_exec_child_graph_node_set_params);
    ASSERT_NE(nullptr, hip_.graph_instantiate);
    ASSERT_NE(nullptr, hip_.graph_launch);
    ASSERT_NE(nullptr, hip_.graph_exec_destroy);
    ASSERT_NE(nullptr, hip_.graph_exec_event_record_node_set_event);
    ASSERT_NE(nullptr, hip_.graph_exec_event_wait_node_set_event);
    ASSERT_NE(nullptr, hip_.ctx_create);
    ASSERT_NE(nullptr, hip_.ctx_destroy);
    ASSERT_NE(nullptr, hip_.ctx_get_current);
    ASSERT_NE(nullptr, hip_.ctx_set_current);

    const hipError_t init_result = hip_.init(/*flags=*/0);
    if (init_result == hipErrorNoDevice) {
      GTEST_SKIP() << "no HIP device available";
    }
    ASSERT_EQ(hipSuccess, init_result) << "hipInit failed";
  }

  // Destroys the handles the body created, in reverse order of the dependencies
  // between handle kinds, so a body that left early on a failed assertion does
  // not also report leaks.
  void TearDown() override {
    for (auto it = graph_execs_.rbegin(); it != graph_execs_.rend(); ++it) {
      EXPECT_EQ(hipSuccess, hip_.graph_exec_destroy(*it));
    }
    for (auto it = graphs_.rbegin(); it != graphs_.rend(); ++it) {
      EXPECT_EQ(hipSuccess, hip_.graph_destroy(*it));
    }
    for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
      EXPECT_EQ(hipSuccess, hip_.event_destroy(*it));
    }
    for (auto it = streams_.rbegin(); it != streams_.rend(); ++it) {
      EXPECT_EQ(hipSuccess, hip_.stream_destroy(*it));
    }
  }

  // Creates a stream owned by the fixture.
  hipStream_t CreateStream() {
    hipStream_t stream = nullptr;
    EXPECT_EQ(hipSuccess, hip_.stream_create(&stream));
    if (stream) streams_.push_back(stream);
    return stream;
  }

  // Destroys a stream the fixture owns ahead of TearDown.
  void DestroyStream(hipStream_t stream) {
    streams_.erase(std::remove(streams_.begin(), streams_.end(), stream),
                   streams_.end());
    EXPECT_EQ(hipSuccess, hip_.stream_destroy(stream));
  }

  // Creates an event owned by the fixture.
  hipEvent_t CreateEvent() {
    hipEvent_t event = nullptr;
    EXPECT_EQ(hipSuccess, hip_.event_create(&event));
    if (event) events_.push_back(event);
    return event;
  }

  // Creates an event carrying |flags|, owned by the fixture.
  hipEvent_t CreateEventWithFlags(unsigned int flags) {
    hipEvent_t event = nullptr;
    EXPECT_EQ(hipSuccess, hip_.event_create_with_flags(&event, flags));
    if (event) events_.push_back(event);
    return event;
  }

  // Destroys an event the fixture owns ahead of TearDown.
  void DestroyEvent(hipEvent_t event) {
    events_.erase(std::remove(events_.begin(), events_.end(), event),
                  events_.end());
    EXPECT_EQ(hipSuccess, hip_.event_destroy(event));
  }

  // Creates an empty graph owned by the fixture.
  hipGraph_t CreateGraph() {
    hipGraph_t graph = nullptr;
    EXPECT_EQ(hipSuccess, hip_.graph_create(&graph, /*flags=*/0));
    TrackGraph(graph);
    return graph;
  }

  // Hands a graph the body obtained some other way - ending a stream capture,
  // for one - to the fixture so TearDown destroys it. Ignores null.
  void TrackGraph(hipGraph_t graph) {
    if (graph) graphs_.push_back(graph);
  }

  // Destroys a graph the fixture owns ahead of TearDown.
  void DestroyGraph(hipGraph_t graph) {
    graphs_.erase(std::remove(graphs_.begin(), graphs_.end(), graph),
                  graphs_.end());
    EXPECT_EQ(hipSuccess, hip_.graph_destroy(graph));
  }

  // Instantiates |graph| into an executable owned by the fixture.
  hipGraphExec_t InstantiateGraph(hipGraph_t graph) {
    hipGraphExec_t graph_exec = nullptr;
    EXPECT_EQ(hipSuccess, hip_.graph_instantiate(
                              &graph_exec, graph, /*error_node=*/nullptr,
                              /*log_buffer=*/nullptr, /*buffer_size=*/0));
    if (graph_exec) graph_execs_.push_back(graph_exec);
    return graph_exec;
  }

  // Destroys an executable the fixture owns ahead of TearDown.
  void DestroyGraphExec(hipGraphExec_t graph_exec) {
    graph_execs_.erase(
        std::remove(graph_execs_.begin(), graph_execs_.end(), graph_exec),
        graph_execs_.end());
    EXPECT_EQ(hipSuccess, hip_.graph_exec_destroy(graph_exec));
  }

  // Records |event| on |stream| and waits for it |count| times, leaving both
  // the stream and the event drained.
  void AdvanceEventOnStream(hipEvent_t event, hipStream_t stream, int count) {
    for (int i = 0; i < count; ++i) {
      ASSERT_EQ(hipSuccess, hip_.event_record(event, stream));
      ASSERT_EQ(hipSuccess, hip_.event_synchronize(event));
    }
    ASSERT_EQ(hipSuccess, hip_.stream_synchronize(stream));
  }

  // Instantiates a graph whose only node records |event|, so the record is the
  // launch's last block and its point lands on the launching stream timeline.
  hipGraphExec_t InstantiateGraphRecordingAtTheEnd(hipEvent_t event) {
    hipGraph_t graph = CreateGraph();
    EXPECT_NE(nullptr, graph);
    if (!graph) return nullptr;
    hipGraphNode_t record_node = nullptr;
    EXPECT_EQ(hipSuccess, hip_.graph_add_event_record_node(
                              &record_node, graph, /*dependencies=*/nullptr,
                              /*dependency_count=*/0, event));
    return InstantiateGraph(graph);
  }

  // Instantiates a graph that records |event| and then runs a host node, which
  // puts the record in a partition of its own so its point lands on a timeline
  // internal to the launch. The host node stores |host_node_ran|, which must
  // outlive every launch of the returned executable.
  hipGraphExec_t InstantiateGraphRecordingBeforeAHostNode(
      hipEvent_t event, std::atomic<bool>* host_node_ran) {
    hipGraph_t graph = CreateGraph();
    EXPECT_NE(nullptr, graph);
    if (!graph) return nullptr;
    hipGraphNode_t record_node = nullptr;
    EXPECT_EQ(hipSuccess, hip_.graph_add_event_record_node(
                              &record_node, graph, /*dependencies=*/nullptr,
                              /*dependency_count=*/0, event));
    hipHostNodeParams host_params = {};
    host_params.fn = &RanHostFunction;
    host_params.userData = host_node_ran;
    hipGraphNode_t host_node = nullptr;
    EXPECT_EQ(hipSuccess,
              hip_.graph_add_host_node(&host_node, graph, &record_node,
                                       /*dependency_count=*/1, &host_params));
    return InstantiateGraph(graph);
  }

  // Instantiates a graph that records |start|, runs |fn| over |user_data|, and
  // then records |stop|, so a replay opens a device interval the host node
  // controls. |user_data| must outlive every launch of the returned executable.
  hipGraphExec_t InstantiateGraphRecordingAroundAHostNode(hipEvent_t start,
                                                          hipEvent_t stop,
                                                          hipHostFn_t fn,
                                                          void* user_data) {
    hipGraph_t graph = CreateGraph();
    EXPECT_NE(nullptr, graph);
    if (!graph) return nullptr;
    hipGraphNode_t start_node = nullptr;
    EXPECT_EQ(hipSuccess, hip_.graph_add_event_record_node(
                              &start_node, graph, /*dependencies=*/nullptr,
                              /*dependency_count=*/0, start));
    hipHostNodeParams host_params = {};
    host_params.fn = fn;
    host_params.userData = user_data;
    hipGraphNode_t host_node = nullptr;
    EXPECT_EQ(hipSuccess,
              hip_.graph_add_host_node(&host_node, graph, &start_node,
                                       /*dependency_count=*/1, &host_params));
    hipGraphNode_t stop_node = nullptr;
    EXPECT_EQ(hipSuccess,
              hip_.graph_add_event_record_node(&stop_node, graph, &host_node,
                                               /*dependency_count=*/1, stop));
    return InstantiateGraph(graph);
  }

  // Parks the test thread for kGateHoldMs, releases the gate, and returns how
  // long the gate was held. The device sees at least this interval between the
  // records either side of it.
  double HoldGateAndRelease(ScopedHostCallbacks* callbacks) {
    const auto held_from = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(
        std::chrono::duration<double, std::milli>(kGateHoldMs));
    const double held_ms = MillisecondsSince(held_from);
    callbacks->ReleaseGate();
    return held_ms;
  }

  // Replays |graph_exec| on |stream|, holds its gate for |hold_ms|, and reports
  // what the event pair measured across the hold together with how long the
  // gate was actually held.
  void ReplayHoldingTheGateAndMeasure(hipGraphExec_t graph_exec,
                                      hipStream_t stream,
                                      ScopedReplayGate* gate, double hold_ms,
                                      hipEvent_t start, hipEvent_t stop,
                                      float* out_reported_ms,
                                      double* out_held_ms) {
    ASSERT_EQ(hipSuccess, hip_.graph_launch(graph_exec, stream));
    const int generation = gate->AddLaunch();
    gate->WaitUntilParked(generation);
    const auto held_from = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(
        std::chrono::duration<double, std::milli>(hold_ms));
    *out_held_ms = MillisecondsSince(held_from);
    gate->Release(generation);
    ASSERT_EQ(hipSuccess, hip_.event_synchronize(stop));
    ASSERT_EQ(hipSuccess,
              hip_.event_elapsed_time(out_reported_ms, start, stop));
  }

  // Asserts |reported_ms| is the device interval a gate held open for
  // |held_ms| rather than the time the host spent issuing the records.
  void ExpectMeasuredTheHold(float reported_ms, double held_ms,
                             const char* what) {
    EXPECT_GT(reported_ms, held_ms * kMinimumHoldFraction)
        << what << ": reported " << reported_ms
        << " ms for a device interval the host held open for " << held_ms
        << " ms, which is what timing the enqueue instead of the device "
           "reports";
    EXPECT_LT(reported_ms, held_ms * kMaximumHoldFactor)
        << what << ": reported " << reported_ms
        << " ms for a device interval the host held open for " << held_ms
        << " ms, so the ticks were converted on a rate nothing advertised";
  }

  // Allocates |size| bytes from the device's default pool on |stream|.
  void* AllocateAsync(hipStream_t stream, size_t size) {
    void* device_ptr = nullptr;
    EXPECT_EQ(hipSuccess, hip_.malloc_async(&device_ptr, size, stream));
    return device_ptr;
  }

  // Enqueues |gate| on |stream|, registers it with |callbacks|, and returns
  // once the callback is running and the stream provably holds unfinished work.
  void EnqueueGateAndWaitUntilEntered(hipStream_t stream, StreamGate* gate,
                                      ScopedHostCallbacks* callbacks) {
    ASSERT_EQ(hipSuccess,
              hip_.launch_host_func(stream, &GateHostFunction, gate));
    ASSERT_NO_FATAL_FAILURE(callbacks->AddGate(gate));
    while (!gate->entered.load(std::memory_order_acquire)) {
      sched_yield();
    }
  }

  // Flag a graph host node built inline by a test body stores through
  // RanHostFunction. It must outlive every launch of an executable holding
  // that node, and the fixture owns executables until TearDown, so it is held
  // here for the whole test.
  std::atomic<bool> graph_host_node_ran_{false};

  // HIP shim under test. Intentionally never dlclose()d: the shim owns
  // process-global device state shared by every test in this file.
  void* library_ = nullptr;
  // Entry points resolved from |library_| by SetUp, each named for the hip*
  // symbol it was resolved from. SetUp fails if any is unresolved, so every
  // one is non-null for the duration of a test body.
  struct {
    HipInitFn init;
    HipStreamCreateFn stream_create;
    HipStreamDestroyFn stream_destroy;
    HipStreamSynchronizeFn stream_synchronize;
    HipStreamWaitEventFn stream_wait_event;
    HipStreamBeginCaptureFn stream_begin_capture;
    HipStreamEndCaptureFn stream_end_capture;
    HipStreamIsCapturingFn stream_is_capturing;
    HipEventCreateFn event_create;
    HipEventCreateWithFlagsFn event_create_with_flags;
    HipEventDestroyFn event_destroy;
    HipEventRecordFn event_record;
    HipEventQueryFn event_query;
    HipEventSynchronizeFn event_synchronize;
    HipEventElapsedTimeFn event_elapsed_time;
    HipLaunchHostFuncFn launch_host_func;
    HipMallocAsyncFn malloc_async;
    HipFreeAsyncFn free_async;
    HipMemcpyAsyncFn memcpy_async;
    HipGraphCreateFn graph_create;
    HipGraphDestroyFn graph_destroy;
    HipGraphAddEventRecordNodeFn graph_add_event_record_node;
    HipGraphAddEventWaitNodeFn graph_add_event_wait_node;
    HipGraphEventRecordNodeSetEventFn graph_event_record_node_set_event;
    HipGraphEventWaitNodeSetEventFn graph_event_wait_node_set_event;
    HipGraphAddHostNodeFn graph_add_host_node;
    HipGraphAddChildGraphNodeFn graph_add_child_graph_node;
    HipGraphExecChildGraphNodeSetParamsFn
        graph_exec_child_graph_node_set_params;
    HipGraphInstantiateFn graph_instantiate;
    HipGraphLaunchFn graph_launch;
    HipGraphExecDestroyFn graph_exec_destroy;
    HipGraphExecEventRecordNodeSetEventFn
        graph_exec_event_record_node_set_event;
    HipGraphExecEventWaitNodeSetEventFn graph_exec_event_wait_node_set_event;
    HipCtxCreateFn ctx_create;
    HipCtxDestroyFn ctx_destroy;
    HipCtxGetCurrentFn ctx_get_current;
    HipCtxSetCurrentFn ctx_set_current;
  } hip_ = {};

 private:
  // Streams created through CreateStream, destroyed by TearDown.
  std::vector<hipStream_t> streams_;
  // Events created through CreateEvent, destroyed by TearDown.
  std::vector<hipEvent_t> events_;
  // Graphs created through CreateGraph, destroyed by TearDown.
  std::vector<hipGraph_t> graphs_;
  // Executables created through InstantiateGraph, destroyed by TearDown.
  std::vector<hipGraphExec_t> graph_execs_;
};

TEST_F(HipEventTest, CrossStreamRerecordDoesNotReportStaleCompletion) {
  hipStream_t stream_a = CreateStream();
  ASSERT_NE(nullptr, stream_a);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, stream_a, /*count=*/3));

  // A fresh stream starts its timeline at zero, behind the stream that has been
  // recording the event.
  hipStream_t stream_b = CreateStream();
  ASSERT_NE(nullptr, stream_b);

  StreamGate gate;
  ScopedHostCallbacks callbacks;
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(stream_b, &gate, &callbacks));

  ASSERT_EQ(hipSuccess, hip_.event_record(event, stream_b));
  EXPECT_EQ(hipErrorNotReady, hip_.event_query(event))
      << "event completed while the gate callback holding its recording stream "
         "is still running";

  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, hip_.event_synchronize(event));
  EXPECT_EQ(hipSuccess, hip_.event_query(event));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(stream_b));
}

// A chained stream hangs in hipStreamSynchronize rather than failing here.
//
// Timing is disabled on the event because a timed record enqueues a device
// timestamp where an untimed one is semaphore bookkeeping the queue never
// sees, and one hardware queue serves every stream: B's record would sit
// behind the gate parked on A and B could not drain until the gate is
// released, which is the one thing this test needs to happen while it is
// held. The two records differ only in which queue operation they enqueue -
// the timeline value they reserve, the point they commit and the recording
// stream they adopt are identical - so the chaining this test names is
// exercised either way.
TEST_F(HipEventTest, CrossStreamRerecordDoesNotChainTheNewStreamToTheOldOne) {
  hipStream_t stream_a = CreateStream();
  ASSERT_NE(nullptr, stream_a);
  hipStream_t stream_b = CreateStream();
  ASSERT_NE(nullptr, stream_b);
  hipEvent_t event = CreateEventWithFlags(hipEventDisableTiming);
  ASSERT_NE(nullptr, event);

  StreamGate gate;
  ScopedHostCallbacks callbacks;
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(stream_a, &gate, &callbacks));

  ASSERT_EQ(hipSuccess, hip_.event_record(event, stream_a));
  ASSERT_EQ(hipSuccess, hip_.event_record(event, stream_b));

  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(stream_b));
  // The event names the last record, on the stream that just drained, so
  // waiting on it must not reach back to the parked stream either.
  EXPECT_EQ(hipSuccess, hip_.event_query(event));
  EXPECT_EQ(hipSuccess, hip_.event_synchronize(event));
  EXPECT_FALSE(gate.finished.load(std::memory_order_acquire))
      << "the gate was released before the independent stream was observed";

  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(stream_a));
}

// Pins the handle-level contract only - the event keeps the stream object alive
// for its capture state, so the timeline reference is the case below.
TEST_F(HipEventTest, EventStaysUsableAfterItsRecordingStreamHandleIsDestroyed) {
  hipStream_t stream_a = CreateStream();
  ASSERT_NE(nullptr, stream_a);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, stream_a, /*count=*/2));

  ASSERT_NO_FATAL_FAILURE(DestroyStream(stream_a));
  EXPECT_EQ(hipSuccess, hip_.event_query(event));
  EXPECT_EQ(hipSuccess, hip_.event_synchronize(event));

  hipStream_t stream_b = CreateStream();
  ASSERT_NE(nullptr, stream_b);
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, stream_b, /*count=*/1));
  EXPECT_EQ(hipSuccess, hip_.event_query(event));
}

// The one case where the event's own reference is all that keeps the timeline
// alive; without it the query below is a use after free.
TEST_F(HipEventTest, EventStaysUsableAfterTheRecordingGraphExecutableIsGone) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  // The node behind the record node puts the record in a partition of its own;
  // otherwise it would land in the final partition and name the stream
  // timeline, which outlives the executable.
  hipGraph_t graph = CreateGraph();
  ASSERT_NE(nullptr, graph);
  hipGraphNode_t record_node = nullptr;
  ASSERT_EQ(hipSuccess, hip_.graph_add_event_record_node(
                            &record_node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, event));
  std::atomic<bool> host_node_ran{false};
  hipHostNodeParams host_params = {};
  host_params.fn = &RanHostFunction;
  host_params.userData = &host_node_ran;
  hipGraphNode_t host_node = nullptr;
  ASSERT_EQ(hipSuccess,
            hip_.graph_add_host_node(&host_node, graph, &record_node,
                                     /*dependency_count=*/1, &host_params));
  hipGraphExec_t graph_exec = InstantiateGraph(graph);
  ASSERT_NE(nullptr, graph_exec);

  ScopedHostCallbacks callbacks;
  ASSERT_EQ(hipSuccess, hip_.graph_launch(graph_exec, stream));
  callbacks.Add(host_node_ran);
  ASSERT_EQ(hipSuccess, hip_.stream_synchronize(stream));

  ASSERT_NO_FATAL_FAILURE(DestroyGraphExec(graph_exec));
  ASSERT_NO_FATAL_FAILURE(DestroyGraph(graph));

  EXPECT_EQ(hipSuccess, hip_.event_query(event));
  EXPECT_EQ(hipSuccess, hip_.event_synchronize(event));

  hipStream_t stream_b = CreateStream();
  ASSERT_NE(nullptr, stream_b);
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, stream_b, /*count=*/1));
  EXPECT_EQ(hipSuccess, hip_.event_query(event));
}

// This is a contract check, not a discriminator - the gated callback shares a
// queue thread with the gate.
TEST_F(HipEventTest, CrossStreamRerecordStaysUsableAsStreamWaitDependency) {
  hipStream_t stream_a = CreateStream();
  ASSERT_NE(nullptr, stream_a);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, stream_a, /*count=*/3));

  hipStream_t stream_b = CreateStream();
  ASSERT_NE(nullptr, stream_b);
  hipStream_t stream_c = CreateStream();
  ASSERT_NE(nullptr, stream_c);

  StreamGate gate;
  StreamMarker marker;
  marker.gate = &gate;
  ScopedHostCallbacks callbacks;
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(stream_b, &gate, &callbacks));

  ASSERT_EQ(hipSuccess, hip_.event_record(event, stream_b));
  ASSERT_EQ(hipSuccess, hip_.stream_wait_event(stream_c, event, /*flags=*/0));
  ASSERT_EQ(hipSuccess,
            hip_.launch_host_func(stream_c, &MarkerHostFunction, &marker));
  callbacks.Add(marker.finished);

  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(stream_c));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(stream_b));

  EXPECT_TRUE(marker.finished.load(std::memory_order_acquire))
      << "work gated on the re-recorded event never ran";
  EXPECT_TRUE(marker.saw_gate_finished.load(std::memory_order_acquire))
      << "work gated on the re-recorded event ran before the stream that "
         "recorded the event drained";
}

// This is a contract check, not a discriminator - the queue also serializes
// signals on one timeline, which covers the same ordering.
TEST_F(HipEventTest, StreamWaitEventStaysOrderedBehindPriorStreamWork) {
  hipStream_t stream_a = CreateStream();
  ASSERT_NE(nullptr, stream_a);
  hipStream_t stream_b = CreateStream();
  ASSERT_NE(nullptr, stream_b);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  // The event is complete before the wait is submitted, so nothing but the
  // stream's own ordering can hold the wait back.
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, stream_a, /*count=*/1));

  StreamGate gate;
  ScopedHostCallbacks callbacks;
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(stream_b, &gate, &callbacks));

  ASSERT_EQ(hipSuccess, hip_.stream_wait_event(stream_b, event, /*flags=*/0));

  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(stream_b));
  EXPECT_TRUE(gate.finished.load(std::memory_order_acquire))
      << "the stream reported itself drained while work submitted before the "
         "event wait was still running";
}

TEST_F(HipEventTest, StreamWaitOnANeverRecordedEventDropsBothWaits) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  ASSERT_EQ(hipSuccess, hip_.stream_wait_event(stream, event, /*flags=*/0));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(stream));

  // The stream has now submitted, so the same wait keeps the stream half and
  // drops only the event half.
  ASSERT_EQ(hipSuccess, hip_.stream_wait_event(stream, event, /*flags=*/0));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(stream));

  // And once the event has a record the wait carries both halves again.
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, stream, /*count=*/1));
  ASSERT_EQ(hipSuccess, hip_.stream_wait_event(stream, event, /*flags=*/0));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(stream));
}

TEST_F(HipEventTest, GraphEventWaitNodeOnANeverRecordedEventDropsTheWait) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  hipGraph_t graph = CreateGraph();
  ASSERT_NE(nullptr, graph);
  hipGraphNode_t wait_node = nullptr;
  ASSERT_EQ(hipSuccess,
            hip_.graph_add_event_wait_node(&wait_node, graph,
                                           /*dependencies=*/nullptr,
                                           /*dependency_count=*/0, event));
  std::atomic<bool> host_node_ran{false};
  hipHostNodeParams host_params = {};
  host_params.fn = &RanHostFunction;
  host_params.userData = &host_node_ran;
  hipGraphNode_t host_node = nullptr;
  ASSERT_EQ(hipSuccess,
            hip_.graph_add_host_node(&host_node, graph, &wait_node,
                                     /*dependency_count=*/1, &host_params));
  hipGraphExec_t graph_exec = InstantiateGraph(graph);
  ASSERT_NE(nullptr, graph_exec);

  ScopedHostCallbacks callbacks;
  ASSERT_EQ(hipSuccess, hip_.graph_launch(graph_exec, stream));
  callbacks.Add(host_node_ran);
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(stream));
  EXPECT_TRUE(host_node_ran.load(std::memory_order_acquire))
      << "the node behind a dropped event wait never ran";
}

TEST_F(HipEventTest, GraphApisRejectDestroyedEventHandles) {
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);
  hipGraph_t graph = CreateGraph();
  ASSERT_NE(nullptr, graph);

  hipGraphNode_t record_node = nullptr;
  ASSERT_EQ(hipSuccess, hip_.graph_add_event_record_node(
                            &record_node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, event));
  hipGraphNode_t wait_node = nullptr;
  ASSERT_EQ(hipSuccess, hip_.graph_add_event_wait_node(
                            &wait_node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, event));
  hipGraphExec_t graph_exec = InstantiateGraph(graph);
  ASSERT_NE(nullptr, graph_exec);

  // Graph ownership keeps the event storage alive after public destruction.
  // The stale public handle must still be rejected by every graph entry point.
  DestroyEvent(event);

  hipGraphNode_t added_node = nullptr;
  EXPECT_EQ(hipErrorInvalidResourceHandle,
            hip_.graph_add_event_record_node(&added_node, graph,
                                             /*dependencies=*/nullptr,
                                             /*dependency_count=*/0, event));
  EXPECT_EQ(nullptr, added_node);
  EXPECT_EQ(hipErrorInvalidResourceHandle,
            hip_.graph_add_event_wait_node(&added_node, graph,
                                           /*dependencies=*/nullptr,
                                           /*dependency_count=*/0, event));
  EXPECT_EQ(nullptr, added_node);
  EXPECT_EQ(hipErrorInvalidResourceHandle,
            hip_.graph_event_record_node_set_event(record_node, event));
  EXPECT_EQ(hipErrorInvalidResourceHandle,
            hip_.graph_event_wait_node_set_event(wait_node, event));
  EXPECT_EQ(hipErrorInvalidResourceHandle,
            hip_.graph_exec_event_record_node_set_event(graph_exec, record_node,
                                                        event));
  EXPECT_EQ(
      hipErrorInvalidResourceHandle,
      hip_.graph_exec_event_wait_node_set_event(graph_exec, wait_node, event));
}

TEST_F(HipEventTest, GraphRecordNodeAtTheEndOfAGraphMarksTheLaunchPoint) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  // Stream records and graph records have to interleave on one event.
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, stream, /*count=*/2));

  hipGraph_t graph = CreateGraph();
  ASSERT_NE(nullptr, graph);
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess, hip_.graph_add_event_record_node(
                            &node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, event));
  hipGraphExec_t graph_exec = InstantiateGraph(graph);
  ASSERT_NE(nullptr, graph_exec);

  StreamGate gate;
  ScopedHostCallbacks callbacks;
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(stream, &gate, &callbacks));

  ASSERT_EQ(hipSuccess, hip_.graph_launch(graph_exec, stream));
  EXPECT_EQ(hipErrorNotReady, hip_.event_query(event))
      << "event completed while the gate callback holding the stream the graph "
         "launched onto is still running";

  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, hip_.event_synchronize(event));
  EXPECT_EQ(hipSuccess, hip_.event_query(event));

  // Repeated launches keep naming a fresh point, and stream records keep
  // working on an event a graph has been recording.
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(hipSuccess, hip_.graph_launch(graph_exec, stream));
    EXPECT_EQ(hipSuccess, hip_.event_synchronize(event));
    EXPECT_EQ(hipSuccess, hip_.event_query(event));
  }
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(stream));
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, stream, /*count=*/1));
}

TEST_F(HipEventTest, GraphRecordNodeInsideAGraphMarksTheNodePoint) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  hipGraph_t graph = CreateGraph();
  ASSERT_NE(nullptr, graph);
  hipGraphNode_t record_node = nullptr;
  ASSERT_EQ(hipSuccess, hip_.graph_add_event_record_node(
                            &record_node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, event));

  StreamGate gate;
  StreamMarker marker;
  marker.gate = &gate;
  hipHostNodeParams host_params = {};
  host_params.fn = &MarkerHostFunction;
  host_params.userData = &marker;
  hipGraphNode_t host_node = nullptr;
  ASSERT_EQ(hipSuccess,
            hip_.graph_add_host_node(&host_node, graph, &record_node,
                                     /*dependency_count=*/1, &host_params));

  hipGraphExec_t graph_exec = InstantiateGraph(graph);
  ASSERT_NE(nullptr, graph_exec);

  ScopedHostCallbacks callbacks;
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(stream, &gate, &callbacks));

  ASSERT_EQ(hipSuccess, hip_.graph_launch(graph_exec, stream));
  callbacks.Add(marker.finished);
  EXPECT_EQ(hipErrorNotReady, hip_.event_query(event))
      << "event completed while the gate callback holding the stream the graph "
         "launched onto is still running";

  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, hip_.event_synchronize(event));
  EXPECT_EQ(hipSuccess, hip_.event_query(event));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(stream));
  EXPECT_TRUE(marker.saw_gate_finished.load(std::memory_order_acquire))
      << "the node after the record node ran before the work in front of the "
         "launch drained";
}

TEST_F(HipEventTest, ElapsedTimeNeedsBothEventsRecorded) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t start = CreateEvent();
  ASSERT_NE(nullptr, start);
  hipEvent_t stop = CreateEvent();
  ASSERT_NE(nullptr, stop);

  float ms = -1.0f;
  EXPECT_EQ(hipErrorInvalidHandle, hip_.event_elapsed_time(&ms, start, stop));

  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(start, stream, /*count=*/1));
  EXPECT_EQ(hipErrorInvalidHandle, hip_.event_elapsed_time(&ms, start, stop))
      << "an interval was reported for an event that was never recorded";

  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(stop, stream, /*count=*/1));
  ms = -1.0f;
  EXPECT_EQ(hipSuccess, hip_.event_elapsed_time(&ms, start, stop));
  EXPECT_GE(ms, 0.0f);
}

TEST_F(HipEventTest, ElapsedTimeIsNotReadyWhileAnEventIsOutstanding) {
  hipStream_t stream_a = CreateStream();
  ASSERT_NE(nullptr, stream_a);
  hipStream_t stream_b = CreateStream();
  ASSERT_NE(nullptr, stream_b);
  hipEvent_t start = CreateEvent();
  ASSERT_NE(nullptr, start);
  hipEvent_t stop = CreateEvent();
  ASSERT_NE(nullptr, stop);

  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(start, stream_a, /*count=*/1));

  StreamGate gate;
  ScopedHostCallbacks callbacks;
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(stream_b, &gate, &callbacks));
  ASSERT_EQ(hipSuccess, hip_.event_record(stop, stream_b));

  float ms = -1.0f;
  EXPECT_EQ(hipErrorNotReady, hip_.event_elapsed_time(&ms, start, stop));

  callbacks.ReleaseGate();
  ASSERT_EQ(hipSuccess, hip_.event_synchronize(stop));
  ms = -1.0f;
  EXPECT_EQ(hipSuccess, hip_.event_elapsed_time(&ms, start, stop));
  EXPECT_GE(ms, 0.0f);
}

TEST_F(HipEventTest, ElapsedTimeRejectsEventsRecordedOnlyDuringCapture) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t start = CreateEvent();
  ASSERT_NE(nullptr, start);
  hipEvent_t stop = CreateEvent();
  ASSERT_NE(nullptr, stop);

  ASSERT_EQ(hipSuccess,
            hip_.stream_begin_capture(stream, hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess, hip_.event_record(start, stream));
  ASSERT_EQ(hipSuccess, hip_.event_record(stop, stream));
  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, hip_.stream_end_capture(stream, &graph));
  TrackGraph(graph);
  ASSERT_NE(nullptr, graph);

  float ms = -1.0f;
  EXPECT_EQ(hipErrorCapturedEvent, hip_.event_elapsed_time(&ms, start, stop))
      << "an interval was reported for records that were never submitted";
}

// An event recorded directly and then captured carries a submitted point that
// measures, but that point belongs to the earlier record and not to what the
// caller asked about when it captured the event. The pair is refused for the
// same reason a query of the event is.
TEST_F(HipEventTest, ElapsedTimeRejectsAnEventLastRecordedIntoACapture) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipStream_t capture_stream = CreateStream();
  ASSERT_NE(nullptr, capture_stream);
  hipEvent_t start = CreateEvent();
  ASSERT_NE(nullptr, start);
  hipEvent_t stop = CreateEvent();
  ASSERT_NE(nullptr, stop);

  // Both events are left carrying a submitted record that has been reached, so
  // the pair measures until one of them is captured.
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(start, stream, /*count=*/1));
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(stop, stream, /*count=*/1));
  float ms = -1.0f;
  ASSERT_EQ(hipSuccess, hip_.event_elapsed_time(&ms, start, stop));
  ASSERT_GE(ms, 0.0f);

  ASSERT_EQ(hipSuccess, hip_.stream_begin_capture(capture_stream,
                                                  hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess, hip_.event_record(stop, capture_stream));
  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, hip_.stream_end_capture(capture_stream, &graph));
  TrackGraph(graph);
  ASSERT_NE(nullptr, graph);

  ms = -1.0f;
  EXPECT_EQ(hipErrorCapturedEvent, hip_.event_elapsed_time(&ms, start, stop))
      << "the interval of a record the caller did not ask about was reported";
  EXPECT_FLOAT_EQ(-1.0f, ms) << "a refused pair reported a duration";
}

// Refusing a pair also invalidates the capture the refused record went into,
// which is what the two neighbouring event entry points do for the same state.
// Only an active capture can be marked: ending one clears the association from
// its stream, so a measurement taken afterwards finds no stream to mark. The
// capture is therefore left running here, and hipStreamEndCapture refusing to
// hand back a graph is the evidence that measuring invalidated it.
TEST_F(HipEventTest, ElapsedTimeInvalidatesTheCaptureItRefuses) {
  hipStream_t capture_stream = CreateStream();
  ASSERT_NE(nullptr, capture_stream);
  hipEvent_t start = CreateEvent();
  ASSERT_NE(nullptr, start);
  hipEvent_t stop = CreateEvent();
  ASSERT_NE(nullptr, stop);

  ASSERT_EQ(hipSuccess, hip_.stream_begin_capture(capture_stream,
                                                  hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess, hip_.event_record(start, capture_stream));
  ASSERT_EQ(hipSuccess, hip_.event_record(stop, capture_stream));

  float ms = -1.0f;
  EXPECT_EQ(hipErrorCapturedEvent, hip_.event_elapsed_time(&ms, start, stop))
      << "an interval was reported for records that went into a capture";

  hipGraph_t graph = nullptr;
  const hipError_t end_result = hip_.stream_end_capture(capture_stream, &graph);
  TrackGraph(graph);
  EXPECT_EQ(hipErrorStreamCaptureInvalidated, end_result)
      << "the capture the measurement refused was left able to complete";
}

// A record submitted after a capture ends the event's association with the
// captured graph in the same transition that installs its point, so every
// entry point that refuses a captured event stops refusing without the caller
// doing anything else.
TEST_F(HipEventTest, ASubmittedRecordEndsTheCaptureAssociation) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipStream_t capture_stream = CreateStream();
  ASSERT_NE(nullptr, capture_stream);
  hipEvent_t start = CreateEvent();
  ASSERT_NE(nullptr, start);
  hipEvent_t stop = CreateEvent();
  ASSERT_NE(nullptr, stop);

  ASSERT_EQ(hipSuccess, hip_.stream_begin_capture(capture_stream,
                                                  hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess, hip_.event_record(start, capture_stream));
  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, hip_.stream_end_capture(capture_stream, &graph));
  TrackGraph(graph);
  ASSERT_NE(nullptr, graph);

  // The capture has ended and the event still belongs to the graph it went
  // into, which is the state the record below has to clear.
  ASSERT_EQ(hipErrorCapturedEvent, hip_.event_query(start));

  ASSERT_EQ(hipSuccess, hip_.event_record(start, stream));
  ASSERT_EQ(hipSuccess, hip_.event_record(stop, stream));
  ASSERT_EQ(hipSuccess, hip_.stream_synchronize(stream));

  EXPECT_EQ(hipSuccess, hip_.event_query(start));
  EXPECT_EQ(hipSuccess, hip_.event_synchronize(start));
  float ms = -1.0f;
  EXPECT_EQ(hipSuccess, hip_.event_elapsed_time(&ms, start, stop));
  EXPECT_GE(ms, 0.0f);
}

// The last reference to a captured graph is dropped after a replaying launch
// has released its locks. Freeing a graph's host allocations synchronizes every
// context, which relocks the stream the launch submits on, so a launch still
// holding that lock when it dropped the reference would hang. The capture here
// owns a host allocation, its graph handle is destroyed before the replay, and
// the replay's records hold the only references left, so the launch is what
// destroys it and the evidence is that the replay completes.
//
// The launch parks the references it drops in chunked storage that starts in
// its own frame and grows onto the heap, and seventeen events exceed the
// sixteen one chunk holds, so the walk grows it. What this pins about the
// growth is that the launch still completes and every event comes back
// unassociated; a reference stranded in a grown chunk is a leak, which none of
// these observables can see, and the drain across the growth is pinned on the
// host by graph_exec_test.cc instead. The capacity is private to the launch
// and nothing here can read it, so a change to it has to be matched by a
// change here.
TEST_F(HipEventTest, GraphReplayDropsTheLastCaptureReferenceAfterUnlocking) {
  static constexpr int kEventCount = 17;
  static constexpr size_t kCapturedCopySize = 256;
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipStream_t capture_stream = CreateStream();
  ASSERT_NE(nullptr, capture_stream);
  std::vector<hipEvent_t> events;
  for (int i = 0; i < kEventCount; ++i) {
    hipEvent_t event = CreateEvent();
    ASSERT_NE(nullptr, event);
    events.push_back(event);
  }

  void* device_ptr = AllocateAsync(stream, kCapturedCopySize);
  ASSERT_NE(nullptr, device_ptr);
  ASSERT_EQ(hipSuccess, hip_.stream_synchronize(stream));

  // The captured copy is what gives the graph a host allocation of its own,
  // and every event is associated with that graph.
  std::vector<uint8_t> host_source(kCapturedCopySize, 0x5a);
  ASSERT_EQ(hipSuccess, hip_.stream_begin_capture(capture_stream,
                                                  hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess,
            hip_.memcpy_async(device_ptr, host_source.data(), kCapturedCopySize,
                              hipMemcpyHostToDevice, capture_stream));
  for (hipEvent_t event : events) {
    ASSERT_EQ(hipSuccess, hip_.event_record(event, capture_stream));
  }
  hipGraph_t captured_graph = nullptr;
  ASSERT_EQ(hipSuccess,
            hip_.stream_end_capture(capture_stream, &captured_graph));
  ASSERT_NE(nullptr, captured_graph);

  // Leaves the events holding the only references to the captured graph.
  ASSERT_EQ(hipSuccess, hip_.graph_destroy(captured_graph));

  hipGraph_t replay_graph = CreateGraph();
  ASSERT_NE(nullptr, replay_graph);
  hipGraphNode_t tail = nullptr;
  for (hipEvent_t event : events) {
    hipGraphNode_t node = nullptr;
    ASSERT_EQ(hipSuccess, hip_.graph_add_event_record_node(
                              &node, replay_graph, tail ? &tail : nullptr,
                              tail ? 1 : 0, event));
    tail = node;
  }
  hipGraphExec_t replay_exec = InstantiateGraph(replay_graph);
  ASSERT_NE(nullptr, replay_exec);

  ASSERT_EQ(hipSuccess, hip_.graph_launch(replay_exec, stream));
  ASSERT_EQ(hipSuccess, hip_.stream_synchronize(stream));

  for (hipEvent_t event : events) {
    EXPECT_EQ(hipSuccess, hip_.event_query(event))
        << "a replayed record left its event associated with a capture";
  }

  ASSERT_EQ(hipSuccess, hip_.free_async(device_ptr, stream));
  ASSERT_EQ(hipSuccess, hip_.stream_synchronize(stream));
}

// A wait on an event whose last record was submitted waits on that record's
// point. It does not join the capture an earlier record of the same event went
// into, so the waiting stream is not left capturing into a graph the caller
// never asked it to build. No gate parks the queue here: a parked queue cannot
// tell a stream that waited on a point from a stream blocked behind the gate,
// so the observable is the waiting stream's capture status.
TEST_F(HipEventTest, StreamWaitAfterARecordDoesNotJoinTheOldCapture) {
  hipStream_t capture_stream = CreateStream();
  ASSERT_NE(nullptr, capture_stream);
  hipStream_t recording_stream = CreateStream();
  ASSERT_NE(nullptr, recording_stream);
  hipStream_t waiting_stream = CreateStream();
  ASSERT_NE(nullptr, waiting_stream);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  ASSERT_EQ(hipSuccess, hip_.stream_begin_capture(capture_stream,
                                                  hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess, hip_.event_record(event, capture_stream));
  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, hip_.stream_end_capture(capture_stream, &graph));
  TrackGraph(graph);
  ASSERT_NE(nullptr, graph);

  // The submitted record replaces the capture association, and synchronizing
  // leaves nothing outstanding so the wait below has only the association to
  // decide on.
  ASSERT_EQ(hipSuccess, hip_.event_record(event, recording_stream));
  ASSERT_EQ(hipSuccess, hip_.stream_synchronize(recording_stream));

  EXPECT_EQ(hipSuccess,
            hip_.stream_wait_event(waiting_stream, event, /*flags=*/0));
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusActive;
  EXPECT_EQ(hipSuccess,
            hip_.stream_is_capturing(waiting_stream, &capture_status));
  EXPECT_EQ(hipStreamCaptureStatusNone, capture_status)
      << "the waiting stream adopted a capture the event no longer belongs to";
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(waiting_stream))
      << "the waiting stream was left capturing, which makes synchronizing it "
         "unsupported";
}

TEST_F(HipEventTest, ElapsedTimeNeedsTimingEnabledOnBothEvents) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t timed_start = CreateEvent();
  ASSERT_NE(nullptr, timed_start);
  hipEvent_t timed_stop = CreateEvent();
  ASSERT_NE(nullptr, timed_stop);
  hipEvent_t untimed_start = CreateEventWithFlags(hipEventDisableTiming);
  ASSERT_NE(nullptr, untimed_start);
  hipEvent_t untimed_stop = CreateEventWithFlags(hipEventDisableTiming);
  ASSERT_NE(nullptr, untimed_stop);

  // Every event is left with a submitted record that has been reached, and the
  // pairs below are drawn from that one ordered run, so the timing flag is the
  // only thing separating a measured interval from a rejected one.
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(timed_start, stream,
                                               /*count=*/1));
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(untimed_start, stream,
                                               /*count=*/1));
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(untimed_stop, stream,
                                               /*count=*/1));
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(timed_stop, stream,
                                               /*count=*/1));

  float ms = -1.0f;
  EXPECT_EQ(hipSuccess, hip_.event_elapsed_time(&ms, timed_start, timed_stop));
  EXPECT_GE(ms, 0.0f);

  EXPECT_EQ(hipErrorInvalidHandle,
            hip_.event_elapsed_time(&ms, untimed_start, untimed_stop))
      << "an interval was reported for two events with timing disabled";
  EXPECT_EQ(hipErrorInvalidHandle,
            hip_.event_elapsed_time(&ms, untimed_start, timed_stop))
      << "an interval was reported for a start with timing disabled";
  EXPECT_EQ(hipErrorInvalidHandle,
            hip_.event_elapsed_time(&ms, timed_start, untimed_stop))
      << "an interval was reported for a stop with timing disabled";
}

// The interval between two records is the device work between them, not the
// time the host spent issuing them. The gate sits between the two records on
// one stream, so the start record runs before the queue parks and the stop
// record after it is released, and nothing has to make progress on a second
// stream while the queue is held.
TEST_F(HipEventTest, ElapsedTimeMeasuresTheDeviceIntervalBetweenDirectRecords) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t start = CreateEvent();
  ASSERT_NE(nullptr, start);
  hipEvent_t stop = CreateEvent();
  ASSERT_NE(nullptr, stop);

  StreamGate gate;
  ScopedHostCallbacks callbacks;
  // Only the two record calls are timed, so the number below is the quantity
  // this test separates the reported interval from.
  const auto first_record_from = std::chrono::steady_clock::now();
  ASSERT_EQ(hipSuccess, hip_.event_record(start, stream));
  double issue_ms = MillisecondsSince(first_record_from);
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(stream, &gate, &callbacks));
  const auto second_record_from = std::chrono::steady_clock::now();
  ASSERT_EQ(hipSuccess, hip_.event_record(stop, stream));
  issue_ms += MillisecondsSince(second_record_from);

  const double held_ms = HoldGateAndRelease(&callbacks);
  ASSERT_EQ(hipSuccess, hip_.event_synchronize(stop));

  float reported_ms = -1.0f;
  ASSERT_EQ(hipSuccess, hip_.event_elapsed_time(&reported_ms, start, stop));
  ASSERT_NO_FATAL_FAILURE(
      ExpectMeasuredTheHold(reported_ms, held_ms, "direct records"));
  EXPECT_LT(issue_ms, reported_ms * kMinimumHoldFraction)
      << "issuing the two records took " << issue_ms << " ms against the "
      << reported_ms
      << " ms reported, so this test cannot separate the two quantities";
}

// The same property through hipGraphLaunch, which reaches the queue by a
// different path than a direct record; the gate is the graph's own host node.
TEST_F(HipEventTest, ElapsedTimeMeasuresTheDeviceIntervalBetweenGraphRecords) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t start = CreateEvent();
  ASSERT_NE(nullptr, start);
  hipEvent_t stop = CreateEvent();
  ASSERT_NE(nullptr, stop);

  StreamGate gate;
  ScopedHostCallbacks callbacks;
  hipGraphExec_t graph_exec = InstantiateGraphRecordingAroundAHostNode(
      start, stop, &GateHostFunction, &gate);
  ASSERT_NE(nullptr, graph_exec);

  const auto launch_from = std::chrono::steady_clock::now();
  ASSERT_EQ(hipSuccess, hip_.graph_launch(graph_exec, stream));
  const double launch_ms = MillisecondsSince(launch_from);
  ASSERT_NO_FATAL_FAILURE(callbacks.AddGate(&gate));
  while (!gate.entered.load(std::memory_order_acquire)) {
    sched_yield();
  }

  const double held_ms = HoldGateAndRelease(&callbacks);
  ASSERT_EQ(hipSuccess, hip_.event_synchronize(stop));

  float reported_ms = -1.0f;
  ASSERT_EQ(hipSuccess, hip_.event_elapsed_time(&reported_ms, start, stop));
  ASSERT_NO_FATAL_FAILURE(
      ExpectMeasuredTheHold(reported_ms, held_ms, "graph-replayed records"));
  EXPECT_LT(launch_ms, reported_ms * kMinimumHoldFraction)
      << "the launch took " << launch_ms << " ms against the " << reported_ms
      << " ms reported, so this test cannot separate the two quantities";
}

// Every replay of one executable has to capture new ticks. Three replays of one
// graph over one event pair, holding long, short and long, report three
// intervals on the same clock, so comparing them to each other cancels the tick
// rate. A long replay either side of the short one is what separates re-timing
// from a second replay happening to read something plausible: a tick written
// once, or a slot recycled between a record and the read of it, collapses the
// ratio.
TEST_F(HipEventTest, GraphReplayRetimesTheEventsOnEveryLaunch) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t start = CreateEvent();
  ASSERT_NE(nullptr, start);
  hipEvent_t stop = CreateEvent();
  ASSERT_NE(nullptr, stop);

  ReplayGate gate;
  ScopedReplayGate replays(&gate);
  hipGraphExec_t graph_exec = InstantiateGraphRecordingAroundAHostNode(
      start, stop, &ReplayGateHostFunction, &gate);
  ASSERT_NE(nullptr, graph_exec);

  float first_long_ms = -1.0f;
  double first_long_held_ms = 0.0;
  ASSERT_NO_FATAL_FAILURE(ReplayHoldingTheGateAndMeasure(
      graph_exec, stream, &replays, kGateHoldMs, start, stop, &first_long_ms,
      &first_long_held_ms));
  float short_ms = -1.0f;
  double short_held_ms = 0.0;
  ASSERT_NO_FATAL_FAILURE(ReplayHoldingTheGateAndMeasure(
      graph_exec, stream, &replays, kShortGateHoldMs, start, stop, &short_ms,
      &short_held_ms));
  float second_long_ms = -1.0f;
  double second_long_held_ms = 0.0;
  ASSERT_NO_FATAL_FAILURE(ReplayHoldingTheGateAndMeasure(
      graph_exec, stream, &replays, kGateHoldMs, start, stop, &second_long_ms,
      &second_long_held_ms));

  ASSERT_NO_FATAL_FAILURE(ExpectMeasuredTheHold(
      first_long_ms, first_long_held_ms, "first long replay"));
  ASSERT_NO_FATAL_FAILURE(ExpectMeasuredTheHold(
      second_long_ms, second_long_held_ms, "second long replay"));
  EXPECT_GT(first_long_ms, short_ms * kMinimumReplayRatio)
      << "the first long replay reported " << first_long_ms
      << " ms against the short replay's " << short_ms
      << " ms for a tenth of the hold, so the replay did not re-time the "
         "events";
  EXPECT_GT(second_long_ms, short_ms * kMinimumReplayRatio)
      << "the second long replay reported " << second_long_ms
      << " ms, close to the preceding short replay's " << short_ms
      << " ms, so the replay reported the ticks of an earlier one";
}

// A submitted record draws its tick slot from the recording stream's context
// pool and hands it to a point the event outlives. Nothing the point names
// keeps that pool alive - a slot reference is a count on the slot alone - so
// only the reference the event holds on its own context does. A record on a
// stream of any other context is refused before a slot is acquired, on the
// direct path and inside a graph launch alike, so an event can never name
// storage a context it does not hold has freed.
//
// A launch settles that refusal before it submits any of the graph, so the
// node ahead of the record never runs either: a launch that had submitted part
// of the graph would report failure and still leave work in flight on a stream
// whose timeline it never advanced.
//
// A second context on the same device is what makes this reachable on a
// machine with one GPU, where a skip would report green for coverage that
// never ran.
TEST_F(HipEventTest, RecordOnAnotherContextsStreamIsRefused) {
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);
  hipStream_t own_stream = CreateStream();
  ASSERT_NE(nullptr, own_stream);

  // The host node depends on nothing and the record depends on it, so the host
  // call is the block a launch submits first and the record the block that
  // refuses it. Whether the host node ran is how the test sees what a refused
  // launch had already submitted.
  hipGraph_t graph = CreateGraph();
  ASSERT_NE(nullptr, graph);
  hipHostNodeParams host_params = {};
  host_params.fn = &RanHostFunction;
  host_params.userData = &graph_host_node_ran_;
  hipGraphNode_t host_node = nullptr;
  ASSERT_EQ(hipSuccess,
            hip_.graph_add_host_node(&host_node, graph,
                                     /*dependencies=*/nullptr,
                                     /*dependency_count=*/0, &host_params));
  hipGraphNode_t record_node = nullptr;
  ASSERT_EQ(hipSuccess,
            hip_.graph_add_event_record_node(&record_node, graph, &host_node,
                                             /*dependency_count=*/1, event));
  hipGraphExec_t graph_exec = InstantiateGraph(graph);
  ASSERT_NE(nullptr, graph_exec);

  hipCtx_t original_context = nullptr;
  ASSERT_EQ(hipSuccess, hip_.ctx_get_current(&original_context));
  ASSERT_NE(nullptr, original_context)
      << "the fixture's handles were created without a current context";

  // Creating a context makes it current, so the stream below is created in it
  // and everything after the restore runs in the original one again.
  hipCtx_t other_context = nullptr;
  ASSERT_EQ(hipSuccess,
            hip_.ctx_create(&other_context, /*flags=*/0, /*device=*/0));
  ASSERT_NE(nullptr, other_context);
  hipStream_t other_stream = nullptr;
  const hipError_t other_stream_result = hip_.stream_create(&other_stream);
  ASSERT_EQ(hipSuccess, hip_.ctx_set_current(original_context));
  ASSERT_EQ(hipSuccess, other_stream_result);
  ASSERT_NE(nullptr, other_stream);

  EXPECT_EQ(hipErrorInvalidHandle, hip_.event_record(event, other_stream))
      << "a record on a stream of another context was accepted, leaving the "
         "event holding a tick slot from a pool that context owns";
  // hipGraphLaunch maps every failed launch to one code, so this pins that the
  // launch failed and nothing about why; the refusal's own message reaches
  // stderr on the way through that mapping.
  EXPECT_EQ(hipErrorInvalidValue, hip_.graph_launch(graph_exec, other_stream))
      << "a replayed record on a stream of another context was accepted";

  // Nothing has been submitted on that stream, so a block the launch enqueued
  // would have waited on nothing and been runnable the moment the queue took
  // it. Draining the stream behind a callback enqueued after the refused launch
  // is what gives such a block its chance to run before the read below. It is
  // not an ordering proof: HAL queues are not FIFO, user-visible order comes
  // from semaphore edges, and these two submissions share none - a refused
  // launch leaves the stream tail where it was, so the callback drops its wait
  // the same way the block would have.
  std::atomic<bool> marker_ran{false};
  ScopedHostCallbacks callbacks;
  ASSERT_EQ(hipSuccess,
            hip_.launch_host_func(other_stream, &RanHostFunction, &marker_ran));
  callbacks.Add(marker_ran);
  ASSERT_EQ(hipSuccess, hip_.stream_synchronize(other_stream));
  ASSERT_TRUE(marker_ran.load(std::memory_order_acquire))
      << "the stream was synchronized without running the callback behind it, "
         "so nothing here says when the launch's own blocks would have run";
  EXPECT_FALSE(graph_host_node_ran_.load(std::memory_order_acquire))
      << "the node ahead of the refused record ran, so the launch submitted "
         "part of the graph and then failed";

  // A refused record commits no point, so the event still carries none and
  // there is no interval between it and a record that was accepted.
  hipEvent_t accepted = CreateEvent();
  ASSERT_NE(nullptr, accepted);
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(accepted, own_stream,
                                               /*count=*/1));
  float ms = -1.0f;
  EXPECT_EQ(hipErrorInvalidHandle,
            hip_.event_elapsed_time(&ms, accepted, event))
      << "an interval was reported for an event whose only records were "
         "refused";

  // A record on a stream of the event's own context is still accepted, so the
  // refusals above are not a blanket one.
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, own_stream, /*count=*/1));
  ms = -1.0f;
  EXPECT_EQ(hipSuccess, hip_.event_elapsed_time(&ms, accepted, event));
  EXPECT_GE(ms, 0.0f);

  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(other_stream));
  EXPECT_EQ(hipSuccess, hip_.stream_destroy(other_stream));
  EXPECT_EQ(hipSuccess, hip_.ctx_destroy(other_context));
}

// Splicing a child graph into an instantiated executable instantiates that
// graph in the context it belongs to and folds the result into the executable,
// so it takes a child graph only from the executable's own context - the rule
// the template's child graph node builder already enforces. A child of another
// context would seat that context's event records in an executable a launch
// answers for by comparing one pair of contexts, and the launch would then
// break on the first of those records with the blocks ahead of it submitted.
TEST_F(HipEventTest, ExecChildGraphNodeTakesOnlyItsOwnContextsGraph) {
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  // A splice keeps the node count, so every child graph here holds one record
  // node, which is also what makes the executable one that records events.
  hipGraph_t child_graph = CreateGraph();
  ASSERT_NE(nullptr, child_graph);
  hipGraphNode_t child_record_node = nullptr;
  ASSERT_EQ(hipSuccess,
            hip_.graph_add_event_record_node(&child_record_node, child_graph,
                                             /*dependencies=*/nullptr,
                                             /*dependency_count=*/0, event));
  hipGraph_t parent_graph = CreateGraph();
  ASSERT_NE(nullptr, parent_graph);
  hipGraphNode_t child_node = nullptr;
  ASSERT_EQ(hipSuccess, hip_.graph_add_child_graph_node(
                            &child_node, parent_graph,
                            /*dependencies=*/nullptr,
                            /*dependency_count=*/0, child_graph));
  hipGraphExec_t graph_exec = InstantiateGraph(parent_graph);
  ASSERT_NE(nullptr, graph_exec);

  hipCtx_t original_context = nullptr;
  ASSERT_EQ(hipSuccess, hip_.ctx_get_current(&original_context));
  ASSERT_NE(nullptr, original_context)
      << "the fixture's handles were created without a current context";

  // Creating a context makes it current, so the event and graph below are
  // created in it and everything after the restore runs in the original one
  // again. Both belong to the same context, which is what lets the record node
  // be added at all.
  hipCtx_t other_context = nullptr;
  ASSERT_EQ(hipSuccess,
            hip_.ctx_create(&other_context, /*flags=*/0, /*device=*/0));
  ASSERT_NE(nullptr, other_context);
  hipEvent_t other_event = nullptr;
  hipGraph_t other_child_graph = nullptr;
  hipGraphNode_t other_record_node = nullptr;
  hipError_t other_result = hip_.event_create(&other_event);
  if (other_result == hipSuccess) {
    other_result = hip_.graph_create(&other_child_graph, /*flags=*/0);
  }
  if (other_result == hipSuccess) {
    other_result = hip_.graph_add_event_record_node(
        &other_record_node, other_child_graph, /*dependencies=*/nullptr,
        /*dependency_count=*/0, other_event);
  }
  ASSERT_EQ(hipSuccess, hip_.ctx_set_current(original_context));
  ASSERT_EQ(hipSuccess, other_result);

  EXPECT_EQ(hipErrorInvalidValue,
            hip_.graph_exec_child_graph_node_set_params(graph_exec, child_node,
                                                        other_child_graph))
      << "an executable took a child graph from another context, whose record "
         "nodes a launch on a stream of this context would refuse one block "
         "into the walk";

  // A child graph of the executable's own context is still taken, so the
  // refusal is the context rule and not a blanket one.
  hipGraph_t replacement_child_graph = CreateGraph();
  ASSERT_NE(nullptr, replacement_child_graph);
  hipGraphNode_t replacement_record_node = nullptr;
  ASSERT_EQ(hipSuccess, hip_.graph_add_event_record_node(
                            &replacement_record_node, replacement_child_graph,
                            /*dependencies=*/nullptr,
                            /*dependency_count=*/0, event));
  EXPECT_EQ(hipSuccess, hip_.graph_exec_child_graph_node_set_params(
                            graph_exec, child_node, replacement_child_graph));

  EXPECT_EQ(hipSuccess, hip_.graph_destroy(other_child_graph));
  EXPECT_EQ(hipSuccess, hip_.event_destroy(other_event));
  EXPECT_EQ(hipSuccess, hip_.ctx_destroy(other_context));
}

// Instantiating a child graph instantiates the child graph nodes it holds in
// turn, so a splice takes only a graph that does not reach the executable's
// template: one that did would make the rebuild follow the containment back
// into itself until the stack ran out. This is the rest of the rule the
// template's child graph node builder already enforces, and the builder takes
// the container built here because containment closes into a cycle only once
// the template's own node is retargeted at it.
TEST_F(HipEventTest, ExecChildGraphNodeRefusesAGraphContainingItsTemplate) {
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  // Every graph here holds exactly one node, so the node counts a splice
  // compares are equal and cannot be what refuses anything below.
  hipGraph_t child_graph = CreateGraph();
  ASSERT_NE(nullptr, child_graph);
  hipGraphNode_t child_record_node = nullptr;
  ASSERT_EQ(hipSuccess,
            hip_.graph_add_event_record_node(&child_record_node, child_graph,
                                             /*dependencies=*/nullptr,
                                             /*dependency_count=*/0, event));
  hipGraph_t template_graph = CreateGraph();
  ASSERT_NE(nullptr, template_graph);
  hipGraphNode_t child_node = nullptr;
  ASSERT_EQ(hipSuccess, hip_.graph_add_child_graph_node(
                            &child_node, template_graph,
                            /*dependencies=*/nullptr,
                            /*dependency_count=*/0, child_graph));
  hipGraphExec_t graph_exec = InstantiateGraph(template_graph);
  ASSERT_NE(nullptr, graph_exec);

  hipGraph_t container_graph = CreateGraph();
  ASSERT_NE(nullptr, container_graph);
  hipGraphNode_t container_child_node = nullptr;
  ASSERT_EQ(hipSuccess, hip_.graph_add_child_graph_node(
                            &container_child_node, container_graph,
                            /*dependencies=*/nullptr,
                            /*dependency_count=*/0, template_graph));

  EXPECT_EQ(hipErrorInvalidValue, hip_.graph_exec_child_graph_node_set_params(
                                      graph_exec, child_node, container_graph))
      << "an executable took a child graph that holds its own template, whose "
         "rebuild instantiates the template again through that child";
  // Both the identity check and the containment walk refuse this: the walk
  // seeds with the offered graph and matches the parent on its first step. The
  // assertion pins the entry point's answer, not the identity check on its
  // own. That check carries the template's child graph node builder, where a
  // graph with no child graph node added to itself would otherwise take the
  // containment early return and succeed.
  EXPECT_EQ(hipErrorInvalidValue, hip_.graph_exec_child_graph_node_set_params(
                                      graph_exec, child_node, template_graph))
      << "an executable took its own template as the child graph of a node of "
         "that template";

  // A child graph that reaches nothing is still taken, so the refusals above
  // are the containment rule and not a blanket one.
  hipGraph_t replacement_child_graph = CreateGraph();
  ASSERT_NE(nullptr, replacement_child_graph);
  hipGraphNode_t replacement_record_node = nullptr;
  ASSERT_EQ(hipSuccess, hip_.graph_add_event_record_node(
                            &replacement_record_node, replacement_child_graph,
                            /*dependencies=*/nullptr,
                            /*dependency_count=*/0, event));
  EXPECT_EQ(hipSuccess, hip_.graph_exec_child_graph_node_set_params(
                            graph_exec, child_node, replacement_child_graph));
}

// Destroying a graph unregisters its handle and drops the reference the handle
// carried, so a handle offered here afterwards names storage the last release
// freed. On the path this test builds the splice reads a node count, a context
// and a child graph node count out of that storage; the walk of the node
// blocks runs only for a graph that holds child graph nodes, and the one
// offered here holds a single record node. It takes only a live handle, the
// gate its template counterpart holds.
TEST_F(HipEventTest, ExecChildGraphNodeRefusesADestroyedGraph) {
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  hipGraph_t child_graph = CreateGraph();
  ASSERT_NE(nullptr, child_graph);
  hipGraphNode_t child_record_node = nullptr;
  ASSERT_EQ(hipSuccess,
            hip_.graph_add_event_record_node(&child_record_node, child_graph,
                                             /*dependencies=*/nullptr,
                                             /*dependency_count=*/0, event));
  hipGraph_t template_graph = CreateGraph();
  ASSERT_NE(nullptr, template_graph);
  hipGraphNode_t child_node = nullptr;
  ASSERT_EQ(hipSuccess, hip_.graph_add_child_graph_node(
                            &child_node, template_graph,
                            /*dependencies=*/nullptr,
                            /*dependency_count=*/0, child_graph));
  hipGraphExec_t graph_exec = InstantiateGraph(template_graph);
  ASSERT_NE(nullptr, graph_exec);

  // Holds one node like the graph it would displace, so the node counts a
  // splice compares are equal and cannot be what refuses it. Nothing else
  // holds a reference to it, so destroying it is what frees it.
  hipGraph_t stale_graph = CreateGraph();
  ASSERT_NE(nullptr, stale_graph);
  hipGraphNode_t stale_record_node = nullptr;
  ASSERT_EQ(hipSuccess,
            hip_.graph_add_event_record_node(&stale_record_node, stale_graph,
                                             /*dependencies=*/nullptr,
                                             /*dependency_count=*/0, event));
  DestroyGraph(stale_graph);

  EXPECT_EQ(hipErrorInvalidValue, hip_.graph_exec_child_graph_node_set_params(
                                      graph_exec, child_node, stale_graph))
      << "an executable took a destroyed graph as a child graph, reading its "
         "node count, its context and its child graph node count out of freed "
         "storage";
}

// Bytes taken from the default pool by the reuse tests below. A pending free is
// only reused for a request of exactly its own size, so every one of them uses
// this constant on both sides of the free.
constexpr size_t kReuseAllocationSize = 4096;

// A stream that waits on an event is ordered behind the stream timeline point
// that reaching the event's recorded point implies, and a pending free on that
// stream up to that point may be handed to it. A record inside a graph launch
// implies nothing about the stream that last recorded the event through the
// stream API, so a free queued on that stream stays unavailable.
TEST_F(HipEventTest, GraphInternalRecordDeniesReuseFromTheLastStreamRecorder) {
  hipStream_t producer = CreateStream();
  ASSERT_NE(nullptr, producer);
  hipStream_t launch_stream = CreateStream();
  ASSERT_NE(nullptr, launch_stream);
  hipStream_t consumer = CreateStream();
  ASSERT_NE(nullptr, consumer);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  std::atomic<bool> host_node_ran{false};
  hipGraphExec_t graph_exec =
      InstantiateGraphRecordingBeforeAHostNode(event, &host_node_ran);
  ASSERT_NE(nullptr, graph_exec);

  // Makes the producer the event's last stream recorder while leaving its
  // timeline far behind the launch timeline the record node goes on to name.
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, producer, /*count=*/1));
  for (int i = 0; i < 8; ++i) {
    ASSERT_EQ(hipSuccess, hip_.graph_launch(graph_exec, launch_stream));
  }
  ASSERT_EQ(hipSuccess, hip_.stream_synchronize(launch_stream));
  ASSERT_TRUE(host_node_ran.load(std::memory_order_acquire));

  // The gate parks the producer ahead of the free, so the free's host callback
  // provably has not run and only an event dependency could release its
  // allocation.
  StreamGate gate;
  ScopedHostCallbacks callbacks;
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(producer, &gate, &callbacks));

  void* freed = AllocateAsync(producer, kReuseAllocationSize);
  ASSERT_NE(nullptr, freed);
  ASSERT_EQ(hipSuccess, hip_.free_async(freed, producer));

  ASSERT_EQ(hipSuccess, hip_.stream_wait_event(consumer, event, /*flags=*/0));
  void* reallocated = AllocateAsync(consumer, kReuseAllocationSize);
  ASSERT_NE(nullptr, reallocated);
  EXPECT_NE(freed, reallocated)
      << "an allocation still queued for free on the producer was handed to a "
         "stream that only waited on a record made inside a graph launch";

  EXPECT_EQ(hipSuccess, hip_.free_async(reallocated, consumer));
  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(producer));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(consumer));
}

// The case the reuse policy exists for: the event's last record is a stream
// record on the producer, behind the free, so waiting on it does order the
// consumer after the free.
TEST_F(HipEventTest, StreamRecordBehindAFreeGrantsReuseToTheWaitingStream) {
  hipStream_t producer = CreateStream();
  ASSERT_NE(nullptr, producer);
  hipStream_t launch_stream = CreateStream();
  ASSERT_NE(nullptr, launch_stream);
  hipStream_t consumer = CreateStream();
  ASSERT_NE(nullptr, consumer);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  std::atomic<bool> host_node_ran{false};
  hipGraphExec_t graph_exec =
      InstantiateGraphRecordingBeforeAHostNode(event, &host_node_ran);
  ASSERT_NE(nullptr, graph_exec);

  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, producer, /*count=*/1));
  for (int i = 0; i < 8; ++i) {
    ASSERT_EQ(hipSuccess, hip_.graph_launch(graph_exec, launch_stream));
  }
  ASSERT_EQ(hipSuccess, hip_.stream_synchronize(launch_stream));
  ASSERT_TRUE(host_node_ran.load(std::memory_order_acquire));

  StreamGate gate;
  ScopedHostCallbacks callbacks;
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(producer, &gate, &callbacks));

  void* freed = AllocateAsync(producer, kReuseAllocationSize);
  ASSERT_NE(nullptr, freed);
  ASSERT_EQ(hipSuccess, hip_.free_async(freed, producer));
  ASSERT_EQ(hipSuccess, hip_.event_record(event, producer));

  ASSERT_EQ(hipSuccess, hip_.stream_wait_event(consumer, event, /*flags=*/0));
  void* reallocated = AllocateAsync(consumer, kReuseAllocationSize);
  ASSERT_NE(nullptr, reallocated);
  EXPECT_EQ(freed, reallocated)
      << "a stream ordered behind the free by an event wait was denied the "
         "freed allocation";

  EXPECT_EQ(hipSuccess, hip_.free_async(reallocated, consumer));
  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(producer));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(consumer));
}

// A stream record names the timeline value it reserved and nothing later. A
// free queued on that same stream after the record lands past that value, so
// waiting on the record does not order the waiter behind the free and the
// allocation stays unavailable.
TEST_F(HipEventTest, StreamRecordAheadOfAFreeDeniesReuseToTheWaitingStream) {
  hipStream_t producer = CreateStream();
  ASSERT_NE(nullptr, producer);
  hipStream_t consumer = CreateStream();
  ASSERT_NE(nullptr, consumer);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  StreamGate gate;
  ScopedHostCallbacks callbacks;
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(producer, &gate, &callbacks));

  ASSERT_EQ(hipSuccess, hip_.event_record(event, producer));

  // Queued behind the record, so the free lands on the producer timeline past
  // the value the record named.
  void* freed = AllocateAsync(producer, kReuseAllocationSize);
  ASSERT_NE(nullptr, freed);
  ASSERT_EQ(hipSuccess, hip_.free_async(freed, producer));

  ASSERT_EQ(hipSuccess, hip_.stream_wait_event(consumer, event, /*flags=*/0));
  void* reallocated = AllocateAsync(consumer, kReuseAllocationSize);
  ASSERT_NE(nullptr, reallocated);
  EXPECT_NE(freed, reallocated)
      << "an allocation freed past the value the record named was handed to a "
         "stream that only waited on that record";

  EXPECT_EQ(hipSuccess, hip_.free_async(reallocated, consumer));
  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(producer));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(consumer));
}

// A record that is the launch's last block names a point on the launching
// stream's own timeline, which is a stream point regardless of which stream
// last recorded the event through the stream API.
TEST_F(HipEventTest, GraphRecordEndingALaunchGrantsReuseOfTheLaunchingStream) {
  hipStream_t producer = CreateStream();
  ASSERT_NE(nullptr, producer);
  hipStream_t other_recorder = CreateStream();
  ASSERT_NE(nullptr, other_recorder);
  hipStream_t consumer = CreateStream();
  ASSERT_NE(nullptr, consumer);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  hipGraphExec_t graph_exec = InstantiateGraphRecordingAtTheEnd(event);
  ASSERT_NE(nullptr, graph_exec);

  StreamGate gate;
  ScopedHostCallbacks callbacks;
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(producer, &gate, &callbacks));

  void* freed = AllocateAsync(producer, kReuseAllocationSize);
  ASSERT_NE(nullptr, freed);
  ASSERT_EQ(hipSuccess, hip_.free_async(freed, producer));

  // The stream record leaves a recorder that has nothing to do with the launch
  // that follows it.
  ASSERT_EQ(hipSuccess, hip_.event_record(event, other_recorder));
  ASSERT_EQ(hipSuccess, hip_.graph_launch(graph_exec, producer));

  ASSERT_EQ(hipSuccess, hip_.stream_wait_event(consumer, event, /*flags=*/0));
  void* reallocated = AllocateAsync(consumer, kReuseAllocationSize);
  ASSERT_NE(nullptr, reallocated);
  EXPECT_EQ(freed, reallocated)
      << "a stream ordered behind the free by a graph record on the producer "
         "was denied the freed allocation";

  EXPECT_EQ(hipSuccess, hip_.free_async(reallocated, consumer));
  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(producer));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(other_recorder));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(consumer));
}

// A record internal to a launch names no stream timeline value, but every block
// of the launch is chained behind the tail the launch waited on, so the free
// queued ahead of the launch is still covered.
TEST_F(HipEventTest, GraphInternalRecordGrantsReuseBehindTheLaunchTail) {
  hipStream_t producer = CreateStream();
  ASSERT_NE(nullptr, producer);
  hipStream_t other_recorder = CreateStream();
  ASSERT_NE(nullptr, other_recorder);
  hipStream_t consumer = CreateStream();
  ASSERT_NE(nullptr, consumer);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  std::atomic<bool> host_node_ran{false};
  hipGraphExec_t graph_exec =
      InstantiateGraphRecordingBeforeAHostNode(event, &host_node_ran);
  ASSERT_NE(nullptr, graph_exec);

  StreamGate gate;
  ScopedHostCallbacks callbacks;
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(producer, &gate, &callbacks));

  void* freed = AllocateAsync(producer, kReuseAllocationSize);
  ASSERT_NE(nullptr, freed);
  ASSERT_EQ(hipSuccess, hip_.free_async(freed, producer));

  ASSERT_EQ(hipSuccess, hip_.event_record(event, other_recorder));
  ASSERT_EQ(hipSuccess, hip_.graph_launch(graph_exec, producer));
  callbacks.Add(host_node_ran);

  ASSERT_EQ(hipSuccess, hip_.stream_wait_event(consumer, event, /*flags=*/0));
  void* reallocated = AllocateAsync(consumer, kReuseAllocationSize);
  ASSERT_NE(nullptr, reallocated);
  EXPECT_EQ(freed, reallocated)
      << "a stream ordered behind the free by the tail the launch waited on "
         "was denied the freed allocation";

  EXPECT_EQ(hipSuccess, hip_.free_async(reallocated, consumer));
  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(producer));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(other_recorder));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(consumer));
}

// The tail a launch waited on is a lower bound on the stream ordering a record
// inside the launch implies, never the point itself. A free queued on the
// launching stream after the launch completes past that tail, so waiting on the
// record does not order the waiter behind it and the allocation stays
// unavailable.
TEST_F(HipEventTest, GraphInternalRecordDeniesReuseAheadOfTheLaunchTail) {
  hipStream_t producer = CreateStream();
  ASSERT_NE(nullptr, producer);
  hipStream_t consumer = CreateStream();
  ASSERT_NE(nullptr, consumer);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  std::atomic<bool> host_node_ran{false};
  hipGraphExec_t graph_exec =
      InstantiateGraphRecordingBeforeAHostNode(event, &host_node_ran);
  ASSERT_NE(nullptr, graph_exec);

  StreamGate gate;
  ScopedHostCallbacks callbacks;
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(producer, &gate, &callbacks));

  ASSERT_EQ(hipSuccess, hip_.graph_launch(graph_exec, producer));
  callbacks.Add(host_node_ran);

  // Queued behind the launch, so the free lands on the producer timeline past
  // the tail the launch waited on.
  void* freed = AllocateAsync(producer, kReuseAllocationSize);
  ASSERT_NE(nullptr, freed);
  ASSERT_EQ(hipSuccess, hip_.free_async(freed, producer));

  ASSERT_EQ(hipSuccess, hip_.stream_wait_event(consumer, event, /*flags=*/0));
  void* reallocated = AllocateAsync(consumer, kReuseAllocationSize);
  ASSERT_NE(nullptr, reallocated);
  EXPECT_NE(freed, reallocated)
      << "an allocation freed past the tail the launch waited on was handed to "
         "a stream that only waited on a record made inside that launch";

  EXPECT_EQ(hipSuccess, hip_.free_async(reallocated, consumer));
  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(producer));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(consumer));
}

// A record landing between a waiter's read of the event and its submission has
// to leave the waiter waiting on the point it read. The interleaving is
// unordered by contract, so there is nothing to assert about which record wins;
// unaided this pins only that every call succeeds and every stream drains. It
// is also a manual race detector probe: no configured job runs it under one, so
// checking that the concurrent reads are synchronized means running it under
// one by hand.
TEST_F(HipEventTest, ConcurrentRecordsDoNotDisturbAConcurrentStreamWait) {
  hipStream_t stream_a = CreateStream();
  ASSERT_NE(nullptr, stream_a);
  hipStream_t stream_b = CreateStream();
  ASSERT_NE(nullptr, stream_b);
  hipStream_t consumer = CreateStream();
  ASSERT_NE(nullptr, consumer);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  // Gives the event a submitted record before either loop starts so the waiter
  // exercises a populated point from its first iteration.
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, stream_a, /*count=*/1));

  constexpr int kIterationCount = 100;
  std::atomic<hipError_t> record_result{hipSuccess};
  std::thread recorder([&] {
    for (int i = 0; i < kIterationCount; ++i) {
      const hipError_t result =
          hip_.event_record(event, (i % 2 == 0) ? stream_a : stream_b);
      if (result != hipSuccess) {
        record_result.store(result, std::memory_order_release);
        return;
      }
    }
  });

  hipError_t wait_result = hipSuccess;
  for (int i = 0; i < kIterationCount && wait_result == hipSuccess; ++i) {
    wait_result = hip_.stream_wait_event(consumer, event, /*flags=*/0);
  }
  recorder.join();

  EXPECT_EQ(hipSuccess, record_result.load(std::memory_order_acquire));
  EXPECT_EQ(hipSuccess, wait_result);
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(stream_a));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(stream_b));
  EXPECT_EQ(hipSuccess, hip_.stream_synchronize(consumer));
  EXPECT_EQ(hipSuccess, hip_.event_synchronize(event));
}

}  // namespace
