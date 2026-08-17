// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <dlfcn.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
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
using HipEventCreateFn = hipError_t (*)(hipEvent_t* event);
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
using HipFreeAsyncFn = hipError_t (*)(void* device_ptr, hipStream_t stream);
using HipGraphCreateFn = hipError_t (*)(hipGraph_t* graph, unsigned int flags);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t graph);
using HipGraphAddEventRecordNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, hipEvent_t event);
using HipGraphAddEventWaitNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, hipEvent_t event);
using HipGraphAddHostNodeFn = hipError_t (*)(hipGraphNode_t* node,
                                             hipGraph_t graph,
                                             const hipGraphNode_t* dependencies,
                                             size_t dependency_count,
                                             const void* node_params);
using HipGraphInstantiateFn = hipError_t (*)(hipGraphExec_t* graph_exec,
                                             hipGraph_t graph,
                                             hipGraphNode_t* error_node,
                                             char* log_buffer,
                                             size_t buffer_size);
using HipGraphLaunchFn = hipError_t (*)(hipGraphExec_t graph_exec,
                                        hipStream_t stream);
using HipGraphExecDestroyFn = hipError_t (*)(hipGraphExec_t graph_exec);

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
    hip_.event_create =
        ResolveHipSymbol<HipEventCreateFn>(library_, "hipEventCreate");
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
    hip_.graph_add_host_node = ResolveHipSymbol<HipGraphAddHostNodeFn>(
        library_, "hipGraphAddHostNode");
    hip_.graph_instantiate = ResolveHipSymbol<HipGraphInstantiateFn>(
        library_, "hipGraphInstantiate");
    hip_.graph_launch =
        ResolveHipSymbol<HipGraphLaunchFn>(library_, "hipGraphLaunch");
    hip_.graph_exec_destroy = ResolveHipSymbol<HipGraphExecDestroyFn>(
        library_, "hipGraphExecDestroy");

    ASSERT_NE(nullptr, hip_.init);
    ASSERT_NE(nullptr, hip_.stream_create);
    ASSERT_NE(nullptr, hip_.stream_destroy);
    ASSERT_NE(nullptr, hip_.stream_synchronize);
    ASSERT_NE(nullptr, hip_.stream_wait_event);
    ASSERT_NE(nullptr, hip_.stream_begin_capture);
    ASSERT_NE(nullptr, hip_.stream_end_capture);
    ASSERT_NE(nullptr, hip_.event_create);
    ASSERT_NE(nullptr, hip_.event_destroy);
    ASSERT_NE(nullptr, hip_.event_record);
    ASSERT_NE(nullptr, hip_.event_query);
    ASSERT_NE(nullptr, hip_.event_synchronize);
    ASSERT_NE(nullptr, hip_.event_elapsed_time);
    ASSERT_NE(nullptr, hip_.launch_host_func);
    ASSERT_NE(nullptr, hip_.malloc_async);
    ASSERT_NE(nullptr, hip_.free_async);
    ASSERT_NE(nullptr, hip_.graph_create);
    ASSERT_NE(nullptr, hip_.graph_destroy);
    ASSERT_NE(nullptr, hip_.graph_add_event_record_node);
    ASSERT_NE(nullptr, hip_.graph_add_event_wait_node);
    ASSERT_NE(nullptr, hip_.graph_add_host_node);
    ASSERT_NE(nullptr, hip_.graph_instantiate);
    ASSERT_NE(nullptr, hip_.graph_launch);
    ASSERT_NE(nullptr, hip_.graph_exec_destroy);

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
    HipEventCreateFn event_create;
    HipEventDestroyFn event_destroy;
    HipEventRecordFn event_record;
    HipEventQueryFn event_query;
    HipEventSynchronizeFn event_synchronize;
    HipEventElapsedTimeFn event_elapsed_time;
    HipLaunchHostFuncFn launch_host_func;
    HipMallocAsyncFn malloc_async;
    HipFreeAsyncFn free_async;
    HipGraphCreateFn graph_create;
    HipGraphDestroyFn graph_destroy;
    HipGraphAddEventRecordNodeFn graph_add_event_record_node;
    HipGraphAddEventWaitNodeFn graph_add_event_wait_node;
    HipGraphAddHostNodeFn graph_add_host_node;
    HipGraphInstantiateFn graph_instantiate;
    HipGraphLaunchFn graph_launch;
    HipGraphExecDestroyFn graph_exec_destroy;
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
TEST_F(HipEventTest, CrossStreamRerecordDoesNotChainTheNewStreamToTheOldOne) {
  hipStream_t stream_a = CreateStream();
  ASSERT_NE(nullptr, stream_a);
  hipStream_t stream_b = CreateStream();
  ASSERT_NE(nullptr, stream_b);
  hipEvent_t event = CreateEvent();
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
  EXPECT_EQ(hipErrorInvalidHandle, hip_.event_elapsed_time(&ms, start, stop))
      << "an interval was reported for records that were never submitted";
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
