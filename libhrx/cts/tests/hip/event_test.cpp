// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <dlfcn.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
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

    init_ = ResolveHipSymbol<HipInitFn>(library_, "hipInit");
    stream_create_ =
        ResolveHipSymbol<HipStreamCreateFn>(library_, "hipStreamCreate");
    stream_destroy_ =
        ResolveHipSymbol<HipStreamDestroyFn>(library_, "hipStreamDestroy");
    stream_synchronize_ = ResolveHipSymbol<HipStreamSynchronizeFn>(
        library_, "hipStreamSynchronize");
    stream_wait_event_ =
        ResolveHipSymbol<HipStreamWaitEventFn>(library_, "hipStreamWaitEvent");
    stream_begin_capture_ = ResolveHipSymbol<HipStreamBeginCaptureFn>(
        library_, "hipStreamBeginCapture");
    stream_end_capture_ = ResolveHipSymbol<HipStreamEndCaptureFn>(
        library_, "hipStreamEndCapture");
    event_create_ =
        ResolveHipSymbol<HipEventCreateFn>(library_, "hipEventCreate");
    event_destroy_ =
        ResolveHipSymbol<HipEventDestroyFn>(library_, "hipEventDestroy");
    event_record_ =
        ResolveHipSymbol<HipEventRecordFn>(library_, "hipEventRecord");
    event_query_ = ResolveHipSymbol<HipEventQueryFn>(library_, "hipEventQuery");
    event_synchronize_ = ResolveHipSymbol<HipEventSynchronizeFn>(
        library_, "hipEventSynchronize");
    event_elapsed_time_ = ResolveHipSymbol<HipEventElapsedTimeFn>(
        library_, "hipEventElapsedTime");
    launch_host_func_ =
        ResolveHipSymbol<HipLaunchHostFuncFn>(library_, "hipLaunchHostFunc");
    graph_create_ =
        ResolveHipSymbol<HipGraphCreateFn>(library_, "hipGraphCreate");
    graph_destroy_ =
        ResolveHipSymbol<HipGraphDestroyFn>(library_, "hipGraphDestroy");
    graph_add_event_record_node_ =
        ResolveHipSymbol<HipGraphAddEventRecordNodeFn>(
            library_, "hipGraphAddEventRecordNode");
    graph_add_event_wait_node_ = ResolveHipSymbol<HipGraphAddEventWaitNodeFn>(
        library_, "hipGraphAddEventWaitNode");
    graph_add_host_node_ = ResolveHipSymbol<HipGraphAddHostNodeFn>(
        library_, "hipGraphAddHostNode");
    graph_instantiate_ = ResolveHipSymbol<HipGraphInstantiateFn>(
        library_, "hipGraphInstantiate");
    graph_launch_ =
        ResolveHipSymbol<HipGraphLaunchFn>(library_, "hipGraphLaunch");
    graph_exec_destroy_ = ResolveHipSymbol<HipGraphExecDestroyFn>(
        library_, "hipGraphExecDestroy");

    ASSERT_NE(nullptr, init_);
    ASSERT_NE(nullptr, stream_create_);
    ASSERT_NE(nullptr, stream_destroy_);
    ASSERT_NE(nullptr, stream_synchronize_);
    ASSERT_NE(nullptr, stream_wait_event_);
    ASSERT_NE(nullptr, stream_begin_capture_);
    ASSERT_NE(nullptr, stream_end_capture_);
    ASSERT_NE(nullptr, event_create_);
    ASSERT_NE(nullptr, event_destroy_);
    ASSERT_NE(nullptr, event_record_);
    ASSERT_NE(nullptr, event_query_);
    ASSERT_NE(nullptr, event_synchronize_);
    ASSERT_NE(nullptr, event_elapsed_time_);
    ASSERT_NE(nullptr, launch_host_func_);
    ASSERT_NE(nullptr, graph_create_);
    ASSERT_NE(nullptr, graph_destroy_);
    ASSERT_NE(nullptr, graph_add_event_record_node_);
    ASSERT_NE(nullptr, graph_add_event_wait_node_);
    ASSERT_NE(nullptr, graph_add_host_node_);
    ASSERT_NE(nullptr, graph_instantiate_);
    ASSERT_NE(nullptr, graph_launch_);
    ASSERT_NE(nullptr, graph_exec_destroy_);

    const hipError_t init_result = init_(/*flags=*/0);
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
      EXPECT_EQ(hipSuccess, graph_exec_destroy_(*it));
    }
    for (auto it = graphs_.rbegin(); it != graphs_.rend(); ++it) {
      EXPECT_EQ(hipSuccess, graph_destroy_(*it));
    }
    for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
      EXPECT_EQ(hipSuccess, event_destroy_(*it));
    }
    for (auto it = streams_.rbegin(); it != streams_.rend(); ++it) {
      EXPECT_EQ(hipSuccess, stream_destroy_(*it));
    }
  }

  // Creates a stream owned by the fixture.
  hipStream_t CreateStream() {
    hipStream_t stream = nullptr;
    EXPECT_EQ(hipSuccess, stream_create_(&stream));
    if (stream) streams_.push_back(stream);
    return stream;
  }

  // Destroys a stream the fixture owns ahead of TearDown.
  void DestroyStream(hipStream_t stream) {
    streams_.erase(std::remove(streams_.begin(), streams_.end(), stream),
                   streams_.end());
    EXPECT_EQ(hipSuccess, stream_destroy_(stream));
  }

  // Creates an event owned by the fixture.
  hipEvent_t CreateEvent() {
    hipEvent_t event = nullptr;
    EXPECT_EQ(hipSuccess, event_create_(&event));
    if (event) events_.push_back(event);
    return event;
  }

  // Creates an empty graph owned by the fixture.
  hipGraph_t CreateGraph() {
    hipGraph_t graph = nullptr;
    EXPECT_EQ(hipSuccess, graph_create_(&graph, /*flags=*/0));
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
    EXPECT_EQ(hipSuccess, graph_destroy_(graph));
  }

  // Instantiates |graph| into an executable owned by the fixture.
  hipGraphExec_t InstantiateGraph(hipGraph_t graph) {
    hipGraphExec_t graph_exec = nullptr;
    EXPECT_EQ(hipSuccess,
              graph_instantiate_(&graph_exec, graph, /*error_node=*/nullptr,
                                 /*log_buffer=*/nullptr, /*buffer_size=*/0));
    if (graph_exec) graph_execs_.push_back(graph_exec);
    return graph_exec;
  }

  // Destroys an executable the fixture owns ahead of TearDown.
  void DestroyGraphExec(hipGraphExec_t graph_exec) {
    graph_execs_.erase(
        std::remove(graph_execs_.begin(), graph_execs_.end(), graph_exec),
        graph_execs_.end());
    EXPECT_EQ(hipSuccess, graph_exec_destroy_(graph_exec));
  }

  // Records |event| on |stream| and waits for it |count| times, leaving both
  // the stream and the event drained.
  void AdvanceEventOnStream(hipEvent_t event, hipStream_t stream, int count) {
    for (int i = 0; i < count; ++i) {
      ASSERT_EQ(hipSuccess, event_record_(event, stream));
      ASSERT_EQ(hipSuccess, event_synchronize_(event));
    }
    ASSERT_EQ(hipSuccess, stream_synchronize_(stream));
  }

  // Enqueues |gate| on |stream|, registers it with |callbacks|, and returns
  // once the callback is running and the stream provably holds unfinished work.
  void EnqueueGateAndWaitUntilEntered(hipStream_t stream, StreamGate* gate,
                                      ScopedHostCallbacks* callbacks) {
    ASSERT_EQ(hipSuccess, launch_host_func_(stream, &GateHostFunction, gate));
    ASSERT_NO_FATAL_FAILURE(callbacks->AddGate(gate));
    while (!gate->entered.load(std::memory_order_acquire)) {
      sched_yield();
    }
  }

  // HIP shim under test. Intentionally never dlclose()d: the shim owns
  // process-global device state shared by every test in this file.
  void* library_ = nullptr;
  // Resolved hipInit.
  HipInitFn init_ = nullptr;
  // Resolved hipStreamCreate.
  HipStreamCreateFn stream_create_ = nullptr;
  // Resolved hipStreamDestroy.
  HipStreamDestroyFn stream_destroy_ = nullptr;
  // Resolved hipStreamSynchronize.
  HipStreamSynchronizeFn stream_synchronize_ = nullptr;
  // Resolved hipStreamWaitEvent.
  HipStreamWaitEventFn stream_wait_event_ = nullptr;
  // Resolved hipStreamBeginCapture.
  HipStreamBeginCaptureFn stream_begin_capture_ = nullptr;
  // Resolved hipStreamEndCapture.
  HipStreamEndCaptureFn stream_end_capture_ = nullptr;
  // Resolved hipEventCreate.
  HipEventCreateFn event_create_ = nullptr;
  // Resolved hipEventDestroy.
  HipEventDestroyFn event_destroy_ = nullptr;
  // Resolved hipEventRecord.
  HipEventRecordFn event_record_ = nullptr;
  // Resolved hipEventQuery.
  HipEventQueryFn event_query_ = nullptr;
  // Resolved hipEventSynchronize.
  HipEventSynchronizeFn event_synchronize_ = nullptr;
  // Resolved hipEventElapsedTime.
  HipEventElapsedTimeFn event_elapsed_time_ = nullptr;
  // Resolved hipLaunchHostFunc.
  HipLaunchHostFuncFn launch_host_func_ = nullptr;
  // Resolved hipGraphCreate.
  HipGraphCreateFn graph_create_ = nullptr;
  // Resolved hipGraphDestroy.
  HipGraphDestroyFn graph_destroy_ = nullptr;
  // Resolved hipGraphAddEventRecordNode.
  HipGraphAddEventRecordNodeFn graph_add_event_record_node_ = nullptr;
  // Resolved hipGraphAddEventWaitNode.
  HipGraphAddEventWaitNodeFn graph_add_event_wait_node_ = nullptr;
  // Resolved hipGraphAddHostNode.
  HipGraphAddHostNodeFn graph_add_host_node_ = nullptr;
  // Resolved hipGraphInstantiate.
  HipGraphInstantiateFn graph_instantiate_ = nullptr;
  // Resolved hipGraphLaunch.
  HipGraphLaunchFn graph_launch_ = nullptr;
  // Resolved hipGraphExecDestroy.
  HipGraphExecDestroyFn graph_exec_destroy_ = nullptr;

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

// A record marks a point on the recording stream's timeline, so re-recording an
// event on a stream whose timeline is behind still names a point that stream
// has not reached.
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

  ASSERT_EQ(hipSuccess, event_record_(event, stream_b));
  EXPECT_EQ(hipErrorNotReady, event_query_(event))
      << "event completed while the gate callback holding its recording stream "
         "is still running";

  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, event_synchronize_(event));
  EXPECT_EQ(hipSuccess, event_query_(event));
  EXPECT_EQ(hipSuccess, stream_synchronize_(stream_b));
}

// Recording a shared event on a second stream must not make that stream wait
// for the first one: nothing was submitted between the two records, so the
// second stream drains while the first is still parked in its gate. A stream
// chained to the other one hangs in hipStreamSynchronize.
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

  ASSERT_EQ(hipSuccess, event_record_(event, stream_a));
  ASSERT_EQ(hipSuccess, event_record_(event, stream_b));

  EXPECT_EQ(hipSuccess, stream_synchronize_(stream_b));
  // The event names the last record, on the stream that just drained, so
  // waiting on it must not reach back to the parked stream either.
  EXPECT_EQ(hipSuccess, event_query_(event));
  EXPECT_EQ(hipSuccess, event_synchronize_(event));
  EXPECT_FALSE(gate.finished.load(std::memory_order_acquire))
      << "the gate was released before the independent stream was observed";

  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, stream_synchronize_(stream_a));
}

// Destroying the stream an event was recorded on has to leave the event
// queryable, and re-recording it elsewhere has to let go of what it was
// holding. This pins the handle-level contract; the event's reference to the
// timeline itself is what the graph executable case below covers.
TEST_F(HipEventTest, EventStaysUsableAfterItsRecordingStreamHandleIsDestroyed) {
  hipStream_t stream_a = CreateStream();
  ASSERT_NE(nullptr, stream_a);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, stream_a, /*count=*/2));

  ASSERT_NO_FATAL_FAILURE(DestroyStream(stream_a));
  EXPECT_EQ(hipSuccess, event_query_(event));
  EXPECT_EQ(hipSuccess, event_synchronize_(event));

  hipStream_t stream_b = CreateStream();
  ASSERT_NE(nullptr, stream_b);
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, stream_b, /*count=*/1));
  EXPECT_EQ(hipSuccess, event_query_(event));
}

// A record node in the middle of a graph marks a point on a timeline the graph
// executable owns, and the event's own reference is all that is left once the
// executable is destroyed: the event has to stay queryable afterwards, and
// re-recording it elsewhere has to let the timeline go.
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
  ASSERT_EQ(hipSuccess, graph_add_event_record_node_(
                            &record_node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, event));
  std::atomic<bool> host_node_ran{false};
  hipHostNodeParams host_params = {};
  host_params.fn = &RanHostFunction;
  host_params.userData = &host_node_ran;
  hipGraphNode_t host_node = nullptr;
  ASSERT_EQ(hipSuccess,
            graph_add_host_node_(&host_node, graph, &record_node,
                                 /*dependency_count=*/1, &host_params));
  hipGraphExec_t graph_exec = InstantiateGraph(graph);
  ASSERT_NE(nullptr, graph_exec);

  ScopedHostCallbacks callbacks;
  ASSERT_EQ(hipSuccess, graph_launch_(graph_exec, stream));
  callbacks.Add(host_node_ran);
  ASSERT_EQ(hipSuccess, stream_synchronize_(stream));

  ASSERT_NO_FATAL_FAILURE(DestroyGraphExec(graph_exec));
  ASSERT_NO_FATAL_FAILURE(DestroyGraph(graph));

  EXPECT_EQ(hipSuccess, event_query_(event));
  EXPECT_EQ(hipSuccess, event_synchronize_(event));

  hipStream_t stream_b = CreateStream();
  ASSERT_NE(nullptr, stream_b);
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, stream_b, /*count=*/1));
  EXPECT_EQ(hipSuccess, event_query_(event));
}

// A cross-stream re-record must leave the event usable as a dependency for
// another stream: work gated on it through hipStreamWaitEvent has to run only
// after the recording stream drains. This is a contract check, not a
// discriminator - the gated callback shares a queue thread with the gate.
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

  ASSERT_EQ(hipSuccess, event_record_(event, stream_b));
  ASSERT_EQ(hipSuccess, stream_wait_event_(stream_c, event, /*flags=*/0));
  ASSERT_EQ(hipSuccess,
            launch_host_func_(stream_c, &MarkerHostFunction, &marker));
  callbacks.Add(marker.finished);

  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, stream_synchronize_(stream_c));
  EXPECT_EQ(hipSuccess, stream_synchronize_(stream_b));

  EXPECT_TRUE(marker.finished.load(std::memory_order_acquire))
      << "work gated on the re-recorded event never ran";
  EXPECT_TRUE(marker.saw_gate_finished.load(std::memory_order_acquire))
      << "work gated on the re-recorded event ran before the stream that "
         "recorded the event drained";
}

// Waiting on an event does not release a stream from its own ordering: the wait
// advances the stream timeline, so it must complete only once the work in front
// of it has run. This is a contract check, not a discriminator - the queue also
// serializes signals on one timeline, which covers the same ordering.
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

  ASSERT_EQ(hipSuccess, stream_wait_event_(stream_b, event, /*flags=*/0));

  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, stream_synchronize_(stream_b));
  EXPECT_TRUE(gate.finished.load(std::memory_order_acquire))
      << "the stream reported itself drained while work submitted before the "
         "event wait was still running";
}

// Both halves of a stream event wait can be absent at once: a stream that has
// never submitted has nothing behind it, and an event with no submitted record
// has no point to wait for. The barrier goes out with an empty wait list and
// still has to be accepted and still has to signal the stream.
TEST_F(HipEventTest, StreamWaitOnANeverRecordedEventDropsBothWaits) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  ASSERT_EQ(hipSuccess, stream_wait_event_(stream, event, /*flags=*/0));
  EXPECT_EQ(hipSuccess, stream_synchronize_(stream));

  // The stream has now submitted, so the same wait keeps the stream half and
  // drops only the event half.
  ASSERT_EQ(hipSuccess, stream_wait_event_(stream, event, /*flags=*/0));
  EXPECT_EQ(hipSuccess, stream_synchronize_(stream));

  // And once the event has a record the wait carries both halves again.
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, stream, /*count=*/1));
  ASSERT_EQ(hipSuccess, stream_wait_event_(stream, event, /*flags=*/0));
  EXPECT_EQ(hipSuccess, stream_synchronize_(stream));
}

// A graph event wait node on an event with no submitted record has nothing to
// wait for, so the wait is dropped and the block goes out with whatever waits
// the schedule gave it. The node behind it still has to run.
TEST_F(HipEventTest, GraphEventWaitNodeOnANeverRecordedEventDropsTheWait) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  hipGraph_t graph = CreateGraph();
  ASSERT_NE(nullptr, graph);
  hipGraphNode_t wait_node = nullptr;
  ASSERT_EQ(hipSuccess,
            graph_add_event_wait_node_(&wait_node, graph,
                                       /*dependencies=*/nullptr,
                                       /*dependency_count=*/0, event));
  std::atomic<bool> host_node_ran{false};
  hipHostNodeParams host_params = {};
  host_params.fn = &RanHostFunction;
  host_params.userData = &host_node_ran;
  hipGraphNode_t host_node = nullptr;
  ASSERT_EQ(hipSuccess,
            graph_add_host_node_(&host_node, graph, &wait_node,
                                 /*dependency_count=*/1, &host_params));
  hipGraphExec_t graph_exec = InstantiateGraph(graph);
  ASSERT_NE(nullptr, graph_exec);

  ScopedHostCallbacks callbacks;
  ASSERT_EQ(hipSuccess, graph_launch_(graph_exec, stream));
  callbacks.Add(host_node_ran);
  EXPECT_EQ(hipSuccess, stream_synchronize_(stream));
  EXPECT_TRUE(host_node_ran.load(std::memory_order_acquire))
      << "the node behind a dropped event wait never ran";
}

// A graph whose only node records an event marks the point the launch itself
// reaches, so launching it onto a parked stream leaves the event not ready
// until that stream drains.
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
  ASSERT_EQ(hipSuccess,
            graph_add_event_record_node_(&node, graph, /*dependencies=*/nullptr,
                                         /*dependency_count=*/0, event));
  hipGraphExec_t graph_exec = InstantiateGraph(graph);
  ASSERT_NE(nullptr, graph_exec);

  StreamGate gate;
  ScopedHostCallbacks callbacks;
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(stream, &gate, &callbacks));

  ASSERT_EQ(hipSuccess, graph_launch_(graph_exec, stream));
  EXPECT_EQ(hipErrorNotReady, event_query_(event))
      << "event completed while the gate callback holding the stream the graph "
         "launched onto is still running";

  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, event_synchronize_(event));
  EXPECT_EQ(hipSuccess, event_query_(event));

  // Repeated launches keep naming a fresh point, and stream records keep
  // working on an event a graph has been recording.
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(hipSuccess, graph_launch_(graph_exec, stream));
    EXPECT_EQ(hipSuccess, event_synchronize_(event));
    EXPECT_EQ(hipSuccess, event_query_(event));
  }
  EXPECT_EQ(hipSuccess, stream_synchronize_(stream));
  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(event, stream, /*count=*/1));
}

// A record node with a node after it marks the point that node reaches rather
// than the point the whole launch reaches: the launch signals the stream, while
// a node in the middle signals one of the executable's internal semaphores. The
// event still must not read as complete while the launch is parked.
TEST_F(HipEventTest, GraphRecordNodeInsideAGraphMarksTheNodePoint) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t event = CreateEvent();
  ASSERT_NE(nullptr, event);

  hipGraph_t graph = CreateGraph();
  ASSERT_NE(nullptr, graph);
  hipGraphNode_t record_node = nullptr;
  ASSERT_EQ(hipSuccess, graph_add_event_record_node_(
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
            graph_add_host_node_(&host_node, graph, &record_node,
                                 /*dependency_count=*/1, &host_params));

  hipGraphExec_t graph_exec = InstantiateGraph(graph);
  ASSERT_NE(nullptr, graph_exec);

  ScopedHostCallbacks callbacks;
  ASSERT_NO_FATAL_FAILURE(
      EnqueueGateAndWaitUntilEntered(stream, &gate, &callbacks));

  ASSERT_EQ(hipSuccess, graph_launch_(graph_exec, stream));
  callbacks.Add(marker.finished);
  EXPECT_EQ(hipErrorNotReady, event_query_(event))
      << "event completed while the gate callback holding the stream the graph "
         "launched onto is still running";

  callbacks.ReleaseGate();
  EXPECT_EQ(hipSuccess, event_synchronize_(event));
  EXPECT_EQ(hipSuccess, event_query_(event));
  EXPECT_EQ(hipSuccess, stream_synchronize_(stream));
  EXPECT_TRUE(marker.saw_gate_finished.load(std::memory_order_acquire))
      << "the node after the record node ran before the work in front of the "
         "launch drained";
}

// Elapsed time reads the timestamp a record leaves behind, which is adopted
// only once the submission carrying it has been accepted, so an event with no
// submitted record has no interval to report.
TEST_F(HipEventTest, ElapsedTimeNeedsBothEventsRecorded) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t start = CreateEvent();
  ASSERT_NE(nullptr, start);
  hipEvent_t stop = CreateEvent();
  ASSERT_NE(nullptr, stop);

  float ms = -1.0f;
  EXPECT_EQ(hipErrorInvalidHandle, event_elapsed_time_(&ms, start, stop));

  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(start, stream, /*count=*/1));
  EXPECT_EQ(hipErrorInvalidHandle, event_elapsed_time_(&ms, start, stop))
      << "an interval was reported for an event that was never recorded";

  ASSERT_NO_FATAL_FAILURE(AdvanceEventOnStream(stop, stream, /*count=*/1));
  ms = -1.0f;
  EXPECT_EQ(hipSuccess, event_elapsed_time_(&ms, start, stop));
  EXPECT_GE(ms, 0.0f);
}

// Both events have to have completed before an interval means anything, so a
// stop event whose record is still parked reports as not ready.
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
  ASSERT_EQ(hipSuccess, event_record_(stop, stream_b));

  float ms = -1.0f;
  EXPECT_EQ(hipErrorNotReady, event_elapsed_time_(&ms, start, stop));

  callbacks.ReleaseGate();
  ASSERT_EQ(hipSuccess, event_synchronize_(stop));
  ms = -1.0f;
  EXPECT_EQ(hipSuccess, event_elapsed_time_(&ms, start, stop));
  EXPECT_GE(ms, 0.0f);
}

// A record issued while its stream is capturing produces neither a submission
// nor a graph node, only a snapshot of the capture frontier, so it leaves the
// event with no timestamp and no interval to report.
TEST_F(HipEventTest, ElapsedTimeRejectsEventsRecordedOnlyDuringCapture) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  hipEvent_t start = CreateEvent();
  ASSERT_NE(nullptr, start);
  hipEvent_t stop = CreateEvent();
  ASSERT_NE(nullptr, stop);

  ASSERT_EQ(hipSuccess,
            stream_begin_capture_(stream, hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess, event_record_(start, stream));
  ASSERT_EQ(hipSuccess, event_record_(stop, stream));
  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, stream_end_capture_(stream, &graph));
  TrackGraph(graph);
  ASSERT_NE(nullptr, graph);

  float ms = -1.0f;
  EXPECT_EQ(hipErrorInvalidHandle, event_elapsed_time_(&ms, start, stop))
      << "an interval was reported for records that were never submitted";
}

}  // namespace
