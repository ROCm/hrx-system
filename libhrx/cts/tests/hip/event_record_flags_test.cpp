// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <atomic>
#include <cstdint>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

class HipEventRecordFlagsTest : public ::testing::TestWithParam<unsigned int> {
 protected:
  void SetUp() override {
    ASSERT_TRUE(library_.Open()) << library_.error();
#define RESOLVE_HIP(name)                             \
  name##_ = library_.Resolve<decltype(&name)>(#name); \
  ASSERT_NE(nullptr, name##_) << library_.error()
    RESOLVE_HIP(hipInit);
    RESOLVE_HIP(hipStreamCreate);
    RESOLVE_HIP(hipStreamDestroy);
    RESOLVE_HIP(hipStreamSynchronize);
    RESOLVE_HIP(hipStreamBeginCapture);
    RESOLVE_HIP(hipStreamEndCapture);
    RESOLVE_HIP(hipEventCreateWithFlags);
    RESOLVE_HIP(hipEventDestroy);
    RESOLVE_HIP(hipEventRecordWithFlags);
    RESOLVE_HIP(hipEventQuery);
    RESOLVE_HIP(hipEventSynchronize);
    RESOLVE_HIP(hipGraphGetNodes);
    RESOLVE_HIP(hipGraphNodeGetType);
    RESOLVE_HIP(hipGraphInstantiate);
    RESOLVE_HIP(hipGraphLaunch);
    RESOLVE_HIP(hipGraphExecDestroy);
    RESOLVE_HIP(hipGraphDestroy);
    RESOLVE_HIP(hipLaunchHostFunc);
#undef RESOLVE_HIP
    ASSERT_EQ(hipSuccess, hipInit_(0));
    ASSERT_EQ(hipSuccess, hipStreamCreate_(&stream_));
    ASSERT_EQ(hipSuccess, hipEventCreateWithFlags_(&event_, GetParam()));
  }

  void TearDown() override {
    if (capturing_) {
      EXPECT_EQ(hipSuccess, hipStreamEndCapture_(stream_, &graph_));
    }
    if (executable_) {
      EXPECT_EQ(hipSuccess, hipGraphExecDestroy_(executable_));
    }
    if (graph_) {
      EXPECT_EQ(hipSuccess, hipGraphDestroy_(graph_));
    }
    if (event_) {
      EXPECT_EQ(hipSuccess, hipEventDestroy_(event_));
    }
    if (stream_) {
      EXPECT_EQ(hipSuccess, hipStreamDestroy_(stream_));
    }
  }

  static void TearDownTestSuite() {
    auto deinitialize = library_.Resolve<hipError_t (*)(void)>("hipHALDeinit");
    ASSERT_NE(nullptr, deinitialize) << library_.error();
    EXPECT_EQ(hipSuccess, deinitialize());
    EXPECT_TRUE(library_.Close()) << library_.error();
  }

  void BeginCapture() {
    ASSERT_EQ(hipSuccess,
              hipStreamBeginCapture_(stream_, hipStreamCaptureModeGlobal));
    capturing_ = true;
  }

  void EndCapture() {
    hipError_t result = hipStreamEndCapture_(stream_, &graph_);
    capturing_ = false;
    ASSERT_EQ(hipSuccess, result);
  }

  void Instantiate() {
    ASSERT_EQ(hipSuccess,
              hipGraphInstantiate_(&executable_, graph_, nullptr, nullptr, 0));
  }

  static void CountLaunch(void* user_data) {
    static_cast<std::atomic<int>*>(user_data)->fetch_add(
        1, std::memory_order_release);
  }

  // Exact runtime under test, retained until suite shutdown.
  static hrx::hip::testing::HipDso library_;
  // Recording and replay stream owned by this test.
  hipStream_t stream_ = nullptr;
  // Public event handle, independent of references retained by graph nodes.
  hipEvent_t event_ = nullptr;
  // Template returned when capture ends, or null before that point.
  hipGraph_t graph_ = nullptr;
  // Executable retaining the captured template's operations.
  hipGraphExec_t executable_ = nullptr;
  // Whether teardown must terminate an unfinished capture.
  bool capturing_ = false;
  // Host callback state, alive until teardown retires the graph executable.
  std::atomic<int> launch_count_{0};

  // Initializes the runtime before creating public handles.
  decltype(&hipInit) hipInit_ = nullptr;
  // Creates the explicit recording stream.
  decltype(&hipStreamCreate) hipStreamCreate_ = nullptr;
  // Retires the stream during teardown.
  decltype(&hipStreamDestroy) hipStreamDestroy_ = nullptr;
  // Waits for replay when the public event handle has been destroyed.
  decltype(&hipStreamSynchronize) hipStreamSynchronize_ = nullptr;
  // Starts ordinary graph capture.
  decltype(&hipStreamBeginCapture) hipStreamBeginCapture_ = nullptr;
  // Publishes the captured graph template.
  decltype(&hipStreamEndCapture) hipStreamEndCapture_ = nullptr;
  // Creates an event with the selected timing mode.
  decltype(&hipEventCreateWithFlags) hipEventCreateWithFlags_ = nullptr;
  // Retires the public event handle before graph-owned references.
  decltype(&hipEventDestroy) hipEventDestroy_ = nullptr;
  // Entry point and flags being qualified.
  decltype(&hipEventRecordWithFlags) hipEventRecordWithFlags_ = nullptr;
  // Observes event state through the public runtime.
  decltype(&hipEventQuery) hipEventQuery_ = nullptr;
  // Waits on the latest submitted record.
  decltype(&hipEventSynchronize) hipEventSynchronize_ = nullptr;
  // Enumerates captured operations without inspecting implementation objects.
  decltype(&hipGraphGetNodes) hipGraphGetNodes_ = nullptr;
  // Distinguishes explicit event records from host operations.
  decltype(&hipGraphNodeGetType) hipGraphNodeGetType_ = nullptr;
  // Prepares the captured template for execution.
  decltype(&hipGraphInstantiate) hipGraphInstantiate_ = nullptr;
  // Replays the captured operations.
  decltype(&hipGraphLaunch) hipGraphLaunch_ = nullptr;
  // Retires the executable during teardown.
  decltype(&hipGraphExecDestroy) hipGraphExecDestroy_ = nullptr;
  // Retires the graph template during teardown.
  decltype(&hipGraphDestroy) hipGraphDestroy_ = nullptr;
  // Captures an observable operation before the event record.
  decltype(&hipLaunchHostFunc) hipLaunchHostFunc_ = nullptr;
};

hrx::hip::testing::HipDso HipEventRecordFlagsTest::library_;

TEST_P(HipEventRecordFlagsTest, BothFlagsRecordOnExplicitAndDefaultStreams) {
  const std::array<hipStream_t, 4> streams = {stream_, nullptr, hipStreamLegacy,
                                              hipStreamPerThread};
  for (unsigned int flags : {hipEventRecordDefault, hipEventRecordExternal}) {
    for (hipStream_t stream : streams) {
      SCOPED_TRACE(flags);
      SCOPED_TRACE(stream);
      ASSERT_EQ(hipSuccess, hipEventRecordWithFlags_(event_, stream, flags));
      ASSERT_EQ(hipSuccess, hipEventSynchronize_(event_));
      EXPECT_EQ(hipSuccess, hipEventQuery_(event_));
    }
  }
}

TEST_P(HipEventRecordFlagsTest, InvalidFlagsAndHandlesAreRejected) {
  EXPECT_EQ(hipErrorInvalidValue, hipEventRecordWithFlags_(event_, stream_, 2));
  EXPECT_EQ(hipErrorInvalidValue,
            hipEventRecordWithFlags_(event_, stream_, UINT32_MAX));
  EXPECT_EQ(hipErrorInvalidHandle,
            hipEventRecordWithFlags_(nullptr, stream_, hipEventRecordDefault));
  EXPECT_EQ(hipErrorInvalidHandle,
            hipEventRecordWithFlags_(reinterpret_cast<hipEvent_t>(0x1234),
                                     stream_, hipEventRecordExternal));
  EXPECT_EQ(
      hipErrorInvalidHandle,
      hipEventRecordWithFlags_(event_, reinterpret_cast<hipStream_t>(0x1234),
                               hipEventRecordDefault));
  EXPECT_EQ(hipSuccess, hipEventQuery_(event_));
}

TEST_P(HipEventRecordFlagsTest, DefaultCaptureRecordsOnlyTheFrontier) {
  ASSERT_NO_FATAL_FAILURE(BeginCapture());
  ASSERT_EQ(hipSuccess,
            hipEventRecordWithFlags_(event_, stream_, hipEventRecordDefault));
  EXPECT_EQ(hipErrorInvalidValue, hipEventRecordWithFlags_(event_, stream_, 2));
  ASSERT_NO_FATAL_FAILURE(EndCapture());
  size_t node_count = 0;
  ASSERT_EQ(hipSuccess, hipGraphGetNodes_(graph_, nullptr, &node_count));
  EXPECT_EQ(0u, node_count);
  ASSERT_EQ(hipSuccess,
            hipEventRecordWithFlags_(event_, stream_, hipEventRecordDefault));
  EXPECT_EQ(hipSuccess, hipEventSynchronize_(event_));
}

TEST_P(HipEventRecordFlagsTest, ExternalCapturePublishesEachReplayToTheEvent) {
  ASSERT_NO_FATAL_FAILURE(BeginCapture());
  ASSERT_EQ(hipSuccess,
            hipLaunchHostFunc_(stream_, CountLaunch, &launch_count_));
  ASSERT_EQ(hipSuccess,
            hipEventRecordWithFlags_(event_, stream_, hipEventRecordExternal));
  ASSERT_NO_FATAL_FAILURE(EndCapture());
  size_t node_count = 0;
  ASSERT_EQ(hipSuccess, hipGraphGetNodes_(graph_, nullptr, &node_count));
  ASSERT_EQ(2u, node_count);
  std::array<hipGraphNode_t, 2> nodes = {};
  ASSERT_EQ(hipSuccess, hipGraphGetNodes_(graph_, nodes.data(), &node_count));
  hipGraphNodeType type;
  ASSERT_EQ(hipSuccess, hipGraphNodeGetType_(nodes[0], &type));
  EXPECT_EQ(hipGraphNodeTypeHost, type);
  ASSERT_EQ(hipSuccess, hipGraphNodeGetType_(nodes[1], &type));
  EXPECT_EQ(hipGraphNodeTypeEventRecord, type);
  ASSERT_NO_FATAL_FAILURE(Instantiate());
  for (int launch = 1; launch <= 3; ++launch) {
    ASSERT_EQ(hipSuccess, hipGraphLaunch_(executable_, stream_));
    ASSERT_EQ(hipSuccess, hipEventSynchronize_(event_));
    EXPECT_EQ(hipSuccess, hipEventQuery_(event_));
    EXPECT_EQ(launch, launch_count_.load(std::memory_order_acquire));
  }
}

TEST_P(HipEventRecordFlagsTest, ReplayRetainsAnEventAfterItsHandleIsDestroyed) {
  ASSERT_NO_FATAL_FAILURE(BeginCapture());
  ASSERT_EQ(hipSuccess,
            hipLaunchHostFunc_(stream_, CountLaunch, &launch_count_));
  ASSERT_EQ(hipSuccess,
            hipEventRecordWithFlags_(event_, stream_, hipEventRecordExternal));
  ASSERT_NO_FATAL_FAILURE(EndCapture());
  ASSERT_EQ(hipSuccess, hipEventDestroy_(event_));
  event_ = nullptr;
  ASSERT_NO_FATAL_FAILURE(Instantiate());
  ASSERT_EQ(hipSuccess, hipGraphDestroy_(graph_));
  graph_ = nullptr;
  ASSERT_EQ(hipSuccess, hipGraphLaunch_(executable_, stream_));
  ASSERT_EQ(hipSuccess, hipStreamSynchronize_(stream_));
  EXPECT_EQ(1, launch_count_.load(std::memory_order_acquire));
}

TEST_P(HipEventRecordFlagsTest, AnUnlaunchedExternalRecordCanBeDestroyed) {
  ASSERT_NO_FATAL_FAILURE(BeginCapture());
  ASSERT_EQ(hipSuccess,
            hipEventRecordWithFlags_(event_, stream_, hipEventRecordExternal));
  ASSERT_NO_FATAL_FAILURE(EndCapture());
  ASSERT_EQ(hipSuccess, hipGraphDestroy_(graph_));
  graph_ = nullptr;
  ASSERT_EQ(hipSuccess, hipEventDestroy_(event_));
  event_ = nullptr;
}

TEST_P(HipEventRecordFlagsTest,
       QueryingACapturedExternalRecordInvalidatesCapture) {
  ASSERT_NO_FATAL_FAILURE(BeginCapture());
  ASSERT_EQ(hipSuccess,
            hipEventRecordWithFlags_(event_, stream_, hipEventRecordExternal));
  EXPECT_EQ(hipErrorCapturedEvent, hipEventQuery_(event_));
  EXPECT_EQ(hipErrorStreamCaptureInvalidated,
            hipStreamEndCapture_(stream_, &graph_));
  capturing_ = false;
  EXPECT_EQ(nullptr, graph_);
}

INSTANTIATE_TEST_SUITE_P(TimingModes, HipEventRecordFlagsTest,
                         ::testing::Values(hipEventDefault,
                                           hipEventDisableTiming));

}  // namespace
