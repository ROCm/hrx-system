// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/hip_graph_memory_test.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace hrx::hip::testing {

hrx::hip::testing::HipDso HipGraphMemoryApiTest::dso_;
HipGraphApi HipGraphMemoryApiTest::api_;

void HipGraphMemoryApiTest::SetUpTestSuite() {
  ASSERT_TRUE(dso_.Open()) << dso_.error();
#define RESOLVE(field, symbol)                             \
  api_.field = dso_.Resolve<decltype(api_.field)>(symbol); \
  ASSERT_NE(api_.field, nullptr) << dso_.error()
  RESOLVE(initialize, "hipInit");
  RESOLVE(deinitialize, "hipHALDeinit");
  RESOLVE(get_device, "hipGetDevice");
  RESOLVE(get_context, "hipCtxGetCurrent");
  RESOLVE(create_context, "hipCtxCreate");
  RESOLVE(set_context, "hipCtxSetCurrent");
  RESOLVE(destroy_context, "hipCtxDestroy");
  RESOLVE(copy, "hipMemcpy");
  RESOLVE(stream_create, "hipStreamCreateWithFlags");
  RESOLVE(stream_destroy, "hipStreamDestroy");
  RESOLVE(stream_synchronize, "hipStreamSynchronize");
  RESOLVE(stream_query, "hipStreamQuery");
  RESOLVE(launch_host, "hipLaunchHostFunc");
  RESOLVE(allocate, "hipMalloc");
  RESOLVE(free, "hipFree");
  RESOLVE(allocate_async, "hipMallocAsync");
  RESOLVE(free_async, "hipFreeAsync");
  RESOLVE(graph_memory_attribute, "hipDeviceGetGraphMemAttribute");
  RESOLVE(graph_memory_trim, "hipDeviceGraphMemTrim");
  RESOLVE(allocate_host, "hipHostAlloc");
  RESOLVE(free_host, "hipFreeHost");
  RESOLVE(graph_create, "hipGraphCreate");
  RESOLVE(graph_destroy, "hipGraphDestroy");
  RESOLVE(graph_clone, "hipGraphClone");
  RESOLVE(find_clone_node, "hipGraphNodeFindInClone");
  RESOLVE(instantiate, "hipGraphInstantiateWithFlags");
  RESOLVE(executable_destroy, "hipGraphExecDestroy");
  RESOLVE(launch, "hipGraphLaunch");
  RESOLVE(add_copy_1d, "hipGraphAddMemcpyNode1D");
  RESOLVE(add_allocation, "hipGraphAddMemAllocNode");
  RESOLVE(add_free, "hipGraphAddMemFreeNode");
  RESOLVE(add_copy, "hipGraphAddMemcpyNode");
  RESOLVE(add_fill, "hipGraphAddMemsetNode");
  RESOLVE(get_fill, "hipGraphMemsetNodeGetParams");
  RESOLVE(set_fill, "hipGraphMemsetNodeSetParams");
  RESOLVE(set_executable_fill, "hipGraphExecMemsetNodeSetParams");
  RESOLVE(fill_2d_async, "hipMemset2DAsync");
  RESOLVE(add_empty, "hipGraphAddEmptyNode");
  RESOLVE(set_copy, "hipGraphMemcpyNodeSetParams1D");
  RESOLVE(set_executable_copy, "hipGraphExecMemcpyNodeSetParams1D");
  RESOLVE(set_executable_copy_3d, "hipGraphExecMemcpyNodeSetParams");
  RESOLVE(update_executable, "hipGraphExecUpdate");
  RESOLVE(set_enabled, "hipGraphNodeSetEnabled");
  RESOLVE(get_node_type, "hipGraphNodeGetType");
  RESOLVE(destroy_node, "hipGraphDestroyNode");
  RESOLVE(add_dependencies, "hipGraphAddDependencies");
  RESOLVE(remove_dependencies, "hipGraphRemoveDependencies");
  RESOLVE(get_edges, "hipGraphGetEdges");
  RESOLVE(get_roots, "hipGraphGetRootNodes");
  RESOLVE(get_dependencies, "hipGraphNodeGetDependencies");
  RESOLVE(get_dependents, "hipGraphNodeGetDependentNodes");
  RESOLVE(begin_capture, "hipStreamBeginCapture");
  RESOLVE(begin_capture_to_graph, "hipStreamBeginCaptureToGraph");
  RESOLVE(end_capture, "hipStreamEndCapture");
  RESOLVE(capture_info, "hipStreamGetCaptureInfo_v2");
  RESOLVE(copy_async, "hipMemcpyAsync");
  RESOLVE(copy_3d_async, "hipMemcpy3DAsync");
  RESOLVE(get_nodes, "hipGraphGetNodes");
  RESOLVE(add_child, "hipGraphAddChildGraphNode");
  RESOLVE(get_child, "hipGraphChildGraphNodeGetGraph");
  RESOLVE(set_executable_node, "hipGraphExecNodeSetParams");
  RESOLVE(get_copy, "hipGraphMemcpyNodeGetParams");
  RESOLVE(add_host, "hipGraphAddHostNode");
  RESOLVE(get_host, "hipGraphHostNodeGetParams");
  RESOLVE(set_host, "hipGraphHostNodeSetParams");
  RESOLVE(set_executable_host, "hipGraphExecHostNodeSetParams");
#undef RESOLVE
  ASSERT_EQ(hipSuccess, api_.initialize(0));
}

void HipGraphMemoryApiTest::TearDownTestSuite() {
  if (!dso_.is_open()) {
    return;
  }
  ASSERT_NE(api_.deinitialize, nullptr);
  EXPECT_EQ(hipSuccess, api_.deinitialize());
  api_ = {};
  EXPECT_TRUE(dso_.Close()) << dso_.error();
}

void HipGraphMemoryApiTest::SetUp() {
  ASSERT_EQ(hipSuccess, api_.stream_create(&stream_, hipStreamNonBlocking));
  ASSERT_EQ(hipSuccess, api_.get_context(&primary_context_));
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph_, 0));
  for (size_t i = 0; i < source_.size(); ++i) {
    source_[i] = static_cast<uint8_t>(i + 1);
    replacement_[i] = static_cast<uint8_t>(i + 101);
  }
}

void HipGraphMemoryApiTest::TearDown() {
  ReleaseStreamGate();
  if (primary_context_) {
    EXPECT_EQ(hipSuccess, api_.set_context(primary_context_));
  }
  if (stream_) {
    EXPECT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  }
  if (other_stream_) {
    EXPECT_EQ(hipSuccess, api_.stream_synchronize(other_stream_));
  }
  if (executable_) {
    EXPECT_EQ(hipSuccess, api_.executable_destroy(executable_));
  }
  if (sibling_) {
    EXPECT_EQ(hipSuccess, api_.executable_destroy(sibling_));
  }
  if (clone_) {
    EXPECT_EQ(hipSuccess, api_.graph_destroy(clone_));
  }
  if (graph_) {
    EXPECT_EQ(hipSuccess, api_.graph_destroy(graph_));
  }
  if (stream_) {
    EXPECT_EQ(hipSuccess, api_.stream_destroy(stream_));
  }
  if (other_stream_) {
    EXPECT_EQ(hipSuccess, api_.stream_destroy(other_stream_));
  }
  if (device_) {
    EXPECT_EQ(hipSuccess, api_.free(device_));
  }
  for (auto* buffer : host_buffers_) {
    if (buffer) {
      EXPECT_EQ(hipSuccess, api_.free_host(buffer));
    }
  }
  if (remote_context_) {
    EXPECT_EQ(hipSuccess, api_.set_context(remote_context_));
    for (auto* buffer : remote_buffers_) {
      if (buffer) {
        EXPECT_EQ(hipSuccess, api_.free(buffer));
      }
    }
    EXPECT_EQ(hipSuccess, api_.destroy_context(remote_context_));
    EXPECT_EQ(hipSuccess, api_.set_context(primary_context_));
  }
}

void HipGraphMemoryApiTest::Replay(hipGraphExec_t executable) {
  ASSERT_EQ(hipSuccess,
            api_.launch(executable ? executable : executable_, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
}

void HipGraphMemoryApiTest::BuildAllocationGraph(unsigned long long flags,
                                                 void** out_pointer) {
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  hipMemAllocNodeParams allocation = {};
  allocation.poolProps.allocType = hipMemAllocationTypePinned;
  allocation.poolProps.location.type = hipMemLocationTypeDevice;
  allocation.poolProps.location.id = device;
  allocation.bytesize = source_.size();
  hipGraphNode_t allocation_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_allocation(&allocation_node, graph_, nullptr,
                                            0, &allocation));
  hipGraphNode_t upload = nullptr;
  ASSERT_EQ(
      hipSuccess,
      api_.add_copy_1d(&upload, graph_, &allocation_node, 1, allocation.dptr,
                       source_.data(), source_.size(), hipMemcpyHostToDevice));
  hipGraphNode_t download = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_copy_1d(&download, graph_, &upload, 1, destination_.data(),
                             allocation.dptr, destination_.size(),
                             hipMemcpyDeviceToHost));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, flags));
  *out_pointer = allocation.dptr;
}

void HipGraphMemoryApiTest::HoldStreamAtGate(hipStream_t stream) {
  gate_.pending = true;
  ASSERT_EQ(hipSuccess, api_.launch_host(
                            stream ? stream : stream_,
                            +[](void* argument) {
                              auto* gate = static_cast<StreamGate*>(argument);
                              gate->entered.release();
                              gate->release.acquire();
                            },
                            &gate_));
  gate_.entered.acquire();
}

void HipGraphMemoryApiTest::ReleaseStreamGate() {
  if (gate_.pending) {
    gate_.release.release();
    gate_.pending = false;
  }
}

void HipGraphMemoryApiTest::CheckPendingStagedCopyUpdates(bool pitched) {
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size()));
  const size_t width = pitched ? 4 : source_.size();
  const size_t height = pitched ? 2 : 1;
  const size_t depth = pitched ? 2 : 1;
  const size_t pitch = pitched ? 8 : width;
  const size_t slice_height = pitched ? 3 : height;
  hipMemcpy3DParms upload_params = {
      .srcPos = {pitched ? size_t{1} : 0, pitched ? size_t{1} : 0, 0},
      .srcPtr = {source_.data(), pitch, pitch, slice_height},
      .dstPos = {pitched ? size_t{1} : 0, pitched ? size_t{1} : 0, 0},
      .dstPtr = {device_, pitch, pitch, slice_height},
      .extent = {width, height, depth},
      .kind = hipMemcpyHostToDevice,
  };
  hipMemcpy3DParms download_params = {
      .srcPos = {pitched ? size_t{1} : 0, pitched ? size_t{1} : 0, 0},
      .srcPtr = {device_, pitch, pitch, slice_height},
      .dstPos = {pitched ? size_t{2} : 0, pitched ? size_t{1} : 0, 0},
      .dstPtr = {destination_.data(), pitched ? 10 : width,
                 pitched ? 10 : width, slice_height},
      .extent = {width, height, depth},
      .kind = hipMemcpyDeviceToHost,
  };
  const auto expected = [pitched](const std::array<uint8_t, 64>& input) {
    if (!pitched) {
      return input;
    }
    std::array<uint8_t, 64> output = {};
    constexpr size_t source_offsets[] = {9, 17, 33, 41};
    constexpr size_t target_offsets[] = {12, 22, 42, 52};
    for (size_t i = 0; i < 4; ++i) {
      std::copy_n(input.data() + source_offsets[i], 4,
                  output.data() + target_offsets[i]);
    }
    return output;
  };
  hipGraphNode_t upload = nullptr;
  hipGraphNode_t download = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_copy(&upload, graph_, nullptr, 0, &upload_params));
  ASSERT_EQ(hipSuccess,
            api_.add_copy(&download, graph_, &upload, 1, &download_params));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));

  HoldStreamAtGate();
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(hipSuccess, api_.launch(executable_, stream_));
  upload_params.srcPtr.ptr = replacement_.data();
  download_params.dstPtr.ptr = observed_.data();
  if (pitched) {
    ASSERT_EQ(hipSuccess,
              api_.set_executable_copy_3d(executable_, upload, &upload_params));
    ASSERT_EQ(hipSuccess, api_.set_executable_copy_3d(executable_, download,
                                                      &download_params));
  } else {
    ASSERT_EQ(hipSuccess, api_.set_executable_copy(
                              executable_, upload, device_, replacement_.data(),
                              replacement_.size(), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, api_.set_executable_copy(
                              executable_, download, observed_.data(), device_,
                              observed_.size(), hipMemcpyDeviceToHost));
  }
  // A rejected replacement must preserve both the accepted private recording
  // and the older invocation still waiting at the stream gate.
  hipMemcpy3DParms invalid_params = upload_params;
  invalid_params.dstPtr.pitch = 1;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.set_executable_copy_3d(executable_, upload, &invalid_params));
  invalid_params = upload_params;
  invalid_params.srcPtr.ysize = 0;
  invalid_params.srcPos.y = SIZE_MAX;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.set_executable_copy_3d(executable_, upload, &invalid_params));
  hipMemcpy3DParms queried_params = {};
  ASSERT_EQ(hipSuccess, api_.get_copy(upload, &queried_params));
  EXPECT_EQ(queried_params.srcPtr.ptr, source_.data());
  EXPECT_EQ(queried_params.dstPtr.pitch, pitch);
  ReleaseStreamGate();
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  EXPECT_EQ(destination_, expected(source_));
  EXPECT_EQ(observed_, (std::array<uint8_t, 64>{}));

  // Replacing the entire private graph releases its node references while
  // the pending callbacks and GPU copies still own the previous storage.
  HoldStreamAtGate();
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(hipSuccess, api_.launch(executable_, stream_));
  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult result;
  ASSERT_EQ(hipSuccess,
            api_.update_executable(executable_, graph_, &error_node, &result));
  EXPECT_EQ(result, hipGraphExecUpdateSuccess);
  ReleaseStreamGate();
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  EXPECT_EQ(observed_, expected(replacement_));
  destination_.fill(0);
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, expected(source_));
}

void HipGraphMemoryApiTest::Increment(void* user_data) {
  ++*static_cast<int*>(user_data);
}

void HipGraphMemoryApiTest::ObserveDestination(void* user_data) {
  auto* test = static_cast<HipGraphMemoryApiTest*>(user_data);
  test->observed_ = test->destination_;
}

void HipGraphMemoryApiTest::ReplaceSource(void* user_data) {
  auto* test = static_cast<HipGraphMemoryApiTest*>(user_data);
  test->source_ = test->replacement_;
}

void HipGraphMemoryApiTest::CheckPitchedCopyClone(bool implicit_slice_height,
                                                  bool registered_host,
                                                  bool capture) {
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size()));
  void* source = source_.data();
  void* destination = destination_.data();
  if (registered_host) {
    ASSERT_EQ(hipSuccess,
              api_.allocate_host(&host_buffers_[0], source_.size(), 0));
    ASSERT_EQ(hipSuccess,
              api_.allocate_host(&host_buffers_[1], source_.size(), 0));
    source = host_buffers_[0];
    destination = host_buffers_[1];
    std::memcpy(source, source_.data(), source_.size());
    std::memset(destination, 0, source_.size());
  }
  const size_t y_origin = implicit_slice_height ? 1 : 0;
  const size_t slice_height = implicit_slice_height ? 0 : 3;
  const hipMemcpy3DParms upload = {
      .srcArray = nullptr,
      .srcPos = {1, y_origin, 0},
      .srcPtr = {source, 8, 8, slice_height},
      .dstArray = nullptr,
      .dstPos = {1, y_origin, 0},
      .dstPtr = {device_, 8, 8, slice_height},
      .extent = {4, 2, 2},
      .kind = hipMemcpyHostToDevice,
  };
  hipMemcpy3DParms download = upload;
  download.srcPtr.ptr = device_;
  download.dstPtr.ptr = destination;
  download.kind = hipMemcpyDeviceToHost;
  if (capture) {
    ASSERT_EQ(hipSuccess,
              api_.begin_capture_to_graph(stream_, graph_, nullptr, nullptr, 0,
                                          hipStreamCaptureModeGlobal));
    EXPECT_EQ(hipSuccess, api_.copy_3d_async(&upload, stream_));
    EXPECT_EQ(hipSuccess, api_.copy_3d_async(&download, stream_));
    hipGraph_t captured_graph = nullptr;
    ASSERT_EQ(hipSuccess, api_.end_capture(stream_, &captured_graph));
    ASSERT_EQ(graph_, captured_graph);
  } else {
    hipGraphNode_t upload_node = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.add_copy(&upload_node, graph_, nullptr, 0, &upload));
    hipGraphNode_t download_node = nullptr;
    ASSERT_EQ(hipSuccess, api_.add_copy(&download_node, graph_, &upload_node, 1,
                                        &download));
  }
  ASSERT_EQ(hipSuccess, api_.graph_clone(&clone_, graph_));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph_));
  graph_ = nullptr;
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, clone_, 0));
  for (int replay = 0; replay < 2; ++replay) {
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    for (size_t i = 0; i < source_.size(); ++i) {
      const size_t row_origin = y_origin * 8 + 1;
      const size_t slice_offset = i % 24;
      const bool copied =
          i < 48 &&
          ((slice_offset >= row_origin && slice_offset < row_origin + 4) ||
           (slice_offset >= row_origin + 8 && slice_offset < row_origin + 12));
      EXPECT_EQ(static_cast<uint8_t*>(destination)[i], copied ? source_[i] : 0)
          << i;
    }
    source_ = replacement_;
    if (registered_host) {
      std::memcpy(source, source_.data(), source_.size());
    }
  }
}

}  // namespace hrx::hip::testing
