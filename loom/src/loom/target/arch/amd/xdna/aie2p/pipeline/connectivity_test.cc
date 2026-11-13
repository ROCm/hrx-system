// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/connectivity.h"

#include <initializer_list>
#include <map>
#include <tuple>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/pass/types.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"
#include "loom/testing/diagnostic_matchers.h"

namespace {

using Endpoint = loom_aie2p_native_connection_endpoint_t;

class ConnectivityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    pass_.arena = &arena_;
    pass_.diagnostic_emitter = diagnostics_.emitter();
    context_.pass = &pass_;
    context_.family = loom_xdna_npu2_array_family();
    tiles_.resize(context_.family->column_count * context_.family->row_count);
    for (uint16_t column = 0; column < context_.family->column_count;
         ++column) {
      for (uint16_t row = 0; row < context_.family->row_count; ++row) {
        auto& tile = tiles_[column * context_.family->row_count + row];
        tile.coordinate = {column, row};
        tile.facts =
            loom_xdna_array_tile_facts(context_.family, tile.coordinate);
      }
    }
    context_.tiles = tiles_.data();
    IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
        context_.family, 256, &arena_, &context_.routing));
    routes_.resize(256);
    context_.routes = routes_.data();
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Route(Endpoint source, Endpoint destination,
             uint8_t packet = UINT8_MAX) {
    auto& tree = trees_[{source.coordinate.column, source.coordinate.row,
                         source.port, source.channel}];
    const size_t first = context_.routing.route_count;
    ASSERT_TRUE(loom_aie2p_array_route_stream(
        &context_.routing, next_label_++, &tree, source.coordinate, source.port,
        source.channel, destination.coordinate, destination.port,
        destination.channel, packet != UINT8_MAX));
    loom_aie2p_native_route_label(&context_, first, packet, UINT32_MAX);
  }

  void Build() {
    bool valid = false;
    IREE_ASSERT_OK(
        loom_aie2p_native_switches_build(&context_, nullptr, &valid));
    ASSERT_TRUE(valid);
    ASSERT_TRUE(diagnostics_.emissions.empty());
    IREE_ASSERT_OK(loom_aie2p_native_connectivity_allocate(
        context_.family, context_.switches, context_.routing.routes,
        context_.routing.route_count, &arena_, &connectivity_));
  }

  loom_aie2p_native_connectivity_result_t Check(
      Endpoint source, std::initializer_list<Endpoint> destinations,
      uint8_t packet = UINT8_MAX, bool retain_header = false) {
    return loom_aie2p_native_connectivity_check(
        connectivity_, source, packet, destinations.begin(),
        destinations.size(), retain_header);
  }

  // Arena and diagnostic services used by real routing and switch admission.
  iree_arena_block_pool_t pool_ = {};
  // All selected plans and the audit share the test's allocation lifetime.
  iree_arena_allocator_t arena_ = {};
  // Resource rejection remains visible when building the physical network.
  loom::testing::DiagnosticEmissionCapture diagnostics_;
  // Compiler services required by native selection.
  loom_pass_t pass_ = {};
  // Valid physical plans consumed by the production switch emitter.
  loom_aie2p_native_context_t context_ = {};
  // Physical tile owners in column-major order.
  std::vector<loom_aie2p_native_tile_t> tiles_;
  // Realized edges labeled with their selected packet identity.
  std::vector<loom_aie2p_native_route_t> routes_;
  // Stable route ownership per physical sender.
  std::map<std::tuple<uint16_t, uint16_t, loom_xdna_stream_port_t, uint8_t>,
           loom_aie2p_array_route_tree_t>
      trees_;
  // Destination labels carry no route-tree ownership.
  uint32_t next_label_ = 0;
  // Independent view of the completed physical connections.
  loom_aie2p_native_connectivity_t* connectivity_ = nullptr;
};

TEST_F(ConnectivityTest, PayloadAndCompletionReachOnlyTheirIntendedConsumers) {
  const Endpoint ingress = {{0, 0}, LOOM_XDNA_STREAM_PORT_DMA, 0};
  const Endpoint payload = {{0, 3}, LOOM_XDNA_STREAM_PORT_DMA, 0};
  const Endpoint completion = {{0, 0}, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0};
  const Endpoint consumer = {{0, 3}, LOOM_XDNA_STREAM_PORT_CORE, 0};
  ASSERT_NO_FATAL_FAILURE(Route(ingress, payload));
  ASSERT_NO_FATAL_FAILURE(Route(completion, consumer, 1));
  ASSERT_NO_FATAL_FAILURE(Build());
  EXPECT_EQ(Check(ingress, {payload}).failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE);
  EXPECT_EQ(Check(completion, {consumer}, 1, true).failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE);
}

TEST_F(ConnectivityTest,
       IndependentIntentDetectsTheIncidentFanoutAndLostReply) {
  const Endpoint ingress = {{0, 0}, LOOM_XDNA_STREAM_PORT_DMA, 0};
  const Endpoint payload = {{0, 3}, LOOM_XDNA_STREAM_PORT_DMA, 0};
  const Endpoint completion = {{0, 0}, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0};
  const Endpoint consumer = {{0, 3}, LOOM_XDNA_STREAM_PORT_CORE, 0};
  // This is a locally legal circuit multicast. Its actual consumers differ
  // from the request: the payload reaches the completion input as well.
  ASSERT_NO_FATAL_FAILURE(Route(ingress, payload));
  ASSERT_NO_FATAL_FAILURE(Route(ingress, consumer));
  ASSERT_NO_FATAL_FAILURE(Build());
  const auto fanout = Check(ingress, {payload});
  EXPECT_EQ(fanout.failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_UNEXPECTED_DESTINATION);
  EXPECT_EQ(fanout.endpoint.port, LOOM_XDNA_STREAM_PORT_CORE);
  const auto reply = Check(completion, {consumer}, 1, true);
  EXPECT_EQ(reply.failure, LOOM_AIE2P_NATIVE_CONNECTIVITY_DEAD_END);
  EXPECT_EQ(reply.endpoint.coordinate.row, 0);
  EXPECT_EQ(reply.endpoint.port, LOOM_XDNA_STREAM_PORT_TILE_CONTROL);
}

TEST_F(ConnectivityTest,
       PacketClassesKeepIndependentMulticastSetsAndDropHeaders) {
  const Endpoint source = {{0, 1}, LOOM_XDNA_STREAM_PORT_DMA, 0};
  const Endpoint first = {{1, 3}, LOOM_XDNA_STREAM_PORT_DMA, 0};
  const Endpoint second = {{2, 3}, LOOM_XDNA_STREAM_PORT_DMA, 0};
  const Endpoint third = {{3, 3}, LOOM_XDNA_STREAM_PORT_DMA, 0};
  for (uint8_t packet : {0, 1, 2, 3}) {
    ASSERT_NO_FATAL_FAILURE(Route(source, first, packet));
    ASSERT_NO_FATAL_FAILURE(Route(source, second, packet));
  }
  ASSERT_NO_FATAL_FAILURE(Route(source, third, 4));
  ASSERT_NO_FATAL_FAILURE(Build());
  for (uint8_t packet : {0, 1, 2, 3}) {
    EXPECT_EQ(Check(source, {first, second}, packet).failure,
              LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE);
  }
  EXPECT_EQ(Check(source, {third}, 4).failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE);
  EXPECT_EQ(Check(source, {first}, 0).failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_UNEXPECTED_DESTINATION);
  EXPECT_EQ(Check(source, {first, second, third}, 0).failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_MISSING_DESTINATION);
  EXPECT_EQ(Check(source, {third}, 5).failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_DEAD_END);
  EXPECT_EQ(Check(source, {third}, UINT8_MAX).failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_HEADER);
  EXPECT_EQ(Check(source, {third}, 4, true).failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_HEADER);
}

TEST_F(ConnectivityTest, CrossColumnShimEgressAndIngressKeepEngineIdentity) {
  const Endpoint source = {{0, 3}, LOOM_XDNA_STREAM_PORT_DMA, 1};
  const Endpoint shim = {{2, 0}, LOOM_XDNA_STREAM_PORT_DMA, 1};
  const Endpoint receiver = {{2, 4}, LOOM_XDNA_STREAM_PORT_DMA, 1};
  ASSERT_NO_FATAL_FAILURE(Route(source, shim, 3));
  ASSERT_NO_FATAL_FAILURE(Route(shim, receiver, 7));
  ASSERT_NO_FATAL_FAILURE(Build());
  EXPECT_EQ(Check(source, {shim}, 3).failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE);
  EXPECT_EQ(Check(shim, {receiver}, 7).failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE);
  EXPECT_EQ(Check(source, {{{2, 0}, LOOM_XDNA_STREAM_PORT_DMA, 0}}, 3).failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_UNEXPECTED_DESTINATION);
}

TEST_F(ConnectivityTest, PrivateCircuitPreservesControlFraming) {
  const Endpoint source = {{0, 3}, LOOM_XDNA_STREAM_PORT_CORE, 0};
  const Endpoint destination = {{0, 1}, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0};
  ASSERT_NO_FATAL_FAILURE(Route(source, destination));
  ASSERT_NO_FATAL_FAILURE(Build());
  EXPECT_EQ(Check(source, {destination}, 2, true).failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE);
  EXPECT_EQ(Check(source, {destination}, 2, false).failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_HEADER);
}

TEST_F(ConnectivityTest, ConfigurationCompletionReachesExternalShimPort) {
  const Endpoint source = {{0, 0}, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0};
  const Endpoint destination = {{0, 0}, LOOM_XDNA_STREAM_PORT_SOUTH, 0};
  ASSERT_NO_FATAL_FAILURE(Route(source, destination, 0));
  ASSERT_NO_FATAL_FAILURE(Build());
  EXPECT_EQ(Check(source, {destination}, 0, true).failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE);
}

TEST_F(ConnectivityTest, CompleteAuditGroupsIntentAndReleasesScratch) {
  const Endpoint source = {{0, 0}, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0};
  const Endpoint first = {{0, 2}, LOOM_XDNA_STREAM_PORT_CORE, 0};
  const Endpoint second = {{0, 3}, LOOM_XDNA_STREAM_PORT_CORE, 0};
  const Endpoint third = {{0, 4}, LOOM_XDNA_STREAM_PORT_CORE, 0};
  loom_aie2p_native_control_connection_t connections[] = {
      {source, first, 1}, {source, second, 1}, {source, third, 2}};
  for (const auto& connection : connections) {
    ASSERT_NO_FATAL_FAILURE(
        Route(connection.source, connection.destination, connection.packet));
  }
  ASSERT_NO_FATAL_FAILURE(Build());
  context_.control_connections = connections;
  context_.control_connection_count = IREE_ARRAYSIZE(connections);
  const auto retained_bytes = arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_aie2p_native_connections_verify(&context_));
  EXPECT_EQ(arena_.used_allocation_size, retained_bytes);
}

TEST_F(ConnectivityTest, PacketCycleIsRejectedWithoutDeviceExecution) {
  const Endpoint source = {{0, 2}, LOOM_XDNA_STREAM_PORT_CORE, 0};
  ASSERT_NO_FATAL_FAILURE(
      Route(source, {{0, 2}, LOOM_XDNA_STREAM_PORT_NORTH, 0}, 1));
  ASSERT_NO_FATAL_FAILURE(Route({{0, 3}, LOOM_XDNA_STREAM_PORT_SOUTH, 0},
                                {{0, 3}, LOOM_XDNA_STREAM_PORT_SOUTH, 0}, 1));
  ASSERT_NO_FATAL_FAILURE(Route({{0, 2}, LOOM_XDNA_STREAM_PORT_NORTH, 0},
                                {{0, 2}, LOOM_XDNA_STREAM_PORT_NORTH, 0}, 1));
  ASSERT_NO_FATAL_FAILURE(Build());
  EXPECT_EQ(
      Check(source, {{{0, 3}, LOOM_XDNA_STREAM_PORT_CORE, 0}}, 1, true).failure,
      LOOM_AIE2P_NATIVE_CONNECTIVITY_REPEATED_ARRIVAL);
}

TEST_F(ConnectivityTest, LocalFifoCarriesThePacketToItsReceiver) {
  const Endpoint source = {{0, 2}, LOOM_XDNA_STREAM_PORT_CORE, 0};
  const Endpoint fifo = {{0, 2}, LOOM_XDNA_STREAM_PORT_FIFO, 0};
  const Endpoint destination = {{0, 2}, LOOM_XDNA_STREAM_PORT_DMA, 0};
  ASSERT_NO_FATAL_FAILURE(Route(source, fifo, 1));
  ASSERT_NO_FATAL_FAILURE(Route(fifo, destination, 1));
  ASSERT_NO_FATAL_FAILURE(Build());
  EXPECT_EQ(Check(source, {destination}, 1).failure,
            LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE);
}

TEST_F(ConnectivityTest, AnOutputBeyondTheArrayDoesNotCountAsDelivery) {
  const Endpoint source = {{0, 5}, LOOM_XDNA_STREAM_PORT_CORE, 0};
  const Endpoint north = {{0, 5}, LOOM_XDNA_STREAM_PORT_NORTH, 0};
  ASSERT_NO_FATAL_FAILURE(Route(source, north));
  ASSERT_NO_FATAL_FAILURE(Build());
  const auto result = Check(source, {{{0, 5}, LOOM_XDNA_STREAM_PORT_DMA, 0}});
  EXPECT_EQ(result.failure, LOOM_AIE2P_NATIVE_CONNECTIVITY_ARRAY_BOUNDARY);
  EXPECT_EQ(result.endpoint.coordinate.row, context_.family->row_count);
}

}  // namespace
