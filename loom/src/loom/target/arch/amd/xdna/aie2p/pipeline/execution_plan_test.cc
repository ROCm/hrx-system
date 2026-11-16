// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/execution_plan.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/pass/types.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/code_transport.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/connectivity.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/payload.h"

namespace {

class ExecutionPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &source_arena_);
    iree_arena_initialize(&blocks_, &retained_arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&retained_arena_);
    iree_arena_deinitialize(&source_arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }
  // Blocks shared by two independently retired compilation owners.
  iree_arena_block_pool_t blocks_ = {};
  // Original source selection and compilation.
  iree_arena_allocator_t source_arena_ = {};
  // Independent projected compilation.
  iree_arena_allocator_t retained_arena_ = {};
};

TEST_F(ExecutionPlanTest, SourceRetirementPreservesRoutesAndPhysicalOwnership) {
  const auto* family = loom_xdna_npu2_array_family();
  const size_t tile_count = family->column_count * family->row_count;
  std::vector<loom_aie2p_native_tile_t> tiles(tile_count);
  for (size_t i = 0; i < tile_count; ++i) {
    tiles[i].coordinate = {static_cast<uint16_t>(i / family->row_count),
                           static_cast<uint16_t>(i % family->row_count)};
    tiles[i].facts = loom_xdna_array_tile_facts(family, tiles[i].coordinate);
  }
  loom_pass_t pass = {.arena = &source_arena_};
  loom_aie2p_native_context_t context = {
      .pass = &pass,
      .family = family,
      .tiles = tiles.data(),
  };
  IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
      family, 64, &source_arena_, &context.routing));
  std::vector<loom_aie2p_native_control_source_t> sources(tile_count * 2);
  std::vector<uint32_t> destinations(tile_count * 2, UINT32_MAX);
  std::vector<uint8_t> command_packets(tile_count, UINT8_MAX);
  command_packets[1] = 3;
  std::vector<loom_aie2p_native_control_path_t> control_paths(
      family->column_count);
  control_paths[0].packets = 1u << command_packets[1];
  loom_aie2p_native_control_connection_t connection = {
      .source = {{0, 2}, LOOM_XDNA_STREAM_PORT_CORE, 0},
      .destination = {{0, 1}, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0},
      .packet = 3,
  };
  sources[4].destination = 3;
  destinations[3] = 4;
  ASSERT_TRUE(loom_aie2p_array_route_stream(
      &context.routing, 0, &sources[4].tree, connection.source.coordinate,
      connection.source.port, 0, connection.destination.coordinate,
      connection.destination.port, 0, true));
  const size_t control_route_count = context.routing.route_count;
  loom_aie2p_native_payload_endpoint_t receivers[] = {
      {&tiles[2], 1}, {&tiles[family->row_count + 2], 0}};
  loom_aie2p_native_payload_source_t payload_source = {
      .endpoint = {&tiles[1], 0},
      .class_count = 1,
      .framed = true,
  };
  loom_aie2p_native_payload_class_t payload = {
      .source = &payload_source,
      .destinations = receivers,
      .destination_count = IREE_ARRAYSIZE(receivers),
      .packet = 9,
      .routed = true,
  };
  payload_source.classes = &payload;
  for (const auto& receiver : receivers) {
    ASSERT_TRUE(loom_aie2p_array_route_workers(
        &context.routing, 1, &payload_source.tree,
        payload_source.endpoint.tile->coordinate, 0, receiver.tile->coordinate,
        receiver.engine, true));
  }
  context.payload_sources = &payload_source;
  context.control_connections = &connection;
  context.control_connection_count = 1;
  std::vector<loom_aie2p_native_route_t> routes(context.routing.route_count);
  std::vector<loom_aie2p_native_route_lifetime_t> route_lifetimes = {
      {}, {.lifetime = 1, .next = 0}};
  context.routes = routes.data();
  context.control_sources = sources.data();
  context.control_destination_sources = destinations.data();
  loom_aie2p_native_route_label(&context, 0, connection.packet, true, 4);
  loom_aie2p_native_route_label(&context, control_route_count, payload.packet,
                                false, UINT32_MAX);
  for (size_t i = control_route_count; i < routes.size(); ++i) {
    routes[i].lifetime_head = 1;
  }
  const loom_aie2p_native_switch_input_t input = {
      .routing = &context.routing,
      .routes = context.routes,
      .control_sources = context.control_sources,
      .control_destination_sources = context.control_destination_sources,
  };
  iree_string_view_t requirement = iree_string_view_empty();
  IREE_ASSERT_OK(loom_aie2p_native_switches_build(
      &input, &source_arena_, &context.switches, &requirement));
  ASSERT_TRUE(iree_string_view_is_empty(requirement));
  IREE_ASSERT_OK(loom_aie2p_native_connections_build(&context));
  IREE_ASSERT_OK(loom_aie2p_native_connections_verify(
      &context.routing, context.switches, &context.connections,
      &source_arena_));

  constexpr auto none = IREE_HOST_SIZE_MAX;
  const loom_pipeline_lifetime_t nodes[] = {
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none, none},
  };
  const iree_host_size_t owners[] = {1};
  loom_pipeline_lifetimes_t lifetimes = {
      .nodes = nodes,
      .count = IREE_ARRAYSIZE(nodes),
      .owners = owners,
  };
  IREE_ASSERT_OK(loom_pipeline_lifetimes_index(&source_arena_, &lifetimes));
  loom_pipeline_entry_scope_t* scope = nullptr;
  IREE_ASSERT_OK(loom_pipeline_entry_scope_allocate(
      &loom_aie2p_execution_plan_type, &lifetimes, 1, &source_arena_, &scope));
  std::vector<loom_aie2p_execution_tile_t> occupancy(tile_count);
  occupancy[1] = {
      .memory_length = 128,
      .descriptor_counts = {1, 0},
      .mm2s_count = 1,
  };
  occupancy[2] = {
      .memory_length = 2560,
      .lock_count = 3,
      .descriptor_counts = {2, 0},
      .mm2s_count = 1,
      .s2mm_count = 1,
  };
  const loom_aie2p_execution_plan_t selected = {
      .tiles = occupancy.data(),
      .routing = context.routing,
      .routes = routes.data(),
      .route_lifetimes = route_lifetimes.data(),
      .route_lifetime_count = route_lifetimes.size(),
      .control_sources = sources.data(),
      .control_destination_sources = destinations.data(),
      .column_command_packets = command_packets.data(),
      .control_paths = control_paths.data(),
      .switches = context.switches,
      .connections = context.connections,
  };
  auto* original = reinterpret_cast<loom_aie2p_execution_plan_t*>(scope);
  IREE_ASSERT_OK(loom_aie2p_execution_plan_copy_resources(
      &selected, 0, &source_arena_, original));
  const loom_pipeline_entry_scope_t* cloned = nullptr;
  IREE_ASSERT_OK(
      loom_pipeline_entry_scope_clone(scope, &retained_arena_, &cloned));
  const auto* retained =
      reinterpret_cast<const loom_aie2p_execution_plan_t*>(cloned);
  EXPECT_EQ(cloned->type, &loom_aie2p_execution_plan_type);
  EXPECT_NE(retained->routing.routes, original->routing.routes);
  EXPECT_NE(retained->switches->banks[0].words,
            original->switches->banks[0].words);
  EXPECT_NE(retained->connections.groups, original->connections.groups);
  EXPECT_NE(retained->connections.destinations,
            original->connections.destinations);
  EXPECT_NE(retained->column_command_packets, original->column_command_packets);
  EXPECT_NE(retained->route_lifetimes, original->route_lifetimes);
  iree_arena_reset(&source_arena_);
  tiles.clear();
  routes.clear();
  route_lifetimes.clear();
  sources.clear();
  destinations.clear();
  command_packets.clear();
  occupancy.clear();

  EXPECT_EQ(retained->base.lifetimes.owners[0], 1u);
  ASSERT_EQ(retained->route_lifetime_count, 2u);
  EXPECT_EQ(retained->route_lifetimes[1].lifetime, 1u);
  EXPECT_EQ(retained->route_lifetimes[1].next, 0u);
  EXPECT_EQ(retained->tiles[2].memory_length, 2560u);
  EXPECT_EQ(retained->tiles[2].lock_count, 3u);
  EXPECT_EQ(retained->tiles[2].descriptor_counts[0], 2u);
  EXPECT_EQ(retained->tiles[2].mm2s_count, 1u);
  EXPECT_EQ(retained->tiles[2].s2mm_count, 1u);
  EXPECT_EQ(retained->control_sources[4].destination, 3u);
  EXPECT_EQ(retained->control_sources[4].packet_mask, 1u << connection.packet);
  EXPECT_EQ(retained->control_sources[4].register_packet_mask,
            1u << connection.packet);
  EXPECT_EQ(retained->control_destination_sources[3], 4u);
  EXPECT_EQ(retained->column_command_packets[1], 3u);
  for (size_t i = 0; i < retained->routing.route_count; ++i) {
    EXPECT_EQ(retained->routes[i].edge, &retained->routing.routes[i]);
    EXPECT_EQ(retained->routes[i].register_command, i < control_route_count);
    EXPECT_EQ(retained->routes[i].control_source,
              i < control_route_count ? 4u : UINT32_MAX);
    EXPECT_EQ(retained->routes[i].lifetime_head,
              i < control_route_count ? 0u : 1u);
  }
  ASSERT_EQ(retained->connections.group_count, 2u);
  ASSERT_EQ(retained->connections.destination_count, 3u);
  const auto& payload_intent = retained->connections.groups[0];
  EXPECT_FALSE(payload_intent.retain_header);
  EXPECT_EQ(payload_intent.packet, 9u);
  EXPECT_EQ(payload_intent.destination_count, 2u);
  const auto& intended = retained->connections.groups[1];
  EXPECT_TRUE(intended.retain_header);
  EXPECT_EQ(intended.destination_count, 1u);
  IREE_ASSERT_OK(loom_aie2p_native_connections_verify(
      &retained->routing, retained->switches, &retained->connections,
      &retained_arena_));

  // Late code delivery consumes the retained physical ownership after source
  // retirement. Payload multicast and the original control delivery survive.
  loom_aie2p_code_transport_t code_transport;
  IREE_ASSERT_OK(loom_aie2p_code_transport_initialize(
      retained, 4, &retained_arena_, &code_transport));
  loom_aie2p_code_transport_binding_t binding;
  for (uint16_t row = 2; row < 6; ++row) {
    loom_aie2p_code_transport_select(&code_transport, {0, row}, 2, &binding,
                                     &requirement);
    ASSERT_TRUE(iree_string_view_is_empty(requirement));
    loom_aie2p_code_transport_bind(&code_transport, 2, &binding, &requirement);
    ASSERT_TRUE(iree_string_view_is_empty(requirement));
    EXPECT_EQ(binding.engine, 1u);
    EXPECT_EQ(binding.first_descriptor, 24u + 2u * (row - 2u));
    EXPECT_EQ(binding.descriptor_count, 2u);
    EXPECT_EQ(binding.queue_packet, 3u);
    EXPECT_EQ(binding.code_packet, row - 2u);
    EXPECT_EQ(binding.reply_packet, 0u);
  }
  EXPECT_EQ(code_transport.plan.tiles[1].memory_length, 128u);
  EXPECT_EQ(code_transport.plan.tiles[1].mm2s_count, 2u);
  EXPECT_EQ(code_transport.plan.tiles[1].descriptor_counts[1], 8u);
  EXPECT_EQ(code_transport.plan.connections.group_count, 14u);
  EXPECT_EQ(code_transport.plan.connections.destination_count, 15u);
  IREE_ASSERT_OK(loom_aie2p_code_transport_finish(
      &code_transport, &retained_arena_, &requirement));
  ASSERT_TRUE(iree_string_view_is_empty(requirement));
  EXPECT_EQ(retained->tiles[1].mm2s_count, 1u);
  EXPECT_EQ(retained->control_sources[4].packet_mask, 1u << connection.packet);

  // Descriptor capacity can end a shared sender before its task queue fills.
  // The next engine uses the other bank but shares the delivery carrier and
  // destination namespace, so descriptor demand consumes no additional lane.
  IREE_ASSERT_OK(loom_aie2p_code_transport_initialize(
      retained, 4, &retained_arena_, &code_transport));
  for (uint16_t row = 2; row < 6; ++row) {
    loom_aie2p_code_transport_select(&code_transport, {0, row}, 7, &binding,
                                     &requirement);
    ASSERT_TRUE(iree_string_view_is_empty(requirement));
    loom_aie2p_code_transport_bind(&code_transport, 7, &binding, &requirement);
    ASSERT_TRUE(iree_string_view_is_empty(requirement));
    EXPECT_EQ(binding.engine, row == 5 ? 2u : 1u);
    EXPECT_EQ(binding.first_descriptor, row == 5 ? 1u : 24u + 7u * (row - 2u));
    EXPECT_EQ(binding.code_packet, row - 2u);
    EXPECT_EQ(code_transport.plan.routing.link_channels
                  .northbound[family->column_count],
              3u);
  }
  EXPECT_EQ(code_transport.plan.tiles[1].mm2s_count, 3u);
  EXPECT_EQ(code_transport.plan.tiles[1].descriptor_counts[0], 8u);
  EXPECT_EQ(code_transport.plan.tiles[1].descriptor_counts[1], 21u);
  IREE_ASSERT_OK(loom_aie2p_code_transport_finish(
      &code_transport, &retained_arena_, &requirement));
  ASSERT_TRUE(iree_string_view_is_empty(requirement));

  // Native placement can discard a selection or use fewer descriptors. Until
  // binding, neither a failed resource check nor an unused loader consumes an
  // engine, packet identity, route, or queue slot from the next worker.
  IREE_ASSERT_OK(loom_aie2p_code_transport_initialize(
      retained, 4, &retained_arena_, &code_transport));
  loom_aie2p_code_transport_select(&code_transport, {0, 2}, 25, &binding,
                                   &requirement);
  ASSERT_FALSE(iree_string_view_is_empty(requirement));
  loom_aie2p_code_transport_select(&code_transport, {0, 2}, 7, &binding,
                                   &requirement);
  ASSERT_TRUE(iree_string_view_is_empty(requirement));
  loom_aie2p_code_transport_bind(&code_transport, 0, &binding, &requirement);
  ASSERT_TRUE(iree_string_view_is_empty(requirement));
  EXPECT_EQ(binding.descriptor_count, 0u);
  EXPECT_EQ(code_transport.plan.tiles[1].mm2s_count, 1u);
  EXPECT_EQ(code_transport.plan.tiles[1].descriptor_counts[1], 0u);
  EXPECT_EQ(code_transport.plan.routing.route_count,
            retained->routing.route_count);
  EXPECT_EQ(code_transport.plan.connections.group_count, 2u);
  for (uint16_t row = 2; row < 6; ++row) {
    loom_aie2p_code_transport_select(&code_transport, {0, row}, 7, &binding,
                                     &requirement);
    ASSERT_TRUE(iree_string_view_is_empty(requirement));
    EXPECT_EQ(binding.engine, 1u);
    EXPECT_EQ(binding.first_descriptor, 24u + 5u * (row - 2u));
    EXPECT_EQ(binding.queue_packet, 3u);
    EXPECT_EQ(binding.reply_packet, 0u);
    loom_aie2p_code_transport_bind(&code_transport, 5, &binding, &requirement);
    ASSERT_TRUE(iree_string_view_is_empty(requirement));
    EXPECT_EQ(binding.descriptor_count, 5u);
  }
  EXPECT_EQ(code_transport.plan.tiles[1].mm2s_count, 2u);
  EXPECT_EQ(code_transport.plan.tiles[1].descriptor_counts[0], 1u);
  EXPECT_EQ(code_transport.plan.tiles[1].descriptor_counts[1], 20u);
  EXPECT_EQ(code_transport.plan.connections.group_count, 14u);
  IREE_ASSERT_OK(loom_aie2p_code_transport_finish(
      &code_transport, &retained_arena_, &requirement));
  ASSERT_TRUE(iree_string_view_is_empty(requirement));

  loom_aie2p_array_route_builder_t extended;
  IREE_ASSERT_OK(loom_aie2p_array_route_builder_clone(
      &retained->routing, retained->routing.route_count + 8, &retained_arena_,
      &extended));
  auto sender = retained->control_sources[4].tree;
  EXPECT_TRUE(loom_aie2p_array_route_stream(
      &extended, 1, &sender, intended.source.coordinate, intended.source.port,
      0, {0, 0}, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0, true));
  EXPECT_GT(extended.route_count, retained->routing.route_count);

  // A later control destination changes the private circuit into a multicast
  // route. Rebuild and audit the complete switch selection using only retained
  // facts, including the untouched payload destinations.
  std::vector<loom_aie2p_native_route_t> extended_routes(extended.route_count);
  for (size_t i = 0; i < extended.route_count; ++i) {
    extended_routes[i] =
        i < retained->routing.route_count
            ? retained->routes[i]
            : loom_aie2p_native_route_t{nullptr, intended.packet, true, 4};
    extended_routes[i].edge = &extended.routes[i];
  }
  std::vector<loom_aie2p_native_control_source_t> extended_sources(
      retained->control_sources, retained->control_sources + tile_count * 2);
  extended_sources[4].destination = UINT32_MAX;
  const loom_aie2p_native_switch_input_t extended_input = {
      .routing = &extended,
      .routes = extended_routes.data(),
      .control_sources = extended_sources.data(),
      .control_destination_sources = retained->control_destination_sources,
  };
  loom_aie2p_native_switch_t* extended_switches = nullptr;
  IREE_ASSERT_OK(loom_aie2p_native_switches_build(
      &extended_input, &retained_arena_, &extended_switches, &requirement));
  ASSERT_TRUE(iree_string_view_is_empty(requirement));
  std::vector<loom_aie2p_native_connection_t> extended_groups(
      retained->connections.groups,
      retained->connections.groups + retained->connections.group_count);
  std::vector<loom_aie2p_native_connection_endpoint_t> extended_destinations(
      retained->connections.destinations,
      retained->connections.destinations +
          retained->connections.destination_count);
  ++extended_groups[1].destination_count;
  extended_destinations.push_back(
      {{0, 0}, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0});
  const loom_aie2p_native_connections_t extended_connections = {
      .groups = extended_groups.data(),
      .group_count = extended_groups.size(),
      .destinations = extended_destinations.data(),
      .destination_count = extended_destinations.size(),
  };
  IREE_ASSERT_OK(loom_aie2p_native_connections_verify(
      &extended, extended_switches, &extended_connections, &retained_arena_));
}

TEST_F(ExecutionPlanTest, CodeRequestsShareCommandsOnASaturatedColumn) {
  const auto* family = loom_xdna_npu2_array_family();
  const size_t tile_count = family->column_count * family->row_count;
  std::vector<loom_aie2p_native_tile_t> tiles(tile_count);
  std::vector<loom_aie2p_native_control_source_t> sources(tile_count * 2);
  std::vector<uint32_t> destinations(tile_count * 2, UINT32_MAX);
  std::vector<uint8_t> command_packets(tile_count, UINT8_MAX);
  std::vector<loom_aie2p_native_control_path_t> paths(family->column_count);
  std::vector<loom_aie2p_native_route_t> routes(128);
  std::vector<loom_aie2p_native_control_connection_t> controls(16);
  for (size_t i = 0; i < tile_count; ++i) {
    tiles[i].coordinate = {static_cast<uint16_t>(i / family->row_count),
                           static_cast<uint16_t>(i % family->row_count)};
    tiles[i].facts = loom_xdna_array_tile_facts(family, tiles[i].coordinate);
  }
  loom_pass_t pass = {.arena = &source_arena_};
  loom_aie2p_native_context_t context = {
      .pass = &pass,
      .family = family,
      .tiles = tiles.data(),
      .routes = routes.data(),
      .control_sources = sources.data(),
      .control_destination_sources = destinations.data(),
      .column_command_packets = command_packets.data(),
      .control_paths = paths.data(),
      .control_connections = controls.data(),
  };
  IREE_ASSERT_OK(loom_aie2p_array_route_builder_initialize(
      family, routes.size(), &source_arena_, &context.routing));
  uint32_t next_source = 0;
  auto command = [&](uint16_t row, uint16_t destination, uint8_t packet) {
    ASSERT_TRUE(loom_aie2p_native_packet_route(
        &context, &next_source, {0, row}, LOOM_XDNA_STREAM_PORT_CORE,
        {0, destination}, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, packet, nullptr));
    command_packets[destination] = packet;
    paths[0].packets |= UINT32_C(1) << packet;
  };
  command(2, 2, 2);
  command(2, 0, 1);
  command(3, 3, 3);
  command(4, 1, 0);
  command(4, 0, 1);
  command(4, 4, 4);
  command(5, 5, 5);
  // A private collective can reuse another core's command ID. Late code
  // loading must preserve that meaning on its original source stream.
  ASSERT_TRUE(loom_aie2p_native_packet_route(
      &context, &next_source, {0, 3}, LOOM_XDNA_STREAM_PORT_CORE, {0, 4},
      LOOM_XDNA_STREAM_PORT_CORE, 0, nullptr));
  ASSERT_EQ(paths[0].source, 8u);
  ASSERT_EQ(paths[0].count, 4u);
  const uint8_t carrier_arrival =
      context.routing.routes[paths[0].first + 2].source_channel;

  // Two independent results and two command sources exhaust the southbound
  // links. Payloads keep circuit framing and cannot carry load requests.
  loom_aie2p_native_payload_source_t payload_sources[2] = {};
  loom_aie2p_native_payload_class_t payloads[2] = {};
  loom_aie2p_native_payload_endpoint_t receivers[2] = {};
  for (uint16_t i = 0; i < 2; ++i) {
    auto& payload_source = payload_sources[i];
    payload_source.endpoint = {&tiles[2 + i * 2], 0};
    payload_source.class_count = 1;
    payload_source.classes = &payloads[i];
    payload_source.next = i == 0 ? &payload_sources[1] : nullptr;
    receivers[i] = {&tiles[0], static_cast<uint8_t>(i)};
    payloads[i] = {
        .source = &payload_source,
        .destinations = &receivers[i],
        .destination_count = 1,
        .packet = UINT8_MAX,
        .routed = true,
    };
    const size_t first = context.routing.route_count;
    ASSERT_TRUE(loom_aie2p_array_route_egress(
        &context.routing, next_source++, &payload_source.tree,
        payload_source.endpoint.tile->coordinate, 0, {0, 0}, i));
    loom_aie2p_native_route_label(&context, first, UINT8_MAX, false,
                                  UINT32_MAX);
  }
  context.payload_sources = payload_sources;
  EXPECT_EQ(context.routing.link_channels.southbound[family->column_count],
            15u);
  IREE_ASSERT_OK(loom_aie2p_native_connections_build(&context));
  const loom_aie2p_native_switch_input_t input = {
      .routing = &context.routing,
      .routes = context.routes,
      .control_sources = context.control_sources,
      .control_destination_sources = context.control_destination_sources,
  };
  iree_string_view_t requirement;
  IREE_ASSERT_OK(loom_aie2p_native_switches_build(
      &input, &source_arena_, &context.switches, &requirement));
  ASSERT_TRUE(iree_string_view_is_empty(requirement));
  IREE_ASSERT_OK(loom_aie2p_native_connections_verify(
      &context.routing, context.switches, &context.connections,
      &source_arena_));
  std::vector<loom_aie2p_execution_tile_t> occupancy(tile_count);
  const loom_aie2p_execution_plan_t selected = {
      .tiles = occupancy.data(),
      .routing = context.routing,
      .routes = context.routes,
      .control_sources = context.control_sources,
      .control_destination_sources = context.control_destination_sources,
      .column_command_packets = context.column_command_packets,
      .control_paths = paths.data(),
      .switches = context.switches,
      .connections = context.connections,
  };
  loom_aie2p_code_transport_t transport;
  IREE_ASSERT_OK(loom_aie2p_code_transport_initialize(
      &selected, 4, &retained_arena_, &transport));
  // The copy retains exact intervals and packet ownership after the routing
  // producer retires. No source IR or native worker objects remain reachable.
  iree_arena_reset(&source_arena_);
  routes.clear();
  sources.clear();
  paths.clear();
  controls.clear();
  command_packets.clear();
  for (uint16_t row = 2; row < 6; ++row) {
    loom_aie2p_code_transport_binding_t binding;
    loom_aie2p_code_transport_select(&transport, {0, row}, 2, &binding,
                                     &requirement);
    ASSERT_TRUE(iree_string_view_is_empty(requirement));
    loom_aie2p_code_transport_bind(&transport, 2, &binding, &requirement);
    ASSERT_TRUE(iree_string_view_is_empty(requirement));
    EXPECT_EQ(binding.queue_packet, row == 3 ? 6u : 0u);
    EXPECT_EQ(binding.code_packet, row - 2u);
    EXPECT_EQ(binding.engine, 0u);
  }
  EXPECT_EQ(
      transport.plan.routing.link_channels.southbound[family->column_count],
      15u);
  EXPECT_EQ(transport.plan.connections.group_count, 22u);
  bool saw_converged_request = false;
  for (size_t i = 0; i < transport.plan.routing.route_count; ++i) {
    const auto& route = transport.plan.routes[i];
    if (route.control_source == 6 && route.edge->coordinate.row == 2) {
      EXPECT_EQ(route.edge->source_port, LOOM_XDNA_STREAM_PORT_NORTH);
      EXPECT_EQ(route.edge->source_channel, carrier_arrival);
      saw_converged_request = true;
    }
  }
  EXPECT_TRUE(saw_converged_request);
  IREE_ASSERT_OK(loom_aie2p_code_transport_finish(&transport, &retained_arena_,
                                                  &requirement));
  ASSERT_TRUE(iree_string_view_is_empty(requirement));
}

}  // namespace
