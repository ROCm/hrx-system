// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/topology.h"

#include <string.h>

#include "loom/analysis/scc.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/facts.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"

// Mutable logical topology under validation before physical planning begins.
typedef struct loom_aie2p_array_topology_t {
  // Source module retaining exact defining operations for logical entities.
  const loom_module_t* module;

  // Exact function-local facts used to resolve dynamic dimensions.
  const loom_value_fact_table_t* facts;

  // Receives diagnostics for unsupported worker dependencies.
  iree_diagnostic_emitter_t diagnostic_emitter;

  // Arena used for validation scratch storage.
  iree_arena_allocator_t* arena;

  // Logical topology and immutable entity arrays under validation.
  loom_aie2p_array_plan_t* plan;

  // Mutable channel array receiving transport and multicast classification.
  loom_aie2p_array_channel_t* channels;
} loom_aie2p_array_topology_t;

static const loom_op_t* loom_aie2p_array_topology_defining_op(
    const loom_aie2p_array_topology_t* topology, loom_value_id_t value_id) {
  return loom_value_def_op(loom_module_value(topology->module, value_id));
}

static iree_status_t loom_aie2p_array_topology_reject_group_lane(
    const loom_aie2p_array_topology_t* topology, const loom_op_t* op,
    uint32_t group_index, uint32_t lane, uint32_t worker_count) {
  const loom_aie2p_array_group_t* group = &topology->plan->groups[group_index];
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(group_index),
      loom_param_u32(lane),
      loom_param_u32(group->lane_count),
      loom_param_u32(worker_count),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_XDNA_019,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(topology->diagnostic_emitter, &emission);
}

// Adjacency over worker-to-worker channels. Binding transfers do not create
// worker dependencies, and multiple channels retain their distinct edges.
typedef struct loom_aie2p_array_worker_graph_t {
  // Validated logical array topology.
  const loom_aie2p_array_plan_t* plan;
  // First outgoing channel per worker, or UINT32_MAX when there are none.
  uint32_t* first_channels;
  // Next outgoing channel from the same worker, or UINT32_MAX at the end.
  uint32_t* next_channels;
} loom_aie2p_array_worker_graph_t;

static iree_status_t loom_aie2p_array_worker_graph_visit(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor) {
  const loom_aie2p_array_worker_graph_t* graph = user_data;
  for (uint32_t channel_index = graph->first_channels[node];
       channel_index != UINT32_MAX;
       channel_index = graph->next_channels[channel_index]) {
    const loom_aie2p_array_channel_t* channel =
        &graph->plan->channels[channel_index];
    const uint32_t receiver =
        graph->plan->endpoints[channel->receiver_endpoint_index].owner_index;
    IREE_RETURN_IF_ERROR(successor.fn(successor.user_data, receiver));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_topology_validate_worker_dependencies(
    const loom_aie2p_array_topology_t* topology, bool* out_valid) {
  *out_valid = false;
  const loom_aie2p_array_plan_t* plan = topology->plan;
  loom_aie2p_array_worker_graph_t graph = {.plan = plan};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      topology->arena, plan->worker_count, sizeof(*graph.first_channels),
      (void**)&graph.first_channels));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      topology->arena, plan->channel_count, sizeof(*graph.next_channels),
      (void**)&graph.next_channels));
  for (uint32_t i = 0; i < plan->worker_count; ++i) {
    graph.first_channels[i] = UINT32_MAX;
  }
  for (uint32_t i = 0; i < plan->channel_count; ++i) {
    const loom_aie2p_array_channel_t* channel = &plan->channels[i];
    const loom_aie2p_array_endpoint_t* sender =
        &plan->endpoints[channel->sender_endpoint_index];
    const loom_aie2p_array_endpoint_t* receiver =
        &plan->endpoints[channel->receiver_endpoint_index];
    if (sender->owner_kind != LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER ||
        receiver->owner_kind != LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER) {
      continue;
    }
    graph.next_channels[i] = graph.first_channels[sender->owner_index];
    graph.first_channels[sender->owner_index] = i;
  }
  const loom_scc_graph_t scc_graph = {
      .node_count = plan->worker_count,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_aie2p_array_worker_graph_visit, &graph),
  };
  loom_scc_list_t components = {0};
  IREE_RETURN_IF_ERROR(loom_scc_compute(&scc_graph, /*options=*/NULL,
                                        topology->arena, &components));
  for (iree_host_size_t i = 0; i < components.count; ++i) {
    if (components.values[i].is_cycle) {
      const uint32_t worker_index = (uint32_t)components.values[i].nodes[0];
      const loom_aie2p_array_worker_t* worker = &plan->workers[worker_index];
      const loom_diagnostic_param_t params[] = {
          loom_param_u32(worker_index),
          loom_param_u32(worker->group_index),
          loom_param_u32(worker->lane),
      };
      const loom_diagnostic_emission_t emission = {
          .op = plan->function_op,
          .error = LOOM_ERR_XDNA_005,
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
      };
      return iree_diagnostic_emit(topology->diagnostic_emitter, &emission);
    }
  }
  *out_valid = true;
  return iree_ok_status();
}

const loom_aie2p_array_endpoint_t* loom_aie2p_array_topology_base_endpoint(
    const loom_aie2p_array_plan_t* plan,
    const loom_aie2p_array_endpoint_t* endpoint) {
  if (endpoint->binding_view_source_endpoint_index == UINT32_MAX) {
    return endpoint;
  }
  return &plan->endpoints[endpoint->binding_view_source_endpoint_index];
}

static iree_status_t loom_aie2p_array_topology_dimension_value(
    const loom_aie2p_array_topology_t* topology, loom_type_t type,
    iree_host_size_t dimension, uint32_t* out_value) {
  if (!loom_type_dim_is_dynamic_at(type, dimension)) {
    const int64_t value = loom_type_dim_static_size_at(type, dimension);
    if (value < 0 || value > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P array dimension is out of range");
    }
    *out_value = (uint32_t)value;
    return iree_ok_status();
  }
  loom_value_facts_t element_facts = loom_value_facts_unknown();
  int64_t value = 0;
  if (!loom_value_facts_query_all_equal_element(
          &topology->facts->context,
          loom_value_fact_table_lookup(
              topology->facts, loom_type_dim_value_id_at(type, dimension)),
          &element_facts) ||
      !loom_value_facts_as_exact_i64(element_facts, &value) || value < 0 ||
      value > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P array dimension must resolve to one exact non-negative u32 "
        "fact");
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static const loom_aie2p_array_channel_t*
loom_aie2p_array_topology_first_endpoint_channel(
    const loom_aie2p_array_topology_t* topology, uint32_t endpoint_index) {
  for (iree_host_size_t i = 0; i < topology->plan->channel_count; ++i) {
    const loom_aie2p_array_channel_t* channel = &topology->channels[i];
    if (channel->sender_endpoint_index == endpoint_index ||
        channel->receiver_endpoint_index == endpoint_index) {
      return channel;
    }
  }
  IREE_ASSERT_UNREACHABLE(
      "validated worker endpoint must belong to one channel");
  return NULL;
}

static iree_status_t loom_aie2p_array_topology_validate_worker_rates(
    const loom_aie2p_array_topology_t* topology) {
  for (uint32_t worker_index = 0; worker_index < topology->plan->worker_count;
       ++worker_index) {
    const loom_aie2p_array_worker_t* worker =
        &topology->plan->workers[worker_index];
    uint32_t common_record_count = 0;
    uint32_t output_record_count = 0;
    uint32_t sender_count = 0;
    const uint64_t fold_output_end =
        (uint64_t)worker->fold_output_port + worker->fold_output_count;
    if (worker->fold_record_count != 0 &&
        (worker->fold_output_count == 0 || fold_output_end > UINT32_MAX)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P folded worker output port range is invalid");
    }
    for (uint32_t endpoint_index = 0;
         endpoint_index < topology->plan->endpoint_count; ++endpoint_index) {
      const loom_aie2p_array_endpoint_t* endpoint =
          &topology->plan->endpoints[endpoint_index];
      if (endpoint->owner_kind != LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER ||
          endpoint->owner_index != worker_index ||
          endpoint->binding_view_source_endpoint_index != UINT32_MAX) {
        continue;
      }
      const loom_aie2p_array_channel_t* channel =
          loom_aie2p_array_topology_first_endpoint_channel(topology,
                                                           endpoint_index);
      if (worker->fold_record_count == 0) {
        if (common_record_count == 0) {
          common_record_count = channel->record_count;
        }
        if (common_record_count != channel->record_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "AIE2P recordwise worker channels must have one record count");
        }
        continue;
      }

      if (endpoint->direction == LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE) {
        if (common_record_count == 0) {
          common_record_count = channel->record_count;
        } else if (common_record_count != channel->record_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "AIE2P folded worker inputs must have one record count");
        }
        continue;
      }
      ++sender_count;
      if (endpoint->port < worker->fold_output_port ||
          endpoint->port >= fold_output_end) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P folded worker must use its folded output port range");
      }
      if (output_record_count == 0) {
        output_record_count = channel->record_count;
      } else if (output_record_count != channel->record_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P folded worker outputs must have one record count");
      }
      const uint32_t record_byte_length = channel->record_byte_length;
      const uint32_t accumulator_lane_byte_length = 16 * sizeof(float);
      const bool supported_f32_shape =
          record_byte_length == sizeof(float) ||
          (record_byte_length >= accumulator_lane_byte_length &&
           record_byte_length % accumulator_lane_byte_length == 0);
      if (loom_type_element_type(endpoint->message_type) !=
              LOOM_SCALAR_TYPE_F32 ||
          !supported_f32_shape) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AIE2P temporal fold requires one F32 element or a multiple of "
            "16 F32 elements");
      }
    }
    if (worker->fold_record_count != 0) {
      if (worker->fold_kind != LOOM_COMBINING_KIND_ADDF) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AIE2P temporal fold currently supports floating-point addition");
      }
      if (sender_count != worker->fold_output_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P folded worker sender ports must exactly cover its output "
            "port range");
      }
      const uint64_t expected_input_record_count =
          (uint64_t)worker->fold_record_count * output_record_count;
      if (common_record_count != expected_input_record_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P folded worker must consume its fold record count for each "
            "output record");
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_topology_validate_binding_view(
    const loom_aie2p_array_topology_t* topology,
    const loom_aie2p_array_endpoint_t* endpoint, uint32_t* out_record_count) {
  const loom_aie2p_array_endpoint_t* source =
      loom_aie2p_array_topology_base_endpoint(topology->plan, endpoint);
  if (source == endpoint ||
      source->binding_view_source_endpoint_index != UINT32_MAX ||
      source->owner_kind != LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING ||
      endpoint->partition_lane_count == 0 ||
      (endpoint->binding_view_partitioned
           ? endpoint->partition_lane >= endpoint->partition_lane_count
           : endpoint->partition_lane != 0 ||
                 endpoint->partition_lane_count != 1)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P channel binding view is malformed");
  }
  const uint8_t source_rank = loom_type_rank(source->message_type);
  const uint8_t endpoint_rank = loom_type_rank(endpoint->message_type);
  const uint8_t sequence_start = endpoint->binding_view_partitioned ? 1 : 0;
  if (loom_type_element_type(source->message_type) !=
          loom_type_element_type(endpoint->message_type) ||
      source_rank < endpoint_rank + sequence_start) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P binding view must preserve a trailing record type");
  }
  if (endpoint->binding_view_partitioned) {
    uint32_t leading_dimension = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_topology_dimension_value(
        topology, source->message_type, 0, &leading_dimension));
    if (leading_dimension != endpoint->partition_lane_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P partition count must match the leading tile dimension");
    }
  }
  const uint8_t suffix_start = source_rank - endpoint_rank;
  uint64_t record_count = 1;
  for (uint8_t i = sequence_start; i < suffix_start; ++i) {
    uint32_t sequence_dimension = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_topology_dimension_value(
        topology, source->message_type, i, &sequence_dimension));
    if (sequence_dimension == 0 ||
        !iree_checked_mul_u64(record_count, sequence_dimension,
                              &record_count) ||
        record_count > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P binding view record sequence must have a non-empty u32 "
          "count");
    }
  }
  for (iree_host_size_t i = 0; i < endpoint_rank; ++i) {
    uint32_t source_dimension = 0;
    uint32_t endpoint_dimension = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_topology_dimension_value(
        topology, source->message_type, suffix_start + i, &source_dimension));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_topology_dimension_value(
        topology, endpoint->message_type, i, &endpoint_dimension));
    if (source_dimension != endpoint_dimension) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P binding view record type must match the source suffix");
    }
  }
  *out_record_count = (uint32_t)record_count;
  return iree_ok_status();
}

iree_status_t loom_aie2p_array_topology_validate(
    const loom_module_t* module, const loom_value_fact_table_t* facts,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_aie2p_array_plan_t* plan, loom_aie2p_array_channel_t* mutable_channels,
    bool* out_valid) {
  *out_valid = false;
  const loom_aie2p_array_topology_t topology_storage = {
      .module = module,
      .facts = facts,
      .diagnostic_emitter = diagnostic_emitter,
      .arena = arena,
      .plan = plan,
      .channels = mutable_channels,
  };
  const loom_aie2p_array_topology_t* topology = &topology_storage;
  if (plan->worker_count == 0 || plan->channel_count == 0) {
    const loom_diagnostic_param_t params[] = {
        loom_param_u32((uint32_t)plan->worker_count),
        loom_param_u32((uint32_t)plan->channel_count),
    };
    const loom_diagnostic_emission_t emission = {
        .op = plan->function_op,
        .error = LOOM_ERR_XDNA_018,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(topology->diagnostic_emitter, &emission);
  }

  for (iree_host_size_t i = 0; i < plan->worker_count; ++i) {
    const loom_aie2p_array_worker_t* worker = &plan->workers[i];
    const loom_aie2p_array_group_t* group = &plan->groups[worker->group_index];
    uint32_t lane_worker_count = 0;
    for (iree_host_size_t j = 0; j < plan->worker_count; ++j) {
      lane_worker_count +=
          plan->workers[j].group_index == worker->group_index &&
          plan->workers[j].lane == worker->lane;
    }
    if (worker->lane >= group->lane_count || lane_worker_count != 1) {
      return loom_aie2p_array_topology_reject_group_lane(
          topology,
          loom_aie2p_array_topology_defining_op(topology, worker->value_id),
          worker->group_index, worker->lane, lane_worker_count);
    }
    if (worker->coordinate.column == UINT16_MAX) {
      const loom_diagnostic_param_t params[] = {
          loom_param_u32((uint32_t)i),
          loom_param_u32(0),
      };
      const loom_diagnostic_emission_t emission = {
          .op =
              loom_aie2p_array_topology_defining_op(topology, worker->value_id),
          .error = LOOM_ERR_XDNA_020,
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
      };
      return iree_diagnostic_emit(topology->diagnostic_emitter, &emission);
    }
  }

  for (iree_host_size_t group_index = 0; group_index < plan->group_count;
       ++group_index) {
    const loom_aie2p_array_group_t* group = &plan->groups[group_index];
    uint32_t group_worker_count = 0;
    for (iree_host_size_t i = 0; i < plan->worker_count; ++i) {
      group_worker_count += plan->workers[i].group_index == group_index;
    }
    if (group_worker_count == group->lane_count) {
      continue;
    }
    // Every admitted worker lane is unique and in range. If the group has
    // fewer workers than lanes, one lane no greater than the worker count is
    // absent, bounding this search by authored worker count rather than a
    // potentially enormous lane-count constant.
    for (uint64_t lane = 0; lane <= group_worker_count; ++lane) {
      bool found = false;
      for (iree_host_size_t i = 0; i < plan->worker_count; ++i) {
        found |= plan->workers[i].group_index == group_index &&
                 plan->workers[i].lane == (uint32_t)lane;
      }
      if (!found) {
        return loom_aie2p_array_topology_reject_group_lane(
            topology,
            loom_aie2p_array_topology_defining_op(topology, group->value_id),
            (uint32_t)group_index, (uint32_t)lane, 0);
      }
    }
    IREE_ASSERT_UNREACHABLE("incomplete group must have one missing lane");
  }

  for (iree_host_size_t i = 0; i < plan->binding_count; ++i) {
    for (iree_host_size_t j = i + 1; j < plan->binding_count; ++j) {
      if (plan->bindings[i].ordinal == plan->bindings[j].ordinal) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AIE2P binding ordinal is duplicated");
      }
    }
  }

  uint32_t* channel_use_counts = NULL;
  uint32_t* binding_view_use_counts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      topology->arena, plan->endpoint_count, sizeof(*channel_use_counts),
      (void**)&channel_use_counts));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      topology->arena, plan->endpoint_count, sizeof(*binding_view_use_counts),
      (void**)&binding_view_use_counts));
  memset(channel_use_counts, 0,
         plan->endpoint_count * sizeof(*channel_use_counts));
  memset(binding_view_use_counts, 0,
         plan->endpoint_count * sizeof(*binding_view_use_counts));

  for (iree_host_size_t i = 0; i < plan->endpoint_count; ++i) {
    const loom_aie2p_array_endpoint_t* endpoint = &plan->endpoints[i];
    if (endpoint->binding_view_source_endpoint_index != UINT32_MAX) {
      uint32_t record_count = 0;
      IREE_RETURN_IF_ERROR(loom_aie2p_array_topology_validate_binding_view(
          topology, endpoint, &record_count));
      ++binding_view_use_counts[endpoint->binding_view_source_endpoint_index];
    }
    for (iree_host_size_t j = i + 1; j < plan->endpoint_count; ++j) {
      const loom_aie2p_array_endpoint_t* other = &plan->endpoints[j];
      if (endpoint->binding_view_source_endpoint_index == UINT32_MAX &&
          other->binding_view_source_endpoint_index == UINT32_MAX &&
          endpoint->owner_kind == other->owner_kind &&
          endpoint->owner_index == other->owner_index &&
          endpoint->port == other->port) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AIE2P owner port is defined more than once");
      }
    }
  }

  for (iree_host_size_t i = 0; i < plan->channel_count; ++i) {
    loom_aie2p_array_channel_t* channel = &topology->channels[i];
    if (channel->capacity == 0 || channel->record_count == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AIE2P channel capacity and record count cannot "
                              "be zero");
    }
    const loom_aie2p_array_endpoint_t* sender =
        &plan->endpoints[channel->sender_endpoint_index];
    const loom_aie2p_array_endpoint_t* receiver =
        &plan->endpoints[channel->receiver_endpoint_index];
    if (sender->direction != LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND ||
        receiver->direction != LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE ||
        !loom_type_equal(sender->message_type, receiver->message_type)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P channel endpoints must have matching typed directions");
    }
    ++channel_use_counts[channel->sender_endpoint_index];
    ++channel_use_counts[channel->receiver_endpoint_index];

    const loom_aie2p_array_endpoint_t* binding_view_endpoint = NULL;
    if (sender->binding_view_source_endpoint_index != UINT32_MAX) {
      binding_view_endpoint = sender;
    } else if (receiver->binding_view_source_endpoint_index != UINT32_MAX) {
      binding_view_endpoint = receiver;
    }
    if (binding_view_endpoint != NULL) {
      uint32_t binding_view_record_count = 0;
      IREE_RETURN_IF_ERROR(loom_aie2p_array_topology_validate_binding_view(
          topology, binding_view_endpoint, &binding_view_record_count));
      if (channel->record_count != binding_view_record_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P channel record count must match its binding view");
      }
    }
    const loom_aie2p_array_endpoint_t* base_sender =
        loom_aie2p_array_topology_base_endpoint(plan, sender);
    if (base_sender->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING &&
        receiver->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER) {
      const loom_aie2p_array_binding_t* binding =
          &plan->bindings[base_sender->owner_index];
      if ((binding->access & LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ) == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P input channel uses a non-readable binding");
      }
      channel->transport = LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA;
    } else if (sender->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER &&
               receiver->owner_kind ==
                   LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING) {
      const loom_aie2p_array_binding_t* binding =
          &plan->bindings[receiver->owner_index];
      if ((binding->access & LOOM_AIE2P_ARRAY_BINDING_ACCESS_WRITE) == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P output channel uses a non-writable binding");
      }
      channel->transport = LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA;
    } else if (sender->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER &&
               receiver->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER) {
      const loom_aie2p_array_worker_t* sender_worker =
          &plan->workers[sender->owner_index];
      const loom_aie2p_array_worker_t* receiver_worker =
          &plan->workers[receiver->owner_index];
      const int row_delta = (int)sender_worker->coordinate.row -
                            (int)receiver_worker->coordinate.row;
      channel->transport =
          sender_worker->coordinate.column ==
                      receiver_worker->coordinate.column &&
                  (row_delta == -1 || row_delta == 1)
              ? LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_NEIGHBOR_MEMORY
              : LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_ROUTED_DMA;
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported AIE2P channel ownership");
    }
  }

  // One sending endpoint owns one physical source even when several consumers
  // use it. Retain the canonical source channel while this traversal owns the
  // endpoint fanout. Worker fanout shares its storage, locks, and compute DMA;
  // binding fanout shares its host-facing shim DMA and runtime patch. Neighbor
  // memory cannot preserve a worker's single-ring contract across a mixed set
  // of destinations, so promote those branches to routed DMA.
  for (iree_host_size_t endpoint_index = 0;
       endpoint_index < plan->endpoint_count; ++endpoint_index) {
    const loom_aie2p_array_endpoint_t* endpoint =
        &plan->endpoints[endpoint_index];
    if (endpoint->direction != LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND ||
        channel_use_counts[endpoint_index] <= 1) {
      continue;
    }
    uint32_t source_channel_index = UINT32_MAX;
    for (iree_host_size_t channel_index = 0;
         channel_index < plan->channel_count; ++channel_index) {
      loom_aie2p_array_channel_t* channel = &topology->channels[channel_index];
      if (channel->sender_endpoint_index != endpoint_index) {
        continue;
      }
      if (source_channel_index == UINT32_MAX) {
        source_channel_index = (uint32_t)channel_index;
      } else if (endpoint->owner_kind ==
                     LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER &&
                 (channel->capacity !=
                      topology->channels[source_channel_index].capacity ||
                  channel->record_count !=
                      topology->channels[source_channel_index].record_count ||
                  channel->record_byte_length !=
                      topology->channels[source_channel_index]
                          .record_byte_length)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P worker multicast channels must have one ring shape");
      }
      channel->source_channel_index = source_channel_index;
      if (channel->transport ==
          LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_NEIGHBOR_MEMORY) {
        channel->transport = LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_ROUTED_DMA;
      }
    }
  }

  for (iree_host_size_t i = 0; i < plan->endpoint_count; ++i) {
    const loom_aie2p_array_endpoint_t* endpoint = &plan->endpoints[i];
    const loom_aie2p_array_endpoint_t* base_endpoint =
        loom_aie2p_array_topology_base_endpoint(plan, endpoint);
    const bool is_multicast_sender =
        endpoint->direction == LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND &&
        (base_endpoint->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING ||
         endpoint->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER);
    if (endpoint->binding_view_source_endpoint_index != UINT32_MAX) {
      if (channel_use_counts[i] == 0 || binding_view_use_counts[i] != 0 ||
          (!is_multicast_sender && channel_use_counts[i] != 1)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P binding view must feed one channel or source multicast");
      }
      continue;
    }
    if (endpoint->owner_kind != LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING) {
      if (binding_view_use_counts[i] != 0 || channel_use_counts[i] == 0 ||
          (!is_multicast_sender && channel_use_counts[i] != 1)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P worker endpoint must feed one channel or source multicast");
      }
      continue;
    }
    if (binding_view_use_counts[i] == 0) {
      if (channel_use_counts[i] == 0 ||
          (!is_multicast_sender && channel_use_counts[i] != 1)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P direct binding endpoint must feed one channel or source "
            "multicast");
      }
      continue;
    }
    if (channel_use_counts[i] != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P binding endpoint must feed binding views exclusively");
    }

    const loom_aie2p_array_endpoint_t* first_view = endpoint;
    for (iree_host_size_t j = 0; j < plan->endpoint_count; ++j) {
      const loom_aie2p_array_endpoint_t* view = &plan->endpoints[j];
      if (view->binding_view_source_endpoint_index == i) {
        first_view = view;
        break;
      }
    }
    if (!first_view->binding_view_partitioned) {
      if (binding_view_use_counts[i] != 1) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P binding endpoint must own one unpartitioned view");
      }
      continue;
    }

    const uint32_t partition_lane_count = first_view->partition_lane_count;
    if (binding_view_use_counts[i] != partition_lane_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P partitioned binding view must materialize every lane");
    }
    for (uint32_t lane = 0; lane < partition_lane_count; ++lane) {
      iree_host_size_t match_count = 0;
      for (iree_host_size_t j = 0; j < plan->endpoint_count; ++j) {
        const loom_aie2p_array_endpoint_t* view = &plan->endpoints[j];
        if (view->binding_view_source_endpoint_index != i) {
          continue;
        }
        if (!view->binding_view_partitioned ||
            view->partition_lane_count != partition_lane_count ||
            view->binding_byte_offset != first_view->binding_byte_offset) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "AIE2P binding endpoint views must share one partition");
        }
        if (view->partition_lane == lane) {
          ++match_count;
        }
      }
      if (match_count != 1) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P binding view partition lanes must be unique and complete");
      }
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_array_topology_validate_worker_rates(topology));
  return loom_aie2p_array_topology_validate_worker_dependencies(topology,
                                                                out_valid);
}
