// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/topology.h"

#include <string.h>

#include "loom/analysis/scc.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/facts.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"

// Mutable logical topology under validation before physical planning begins.
typedef struct loom_aie2p_array_topology_t {
  // Source module retaining exact defining operations for logical entities.
  const loom_module_t* module;

  // Exact function-local facts used to resolve dynamic dimensions.
  const loom_value_fact_table_t* facts;

  // Receives diagnostics for rejected authored topology.
  iree_diagnostic_emitter_t diagnostic_emitter;

  // Arena used for validation scratch storage.
  iree_arena_allocator_t* arena;

  // Logical topology under validation.
  loom_aie2p_array_plan_t* plan;

  // Mutable worker array receiving active endpoint counts.
  loom_aie2p_array_worker_t* workers;

  // Mutable endpoint array receiving channel and leaf-resource facts.
  loom_aie2p_array_endpoint_t* endpoints;

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

static iree_status_t loom_aie2p_array_topology_reject_binding_ordinal(
    const loom_aie2p_array_topology_t* topology,
    const loom_aie2p_array_binding_t* binding, iree_string_view_t reason) {
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(binding->ordinal),
      loom_param_string(reason),
      loom_param_u32(topology->plan->binding_slot_count),
  };
  const loom_diagnostic_emission_t emission = {
      .op = loom_aie2p_array_topology_defining_op(topology, binding->value_id),
      .error = LOOM_ERR_XDNA_031,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(topology->diagnostic_emitter, &emission);
}

static iree_status_t loom_aie2p_array_topology_reject_worker_port(
    const loom_aie2p_array_topology_t* topology, const loom_op_t* op,
    uint32_t worker_index, uint64_t port, uint32_t active_endpoint_count,
    uint32_t resource_count) {
  const loom_aie2p_array_worker_t* worker = &topology->workers[worker_index];
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(worker_index),
      loom_param_string(
          loom_low_diagnostic_symbol_name(topology->module, worker->entry)),
      loom_param_u64(port),
      loom_param_u32(active_endpoint_count),
      loom_param_u32(resource_count),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_XDNA_022,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(topology->diagnostic_emitter, &emission);
}

static iree_status_t loom_aie2p_array_topology_reject_binding_view(
    const loom_aie2p_array_topology_t* topology,
    const loom_aie2p_array_endpoint_t* endpoint,
    const loom_aie2p_array_endpoint_t* source, iree_string_view_t reason) {
  loom_diagnostic_param_t lane = loom_param_u32(endpoint->partition_lane);
  loom_diagnostic_param_t lane_count =
      loom_param_u32(endpoint->partition_lane_count);
  if (endpoint->binding_view_partitioned) {
    lane = loom_param_with_field_ref(
        lane, loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 2));
    lane_count = loom_param_with_field_ref(
        lane_count,
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 3));
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_type(endpoint->message_type),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0)),
      lane,
      lane_count,
      loom_param_type(source->message_type),
      loom_param_string(reason),
  };
  const loom_diagnostic_emission_t emission = {
      .module = topology->module,
      .op = loom_aie2p_array_topology_defining_op(topology, endpoint->value_id),
      .error = LOOM_ERR_XDNA_023,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(topology->diagnostic_emitter, &emission);
}

static iree_status_t loom_aie2p_array_topology_reject_channel_record(
    const loom_aie2p_array_topology_t* topology, uint32_t channel_index,
    loom_type_t record_type, iree_string_view_t reason) {
  const loom_aie2p_array_channel_t* channel =
      &topology->channels[channel_index];
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(channel_index),
      loom_param_with_field_ref(
          loom_param_type(record_type),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0)),
      loom_param_string(reason),
  };
  const loom_diagnostic_emission_t emission = {
      .module = topology->module,
      .op = loom_aie2p_array_topology_defining_op(topology, channel->value_id),
      .error = LOOM_ERR_XDNA_024,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(topology->diagnostic_emitter, &emission);
}

static iree_status_t loom_aie2p_array_topology_reject_channel_ring(
    const loom_aie2p_array_topology_t* topology, uint32_t channel_index,
    iree_string_view_t quantity, uint32_t actual, uint64_t required,
    iree_string_view_t relationship, uint16_t operand_index) {
  const loom_aie2p_array_channel_t* channel =
      &topology->channels[channel_index];
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(channel_index),
      loom_param_string(quantity),
      loom_param_with_field_ref(
          loom_param_u32(actual),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                    operand_index)),
      loom_param_string(relationship),
      loom_param_u64(required),
  };
  const loom_diagnostic_emission_t emission = {
      .op = loom_aie2p_array_topology_defining_op(topology, channel->value_id),
      .error = LOOM_ERR_XDNA_025,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(topology->diagnostic_emitter, &emission);
}

static iree_string_view_t loom_aie2p_array_topology_owner_kind_name(
    loom_aie2p_array_endpoint_owner_kind_t owner_kind) {
  return owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING
             ? IREE_SV("binding")
             : IREE_SV("worker");
}

static iree_status_t loom_aie2p_array_topology_reject_channel_connection(
    const loom_aie2p_array_topology_t* topology, uint32_t channel_index,
    iree_string_view_t reason) {
  const loom_aie2p_array_channel_t* channel =
      &topology->channels[channel_index];
  const loom_aie2p_array_endpoint_t* sender =
      &topology->endpoints[channel->sender_endpoint_index];
  const loom_aie2p_array_endpoint_t* receiver =
      &topology->endpoints[channel->receiver_endpoint_index];
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(channel_index),
      loom_param_string(
          loom_aie2p_array_topology_owner_kind_name(sender->owner_kind)),
      loom_param_with_field_ref(
          loom_param_u32(sender->owner_index),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 0)),
      loom_param_u32(sender->port),
      loom_param_string(
          loom_aie2p_array_topology_owner_kind_name(receiver->owner_kind)),
      loom_param_with_field_ref(
          loom_param_u32(receiver->owner_index),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 1)),
      loom_param_u32(receiver->port),
      loom_param_string(reason),
  };
  const loom_diagnostic_emission_t emission = {
      .op = loom_aie2p_array_topology_defining_op(topology, channel->value_id),
      .error = LOOM_ERR_XDNA_026,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(topology->diagnostic_emitter, &emission);
}

static iree_status_t loom_aie2p_array_topology_reject_receiver_use(
    const loom_aie2p_array_topology_t* topology, uint32_t channel_index,
    uint32_t endpoint_index, uint32_t first_channel_index) {
  const loom_aie2p_array_channel_t* channel =
      &topology->channels[channel_index];
  const loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_u32(endpoint_index),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 1)),
      loom_param_u32(first_channel_index),
      loom_param_u32(channel_index),
  };
  const loom_diagnostic_emission_t emission = {
      .op = loom_aie2p_array_topology_defining_op(topology, channel->value_id),
      .error = LOOM_ERR_XDNA_027,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(topology->diagnostic_emitter, &emission);
}

static iree_status_t loom_aie2p_array_topology_reject_fold_output_range(
    const loom_aie2p_array_topology_t* topology, const loom_op_t* op,
    uint32_t worker_index, iree_string_view_t quantity, uint64_t actual,
    iree_string_view_t requirement) {
  const loom_aie2p_array_worker_t* worker = &topology->workers[worker_index];
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(worker_index),
      loom_param_u32(worker->fold_output_port),
      loom_param_u32(worker->fold_output_count),
      loom_param_string(quantity),
      loom_param_u64(actual),
      loom_param_string(requirement),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_XDNA_028,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(topology->diagnostic_emitter, &emission);
}

static iree_status_t loom_aie2p_array_topology_reject_fold_materialization(
    const loom_aie2p_array_topology_t* topology, const loom_op_t* op,
    uint32_t worker_index, uint32_t channel_index, iree_string_view_t reason) {
  const loom_aie2p_array_worker_t* worker = &topology->workers[worker_index];
  const loom_aie2p_array_channel_t* channel =
      &topology->channels[channel_index];
  const loom_aie2p_array_endpoint_t* sender =
      &topology->endpoints[channel->sender_endpoint_index];
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(worker_index),
      loom_param_u32(channel_index),
      loom_param_type(sender->message_type),
      loom_param_u32((uint32_t)worker->fold_kind),
      loom_param_string(reason),
  };
  const loom_diagnostic_emission_t emission = {
      .module = topology->module,
      .op = op,
      .error = LOOM_ERR_XDNA_029,
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

static bool loom_aie2p_array_topology_dimension_value(
    const loom_aie2p_array_topology_t* topology, loom_type_t type,
    iree_host_size_t dimension, uint32_t* out_value) {
  if (!loom_type_dim_is_dynamic_at(type, dimension)) {
    const int64_t value = loom_type_dim_static_size_at(type, dimension);
    if (value < 0 || value > UINT32_MAX) {
      return false;
    }
    *out_value = (uint32_t)value;
    return true;
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
    return false;
  }
  *out_value = (uint32_t)value;
  return true;
}

static bool loom_aie2p_array_topology_record_byte_length(
    const loom_aie2p_array_topology_t* topology, loom_type_t record_type,
    uint32_t* out_byte_length, iree_string_view_t* out_reason) {
  uint64_t element_count = 1;
  for (iree_host_size_t i = 0; i < loom_type_rank(record_type); ++i) {
    uint32_t dimension = 0;
    if (!loom_aie2p_array_topology_dimension_value(topology, record_type, i,
                                                   &dimension)) {
      *out_reason = IREE_SV(
          "every dimension must resolve to an exact non-negative u32 value");
      return false;
    }
    if (dimension == 0) {
      *out_reason = IREE_SV("every dimension must be positive");
      return false;
    }
    if (!iree_checked_mul_u64(element_count, dimension, &element_count)) {
      *out_reason = IREE_SV("the element count must fit in 64 bits");
      return false;
    }
  }
  const uint64_t element_bit_width =
      (uint64_t)loom_scalar_type_bitwidth(loom_type_element_type(record_type));
  uint64_t bit_length = 0;
  if (!iree_checked_mul_u64(element_count, element_bit_width, &bit_length)) {
    *out_reason = IREE_SV("the total bit count must fit in 64 bits");
    return false;
  }
  if ((bit_length & 7u) != 0) {
    *out_reason = IREE_SV("the total bit count must form whole bytes");
    return false;
  }
  const uint64_t byte_length = bit_length / 8u;
  if (byte_length > UINT32_MAX) {
    *out_reason = IREE_SV("the byte count must fit in u32");
    return false;
  }
  *out_byte_length = (uint32_t)byte_length;
  return true;
}

static bool loom_aie2p_array_topology_is_active_worker_endpoint(
    const loom_aie2p_array_endpoint_t* endpoint) {
  return endpoint->binding_view_source_endpoint_index == UINT32_MAX &&
         endpoint->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER &&
         endpoint->channel_use_count != 0;
}

static iree_status_t loom_aie2p_array_topology_validate_worker_interfaces(
    const loom_aie2p_array_topology_t* topology, bool* out_valid) {
  *out_valid = false;
  for (iree_host_size_t i = 0; i < topology->plan->worker_count; ++i) {
    topology->workers[i].active_endpoint_count = 0;
  }

  for (iree_host_size_t endpoint_index = 0;
       endpoint_index < topology->plan->endpoint_count; ++endpoint_index) {
    const loom_aie2p_array_endpoint_t* endpoint =
        &topology->endpoints[endpoint_index];
    if (!loom_aie2p_array_topology_is_active_worker_endpoint(endpoint)) {
      continue;
    }

    const uint32_t worker_index = endpoint->owner_index;
    loom_aie2p_array_worker_t* worker = &topology->workers[worker_index];
    const loom_low_function_requirements_t* requirements =
        &worker->leaf->requirements;
    ++worker->active_endpoint_count;

    uint32_t active_endpoint_count = 0;
    const loom_op_t* duplicate_endpoint_op = NULL;
    for (iree_host_size_t i = 0; i < topology->plan->endpoint_count; ++i) {
      const loom_aie2p_array_endpoint_t* candidate = &topology->endpoints[i];
      if (loom_aie2p_array_topology_is_active_worker_endpoint(candidate) &&
          candidate->owner_index == worker_index &&
          candidate->port == endpoint->port) {
        ++active_endpoint_count;
        if (active_endpoint_count == 2) {
          duplicate_endpoint_op = loom_aie2p_array_topology_defining_op(
              topology, candidate->value_id);
        }
      }
    }

    uint32_t resource_count = 0;
    uint32_t resource_ordinal = UINT32_MAX;
    const loom_op_t* duplicate_resource_op = NULL;
    for (iree_host_size_t i = 0; i < requirements->resource_count; ++i) {
      if ((uint64_t)loom_low_resource_index(requirements->resources[i]) ==
          endpoint->port) {
        ++resource_count;
        resource_ordinal = (uint32_t)i;
        if (resource_count == 2) {
          duplicate_resource_op = requirements->resources[i];
        }
      }
    }
    if (active_endpoint_count != 1 || resource_count > 1) {
      return loom_aie2p_array_topology_reject_worker_port(
          topology,
          duplicate_endpoint_op != NULL ? duplicate_endpoint_op
                                        : duplicate_resource_op,
          worker_index, endpoint->port, active_endpoint_count, resource_count);
    }
    if (resource_ordinal != UINT32_MAX) {
      topology->endpoints[endpoint_index].worker_resource_ordinal =
          resource_ordinal;
    }
  }

  for (uint32_t worker_index = 0; worker_index < topology->plan->worker_count;
       ++worker_index) {
    const loom_low_function_requirements_t* requirements =
        &topology->workers[worker_index].leaf->requirements;
    for (iree_host_size_t resource_ordinal = 0;
         resource_ordinal < requirements->resource_count; ++resource_ordinal) {
      const loom_op_t* resource = requirements->resources[resource_ordinal];
      const uint64_t port = (uint64_t)loom_low_resource_index(resource);
      bool is_first_resource = true;
      for (iree_host_size_t i = 0; i < resource_ordinal; ++i) {
        is_first_resource &= (uint64_t)loom_low_resource_index(
                                 requirements->resources[i]) != port;
      }
      if (!is_first_resource) {
        continue;
      }

      uint32_t resource_count = 0;
      for (iree_host_size_t i = resource_ordinal;
           i < requirements->resource_count; ++i) {
        resource_count += (uint64_t)loom_low_resource_index(
                              requirements->resources[i]) == port;
      }
      uint32_t active_endpoint_count = 0;
      for (iree_host_size_t i = 0; i < topology->plan->endpoint_count; ++i) {
        const loom_aie2p_array_endpoint_t* endpoint = &topology->endpoints[i];
        active_endpoint_count +=
            loom_aie2p_array_topology_is_active_worker_endpoint(endpoint) &&
            endpoint->owner_index == worker_index && endpoint->port == port;
      }
      if (active_endpoint_count == 0) {
        return loom_aie2p_array_topology_reject_worker_port(
            topology, resource, worker_index, port, active_endpoint_count,
            resource_count);
      }
    }
  }
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_topology_validate_worker_rates(
    const loom_aie2p_array_topology_t* topology, bool* out_valid) {
  *out_valid = false;
  for (uint32_t worker_index = 0; worker_index < topology->plan->worker_count;
       ++worker_index) {
    const loom_aie2p_array_worker_t* worker =
        &topology->plan->workers[worker_index];
    uint32_t common_record_count = 0;
    uint32_t first_input_channel_index = UINT32_MAX;
    uint32_t output_record_count = 0;
    uint32_t first_output_channel_index = UINT32_MAX;
    uint32_t sender_count = 0;
    const uint64_t fold_output_end =
        (uint64_t)worker->fold_output_port + worker->fold_output_count;
    if (worker->fold_record_count != 0) {
      if (worker->fold_output_count == 0) {
        return loom_aie2p_array_topology_reject_fold_output_range(
            topology,
            loom_aie2p_array_topology_defining_op(topology, worker->value_id),
            worker_index, IREE_SV("output count"), 0,
            IREE_SV("must be positive"));
      }
      if (fold_output_end > UINT32_MAX) {
        return loom_aie2p_array_topology_reject_fold_output_range(
            topology,
            loom_aie2p_array_topology_defining_op(topology, worker->value_id),
            worker_index, IREE_SV("exclusive output range end"),
            fold_output_end, IREE_SV("must fit in u32"));
      }
    }
    for (uint32_t endpoint_index = 0;
         endpoint_index < topology->plan->endpoint_count; ++endpoint_index) {
      const loom_aie2p_array_endpoint_t* endpoint =
          &topology->plan->endpoints[endpoint_index];
      if (endpoint->owner_kind != LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER ||
          endpoint->owner_index != worker_index ||
          endpoint->binding_view_source_endpoint_index != UINT32_MAX ||
          endpoint->channel_use_count == 0) {
        continue;
      }
      const uint32_t channel_index = endpoint->first_channel_index;
      const loom_aie2p_array_channel_t* channel =
          &topology->channels[channel_index];
      if (worker->fold_record_count == 0) {
        if (common_record_count == 0) {
          common_record_count = channel->record_count;
        }
        if (common_record_count != channel->record_count) {
          return loom_aie2p_array_topology_reject_channel_ring(
              topology, channel_index, IREE_SV("record count"),
              channel->record_count, common_record_count,
              IREE_SV("the first active recordwise channel record count"), 3);
        }
        continue;
      }

      if (endpoint->direction == LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE) {
        if (common_record_count == 0) {
          common_record_count = channel->record_count;
          first_input_channel_index = channel_index;
        } else if (common_record_count != channel->record_count) {
          return loom_aie2p_array_topology_reject_channel_ring(
              topology, channel_index, IREE_SV("record count"),
              channel->record_count, common_record_count,
              IREE_SV("the first folded input channel record count"), 3);
        }
        continue;
      }
      ++sender_count;
      if (endpoint->port < worker->fold_output_port ||
          endpoint->port >= fold_output_end) {
        return loom_aie2p_array_topology_reject_fold_output_range(
            topology,
            loom_aie2p_array_topology_defining_op(topology, endpoint->value_id),
            worker_index, IREE_SV("active sender port"), endpoint->port,
            IREE_SV("must lie in the declared output range"));
      }
      if (output_record_count == 0) {
        output_record_count = channel->record_count;
        first_output_channel_index = channel_index;
      } else if (output_record_count != channel->record_count) {
        return loom_aie2p_array_topology_reject_channel_ring(
            topology, channel_index, IREE_SV("record count"),
            channel->record_count, output_record_count,
            IREE_SV("the first folded output channel record count"), 3);
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
        return loom_aie2p_array_topology_reject_fold_materialization(
            topology,
            loom_aie2p_array_topology_defining_op(topology, channel->value_id),
            worker_index, channel_index,
            IREE_SV("the resident accumulator supports one f32 element or a "
                    "multiple of 16 f32 elements"));
      }
    }
    if (worker->fold_record_count != 0) {
      if (sender_count != worker->fold_output_count) {
        return loom_aie2p_array_topology_reject_fold_output_range(
            topology,
            loom_aie2p_array_topology_defining_op(topology, worker->value_id),
            worker_index, IREE_SV("active sender count"), sender_count,
            IREE_SV("must equal the declared output count"));
      }
      if (worker->fold_kind != LOOM_COMBINING_KIND_ADDF) {
        return loom_aie2p_array_topology_reject_fold_materialization(
            topology,
            loom_aie2p_array_topology_defining_op(topology, worker->value_id),
            worker_index, first_output_channel_index,
            IREE_SV("only addf (combiner 1) is implemented"));
      }
      if (first_input_channel_index == UINT32_MAX) {
        return loom_aie2p_array_topology_reject_fold_materialization(
            topology,
            loom_aie2p_array_topology_defining_op(topology, worker->value_id),
            worker_index, first_output_channel_index,
            IREE_SV("the fold has no active input channel"));
      }
      const uint64_t expected_input_record_count =
          (uint64_t)worker->fold_record_count * output_record_count;
      if (common_record_count != expected_input_record_count) {
        return loom_aie2p_array_topology_reject_channel_ring(
            topology, first_input_channel_index, IREE_SV("record count"),
            common_record_count, expected_input_record_count,
            IREE_SV("the fold count multiplied by the output record count"), 3);
      }
    }
  }
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_topology_validate_binding_view(
    const loom_aie2p_array_topology_t* topology,
    const loom_aie2p_array_endpoint_t* endpoint, uint32_t* out_record_count,
    bool* out_valid) {
  *out_record_count = 0;
  *out_valid = false;
  const loom_aie2p_array_endpoint_t* source =
      loom_aie2p_array_topology_base_endpoint(topology->plan, endpoint);
  if (source->binding_view_source_endpoint_index != UINT32_MAX ||
      source->owner_kind != LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING) {
    return loom_aie2p_array_topology_reject_binding_view(
        topology, endpoint, source,
        IREE_SV("the source must be one direct binding endpoint"));
  }
  if (endpoint->binding_view_partitioned &&
      (endpoint->partition_lane_count == 0 ||
       endpoint->partition_lane >= endpoint->partition_lane_count)) {
    return loom_aie2p_array_topology_reject_binding_view(
        topology, endpoint, source,
        IREE_SV("the partition lane must be within a non-empty lane set"));
  }
  const uint8_t source_rank = loom_type_rank(source->message_type);
  const uint8_t endpoint_rank = loom_type_rank(endpoint->message_type);
  const uint8_t sequence_start = endpoint->binding_view_partitioned ? 1 : 0;
  if (loom_type_element_type(source->message_type) !=
          loom_type_element_type(endpoint->message_type) ||
      source_rank < endpoint_rank + sequence_start) {
    return loom_aie2p_array_topology_reject_binding_view(
        topology, endpoint, source,
        IREE_SV("the result element type and rank must form a trailing source "
                "record"));
  }
  if (endpoint->binding_view_partitioned) {
    uint32_t leading_dimension = 0;
    if (!loom_aie2p_array_topology_dimension_value(
            topology, source->message_type, 0, &leading_dimension)) {
      return loom_aie2p_array_topology_reject_binding_view(
          topology, endpoint, source,
          IREE_SV("the leading source dimension must resolve to an exact "
                  "non-negative u32 value"));
    }
    if (leading_dimension != endpoint->partition_lane_count) {
      return loom_aie2p_array_topology_reject_binding_view(
          topology, endpoint, source,
          IREE_SV("the leading source dimension must equal the partition lane "
                  "count"));
    }
  }
  const uint8_t suffix_start = source_rank - endpoint_rank;
  uint64_t record_count = 1;
  for (uint8_t i = sequence_start; i < suffix_start; ++i) {
    uint32_t sequence_dimension = 0;
    if (!loom_aie2p_array_topology_dimension_value(
            topology, source->message_type, i, &sequence_dimension)) {
      return loom_aie2p_array_topology_reject_binding_view(
          topology, endpoint, source,
          IREE_SV("every source sequence dimension must resolve to an exact "
                  "non-negative u32 value"));
    }
    if (sequence_dimension == 0 ||
        !iree_checked_mul_u64(record_count, sequence_dimension,
                              &record_count) ||
        record_count > UINT32_MAX) {
      return loom_aie2p_array_topology_reject_binding_view(
          topology, endpoint, source,
          IREE_SV("the source record sequence must have a positive u32 "
                  "element count"));
    }
  }
  for (iree_host_size_t i = 0; i < endpoint_rank; ++i) {
    uint32_t source_dimension = 0;
    uint32_t endpoint_dimension = 0;
    if (!loom_aie2p_array_topology_dimension_value(
            topology, source->message_type, suffix_start + i,
            &source_dimension) ||
        !loom_aie2p_array_topology_dimension_value(
            topology, endpoint->message_type, i, &endpoint_dimension) ||
        source_dimension != endpoint_dimension) {
      return loom_aie2p_array_topology_reject_binding_view(
          topology, endpoint, source,
          IREE_SV("source and result record dimensions must resolve to equal "
                  "non-negative u32 values"));
    }
  }
  *out_record_count = (uint32_t)record_count;
  *out_valid = true;
  return iree_ok_status();
}

iree_status_t loom_aie2p_array_topology_validate(
    const loom_module_t* module, const loom_value_fact_table_t* facts,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_aie2p_array_plan_t* plan, loom_aie2p_array_worker_t* mutable_workers,
    loom_aie2p_array_endpoint_t* mutable_endpoints,
    loom_aie2p_array_channel_t* mutable_channels, bool* out_valid) {
  *out_valid = false;
  const loom_aie2p_array_topology_t topology_storage = {
      .module = module,
      .facts = facts,
      .diagnostic_emitter = diagnostic_emitter,
      .arena = arena,
      .plan = plan,
      .workers = mutable_workers,
      .endpoints = mutable_endpoints,
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

  const iree_host_size_t binding_ordinal_word_count =
      ((iree_host_size_t)plan->binding_slot_count + 63u) / 64u;
  uint64_t* binding_ordinals = NULL;
  if (plan->binding_count > 1 && binding_ordinal_word_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        topology->arena, binding_ordinal_word_count, sizeof(*binding_ordinals),
        (void**)&binding_ordinals));
    memset(binding_ordinals, 0,
           binding_ordinal_word_count * sizeof(*binding_ordinals));
  }
  for (iree_host_size_t i = 0; i < plan->binding_count; ++i) {
    const loom_aie2p_array_binding_t* binding = &plan->bindings[i];
    if (binding->ordinal >= plan->binding_slot_count) {
      return loom_aie2p_array_topology_reject_binding_ordinal(
          topology, binding, IREE_SV("outside the dense binding table"));
    }
    if (binding_ordinals == NULL) {
      continue;
    }
    const uint64_t mask = UINT64_C(1) << (binding->ordinal & 63u);
    uint64_t* word = &binding_ordinals[binding->ordinal / 64u];
    if ((*word & mask) != 0) {
      return loom_aie2p_array_topology_reject_binding_ordinal(
          topology, binding,
          IREE_SV("declared by more than one active binding"));
    }
    *word |= mask;
  }

  for (iree_host_size_t i = 0; i < plan->channel_count; ++i) {
    const loom_aie2p_array_channel_t* channel = &topology->channels[i];
    loom_aie2p_array_endpoint_t* sender =
        &topology->endpoints[channel->sender_endpoint_index];
    loom_aie2p_array_endpoint_t* receiver =
        &topology->endpoints[channel->receiver_endpoint_index];
    if (receiver->first_channel_index != UINT32_MAX) {
      return loom_aie2p_array_topology_reject_receiver_use(
          topology, (uint32_t)i, channel->receiver_endpoint_index,
          receiver->first_channel_index);
    }
    if (sender->first_channel_index == UINT32_MAX) {
      sender->first_channel_index = (uint32_t)i;
    }
    receiver->first_channel_index = (uint32_t)i;
    ++sender->channel_use_count;
    ++receiver->channel_use_count;
  }

  for (iree_host_size_t i = 0; i < plan->endpoint_count; ++i) {
    loom_aie2p_array_endpoint_t* endpoint = &topology->endpoints[i];
    if (endpoint->binding_view_source_endpoint_index == UINT32_MAX ||
        endpoint->channel_use_count == 0) {
      continue;
    }
    bool binding_view_valid = false;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_topology_validate_binding_view(
        topology, endpoint, &endpoint->binding_view_record_count,
        &binding_view_valid));
    if (!binding_view_valid) {
      return iree_ok_status();
    }
  }

  for (iree_host_size_t i = 0; i < plan->channel_count; ++i) {
    loom_aie2p_array_channel_t* channel = &topology->channels[i];
    const loom_aie2p_array_endpoint_t* sender =
        &plan->endpoints[channel->sender_endpoint_index];
    const loom_aie2p_array_endpoint_t* receiver =
        &plan->endpoints[channel->receiver_endpoint_index];
    if (channel->capacity == 0) {
      return loom_aie2p_array_topology_reject_channel_ring(
          topology, (uint32_t)i, IREE_SV("capacity"), channel->capacity, 1,
          IREE_SV("the minimum non-empty ring capacity"), 2);
    }
    if (channel->capacity > INT8_MAX) {
      return loom_aie2p_array_topology_reject_channel_ring(
          topology, (uint32_t)i, IREE_SV("capacity"), channel->capacity,
          INT8_MAX, IREE_SV("the maximum credit-lock capacity"), 2);
    }
    if (channel->record_count == 0) {
      return loom_aie2p_array_topology_reject_channel_ring(
          topology, (uint32_t)i, IREE_SV("record count"), channel->record_count,
          1, IREE_SV("the minimum records transferred per activation"), 3);
    }
    iree_string_view_t record_reason = iree_string_view_empty();
    if (!loom_aie2p_array_topology_record_byte_length(
            topology, sender->message_type, &channel->record_byte_length,
            &record_reason)) {
      return loom_aie2p_array_topology_reject_channel_record(
          topology, (uint32_t)i, sender->message_type, record_reason);
    }

    const loom_aie2p_array_endpoint_t* binding_view_endpoint = NULL;
    if (sender->binding_view_source_endpoint_index != UINT32_MAX) {
      binding_view_endpoint = sender;
    } else if (receiver->binding_view_source_endpoint_index != UINT32_MAX) {
      binding_view_endpoint = receiver;
    }
    if (binding_view_endpoint != NULL) {
      if (channel->record_count !=
          binding_view_endpoint->binding_view_record_count) {
        return loom_aie2p_array_topology_reject_channel_ring(
            topology, (uint32_t)i, IREE_SV("record count"),
            channel->record_count,
            binding_view_endpoint->binding_view_record_count,
            IREE_SV("the active binding view record count"), 3);
      }
    }
    const loom_aie2p_array_endpoint_t* base_sender =
        loom_aie2p_array_topology_base_endpoint(plan, sender);
    if (base_sender->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING &&
        receiver->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER) {
      const loom_aie2p_array_binding_t* binding =
          &plan->bindings[base_sender->owner_index];
      if ((binding->access & LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ) == 0) {
        return loom_aie2p_array_topology_reject_channel_connection(
            topology, (uint32_t)i,
            IREE_SV("the sending binding is write-only"));
      }
      channel->transport = LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA;
    } else if (sender->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER &&
               receiver->owner_kind ==
                   LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING) {
      const loom_aie2p_array_binding_t* binding =
          &plan->bindings[receiver->owner_index];
      if ((binding->access & LOOM_AIE2P_ARRAY_BINDING_ACCESS_WRITE) == 0) {
        return loom_aie2p_array_topology_reject_channel_connection(
            topology, (uint32_t)i,
            IREE_SV("the receiving binding is read-only"));
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
      return loom_aie2p_array_topology_reject_channel_connection(
          topology, (uint32_t)i,
          IREE_SV("binding-to-binding channels have no resident worker "
                  "endpoint"));
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
        endpoint->channel_use_count <= 1) {
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
      } else {
        const loom_aie2p_array_channel_t* source_channel =
            &topology->channels[source_channel_index];
        if (endpoint->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER &&
            channel->capacity != source_channel->capacity) {
          return loom_aie2p_array_topology_reject_channel_ring(
              topology, (uint32_t)channel_index, IREE_SV("capacity"),
              channel->capacity, source_channel->capacity,
              IREE_SV("the shared worker multicast capacity"), 2);
        }
        if (channel->record_count != source_channel->record_count) {
          const iree_string_view_t relationship =
              endpoint->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER
                  ? IREE_SV("the shared worker multicast record count")
                  : IREE_SV("the shared binding multicast record count");
          return loom_aie2p_array_topology_reject_channel_ring(
              topology, (uint32_t)channel_index, IREE_SV("record count"),
              channel->record_count, source_channel->record_count, relationship,
              3);
        }
      }
      channel->source_channel_index = source_channel_index;
      if (channel->transport ==
          LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_NEIGHBOR_MEMORY) {
        channel->transport = LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_ROUTED_DMA;
      }
    }
  }

  bool worker_interfaces_valid = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_topology_validate_worker_interfaces(
      topology, &worker_interfaces_valid));
  if (!worker_interfaces_valid) {
    return iree_ok_status();
  }
  bool worker_rates_valid = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_topology_validate_worker_rates(
      topology, &worker_rates_valid));
  if (!worker_rates_valid) {
    return iree_ok_status();
  }
  return loom_aie2p_array_topology_validate_worker_dependencies(topology,
                                                                out_valid);
}
