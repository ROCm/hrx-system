// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/bank_sroa_projection.h"

#include <inttypes.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/reporting/report.h"
#include "loom/transforms/boundary/projection_loop.h"

typedef struct loom_vector_bank_sroa_bank_plan_t
    loom_vector_bank_sroa_bank_plan_t;
typedef struct loom_vector_bank_sroa_endpoint_plan_t
    loom_vector_bank_sroa_endpoint_plan_t;

// Which independently typed LoopLike state domain owns a bank.
typedef enum loom_vector_bank_sroa_bank_domain_e {
  // Body arguments and operation results.
  LOOM_VECTOR_BANK_SROA_BANK_DOMAIN_RESULT = 0,
  // Condition-region arguments and the initial/backedge tuple.
  LOOM_VECTOR_BANK_SROA_BANK_DOMAIN_HEADER = 1,
} loom_vector_bank_sroa_bank_domain_t;

// Why a vector recurrence could not use one homogeneous component schema.
// The numeric order is the diagnostic priority when several blockers are
// discovered on the same recurrence.
typedef enum loom_vector_bank_sroa_blocker_e {
  LOOM_VECTOR_BANK_SROA_BLOCKER_NONE = 0,
  LOOM_VECTOR_BANK_SROA_BLOCKER_VALUE_METADATA_USE = 1,
  LOOM_VECTOR_BANK_SROA_BLOCKER_INCONSISTENT_COMPONENT_ACCESS = 2,
  LOOM_VECTOR_BANK_SROA_BLOCKER_NON_STATIC_COMPONENT_ACCESS = 3,
  LOOM_VECTOR_BANK_SROA_BLOCKER_COMPONENT_COUNT_LIMIT = 4,
} loom_vector_bank_sroa_blocker_t;

typedef struct loom_vector_bank_sroa_state_node_t {
  // Bank whose aggregate state this value represents.
  loom_vector_bank_sroa_bank_plan_t* bank;
  // Recurrence endpoint at the root of this insert chain.
  loom_vector_bank_sroa_endpoint_plan_t* endpoint;
  // Previous aggregate state, or NULL for an endpoint root.
  struct loom_vector_bank_sroa_state_node_t* parent;
  // Insert operation defining this state, or NULL for an endpoint root.
  loom_op_t* insert_op;
  // Payload value installed by insert_op.
  loom_value_id_t inserted_value_id;
  // Physical bank component installed by insert_op.
  uint16_t inserted_component;
  // Aggregate value represented by this node.
  loom_value_id_t value_id;
} loom_vector_bank_sroa_state_node_t;

typedef struct loom_vector_bank_sroa_extract_use_t {
  // Exact extract operation admitted during planning.
  loom_op_t* op;
  // Aggregate state node read by op.
  loom_vector_bank_sroa_state_node_t* source;
  // Physical component replacing the extract result.
  uint16_t component;
} loom_vector_bank_sroa_extract_use_t;

struct loom_vector_bank_sroa_endpoint_plan_t {
  // Original recurrence endpoint definition.
  loom_value_id_t value_id;
  // Generic projection candidate for value_id.
  iree_host_size_t candidate;
  // Block containing admitted insert chains, or NULL for the loop result.
  loom_block_t* block;
  // Boundary terminator consuming forwarded state, or NULL for a result.
  loom_op_t* boundary_terminator;
  // First boundary payload operand accepted from this endpoint.
  uint16_t boundary_operand_offset;
  // Number of boundary payload operands accepted from this endpoint.
  uint16_t boundary_operand_count;
  // Exact admitted extracts reachable from this endpoint.
  loom_vector_bank_sroa_extract_use_t* extracts;
  // Number of entries in extracts.
  iree_host_size_t extract_count;
  // Allocated capacity of extracts.
  iree_host_size_t extract_capacity;
  // Exact admitted inserts in definition order.
  loom_op_t** inserts;
  // Number of entries in inserts.
  iree_host_size_t insert_count;
  // Allocated capacity of inserts.
  iree_host_size_t insert_capacity;
  // Aggregate state nodes in discovery order.
  loom_vector_bank_sroa_state_node_t** nodes;
  // Number of entries in nodes.
  iree_host_size_t node_count;
  // Allocated capacity of nodes.
  iree_host_size_t node_capacity;
  // Whether every use can consume physical components directly.
  bool uses_supported;
};

typedef struct loom_vector_bank_sroa_loop_plan_t {
  // Generic recurrence topology owned by the boundary engine.
  loom_boundary_projection_loop_t* loop;
  // One potential bank per recurrence column.
  loom_vector_bank_sroa_bank_plan_t* banks;
  // Number of entries in banks.
  uint32_t bank_count;
} loom_vector_bank_sroa_loop_plan_t;

struct loom_vector_bank_sroa_bank_plan_t {
  // Loop containing this recurrence column.
  loom_vector_bank_sroa_loop_plan_t* loop;
  // Result/body or condition-header endpoint plans.
  loom_vector_bank_sroa_endpoint_plan_t endpoints[2];
  // Number of initialized entries in endpoints.
  uint8_t endpoint_count;
  // Original aggregate vector type.
  loom_type_t bank_type;
  // Scalar or trailing-vector type carried by every physical component.
  loom_type_t payload_type;
  // Component types retained by the generic schema.
  loom_type_t* component_types;
  // Derived-name suffixes retained by the generic schema.
  iree_string_view_t* component_name_suffixes;
  // Number of leading dimensions selecting one payload component.
  uint8_t prefix_rank;
  // Number of physical payload components.
  uint16_t component_count;
  // Independently typed LoopLike state domain owning this bank.
  loom_vector_bank_sroa_bank_domain_t domain;
  // Recurrence column ordinal within domain.
  uint16_t state_ordinal;
  // Highest-priority semantic blocker retained by the existing use-def scan.
  loom_vector_bank_sroa_blocker_t blocker;
  // Whether an admitted condition/body access selected this bank.
  bool active;
};

typedef struct loom_vector_bank_sroa_function_state_t {
  // Dense local-value lookup for aggregate state nodes.
  loom_vector_bank_sroa_state_node_t** nodes_by_ordinal;
  // Number of entries in nodes_by_ordinal.
  loom_value_ordinal_t value_count;
  // One retained plan per generic loop plan.
  loom_vector_bank_sroa_loop_plan_t* loops;
  // Number of entries in loops.
  iree_host_size_t loop_count;
} loom_vector_bank_sroa_function_state_t;

typedef enum loom_vector_bank_sroa_source_component_kind_e {
  // Extracts a component from a stable aggregate value.
  LOOM_VECTOR_BANK_SROA_SOURCE_EXTRACT = 0,
  // Forwards one component of an already projected endpoint.
  LOOM_VECTOR_BANK_SROA_SOURCE_SLOT = 1,
  // Forwards the payload installed by an admitted insert.
  LOOM_VECTOR_BANK_SROA_SOURCE_VALUE = 2,
} loom_vector_bank_sroa_source_component_kind_t;

typedef struct loom_vector_bank_sroa_source_component_t {
  // How this outgoing physical component is obtained.
  loom_vector_bank_sroa_source_component_kind_t kind;
  union {
    // Stable aggregate extracted by SOURCE_EXTRACT.
    loom_value_id_t aggregate_value_id;
    struct {
      // Projected endpoint supplying SOURCE_SLOT.
      iree_host_size_t candidate;
      // Component ordinal within candidate.
      uint16_t component;
    } slot;
    // Existing payload forwarded by SOURCE_VALUE.
    loom_value_id_t value_id;
  } source;
} loom_vector_bank_sroa_source_component_t;

typedef struct loom_vector_bank_sroa_source_plan_t {
  // Destination bank schema.
  const loom_vector_bank_sroa_bank_plan_t* bank;
  // One retained recipe per physical component.
  loom_vector_bank_sroa_source_component_t* components;
  // Number of entries in components.
  uint16_t component_count;
  // Whether newly extracted components receive source-derived names.
  bool name_components;
} loom_vector_bank_sroa_source_plan_t;

static bool loom_vector_bank_sroa_bank_uses_supported(
    const loom_vector_bank_sroa_bank_plan_t* bank);

static loom_vector_bank_sroa_function_state_t*
loom_vector_bank_sroa_function_state(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function) {
  return (loom_vector_bank_sroa_function_state_t*)
      loom_boundary_projection_function_rule_state(plan, function, rule);
}

static loom_op_t* loom_vector_bank_sroa_loop_op(
    const loom_vector_bank_sroa_bank_plan_t* bank) {
  return bank->loop->loop->loop.op;
}

static void loom_vector_bank_sroa_record_blocker(
    const loom_boundary_projection_plan_t* plan,
    loom_vector_bank_sroa_bank_plan_t* bank,
    loom_vector_bank_sroa_blocker_t blocker) {
  if (!plan->observation_requested) {
    return;
  }
  if (blocker > bank->blocker) {
    bank->blocker = blocker;
  }
}

static loom_vector_bank_sroa_state_node_t* loom_vector_bank_sroa_lookup_node(
    const loom_boundary_projection_function_t* function,
    const loom_vector_bank_sroa_function_state_t* state,
    loom_value_id_t value_id) {
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_try_ordinal(&function->domain, value_id);
  return ordinal == LOOM_VALUE_ORDINAL_INVALID
             ? NULL
             : state->nodes_by_ordinal[ordinal];
}

static void loom_vector_bank_sroa_set_node(
    const loom_boundary_projection_function_t* function,
    loom_vector_bank_sroa_function_state_t* state,
    loom_vector_bank_sroa_state_node_t* node) {
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_ordinal(&function->domain, node->value_id);
  IREE_ASSERT(state->nodes_by_ordinal[ordinal] == NULL);
  state->nodes_by_ordinal[ordinal] = node;
}

static iree_status_t loom_vector_bank_sroa_append_node(
    loom_boundary_projection_plan_t* plan,
    loom_vector_bank_sroa_endpoint_plan_t* endpoint,
    loom_vector_bank_sroa_state_node_t* node) {
  if (endpoint->node_count == endpoint->node_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, endpoint->node_count, endpoint->node_count + 1,
        sizeof(*endpoint->nodes), &endpoint->node_capacity,
        (void**)&endpoint->nodes));
  }
  endpoint->nodes[endpoint->node_count++] = node;
  return iree_ok_status();
}

static iree_status_t loom_vector_bank_sroa_append_extract(
    loom_boundary_projection_plan_t* plan,
    loom_vector_bank_sroa_endpoint_plan_t* endpoint, loom_op_t* op,
    loom_vector_bank_sroa_state_node_t* source, uint16_t component) {
  if (endpoint->extract_count == endpoint->extract_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, endpoint->extract_count, endpoint->extract_count + 1,
        sizeof(*endpoint->extracts), &endpoint->extract_capacity,
        (void**)&endpoint->extracts));
  }
  endpoint->extracts[endpoint->extract_count++] =
      (loom_vector_bank_sroa_extract_use_t){
          .op = op,
          .source = source,
          .component = component,
      };
  return iree_ok_status();
}

static iree_status_t loom_vector_bank_sroa_append_insert(
    loom_boundary_projection_plan_t* plan,
    loom_vector_bank_sroa_endpoint_plan_t* endpoint, loom_op_t* op) {
  if (endpoint->insert_count == endpoint->insert_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, endpoint->insert_count, endpoint->insert_count + 1,
        sizeof(*endpoint->inserts), &endpoint->insert_capacity,
        (void**)&endpoint->inserts));
  }
  endpoint->inserts[endpoint->insert_count++] = op;
  return iree_ok_status();
}

static iree_status_t loom_vector_bank_sroa_payload_type(
    loom_module_t* module, loom_type_t bank_type, uint8_t prefix_rank,
    loom_type_t* out_payload_type) {
  const uint8_t bank_rank = loom_type_rank(bank_type);
  const uint8_t payload_rank = (uint8_t)(bank_rank - prefix_rank);
  if (payload_rank == 0) {
    *out_payload_type = loom_type_scalar(loom_type_element_type(bank_type));
    return iree_ok_status();
  }

  loom_overflow_dim_t payload_dims[LOOM_TYPE_MAX_RANK] = {0};
  for (uint8_t i = 0; i < payload_rank; ++i) {
    payload_dims[i] = loom_type_dim(bank_type, (uint8_t)(prefix_rank + i));
  }
  loom_type_t payload_type = {0};
  uint8_t flags = LOOM_TYPE_FLAG_ALL_STATIC;
  if (payload_rank <= 2) {
    flags |= LOOM_TYPE_FLAG_INLINE_DIMS;
  }
  payload_type.header = loom_type_make_header(
      LOOM_TYPE_VECTOR, loom_type_element_type(bank_type), payload_rank, flags);
  if (payload_rank <= 2) {
    for (uint8_t i = 0; i < payload_rank; ++i) {
      payload_type.dims[i] = payload_dims[i];
    }
  } else {
    payload_type.dims[0] = (uint64_t)(uintptr_t)payload_dims;
  }
  return loom_module_intern_type(module, payload_type, out_payload_type);
}

static bool loom_vector_bank_sroa_static_access_component(
    const loom_vector_bank_sroa_bank_plan_t* bank,
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    loom_type_t payload_type, uint16_t* out_component) {
  *out_component = 0;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count != bank->prefix_rank ||
      (static_indices.count != 0 && !static_indices.i64_array) ||
      dynamic_indices.count != 0 ||
      !loom_type_equal(payload_type, bank->payload_type)) {
    return false;
  }

  uint32_t component = 0;
  for (uint8_t axis = 0; axis < bank->prefix_rank; ++axis) {
    const int64_t index = static_indices.i64_array[axis];
    const int64_t extent = loom_type_dim_static_size_at(bank->bank_type, axis);
    if (index < 0 || index == INT64_MIN || index >= extent) {
      return false;
    }
    component = component * (uint32_t)extent + (uint32_t)index;
  }
  if (component >= bank->component_count) {
    return false;
  }
  *out_component = (uint16_t)component;
  return true;
}

static iree_status_t loom_vector_bank_sroa_prepare_access(
    loom_boundary_projection_plan_t* plan,
    loom_vector_bank_sroa_bank_plan_t* bank, loom_attribute_t static_indices,
    loom_value_slice_t dynamic_indices, loom_type_t payload_type,
    bool allow_activation, uint16_t* out_component, bool* out_supported) {
  *out_component = 0;
  *out_supported = false;
  const uint8_t bank_rank = loom_type_rank(bank->bank_type);
  if (!loom_type_is_all_static(bank->bank_type) ||
      static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count == 0 ||
      static_indices.count > bank_rank || !static_indices.i64_array ||
      dynamic_indices.count != 0) {
    loom_vector_bank_sroa_record_blocker(
        plan, bank, LOOM_VECTOR_BANK_SROA_BLOCKER_NON_STATIC_COMPONENT_ACCESS);
    return iree_ok_status();
  }
  for (uint16_t axis = 0; axis < static_indices.count; ++axis) {
    const int64_t index = static_indices.i64_array[axis];
    const int64_t extent =
        loom_type_dim_static_size_at(bank->bank_type, (uint8_t)axis);
    if (index < 0 || index == INT64_MIN || index >= extent) {
      loom_vector_bank_sroa_record_blocker(
          plan, bank,
          LOOM_VECTOR_BANK_SROA_BLOCKER_INCONSISTENT_COMPONENT_ACCESS);
      return iree_ok_status();
    }
  }

  if (!bank->active) {
    if (!allow_activation) {
      return iree_ok_status();
    }
    bank->prefix_rank = (uint8_t)static_indices.count;
    IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_payload_type(
        plan->module, bank->bank_type, bank->prefix_rank, &bank->payload_type));
    uint32_t component_count = 1;
    for (uint8_t axis = 0; axis < bank->prefix_rank; ++axis) {
      const int64_t extent =
          loom_type_dim_static_size_at(bank->bank_type, axis);
      if (extent <= 0 ||
          (uint64_t)component_count * (uint64_t)extent > UINT16_MAX) {
        loom_vector_bank_sroa_record_blocker(
            plan, bank, LOOM_VECTOR_BANK_SROA_BLOCKER_COMPONENT_COUNT_LIMIT);
        return iree_ok_status();
      }
      component_count *= (uint32_t)extent;
    }
    bank->component_count = (uint16_t)component_count;
    bank->active = true;
  }

  *out_supported = loom_vector_bank_sroa_static_access_component(
      bank, static_indices, dynamic_indices, payload_type, out_component);
  if (!*out_supported) {
    loom_vector_bank_sroa_record_blocker(
        plan, bank,
        LOOM_VECTOR_BANK_SROA_BLOCKER_INCONSISTENT_COMPONENT_ACCESS);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_bank_sroa_initialize_endpoint(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_vector_bank_sroa_function_state_t* function_state,
    loom_vector_bank_sroa_bank_plan_t* bank,
    loom_vector_bank_sroa_endpoint_plan_t* endpoint, loom_value_id_t value_id,
    iree_host_size_t candidate, loom_block_t* block,
    loom_op_t* boundary_terminator, uint16_t boundary_operand_offset,
    uint16_t boundary_operand_count) {
  *endpoint = (loom_vector_bank_sroa_endpoint_plan_t){
      .value_id = value_id,
      .candidate = candidate,
      .block = block,
      .boundary_terminator = boundary_terminator,
      .boundary_operand_offset = boundary_operand_offset,
      .boundary_operand_count = boundary_operand_count,
      .uses_supported = true,
  };
  loom_vector_bank_sroa_state_node_t* root = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan->arena, sizeof(*root), (void**)&root));
  *root = (loom_vector_bank_sroa_state_node_t){
      .bank = bank,
      .endpoint = endpoint,
      .inserted_value_id = LOOM_VALUE_ID_INVALID,
      .value_id = value_id,
  };
  loom_vector_bank_sroa_set_node(function, function_state, root);
  return loom_vector_bank_sroa_append_node(plan, endpoint, root);
}

static iree_status_t loom_vector_bank_sroa_scan_endpoint(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_vector_bank_sroa_function_state_t* function_state,
    loom_vector_bank_sroa_endpoint_plan_t* endpoint, bool allow_activation,
    bool* out_invalid_access) {
  *out_invalid_access = false;
  loom_vector_bank_sroa_bank_plan_t* bank = endpoint->nodes[0]->bank;
  for (iree_host_size_t node_index = 0; node_index < endpoint->node_count;
       ++node_index) {
    loom_vector_bank_sroa_state_node_t* node = endpoint->nodes[node_index];
    const loom_value_t* value = loom_module_value(plan->module, node->value_id);
    if (loom_value_has_attribute_uses(value) ||
        loom_module_value_has_type_uses(plan->module, node->value_id)) {
      endpoint->uses_supported = false;
      loom_vector_bank_sroa_record_blocker(
          plan, bank, LOOM_VECTOR_BANK_SROA_BLOCKER_VALUE_METADATA_USE);
    }
    const loom_use_t* use = NULL;
    loom_value_for_each_use(value, use) {
      loom_op_t* user = loom_use_user_op(*use);
      const uint16_t operand_index = loom_use_operand_index(*use);
      if (user == endpoint->boundary_terminator &&
          operand_index >= endpoint->boundary_operand_offset &&
          operand_index < endpoint->boundary_operand_offset +
                              endpoint->boundary_operand_count) {
        continue;
      }

      if (loom_vector_extract_isa(user) && operand_index == 0 &&
          (!endpoint->block || user->parent_block == endpoint->block)) {
        uint16_t component = 0;
        bool supported = false;
        IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_prepare_access(
            plan, bank, loom_vector_extract_static_indices(user),
            loom_vector_extract_indices(user),
            loom_module_value_type(plan->module,
                                   loom_vector_extract_result(user)),
            allow_activation, &component, &supported));
        if (!supported) {
          endpoint->uses_supported = false;
          *out_invalid_access = true;
          return iree_ok_status();
        }
        IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_append_extract(
            plan, endpoint, user, node, component));
        continue;
      }

      if (endpoint->block && loom_vector_insert_isa(user) &&
          operand_index == 1 && user->parent_block == endpoint->block) {
        uint16_t component = 0;
        bool supported = false;
        IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_prepare_access(
            plan, bank, loom_vector_insert_static_indices(user),
            loom_vector_insert_indices(user),
            loom_module_value_type(plan->module,
                                   loom_vector_insert_value(user)),
            allow_activation, &component, &supported));
        if (!supported) {
          endpoint->uses_supported = false;
          *out_invalid_access = true;
          return iree_ok_status();
        }
        loom_vector_bank_sroa_state_node_t* child = NULL;
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(plan->arena, sizeof(*child), (void**)&child));
        *child = (loom_vector_bank_sroa_state_node_t){
            .bank = bank,
            .endpoint = endpoint,
            .parent = node,
            .insert_op = user,
            .inserted_value_id = loom_vector_insert_value(user),
            .inserted_component = component,
            .value_id = loom_vector_insert_result(user),
        };
        loom_vector_bank_sroa_set_node(function, function_state, child);
        IREE_RETURN_IF_ERROR(
            loom_vector_bank_sroa_append_node(plan, endpoint, child));
        IREE_RETURN_IF_ERROR(
            loom_vector_bank_sroa_append_insert(plan, endpoint, user));
        continue;
      }

      endpoint->uses_supported = false;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_bank_sroa_prepare_bank_schema(
    loom_boundary_projection_plan_t* plan,
    loom_vector_bank_sroa_bank_plan_t* bank) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, bank->component_count, sizeof(*bank->component_types),
      (void**)&bank->component_types));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan->arena, bank->component_count,
                                sizeof(*bank->component_name_suffixes),
                                (void**)&bank->component_name_suffixes));
  const iree_host_size_t suffix_capacity = 32;
  char* suffix_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      plan->arena, (iree_host_size_t)bank->component_count * suffix_capacity,
      (void**)&suffix_storage));
  for (uint16_t component = 0; component < bank->component_count; ++component) {
    bank->component_types[component] = bank->payload_type;
    char* suffix = suffix_storage + component * suffix_capacity;
    const int length =
        iree_snprintf(suffix, suffix_capacity, "slot_%" PRIu16, component);
    IREE_ASSERT(length > 0 && (iree_host_size_t)length < suffix_capacity);
    bank->component_name_suffixes[component] =
        iree_make_string_view(suffix, (iree_host_size_t)length);
  }
  return iree_ok_status();
}

// An eliminative source bank can remain selected only when every loop edge
// consuming one of its aggregate states also projects the destination. The
// ordinary source planner adds the converse dependency when it chooses direct
// component forwarding, making the two banks one availability component only
// when both sides are selected.
static iree_status_t loom_vector_bank_sroa_require_projected_destination(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_vector_bank_sroa_function_state_t* function_state,
    loom_value_id_t source_value_id, iree_host_size_t destination_candidate) {
  loom_vector_bank_sroa_state_node_t* source_node =
      loom_vector_bank_sroa_lookup_node(function, function_state,
                                        source_value_id);
  if (!source_node) {
    return iree_ok_status();
  }
  if (destination_candidate == IREE_HOST_SIZE_MAX) {
    source_node->endpoint->uses_supported = false;
    return iree_ok_status();
  }
  const iree_host_size_t source_candidate = source_node->endpoint->candidate;
  if (source_candidate == destination_candidate) {
    return iree_ok_status();
  }
  return loom_boundary_projection_add_dependency(
      plan, function, destination_candidate, source_candidate,
      /*orders_realization=*/false);
}

static iree_status_t loom_vector_bank_sroa_plan_loop_edge_dependencies(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_vector_bank_sroa_function_state_t* function_state,
    loom_vector_bank_sroa_loop_plan_t* loop_plan) {
  const loom_boundary_projection_loop_t* loop = loop_plan->loop;
  const loom_value_slice_t initial_values =
      loom_loop_like_iter_args(loop->loop);
  if (loop->condition_terminator) {
    const loom_value_id_t* condition_values =
        loom_op_const_operands(loop->condition_terminator) + 1;
    for (uint16_t i = 0; i < loop->result_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_require_projected_destination(
          plan, function, function_state, condition_values[i],
          loop->result_states[i].body_candidate));
    }
    const loom_value_id_t* backedge_values =
        loom_op_const_operands(loop->body_terminator);
    for (uint16_t i = 0; i < loop->header_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_require_projected_destination(
          plan, function, function_state, initial_values.values[i],
          loop->header_states[i].candidate));
      IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_require_projected_destination(
          plan, function, function_state, backedge_values[i],
          loop->header_states[i].candidate));
    }
    return iree_ok_status();
  }

  const loom_value_id_t* backedge_values =
      loom_op_const_operands(loop->body_terminator);
  for (uint16_t i = 0; i < loop->result_count; ++i) {
    const iree_host_size_t destination_candidate =
        loop->result_states[i].body_candidate;
    IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_require_projected_destination(
        plan, function, function_state, initial_values.values[i],
        destination_candidate));
    IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_require_projected_destination(
        plan, function, function_state, backedge_values[i],
        destination_candidate));
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_bank_sroa_prepare_loop(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_vector_bank_sroa_function_state_t* function_state,
    loom_vector_bank_sroa_loop_plan_t* loop_plan) {
  loom_boundary_projection_loop_t* loop = loop_plan->loop;
  loop_plan->bank_count =
      (uint32_t)loop->result_count +
      (loop->condition_terminator ? (uint32_t)loop->header_count : 0u);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, loop_plan->bank_count, sizeof(*loop_plan->banks),
      (void**)&loop_plan->banks));
  memset(loop_plan->banks, 0,
         loop_plan->bank_count * sizeof(*loop_plan->banks));

  const loom_value_slice_t initial_values =
      loom_loop_like_iter_args(loop->loop);
  const uint16_t body_boundary_count =
      loop->condition_terminator ? loop->header_count : loop->result_count;
  for (uint16_t i = 0; i < loop->result_count; ++i) {
    loom_vector_bank_sroa_bank_plan_t* bank = &loop_plan->banks[i];
    bank->loop = loop_plan;
    bank->domain = LOOM_VECTOR_BANK_SROA_BANK_DOMAIN_RESULT;
    bank->state_ordinal = i;
    const loom_boundary_projection_loop_result_state_t* state =
        &loop->result_states[i];
    if (state->result_candidate == IREE_HOST_SIZE_MAX) {
      continue;
    }
    bank->bank_type =
        loom_module_value_type(plan->module, state->result_value_id);
    if (!loom_type_is_vector(bank->bank_type) ||
        loom_type_rank(bank->bank_type) == 0 ||
        !loom_type_equal(
            bank->bank_type,
            loom_module_value_type(plan->module, state->body_value_id)) ||
        (!loop->condition_terminator &&
         !loom_type_equal(
             bank->bank_type,
             loom_module_value_type(plan->module, initial_values.values[i])))) {
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_initialize_endpoint(
        plan, function, function_state, bank, &bank->endpoints[0],
        state->result_value_id, state->result_candidate,
        /*block=*/NULL, /*boundary_terminator=*/NULL,
        /*boundary_operand_offset=*/0, /*boundary_operand_count=*/0));
    IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_initialize_endpoint(
        plan, function, function_state, bank, &bank->endpoints[1],
        state->body_value_id, state->body_candidate,
        loom_region_entry_block(loom_loop_like_body(loop->loop)),
        loop->body_terminator, /*boundary_operand_offset=*/0,
        body_boundary_count));
    bank->endpoint_count = 2;
  }

  if (loop->condition_terminator) {
    for (uint16_t i = 0; i < loop->header_count; ++i) {
      loom_vector_bank_sroa_bank_plan_t* bank =
          &loop_plan->banks[loop->result_count + i];
      bank->loop = loop_plan;
      bank->domain = LOOM_VECTOR_BANK_SROA_BANK_DOMAIN_HEADER;
      bank->state_ordinal = i;
      const loom_boundary_projection_loop_header_state_t* state =
          &loop->header_states[i];
      if (state->candidate == IREE_HOST_SIZE_MAX) {
        continue;
      }
      bank->bank_type = loom_module_value_type(plan->module, state->value_id);
      if (!loom_type_is_vector(bank->bank_type) ||
          loom_type_rank(bank->bank_type) == 0) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_initialize_endpoint(
          plan, function, function_state, bank, &bank->endpoints[0],
          state->value_id, state->candidate,
          loom_region_entry_block(loom_loop_like_condition_region(loop->loop)),
          loop->condition_terminator, /*boundary_operand_offset=*/1,
          loop->result_count));
      bank->endpoint_count = 1;
    }
  }

  for (uint32_t bank_index = 0; bank_index < loop_plan->bank_count;
       ++bank_index) {
    loom_vector_bank_sroa_bank_plan_t* bank = &loop_plan->banks[bank_index];
    if (bank->endpoint_count == 0) {
      continue;
    }
    bool invalid_access = false;
    const uint8_t activation_endpoint =
        bank->domain == LOOM_VECTOR_BANK_SROA_BANK_DOMAIN_RESULT ? 1 : 0;
    for (uint8_t endpoint_index = activation_endpoint;
         endpoint_index < bank->endpoint_count; ++endpoint_index) {
      IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_scan_endpoint(
          plan, function, function_state, &bank->endpoints[endpoint_index],
          /*allow_activation=*/true, &invalid_access));
      if (invalid_access) {
        break;
      }
    }
    if (invalid_access || !bank->active || activation_endpoint == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_scan_endpoint(
        plan, function, function_state, &bank->endpoints[0],
        /*allow_activation=*/false, &invalid_access));
  }
  IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_plan_loop_edge_dependencies(
      plan, function, function_state, loop_plan));
  // Leave independent unsupported or target-native recurrence columns intact.
  // The shared projection plan propagates rejection only across actual SSA
  // dependencies between columns.
  for (uint32_t bank_index = 0; bank_index < loop_plan->bank_count;
       ++bank_index) {
    loom_vector_bank_sroa_bank_plan_t* bank = &loop_plan->banks[bank_index];
    if (bank->active && loom_vector_bank_sroa_bank_uses_supported(bank)) {
      IREE_RETURN_IF_ERROR(
          loom_vector_bank_sroa_prepare_bank_schema(plan, bank));
    }
  }
  return iree_ok_status();
}

static bool loom_vector_bank_sroa_function_applies(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function) {
  (void)rule;
  // Target pipelines provide a concrete version set. Keep the transform on
  // that selected set while direct pass invocations without versions retain
  // the ordinary all-functions behavior.
  return plan->versions.version_handles_by_symbol == NULL ||
         function->version != NULL;
}

static bool loom_vector_bank_sroa_slot_matches(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block) {
  (void)rule;
  (void)function;
  (void)block;
  const loom_type_t type = loom_module_value_type(plan->module, value_id);
  return role == LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE &&
         loom_type_is_vector(type) && loom_type_rank(type) != 0;
}

static iree_status_t loom_vector_bank_sroa_prepare_function(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  loom_vector_bank_sroa_function_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan->arena, sizeof(*state), (void**)&state));
  *state = (loom_vector_bank_sroa_function_state_t){
      .value_count = function->domain.value_count,
      .loop_count = function->loop_count,
  };
  if (state->value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, state->value_count, sizeof(*state->nodes_by_ordinal),
        (void**)&state->nodes_by_ordinal));
    memset(state->nodes_by_ordinal, 0,
           state->value_count * sizeof(*state->nodes_by_ordinal));
  }
  if (state->loop_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, state->loop_count, sizeof(*state->loops),
        (void**)&state->loops));
    memset(state->loops, 0, state->loop_count * sizeof(*state->loops));
  }
  loom_boundary_projection_set_function_rule_state(plan, function, rule, state);

  for (iree_host_size_t i = 0; i < state->loop_count; ++i) {
    state->loops[i].loop = &function->loops[i];
    IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_prepare_loop(
        plan, function, state, &state->loops[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_bank_sroa_plan_slot(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block, loom_boundary_projection_schema_t* out_schema,
    bool* out_claimed) {
  (void)block;
  *out_schema = (loom_boundary_projection_schema_t){0};
  *out_claimed = false;
  if (role != LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE) {
    return iree_ok_status();
  }
  const loom_vector_bank_sroa_function_state_t* state =
      loom_vector_bank_sroa_function_state(rule, plan, function);
  IREE_ASSERT(state != NULL);
  loom_vector_bank_sroa_state_node_t* node =
      loom_vector_bank_sroa_lookup_node(function, state, value_id);
  if (!node || node->parent || node->endpoint != &node->bank->endpoints[0] ||
      !node->bank->active ||
      !loom_vector_bank_sroa_bank_uses_supported(node->bank)) {
    return iree_ok_status();
  }

  const loom_vector_bank_sroa_bank_plan_t* bank = node->bank;
  *out_schema = (loom_boundary_projection_schema_t){
      .rule = rule,
      .component_types = bank->component_types,
      .component_name_suffixes = bank->component_name_suffixes,
      .component_count = bank->component_count,
      .destination_mode = LOOM_BOUNDARY_PROJECTION_DESTINATION_ELIMINATE,
      .rule_plan = node->bank,
  };
  *out_claimed = true;
  return iree_ok_status();
}

static loom_value_id_t loom_vector_bank_sroa_inserted_component(
    const loom_vector_bank_sroa_state_node_t* node, uint16_t component) {
  for (const loom_vector_bank_sroa_state_node_t* current = node;
       current->parent; current = current->parent) {
    if (current->inserted_component == component) {
      return current->inserted_value_id;
    }
  }
  return LOOM_VALUE_ID_INVALID;
}

static iree_status_t loom_vector_bank_sroa_plan_source(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_slot_t* destination,
    const loom_boundary_projection_schema_t* schema,
    loom_value_id_t source_value_id, loom_op_t* boundary_op,
    loom_boundary_projection_source_t* out_source, bool* out_planned) {
  *out_source = (loom_boundary_projection_source_t){0};
  *out_planned = false;
  IREE_ASSERT(destination != NULL);
  IREE_ASSERT(schema->rule == rule);
  if (!loom_type_equal(loom_module_value_type(plan->module, source_value_id),
                       destination->logical_type)) {
    return iree_ok_status();
  }

  const loom_vector_bank_sroa_bank_plan_t* destination_bank =
      (const loom_vector_bank_sroa_bank_plan_t*)schema->rule_plan;
  loom_vector_bank_sroa_source_plan_t* source_plan = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(plan->arena, sizeof(*source_plan),
                                           (void**)&source_plan));
  source_plan->bank = destination_bank;
  source_plan->component_count = schema->component_count;
  source_plan->name_components =
      boundary_op == loom_vector_bank_sroa_loop_op(destination_bank);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, source_plan->component_count,
      sizeof(*source_plan->components), (void**)&source_plan->components));

  const loom_vector_bank_sroa_function_state_t* state =
      loom_vector_bank_sroa_function_state(rule, plan, function);
  IREE_ASSERT(state != NULL);
  loom_vector_bank_sroa_state_node_t* source_node =
      loom_vector_bank_sroa_lookup_node(function, state, source_value_id);
  iree_host_size_t source_candidate = IREE_HOST_SIZE_MAX;
  if (source_node) {
    source_candidate = source_node->endpoint->candidate;
    const loom_boundary_projection_slot_t* source_slot =
        &function->candidates[source_candidate];
    if (!source_slot->selected) {
      source_node = NULL;
    } else if (source_slot->schema.rule != rule ||
               source_slot->schema.component_count != schema->component_count ||
               !loom_type_equal(source_node->bank->payload_type,
                                destination_bank->payload_type)) {
      return iree_ok_status();
    }
  }

  for (uint16_t component = 0; component < source_plan->component_count;
       ++component) {
    loom_vector_bank_sroa_source_component_t* source_component =
        &source_plan->components[component];
    if (!source_node) {
      *source_component = (loom_vector_bank_sroa_source_component_t){
          .kind = LOOM_VECTOR_BANK_SROA_SOURCE_EXTRACT,
          .source.aggregate_value_id = source_value_id,
      };
      continue;
    }
    const loom_value_id_t inserted =
        loom_vector_bank_sroa_inserted_component(source_node, component);
    if (inserted != LOOM_VALUE_ID_INVALID) {
      *source_component = (loom_vector_bank_sroa_source_component_t){
          .kind = LOOM_VECTOR_BANK_SROA_SOURCE_VALUE,
          .source.value_id = inserted,
      };
    } else {
      *source_component = (loom_vector_bank_sroa_source_component_t){
          .kind = LOOM_VECTOR_BANK_SROA_SOURCE_SLOT,
          .source.slot =
              {
                  .candidate = source_candidate,
                  .component = component,
              },
      };
    }
  }

  if (source_node) {
    const iree_host_size_t destination_candidate =
        loom_boundary_projection_slot_index(function, destination->value_id);
    IREE_ASSERT_NE(destination_candidate, IREE_HOST_SIZE_MAX);
    if (source_candidate != destination_candidate) {
      IREE_RETURN_IF_ERROR(loom_boundary_projection_add_dependency(
          plan, function, source_candidate, destination_candidate,
          /*orders_realization=*/false));
    }
  }
  *out_source = (loom_boundary_projection_source_t){
      .rule = rule,
      .rule_plan = source_plan,
      .boundary_op = boundary_op,
  };
  *out_planned = true;
  return iree_ok_status();
}

static void loom_vector_bank_sroa_component_indices(
    const loom_vector_bank_sroa_bank_plan_t* bank, uint16_t component,
    int64_t* out_indices) {
  uint32_t remaining = component;
  for (uint8_t i = bank->prefix_rank; i > 0; --i) {
    const uint8_t axis = (uint8_t)(i - 1);
    const uint32_t extent =
        (uint32_t)loom_type_dim_static_size_at(bank->bank_type, axis);
    out_indices[axis] = remaining % extent;
    remaining /= extent;
  }
}

static iree_status_t loom_vector_bank_sroa_materialize_source(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_source_t* source,
    loom_value_id_t* out_component_values) {
  IREE_ASSERT(source->rule == rule);
  const loom_vector_bank_sroa_source_plan_t* source_plan =
      (const loom_vector_bank_sroa_source_plan_t*)source->rule_plan;
  IREE_ASSERT(source_plan != NULL);
  for (uint16_t component = 0; component < source_plan->component_count;
       ++component) {
    const loom_vector_bank_sroa_source_component_t* source_component =
        &source_plan->components[component];
    if (source_component->kind == LOOM_VECTOR_BANK_SROA_SOURCE_SLOT) {
      const loom_boundary_projection_slot_t* slot =
          &function->candidates[source_component->source.slot.candidate];
      out_component_values[component] =
          slot->component_value_ids[source_component->source.slot.component];
      IREE_ASSERT_NE(out_component_values[component], LOOM_VALUE_ID_INVALID);
      continue;
    }
    if (source_component->kind == LOOM_VECTOR_BANK_SROA_SOURCE_VALUE) {
      out_component_values[component] = loom_boundary_projection_resolve_value(
          function, source_component->source.value_id);
      continue;
    }

    const loom_value_id_t aggregate_value_id =
        loom_boundary_projection_resolve_value(
            function, source_component->source.aggregate_value_id);
    int64_t indices[LOOM_TYPE_MAX_RANK] = {0};
    loom_vector_bank_sroa_component_indices(source_plan->bank, component,
                                            indices);
    loom_op_t* extract_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_extract_build(
        &plan->rewriter.builder, aggregate_value_id,
        /*indices=*/NULL, /*indices_count=*/0, indices,
        source_plan->bank->prefix_rank, source_plan->bank->payload_type,
        source->boundary_op->location, &extract_op));
    out_component_values[component] = loom_vector_extract_result(extract_op);
    if (source_plan->name_components) {
      IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
          &plan->rewriter, aggregate_value_id, out_component_values[component],
          source_plan->bank->component_name_suffixes[component]));
    }
  }
  return iree_ok_status();
}

static loom_value_id_t loom_vector_bank_sroa_resolve_component(
    const loom_boundary_projection_slot_t* slot,
    const loom_vector_bank_sroa_state_node_t* node, uint16_t component) {
  const loom_value_id_t inserted =
      loom_vector_bank_sroa_inserted_component(node, component);
  return inserted != LOOM_VALUE_ID_INVALID
             ? inserted
             : slot->component_value_ids[component];
}

static iree_status_t loom_vector_bank_sroa_eliminate(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_t* slot, loom_type_t logical_type,
    loom_location_id_t location) {
  (void)logical_type;
  (void)location;
  IREE_ASSERT(slot->schema.rule == rule);
  loom_vector_bank_sroa_bank_plan_t* bank =
      (loom_vector_bank_sroa_bank_plan_t*)slot->schema.rule_plan;
  IREE_ASSERT(bank != NULL);
  loom_vector_bank_sroa_endpoint_plan_t* endpoint = NULL;
  for (uint8_t i = 0; i < bank->endpoint_count; ++i) {
    if (bank->endpoints[i].value_id == slot->value_id) {
      endpoint = &bank->endpoints[i];
      break;
    }
  }
  IREE_ASSERT(endpoint != NULL && endpoint->node_count != 0);
  loom_vector_bank_sroa_state_node_t* root = endpoint->nodes[0];
  IREE_ASSERT(root->parent == NULL);

  for (iree_host_size_t i = 0; i < endpoint->extract_count; ++i) {
    const loom_vector_bank_sroa_extract_use_t* extract = &endpoint->extracts[i];
    const loom_value_id_t replacement = loom_vector_bank_sroa_resolve_component(
        slot, extract->source, extract->component);
    IREE_RETURN_IF_ERROR(loom_boundary_projection_record_value_replacement(
        function, loom_vector_extract_result(extract->op), replacement));
    IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
        &plan->rewriter, loom_vector_extract_result(extract->op), replacement));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        &plan->rewriter, extract->op, &replacement, 1));
  }
  for (iree_host_size_t i = endpoint->insert_count; i > 0; --i) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_erase(&plan->rewriter, endpoint->inserts[i - 1]));
  }

  loom_boundary_projection_record_destination_uses(
      plan, rule, (int64_t)endpoint->extract_count);
  loom_boundary_projection_record_source_operations(
      plan, rule, (int64_t)endpoint->insert_count);
  if (endpoint == &bank->endpoints[0]) {
    loom_boundary_projection_record(plan, rule, 1,
                                    slot->schema.component_count);
  }
  return iree_ok_status();
}

static iree_string_view_t loom_vector_bank_sroa_function_name(
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function) {
  const loom_symbol_ref_t symbol_ref =
      loom_func_like_callee(function->function);
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= plan->module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol =
      &plan->module->symbols.entries[symbol_ref.symbol_id];
  return symbol->name_id < plan->module->strings.count
             ? loom_string_table_get(&plan->module->strings, symbol->name_id)
             : IREE_SV("<unnamed>");
}

static iree_string_view_t loom_vector_bank_sroa_blocker_name(
    loom_vector_bank_sroa_blocker_t blocker) {
  switch (blocker) {
    case LOOM_VECTOR_BANK_SROA_BLOCKER_NONE:
      return IREE_SV("none");
    case LOOM_VECTOR_BANK_SROA_BLOCKER_VALUE_METADATA_USE:
      return IREE_SV("value_metadata_use");
    case LOOM_VECTOR_BANK_SROA_BLOCKER_INCONSISTENT_COMPONENT_ACCESS:
      return IREE_SV("inconsistent_component_access");
    case LOOM_VECTOR_BANK_SROA_BLOCKER_NON_STATIC_COMPONENT_ACCESS:
      return IREE_SV("non_static_component_access");
    case LOOM_VECTOR_BANK_SROA_BLOCKER_COMPONENT_COUNT_LIMIT:
      return IREE_SV("component_count_limit");
  }
  return IREE_SV("none");
}

static bool loom_vector_bank_sroa_blocker_is_component_access(
    loom_vector_bank_sroa_blocker_t blocker) {
  return blocker ==
             LOOM_VECTOR_BANK_SROA_BLOCKER_INCONSISTENT_COMPONENT_ACCESS ||
         blocker == LOOM_VECTOR_BANK_SROA_BLOCKER_NON_STATIC_COMPONENT_ACCESS ||
         blocker == LOOM_VECTOR_BANK_SROA_BLOCKER_COMPONENT_COUNT_LIMIT;
}

static bool loom_vector_bank_sroa_bank_uses_supported(
    const loom_vector_bank_sroa_bank_plan_t* bank) {
  for (uint8_t i = 0; i < bank->endpoint_count; ++i) {
    if (!bank->endpoints[i].uses_supported) {
      return false;
    }
  }
  return true;
}

static void loom_vector_bank_sroa_report_decision(
    const loom_boundary_projection_function_t* function,
    const loom_vector_bank_sroa_bank_plan_t* bank,
    iree_string_view_t* out_outcome, iree_string_view_t* out_reason) {
  const loom_boundary_projection_slot_t* candidate =
      &function->candidates[bank->endpoints[0].candidate];
  const bool uses_supported = loom_vector_bank_sroa_bank_uses_supported(bank);
  if (function->selected && candidate->selected) {
    *out_outcome = IREE_SV("selected");
    *out_reason = IREE_SV("static_component_accesses");
    return;
  }

  if (bank->active) {
    *out_outcome = IREE_SV("rejected");
    if (bank->blocker != LOOM_VECTOR_BANK_SROA_BLOCKER_NONE) {
      *out_reason = loom_vector_bank_sroa_blocker_name(bank->blocker);
    } else if (!uses_supported) {
      *out_reason = IREE_SV("unsupported_value_use");
    } else {
      *out_reason = IREE_SV("boundary_projection_rejected");
    }
    return;
  }

  if (loom_vector_bank_sroa_blocker_is_component_access(bank->blocker)) {
    *out_outcome = IREE_SV("rejected");
    *out_reason = loom_vector_bank_sroa_blocker_name(bank->blocker);
  } else {
    *out_outcome = IREE_SV("preserved");
    *out_reason =
        bank->blocker == LOOM_VECTOR_BANK_SROA_BLOCKER_VALUE_METADATA_USE
            ? IREE_SV("value_metadata_use")
        : !uses_supported ? IREE_SV("whole_value_use")
                          : IREE_SV("no_component_access");
  }
}

iree_status_t loom_vector_bank_sroa_record_projection_plan(
    const loom_boundary_projection_plan_t* plan,
    loom_target_compile_report_t* report) {
  const loom_boundary_projection_rule_t* rule =
      loom_vector_bank_sroa_boundary_projection_rule();
  for (iree_host_size_t function_index = 0;
       function_index < plan->function_count; ++function_index) {
    const loom_boundary_projection_function_t* function =
        &plan->functions[function_index];
    if (!function->rule_states) {
      continue;
    }
    const loom_vector_bank_sroa_function_state_t* state =
        loom_vector_bank_sroa_function_state(rule, plan, function);
    if (!state) {
      continue;
    }
    const iree_string_view_t function_name =
        loom_vector_bank_sroa_function_name(plan, function);
    for (iree_host_size_t loop_index = 0; loop_index < state->loop_count;
         ++loop_index) {
      const loom_vector_bank_sroa_loop_plan_t* loop = &state->loops[loop_index];
      const loom_op_t* loop_op = loop->loop->loop.op;
      for (uint32_t bank_index = 0; bank_index < loop->bank_count;
           ++bank_index) {
        const loom_vector_bank_sroa_bank_plan_t* bank =
            &loop->banks[bank_index];
        if (bank->endpoint_count == 0) {
          continue;
        }
        iree_string_view_t outcome = iree_string_view_empty();
        iree_string_view_t reason = iree_string_view_empty();
        loom_vector_bank_sroa_report_decision(function, bank, &outcome,
                                              &reason);
        loom_target_compile_report_source_boundary_projection_row_t row = {
            .function_name = function_name,
            .source_op_name = loom_op_name(plan->module, loop_op),
            .source_op_kind = loop_op->kind,
            .projection_key = rule->name,
            .boundary_key =
                bank->domain == LOOM_VECTOR_BANK_SROA_BANK_DOMAIN_HEADER
                    ? IREE_SV("loop_header_state")
                    : IREE_SV("loop_state"),
            .outcome = outcome,
            .reason = reason,
            .operation_ordinal = (uint32_t)loop_index,
            .source_value_ordinal = bank->state_ordinal,
            .source_type_kind = loom_type_kind(bank->bank_type),
            .source_element_type = loom_type_element_type(bank->bank_type),
            .source_rank = loom_type_rank(bank->bank_type),
            .projected_prefix_rank = bank->active ? bank->prefix_rank : 0,
            .component_count = bank->active ? bank->component_count : 0,
        };
        for (uint8_t axis = 0; axis < row.source_rank; ++axis) {
          row.source_dimensions[axis] =
              loom_type_dim_is_dynamic_at(bank->bank_type, axis)
                  ? LOOM_TARGET_COMPILE_REPORT_DIMENSION_DYNAMIC
                  : loom_type_dim_static_size_at(bank->bank_type, axis);
        }
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_record_source_boundary_projection_row(
                report, &row));
      }
    }
  }
  return iree_ok_status();
}

static const loom_boundary_projection_rule_t kVectorBankSroaRule = {
    .name = IREE_SVL("loop-vector-bank"),
    .type_kind_bits = LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_VECTOR),
    .slot_role_bits = LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
        LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE),
    .function_applies = loom_vector_bank_sroa_function_applies,
    .slot_matches = loom_vector_bank_sroa_slot_matches,
    .prepare_function = loom_vector_bank_sroa_prepare_function,
    .plan_slot = loom_vector_bank_sroa_plan_slot,
    .transport =
        {
            .plan_source = loom_vector_bank_sroa_plan_source,
            .materialize_source = loom_vector_bank_sroa_materialize_source,
            .eliminate = loom_vector_bank_sroa_eliminate,
        },
};

const loom_boundary_projection_rule_t*
loom_vector_bank_sroa_boundary_projection_rule(void) {
  return &kVectorBankSroaRule;
}
