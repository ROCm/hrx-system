// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/lower/private_storage.h"

#include "loom/analysis/storage_layout.h"
#include "loom/codegen/low/lower/source_memory.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/error_catalog.h"
#include "loom/target/arch/spirv/features.h"
#include "loom/target/arch/spirv/registers.h"
#include "loom/util/ordinal_map.h"

// Private arrays have at most UINT32_MAX cells of at most eight bytes. A
// 64-GiB logical interval separates each component, including one-past values.
// Components are function-local 24-bit ordinals, so positions fit signed i64.
#define LOOM_SPIRV_PRIVATE_SEGMENT_STRIDE (UINT64_C(1) << 36)
#define LOOM_SPIRV_PRIVATE_ALLOCA_PLAN 16

static const char kPrivateStorageKey = 0;
static const loom_spirv_scalar_type_t kCellTypes[] = {
    LOOM_SPIRV_SCALAR_TYPE_S8, LOOM_SPIRV_SCALAR_TYPE_S16,
    LOOM_SPIRV_SCALAR_TYPE_S32, LOOM_SPIRV_SCALAR_TYPE_S64};

typedef struct loom_spirv_private_segment_t {
  // Aggregate physical array extent in bytes.
  uint64_t byte_length;
  // Strongest source allocation alignment in bytes.
  uint64_t byte_alignment;
  // Selected log2(cell byte width).
  uint8_t cell_shift;
  // Whether a retained memory recipe consumes the physical array.
  bool requires_array;
  // Typed Low pointer to the emitted array.
  loom_value_id_t array;
} loom_spirv_private_segment_t;

typedef struct loom_spirv_private_storage_t {
  // Sparse component ordinal to segment ordinal, allocated only for private
  // use.
  loom_ordinal_map_t component_segments;
  // Retained physical arrays in first-use order.
  loom_spirv_private_segment_t* segments;
  // Number of initialized physical arrays.
  iree_host_size_t count;
  // Allocated array capacity.
  iree_host_size_t capacity;
} loom_spirv_private_storage_t;

typedef struct loom_spirv_private_operand_t {
  // Function-owned physical segment ordinal.
  uint32_t segment;
  // Result type of the array materializer.
  loom_type_t array_type;
  // Result type of the logical-origin materializer.
  loom_type_t position_type;
} loom_spirv_private_operand_t;

typedef struct loom_spirv_private_position_t {
  // Native logical coordinate type.
  loom_type_t type;
  // True when this use supplies null instead of a private coordinate.
  bool is_null;
} loom_spirv_private_position_t;

bool loom_spirv_value_is_private(const loom_module_t* module,
                                 const loom_value_fact_table_t* facts,
                                 loom_value_id_t value) {
  const loom_type_t type = loom_module_value_type(module, value);
  if (!loom_type_is_buffer(type) && !loom_type_is_view(type)) {
    return false;
  }
  const loom_value_facts_t value_facts =
      loom_value_fact_table_lookup(facts, value);
  if (loom_type_is_buffer(type)) {
    loom_value_fact_buffer_reference_t reference;
    return loom_value_facts_query_buffer_reference(&facts->context, value_facts,
                                                   &reference) &&
           reference.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE;
  }
  if (loom_type_is_view(type)) {
    loom_value_fact_view_reference_t reference;
    return loom_value_facts_query_view_reference(&facts->context, value_facts,
                                                 &reference) &&
           reference.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE;
  }
  return false;
}

bool loom_spirv_value_is_null(const loom_value_fact_table_t* facts,
                              loom_value_id_t value) {
  loom_value_fact_buffer_reference_t reference;
  return loom_value_facts_query_buffer_reference(
             &facts->context, loom_value_fact_table_lookup(facts, value),
             &reference) &&
         reference.nullability == LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NULL;
}

void loom_spirv_private_observe_relation(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_relation_t* relation,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* facts =
      loom_low_lower_context_fact_table(context);
  const loom_type_t destination_type =
      loom_module_value_type(module, relation->destination_value_id);
  if (!loom_type_is_buffer(destination_type) &&
      !loom_type_is_view(destination_type)) {
    return;
  }
  const bool source_private =
      loom_spirv_value_is_private(module, facts, relation->source_value_id);
  const bool destination_private = loom_spirv_value_is_private(
      module, facts, relation->destination_value_id);
  if (!source_private && !destination_private) {
    return;
  }
  if (relation->source_operand_index != UINT16_MAX) {
    loom_low_lower_representation_claim_operand(recorder,
                                                relation->source_operand_index);
  }
  if (loom_spirv_value_is_null(facts, relation->source_value_id) ||
      loom_spirv_value_is_null(facts, relation->destination_value_id)) {
    return;
  }
  if (!source_private || !destination_private) {
    const loom_diagnostic_param_t parameter =
        loom_param_string(loom_low_lower_context_function_name(context));
    loom_low_lower_representation_record_failure(
        recorder,
        loom_low_lower_emit_error_ref(context, source_op,
                                      LOOM_ERR_SPIRV_046_REF, &parameter, 1));
    return;
  }
  loom_low_lower_representation_record_union(
      recorder, relation->source_value_id, relation->destination_value_id);
}

static void loom_spirv_private_constrain(
    loom_low_lower_context_t* context, loom_value_id_t value,
    uint32_t maximum_cell_width,
    loom_low_lower_representation_recorder_t* recorder) {
  const uint64_t features =
      loom_low_lower_context_bundle(context)->config->contract_feature_bits;
  loom_low_representation_candidate_t candidates[4];
  iree_host_size_t count = 0;
  for (uint8_t shift = 0; shift < 4; ++shift) {
    const uint32_t width = 1u << shift;
    if (width > maximum_cell_width ||
        (shift == 1 && !(features & LOOM_SPIRV_FEATURE_INT16))) {
      continue;
    }
    candidates[count++] = (loom_low_representation_candidate_t){
        .representation = LOOM_SPIRV_PRIVATE_REPRESENTATION_BASE + shift,
        .cost = {.runtime = 8u / width}};
  }
  loom_low_lower_representation_record_candidates(recorder, value, candidates,
                                                  count);
}

void loom_spirv_private_observe_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  if (loom_buffer_alloca_memory_space(source_op) ==
      LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE) {
    loom_spirv_private_constrain(context, loom_buffer_alloca_result(source_op),
                                 8, recorder);
  }
}

void loom_spirv_private_observe_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t operand, loom_low_lower_representation_recorder_t* recorder) {
  const loom_value_id_t value = loom_op_const_operands(source_op)[operand];
  if (!loom_spirv_value_is_private(loom_low_lower_context_module(context),
                                   loom_low_lower_context_fact_table(context),
                                   value) ||
      !loom_memory_access_isa(loom_memory_access_cast(
          loom_low_lower_context_module(context), source_op))) {
    return;
  }
  loom_low_source_memory_access_diagnostic_t diagnostic;
  const loom_low_source_memory_access_plan_t* access =
      loom_low_lower_source_memory_access(context, source_op, &diagnostic);
  if (!access) {
    return;  // The canonical access diagnostic owns rejection.
  }
  const uint32_t width =
      iree_min(access->element_byte_count, access->minimum_alignment);
  loom_spirv_private_constrain(context, value, width, recorder);
}

static const loom_spirv_private_storage_t* loom_spirv_private_storage_lookup(
    const loom_low_lower_context_t* context) {
  return loom_low_lower_lookup_target_state(
      context, &kPrivateStorageKey, sizeof(loom_spirv_private_storage_t));
}

static iree_status_t loom_spirv_private_segment(
    loom_low_lower_context_t* context, loom_value_id_t value,
    loom_spirv_private_storage_t** out_storage, uint32_t* out_segment) {
  loom_spirv_private_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &kPrivateStorageKey, sizeof(*storage), (void**)&storage));
  iree_arena_allocator_t* arena =
      loom_low_lower_context_function_arena(context);
  if (!storage->component_segments.arena) {
    loom_ordinal_map_initialize(arena, &storage->component_segments);
  }
  loom_low_representation_plan_t* representations =
      loom_low_lower_representation_plan_if_ready(context);
  const loom_value_ordinal_t ordinal = loom_local_value_domain_try_ordinal(
      loom_low_lower_context_value_domain(context), value);
  const loom_value_ordinal_t component =
      loom_low_representation_plan_component(representations, ordinal);
  uint32_t segment;
  if (!loom_ordinal_map_lookup(&storage->component_segments, 0, component,
                               &segment)) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, storage->count, storage->count + 1, sizeof(*storage->segments),
        &storage->capacity, (void**)&storage->segments));
    loom_low_representation_id_t representation;
    loom_low_representation_plan_lookup(representations, ordinal,
                                        &representation);
    segment = (uint32_t)storage->count++;
    storage->segments[segment] = (loom_spirv_private_segment_t){
        .byte_alignment = 1,
        .cell_shift = representation - LOOM_SPIRV_PRIVATE_REPRESENTATION_BASE,
        .array = LOOM_VALUE_ID_INVALID};
    IREE_RETURN_IF_ERROR(loom_ordinal_map_insert(&storage->component_segments,
                                                 0, component, segment));
  }
  *out_storage = storage;
  *out_segment = segment;
  return iree_ok_status();
}

iree_status_t loom_spirv_select_private_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  if (!loom_buffer_alloca_isa(source_op) ||
      loom_buffer_alloca_pool(source_op) != LOOM_VALUE_ID_INVALID ||
      loom_buffer_alloca_memory_space(source_op) !=
          LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE) {
    return iree_ok_status();
  }
  *out_plan = loom_low_lower_plan_make(LOOM_SPIRV_PRIVATE_ALLOCA_PLAN, NULL);
  bool lifetime_supported;
  IREE_RETURN_IF_ERROR(loom_low_lower_function_storage_check_lifetime(
      context, source_op, &lifetime_supported));
  if (!lifetime_supported) {
    return iree_ok_status();
  }
  int64_t extent;
  if (!loom_value_facts_as_non_negative_i64_maximum(
          loom_value_fact_table_lookup(
              loom_low_lower_context_fact_table(context),
              loom_buffer_alloca_byte_length(source_op)),
          &extent)) {
    return loom_low_lower_emit_function_storage_extent_unsupported(
        context, source_op, LOOM_STORAGE_SPACE_PRIVATE,
        loom_buffer_alloca_byte_length(source_op));
  }
  loom_spirv_private_storage_t* storage;
  uint32_t index;
  IREE_RETURN_IF_ERROR(loom_spirv_private_segment(
      context, loom_buffer_alloca_result(source_op), &storage, &index));
  loom_spirv_private_segment_t* segment = &storage->segments[index];
  const uint64_t cell_width = UINT64_C(1) << segment->cell_shift;
  const uint64_t alignment = iree_max(
      (uint64_t)loom_buffer_alloca_base_alignment(source_op), cell_width);
  uint64_t offset;
  if (!loom_storage_layout_append(iree_max((uint64_t)extent, cell_width),
                                  alignment, &segment->byte_length, &offset) ||
      segment->byte_length > UINT32_MAX * cell_width) {
    return loom_low_lower_emit_function_storage_extent_unsupported(
        context, source_op, LOOM_STORAGE_SPACE_PRIVATE,
        loom_buffer_alloca_byte_length(source_op));
  }
  segment->byte_alignment = iree_max(segment->byte_alignment, alignment);
  uint64_t* position = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*position), (void**)&position));
  *position =
      (index + UINT64_C(1)) * LOOM_SPIRV_PRIVATE_SEGMENT_STRIDE + offset;
  loom_type_t position_type;
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64, 1, &position_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
      context, loom_buffer_alloca_result(source_op), position_type));
  *out_plan =
      loom_low_lower_plan_make(LOOM_SPIRV_PRIVATE_ALLOCA_PLAN, position);
  return iree_ok_status();
}

bool loom_spirv_private_alloca_plan_isa(loom_low_lower_plan_t plan) {
  return plan.id == LOOM_SPIRV_PRIVATE_ALLOCA_PLAN;
}

static iree_status_t loom_spirv_emit_position_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    int64_t position, loom_value_id_t* out_value) {
  loom_type_t type;
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64, 1, &type));
  loom_string_id_t key;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      loom_low_lower_context_module(context), IREE_SV("offset64_value"), &key));
  const loom_named_attr_t attr = {.name_id = key,
                                  .value = loom_attr_i64(position)};
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor = loom_low_descriptor_set_descriptor_at(
          loom_low_lower_context_descriptor_set(context),
          SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONSTANT_OFFSET64)};
  loom_op_t* result;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, &descriptor, loom_make_named_attr_slice(&attr, 1), type,
      source_op->location, &result));
  *out_value = loom_op_const_results(result)[0];
  return iree_ok_status();
}

iree_status_t loom_spirv_emit_private_alloca(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t plan) {
  loom_value_id_t position;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_position_constant(
      context, source_op, (int64_t)*(const uint64_t*)plan.target_data,
      &position));
  return loom_low_lower_bind_value(
      context, loom_buffer_alloca_result(source_op), position);
}

iree_status_t loom_spirv_emit_private_storage(
    loom_low_lower_context_t* context) {
  loom_spirv_private_storage_t* storage =
      (loom_spirv_private_storage_t*)loom_spirv_private_storage_lookup(context);
  if (!storage) {
    return iree_ok_status();
  }
  const loom_location_id_t location =
      loom_low_lower_context_source_function(context).op->location;
  for (iree_host_size_t i = 0; i < storage->count; ++i) {
    loom_spirv_private_segment_t* segment = &storage->segments[i];
    if (!segment->requires_array) {
      continue;
    }
    const uint64_t cell_width = UINT64_C(1) << segment->cell_shift;
    const uint64_t byte_length =
        (segment->byte_length + cell_width - 1) & ~(cell_width - 1);
    loom_op_t* reservation;
    IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
        loom_low_lower_context_builder(context), (int64_t)byte_length,
        (int64_t)segment->byte_alignment,
        loom_type_storage(LOOM_STORAGE_SPACE_PRIVATE), location, &reservation));
    loom_type_t type;
    IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
        context,
        loom_spirv_logical_pointer_reg_class_id(
            LOOM_SPIRV_VALUE_CLASS_PTR_FUNCTION_ARRAY,
            kCellTypes[segment->cell_shift]),
        1, &type));
    loom_op_t* address;
    IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
        loom_low_lower_context_builder(context),
        loom_low_storage_reserve_storage(reservation), 0, type, location,
        &address));
    segment->array = loom_low_storage_address_result(address);
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_can_materialize_private_storage(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, bool* out_can_materialize) {
  (void)source_op;
  *out_can_materialize = loom_spirv_value_is_private(
      loom_low_lower_context_module(context),
      loom_low_lower_context_fact_table(context), source_value);
  return iree_ok_status();
}

iree_status_t loom_spirv_prepare_private_storage(
    loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void** out_plan) {
  loom_spirv_private_storage_t* storage;
  uint32_t segment;
  IREE_RETURN_IF_ERROR(
      loom_spirv_private_segment(context, source_value, &storage, &segment));
  storage->segments[segment].requires_array = true;
  loom_spirv_private_operand_t* plan;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  plan->segment = segment;
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context,
      loom_spirv_logical_pointer_reg_class_id(
          LOOM_SPIRV_VALUE_CLASS_PTR_FUNCTION_ARRAY,
          kCellTypes[storage->segments[segment].cell_shift]),
      1, &plan->array_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64, 1,
      &plan->position_type));
  *out_plan = plan;
  return iree_ok_status();
}

loom_type_t loom_spirv_private_storage_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void* opaque_plan) {
  (void)context;
  (void)source_value;
  return ((const loom_spirv_private_operand_t*)opaque_plan)->array_type;
}

loom_type_t loom_spirv_private_origin_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void* opaque_plan) {
  (void)context;
  (void)source_value;
  return ((const loom_spirv_private_operand_t*)opaque_plan)->position_type;
}

iree_status_t loom_spirv_materialize_private_storage(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, const void* opaque_plan,
    loom_value_id_t* out_value) {
  (void)source_op;
  (void)source_value;
  const loom_spirv_private_operand_t* plan = opaque_plan;
  *out_value =
      loom_spirv_private_storage_lookup(context)->segments[plan->segment].array;
  return iree_ok_status();
}

iree_status_t loom_spirv_materialize_private_origin(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, const void* opaque_plan,
    loom_value_id_t* out_value) {
  (void)source_value;
  const loom_spirv_private_operand_t* plan = opaque_plan;
  return loom_spirv_emit_position_constant(
      context, source_op,
      (int64_t)((plan->segment + UINT64_C(1)) *
                LOOM_SPIRV_PRIVATE_SEGMENT_STRIDE),
      out_value);
}

iree_status_t loom_spirv_emit_private_null(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_value_id_t* out_value) {
  return loom_spirv_emit_position_constant(context, source_op, 0, out_value);
}

iree_status_t loom_spirv_can_materialize_private_position(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, bool* out_can_materialize) {
  (void)source_op;
  *out_can_materialize =
      loom_spirv_value_is_private(loom_low_lower_context_module(context),
                                  loom_low_lower_context_fact_table(context),
                                  source_value) ||
      loom_spirv_value_is_null(loom_low_lower_context_fact_table(context),
                               source_value);
  return iree_ok_status();
}

iree_status_t loom_spirv_prepare_private_position(
    loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void** out_plan) {
  loom_spirv_private_position_t* plan;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  plan->is_null = loom_spirv_value_is_null(
      loom_low_lower_context_fact_table(context), source_value);
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64, 1, &plan->type));
  *out_plan = plan;
  return iree_ok_status();
}

loom_type_t loom_spirv_private_position_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void* opaque_plan) {
  (void)context;
  (void)source_value;
  return ((const loom_spirv_private_position_t*)opaque_plan)->type;
}

iree_status_t loom_spirv_materialize_private_position(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, const void* opaque_plan,
    loom_value_id_t* out_value) {
  const loom_spirv_private_position_t* plan = opaque_plan;
  if (plan->is_null) {
    return loom_spirv_emit_private_null(context, source_op, out_value);
  }
  *out_value = loom_low_lower_lookup_value(context, source_value);
  return iree_ok_status();
}
