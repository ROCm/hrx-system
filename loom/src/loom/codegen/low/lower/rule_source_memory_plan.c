// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/rule_source_memory_plan.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/lower/bindings.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/realization.h"
#include "loom/codegen/low/lower/rule_descriptor.h"
#include "loom/codegen/low/lower/rule_plan.h"
#include "loom/codegen/low/lower/rule_source_memory.h"
#include "loom/codegen/low/lower/source_value_storage.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ir/structural_hash.h"

typedef struct loom_low_lower_rule_byte_offset_binding_t {
  // Resolved emit owning the arithmetic definition at its original position.
  const loom_low_lower_resolved_emit_t* definition;
  // Selected carried value replacing one canonical dynamic term.
  const loom_low_lower_realization_t* term_realization;
  // Last physical use governing placement of the destructive backedge update.
  const loom_op_t* last_use_op;
  // Low SSA published by the definition and read directly by later uses.
  loom_value_id_t value;
  // First operand using the byte offset within the defining emit.
  uint16_t operand_ordinal;
  // Dynamic-term ordinal replaced by |term_realization|.
  uint8_t term_ordinal;
} loom_low_lower_rule_byte_offset_binding_t;

struct loom_low_lower_rule_byte_offset_entry_t {
  // Actual source emission block; definitions never cross block boundaries.
  const loom_block_t* block;
  // Descriptor namespace for the selected arithmetic contract.
  const loom_low_lower_rule_set_t* rule_set;
  // Interned arithmetic carrier, conversions, and descriptor choices.
  const loom_low_lower_source_memory_byte_offset_materializer_t* materializer;
  // Source observation's canonical dynamic term-set identity.
  iree_host_size_t identity;
  // Exact static byte contribution included in the returned physical value.
  int64_t static_bias;
  // Function-owned binding shared by all uses of this physical recipe.
  loom_low_lower_rule_byte_offset_binding_t* binding;
  // Next colliding physical recipe in the bucket.
  loom_low_lower_rule_byte_offset_entry_t* next;
  // Actual source position of the current defining emit.
  uint64_t block_ordinal;
  // Row ordinal within the defining selected rule.
  uint16_t emit_ordinal;
  // Retained hash used when growing the construction table.
  uint32_t hash;
};

iree_status_t loom_low_lower_rule_source_memory_plan_prepare_producers(
    loom_low_lower_context_t* context,
    loom_low_lower_rule_source_memory_plan_builder_t* builder) {
  const loom_value_ordinal_t value_count =
      context->lowering->value_domain.value_count;
  if (value_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, value_count, sizeof(*builder->producers_by_value_ordinal),
      (void**)&builder->producers_by_value_ordinal));
  memset(builder->producers_by_value_ordinal, 0,
         value_count * sizeof(*builder->producers_by_value_ordinal));
  return iree_ok_status();
}

bool loom_low_lower_rule_source_memory_plan_requires_producer(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_source_memory_plan_builder_t* builder,
    const loom_low_lower_selected_plan_t* selected_plan) {
  if (builder->producers_by_value_ordinal == NULL ||
      selected_plan->source_op->result_count != 1 ||
      selected_plan->kind != LOOM_LOW_LOWER_SELECTED_PLAN_RULE ||
      iree_any_bit_set(selected_plan->flags,
                       LOOM_LOW_LOWER_SELECTED_PLAN_CLAIMED) ||
      selected_plan->source_node_count != 1 || selected_plan->rule == NULL ||
      selected_plan->rule->emit_count == 0 ||
      loom_low_lower_source_value_storage_op_requires_emission(
          selected_plan->source_op, selected_plan->source_op->traits) ||
      !loom_traits_are_safe_to_speculate(selected_plan->source_op->traits)) {
    return false;
  }
  const loom_value_id_t result =
      loom_op_const_results(selected_plan->source_op)[0];
  const loom_value_ordinal_t ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, result);
  return iree_any_bit_set(context->lowering->source_plan.value_flags[ordinal],
                          LOOM_LOW_LOWER_VALUE_MEMORY_RECURRENCE_PRODUCER);
}

void loom_low_lower_rule_source_memory_plan_record_producer(
    loom_low_lower_context_t* context,
    loom_low_lower_rule_source_memory_plan_builder_t* builder,
    loom_low_lower_selected_plan_t* selected_plan) {
  const loom_value_id_t result =
      loom_op_const_results(selected_plan->source_op)[0];
  const loom_value_ordinal_t ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, result);
  builder->producers_by_value_ordinal[ordinal] = selected_plan;
}

static uint32_t loom_low_lower_rule_source_memory_descriptor_cost(
    const loom_low_lower_context_t* context,
    const loom_low_descriptor_t* descriptor);

typedef struct loom_low_lower_rule_source_memory_recurrence_key_t {
  // Function-local canonical term identity, or zero for an authored producer.
  iree_host_size_t term_identity;
  // Resolved initializer descriptor defining the carrier's zero value.
  loom_low_lower_resolved_descriptor_t initializer_descriptor;
  // Resolved update descriptor shared by every member of an offer group.
  loom_low_lower_resolved_descriptor_t update_descriptor;
  // Exact update immediate, or zero when the stride is a source input.
  int64_t constant_stride;
  // Dynamic stride input participating in the shared group, or INVALID.
  loom_value_id_t stride_value_id;
  // Module string naming the initializer's constant immediate.
  loom_string_id_t initializer_immediate_name;
  // Module string naming the update immediate, or INVALID for dynamic strides.
  loom_string_id_t update_immediate_name;
} loom_low_lower_rule_source_memory_recurrence_key_t;

typedef struct loom_low_lower_rule_source_memory_recurrence_state_t {
  // Invariant byte stride consumed directly by the backedge update.
  loom_low_lower_realization_input_t stride_input;
  // Physical carrier shared by initialization and update.
  loom_type_t carrier_type;
  // Canonical physical identity shared across generated rule-set instances.
  loom_low_lower_rule_source_memory_recurrence_key_t key;
} loom_low_lower_rule_source_memory_recurrence_state_t;

typedef struct loom_low_lower_rule_source_memory_recurrence_resolution_t {
  // Fully resolved physical state and grouping identity.
  loom_low_lower_rule_source_memory_recurrence_state_t state;
  // One-time zero-carrier initialization cost.
  uint32_t initializer_cost;
  // Per-backedge update cost.
  uint32_t update_cost;
  // True when the update consumes a source or deferred stride carrier.
  bool uses_stride_operand;
} loom_low_lower_rule_source_memory_recurrence_resolution_t;

struct loom_low_lower_rule_source_memory_recurrence_t {
  // Selected producer plan replaced when the complete offer group wins.
  loom_low_lower_selected_plan_t* producer_plan;
  // Exact loop domain shared by selection and physical placement.
  const loom_low_lower_realization_loop_t* loop;
  // Next retained producer candidate in first-observation order.
  loom_low_lower_rule_source_memory_recurrence_t* next;
  // Deferred physical recipe owned by the shared offer selector.
  loom_low_lower_realization_recipe_t recipe;
  // Physical recurrence state consumed by the deferred recipe.
  loom_low_lower_rule_source_memory_recurrence_state_t state;
  // Shared one-time initialization cost.
  uint32_t initializer_cost;
  // Shared cost paid on every backedge.
  uint32_t update_cost;
  // Per-iteration cost of the direct producer plan.
  uint32_t direct_cost;
};

typedef struct loom_low_lower_rule_source_memory_term_recurrence_t {
  // Callback-shared state. First so the common initializer accepts this record.
  loom_low_lower_rule_source_memory_recurrence_state_t state;
  // Selected byte-offset binding receiving the carried canonical term.
  loom_low_lower_rule_byte_offset_binding_t* binding;
  // Exact loop domain shared by selection and physical placement.
  const loom_low_lower_realization_loop_t* loop;
  // Descriptor namespace used to materialize the invariant stride.
  const loom_low_lower_rule_set_t* rule_set;
  // Interned byte-offset arithmetic contract.
  const loom_low_lower_source_memory_byte_offset_materializer_t* materializer;
  // Analysis-owned canonical term; its invariant suffix defines the stride.
  const loom_low_source_memory_dynamic_term_t* term;
  // Static byte factor advanced on each loop backedge.
  int64_t update_byte_stride;
  // Shift spelling of |update_byte_stride|, when available.
  uint32_t update_byte_shift;
  // Deferred carried-term recipe requested only when the offer wins.
  loom_low_lower_realization_recipe_t recipe;
  // Deferred invariant-stride recipe requested only when needed.
  loom_low_lower_realization_recipe_t stride_recipe;
  // Exact invariant byte stride materialized outside the loop.
  int64_t constant_stride;
  // Canonical term ordinal replaced in |binding|.
  uint8_t term_ordinal;
} loom_low_lower_rule_source_memory_term_recurrence_t;

static const uint8_t kSourceMemoryRecurrencePlanId = 0;
static const uint8_t kSourceMemoryStridePlanId = 0;

static uint32_t loom_low_lower_rule_source_memory_descriptor_cost(
    const loom_low_lower_context_t* context,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor == NULL) {
    return 0;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_view_t* descriptor_view =
      loom_low_descriptor_set_descriptor_view(descriptor_set, descriptor);
  const loom_low_schedule_model_t* schedule_model =
      loom_low_lower_context_schedule_model(context);
  if (schedule_model == NULL ||
      descriptor_view->schedule_class_id == LOOM_LOW_SCHEDULE_CLASS_NONE) {
    return 1;
  }
  const loom_low_schedule_class_t* schedule_class =
      loom_low_schedule_model_schedule_class(schedule_model, descriptor_view);
  return iree_max(
      iree_max(loom_low_schedule_class_schedule_distance_cycles(schedule_class),
               schedule_class->minimum_issue_cycles),
      1u);
}

static bool loom_low_lower_rule_source_memory_immediate_fits(
    const loom_low_immediate_t* immediate, int64_t value) {
  const uint64_t magnitude = iree_math_magnitude_i64(value);
  if (magnitude % immediate->value_step != 0) {
    return false;
  }
  if (immediate->kind == LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
    const int64_t maximum = immediate->unsigned_max > INT64_MAX
                                ? INT64_MAX
                                : (int64_t)immediate->unsigned_max;
    return value >= immediate->signed_min && value <= maximum;
  }
  return (immediate->kind == LOOM_LOW_IMMEDIATE_KIND_UNSIGNED ||
          immediate->kind == LOOM_LOW_IMMEDIATE_KIND_ORDINAL) &&
         value >= 0 && (uint64_t)value <= immediate->unsigned_max;
}

// Resolves the physical recurrence contract shared by authored products and
// canonical source-memory terms. A source stride names an existing carrier;
// |can_materialize_stride| admits a deferred invariant stride realization when
// no authored carrier exists. A NULL update descriptor means neither form can
// represent the requested stride.
static iree_status_t loom_low_lower_rule_source_memory_resolve_recurrence(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_source_memory_byte_offset_materializer_t* materializer,
    iree_host_size_t term_identity, int64_t constant_stride,
    loom_value_id_t stride_value_id, bool can_materialize_stride,
    loom_low_lower_rule_source_memory_recurrence_resolution_t* out_resolution) {
  *out_resolution =
      (loom_low_lower_rule_source_memory_recurrence_resolution_t){0};
  loom_low_lower_rule_source_memory_recurrence_state_t* state =
      &out_resolution->state;

  loom_low_lower_resolved_descriptor_t constant_descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
          context, rule_set, materializer->constant_descriptor_ref,
          &constant_descriptor));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_result_type(
      context, constant_descriptor.descriptor, 0, &state->carrier_type));

  loom_low_lower_resolved_descriptor_t update_descriptor = {0};
  loom_string_ref_t update_immediate_string_ref = LOOM_STRING_REF_NONE;
  bool uses_immediate = false;
  if (constant_stride != 0) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
            context, rule_set, materializer->increment_descriptor_ref,
            &update_descriptor));
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_lower_context_descriptor_set(context);
    const loom_low_descriptor_layout_t* layout = loom_low_descriptor_layout(
        descriptor_set, update_descriptor.descriptor);
    IREE_ASSERT_LT(materializer->increment_immediate_ordinal,
                   layout->immediate_count);
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[layout->immediate_start +
                                    materializer->increment_immediate_ordinal];
    if (loom_low_lower_rule_source_memory_immediate_fits(immediate,
                                                         constant_stride)) {
      uses_immediate = true;
      stride_value_id = LOOM_VALUE_ID_INVALID;
      update_immediate_string_ref = immediate->field_name_string_ref;
    }
  }
  if (!uses_immediate) {
    if (stride_value_id == LOOM_VALUE_ID_INVALID && !can_materialize_stride) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
            context, rule_set, materializer->add_descriptor_ref,
            &update_descriptor));
    out_resolution->uses_stride_operand = true;
  }

  loom_string_id_t initializer_immediate_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      loom_low_lower_context_module(context),
      loom_low_lower_rule_set_string(
          rule_set, materializer->constant_immediate_string_ref),
      &initializer_immediate_name));
  loom_string_id_t update_immediate_name = LOOM_STRING_ID_INVALID;
  if (uses_immediate) {
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        loom_low_lower_context_module(context),
        loom_low_descriptor_set_string(
            loom_low_lower_context_descriptor_set(context),
            update_immediate_string_ref),
        &update_immediate_name));
  }

  state->stride_input = (loom_low_lower_realization_input_t){
      .kind = LOOM_LOW_LOWER_REALIZATION_INPUT_SOURCE,
      .value.identity = stride_value_id,
  };
  state->key.term_identity = term_identity;
  state->key.initializer_descriptor = constant_descriptor;
  state->key.update_descriptor = update_descriptor;
  state->key.constant_stride = constant_stride;
  state->key.stride_value_id = stride_value_id;
  state->key.initializer_immediate_name = initializer_immediate_name;
  state->key.update_immediate_name = update_immediate_name;
  out_resolution->initializer_cost =
      loom_low_lower_rule_source_memory_descriptor_cost(
          context, constant_descriptor.descriptor);
  out_resolution->update_cost =
      loom_low_lower_rule_source_memory_descriptor_cost(
          context, update_descriptor.descriptor);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_source_memory_initialize_recurrence(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const void* data, loom_value_id_t carried_value,
    loom_value_id_t* out_value) {
  (void)carried_value;
  const loom_low_lower_rule_source_memory_recurrence_state_t* state =
      (const loom_low_lower_rule_source_memory_recurrence_state_t*)data;
  const loom_named_attr_t attr = {
      .name_id = state->key.initializer_immediate_name,
      .value = loom_attr_i64(0),
  };
  loom_op_t* constant = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, &state->key.initializer_descriptor,
      loom_make_named_attr_slice(&attr, 1), state->carrier_type,
      source_op->location, &constant));
  *out_value = loom_low_const_result(constant);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_source_memory_update_recurrence(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const void* data, loom_value_id_t carried_value,
    loom_value_id_t* out_value) {
  const loom_low_lower_rule_source_memory_recurrence_state_t* state =
      (const loom_low_lower_rule_source_memory_recurrence_state_t*)data;
  loom_value_id_t operands[2] = {carried_value, LOOM_VALUE_ID_INVALID};
  uint16_t operand_count = 1;
  loom_named_attr_slice_t attributes = loom_named_attr_slice_empty();
  loom_named_attr_t attr = {0};
  if (state->stride_input.kind ==
      LOOM_LOW_LOWER_REALIZATION_INPUT_REALIZATION) {
    operands[1] =
        loom_low_lower_realization_value(state->stride_input.value.realization);
    operand_count = 2;
  } else if (state->stride_input.value.identity != LOOM_VALUE_ID_INVALID) {
    operands[1] = loom_low_lower_lookup_value(
        context, state->stride_input.value.identity);
    operand_count = 2;
  } else {
    attr = (loom_named_attr_t){
        .name_id = state->key.update_immediate_name,
        .value = loom_attr_i64(state->key.constant_stride),
    };
    attributes = loom_make_named_attr_slice(&attr, 1);
  }
  return loom_low_lower_rule_source_memory_emit_resolved_op(
      context, &state->key.update_descriptor, operands, operand_count,
      /*consumable_operand_mask=*/1, state->carrier_type, attributes,
      source_op->location, out_value);
}

static iree_status_t loom_low_lower_rule_source_memory_update_term_recurrence(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const void* data, loom_value_id_t carried_value,
    loom_value_id_t* out_value) {
  const loom_low_lower_rule_source_memory_term_recurrence_t* recurrence =
      (const loom_low_lower_rule_source_memory_term_recurrence_t*)data;
  loom_low_source_memory_dynamic_term_t update_term = *recurrence->term;
  update_term.byte_stride = recurrence->update_byte_stride;
  update_term.byte_shift = recurrence->update_byte_shift;
  loom_value_id_t stride = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_materialize_source_memory_term_stride(
          context, recurrence->rule_set, source_op, recurrence->materializer,
          &update_term, recurrence->state.carrier_type, &stride));
  loom_value_id_t operands[2] = {carried_value, stride};
  return loom_low_lower_rule_source_memory_emit_resolved_op(
      context, &recurrence->state.key.update_descriptor, operands,
      IREE_ARRAYSIZE(operands), /*consumable_operand_mask=*/1,
      recurrence->state.carrier_type, loom_named_attr_slice_empty(),
      source_op->location, out_value);
}

static iree_status_t loom_low_lower_rule_source_memory_apply(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_realization_t* realization, void* data) {
  (void)source_op;
  loom_low_lower_rule_source_memory_recurrence_t* recurrence =
      (loom_low_lower_rule_source_memory_recurrence_t*)data;
  IREE_ASSERT(realization != NULL);
  IREE_ASSERT_EQ(recurrence->producer_plan->source_node_count, 1u);
  const loom_value_id_t producer_result =
      loom_op_const_results(recurrence->producer_plan->source_op)[0];
  loom_low_lower_require_source_value_storage(context, producer_result);
  IREE_RETURN_IF_ERROR(loom_low_lower_realization_bind_source_value(
      context, realization, producer_result));
  recurrence->producer_plan->flags |= LOOM_LOW_LOWER_SELECTED_PLAN_CLAIMED |
                                      LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_source_memory_materialize_stride(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const void* data, loom_value_id_t carried_value,
    loom_value_id_t* out_value) {
  (void)carried_value;
  const loom_low_lower_rule_source_memory_term_recurrence_t* recurrence =
      (const loom_low_lower_rule_source_memory_term_recurrence_t*)data;
  const loom_named_attr_t attr = {
      .name_id = recurrence->state.key.initializer_immediate_name,
      .value = loom_attr_i64(recurrence->constant_stride),
  };
  loom_op_t* constant = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, &recurrence->state.key.initializer_descriptor,
      loom_make_named_attr_slice(&attr, 1), recurrence->state.carrier_type,
      source_op->location, &constant));
  *out_value = loom_low_const_result(constant);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_source_memory_apply_term_recurrence(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_realization_t* realization, void* data) {
  (void)source_op;
  (void)realization;
  loom_low_lower_rule_source_memory_term_recurrence_t* recurrence =
      (loom_low_lower_rule_source_memory_term_recurrence_t*)data;
  if (recurrence->stride_recipe.initialize != NULL) {
    const loom_low_lower_realization_t* stride_realization = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_realization_request(
        context, recurrence->loop->entry, /*loop=*/NULL,
        &recurrence->stride_recipe, &stride_realization));
    IREE_ASSERT(stride_realization != NULL,
                "proved invariant stride became unplaceable");
    recurrence->state.stride_input = (loom_low_lower_realization_input_t){
        .kind = LOOM_LOW_LOWER_REALIZATION_INPUT_REALIZATION,
        .value.realization = stride_realization,
    };
    recurrence->recipe.inputs = &recurrence->state.stride_input;
    recurrence->recipe.input_count = 1;
  }
  const loom_low_lower_realization_t* term_realization = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_realization_request(
      context, recurrence->binding->last_use_op, recurrence->loop,
      &recurrence->recipe, &term_realization));
  IREE_ASSERT(term_realization != NULL,
              "proved canonical term recurrence became unplaceable");
  recurrence->binding->term_realization = term_realization;
  recurrence->binding->term_ordinal = recurrence->term_ordinal;
  return iree_ok_status();
}

static bool loom_low_lower_rule_source_memory_exact_term_stride(
    const loom_low_lower_context_t* context,
    const loom_low_source_memory_dynamic_term_t* term, int64_t* out_stride) {
  int64_t stride = term->byte_stride;
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    int64_t factor = 0;
    if (!loom_value_facts_as_exact_i64(
            loom_value_fact_table_lookup(context->fact_table,
                                         term->stride_values[i]),
            &factor) ||
        !iree_checked_mul_i64(stride, factor, &stride)) {
      return false;
    }
  }
  *out_stride = stride;
  return true;
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_low_lower_rule_source_memory_consider_term_recurrence(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_lower_source_memory_byte_offset_materializer_t* materializer,
    const loom_low_lower_source_memory_term_recurrence_t* source_recurrence,
    const loom_op_t* emission_op,
    loom_low_lower_rule_byte_offset_binding_t* binding) {
  if (binding->term_ordinal != UINT8_MAX ||
      materializer->increment_descriptor_ref ==
          LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE ||
      materializer->increment_immediate_ordinal == LOOM_LOW_ID_NONE) {
    return iree_ok_status();
  }
  const loom_low_source_memory_access_plan_t* access =
      &selected_plan->source_memory->access;
  if (source_recurrence->term_ordinal >= access->dynamic_term_count ||
      loom_low_lower_rule_source_memory_reuses_view_base(source_memory,
                                                         access) ||
      loom_low_source_memory_access_dynamic_offset_has_materialized_view_base(
          access)) {
    return iree_ok_status();
  }
  const loom_low_lower_realization_loop_t* loop =
      loom_low_lower_realization_loop(context, emission_op);
  if (loop == NULL ||
      emission_op->parent_block != loop->backedge->parent_block ||
      loop->initial_value != 0) {
    return iree_ok_status();
  }
  const loom_low_source_memory_dynamic_term_t* term =
      &access->dynamic_terms[source_recurrence->term_ordinal];

  const uint16_t term_mask = (uint16_t)(1u << source_recurrence->term_ordinal);
  if (access->retained_component.term != NULL &&
      (access->retained_component.term_mask & term_mask) != 0) {
    // The retained component already shares this calculation with a larger
    // byte sum. Replacing one member would dissolve that reuse and duplicate
    // the remaining terms at each physical byte-offset binding.
    return iree_ok_status();
  }
  // Preserve an available authored realization for the same reason; the
  // ordinary materializer already consumes it without rebuilding its terms.
  for (uint8_t i = 0; i < access->dynamic_realization_count; ++i) {
    const loom_low_source_memory_dynamic_realization_t* realization =
        &access->dynamic_realizations[i];
    if ((realization->term_mask & term_mask) != 0 &&
        loom_low_lower_source_value_has_low_mapping(context,
                                                    realization->term.index)) {
      return iree_ok_status();
    }
  }

  int64_t update_byte_stride = 0;
  if (!iree_checked_mul_i64(term->byte_stride, loop->step,
                            &update_byte_stride)) {
    return iree_ok_status();
  }
  uint32_t update_byte_shift = LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE;
  if (update_byte_stride > 0 &&
      iree_math_is_power_of_two_i64(update_byte_stride)) {
    update_byte_shift = (uint32_t)iree_math_count_trailing_zeros_u64(
        (uint64_t)update_byte_stride);
  }
  int64_t constant_stride = 0;
  const bool has_constant_stride =
      loom_low_lower_rule_source_memory_exact_term_stride(context, term,
                                                          &constant_stride);
  if (has_constant_stride &&
      !iree_checked_mul_i64(constant_stride, loop->step, &constant_stride)) {
    return iree_ok_status();
  }
  if (has_constant_stride && constant_stride == 0) {
    return iree_ok_status();
  }
  loom_low_lower_rule_source_memory_recurrence_resolution_t resolution;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_resolve_recurrence(
      context, selected_plan->rule_set, materializer,
      source_recurrence->term_identity,
      has_constant_stride ? constant_stride : 0, LOOM_VALUE_ID_INVALID,
      /*can_materialize_stride=*/true, &resolution));

  loom_low_lower_resolved_descriptor_t multiply_descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
          context, selected_plan->rule_set,
          materializer->multiply_descriptor_ref, &multiply_descriptor));
  loom_low_lower_resolved_descriptor_t shift_descriptor = {0};
  if ((term->byte_shift != LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE ||
       update_byte_shift != LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE) &&
      materializer->shift_left_descriptor_ref !=
          LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
            context, selected_plan->rule_set,
            materializer->shift_left_descriptor_ref, &shift_descriptor));
  }
  const uint32_t multiply_cost =
      loom_low_lower_rule_source_memory_descriptor_cost(
          context, multiply_descriptor.descriptor);
  const uint32_t shift_cost = loom_low_lower_rule_source_memory_descriptor_cost(
      context, shift_descriptor.descriptor);
  const uint32_t direct_scale_cost =
      term->byte_shift != LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE &&
              shift_descriptor.descriptor != NULL
          ? shift_cost
          : multiply_cost;
  const uint32_t update_scale_cost =
      update_byte_shift != LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE &&
              shift_descriptor.descriptor != NULL
          ? shift_cost
          : multiply_cost;
  uint32_t stride_materialization_cost = 0;
  if (resolution.uses_stride_operand) {
    if (has_constant_stride) {
      stride_materialization_cost = resolution.initializer_cost;
    } else {
      stride_materialization_cost =
          (uint32_t)(term->stride_value_count - 1u) * multiply_cost;
      if (update_byte_stride != 1) {
        stride_materialization_cost +=
            resolution.initializer_cost + update_scale_cost;
      }
    }
  }
  uint32_t direct_cost = term->stride_value_count * multiply_cost;
  if (term->byte_stride != 1) {
    direct_cost += resolution.initializer_cost + direct_scale_cost;
  }

  loom_low_lower_rule_source_memory_term_recurrence_t* recurrence = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, 1, sizeof(*recurrence), (void**)&recurrence));
  *recurrence = (loom_low_lower_rule_source_memory_term_recurrence_t){
      .loop = loop,
      .binding = binding,
      .rule_set = selected_plan->rule_set,
      .materializer = materializer,
      .term = term,
      .update_byte_stride = update_byte_stride,
      .update_byte_shift = update_byte_shift,
      .state = resolution.state,
      .constant_stride = has_constant_stride ? constant_stride : 0,
      .term_ordinal = source_recurrence->term_ordinal,
  };
  recurrence->recipe = (loom_low_lower_realization_recipe_t){
      .type = resolution.state.carrier_type,
      .id = (loom_low_lower_plan_id_t)(uintptr_t)&kSourceMemoryRecurrencePlanId,
      .key = iree_make_const_byte_span(&recurrence->state.key,
                                       sizeof(recurrence->state.key)),
      .initialize = loom_low_lower_rule_source_memory_initialize_recurrence,
      .update = loom_low_lower_rule_source_memory_update_recurrence,
      .data = &recurrence->state,
  };

  if (resolution.uses_stride_operand) {
    loom_low_lower_realization_input_t* inputs = NULL;
    if (!has_constant_stride) {
      recurrence->recipe.update =
          loom_low_lower_rule_source_memory_update_term_recurrence;
      recurrence->recipe.data = recurrence;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
          context, term->stride_value_count, sizeof(*inputs), (void**)&inputs));
      for (uint8_t i = 0; i < term->stride_value_count; ++i) {
        inputs[i] = (loom_low_lower_realization_input_t){
            .kind = LOOM_LOW_LOWER_REALIZATION_INPUT_SOURCE,
            .value.identity = term->stride_values[i],
        };
      }
      recurrence->recipe.inputs = inputs;
      recurrence->recipe.input_count = term->stride_value_count;
    } else {
      recurrence->stride_recipe = (loom_low_lower_realization_recipe_t){
          .type = resolution.state.carrier_type,
          .id = (loom_low_lower_plan_id_t)(uintptr_t)&kSourceMemoryStridePlanId,
          .key = iree_make_const_byte_span(&recurrence->constant_stride,
                                           sizeof(recurrence->constant_stride)),
          .initialize = loom_low_lower_rule_source_memory_materialize_stride,
          .data = recurrence,
      };
    }
  }

  binding->term_ordinal = source_recurrence->term_ordinal;
  const loom_low_lower_realization_offer_t offer = {
      .id = (loom_low_lower_plan_id_t)(uintptr_t)&kSourceMemoryRecurrencePlanId,
      .key = iree_make_const_byte_span(&recurrence->state.key,
                                       sizeof(recurrence->state.key)),
      .group_setup_cost =
          resolution.initializer_cost +
          (has_constant_stride ? stride_materialization_cost : 0),
      .group_iteration_cost =
          resolution.update_cost +
          (has_constant_stride ? 0 : stride_materialization_cost),
      .removed_key = (uint64_t)(uintptr_t)binding,
      .removed_iteration_cost = direct_cost,
      .minimum_removed_count = 2,
      .apply = loom_low_lower_rule_source_memory_apply_term_recurrence,
      .data = recurrence,
  };
  return loom_low_lower_realization_offer(context, emission_op, loop, &offer);
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_low_lower_rule_source_memory_consider_recurrence(
    loom_low_lower_context_t* context,
    loom_low_lower_rule_source_memory_plan_builder_t* builder,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_source_memory_byte_offset_materializer_t* materializer,
    const loom_low_lower_source_memory_recurrence_t* source_recurrence) {
  const loom_value_id_t producer_value_id =
      source_recurrence->producer_value_id;
  if (builder->producers_by_value_ordinal == NULL ||
      materializer->increment_descriptor_ref ==
          LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE ||
      materializer->increment_immediate_ordinal == LOOM_LOW_ID_NONE) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t producer_ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering,
                                            producer_value_id);
  loom_low_lower_selected_plan_t* producer_plan =
      builder->producers_by_value_ordinal[producer_ordinal];
  if (producer_plan == NULL) {
    return iree_ok_status();
  }
  const loom_op_t* producer_op = producer_plan->source_op;
  const loom_low_lower_realization_loop_t* loop =
      loom_low_lower_realization_loop(context, producer_op);
  if (loop == NULL ||
      producer_op->parent_block != loop->backedge->parent_block) {
    return iree_ok_status();
  }
  // This recipe initializes the carrier with zero and advances it by the
  // authored byte stride. Other induction domains require additional setup or
  // stride-scaling arithmetic that this materializer contract does not name.
  if (loop->initial_value != 0 || loop->step != 1) {
    return iree_ok_status();
  }
  const int64_t constant_stride = source_recurrence->constant_stride;
  const loom_value_id_t stride_value = source_recurrence->stride_value_id;

  loom_type_t carrier_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, producer_op, producer_value_id, &carrier_type));
  if (context->result->error_count != 0) {
    return iree_ok_status();
  }
  loom_low_lower_rule_source_memory_recurrence_resolution_t resolution;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_resolve_recurrence(
      context, rule_set, materializer,
      /*term_identity=*/0, constant_stride, stride_value,
      /*can_materialize_stride=*/false, &resolution));
  if (resolution.state.key.update_descriptor.descriptor == NULL ||
      !loom_type_equal(carrier_type, resolution.state.carrier_type)) {
    return iree_ok_status();
  }
  if (resolution.uses_stride_operand) {
    loom_type_t stride_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, producer_op,
                                                  stride_value, &stride_type));
    if (context->result->error_count != 0 ||
        !loom_type_equal(carrier_type, stride_type)) {
      return iree_ok_status();
    }
  }

  uint32_t direct_cost = 0;
  for (uint16_t i = 0; i < producer_plan->rule->emit_count; ++i) {
    direct_cost +=
        loom_low_lower_rule_source_memory_descriptor_cost(
            context, producer_plan->resolved_emits[i].descriptor.descriptor) +
        iree_math_count_ones_u32(
            producer_plan->resolved_emits[i].emit->copy_operand_mask);
  }
  loom_low_lower_rule_source_memory_recurrence_t* recurrence = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, 1, sizeof(*recurrence), (void**)&recurrence));
  *recurrence = (loom_low_lower_rule_source_memory_recurrence_t){
      .producer_plan = producer_plan,
      .loop = loop,
      .state = resolution.state,
      .initializer_cost = resolution.initializer_cost,
      .update_cost = resolution.update_cost,
      .direct_cost = direct_cost,
  };
  recurrence->recipe = (loom_low_lower_realization_recipe_t){
      .type = resolution.state.carrier_type,
      .id = (loom_low_lower_plan_id_t)(uintptr_t)&kSourceMemoryRecurrencePlanId,
      .key = iree_make_const_byte_span(&recurrence->state.key,
                                       sizeof(recurrence->state.key)),
      .inputs = resolution.uses_stride_operand ? &recurrence->state.stride_input
                                               : NULL,
      .input_count = resolution.uses_stride_operand,
      .initialize = loom_low_lower_rule_source_memory_initialize_recurrence,
      .update = loom_low_lower_rule_source_memory_update_recurrence,
      .data = &recurrence->state,
  };
  if (builder->last_recurrence != NULL) {
    builder->last_recurrence->next = recurrence;
  } else {
    builder->first_recurrence = recurrence;
  }
  builder->last_recurrence = recurrence;
  builder->producers_by_value_ordinal[producer_ordinal] = NULL;
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_source_memory_plan_offer_recurrences(
    loom_low_lower_context_t* context,
    loom_low_lower_rule_source_memory_plan_builder_t* builder) {
  for (loom_low_lower_rule_source_memory_recurrence_t* recurrence =
           builder->first_recurrence;
       recurrence != NULL; recurrence = recurrence->next) {
    const loom_value_id_t producer_value_id =
        loom_op_const_results(recurrence->producer_plan->source_op)[0];
    const loom_low_lower_realization_offer_t offer = {
        .id =
            (loom_low_lower_plan_id_t)(uintptr_t)&kSourceMemoryRecurrencePlanId,
        .key = iree_make_const_byte_span(&recurrence->state.key,
                                         sizeof(recurrence->state.key)),
        .group_setup_cost = recurrence->initializer_cost,
        .group_iteration_cost = recurrence->update_cost,
        .removed_key = producer_value_id,
        .removed_iteration_cost = recurrence->direct_cost,
        .recipe = &recurrence->recipe,
        .apply = loom_low_lower_rule_source_memory_apply,
        .data = recurrence,
    };
    IREE_RETURN_IF_ERROR(loom_low_lower_realization_offer(
        context, recurrence->loop->backedge, recurrence->loop, &offer));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_byte_offsets_grow(
    loom_low_lower_rule_source_memory_plan_builder_t* builder) {
  const iree_host_size_t bucket_count =
      builder->bucket_count ? builder->bucket_count * 2 : 32;
  loom_low_lower_rule_byte_offset_entry_t** buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, bucket_count, sizeof(*buckets), (void**)&buckets));
  memset(buckets, 0, bucket_count * sizeof(*buckets));
  for (iree_host_size_t i = 0; i < builder->bucket_count; ++i) {
    loom_low_lower_rule_byte_offset_entry_t* entry = builder->buckets[i];
    while (entry != NULL) {
      loom_low_lower_rule_byte_offset_entry_t* next = entry->next;
      const iree_host_size_t bucket = entry->hash & (bucket_count - 1);
      entry->next = buckets[bucket];
      buckets[bucket] = entry;
      entry = next;
    }
  }
  builder->buckets = buckets;
  builder->bucket_count = bucket_count;
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_source_memory_plan_byte_offset(
    loom_low_lower_context_t* context,
    loom_low_lower_rule_source_memory_plan_builder_t* builder,
    const loom_low_lower_selected_plan_t* selected_plan,
    const loom_low_lower_resolved_emit_t* resolved_emit, uint16_t emit_ordinal,
    uint16_t operand_ordinal, const void** out_binding) {
  const loom_low_lower_rule_set_t* rule_set = selected_plan->rule_set;
  const loom_low_lower_source_memory_t* source_memory =
      &rule_set
           ->source_memories[resolved_emit->emit->source_memory_ordinal - 1];
  const loom_low_source_memory_access_plan_t* access =
      &selected_plan->source_memory->access;
  const loom_low_lower_source_memory_byte_offset_materializer_t* materializer =
      loom_low_lower_rule_set_source_memory_byte_offset_materializer(
          rule_set, source_memory);
  const loom_low_lower_value_ref_t* ref =
      &rule_set->value_refs[resolved_emit->emit->operand_ref_start +
                            operand_ordinal];
  // A reused view base may already include a constant. Complete offsets always
  // include the canonical bias; dynamic offsets include only that embedded
  // contribution. Equal physical values can share across both reference kinds.
  const int64_t static_bias =
      ref->kind == LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_BYTE_OFFSET
          ? access->static_byte_offset
          : (loom_low_lower_rule_source_memory_reuses_view_base(source_memory,
                                                                access)
                 ? access->dynamic_view_base_value_static_byte_offset
                 : 0);
  const loom_low_lower_source_memory_record_t* memory_record =
      selected_plan->source_memory->record;
  const iree_host_size_t identity = memory_record->dynamic_byte_offset_identity;
  const loom_op_t* emission_op =
      selected_plan->emission_source_node_index
          ? selected_plan->data
                .source_nodes[selected_plan->emission_source_node_index]
          : selected_plan->source_op;
  const loom_low_lower_source_memory_recurrence_set_t* recurrences =
      memory_record->recurrences;
  if (recurrences != NULL) {
    for (uint8_t i = 0; i < recurrences->producer_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_rule_source_memory_consider_recurrence(
              context, builder, rule_set, materializer,
              &recurrences->producers[i]));
    }
  }
  const loom_block_t* block = emission_op->parent_block;
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_u64(hash, identity);
  hash = loom_structural_hash_mix_u64(hash, (uintptr_t)block);
  hash = loom_structural_hash_mix_u64(hash, (uintptr_t)rule_set);
  hash = loom_structural_hash_mix_u64(hash, (uintptr_t)materializer);
  hash = loom_structural_hash_mix_u64(hash, (uint64_t)static_bias);
  hash = loom_structural_hash_finalize(hash);
  loom_low_lower_rule_byte_offset_entry_t* entry =
      builder->bucket_count
          ? builder->buckets[hash & (builder->bucket_count - 1)]
          : NULL;
  for (; entry != NULL; entry = entry->next) {
    if (entry->hash == hash && entry->identity == identity &&
        entry->block == block && entry->rule_set == rule_set &&
        entry->materializer == materializer &&
        entry->static_bias == static_bias) {
      break;
    }
  }
  if (entry == NULL) {
    if (builder->entry_count >= builder->bucket_count * 3 / 4) {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_byte_offsets_grow(builder));
    }
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(builder->arena, sizeof(*entry), (void**)&entry));
    loom_low_lower_rule_byte_offset_binding_t* binding = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
        context, sizeof(*binding), (void**)&binding));
    *binding = (loom_low_lower_rule_byte_offset_binding_t){
        .definition = resolved_emit,
        .last_use_op = emission_op,
        .value = LOOM_VALUE_ID_INVALID,
        .operand_ordinal = operand_ordinal,
        .term_ordinal = UINT8_MAX,
    };
    const iree_host_size_t bucket = hash & (builder->bucket_count - 1);
    *entry = (loom_low_lower_rule_byte_offset_entry_t){
        .block = block,
        .rule_set = rule_set,
        .materializer = materializer,
        .identity = identity,
        .static_bias = static_bias,
        .binding = binding,
        .next = builder->buckets[bucket],
        .block_ordinal = emission_op->block_ordinal,
        .emit_ordinal = emit_ordinal,
        .hash = hash,
    };
    builder->buckets[bucket] = entry;
    ++builder->entry_count;
    if (recurrences != NULL && recurrences->has_term) {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_rule_source_memory_consider_term_recurrence(
              context, selected_plan, source_memory, materializer,
              &recurrences->term, emission_op, binding));
    }
  } else if (emission_op->block_ordinal < entry->block_ordinal ||
             (emission_op->block_ordinal == entry->block_ordinal &&
              (emit_ordinal < entry->emit_ordinal ||
               (emit_ordinal == entry->emit_ordinal &&
                operand_ordinal < entry->binding->operand_ordinal)))) {
    entry->block_ordinal = emission_op->block_ordinal;
    entry->emit_ordinal = emit_ordinal;
    entry->binding->definition = resolved_emit;
    entry->binding->operand_ordinal = operand_ordinal;
  }
  if (emission_op->block_ordinal > entry->binding->last_use_op->block_ordinal) {
    entry->binding->last_use_op = emission_op;
  }
  *out_binding = entry->binding;
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_source_memory_emit_byte_offset(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* access,
    const loom_low_lower_resolved_emit_t* resolved_emit,
    uint16_t operand_ordinal, loom_value_id_t* out_value) {
  loom_low_lower_rule_byte_offset_binding_t* binding =
      (loom_low_lower_rule_byte_offset_binding_t*)
          loom_low_lower_resolved_emit_materializer(resolved_emit,
                                                    operand_ordinal);
  if (binding->definition == resolved_emit &&
      binding->operand_ordinal == operand_ordinal) {
    loom_low_lower_source_memory_term_replacement_t replacement = {0};
    const loom_low_lower_source_memory_term_replacement_t* replacement_ptr =
        NULL;
    if (binding->term_realization != NULL) {
      replacement = (loom_low_lower_source_memory_term_replacement_t){
          .value_id =
              loom_low_lower_realization_value(binding->term_realization),
          .term_ordinal = binding->term_ordinal,
      };
      replacement_ptr = &replacement;
    }
    const loom_low_lower_value_ref_t* ref =
        &rule_set->value_refs[resolved_emit->emit->operand_ref_start +
                              operand_ordinal];
    if (ref->kind == LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_BYTE_OFFSET) {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_rule_materialize_source_memory_complete_byte_offset(
              context, rule_set, source_op, source_memory, access,
              replacement_ptr, &binding->value));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_rule_materialize_source_memory_byte_offset(
              context, rule_set, source_op, source_memory, access,
              replacement_ptr, &binding->value));
    }
  }
  *out_value = binding->value;
  return iree_ok_status();
}
