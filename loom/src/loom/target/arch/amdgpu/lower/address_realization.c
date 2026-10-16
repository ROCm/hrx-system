// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/address_realization.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/analysis/symbolic_projection.h"
#include "loom/codegen/low/lower/source_memory.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/buffer_resource.h"
#include "loom/target/arch/amdgpu/lower/buffer_descriptor.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/fragment_memory/address.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/memory_address.h"
#include "loom/target/arch/amdgpu/lower/memory_packet.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"

typedef enum loom_amdgpu_address_realization_kind_e {
  LOOM_AMDGPU_ADDRESS_REALIZATION_COMPONENT = 0,
  LOOM_AMDGPU_ADDRESS_REALIZATION_DESCRIPTOR_ROOT = 1,
  LOOM_AMDGPU_ADDRESS_REALIZATION_DESCRIPTOR_OFFSET = 2,
  LOOM_AMDGPU_ADDRESS_REALIZATION_SCALAR_OFFSET = 3,
} loom_amdgpu_address_realization_kind_t;

typedef struct loom_amdgpu_address_component_key_t {
  // Constant byte coefficient of each invariant source term.
  int64_t coefficients[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // ABI lane-coordinate recipe; all unused entries are zero.
  loom_amdgpu_fragment_memory_lane_term_t
      lane_terms[LOOM_MATRIX_FRAGMENT_AXIS_COUNT];
  // Linear lane byte stride when the physical layout selects that recipe.
  uint32_t linear_lane_byte_stride;
  // Alternating byte-address bit, or zero for an invariant.
  uint32_t bank_bit;
  // Initial bank contribution to the physical address.
  uint32_t initial_byte_offset;
  // Number of invariant source terms.
  uint16_t term_count;
  // Number of lane projection terms.
  uint16_t lane_term_count;
} loom_amdgpu_address_component_key_t;

typedef struct loom_amdgpu_address_component_t {
  // Selected terms borrowed from the function-owned canonical access plan.
  loom_amdgpu_memory_dynamic_term_sequence_t terms;
  // Lane recipe borrowed from a function-owned fragment plan, or NULL.
  const loom_amdgpu_fragment_memory_address_layout_t* layout;
  // Alternating byte-address bit used by the backedge update.
  uint32_t bank_bit;
  // Entry-phase bank contribution.
  uint32_t initial_byte_offset;
} loom_amdgpu_address_component_t;

static bool loom_amdgpu_address_alternating_bank(
    loom_low_lower_context_t* context,
    const loom_low_lower_realization_loop_t* loop,
    const loom_low_source_memory_access_plan_t* source, uint16_t invariant_mask,
    uint32_t* out_bank_bit, uint32_t* out_initial_byte_offset) {
  const uint16_t all_terms =
      (uint16_t)((1u << source->dynamic_term_count) - 1u);
  const uint16_t periodic_mask = all_terms & (uint16_t)~invariant_mask;
  if (periodic_mask == 0 || (periodic_mask & (periodic_mask - 1u)) != 0 ||
      loop->initial_value < 0 || !(loop->step & 1)) {
    return false;
  }
  const uint8_t term_index =
      (uint8_t)iree_math_count_trailing_zeros_u32(periodic_mask);
  const loom_low_source_memory_dynamic_term_t* term =
      &source->dynamic_terms[term_index];
  if (term->source != LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE ||
      term->stride_value_count != 0 || term->byte_stride <= 0 ||
      term->byte_stride > INT32_MAX ||
      ((uint32_t)term->byte_stride & ((uint32_t)term->byte_stride - 1u)) != 0) {
    return false;
  }
  loom_symbolic_expr_summary_t summary;
  if (!loom_symbolic_expr_context_try_lookup_summary(
          loom_low_lower_context_symbolic_expr_context(context), term->index,
          &summary) ||
      summary.projection == NULL) {
    return false;
  }
  const loom_symbolic_projection_t* projection = summary.projection;
  if (projection->value_id != loop->induction_value || projection->scale != 1 ||
      projection->divisor != 1 || projection->modulus != 2) {
    return false;
  }
  *out_bank_bit = (uint32_t)term->byte_stride;
  *out_initial_byte_offset =
      ((uint64_t)loop->initial_value + (uint64_t)projection->offset) % 2
          ? *out_bank_bit
          : 0;
  return true;
}

static bool loom_amdgpu_address_invariant_bound(
    const loom_low_source_memory_access_plan_t* source, uint16_t mask,
    uint64_t* out_maximum) {
  uint64_t maximum = 0;
  for (uint8_t i = 0; i < source->dynamic_term_count; ++i) {
    if (!(mask & (1u << i))) {
      continue;
    }
    const loom_low_source_memory_dynamic_term_t* term =
        &source->dynamic_terms[i];
    int64_t term_maximum = 0;
    if (term->stride_value_count != 0 ||
        !loom_value_facts_as_non_negative_i64_maximum(term->byte_facts,
                                                      &term_maximum) ||
        (uint64_t)term_maximum > UINT32_MAX - maximum) {
      return false;
    }
    maximum += (uint64_t)term_maximum;
  }
  *out_maximum = maximum;
  return true;
}

static iree_status_t loom_amdgpu_initialize_address_component(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const void* data, loom_value_id_t carried_value,
    loom_value_id_t* out_value) {
  (void)carried_value;
  const loom_amdgpu_address_component_t* component = data;
  const loom_amdgpu_memory_access_t access = {
      .vaddr_static_byte_offset = component->initial_byte_offset,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vaddr(
      context, source_op, &access, &component->terms, LOOM_VALUE_ID_INVALID,
      out_value));
  if (component->layout) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_lane_offset(
        context, source_op, component->layout, *out_value, out_value));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_update_address_component(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const void* data, loom_value_id_t carried_value,
    loom_value_id_t* out_value) {
  const loom_amdgpu_address_component_t* component = data;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT,
      carried_value, component->bank_bit, vgpr_type, out_value);
}

static iree_status_t loom_amdgpu_request_address_component(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_realization_loop_t* loop,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_fragment_memory_address_layout_t* layout,
    uint16_t invariant_mask, uint32_t bank_bit, uint32_t initial_byte_offset,
    const loom_low_lower_realization_t** out_realization) {
  loom_amdgpu_address_component_key_t key;
  memset(&key, 0, sizeof(key));
  key.bank_bit = bank_bit;
  key.initial_byte_offset = initial_byte_offset;
  loom_low_lower_realization_input_t
      inputs[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY + 1];
  uint16_t input_count = 0;
  loom_amdgpu_memory_dynamic_term_sequence_t terms = {0};
  for (uint8_t i = 0; i < source->dynamic_term_count; ++i) {
    if (invariant_mask & (1u << i)) {
      terms.terms[terms.count++] = &source->dynamic_terms[i];
    }
  }
  for (uint8_t i = 0; i < terms.count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term = terms.terms[i];
    terms.kinds[i] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
    inputs[input_count++] = (loom_low_lower_realization_input_t){
        .kind = LOOM_LOW_LOWER_REALIZATION_INPUT_SOURCE,
        .value.identity = term->index,
    };
    key.coefficients[key.term_count++] = term->byte_stride;
  }
  if (layout) {
    key.linear_lane_byte_stride = layout->linear_lane_byte_stride;
    key.lane_term_count = layout->lane_term_count;
    memcpy(key.lane_terms, layout->lane_terms,
           layout->lane_term_count * sizeof(*layout->lane_terms));
    inputs[input_count++] = (loom_low_lower_realization_input_t){
        .kind = LOOM_LOW_LOWER_REALIZATION_INPUT_ENTRY,
        .value.identity = LOOM_OP_KERNEL_SUBGROUP_LANE_ID,
    };
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_plan_memory_term_sequence(context, &terms));
  loom_amdgpu_address_component_t* component = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*component), (void**)&component));
  *component = (loom_amdgpu_address_component_t){
      .terms = terms,
      .layout = layout,
      .bank_bit = bank_bit,
      .initial_byte_offset = initial_byte_offset,
  };
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  const loom_low_lower_realization_recipe_t recipe = {
      .type = vgpr_type,
      .id = LOOM_AMDGPU_ADDRESS_REALIZATION_COMPONENT,
      .key = iree_make_const_byte_span(&key, sizeof(key)),
      .inputs = inputs,
      .input_count = input_count,
      .initialize = loom_amdgpu_initialize_address_component,
      .update = loop ? loom_amdgpu_update_address_component : NULL,
      .data = component,
  };
  return loom_low_lower_realization_request(context, source_op, loop, &recipe,
                                            out_realization);
}

typedef struct loom_amdgpu_scalar_offset_t {
  // Canonical uniform source contribution, or NULL for a constant offset.
  const loom_low_source_memory_dynamic_term_t* term;
  // Constant bytes added to the dynamic contribution.
  uint32_t static_byte_offset;
} loom_amdgpu_scalar_offset_t;

static iree_status_t loom_amdgpu_initialize_scalar_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const void* data, loom_value_id_t carried_value,
    loom_value_id_t* out_value) {
  (void)carried_value;
  const loom_amdgpu_scalar_offset_t* offset = data;
  const loom_amdgpu_memory_dynamic_term_sequence_t terms = {
      .terms = {offset->term},
      .kinds = {LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET},
      .count = offset->term ? 1 : 0,
  };
  return loom_amdgpu_emit_sgpr_byte_offset_terms(
      context, source_op, &terms, offset->static_byte_offset, out_value);
}

static iree_status_t loom_amdgpu_request_scalar_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_dynamic_term_t* term,
    uint32_t static_byte_offset,
    const loom_low_lower_realization_t** out_realization) {
  const int64_t key[] = {term ? term->byte_stride : 0, static_byte_offset};
  const loom_low_lower_realization_input_t input = {
      .kind = LOOM_LOW_LOWER_REALIZATION_INPUT_SOURCE,
      .value.identity = term ? term->index : LOOM_VALUE_ID_INVALID,
  };
  loom_amdgpu_scalar_offset_t* offset = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*offset), (void**)&offset));
  *offset = (loom_amdgpu_scalar_offset_t){
      .term = term, .static_byte_offset = static_byte_offset};
  loom_type_t scalar_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &scalar_type));
  const loom_low_lower_realization_recipe_t recipe = {
      .type = scalar_type,
      .id = LOOM_AMDGPU_ADDRESS_REALIZATION_SCALAR_OFFSET,
      .key = iree_make_const_byte_span(key, sizeof(key)),
      .inputs = &input,
      .input_count = term ? 1 : 0,
      .initialize = loom_amdgpu_initialize_scalar_offset,
      .data = offset,
  };
  return loom_low_lower_realization_request(context, source_op, NULL, &recipe,
                                            out_realization);
}

typedef struct loom_amdgpu_descriptor_offset_t {
  // Immutable descriptor retaining the original resource's extent and control.
  const loom_low_lower_realization_t* root;
  // Shared uniform nonnegative byte offset used by every participating root.
  const loom_low_lower_realization_t* byte_offset;
} loom_amdgpu_descriptor_offset_t;

static iree_status_t loom_amdgpu_initialize_descriptor_root(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const void* data, loom_value_id_t carried_value,
    loom_value_id_t* out_value) {
  (void)carried_value;
  const loom_low_source_memory_access_plan_t* source = data;
  loom_value_id_t binding = loom_low_lower_lookup_value(
      context, loom_low_source_memory_access_base_view_value_id(source));
  // Shared descriptor selection requires a fixed explicit resource extent.
  return loom_amdgpu_emit_hal_buffer_descriptor(
      context, source_op, binding, source, /*plan=*/NULL, out_value);
}

static iree_status_t loom_amdgpu_initialize_descriptor_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const void* data, loom_value_id_t carried_value,
    loom_value_id_t* out_value) {
  (void)carried_value;
  const loom_amdgpu_descriptor_offset_t* offset = data;
  const loom_value_id_t root = loom_low_lower_realization_value(offset->root);
  loom_type_t scalar_type = loom_type_none();
  loom_type_t pointer_type = loom_type_none();
  loom_type_t descriptor_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &scalar_type));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &pointer_type));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 4, &descriptor_type));
  loom_value_id_t base = LOOM_VALUE_ID_INVALID;
  loom_value_id_t extent = LOOM_VALUE_ID_INVALID;
  loom_value_id_t control = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, root, 0,
                                                  pointer_type, &base));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, root, 2,
                                                  scalar_type, &extent));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, root, 3,
                                                  scalar_type, &control));
  const loom_value_id_t byte_offset =
      loom_low_lower_realization_value(offset->byte_offset);
  loom_value_id_t shifted_words[2];
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_add_u32_offset(
      context, source_op, base, byte_offset, shifted_words));
  // Preparation proves the offset is bounded by both the imported resource
  // and source view extents. Subtraction cannot wrap for either descriptor
  // extent source, including an offset exactly at the end of the resource.
  loom_value_id_t remaining_extent = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_SUB_U32, extent,
      byte_offset, scalar_type, &remaining_extent));
  const loom_value_id_t fields[] = {shifted_words[0], shifted_words[1],
                                    remaining_extent, control};
  return loom_amdgpu_build_low_register_range(context, source_op, fields,
                                              IREE_ARRAYSIZE(fields),
                                              descriptor_type, out_value);
}

static iree_status_t loom_amdgpu_request_offset_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source, uint8_t varying_index,
    const loom_low_lower_realization_t** out_realization) {
  loom_type_t descriptor_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 4, &descriptor_type));
  const loom_low_lower_realization_input_t root_input = {
      .kind = LOOM_LOW_LOWER_REALIZATION_INPUT_SOURCE,
      .value.identity =
          loom_low_source_memory_access_base_view_value_id(source),
  };
  const loom_low_lower_realization_recipe_t root_recipe = {
      .type = descriptor_type,
      .id = LOOM_AMDGPU_ADDRESS_REALIZATION_DESCRIPTOR_ROOT,
      .key = iree_make_const_byte_span(&source->view_value_id,
                                       sizeof(source->view_value_id)),
      .inputs = &root_input,
      .input_count = 1,
      .initialize = loom_amdgpu_initialize_descriptor_root,
      .data = source,
  };
  const loom_low_lower_realization_t* root = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_realization_request(
      context, source_op, NULL, &root_recipe, &root));
  if (root == NULL) {
    return iree_ok_status();
  }
  const loom_low_source_memory_dynamic_term_t* term =
      &source->dynamic_terms[varying_index];
  const loom_low_lower_realization_t* byte_offset = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_request_scalar_offset(
      context, source_op, term, 0, &byte_offset));
  if (byte_offset == NULL) {
    return iree_ok_status();
  }
  const loom_low_lower_realization_input_t inputs[] = {
      {.kind = LOOM_LOW_LOWER_REALIZATION_INPUT_REALIZATION,
       .value.realization = root},
      {.kind = LOOM_LOW_LOWER_REALIZATION_INPUT_REALIZATION,
       .value.realization = byte_offset},
  };
  loom_amdgpu_descriptor_offset_t* offset = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*offset), (void**)&offset));
  *offset = (loom_amdgpu_descriptor_offset_t){.root = root,
                                              .byte_offset = byte_offset};
  const loom_low_lower_realization_recipe_t recipe = {
      .type = descriptor_type,
      .id = LOOM_AMDGPU_ADDRESS_REALIZATION_DESCRIPTOR_OFFSET,
      .key = iree_const_byte_span_empty(),
      .inputs = inputs,
      .input_count = IREE_ARRAYSIZE(inputs),
      .initialize = loom_amdgpu_initialize_descriptor_offset,
      .data = offset,
  };
  return loom_low_lower_realization_request(context, source_op, NULL, &recipe,
                                            out_realization);
}

typedef struct loom_amdgpu_global_address_replacement_t {
  // Direct packet plan replaced only after shared cost selection succeeds.
  loom_amdgpu_memory_access_t* access;
  // Selected buffer packet descriptor.
  const loom_low_descriptor_t* descriptor;
  // Static byte offset encoded in the buffer packet.
  int64_t immediate_offset;
  // Invariant scalar bytes outside the buffer packet's immediate range.
  uint32_t scalar_byte_offset;
  // Canonical terms belonging to the invariant vector component.
  uint16_t invariant_mask;
  // Sole uniform loop-varying source term.
  uint8_t varying_index;
} loom_amdgpu_global_address_replacement_t;

static iree_status_t loom_amdgpu_apply_global_address_realizations(
    loom_low_lower_context_t* context, const loom_op_t* source_op, void* data) {
  const loom_amdgpu_global_address_replacement_t* replacement = data;
  loom_amdgpu_memory_access_t candidate = *replacement->access;
  candidate.address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
  candidate.descriptor = replacement->descriptor;
  candidate.immediate_offset = replacement->immediate_offset;
  candidate.secondary_immediate_offset = 0;
  candidate.vaddr_static_byte_offset = 0;
  candidate.scalar_byte_offset = replacement->scalar_byte_offset;
  candidate.scalar_base_byte_offset = 0;
  const loom_low_source_memory_access_plan_t* source =
      replacement->access->source;
  IREE_RETURN_IF_ERROR(loom_amdgpu_request_address_component(
      context, source_op, NULL, source, NULL, replacement->invariant_mask, 0, 0,
      &candidate.realization.vaddr));
  if (candidate.realization.vaddr == NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_request_offset_descriptor(
      context, source_op, source, replacement->varying_index,
      &candidate.realization.descriptor));
  if (candidate.realization.descriptor != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_request_scalar_offset(
        context, source_op, NULL, candidate.scalar_byte_offset,
        &candidate.realization.soffset));
    *replacement->access = candidate;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_prepare_global_address_realizations(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_realization_loop_t* loop,
    loom_amdgpu_memory_access_t* access) {
  const loom_low_source_memory_access_plan_t* source = access->source;
  if (source->operation_kind != LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD ||
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR ||
      source->dynamic_view_base_term_count != 0 ||
      source->static_view_base_byte_offset != 0 ||
      access->scalar_base_byte_offset != 0 ||
      access->vaddr_static_byte_offset != 0) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_amdgpu_descriptor_set_info_t* target_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  if (target_info->buffer_resource.record_encoding ==
          LOOM_AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_NONE ||
      loom_amdgpu_buffer_resource_record_encoding_info(
          target_info->buffer_resource.record_encoding)
              ->num_records_word1_bit_count != 0) {
    return iree_ok_status();
  }
  // A fixed, root-relative source view gives the initializer all range facts
  // without introducing undeclared dependencies on dynamic view dimensions.
  const loom_value_fact_table_t* facts =
      loom_low_lower_context_fact_table(context);
  loom_value_fact_view_reference_t view = {0};
  int64_t extent = 0;
  if (!loom_value_facts_query_view_reference(
          &facts->context,
          loom_value_fact_table_lookup(facts, source->view_value_id), &view) ||
      !loom_value_facts_is_zero(view.base_byte_offset) ||
      !loom_value_facts_as_exact_i64(view.footprint_byte_length, &extent) ||
      extent <= 0 || extent > UINT32_MAX) {
    return iree_ok_status();
  }
  const uint16_t invariant_mask =
      loom_low_lower_source_memory_invariant_terms(context);
  const uint16_t varying_mask =
      (uint16_t)((1u << source->dynamic_term_count) - 1u) &
      (uint16_t)~invariant_mask;
  uint64_t invariant_maximum = 0;
  if (invariant_mask == 0 || varying_mask == 0 ||
      (varying_mask & (varying_mask - 1u)) != 0 ||
      !loom_amdgpu_address_invariant_bound(source, invariant_mask,
                                           &invariant_maximum)) {
    return iree_ok_status();
  }
  // The setup estimate covers a single retained invariant term and its byte
  // conversion. Larger sums need a shared source realization, not an assumed
  // cheap reconstruction of arbitrary canonical expressions.
  if ((invariant_mask & (invariant_mask - 1u)) != 0 &&
      (access->retained_component_kind !=
           LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR ||
       source->retained_component.term_mask != invariant_mask)) {
    return iree_ok_status();
  }
  for (uint8_t i = 0; i < source->dynamic_term_count; ++i) {
    if (access->dynamic_term_kinds[i] !=
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR) {
      return iree_ok_status();
    }
  }
  const uint8_t varying_index =
      (uint8_t)iree_math_count_trailing_zeros_u32(varying_mask);
  const loom_low_source_memory_dynamic_term_t* term =
      &source->dynamic_terms[varying_index];
  int64_t varying_maximum = 0;
  if (term->source != LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE ||
      term->stride_value_count != 0 || term->byte_stride <= 0 ||
      term->byte_stride > UINT32_MAX ||
      !loom_value_facts_is_subgroup_uniform(
          loom_value_fact_table_lookup(facts, term->index)) ||
      !loom_value_facts_as_non_negative_i64_maximum(term->byte_facts,
                                                    &varying_maximum) ||
      varying_maximum > extent) {
    return iree_ok_status();
  }
  const uint16_t root_argument =
      loom_low_lower_source_memory_root_argument_index(context, source);
  if (root_argument == UINT16_MAX) {
    return iree_ok_status();
  }
  uint16_t argument_count = 0;
  const loom_low_lower_abi_argument_t* argument =
      &loom_low_lower_context_argument_map(context,
                                           &argument_count)[root_argument];
  const loom_low_resource_build_flags_t extent_flags =
      LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_EXTENT |
      LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_EXTENT_VALUE;
  if (argument->kind != LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE ||
      (argument->resource_build_flags & extent_flags) !=
          LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_EXTENT ||
      varying_maximum > argument->resource_extent) {
    return iree_ok_status();
  }
  loom_amdgpu_memory_access_t candidate = *access;
  candidate.immediate_offset = 0;
  candidate.secondary_immediate_offset = 0;
  candidate.vaddr_static_byte_offset = 0;
  candidate.scalar_byte_offset = 0;
  candidate.scalar_base_byte_offset = 0;
  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  if (!loom_amdgpu_memory_access_try_select_buffer(
          descriptor_set, source->operation_kind, &candidate, &diagnostic) ||
      !loom_amdgpu_memory_cache_policy_can_lower(descriptor_set, &candidate)) {
    return iree_ok_status();
  }
  loom_amdgpu_global_address_replacement_t* replacement = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*replacement), (void**)&replacement));
  *replacement = (loom_amdgpu_global_address_replacement_t){
      .access = access,
      .descriptor = candidate.descriptor,
      .immediate_offset = candidate.immediate_offset,
      .scalar_byte_offset = candidate.scalar_byte_offset,
      .invariant_mask = invariant_mask,
      .varying_index = varying_index,
  };
  const uint64_t key[] = {source->view_value_id, term->index,
                          (uint64_t)term->byte_stride};
  // Each distinct biased SADDR costs a full-width add/carry pair. The shared
  // SRD costs an add/carry pair, extent subtraction and uniform byte scaling
  // per iteration. Root setup and each packet's invariant conversion/constant
  // are charged separately; vector arithmetic savings receive no credit.
  // These are issue-cost estimates, not guarantees about final allocation.
  const loom_low_lower_realization_offer_t offer = {
      .id = LOOM_AMDGPU_ADDRESS_REALIZATION_DESCRIPTOR_OFFSET,
      .key = iree_make_const_byte_span(key, sizeof(key)),
      .group_setup_cost = 6,
      .group_iteration_cost = 4,
      .removed_key = access->scalar_byte_offset,
      .removed_iteration_cost = access->scalar_byte_offset ? 2 : 0,
      .setup_cost = 2 + (candidate.scalar_byte_offset != 0),
      .apply = loom_amdgpu_apply_global_address_realizations,
      .data = replacement,
  };
  return loom_low_lower_realization_offer(context, source_op, loop, &offer);
}

iree_status_t loom_amdgpu_prepare_memory_address_realizations(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_t* access) {
  const loom_low_lower_realization_loop_t* loop =
      loom_low_lower_realization_loop(context, source_op);
  if (loop == NULL) {
    return iree_ok_status();
  }
  if (access->source->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL) {
    return loom_amdgpu_prepare_global_address_realizations(context, source_op,
                                                           loop, access);
  }
  if (access->source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT ||
      access->static_byte_offset < 0 || access->vaddr_static_byte_offset != 0) {
    return iree_ok_status();
  }
  const uint16_t invariant_mask =
      loom_low_lower_source_memory_invariant_terms(context);
  uint32_t bank_bit = 0;
  uint32_t initial_byte_offset = 0;
  uint64_t invariant_maximum = 0;
  if (!loom_amdgpu_address_alternating_bank(context, loop, access->source,
                                            invariant_mask, &bank_bit,
                                            &initial_byte_offset) ||
      !loom_amdgpu_address_invariant_bound(access->source, invariant_mask,
                                           &invariant_maximum) ||
      invariant_maximum + (uint64_t)access->static_byte_offset +
              access->packet_byte_count >
          bank_bit) {
    return iree_ok_status();
  }
  return loom_amdgpu_request_address_component(
      context, source_op, loop, access->source, NULL, invariant_mask, bank_bit,
      initial_byte_offset, &access->realization.vaddr);
}

iree_status_t loom_amdgpu_prepare_fragment_address_realization(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* plan) {
  const loom_low_lower_realization_loop_t* loop =
      loom_low_lower_realization_loop(context, source_op);
  if (loop == NULL ||
      plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      plan->operation_kind != LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD ||
      plan->payload_form != LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE) {
    return iree_ok_status();
  }
  for (uint8_t i = 0; i < plan->view_rank; ++i) {
    if (plan->runtime_axes[i].byte_stride.kind ==
        LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_DYNAMIC) {
      return iree_ok_status();
    }
  }
  const uint16_t invariant_mask =
      loom_low_lower_source_memory_invariant_terms(context);
  uint32_t bank_bit = 0;
  uint32_t initial_byte_offset = 0;
  uint64_t invariant_maximum = 0;
  if (!loom_amdgpu_address_alternating_bank(context, loop, &plan->source,
                                            invariant_mask, &bank_bit,
                                            &initial_byte_offset) ||
      !loom_amdgpu_address_invariant_bound(&plan->source, invariant_mask,
                                           &invariant_maximum)) {
    return iree_ok_status();
  }
  uint64_t lane_maximum = 0;
  const uint32_t wave_size =
      loom_low_lower_context_bundle(context)->snapshot->subgroup_size;
  for (uint32_t lane = 0; lane < wave_size; ++lane) {
    const uint64_t offset =
        loom_amdgpu_fragment_memory_relative_lane_byte_offset(
            &plan->address_layout, (uint8_t)lane);
    if (offset > lane_maximum) {
      lane_maximum = offset;
    }
  }
  for (uint16_t i = 0; i < plan->register_count; ++i) {
    int64_t offset = 0;
    if (!loom_amdgpu_fragment_memory_static_offset_i64(
            plan, i, plan->address_layout.payload_elements_per_register - 1,
            &offset) ||
        offset < 0 ||
        invariant_maximum + lane_maximum + (uint64_t)offset +
                plan->element_byte_count >
            bank_bit) {
      return iree_ok_status();
    }
  }
  return loom_amdgpu_request_address_component(
      context, source_op, loop, &plan->source, &plan->address_layout,
      invariant_mask, bank_bit, initial_byte_offset,
      &plan->address_realization);
}
