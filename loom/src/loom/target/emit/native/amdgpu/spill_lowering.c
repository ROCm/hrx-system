// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/spill_lowering.h"

#include <inttypes.h>

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"
#include "loom/target/registers.h"

#define LOOM_AMDGPU_SCRATCH_SPILL_UNIT_BITS 32u

static const loom_low_storage_space_set_t kLoomAmdgpuSpillStorageSpaces =
    LOOM_LOW_STORAGE_SPACE_SET_SCRATCH | LOOM_LOW_STORAGE_SPACE_SET_PRIVATE;

typedef struct loom_amdgpu_spill_descriptor_t {
  // Descriptor table row for the selected scratch packet.
  const loom_low_descriptor_t* descriptor;
  // Offset immediate row used to validate lowering-created attributes.
  const loom_low_immediate_t* offset_immediate;
} loom_amdgpu_spill_descriptor_t;

typedef struct loom_amdgpu_spill_access_t {
  // Byte offset relative to the referenced low storage handle.
  uint64_t storage_offset;
  // Byte offset relative to the AMDGPU private segment.
  int64_t segment_offset;
} loom_amdgpu_spill_access_t;

typedef enum loom_amdgpu_spill_register_kind_e {
  LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR = 1,
  LOOM_AMDGPU_SPILL_REGISTER_KIND_VGPR = 2,
} loom_amdgpu_spill_register_kind_t;

typedef struct loom_amdgpu_spill_register_t {
  // Register file being spilled or reloaded.
  loom_amdgpu_spill_register_kind_t kind;
  // Bytes in one allocation unit.
  uint32_t unit_bytes;
} loom_amdgpu_spill_register_t;

typedef struct loom_amdgpu_spill_lowering_context_t {
  // Module being rewritten.
  loom_module_t* module;
  // Low function owning the spill traffic being rewritten.
  loom_op_t* function_op;
  // Target-low descriptor set selected for this function.
  const loom_low_descriptor_set_t* descriptor_set;
  // Built fixed-segment layout for the function storage reservations.
  loom_amdgpu_storage_layout_t storage_layout;
  // Descriptor-set-local ID for the AMDGPU SGPR register class.
  uint16_t sgpr_class_id;
  // Bytes in one SGPR allocation unit.
  uint32_t sgpr_unit_bytes;
  // Descriptor-set-local ID for the AMDGPU VGPR register class.
  uint16_t vgpr_class_id;
  // Bytes in one VGPR allocation unit.
  uint32_t vgpr_unit_bytes;
  // Attribute name used by scratch packet descriptors.
  loom_string_id_t offset_attr_id;
  // Immediate attribute name used by M0 constant descriptors.
  loom_string_id_t imm32_attr_id;
  // Optional structured diagnostic emitter for user-authored spill traffic.
  iree_diagnostic_emitter_t emitter;
  // Mutable result receiving emitted diagnostic counts.
  loom_amdgpu_spill_lowering_result_t* result;
  // Scratch arena backing transient lowering state and result lists.
  iree_arena_allocator_t* scratch_arena;
  // Value IDs that the lowered traffic requires in registers.
  loom_value_id_t* required_register_value_ids;
  // Number of entries in |required_register_value_ids|.
  iree_host_size_t required_register_value_count;
  // Capacity of |required_register_value_ids|.
  iree_host_size_t required_register_value_capacity;
} loom_amdgpu_spill_lowering_context_t;

static iree_status_t loom_amdgpu_spill_lowering_record_required_register_value(
    loom_amdgpu_spill_lowering_context_t* context, loom_value_id_t value_id) {
  const iree_host_size_t minimum_capacity =
      context->required_register_value_count + 1;
  if (minimum_capacity > context->required_register_value_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        context->scratch_arena, context->required_register_value_count,
        iree_max(minimum_capacity, 8u),
        sizeof(*context->required_register_value_ids),
        &context->required_register_value_capacity,
        (void**)&context->required_register_value_ids));
  }
  context
      ->required_register_value_ids[context->required_register_value_count++] =
      value_id;
  return iree_ok_status();
}

static const loom_low_descriptor_t* loom_amdgpu_spill_lowering_descriptor_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set, descriptor_ref);
  IREE_ASSERT(descriptor != NULL,
              "generated descriptor set must satisfy spill lowering refs");
  return descriptor;
}

static loom_amdgpu_spill_descriptor_t
loom_amdgpu_spill_lowering_scratch_descriptor_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_spill_lowering_descriptor_ref(descriptor_set, descriptor_ref);

  const loom_amdgpu_descriptor_immediate_slots_t immediate_slots =
      loom_amdgpu_descriptor_immediate_slots(descriptor_set, descriptor);
  IREE_ASSERT(immediate_slots.address_offset != LOOM_LOW_ID_NONE,
              "generated spill descriptor must have an address offset");
  IREE_ASSERT(immediate_slots.address_offset < descriptor->immediate_count,
              "generated address offset slot must be descriptor-local");
  const uint32_t immediate_index =
      descriptor->immediate_start + immediate_slots.address_offset;
  IREE_ASSERT(immediate_index < descriptor_set->immediate_count,
              "generated address offset slot must reference an immediate");
  const loom_low_immediate_t* offset_immediate =
      &descriptor_set->immediates[immediate_index];

  return (loom_amdgpu_spill_descriptor_t){
      .descriptor = descriptor,
      .offset_immediate = offset_immediate,
  };
}

static bool loom_amdgpu_spill_lowering_offset_fits_immediate(
    const loom_low_immediate_t* immediate, int64_t offset) {
  switch (immediate->kind) {
    case LOOM_LOW_IMMEDIATE_KIND_SIGNED: {
      const int64_t maximum = immediate->unsigned_max > INT64_MAX
                                  ? INT64_MAX
                                  : (int64_t)immediate->unsigned_max;
      return offset >= immediate->signed_min && offset <= maximum;
    }
    case LOOM_LOW_IMMEDIATE_KIND_UNSIGNED:
    case LOOM_LOW_IMMEDIATE_KIND_ORDINAL:
      return offset >= 0 && (uint64_t)offset <= immediate->unsigned_max;
    default:
      return false;
  }
}

static void loom_amdgpu_spill_lowering_resolve_storage_reference(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reference_t* out_reference) {
  loom_amdgpu_storage_layout_lookup_reference(&context->storage_layout,
                                              context->module, storage_value_id,
                                              out_reference);
}

static bool loom_amdgpu_spill_lowering_storage_space_supported(
    const loom_amdgpu_storage_layout_reference_t* storage_reference) {
  return loom_low_storage_space_set_contains(
      kLoomAmdgpuSpillStorageSpaces, storage_reference->reservation.space);
}

static iree_status_t loom_amdgpu_spill_lowering_emit_unsupported_storage_space(
    const loom_amdgpu_spill_lowering_context_t* context, const loom_op_t* op,
    loom_value_id_t storage_value_id,
    const loom_amdgpu_storage_layout_reference_t* storage_reference) {
  const iree_string_view_t storage_space =
      loom_low_storage_type_space_name(storage_reference->reservation.space);
  if (context->emitter.fn == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU spill lowering cannot lower storage space '%.*s'",
        (int)storage_space.size, storage_space.data);
  }

  iree_string_view_t supported_storage_space_names[LOOM_STORAGE_SPACE_COUNT_];
  const iree_host_size_t supported_storage_space_count =
      loom_low_storage_space_set_names(
          kLoomAmdgpuSpillStorageSpaces,
          IREE_ARRAYSIZE(supported_storage_space_names),
          supported_storage_space_names);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(
          context->module, context->function_op)),
      loom_param_string(loom_op_name(context->module, op)),
      loom_param_string(
          loom_low_diagnostic_value_name(context->module, storage_value_id)),
      loom_param_string(storage_space),
      loom_param_string_list(supported_storage_space_names,
                             supported_storage_space_count),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_AMDGPU_037,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(context->emitter, &emission));
  ++context->result->error_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_resolve_access(
    const loom_amdgpu_storage_layout_reference_t* storage_reference,
    int64_t operation_offset, uint64_t chunk_byte_offset,
    uint64_t chunk_byte_length, loom_amdgpu_spill_access_t* out_access) {
  *out_access = (loom_amdgpu_spill_access_t){0};
  IREE_ASSERT(operation_offset >= 0,
              "spill access range must be validated before chunk lowering");
  const uint64_t access_offset = (uint64_t)operation_offset + chunk_byte_offset;
  const uint64_t access_end = access_offset + chunk_byte_length;
  IREE_ASSERT(access_end <= storage_reference->byte_length,
              "spill access range must fit before chunk lowering");

  uint64_t absolute_offset = 0;
  if (!iree_checked_add_u64(storage_reference->reservation.byte_offset,
                            storage_reference->byte_offset, &absolute_offset) ||
      !iree_checked_add_u64(absolute_offset, access_offset, &absolute_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU scratch spill offset overflows");
  }
  if (absolute_offset > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU scratch spill offset %" PRIu64
                            " exceeds int64_t range",
                            absolute_offset);
  }
  out_access->storage_offset = access_offset;
  out_access->segment_offset = (int64_t)absolute_offset;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_emit_out_of_bounds_access(
    const loom_amdgpu_spill_lowering_context_t* context, const loom_op_t* op,
    loom_value_id_t storage_value_id, int64_t access_byte_offset,
    uint64_t access_byte_length,
    const loom_amdgpu_storage_layout_reference_t* storage_reference) {
  if (context->emitter.fn == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU spill lowering cannot lower out-of-bounds spill access");
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(
          context->module, context->function_op)),
      loom_param_string(loom_op_name(context->module, op)),
      loom_param_string(
          loom_low_diagnostic_value_name(context->module, storage_value_id)),
      loom_param_i64(access_byte_offset),
      loom_param_u64(access_byte_length),
      loom_param_u64(storage_reference->byte_length),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_AMDGPU_039,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(context->emitter, &emission));
  ++context->result->error_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_validate_access_range(
    const loom_amdgpu_spill_lowering_context_t* context, const loom_op_t* op,
    loom_value_id_t storage_value_id,
    const loom_amdgpu_storage_layout_reference_t* storage_reference,
    int64_t operation_offset, uint64_t access_byte_length,
    bool* out_supported) {
  *out_supported = false;
  if (operation_offset < 0) {
    return loom_amdgpu_spill_lowering_emit_out_of_bounds_access(
        context, op, storage_value_id, operation_offset, access_byte_length,
        storage_reference);
  }
  const uint64_t access_offset = (uint64_t)operation_offset;
  if (access_offset > UINT64_MAX - access_byte_length) {
    return loom_amdgpu_spill_lowering_emit_out_of_bounds_access(
        context, op, storage_value_id, (int64_t)access_offset,
        access_byte_length, storage_reference);
  }
  const uint64_t access_end = access_offset + access_byte_length;
  if (access_end > storage_reference->byte_length) {
    return loom_amdgpu_spill_lowering_emit_out_of_bounds_access(
        context, op, storage_value_id, (int64_t)access_offset,
        access_byte_length, storage_reference);
  }
  *out_supported = true;
  return iree_ok_status();
}

static uint32_t loom_amdgpu_spill_lowering_chunk_units(
    uint32_t remaining_units) {
  if (remaining_units >= 4) {
    return 4;
  }
  if (remaining_units >= 2) {
    return 2;
  }
  return 1;
}

static loom_amdgpu_descriptor_ref_t
loom_amdgpu_spill_lowering_load_descriptor_ref(uint32_t chunk_units) {
  switch (chunk_units) {
    case 1:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B32_OFFSET_ONLY;
    case 2:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B64_OFFSET_ONLY;
    default:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B128_OFFSET_ONLY;
  }
}

static loom_amdgpu_descriptor_ref_t
loom_amdgpu_spill_lowering_load_vaddr_descriptor_ref(uint32_t chunk_units) {
  switch (chunk_units) {
    case 1:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B32_VADDR;
    case 2:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B64_VADDR;
    default:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B128_VADDR;
  }
}

static loom_amdgpu_descriptor_ref_t
loom_amdgpu_spill_lowering_store_descriptor_ref(uint32_t chunk_units) {
  switch (chunk_units) {
    case 1:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B32_OFFSET_ONLY;
    case 2:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B64_OFFSET_ONLY;
    default:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B128_OFFSET_ONLY;
  }
}

static loom_amdgpu_descriptor_ref_t
loom_amdgpu_spill_lowering_store_vaddr_descriptor_ref(uint32_t chunk_units) {
  switch (chunk_units) {
    case 1:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B32_VADDR;
    case 2:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B64_VADDR;
    default:
      return LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B128_VADDR;
  }
}

static iree_status_t loom_amdgpu_spill_lowering_make_offset_attr(
    const loom_amdgpu_spill_lowering_context_t* context, int64_t offset,
    loom_named_attr_t* out_attr) {
  *out_attr = (loom_named_attr_t){
      .name_id = context->offset_attr_id,
      .value = loom_attr_i64(offset),
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_make_chunk_type(
    loom_module_t* module, loom_type_t base_type, uint32_t chunk_units,
    loom_type_t* out_type) {
  *out_type = loom_type_none();
  loom_type_t chunk_type =
      loom_low_register_carrier_type_with_unit_count(base_type, chunk_units);
  return loom_module_intern_type(module, chunk_type, out_type);
}

static iree_status_t loom_amdgpu_spill_lowering_make_register_type(
    const loom_amdgpu_spill_lowering_context_t* context, uint16_t class_id,
    uint32_t unit_count, loom_type_t* out_type) {
  *out_type = loom_type_none();
  loom_type_t register_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      context->descriptor_set, class_id, unit_count, &register_type));
  return loom_module_intern_type(context->module, register_type, out_type);
}

static iree_status_t loom_amdgpu_spill_lowering_resolve_register_type(
    const loom_amdgpu_spill_lowering_context_t* context, loom_type_t type,
    loom_amdgpu_spill_register_t* out_register, bool* out_supported) {
  *out_register = (loom_amdgpu_spill_register_t){0};
  *out_supported = false;
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          context->descriptor_set->stable_id) {
    return iree_ok_status();
  }
  const uint16_t class_id = loom_low_register_type_class_id(type);
  if (class_id == context->vgpr_class_id) {
    *out_register = (loom_amdgpu_spill_register_t){
        .kind = LOOM_AMDGPU_SPILL_REGISTER_KIND_VGPR,
        .unit_bytes = context->vgpr_unit_bytes,
    };
    *out_supported = true;
    return iree_ok_status();
  }
  if (class_id == context->sgpr_class_id) {
    *out_register = (loom_amdgpu_spill_register_t){
        .kind = LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR,
        .unit_bytes = context->sgpr_unit_bytes,
    };
    *out_supported = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_emit_unsupported_register_type(
    const loom_amdgpu_spill_lowering_context_t* context, const loom_op_t* op,
    loom_value_id_t value_id, loom_type_t type) {
  if (context->emitter.fn == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU spill lowering cannot lower spill value type");
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(
          context->module, context->function_op)),
      loom_param_string(loom_op_name(context->module, op)),
      loom_param_string(
          loom_low_diagnostic_value_name(context->module, value_id)),
      loom_param_type(type),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_AMDGPU_038,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(context->emitter, &emission));
  ++context->result->error_count;
  return iree_ok_status();
}

static uint32_t loom_amdgpu_spill_lowering_register_chunk_units(
    const loom_amdgpu_spill_register_t* spill_register,
    uint32_t remaining_units) {
  if (spill_register->kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR) {
    return 1;
  }
  return loom_amdgpu_spill_lowering_chunk_units(remaining_units);
}

static uint32_t loom_amdgpu_spill_lowering_access_chunk_units(
    const loom_amdgpu_spill_register_t* spill_register, uint32_t chunk_units,
    int64_t segment_offset) {
  if (spill_register->kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR) {
    return 1;
  }
  IREE_ASSERT(segment_offset >= 0,
              "spill access range must be resolved before chunk selection");
  while (chunk_units > 1) {
    const uint32_t chunk_byte_length = chunk_units * spill_register->unit_bytes;
    if (((uint64_t)segment_offset % chunk_byte_length) == 0) {
      return chunk_units;
    }
    chunk_units /= 2;
  }
  return chunk_units;
}

static iree_status_t loom_amdgpu_spill_lowering_build_register_convert(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_value_id_t source, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_spill_lowering_descriptor_ref(context->descriptor_set,
                                                descriptor_ref);
  const loom_value_id_t operands[] = {source};
  const loom_type_t result_types[] = {result_type};
  loom_op_t* convert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, context->descriptor_set, descriptor, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
      result_types, IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &convert_op));
  *out_value = loom_low_op_results(convert_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_build_m0_zero(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, const loom_low_descriptor_t* consumer_descriptor,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t m0_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_descriptor_implicit_resource_type(
      context->descriptor_set, consumer_descriptor, &m0_type));
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_spill_lowering_descriptor_ref(
          context->descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM);
  const loom_named_attr_t imm32_attr = {
      .name_id = context->imm32_attr_id,
      .value = loom_attr_i64(0),
  };
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      &rewriter->builder, context->descriptor_set, descriptor,
      loom_make_named_attr_slice(&imm32_attr, 1), m0_type, location,
      &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_materialize_store_value(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter,
    const loom_amdgpu_spill_register_t* spill_register, loom_value_id_t value,
    uint32_t chunk_units, loom_location_id_t location,
    loom_value_id_t* out_value, loom_type_t* out_type) {
  *out_value = value;
  *out_type = loom_module_value_type(context->module, value);
  if (spill_register->kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_VGPR) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(chunk_units, 1u);
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_make_register_type(
      context, context->vgpr_class_id, 1, out_type));
  return loom_amdgpu_spill_lowering_build_register_convert(
      context, rewriter, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY, value,
      *out_type, location, out_value);
}

static iree_status_t loom_amdgpu_spill_lowering_materialize_loaded_value(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter,
    const loom_amdgpu_spill_register_t* spill_register, loom_value_id_t value,
    uint32_t chunk_units, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = value;
  if (spill_register->kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_VGPR) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(chunk_units, 1u);
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_make_register_type(
      context, context->sgpr_class_id, 1, &sgpr_type));
  return loom_amdgpu_spill_lowering_build_register_convert(
      context, rewriter, LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32, value,
      sgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_spill_lowering_build_exec_read(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_location_id_t location,
    loom_value_id_t* out_exec) {
  *out_exec = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_spill_lowering_descriptor_ref(
          context->descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ);
  loom_type_t exec_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_make_register_type(
      context, context->sgpr_class_id, 2, &exec_type));
  loom_op_t* read_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, context->descriptor_set, descriptor,
      /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
      &exec_type, 1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &read_op));
  *out_exec = loom_low_op_results(read_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_build_exec_write(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_value_id_t exec,
    loom_location_id_t location) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_spill_lowering_descriptor_ref(
          context->descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC);
  loom_op_t* write_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      &rewriter->builder, context->descriptor_set, descriptor, &exec, 1,
      loom_named_attr_slice_empty(),
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &write_op);
}

static iree_status_t loom_amdgpu_spill_lowering_build_full_exec_write(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_location_id_t location) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_spill_lowering_descriptor_ref(
          context->descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_FULL);
  loom_op_t* write_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      &rewriter->builder, context->descriptor_set, descriptor,
      /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
      /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &write_op);
}

static iree_status_t loom_amdgpu_spill_lowering_enter_full_exec(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_location_id_t location,
    loom_value_id_t* out_saved_exec) {
  *out_saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_exec_read(
      context, rewriter, location, out_saved_exec));
  return loom_amdgpu_spill_lowering_build_full_exec_write(context, rewriter,
                                                          location);
}

static iree_status_t loom_amdgpu_spill_lowering_make_scratch_load_type(
    const loom_amdgpu_spill_lowering_context_t* context,
    const loom_amdgpu_spill_register_t* spill_register, uint32_t chunk_units,
    loom_type_t source_type, loom_type_t* out_type) {
  if (spill_register->kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR) {
    return loom_amdgpu_spill_lowering_make_register_type(
        context, context->vgpr_class_id, chunk_units, out_type);
  }
  if (chunk_units == loom_low_register_type_unit_count(source_type)) {
    *out_type = source_type;
    return iree_ok_status();
  }
  return loom_amdgpu_spill_lowering_make_chunk_type(
      context->module, source_type, chunk_units, out_type);
}

static uint32_t loom_amdgpu_spill_lowering_register_class_unit_bytes(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t class_id) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT(class_id < descriptor_set->reg_class_count,
              "AMDGPU descriptor set must provide generated register class");
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[class_id];
  IREE_ASSERT_EQ(reg_class->alloc_unit_bits,
                 LOOM_AMDGPU_SCRATCH_SPILL_UNIT_BITS);
  return reg_class->alloc_unit_bits / 8u;
}

static iree_status_t loom_amdgpu_spill_lowering_build_slice(
    loom_rewriter_t* rewriter, loom_value_id_t source, uint32_t source_units,
    uint32_t chunk_start, uint32_t chunk_units, loom_type_t source_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (source_units == chunk_units) {
    *out_value = source;
    return iree_ok_status();
  }
  loom_type_t chunk_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_make_chunk_type(
      rewriter->module, source_type, chunk_units, &chunk_type));
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(&rewriter->builder, source,
                                            chunk_start, chunk_type, location,
                                            &slice_op));
  *out_value = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_build_storage_address(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_value_id_t storage,
    const loom_amdgpu_spill_access_t* access, loom_type_t base_type,
    loom_location_id_t location, loom_value_id_t* out_address) {
  *out_address = LOOM_VALUE_ID_INVALID;
  if (access->storage_offset > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU scratch spill storage offset %" PRIu64
                            " exceeds int64_t range",
                            access->storage_offset);
  }
  loom_type_t address_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_make_chunk_type(
      context->module, base_type, 1, &address_type));
  loom_op_t* address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
      &rewriter->builder, storage, (int64_t)access->storage_offset,
      address_type, location, &address_op));
  *out_address = loom_low_storage_address_result(address_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_store_chunk(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_value_id_t storage, loom_value_id_t value,
    uint32_t chunk_units, const loom_amdgpu_spill_access_t* access,
    loom_type_t value_type, loom_location_id_t location) {
  loom_amdgpu_spill_descriptor_t spill_descriptor =
      loom_amdgpu_spill_lowering_scratch_descriptor_ref(
          context->descriptor_set,
          loom_amdgpu_spill_lowering_store_descriptor_ref(chunk_units));
  loom_value_id_t operands[3] = {value, LOOM_VALUE_ID_INVALID,
                                 LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 1;
  int64_t offset = access->segment_offset;
  if (!loom_amdgpu_spill_lowering_offset_fits_immediate(
          spill_descriptor.offset_immediate, offset)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_storage_address(
        context, rewriter, storage, access, value_type, location,
        &operands[0]));
    operands[1] = value;
    operand_count = 2;
    offset = 0;
    spill_descriptor = loom_amdgpu_spill_lowering_scratch_descriptor_ref(
        context->descriptor_set,
        loom_amdgpu_spill_lowering_store_vaddr_descriptor_ref(chunk_units));
  }
  IREE_ASSERT(
      loom_amdgpu_spill_lowering_offset_fits_immediate(
          spill_descriptor.offset_immediate, offset),
      "selected spill descriptor must encode the lowering-created offset");
  if (loom_low_descriptor_implicit_resource_operand(
          context->descriptor_set, spill_descriptor.descriptor) != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_m0_zero(
        context, rewriter, spill_descriptor.descriptor, location,
        &operands[operand_count++]));
  }
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_make_offset_attr(context, offset, &attr));
  loom_op_t* store_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      &rewriter->builder, context->descriptor_set, spill_descriptor.descriptor,
      operands, operand_count, loom_make_named_attr_slice(&attr, 1),
      /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &store_op);
}

static iree_status_t loom_amdgpu_spill_lowering_load_chunk(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_value_id_t storage, uint32_t chunk_units,
    loom_type_t result_type, const loom_amdgpu_spill_access_t* access,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_spill_descriptor_t spill_descriptor =
      loom_amdgpu_spill_lowering_scratch_descriptor_ref(
          context->descriptor_set,
          loom_amdgpu_spill_lowering_load_descriptor_ref(chunk_units));
  loom_value_id_t operands[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 0;
  int64_t offset = access->segment_offset;
  if (!loom_amdgpu_spill_lowering_offset_fits_immediate(
          spill_descriptor.offset_immediate, offset)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_storage_address(
        context, rewriter, storage, access, result_type, location,
        &operands[0]));
    operand_count = 1;
    offset = 0;
    spill_descriptor = loom_amdgpu_spill_lowering_scratch_descriptor_ref(
        context->descriptor_set,
        loom_amdgpu_spill_lowering_load_vaddr_descriptor_ref(chunk_units));
  }
  IREE_ASSERT(
      loom_amdgpu_spill_lowering_offset_fits_immediate(
          spill_descriptor.offset_immediate, offset),
      "selected reload descriptor must encode the lowering-created offset");
  if (loom_low_descriptor_implicit_resource_operand(
          context->descriptor_set, spill_descriptor.descriptor) != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_m0_zero(
        context, rewriter, spill_descriptor.descriptor, location,
        &operands[operand_count++]));
  }
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_make_offset_attr(context, offset, &attr));
  const loom_type_t result_types[] = {result_type};
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, context->descriptor_set, spill_descriptor.descriptor,
      operands, operand_count, loom_make_named_attr_slice(&attr, 1),
      result_types, IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &load_op));
  *out_value = loom_low_op_results(load_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_rewrite_spill(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_op_t* op) {
  const loom_value_id_t value = loom_low_spill_value(op);
  const loom_type_t value_type = loom_module_value_type(context->module, value);
  loom_amdgpu_spill_register_t spill_register = {0};
  bool register_supported = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_register_type(
      context, value_type, &spill_register, &register_supported));
  if (!register_supported) {
    return loom_amdgpu_spill_lowering_emit_unsupported_register_type(
        context, op, value, value_type);
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(value_type);
  if (unit_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering found a zero-unit spill value");
  }

  const loom_value_id_t storage = loom_low_spill_storage(op);
  loom_amdgpu_storage_layout_reference_t storage_reference = {0};
  loom_amdgpu_spill_lowering_resolve_storage_reference(context, storage,
                                                       &storage_reference);
  if (!loom_amdgpu_spill_lowering_storage_space_supported(&storage_reference)) {
    return loom_amdgpu_spill_lowering_emit_unsupported_storage_space(
        context, op, storage, &storage_reference);
  }
  const uint64_t access_byte_length =
      (uint64_t)unit_count * spill_register.unit_bytes;
  bool access_supported = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_validate_access_range(
      context, op, storage, &storage_reference, loom_low_spill_offset(op),
      access_byte_length, &access_supported));
  if (!access_supported) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
  // low.spill has no lane-mask operand and snapshots the whole logical register
  // value. AMDGPU scratch/private packets are lane-private and EXEC-gated, so
  // emit spill traffic under full EXEC for every supported register class.
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_enter_full_exec(
      context, rewriter, op->location, &saved_exec));
  for (uint32_t chunk_start = 0; chunk_start < unit_count;) {
    uint32_t chunk_units = loom_amdgpu_spill_lowering_register_chunk_units(
        &spill_register, unit_count - chunk_start);
    const uint64_t chunk_byte_offset =
        (uint64_t)chunk_start * spill_register.unit_bytes;
    const uint64_t chunk_byte_length =
        (uint64_t)chunk_units * spill_register.unit_bytes;
    loom_amdgpu_spill_access_t access = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_access(
        &storage_reference, loom_low_spill_offset(op), chunk_byte_offset,
        chunk_byte_length, &access));
    chunk_units = loom_amdgpu_spill_lowering_access_chunk_units(
        &spill_register, chunk_units, access.segment_offset);
    loom_value_id_t chunk_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_slice(
        rewriter, value, unit_count, chunk_start, chunk_units, value_type,
        op->location, &chunk_value));
    loom_type_t scratch_value_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_materialize_store_value(
        context, rewriter, &spill_register, chunk_value, chunk_units,
        op->location, &chunk_value, &scratch_value_type));
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_store_chunk(
        context, rewriter, storage, chunk_value, chunk_units, &access,
        scratch_value_type, op->location));
    chunk_start += chunk_units;
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_exec_write(
      context, rewriter, saved_exec, op->location));
  return loom_rewriter_erase(rewriter, op);
}

static iree_status_t loom_amdgpu_spill_lowering_rewrite_reload(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_op_t* op) {
  const loom_value_id_t result = loom_low_reload_result(op);
  const loom_type_t result_type =
      loom_module_value_type(context->module, result);
  loom_amdgpu_spill_register_t spill_register = {0};
  bool register_supported = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_register_type(
      context, result_type, &spill_register, &register_supported));
  if (!register_supported) {
    return loom_amdgpu_spill_lowering_emit_unsupported_register_type(
        context, op, result, result_type);
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(result_type);
  if (unit_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering found a zero-unit reload result");
  }

  const loom_value_id_t storage = loom_low_reload_storage(op);
  loom_amdgpu_storage_layout_reference_t storage_reference = {0};
  loom_amdgpu_spill_lowering_resolve_storage_reference(context, storage,
                                                       &storage_reference);
  if (!loom_amdgpu_spill_lowering_storage_space_supported(&storage_reference)) {
    return loom_amdgpu_spill_lowering_emit_unsupported_storage_space(
        context, op, storage, &storage_reference);
  }
  const uint64_t access_byte_length =
      (uint64_t)unit_count * spill_register.unit_bytes;
  bool access_supported = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_validate_access_range(
      context, op, storage, &storage_reference, loom_low_reload_offset(op),
      access_byte_length, &access_supported));
  if (!access_supported) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
  // low.reload restores the whole logical register value. Load every
  // lane-private slot under full EXEC before restoring the caller's EXEC mask.
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_enter_full_exec(
      context, rewriter, op->location, &saved_exec));
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t* loaded_chunks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, unit_count,
                                                 sizeof(*loaded_chunks),
                                                 (void**)&loaded_chunks));
  iree_host_size_t loaded_chunk_count = 0;
  for (uint32_t chunk_start = 0; chunk_start < unit_count;) {
    uint32_t chunk_units = loom_amdgpu_spill_lowering_register_chunk_units(
        &spill_register, unit_count - chunk_start);
    const uint64_t chunk_byte_offset =
        (uint64_t)chunk_start * spill_register.unit_bytes;
    const uint64_t chunk_byte_length =
        (uint64_t)chunk_units * spill_register.unit_bytes;
    loom_amdgpu_spill_access_t access = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_access(
        &storage_reference, loom_low_reload_offset(op), chunk_byte_offset,
        chunk_byte_length, &access));
    chunk_units = loom_amdgpu_spill_lowering_access_chunk_units(
        &spill_register, chunk_units, access.segment_offset);
    loom_type_t chunk_type = result_type;
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_make_scratch_load_type(
        context, &spill_register, chunk_units, result_type, &chunk_type));
    loom_value_id_t loaded_chunk = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_load_chunk(
        context, rewriter, storage, chunk_units, chunk_type, &access,
        op->location, &loaded_chunk));
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_materialize_loaded_value(
        context, rewriter, &spill_register, loaded_chunk, chunk_units,
        op->location, &loaded_chunks[loaded_chunk_count++]));
    chunk_start += chunk_units;
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_exec_write(
      context, rewriter, saved_exec, op->location));

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (loaded_chunk_count == 1) {
    replacement = loaded_chunks[0];
  } else {
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_concat_build(
        &rewriter->builder, loaded_chunks, loaded_chunk_count, result_type,
        op->location, &concat_op));
    replacement = loom_low_concat_result(concat_op);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_amdgpu_spill_lowering_collect_ops(
    loom_region_t* body, iree_arena_allocator_t* arena, loom_op_t*** out_ops,
    iree_host_size_t* out_op_count) {
  *out_ops = NULL;
  *out_op_count = 0;
  iree_host_size_t op_capacity = 0;
  loom_block_t* block = NULL;
  loom_op_t* op = NULL;
  loom_region_for_each_block(body, block) {
    loom_block_for_each_op(block, op) {
      if (!loom_low_spill_isa(op) && !loom_low_reload_isa(op)) {
        continue;
      }
      const iree_host_size_t minimum_capacity = *out_op_count + 1;
      if (minimum_capacity > op_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            arena, *out_op_count, iree_max(minimum_capacity, 8u),
            sizeof(**out_ops), &op_capacity, (void**)out_ops));
      }
      (*out_ops)[(*out_op_count)++] = op;
    }
  }
  return iree_ok_status();
}

// Spill lowering precedes schedule construction. Build its private storage
// projection only after spill discovery proves that rewriting is required.
static iree_status_t loom_amdgpu_spill_lowering_build_storage_layout(
    const loom_module_t* module, const loom_region_t* body,
    iree_arena_allocator_t* arena,
    loom_amdgpu_storage_layout_t* out_storage_layout) {
  loom_low_storage_layout_builder_t builder;
  loom_low_storage_layout_builder_initialize(&builder);
  const loom_block_t* block = NULL;
  const loom_op_t* op = NULL;
  loom_region_for_each_block(body, block) {
    loom_block_for_each_op(block, op) {
      if (!loom_low_storage_reserve_isa(op)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_low_storage_layout_builder_append(module, op, arena, &builder));
    }
  }
  loom_low_storage_layout_t source_layout;
  loom_low_storage_layout_builder_finish(&builder, &source_layout);
  return loom_amdgpu_storage_layout_build(&source_layout, arena,
                                          out_storage_layout);
}

static iree_status_t loom_amdgpu_spill_lowering_initialize_context(
    loom_module_t* module, loom_op_t* function_op, loom_region_t* body,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_spill_lowering_result_t* result,
    loom_amdgpu_spill_lowering_context_t* out_context,
    iree_arena_allocator_t* scratch_arena) {
  *out_context = (loom_amdgpu_spill_lowering_context_t){
      .module = module,
      .function_op = function_op,
      .descriptor_set = descriptor_set,
      .emitter = emitter,
      .result = result,
      .scratch_arena = scratch_arena,
  };
  out_context->sgpr_class_id = LOOM_AMDGPU_REG_CLASS_ID_SGPR;
  out_context->sgpr_unit_bytes =
      loom_amdgpu_spill_lowering_register_class_unit_bytes(
          descriptor_set, out_context->sgpr_class_id);
  out_context->vgpr_class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR;
  out_context->vgpr_unit_bytes =
      loom_amdgpu_spill_lowering_register_class_unit_bytes(
          descriptor_set, out_context->vgpr_class_id);
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("offset"),
                                                 &out_context->offset_attr_id));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("imm32"),
                                                 &out_context->imm32_attr_id));
  return loom_amdgpu_spill_lowering_build_storage_layout(
      module, body, scratch_arena, &out_context->storage_layout);
}

iree_status_t loom_amdgpu_lower_spill_traffic(
    loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_spill_lowering_result_t* out_result,
    iree_arena_allocator_t* scratch_arena) {
  if (module == NULL || function_op == NULL || descriptor_set == NULL ||
      out_result == NULL || scratch_arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering requires module, function, descriptor set, "
        "result, and scratch arena");
  }
  *out_result = (loom_amdgpu_spill_lowering_result_t){0};
  loom_region_t* body = loom_low_function_body(function_op);
  if (body == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering requires a low function body");
  }

  loom_op_t** ops = NULL;
  iree_host_size_t op_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_collect_ops(
      body, scratch_arena, &ops, &op_count));
  if (op_count == 0) {
    return iree_ok_status();
  }

  loom_amdgpu_spill_lowering_context_t context = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_initialize_context(
      module, function_op, body, descriptor_set, emitter, out_result, &context,
      scratch_arena));

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, scratch_arena));
  // This is terminal target lowering: no later structural spill layer can
  // recursively spill its packet helpers. Preserve every created helper and
  // every existing value consumed by a lowered spill store in registers.
  const loom_value_id_t first_generated_value_id = module->values.count;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < op_count && iree_status_is_ok(status); ++i) {
    if (loom_low_spill_isa(ops[i])) {
      status = loom_amdgpu_spill_lowering_record_required_register_value(
          &context, loom_low_spill_value(ops[i]));
      if (iree_status_is_ok(status)) {
        status = loom_amdgpu_spill_lowering_rewrite_spill(&context, &rewriter,
                                                          ops[i]);
      }
    } else if (loom_low_reload_isa(ops[i])) {
      status = loom_amdgpu_spill_lowering_rewrite_reload(&context, &rewriter,
                                                         ops[i]);
    }
  }
  for (loom_value_id_t value_id = first_generated_value_id;
       value_id < module->values.count && iree_status_is_ok(status);
       ++value_id) {
    status = loom_amdgpu_spill_lowering_record_required_register_value(
        &context, value_id);
  }
  loom_rewriter_deinitialize(&rewriter);
  if (iree_status_is_ok(status)) {
    out_result->required_register_value_ids =
        context.required_register_value_ids;
    out_result->required_register_value_count =
        context.required_register_value_count;
  }
  return status;
}
