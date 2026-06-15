// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/spill_lowering.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"
#include "loom/target/registers.h"

#define LOOM_AMDGPU_SCRATCH_SPILL_UNIT_BITS 32u

typedef struct loom_amdgpu_spill_descriptor_t {
  // Descriptor table row for the selected scratch packet.
  const loom_low_descriptor_t* descriptor;
  // Module string ID for the descriptor key.
  loom_string_id_t opcode_id;
  // Offset immediate row used to validate lowering-created attributes.
  const loom_low_immediate_t* offset_immediate;
} loom_amdgpu_spill_descriptor_t;

typedef struct loom_amdgpu_spill_opcode_descriptor_t {
  // Descriptor table row for the selected packet.
  const loom_low_descriptor_t* descriptor;
  // Module string ID for the descriptor key.
  loom_string_id_t opcode_id;
} loom_amdgpu_spill_opcode_descriptor_t;

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
} loom_amdgpu_spill_lowering_context_t;

static iree_status_t loom_amdgpu_spill_lowering_checked_add_u64(
    uint64_t lhs, uint64_t rhs, uint64_t* out_result) {
  *out_result = 0;
  if (lhs > UINT64_MAX - rhs) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU scratch spill offset overflows");
  }
  *out_result = lhs + rhs;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_resolve_descriptor_ref(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_spill_opcode_descriptor_t* out_opcode_descriptor) {
  *out_opcode_descriptor = (loom_amdgpu_spill_opcode_descriptor_t){0};
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set, descriptor_ref);
  if (descriptor == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "AMDGPU spill lowering references missing descriptor ref %" PRIu16,
        descriptor_ref);
  }
  iree_string_view_t key = loom_low_descriptor_set_string(
      descriptor_set, descriptor->key_string_offset);
  if (iree_string_view_is_empty(key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU spill lowering descriptor ref %" PRIu16
                            " has no descriptor key",
                            descriptor_ref);
  }
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(rewriter->module, key, &opcode_id));

  *out_opcode_descriptor = (loom_amdgpu_spill_opcode_descriptor_t){
      .descriptor = descriptor,
      .opcode_id = opcode_id,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_resolve_scratch_descriptor_ref(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_spill_descriptor_t* out_spill_descriptor) {
  *out_spill_descriptor = (loom_amdgpu_spill_descriptor_t){0};
  loom_amdgpu_spill_opcode_descriptor_t opcode_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_descriptor_ref(
      rewriter, descriptor_set, descriptor_ref, &opcode_descriptor));
  const loom_low_descriptor_t* descriptor = opcode_descriptor.descriptor;

  const loom_low_immediate_t* offset_immediate = NULL;
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const uint32_t immediate_index = descriptor->immediate_start + i;
    if (immediate_index >= descriptor_set->immediate_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU spill lowering descriptor ref %" PRIu16
                              " references immediate row %" PRIu32
                              " outside the descriptor set",
                              descriptor_ref, immediate_index);
    }
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[immediate_index];
    iree_string_view_t immediate_name = loom_low_descriptor_set_string(
        descriptor_set, immediate->field_name_string_offset);
    if (iree_string_view_equal(immediate_name, IREE_SV("offset"))) {
      offset_immediate = immediate;
      break;
    }
  }
  if (offset_immediate == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "AMDGPU spill lowering descriptor ref %" PRIu16
                            " has no offset immediate",
                            descriptor_ref);
  }

  *out_spill_descriptor = (loom_amdgpu_spill_descriptor_t){
      .descriptor = opcode_descriptor.descriptor,
      .opcode_id = opcode_descriptor.opcode_id,
      .offset_immediate = offset_immediate,
  };
  return iree_ok_status();
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

static iree_status_t loom_amdgpu_spill_lowering_validate_offset_immediate(
    const loom_amdgpu_spill_descriptor_t* spill_descriptor, int64_t offset) {
  const loom_low_immediate_t* immediate = spill_descriptor->offset_immediate;
  switch (immediate->kind) {
    case LOOM_LOW_IMMEDIATE_KIND_SIGNED: {
      const int64_t maximum = immediate->unsigned_max > INT64_MAX
                                  ? INT64_MAX
                                  : (int64_t)immediate->unsigned_max;
      if (offset >= immediate->signed_min && offset <= maximum) {
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU scratch spill offset %" PRId64
          " does not fit the selected offset-only scratch encoding range "
          "[%" PRId64 ", %" PRId64 "]",
          offset, immediate->signed_min, maximum);
    }
    case LOOM_LOW_IMMEDIATE_KIND_UNSIGNED:
    case LOOM_LOW_IMMEDIATE_KIND_ORDINAL:
      if (offset >= 0 && (uint64_t)offset <= immediate->unsigned_max) {
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU scratch spill offset %" PRId64
          " does not fit the selected offset-only scratch encoding range "
          "[0, %" PRIu64 "]",
          offset, immediate->unsigned_max);
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU scratch spill offset immediate has unsupported kind %u",
          (unsigned)immediate->kind);
  }
}

static iree_status_t loom_amdgpu_spill_lowering_resolve_storage_reference(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reference_t* out_reference) {
  return loom_amdgpu_storage_layout_lookup_reference(
      &context->storage_layout, context->module, storage_value_id,
      out_reference);
}

static iree_status_t loom_amdgpu_spill_lowering_validate_storage_space(
    const loom_amdgpu_storage_layout_reference_t* storage_reference) {
  switch (storage_reference->reservation.space) {
    case LOOM_STORAGE_SPACE_PRIVATE:
    case LOOM_STORAGE_SPACE_SCRATCH:
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AMDGPU spill lowering only supports private or scratch storage, not "
          "%s storage",
          loom_storage_space_name(storage_reference->reservation.space));
  }
}

static iree_status_t loom_amdgpu_spill_lowering_resolve_access(
    const loom_amdgpu_storage_layout_reference_t* storage_reference,
    int64_t operation_offset, uint64_t chunk_byte_offset,
    uint64_t chunk_byte_length, loom_amdgpu_spill_access_t* out_access) {
  *out_access = (loom_amdgpu_spill_access_t){0};
  if (operation_offset < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering requires non-negative spill offsets");
  }
  uint64_t access_offset = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_checked_add_u64(
      (uint64_t)operation_offset, chunk_byte_offset, &access_offset));
  uint64_t access_end = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_checked_add_u64(
      access_offset, chunk_byte_length, &access_end));
  if (access_end > storage_reference->byte_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU spill lowering access range [%" PRIu64 ", %" PRIu64
        ") exceeds storage reference size %" PRIu64,
        access_offset, access_end, storage_reference->byte_length);
  }

  uint64_t absolute_offset = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_checked_add_u64(
      storage_reference->reservation.byte_offset,
      storage_reference->byte_offset, &absolute_offset));
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_checked_add_u64(
      absolute_offset, access_offset, &absolute_offset));
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
      loom_low_register_type_with_unit_count(base_type, chunk_units);
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
    loom_amdgpu_spill_register_t* out_register) {
  *out_register = (loom_amdgpu_spill_register_t){0};
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          context->descriptor_set->stable_id) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU spill lowering supports only AMDGPU SGPR or VGPR register "
        "values");
  }
  const uint16_t class_id = loom_low_register_type_class_id(type);
  if (class_id == context->vgpr_class_id) {
    if (context->vgpr_unit_bytes == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU spill lowering found a zero-byte VGPR allocation unit");
    }
    *out_register = (loom_amdgpu_spill_register_t){
        .kind = LOOM_AMDGPU_SPILL_REGISTER_KIND_VGPR,
        .unit_bytes = context->vgpr_unit_bytes,
    };
    return iree_ok_status();
  }
  if (class_id == context->sgpr_class_id) {
    if (context->sgpr_unit_bytes == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU spill lowering found a zero-byte SGPR allocation unit");
    }
    *out_register = (loom_amdgpu_spill_register_t){
        .kind = LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR,
        .unit_bytes = context->sgpr_unit_bytes,
    };
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU spill lowering supports only AMDGPU SGPR or VGPR register "
      "values");
}

static uint32_t loom_amdgpu_spill_lowering_register_chunk_units(
    const loom_amdgpu_spill_register_t* spill_register,
    uint32_t remaining_units) {
  if (spill_register->kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR) {
    return 1;
  }
  return loom_amdgpu_spill_lowering_chunk_units(remaining_units);
}

static iree_status_t loom_amdgpu_spill_lowering_build_register_convert(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_value_id_t source, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_spill_opcode_descriptor_t opcode_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_descriptor_ref(
      rewriter, context->descriptor_set, descriptor_ref, &opcode_descriptor));
  const loom_value_id_t operands[] = {source};
  const loom_type_t result_types[] = {result_type};
  loom_op_t* convert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, context->descriptor_set, opcode_descriptor.descriptor,
      opcode_descriptor.opcode_id, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &convert_op));
  *out_value = loom_low_op_results(convert_op).values[0];
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
  if (chunk_units != 1) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU SGPR spill lowering expects one-unit scratch chunks");
  }
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
  if (chunk_units != 1) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU SGPR reload lowering expects one-unit scratch chunks");
  }
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
  loom_amdgpu_spill_opcode_descriptor_t opcode_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_descriptor_ref(
      rewriter, context->descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ, &opcode_descriptor));
  loom_type_t exec_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_make_register_type(
      context, context->sgpr_class_id, 2, &exec_type));
  loom_op_t* read_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, context->descriptor_set, opcode_descriptor.descriptor,
      opcode_descriptor.opcode_id, /*operands=*/NULL, /*operand_count=*/0,
      loom_named_attr_slice_empty(), &exec_type, 1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &read_op));
  *out_exec = loom_low_op_results(read_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_build_exec_write(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_value_id_t exec,
    loom_location_id_t location) {
  loom_amdgpu_spill_opcode_descriptor_t opcode_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_descriptor_ref(
      rewriter, context->descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC, &opcode_descriptor));
  loom_op_t* write_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      &rewriter->builder, context->descriptor_set, opcode_descriptor.descriptor,
      opcode_descriptor.opcode_id, &exec, 1, loom_named_attr_slice_empty(),
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &write_op);
}

static iree_status_t loom_amdgpu_spill_lowering_build_full_exec_write(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_location_id_t location) {
  loom_amdgpu_spill_opcode_descriptor_t opcode_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_descriptor_ref(
      rewriter, context->descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_FULL, &opcode_descriptor));
  loom_op_t* write_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      &rewriter->builder, context->descriptor_set, opcode_descriptor.descriptor,
      opcode_descriptor.opcode_id, /*operands=*/NULL, /*operand_count=*/0,
      loom_named_attr_slice_empty(), /*result_types=*/NULL,
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

static iree_status_t loom_amdgpu_spill_lowering_initialize_register_class(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t class_id,
    iree_string_view_t expected_name, uint32_t* out_unit_bytes) {
  *out_unit_bytes = 0;
  if (descriptor_set->reg_class_count <= class_id) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "AMDGPU spill lowering descriptor set has no %.*s register class",
        (int)expected_name.size, expected_name.data);
  }
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[class_id];
  iree_string_view_t class_name = loom_low_descriptor_set_string(
      descriptor_set, reg_class->name_string_offset);
  if (!iree_string_view_equal(class_name, expected_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering expected descriptor register class %" PRIu16
        " to be '%.*s', but found '%.*s'",
        class_id, (int)expected_name.size, expected_name.data,
        (int)class_name.size, class_name.data);
  }
  if (reg_class->alloc_unit_bits != LOOM_AMDGPU_SCRATCH_SPILL_UNIT_BITS) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU spill lowering expects 32-bit %.*s allocation units for "
        "scratch packets, but register class '%.*s' has %" PRIu16 " bits",
        (int)expected_name.size, expected_name.data, (int)class_name.size,
        class_name.data, reg_class->alloc_unit_bits);
  }
  *out_unit_bytes = reg_class->alloc_unit_bits / 8u;
  return iree_ok_status();
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
  loom_amdgpu_spill_descriptor_t spill_descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_resolve_scratch_descriptor_ref(
          rewriter, context->descriptor_set,
          loom_amdgpu_spill_lowering_store_descriptor_ref(chunk_units),
          &spill_descriptor));
  loom_value_id_t operands[2] = {value, LOOM_VALUE_ID_INVALID};
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
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_spill_lowering_resolve_scratch_descriptor_ref(
            rewriter, context->descriptor_set,
            loom_amdgpu_spill_lowering_store_vaddr_descriptor_ref(chunk_units),
            &spill_descriptor));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_validate_offset_immediate(
      &spill_descriptor, offset));
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_make_offset_attr(context, offset, &attr));
  loom_op_t* store_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      &rewriter->builder, context->descriptor_set, spill_descriptor.descriptor,
      spill_descriptor.opcode_id, operands, operand_count,
      loom_make_named_attr_slice(&attr, 1), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &store_op);
}

static iree_status_t loom_amdgpu_spill_lowering_load_chunk(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_value_id_t storage, uint32_t chunk_units,
    loom_type_t result_type, const loom_amdgpu_spill_access_t* access,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_spill_descriptor_t spill_descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_resolve_scratch_descriptor_ref(
          rewriter, context->descriptor_set,
          loom_amdgpu_spill_lowering_load_descriptor_ref(chunk_units),
          &spill_descriptor));
  loom_value_id_t operands[1] = {LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 0;
  int64_t offset = access->segment_offset;
  if (!loom_amdgpu_spill_lowering_offset_fits_immediate(
          spill_descriptor.offset_immediate, offset)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_storage_address(
        context, rewriter, storage, access, result_type, location,
        &operands[0]));
    operand_count = 1;
    offset = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_spill_lowering_resolve_scratch_descriptor_ref(
            rewriter, context->descriptor_set,
            loom_amdgpu_spill_lowering_load_vaddr_descriptor_ref(chunk_units),
            &spill_descriptor));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_validate_offset_immediate(
      &spill_descriptor, offset));
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_make_offset_attr(context, offset, &attr));
  const loom_type_t result_types[] = {result_type};
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, context->descriptor_set, spill_descriptor.descriptor,
      spill_descriptor.opcode_id, operands, operand_count,
      loom_make_named_attr_slice(&attr, 1), result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_register_type(
      context, value_type, &spill_register));
  const uint32_t unit_count = loom_low_register_type_unit_count(value_type);
  if (unit_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering found a zero-unit spill value");
  }

  loom_amdgpu_storage_layout_reference_t storage_reference = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_storage_reference(
      context, loom_low_spill_storage(op), &storage_reference));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_validate_storage_space(&storage_reference));

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
  if (spill_register.kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR) {
    // SGPR values are wave-uniform, but scratch is lane-private and vector
    // packets obey EXEC. Store every lane's private spill slot so a later SGPR
    // reload cannot sample a lane that was inactive at spill time.
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_enter_full_exec(
        context, rewriter, op->location, &saved_exec));
  }
  for (uint32_t chunk_start = 0; chunk_start < unit_count;) {
    const uint32_t chunk_units =
        loom_amdgpu_spill_lowering_register_chunk_units(
            &spill_register, unit_count - chunk_start);
    const uint64_t chunk_byte_offset =
        (uint64_t)chunk_start * spill_register.unit_bytes;
    const uint64_t chunk_byte_length =
        (uint64_t)chunk_units * spill_register.unit_bytes;
    loom_amdgpu_spill_access_t access = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_access(
        &storage_reference, loom_low_spill_offset(op), chunk_byte_offset,
        chunk_byte_length, &access));
    loom_value_id_t chunk_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_slice(
        rewriter, value, unit_count, chunk_start, chunk_units, value_type,
        op->location, &chunk_value));
    loom_type_t scratch_value_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_materialize_store_value(
        context, rewriter, &spill_register, chunk_value, chunk_units,
        op->location, &chunk_value, &scratch_value_type));
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_store_chunk(
        context, rewriter, loom_low_spill_storage(op), chunk_value, chunk_units,
        &access, scratch_value_type, op->location));
    chunk_start += chunk_units;
  }
  if (saved_exec != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_exec_write(
        context, rewriter, saved_exec, op->location));
  }
  return loom_rewriter_erase(rewriter, op);
}

static iree_status_t loom_amdgpu_spill_lowering_rewrite_reload(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_rewriter_t* rewriter, loom_op_t* op) {
  const loom_value_id_t result = loom_low_reload_result(op);
  const loom_type_t result_type =
      loom_module_value_type(context->module, result);
  loom_amdgpu_spill_register_t spill_register = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_register_type(
      context, result_type, &spill_register));
  const uint32_t unit_count = loom_low_register_type_unit_count(result_type);
  if (unit_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering found a zero-unit reload result");
  }

  loom_amdgpu_storage_layout_reference_t storage_reference = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_storage_reference(
      context, loom_low_reload_storage(op), &storage_reference));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_validate_storage_space(&storage_reference));

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
  if (spill_register.kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR) {
    // SGPR reloads move through lane-private scratch and collapse the value
    // with readfirstlane. Load every lane's private slot before sampling one
    // lane so reloads remain correct across divergent EXEC regions.
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_enter_full_exec(
        context, rewriter, op->location, &saved_exec));
  }
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t* loaded_chunks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, unit_count,
                                                 sizeof(*loaded_chunks),
                                                 (void**)&loaded_chunks));
  iree_host_size_t loaded_chunk_count = 0;
  for (uint32_t chunk_start = 0; chunk_start < unit_count;) {
    const uint32_t chunk_units =
        loom_amdgpu_spill_lowering_register_chunk_units(
            &spill_register, unit_count - chunk_start);
    const uint64_t chunk_byte_offset =
        (uint64_t)chunk_start * spill_register.unit_bytes;
    const uint64_t chunk_byte_length =
        (uint64_t)chunk_units * spill_register.unit_bytes;
    loom_amdgpu_spill_access_t access = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_resolve_access(
        &storage_reference, loom_low_reload_offset(op), chunk_byte_offset,
        chunk_byte_length, &access));
    loom_type_t chunk_type = result_type;
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_make_scratch_load_type(
        context, &spill_register, chunk_units, result_type, &chunk_type));
    loom_value_id_t loaded_chunk = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_load_chunk(
        context, rewriter, loom_low_reload_storage(op), chunk_units, chunk_type,
        &access, op->location, &loaded_chunk));
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_materialize_loaded_value(
        context, rewriter, &spill_register, loaded_chunk, chunk_units,
        op->location, &loaded_chunks[loaded_chunk_count++]));
    chunk_start += chunk_units;
  }
  if (saved_exec != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_exec_write(
        context, rewriter, saved_exec, op->location));
  }

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
    loom_region_t* body, loom_op_t** ops, iree_host_size_t op_capacity,
    iree_host_size_t* out_op_count) {
  *out_op_count = 0;
  loom_block_t* block = NULL;
  loom_op_t* op = NULL;
  loom_region_for_each_block(body, block) {
    loom_block_for_each_op(block, op) {
      if (!loom_low_spill_isa(op) && !loom_low_reload_isa(op)) {
        continue;
      }
      if (ops != NULL && *out_op_count >= op_capacity) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU spill lowering op count changed while collecting ops");
      }
      if (ops != NULL) {
        ops[*out_op_count] = op;
      }
      ++(*out_op_count);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_initialize_context(
    loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_spill_lowering_context_t* out_context,
    iree_arena_allocator_t* scratch_arena) {
  *out_context = (loom_amdgpu_spill_lowering_context_t){
      .module = module,
      .descriptor_set = descriptor_set,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_initialize_register_class(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, IREE_SV("amdgpu.sgpr"),
      &out_context->sgpr_unit_bytes));
  out_context->sgpr_class_id = LOOM_AMDGPU_REG_CLASS_ID_SGPR;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_initialize_register_class(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, IREE_SV("amdgpu.vgpr"),
      &out_context->vgpr_unit_bytes));
  out_context->vgpr_class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("offset"),
                                                 &out_context->offset_attr_id));
  return loom_amdgpu_storage_layout_build(module, function_op, scratch_arena,
                                          &out_context->storage_layout);
}

iree_status_t loom_amdgpu_lower_spill_traffic(
    loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* scratch_arena) {
  if (module == NULL || function_op == NULL || descriptor_set == NULL ||
      scratch_arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering requires module, function, descriptor set, and "
        "scratch arena");
  }
  loom_region_t* body = loom_low_function_body(function_op);
  if (body == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering requires a low function body");
  }

  iree_host_size_t op_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_spill_lowering_collect_ops(body, NULL, 0, &op_count));
  if (op_count == 0) {
    return iree_ok_status();
  }

  loom_op_t** ops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, op_count,
                                                 sizeof(*ops), (void**)&ops));
  iree_host_size_t collected_op_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_collect_ops(
      body, ops, op_count, &collected_op_count));
  if (collected_op_count != op_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU spill lowering op count changed while collecting ops");
  }

  loom_amdgpu_spill_lowering_context_t context = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_initialize_context(
      module, function_op, descriptor_set, &context, scratch_arena));

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, scratch_arena));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < op_count && iree_status_is_ok(status); ++i) {
    if (loom_low_spill_isa(ops[i])) {
      status =
          loom_amdgpu_spill_lowering_rewrite_spill(&context, &rewriter, ops[i]);
    } else if (loom_low_reload_isa(ops[i])) {
      status = loom_amdgpu_spill_lowering_rewrite_reload(&context, &rewriter,
                                                         ops[i]);
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
