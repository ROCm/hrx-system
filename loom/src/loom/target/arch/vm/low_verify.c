// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/low_verify.h"

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/packet.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/registers.h"

typedef enum loom_vm_packet_constraint_kind_e {
  LOOM_VM_PACKET_CONSTRAINT_KIND_IMMEDIATE_NONZERO = 1,
  LOOM_VM_PACKET_CONSTRAINT_KIND_MEMORY_FORMAT_UNIT_COUNT = 2,
  LOOM_VM_PACKET_CONSTRAINT_KIND_INTEGER_BITSTREAM_PACK = 3,
  LOOM_VM_PACKET_CONSTRAINT_KIND_INTEGER_BITSTREAM_UNPACK = 4,
  LOOM_VM_PACKET_CONSTRAINT_KIND_PACKED_IMMEDIATE_MASK = 5,
} loom_vm_packet_constraint_kind_t;

typedef struct loom_vm_packet_constraint_range_t {
  // First constraint row for one descriptor ordinal.
  uint8_t start;
  // Number of consecutive constraint rows for the descriptor.
  uint8_t count;
} loom_vm_packet_constraint_range_t;

typedef struct loom_vm_packet_constraint_t {
  // Operation interpreted from |arguments| and |parameter|.
  uint8_t kind;
  // Kind-specific descriptor operand and immediate ordinals.
  uint8_t arguments[7];
  // Kind-specific unsigned parameter.
  uint32_t parameter;
} loom_vm_packet_constraint_t;

typedef struct loom_vm_packed_immediate_mask_t {
  // One validity bit for each possible u8 immediate value.
  uint64_t words[4];
} loom_vm_packed_immediate_mask_t;

#define LOOM_VM_PACKET_CONSTRAINT_DEFINE_LIMITS
#define LOOM_VM_PACKET_CONSTRAINT_LIMITS(descriptor_count, constraint_count, \
                                         packed_mask_count,                  \
                                         memory_format_count)                \
  enum {                                                                     \
    LOOM_VM_PACKET_CONSTRAINT_DESCRIPTOR_COUNT = descriptor_count,           \
    LOOM_VM_PACKET_CONSTRAINT_COUNT = constraint_count,                      \
    LOOM_VM_PACKED_IMMEDIATE_MASK_COUNT = packed_mask_count,                 \
    LOOM_VM_MEMORY_FORMAT_COUNT = memory_format_count,                       \
  };
#include "loom/target/arch/vm/verification_rows.inl"
#undef LOOM_VM_PACKET_CONSTRAINT_LIMITS
#undef LOOM_VM_PACKET_CONSTRAINT_DEFINE_LIMITS

static const uint8_t kLoomVmMemoryFormatUnitCounts[] = {
#define LOOM_VM_PACKET_CONSTRAINT_DEFINE_MEMORY_FORMAT_ROWS
#define LOOM_VM_MEMORY_FORMAT_UNIT_COUNT_ROW(unit_count) unit_count,
#include "loom/target/arch/vm/verification_rows.inl"
#undef LOOM_VM_MEMORY_FORMAT_UNIT_COUNT_ROW
#undef LOOM_VM_PACKET_CONSTRAINT_DEFINE_MEMORY_FORMAT_ROWS
};

static const loom_vm_packed_immediate_mask_t kLoomVmPackedImmediateMasks[] = {
#define LOOM_VM_PACKET_CONSTRAINT_DEFINE_PACKED_MASK_ROWS
#define LOOM_VM_PACKED_IMMEDIATE_MASK_ROW(word0, word1, word2, word3) \
  {{word0, word1, word2, word3}},
#include "loom/target/arch/vm/verification_rows.inl"
#undef LOOM_VM_PACKED_IMMEDIATE_MASK_ROW
#undef LOOM_VM_PACKET_CONSTRAINT_DEFINE_PACKED_MASK_ROWS
};

static const loom_vm_packet_constraint_range_t kLoomVmPacketConstraintRanges[] =
    {
#define LOOM_VM_PACKET_CONSTRAINT_DEFINE_RANGE_ROWS
#define LOOM_VM_PACKET_CONSTRAINT_RANGE_ROW(constraint_start, \
                                            constraint_count) \
  {constraint_start, constraint_count},
#include "loom/target/arch/vm/verification_rows.inl"
#undef LOOM_VM_PACKET_CONSTRAINT_RANGE_ROW
#undef LOOM_VM_PACKET_CONSTRAINT_DEFINE_RANGE_ROWS
};

static const loom_vm_packet_constraint_t kLoomVmPacketConstraints[] = {
#define LOOM_VM_PACKET_CONSTRAINT_DEFINE_ROWS
#define LOOM_VM_PACKET_CONSTRAINT_ROW(kind, parameter, argument0, argument1, \
                                      argument2, argument3, argument4,       \
                                      argument5, argument6)                  \
  {LOOM_VM_PACKET_CONSTRAINT_KIND_##kind,                                    \
   {argument0, argument1, argument2, argument3, argument4, argument5,        \
    argument6},                                                              \
   parameter},
#include "loom/target/arch/vm/verification_rows.inl"
#undef LOOM_VM_PACKET_CONSTRAINT_ROW
#undef LOOM_VM_PACKET_CONSTRAINT_DEFINE_ROWS
};

static_assert(sizeof(loom_vm_packet_constraint_range_t) == 2,
              "VM packet constraint ranges must remain compact");
static_assert(sizeof(loom_vm_packet_constraint_t) == 12,
              "VM packet constraints must remain compact");
static_assert(IREE_ARRAYSIZE(kLoomVmMemoryFormatUnitCounts) ==
                  LOOM_VM_MEMORY_FORMAT_COUNT,
              "VM memory-format table count drifted");
static_assert(IREE_ARRAYSIZE(kLoomVmPackedImmediateMasks) ==
                  LOOM_VM_PACKED_IMMEDIATE_MASK_COUNT,
              "VM packed-immediate mask count drifted");
static_assert(IREE_ARRAYSIZE(kLoomVmPacketConstraintRanges) ==
                  LOOM_VM_PACKET_CONSTRAINT_DESCRIPTOR_COUNT,
              "VM packet constraint range count drifted");
static_assert(IREE_ARRAYSIZE(kLoomVmPacketConstraints) ==
                  LOOM_VM_PACKET_CONSTRAINT_COUNT,
              "VM packet constraint count drifted");

static const loom_named_attr_t* loom_vm_low_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id >= module->strings.count) continue;
    if (iree_string_view_equal(module->strings.entries[attr->name_id], name)) {
      return attr;
    }
  }
  return NULL;
}

static bool loom_vm_low_try_resolve_enum_token(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_immediate_t* immediate, loom_string_id_t token_id,
    uint64_t* out_value) {
  if (token_id >= module->strings.count) return false;
  const iree_string_view_t token = module->strings.entries[token_id];
  int64_t value = 0;
  if (!loom_low_descriptor_set_lookup_enum_value_by_token(
          descriptor_set, immediate->enum_domain_id, token, &value)) {
    return false;
  }
  *out_value = (uint64_t)value;
  return true;
}

static bool loom_vm_low_try_immediate_value(
    const loom_low_verify_context_t* context,
    const loom_low_descriptor_packet_t* packet, uint8_t immediate_ordinal,
    uint64_t* out_value) {
  const loom_low_resolved_target_t* target =
      loom_low_verify_context_target(context);
  const loom_low_descriptor_set_t* descriptor_set = target->descriptor_set;
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  if (immediate_ordinal >= descriptor->immediate_count) return false;
  const loom_low_immediate_t* immediate =
      &descriptor_set
           ->immediates[descriptor->immediate_start + immediate_ordinal];
  const iree_string_view_t name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_offset);
  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  if (!loom_low_packet_try_op_attrs(packet->op, &attrs, NULL)) return false;
  const loom_module_t* module = loom_low_verify_context_module(context);
  const loom_named_attr_t* attr = loom_vm_low_find_attr(module, attrs, name);
  if (attr == NULL) {
    if (!iree_any_bit_set(immediate->flags,
                          LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
      return false;
    }
    *out_value = (uint64_t)immediate->default_value;
    return true;
  }
  if (attr->value.kind == LOOM_ATTR_I64) {
    *out_value = (uint64_t)attr->value.i64;
    return true;
  }
  if (attr->value.kind == LOOM_ATTR_STRING &&
      immediate->kind == LOOM_LOW_IMMEDIATE_KIND_ENUM) {
    return loom_vm_low_try_resolve_enum_token(module, descriptor_set, immediate,
                                              attr->value.string_id, out_value);
  }
  return false;
}

static bool loom_vm_low_try_operand_unit_count(
    const loom_low_verify_context_t* context,
    const loom_low_descriptor_packet_t* packet, uint8_t operand_ordinal,
    uint32_t* out_unit_count) {
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  if (operand_ordinal >= descriptor->operand_count) return false;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_verify_context_target(context)->descriptor_set;
  const loom_low_operand_t* operand =
      &descriptor_set->operands[descriptor->operand_start + operand_ordinal];
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  if (operand_ordinal < descriptor->result_count) {
    if (operand->source_value_index >= packet->op->result_count) return false;
    value_id = loom_op_const_results(packet->op)[operand->source_value_index];
  } else {
    if (operand->role == LOOM_LOW_OPERAND_ROLE_IMPLICIT ||
        operand->source_value_index >= packet->op->operand_count) {
      return false;
    }
    value_id = loom_op_const_operands(packet->op)[operand->source_value_index];
  }
  const loom_module_t* module = loom_low_verify_context_module(context);
  if (value_id >= module->values.count) return false;
  const loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_low_type_is_register(type)) return false;
  *out_unit_count = loom_low_register_type_unit_count(type);
  return true;
}

static bool loom_vm_low_evaluate_immediate_nonzero(
    const loom_low_verify_context_t* context,
    const loom_low_descriptor_packet_t* packet,
    const loom_vm_packet_constraint_t* constraint, bool* out_satisfied) {
  uint64_t value = 0;
  if (!loom_vm_low_try_immediate_value(context, packet,
                                       constraint->arguments[0], &value)) {
    return false;
  }
  *out_satisfied = value != 0;
  return true;
}

static bool loom_vm_low_evaluate_memory_format_unit_count(
    const loom_low_verify_context_t* context,
    const loom_low_descriptor_packet_t* packet,
    const loom_vm_packet_constraint_t* constraint, bool* out_satisfied) {
  uint32_t unit_count = 0;
  uint64_t format = 0;
  if (!loom_vm_low_try_operand_unit_count(
          context, packet, constraint->arguments[0], &unit_count) ||
      !loom_vm_low_try_immediate_value(context, packet,
                                       constraint->arguments[1], &format) ||
      format >= IREE_ARRAYSIZE(kLoomVmMemoryFormatUnitCounts)) {
    return false;
  }
  const uint8_t expected_unit_count = kLoomVmMemoryFormatUnitCounts[format];
  *out_satisfied = expected_unit_count <= constraint->parameter &&
                   unit_count == expected_unit_count;
  return true;
}

static bool loom_vm_low_evaluate_integer_bitstream(
    const loom_low_verify_context_t* context,
    const loom_low_descriptor_packet_t* packet,
    const loom_vm_packet_constraint_t* constraint, bool is_pack,
    bool* out_satisfied) {
  uint32_t result_unit_count = 0;
  uint32_t source_unit_count = 0;
  uint64_t field_width = 0;
  uint64_t source_count = 0;
  uint64_t result_count = 0;
  uint64_t source_width = 0;
  uint64_t result_width = 0;
  if (!loom_vm_low_try_operand_unit_count(
          context, packet, constraint->arguments[0], &result_unit_count) ||
      !loom_vm_low_try_operand_unit_count(
          context, packet, constraint->arguments[1], &source_unit_count) ||
      !loom_vm_low_try_immediate_value(
          context, packet, constraint->arguments[2], &field_width) ||
      !loom_vm_low_try_immediate_value(
          context, packet, constraint->arguments[3], &source_count) ||
      !loom_vm_low_try_immediate_value(
          context, packet, constraint->arguments[4], &result_count) ||
      !loom_vm_low_try_immediate_value(
          context, packet, constraint->arguments[5], &source_width) ||
      !loom_vm_low_try_immediate_value(
          context, packet, constraint->arguments[6], &result_width)) {
    return false;
  }
  const uint64_t source_bit_count =
      source_count * (is_pack ? field_width : source_width);
  const uint64_t result_bit_count =
      result_count * (is_pack ? result_width : field_width);
  const uint64_t field_carrier_width = is_pack ? source_width : result_width;
  *out_satisfied = result_unit_count == result_count &&
                   source_unit_count == source_count && field_width != 0 &&
                   field_width <= field_carrier_width &&
                   source_bit_count == result_bit_count &&
                   source_bit_count <= constraint->parameter;
  return true;
}

static bool loom_vm_low_evaluate_packed_immediate_mask(
    const loom_low_verify_context_t* context,
    const loom_low_descriptor_packet_t* packet,
    const loom_vm_packet_constraint_t* constraint, bool* out_satisfied) {
  uint64_t value = 0;
  const uint8_t mask_ordinal = constraint->arguments[1];
  if (!loom_vm_low_try_immediate_value(context, packet,
                                       constraint->arguments[0], &value) ||
      value > UINT8_MAX ||
      mask_ordinal >= IREE_ARRAYSIZE(kLoomVmPackedImmediateMasks)) {
    return false;
  }
  const loom_vm_packed_immediate_mask_t* mask =
      &kLoomVmPackedImmediateMasks[mask_ordinal];
  *out_satisfied =
      (mask->words[value / 64] & (UINT64_C(1) << (value % 64))) != 0;
  return true;
}

static bool loom_vm_low_evaluate_constraint(
    const loom_low_verify_context_t* context,
    const loom_low_descriptor_packet_t* packet,
    const loom_vm_packet_constraint_t* constraint, bool* out_satisfied) {
  *out_satisfied = true;
  switch (constraint->kind) {
    case LOOM_VM_PACKET_CONSTRAINT_KIND_IMMEDIATE_NONZERO:
      return loom_vm_low_evaluate_immediate_nonzero(context, packet, constraint,
                                                    out_satisfied);
    case LOOM_VM_PACKET_CONSTRAINT_KIND_MEMORY_FORMAT_UNIT_COUNT:
      return loom_vm_low_evaluate_memory_format_unit_count(
          context, packet, constraint, out_satisfied);
    case LOOM_VM_PACKET_CONSTRAINT_KIND_INTEGER_BITSTREAM_PACK:
      return loom_vm_low_evaluate_integer_bitstream(
          context, packet, constraint, /*is_pack=*/true, out_satisfied);
    case LOOM_VM_PACKET_CONSTRAINT_KIND_INTEGER_BITSTREAM_UNPACK:
      return loom_vm_low_evaluate_integer_bitstream(
          context, packet, constraint, /*is_pack=*/false, out_satisfied);
    case LOOM_VM_PACKET_CONSTRAINT_KIND_PACKED_IMMEDIATE_MASK:
      return loom_vm_low_evaluate_packed_immediate_mask(
          context, packet, constraint, out_satisfied);
    default:
      return false;
  }
}

static iree_string_view_t loom_vm_low_constraint_name(uint8_t kind) {
  switch (kind) {
    case LOOM_VM_PACKET_CONSTRAINT_KIND_IMMEDIATE_NONZERO:
      return IREE_SV("immediate_nonzero");
    case LOOM_VM_PACKET_CONSTRAINT_KIND_MEMORY_FORMAT_UNIT_COUNT:
      return IREE_SV("memory_format_unit_count");
    case LOOM_VM_PACKET_CONSTRAINT_KIND_INTEGER_BITSTREAM_PACK:
    case LOOM_VM_PACKET_CONSTRAINT_KIND_INTEGER_BITSTREAM_UNPACK:
      return IREE_SV("integer_bitstream_shape");
    case LOOM_VM_PACKET_CONSTRAINT_KIND_PACKED_IMMEDIATE_MASK:
      return IREE_SV("packed_selector");
    default:
      return IREE_SV("vm_packet");
  }
}

static iree_status_t loom_vm_low_emit_constraint_error(
    loom_low_verify_context_t* context,
    const loom_low_descriptor_packet_t* packet,
    const loom_vm_packet_constraint_t* constraint) {
  const loom_low_resolved_target_t* target =
      loom_low_verify_context_target(context);
  const loom_module_t* module = loom_low_verify_context_module(context);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(target)),
      loom_param_string(loom_low_diagnostic_export_name(target)),
      loom_param_string(loom_low_diagnostic_config_key(target)),
      loom_param_string(loom_low_diagnostic_function_name(
          module, loom_low_verify_context_function_op(context))),
      loom_param_string(loom_op_name(module, packet->op)),
      loom_param_string(loom_vm_low_constraint_name(constraint->kind)),
  };
  return loom_low_verify_context_emit(context, packet->op, LOOM_ERR_TARGET_032,
                                      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vm_low_verify_op(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void* provider_state,
    const loom_low_descriptor_packet_t* packet) {
  (void)provider;
  (void)provider_state;
  if (loom_low_verify_context_should_stop(context) ||
      packet->descriptor == NULL ||
      loom_low_verify_context_target(context)->descriptor_set !=
          loom_vm_core_descriptor_set() ||
      packet->descriptor_ordinal >=
          IREE_ARRAYSIZE(kLoomVmPacketConstraintRanges)) {
    return iree_ok_status();
  }
  const loom_vm_packet_constraint_range_t range =
      kLoomVmPacketConstraintRanges[packet->descriptor_ordinal];
  for (uint8_t i = 0; i < range.count; ++i) {
    const loom_vm_packet_constraint_t* constraint =
        &kLoomVmPacketConstraints[range.start + i];
    bool satisfied = true;
    const bool could_evaluate = loom_vm_low_evaluate_constraint(
        context, packet, constraint, &satisfied);
    // Generic Low verification runs before target providers and owns malformed
    // immediates, register types, and packet shape. Only add the relational
    // target diagnostic when those inputs were available and well-formed.
    if (!could_evaluate || satisfied) {
      continue;
    }
    return loom_vm_low_emit_constraint_error(context, packet, constraint);
  }
  return iree_ok_status();
}

const loom_low_verify_provider_t loom_vm_low_verify_provider = {
    .name = IREE_SVL("vm"),
    .verify_op = loom_vm_low_verify_op,
};
