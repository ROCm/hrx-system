// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower/realization.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cache.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/buffer_descriptor.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/memory_coherence.h"
#include "loom/target/arch/amdgpu/lower/memory_ordering.h"
#include "loom/target/arch/amdgpu/lower/system_memory.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/value/integer64.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static iree_status_t loom_amdgpu_emit_memory_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_packet_plan_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_op_t** out_op) {
  IREE_ASSERT(packet->access.descriptor != NULL);
  *out_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      loom_low_lower_context_builder(context),
      loom_low_lower_context_descriptor_set(context), packet->access.descriptor,
      packet->access.source.access_flags, operands, operand_count, attrs,
      result_types, result_count, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, out_op));
  return loom_low_lower_record_memory_packet(
      context, *out_op, packet->access.descriptor, &packet->access.source,
      loom_value_facts_exact_i64(0));
}

static iree_status_t loom_amdgpu_memory_payload_low_type(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_t* access, loom_type_t* out_type) {
  if (access->payload_register_class ==
      LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_SGPR) {
    if (access->payload_register_count == 1) {
      return loom_amdgpu_make_sgpr_type(context, out_type);
    }
    return loom_amdgpu_make_sgpr_range_type(
        context, access->payload_register_count, out_type);
  }
  if (access->payload_register_count == 1) {
    return loom_amdgpu_make_vgpr_type(context, out_type);
  }
  return loom_amdgpu_make_vgpr_range_type(
      context, access->payload_register_count, out_type);
}

static iree_status_t loom_amdgpu_emit_constant_store_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_packet_plan_t* packet,
    loom_value_id_t* out_low_value) {
  const uint32_t payload_register_count = packet->access.payload_register_count;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t low_lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < payload_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        packet->constant_words[i], vgpr_type, &low_lanes[i]));
  }
  if (payload_register_count == 1) {
    *out_low_value = low_lanes[0];
    return iree_ok_status();
  }

  loom_type_t vgpr_range_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
      context, payload_register_count, &vgpr_range_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(context), low_lanes,
                            payload_register_count, vgpr_range_type,
                            source_op->location, &concat_op));
  *out_low_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_ensure_memory_store_payload_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_packet_plan_t* packet, loom_value_id_t low_value,
    loom_value_id_t* out_low_value) {
  const loom_amdgpu_memory_access_t* access = &packet->access;
  *out_low_value = low_value;
  if (access->payload_register_class !=
      LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_VGPR) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr) {
    if (access->payload_register_count == 1 && access->packet_byte_count == 2) {
      return loom_amdgpu_materialize_full_low_vgpr_b32(
          context, source_op, low_value, out_low_value);
    }
    return iree_ok_status();
  }
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (is_sgpr && loom_low_register_type_unit_count(low_type) ==
                     access->payload_register_count) {
    if (packet->constant_words != NULL) {
      return loom_amdgpu_emit_constant_store_payload(context, source_op, packet,
                                                     out_low_value);
    }
    return loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_value, out_low_value);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_memory_buffer_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_binding, const loom_amdgpu_memory_access_t* access,
    loom_value_id_t* out_descriptor) {
  if (access->realization.descriptor) {
    *out_descriptor =
        loom_low_lower_realization_value(access->realization.descriptor);
    return iree_ok_status();
  }
  return loom_amdgpu_emit_hal_buffer_descriptor(
      context, source_op, low_binding, &access->source, access->buffer_extent,
      out_descriptor);
}

static iree_status_t loom_amdgpu_emit_memory_buffer_soffset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access,
    const loom_amdgpu_memory_dynamic_term_sequence_t* sequence,
    loom_value_id_t* out_soffset) {
  if (access->realization.soffset) {
    *out_soffset =
        loom_low_lower_realization_value(access->realization.soffset);
    return iree_ok_status();
  }
  return loom_amdgpu_emit_sgpr_byte_offset_terms(
      context, source_op, sequence, access->scalar_byte_offset, out_soffset);
}

static bool loom_amdgpu_memory_packet_operand_matches_field(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand, iree_string_view_t field_name) {
  if (!loom_low_operand_role_is_packet_operand(operand->role)) {
    return false;
  }
  const iree_string_view_t operand_field_name = loom_low_descriptor_set_string(
      descriptor_set, operand->field_name_string_ref);
  return iree_string_view_equal(operand_field_name, field_name);
}

static uint32_t loom_amdgpu_memory_packet_addr_operand_unit_count(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_packet_plan_t* packet) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor = packet->access.descriptor;
  for (uint16_t descriptor_operand_index = descriptor->result_count;
       descriptor_operand_index < descriptor->operand_count;
       ++descriptor_operand_index) {
    const loom_low_operand_t* operand = loom_low_descriptor_operand(
        descriptor_set, descriptor, descriptor_operand_index);
    if (loom_amdgpu_memory_packet_operand_matches_field(descriptor_set, operand,
                                                        IREE_SV("addr"))) {
      return operand->unit_count;
    }
  }
  IREE_ASSERT_UNREACHABLE(
      "AMDGPU flat memory packet selected without an addr operand");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_fit_memory_flat_vaddr_to_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_packet_plan_t* packet, loom_value_id_t low_vaddr,
    loom_value_id_t* out_low_vaddr) {
  *out_low_vaddr = low_vaddr;
  const uint32_t required_unit_count =
      loom_amdgpu_memory_packet_addr_operand_unit_count(context, packet);

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_vaddr_type = loom_module_value_type(module, low_vaddr);
  IREE_ASSERT(loom_low_type_is_register(low_vaddr_type));
  const uint32_t actual_unit_count =
      loom_low_register_type_unit_count(low_vaddr_type);
  if (actual_unit_count == required_unit_count) {
    return iree_ok_status();
  }

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  if (actual_unit_count == 2 && required_unit_count == 1) {
    return loom_amdgpu_emit_low_slice(context, source_op, low_vaddr,
                                      /*offset=*/0, vgpr_type, out_low_vaddr);
  }
  if (actual_unit_count == 1 && required_unit_count == 2) {
    return loom_amdgpu_emit_vgpr64_from_u32(context, source_op, low_vaddr,
                                            out_low_vaddr);
  }

  IREE_ASSERT_UNREACHABLE(
      "AMDGPU flat memory packet selected unsupported vaddr width");
  IREE_BUILTIN_UNREACHABLE();
}

typedef struct loom_amdgpu_memory_cache_attr_field_t {
  // Presence bit required for this descriptor attribute.
  loom_amdgpu_memory_cache_policy_attr_flags_t flag;
  // Descriptor attribute name.
  iree_string_view_t name;
  // Byte offset to the encoded attribute value field.
  iree_host_size_t value_offset;
} loom_amdgpu_memory_cache_attr_field_t;

static const loom_amdgpu_memory_cache_attr_field_t
    kLoomAmdgpuMemoryCacheAttrFields[] = {
        {
            .flag = LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_SCOPE,
            .name = {.data = "scope", .size = 5},
            .value_offset =
                offsetof(loom_amdgpu_memory_cache_policy_attrs_t, scope),
        },
        {
            .flag = LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_TH,
            .name = {.data = "th", .size = 2},
            .value_offset =
                offsetof(loom_amdgpu_memory_cache_policy_attrs_t, th),
        },
        {
            .flag = LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_NT,
            .name = {.data = "nt", .size = 2},
            .value_offset =
                offsetof(loom_amdgpu_memory_cache_policy_attrs_t, nt),
        },
};

static int64_t loom_amdgpu_memory_cache_attr_field_value(
    const loom_amdgpu_memory_cache_policy_attrs_t* cache_attrs,
    const loom_amdgpu_memory_cache_attr_field_t* field) {
  const uint8_t* attrs_bytes = (const uint8_t*)cache_attrs;
  const void* value_bytes = attrs_bytes + field->value_offset;
  return *(const int64_t*)value_bytes;
}

static iree_status_t loom_amdgpu_append_memory_cache_attrs(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_t* access, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  // LDS observations participate in ordering without a global cache policy.
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }
  if (access->source.operation_kind ==
          LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_STORE &&
      access->source.atomic.scope > LOOM_ATOMIC_SCOPE_SUBGROUP) {
    return loom_amdgpu_system_memory_append_release_store_attrs_scoped(
        loom_low_lower_context_builder(context),
        loom_low_lower_context_descriptor_set(context),
        loom_amdgpu_memory_coherence_scope(
            loom_low_lower_context_descriptor_set(context),
            access->source.atomic.scope),
        attrs, attr_capacity, inout_attr_count);
  }
  uint8_t read_scope = access->source.read_visibility_scope;
  if (access->source.operation_kind ==
      LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_LOAD) {
    read_scope = iree_max(read_scope, access->source.atomic.scope);
  }
  if (read_scope > LOOM_ATOMIC_SCOPE_SUBGROUP) {
    // The retained visibility obligation and atomic observation scope both
    // constrain these reads. Advisory cache preferences cannot weaken them.
    return loom_amdgpu_system_memory_append_load_attrs_scoped(
        loom_low_lower_context_builder(context),
        loom_low_lower_context_descriptor_set(context),
        loom_amdgpu_memory_coherence_scope(
            loom_low_lower_context_descriptor_set(context), read_scope),
        attrs, attr_capacity, inout_attr_count);
  }
  const loom_vector_memory_cache_policy_t* policy =
      &access->source.cache_policy;
  if (!loom_amdgpu_memory_cache_policy_is_present(policy)) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  loom_amdgpu_memory_cache_policy_attrs_t cache_attrs = {0};
  const loom_amdgpu_memory_cache_policy_resolution_t resolution =
      loom_amdgpu_memory_cache_policy_resolve(descriptor_set, access,
                                              &cache_attrs);
  IREE_ASSERT_NE(resolution,
                 LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_REJECTED);
  if (resolution != LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_ENCODED) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuMemoryCacheAttrFields); ++i) {
    const loom_amdgpu_memory_cache_attr_field_t* field =
        &kLoomAmdgpuMemoryCacheAttrFields[i];
    if (!iree_any_bit_set(cache_attrs.flags, field->flag)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, field->name,
        loom_amdgpu_memory_cache_attr_field_value(&cache_attrs, field), attrs,
        attr_capacity, inout_attr_count));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_make_memory_cache_attrs(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_t* access, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* out_attr_count) {
  *out_attr_count = 0;
  return loom_amdgpu_append_memory_cache_attrs(context, access, attrs,
                                               attr_capacity, out_attr_count);
}

iree_status_t loom_amdgpu_make_memory_attrs(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_t* access, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* out_attr_count) {
  *out_attr_count = 0;
  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("offset0"), access->immediate_offset, attrs,
        attr_capacity, out_attr_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("offset1"), access->secondary_immediate_offset, attrs,
        attr_capacity, out_attr_count));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("offset"), access->immediate_offset, attrs,
        attr_capacity, out_attr_count));
  }
  return loom_amdgpu_append_memory_cache_attrs(context, access, attrs,
                                               attr_capacity, out_attr_count);
}

static loom_value_id_t loom_amdgpu_memory_load_result(
    const loom_op_t* source_op) {
  IREE_ASSERT_EQ(source_op->result_count, 1u);
  return loom_op_const_results(source_op)[0];
}

static iree_status_t loom_amdgpu_materialize_memory_load_packet_for_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool result_is_vgpr, loom_value_id_t low_packet,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = low_packet;
  if (!result_is_vgpr) {
    return iree_ok_status();
  }
  return loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_packet, out_low_packet);
}

static bool loom_amdgpu_memory_load_packet_needs_signed_i16_repair(
    const loom_amdgpu_memory_access_t* access) {
  return access->source.memory_space ==
             LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
         access->payload_format ==
             LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_SIGNED_16BIT_INTEGER;
}

static iree_status_t loom_amdgpu_emit_signed_i16_memory_load_repair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_packet, loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("offset"), 0, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("width"), 16, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));

  const loom_value_id_t operands[] = {low_packet};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_I32_OFFSET_WIDTH_INLINE, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      &lane_type, 1, &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_repair_memory_load_packet_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access, loom_value_id_t low_packet,
    loom_value_id_t* out_low_result) {
  if (!loom_amdgpu_memory_load_packet_needs_signed_i16_repair(access)) {
    *out_low_result = low_packet;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_signed_i16_memory_load_repair(
      context, source_op, low_packet, out_low_result);
}

static loom_value_id_t loom_amdgpu_memory_store_value(
    const loom_module_t* module, const loom_op_t* source_op) {
  const loom_memory_access_t access =
      loom_memory_access_cast(module, source_op);
  IREE_ASSERT(loom_memory_access_isa(access));
  const loom_value_id_t value = loom_memory_access_value(access);
  IREE_ASSERT_NE(value, LOOM_VALUE_ID_INVALID);
  return value;
}

static iree_status_t loom_amdgpu_bind_memory_load_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t result_type, loom_value_id_t low_result) {
  const loom_value_id_t source_result =
      loom_amdgpu_memory_load_result(source_op);
  const bool result_is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_memory_load_packet_for_result(
      context, source_op, result_is_vgpr, low_result, &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static bool loom_amdgpu_memory_access_needs_hal_resource(
    const loom_amdgpu_memory_access_t* access) {
  return access->source.memory_space !=
             LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
         access->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE;
}

static iree_status_t loom_amdgpu_lower_memory_packet_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_packet_plan_t* packet,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  const loom_amdgpu_memory_access_t* access = &packet->access;
  loom_amdgpu_memory_dynamic_term_sequence_t sequence = {0};
  loom_amdgpu_memory_access_resolve_dynamic_terms(context, access, &sequence);
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_memory_access_needs_hal_resource(access)) {
    low_resource = loom_low_lower_lookup_value(
        context,
        loom_low_source_memory_access_base_view_value_id(&access->source));
  }

  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  if (access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_vaddr(
          context, source_op, access, &sequence, low_resource, &low_vaddr));
      IREE_RETURN_IF_ERROR(loom_amdgpu_fit_memory_flat_vaddr_to_packet(
          context, source_op, packet, low_vaddr, &low_vaddr));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_memory_vaddr(context, source_op, access, &sequence,
                                        LOOM_VALUE_ID_INVALID, &low_vaddr));
    }
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_memory_payload_low_type(context, access, &result_type));

  loom_named_attr_t attrs[5] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_memory_attrs(
      context, access, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
      const loom_low_lower_resolved_descriptor_t packet_descriptor = {
          .descriptor = access->descriptor,
      };
      loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_m0_u32(
          context, source_op, &packet_descriptor, 0, &low_m0));
      loom_value_id_t operands[] = {low_m0};
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
          context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
          &low_op));
      const loom_value_id_t raw_result =
          loom_value_slice_get(loom_low_op_results(low_op), 0);
      return loom_amdgpu_repair_memory_load_packet_result(
          context, source_op, access, raw_result, out_low_result);
    }
    loom_value_id_t operands[] = {low_vaddr};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
        &low_op));
    const loom_value_id_t raw_result =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    return loom_amdgpu_repair_memory_load_packet_result(
        context, source_op, access, raw_result, out_low_result);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    loom_value_id_t low_saddr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
        context, source_op, access, &sequence, low_resource, &low_saddr));
    const loom_value_id_t operands[] = {low_vaddr, low_saddr};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
        &low_op));
    const loom_value_id_t raw_result =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    return loom_amdgpu_repair_memory_load_packet_result(
        context, source_op, access, raw_result, out_low_result);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM) {
    loom_value_id_t low_sbase = low_resource;
    loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
    if (access->scalar_offset_placement ==
        LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_BASE) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
          context, source_op, access, &sequence, low_resource, &low_sbase));
      loom_type_t sgpr_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0,
          sgpr_type, &low_soffset));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_byte_offset_terms(
          context, source_op, &sequence, access->scalar_byte_offset,
          &low_soffset));
    }
    loom_value_id_t operands[] = {
        low_sbase,
        low_soffset,
    };
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
        &low_op));
    const loom_value_id_t raw_result =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    return loom_amdgpu_repair_memory_load_packet_result(
        context, source_op, access, raw_result, out_low_result);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO) {
    loom_value_id_t low_descriptor = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_buffer_descriptor(
        context, source_op, low_resource, access, &low_descriptor));
    loom_value_id_t operands[] = {low_descriptor};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
        &low_op));
    const loom_value_id_t raw_result =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    return loom_amdgpu_repair_memory_load_packet_result(
        context, source_op, access, raw_result, out_low_result);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT ||
      access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_SCRATCH_VADDR) {
    const loom_value_id_t operands[] = {low_vaddr};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
        &low_op));
    const loom_value_id_t raw_result =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    return loom_amdgpu_repair_memory_load_packet_result(
        context, source_op, access, raw_result, out_low_result);
  }

  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_buffer_soffset(
      context, source_op, access, &sequence, &low_soffset));
  loom_value_id_t low_descriptor = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_buffer_descriptor(
      context, source_op, low_resource, access, &low_descriptor));
  loom_value_id_t operands[] = {
      low_descriptor,
      low_vaddr,
      low_soffset,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
      context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &result_type, 1, &low_op));
  const loom_value_id_t raw_result =
      loom_value_slice_get(loom_low_op_results(low_op), 0);
  return loom_amdgpu_repair_memory_load_packet_result(
      context, source_op, access, raw_result, out_low_result);
}

static iree_status_t loom_amdgpu_lower_memory_packet_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_packet_plan_t* packet, loom_value_id_t low_value) {
  const loom_amdgpu_memory_access_t* access = &packet->access;
  loom_amdgpu_memory_dynamic_term_sequence_t sequence = {0};
  loom_amdgpu_memory_access_resolve_dynamic_terms(context, access, &sequence);
  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_memory_store_payload_vgpr(
      context, source_op, packet, low_value, &low_value));
  const uint32_t register_byte_offset = packet->payload_byte_offset % 4u;
  if (register_byte_offset != 0) {
    loom_type_t lane_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        register_byte_offset * 8u, low_value, lane_type, &low_value));
  }
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_memory_access_needs_hal_resource(access)) {
    low_resource = loom_low_lower_lookup_value(
        context,
        loom_low_source_memory_access_base_view_value_id(&access->source));
  }

  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  if (access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_vaddr(
          context, source_op, access, &sequence, low_resource, &low_vaddr));
      IREE_RETURN_IF_ERROR(loom_amdgpu_fit_memory_flat_vaddr_to_packet(
          context, source_op, packet, low_vaddr, &low_vaddr));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_memory_vaddr(context, source_op, access, &sequence,
                                        LOOM_VALUE_ID_INVALID, &low_vaddr));
    }
  }

  loom_named_attr_t attrs[5] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_memory_attrs(
      context, access, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
      const loom_low_lower_resolved_descriptor_t packet_descriptor = {
          .descriptor = access->descriptor,
      };
      loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_m0_u32(
          context, source_op, &packet_descriptor, 0, &low_m0));
      loom_value_id_t operands[] = {
          low_value,
          low_m0,
      };
      loom_op_t* low_op = NULL;
      return loom_amdgpu_emit_memory_packet(
          context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
          /*result_count=*/0, &low_op);
    }
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR) {
      loom_type_t lane_type = loom_type_none();
      const uint32_t lane_register_count = access->payload_register_count / 2u;
      if (lane_register_count == 1u) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
            context, lane_register_count, &lane_type));
      }
      loom_value_id_t low_value0 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_value, 0, lane_type, &low_value0));
      loom_value_id_t low_value1 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_value, lane_register_count, lane_type,
          &low_value1));
      loom_value_id_t operands[] = {
          low_vaddr,
          low_value0,
          low_value1,
      };
      loom_op_t* low_op = NULL;
      return loom_amdgpu_emit_memory_packet(
          context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
          /*result_count=*/0, &low_op);
    }
    loom_value_id_t operands[] = {
        low_vaddr,
        low_value,
    };
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
        /*result_count=*/0, &low_op);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO) {
    loom_value_id_t low_descriptor = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_buffer_descriptor(
        context, source_op, low_resource, access, &low_descriptor));
    loom_value_id_t operands[] = {
        low_value,
        low_descriptor,
    };
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
        /*result_count=*/0, &low_op);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    loom_value_id_t low_saddr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
        context, source_op, access, &sequence, low_resource, &low_saddr));
    const loom_value_id_t operands[] = {low_vaddr, low_value, low_saddr};
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
        /*result_count=*/0, &low_op);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT ||
      access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_SCRATCH_VADDR) {
    const loom_value_id_t operands[] = {low_vaddr, low_value};
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
        /*result_count=*/0, &low_op);
  }

  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_buffer_soffset(
      context, source_op, access, &sequence, &low_soffset));
  loom_value_id_t low_descriptor = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_buffer_descriptor(
      context, source_op, low_resource, access, &low_descriptor));
  loom_value_id_t operands[] = {
      low_value,
      low_descriptor,
      low_vaddr,
      low_soffset,
  };
  loom_op_t* low_op = NULL;
  return loom_amdgpu_emit_memory_packet(
      context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
      /*result_count=*/0, &low_op);
}

iree_status_t loom_amdgpu_lower_memory_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* plan) {
  IREE_ASSERT_GT(plan->packet_count, 0);
  const loom_type_t result_type = loom_low_lower_value_binding_type(
      context, loom_amdgpu_memory_load_result(source_op));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_ordering_prefix(
      context, source_op, &plan->packets[0].access.source));
  if (plan->packet_count == 1) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lower_memory_packet_load(
        context, source_op, &plan->packets[0], &low_result));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_ordering_suffix(
        context, source_op, &plan->packets[0].access.source));
    return loom_amdgpu_bind_memory_load_result(context, source_op, result_type,
                                               low_result);
  }

  loom_value_id_t low_results[LOOM_AMDGPU_MAX_MEMORY_PACKET_COUNT];
  uint32_t low_result_count = 0;
  const bool result_is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  for (uint32_t i = 0; i < plan->packet_count; ++i) {
    const loom_amdgpu_memory_packet_plan_t* packet = &plan->packets[i];
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lower_memory_packet_load(
        context, source_op, packet, &low_result));
    const uint32_t register_byte_offset = packet->payload_byte_offset % 4u;
    if (register_byte_offset == 0) {
      low_results[low_result_count++] = low_result;
    } else {
      // The final byte completes a zero-extended halfword in the same VGPR.
      // Assemble before bank materialization so uniform results cross once.
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
          register_byte_offset * 8u, low_result, lane_type, &low_result));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
          low_results[low_result_count - 1u], low_result, lane_type,
          &low_results[low_result_count - 1u]));
    }
    // Materialize complete words immediately, retaining the existing ordering
    // for full-register packets. A following partial packet finishes this word.
    if (i + 1u == plan->packet_count ||
        plan->packets[i + 1u].payload_byte_offset % 4u == 0) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_materialize_memory_load_packet_for_result(
              context, source_op, result_is_vgpr,
              low_results[low_result_count - 1u],
              &low_results[low_result_count - 1u]));
    }
  }
  if (low_result_count == 1) {
    return loom_amdgpu_bind_memory_load_result(context, source_op, result_type,
                                               low_results[0]);
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_results, low_result_count,
      result_type, source_op->location, &concat_op));
  return loom_amdgpu_bind_memory_load_result(context, source_op, result_type,
                                             loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_lower_memory_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* plan) {
  IREE_ASSERT_GT(plan->packet_count, 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_ordering_prefix(
      context, source_op, &plan->packets[0].access.source));
  const loom_value_id_t source_value = loom_amdgpu_memory_store_value(
      loom_low_lower_context_module(context), source_op);
  loom_value_id_t low_value =
      loom_low_lower_lookup_value(context, source_value);
  if (plan->packet_count == 1) {
    return loom_amdgpu_lower_memory_packet_store(context, source_op,
                                                 &plan->packets[0], low_value);
  }

  for (uint32_t i = 0; i < plan->packet_count; ++i) {
    const loom_amdgpu_memory_packet_plan_t* packet = &plan->packets[i];
    loom_type_t packet_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_memory_payload_low_type(
        context, &packet->access, &packet_type));
    loom_value_id_t packet_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_value, packet->payload_byte_offset / 4u,
        packet_type, &packet_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_lower_memory_packet_store(
        context, source_op, packet, packet_value));
  }
  return iree_ok_status();
}
