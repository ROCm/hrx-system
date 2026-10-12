// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/system_memory.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/cache.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/memory_coherence.h"
#include "loom/target/arch/amdgpu/planning/wait_packets.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

iree_status_t loom_amdgpu_system_memory_build_u32_attr(
    loom_builder_t* builder, iree_string_view_t name, uint32_t value,
    loom_named_attr_t* out_attr) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, name, &name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_system_memory_build_offset_attr(
    loom_builder_t* builder, uint32_t byte_offset,
    loom_named_attr_t* out_attr) {
  return loom_amdgpu_system_memory_build_u32_attr(builder, IREE_SV("offset"),
                                                  byte_offset, out_attr);
}

static bool loom_amdgpu_system_memory_type_is_register_class(
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t type,
    uint16_t reg_class_id) {
  return loom_low_type_is_register(type) &&
         loom_low_register_type_descriptor_set_stable_id(type) ==
             descriptor_set->stable_id &&
         loom_low_register_type_class_id(type) == reg_class_id;
}

static void loom_amdgpu_system_memory_require_register_class(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, uint16_t reg_class_id, uint32_t unit_count) {
  IREE_ASSERT_LT(value, builder->module->values.count);
  const loom_type_t type = loom_module_value_type(builder->module, value);
  IREE_ASSERT(loom_amdgpu_system_memory_type_is_register_class(
      descriptor_set, type, reg_class_id));
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(type), unit_count);
}

static iree_status_t loom_amdgpu_system_memory_build_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t value,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);

  loom_named_attr_t imm32_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_u32_attr(
      builder, IREE_SV("imm32"), value, &imm32_attr));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, descriptor_set, descriptor,
      loom_make_named_attr_slice(&imm32_attr, 1), result_type, location,
      &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_system_memory_build_vgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  return loom_amdgpu_system_memory_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, value,
      vgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_system_memory_build_sgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  return loom_amdgpu_system_memory_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
      sgpr_type, location, out_value);
}

iree_status_t loom_amdgpu_system_memory_build_saddr_byte_offset(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, uint32_t byte_offset,
    loom_location_id_t location, loom_value_id_t* out_address) {
  *out_address = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_system_memory_require_register_class(
      builder, descriptor_set, base_address, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  if (byte_offset == 0) {
    *out_address = base_address;
    return iree_ok_status();
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  loom_value_id_t base_lo = LOOM_VALUE_ID_INVALID;
  loom_op_t* slice_lo_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, base_address, /*offset=*/0,
                                            sgpr_type, location, &slice_lo_op));
  base_lo = loom_low_slice_result(slice_lo_op);
  loom_value_id_t base_hi = LOOM_VALUE_ID_INVALID;
  loom_op_t* slice_hi_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, base_address, /*offset=*/1,
                                            sgpr_type, location, &slice_hi_op));
  base_hi = loom_low_slice_result(slice_hi_op);

  loom_value_id_t offset_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_sgpr_u32_const(
      builder, descriptor_set, byte_offset, location, &offset_lo));
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_sgpr_u32_const(
      builder, descriptor_set, 0, location, &zero));

  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, &scc_type));
  const loom_type_t result_types[] = {sgpr_type, scc_type};
  const loom_value_id_t low_operands[] = {base_lo, offset_lo};
  const loom_low_descriptor_t* low_descriptor =
      loom_amdgpu_lookup_descriptor_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_CO_U32);
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, low_descriptor, /*access_flags=*/0, low_operands,
      IREE_ARRAYSIZE(low_operands), loom_named_attr_slice_empty(), result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &low_op));
  const loom_value_id_t high_operands[] = {
      base_hi, zero, loom_value_slice_get(loom_low_op_results(low_op), 1)};
  const loom_low_descriptor_t* high_descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set,
                                        LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32);
  loom_op_t* high_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, high_descriptor, /*access_flags=*/0,
      high_operands, IREE_ARRAYSIZE(high_operands),
      loom_named_attr_slice_empty(), result_types, IREE_ARRAYSIZE(result_types),
      /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &high_op));

  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));
  const loom_value_id_t parts[] = {
      loom_value_slice_get(loom_low_op_results(low_op), 0),
      loom_value_slice_get(loom_low_op_results(high_op), 0),
  };
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts), sgpr_x2_type,
                            location, &concat_op));
  *out_address = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_system_memory_build_global_load_saddr(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t register_count,
    loom_value_id_t base_address, uint32_t byte_offset,
    loom_amdgpu_system_memory_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_system_memory_require_register_class(
      builder, descriptor_set, base_address, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);

  loom_value_id_t zero_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero_vaddr));

  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, register_count,
      &result_type));
  loom_named_attr_t attrs[3] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_offset_attr(
      builder, byte_offset, &attrs[attr_count++]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_load_attrs(
      builder, descriptor_set, attrs, IREE_ARRAYSIZE(attrs), &attr_count));

  const loom_value_id_t operands[] = {zero_vaddr, base_address};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, /*access_flags=*/0, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  const loom_value_id_t value =
      loom_value_slice_get(loom_low_op_results(op), 0);
  if (iree_any_bit_set(flags, LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_ACQUIRE)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_acquire_ordering(
        builder, descriptor_set, location));
  }
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_system_memory_build_readfirstlane_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_system_memory_require_register_class(
      builder, descriptor_set, source, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &result_type));
  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32);
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, /*access_flags=*/0, &source,
      /*operand_count=*/1, loom_make_named_attr_slice(NULL, 0), &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_system_memory_append_u32_attr(
    loom_builder_t* builder, iree_string_view_t name, uint32_t value,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  IREE_ASSERT_LT(*inout_attr_count, attr_capacity);
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_system_memory_build_u32_attr(builder, name, value, &attr));
  attrs[(*inout_attr_count)++] = attr;
  return iree_ok_status();
}

bool loom_amdgpu_system_memory_release_ordering_available(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_memory_coherence_rule_t* rule =
      loom_amdgpu_memory_coherence_rule(descriptor_set);
  if (!rule) {
    return false;
  }
  loom_amdgpu_wait_packet_selection_t selection = {0};
  if (rule->writeback) {
    return loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                              rule->writeback) &&
           loom_amdgpu_wait_packet_try_select_counter_mask(
               descriptor_set, rule->writeback_wait_mask, 0, &selection);
  }
  return loom_amdgpu_wait_packet_try_select_counter_mask(
             descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD, 0,
             &selection) &&
         loom_amdgpu_wait_packet_try_select_counter_mask(
             descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE, 0,
             &selection);
}

bool loom_amdgpu_system_memory_acquire_ordering_available(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_memory_coherence_rule_t* rule =
      loom_amdgpu_memory_coherence_rule(descriptor_set);
  if (!rule) {
    return false;
  }
  for (uint8_t i = 0; i < rule->invalidate_count; ++i) {
    if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                            rule->invalidates[i])) {
      return false;
    }
  }
  loom_amdgpu_wait_packet_selection_t selection = {0};
  return loom_amdgpu_wait_packet_try_select_counter_mask(
      descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD, 0, &selection);
}

static iree_status_t loom_amdgpu_system_memory_append_attrs(
    loom_builder_t* builder, loom_amdgpu_memory_coherence_attrs_t flags,
    loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  loom_amdgpu_memory_coherence_attr_t
      selected[LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_CAPACITY];
  const uint8_t count =
      loom_amdgpu_memory_coherence_select_attrs(flags, scope, selected);
  for (uint8_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_u32_attr(
        builder, selected[i].name, selected[i].value, attrs, attr_capacity,
        inout_attr_count));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_system_memory_append_load_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  return loom_amdgpu_system_memory_append_load_attrs_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs, attr_capacity,
      inout_attr_count);
}

iree_status_t loom_amdgpu_system_memory_append_load_attrs_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  const loom_amdgpu_memory_coherence_rule_t* rule =
      loom_amdgpu_memory_coherence_rule(descriptor_set);
  return loom_amdgpu_system_memory_append_attrs(
      builder, rule->load_attrs[scope], scope, attrs, attr_capacity,
      inout_attr_count);
}

iree_status_t loom_amdgpu_system_memory_append_release_store_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  return loom_amdgpu_system_memory_append_release_store_attrs_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs, attr_capacity,
      inout_attr_count);
}

iree_status_t loom_amdgpu_system_memory_append_release_store_attrs_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  const loom_amdgpu_memory_coherence_rule_t* rule =
      loom_amdgpu_memory_coherence_rule(descriptor_set);
  return loom_amdgpu_system_memory_append_attrs(
      builder, rule->store_attrs[scope], scope, attrs, attr_capacity,
      inout_attr_count);
}

iree_status_t loom_amdgpu_system_memory_append_atomic_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  return loom_amdgpu_system_memory_append_atomic_attrs_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs, attr_capacity,
      inout_attr_count);
}

iree_status_t loom_amdgpu_system_memory_append_atomic_attrs_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  const loom_amdgpu_memory_coherence_rule_t* rule =
      loom_amdgpu_memory_coherence_rule(descriptor_set);
  return loom_amdgpu_system_memory_append_attrs(
      builder, rule->atomic_attrs[scope], scope, attrs, attr_capacity,
      inout_attr_count);
}

static iree_status_t loom_amdgpu_system_memory_build_resolved_packet(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_named_attr_slice_t attrs,
    loom_location_id_t location) {
  loom_op_t* op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, /*access_flags=*/0,
      /*operands=*/NULL,
      /*operand_count=*/0, attrs, /*result_types=*/NULL, /*result_count=*/0,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, &op);
}

static iree_status_t loom_amdgpu_system_memory_build_descriptor_packet(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_named_attr_slice_t attrs,
    loom_location_id_t location) {
  return loom_amdgpu_system_memory_build_resolved_packet(
      builder, descriptor_set, descriptor, attrs, location);
}

static iree_status_t loom_amdgpu_system_memory_build_explicit_packet(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_named_attr_slice_t attrs,
    loom_location_id_t location) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);
  return loom_amdgpu_system_memory_build_resolved_packet(
      builder, descriptor_set, descriptor, attrs, location);
}

static iree_status_t loom_amdgpu_system_memory_build_wait_counter_mask(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t counter_mask, uint16_t target_count, loom_location_id_t location) {
  loom_amdgpu_wait_packet_selection_t selection = {0};
  if (!loom_amdgpu_wait_packet_try_select_counter_mask(
          descriptor_set, counter_mask, target_count, &selection)) {
    IREE_ASSERT_UNREACHABLE(
        "validated AMDGPU system-memory wait counter packet");
    IREE_BUILTIN_UNREACHABLE();
  }
  loom_named_attr_t
      attrs[LOOM_AMDGPU_WAIT_PACKET_SELECTION_IMMEDIATE_CAPACITY] = {0};
  for (iree_host_size_t i = 0; i < selection.immediate_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_u32_attr(
        builder, selection.immediates[i].name, selection.immediates[i].value,
        &attrs[i]));
  }
  return loom_amdgpu_system_memory_build_descriptor_packet(
      builder, descriptor_set, selection.descriptor,
      loom_make_named_attr_slice(attrs, selection.immediate_count), location);
}

static iree_status_t loom_amdgpu_system_memory_build_cache_packet(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_coherence_rule_t* rule,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_cache_scope_t scope,
    loom_location_id_t location) {
  loom_named_attr_t attrs[LOOM_AMDGPU_MEMORY_COHERENCE_ATTR_CAPACITY] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_attrs(
      builder, rule->cache_attrs[scope], scope, attrs, IREE_ARRAYSIZE(attrs),
      &attr_count));
  return loom_amdgpu_system_memory_build_explicit_packet(
      builder, descriptor_set, descriptor_ref,
      loom_make_named_attr_slice(attrs, attr_count), location);
}

iree_status_t loom_amdgpu_system_memory_build_release_ordering(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  return loom_amdgpu_system_memory_build_release_ordering_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location);
}

iree_status_t loom_amdgpu_system_memory_build_release_ordering_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_location_id_t location) {
  const loom_amdgpu_memory_coherence_rule_t* rule =
      loom_amdgpu_memory_coherence_rule(descriptor_set);
  if (rule->writeback && scope >= rule->writeback_scope) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_cache_packet(
        builder, descriptor_set, rule, rule->writeback, scope, location));
    return loom_amdgpu_system_memory_build_wait_counter_mask(
        builder, descriptor_set, rule->writeback_wait_mask, 0, location);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_wait_counter_mask(
      builder, descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD, 0,
      location));
  return loom_amdgpu_system_memory_build_wait_counter_mask(
      builder, descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE, 0,
      location);
}

iree_status_t loom_amdgpu_system_memory_build_load_wait(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  return loom_amdgpu_system_memory_build_wait_counter_mask(
      builder, descriptor_set, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD, 0,
      location);
}

iree_status_t loom_amdgpu_system_memory_build_acquire_ordering(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  return loom_amdgpu_system_memory_build_acquire_ordering_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location);
}

iree_status_t loom_amdgpu_system_memory_build_acquire_ordering_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_location_id_t location) {
  const loom_amdgpu_memory_coherence_rule_t* rule =
      loom_amdgpu_memory_coherence_rule(descriptor_set);
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_load_wait(
      builder, descriptor_set, location));
  for (uint8_t i = 0; i < rule->invalidate_count; ++i) {
    if (!loom_amdgpu_memory_coherence_invalidates_at(rule, i, scope)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_cache_packet(
        builder, descriptor_set, rule, rule->invalidates[i], scope, location));
  }
  if (rule->invalidate_wait_mask) {
    return loom_amdgpu_system_memory_build_wait_counter_mask(
        builder, descriptor_set, rule->invalidate_wait_mask, 0, location);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_system_memory_build_uniform_load_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, uint32_t byte_offset,
    loom_amdgpu_system_memory_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t vector_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_global_load_saddr(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B32_SADDR,
      /*register_count=*/1, base_address, byte_offset, flags, location,
      &vector_value));
  return loom_amdgpu_system_memory_build_readfirstlane_b32(
      builder, descriptor_set, vector_value, location, out_value);
}

iree_status_t loom_amdgpu_system_memory_build_uniform_load_b64(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, uint32_t byte_offset,
    loom_amdgpu_system_memory_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t vector_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_global_load_saddr(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR,
      /*register_count=*/2, base_address, byte_offset, flags, location,
      &vector_value));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  loom_value_id_t vector_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(vector_lanes); ++i) {
    loom_op_t* slice_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, vector_value, i,
                                              vgpr_type, location, &slice_op));
    vector_lanes[i] = loom_low_slice_result(slice_op);
  }

  loom_value_id_t scalar_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(scalar_lanes); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_readfirstlane_b32(
        builder, descriptor_set, vector_lanes[i], location, &scalar_lanes[i]));
  }

  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, scalar_lanes, IREE_ARRAYSIZE(scalar_lanes),
                            sgpr_x2_type, location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}
