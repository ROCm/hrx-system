// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race.h"

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/module.h"
#include "loom/ops/atomic.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/data_symbol.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/feedback.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/preamble.h"
#include "loom/target/arch/amdgpu/lower/sanitizer.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race_report.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_report.h"
#include "loom/target/arch/amdgpu/lower/sync.h"
#include "loom/target/arch/amdgpu/lower/system_memory.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

typedef struct loom_amdgpu_sanitizer_race_config_values_t {
  // Address of the runtime-published TSAN config global.
  loom_value_id_t address;
  // TSAN config flags from loom_amdgpu_tsan_config_flag_bits_e.
  loom_value_id_t flags;
  // Log2 application bytes represented by one shadow entry.
  loom_value_id_t memory_granule_shift;
  // Queue-local shadow allocation base.
  loom_value_id_t shadow_base;
  // Bytes reserved for one dispatch shadow slot.
  loom_value_id_t dispatch_shadow_stride;
  // Bytes reserved for one workgroup shadow record.
  loom_value_id_t workgroup_shadow_stride;
  // Number of queue-local dispatch shadow slots available.
  loom_value_id_t shadow_slot_count;
} loom_amdgpu_sanitizer_race_config_values_t;

typedef struct loom_amdgpu_sanitizer_race_shadow_entry_base_t {
  // Low entry bits common to every access by this workitem in this dispatch.
  loom_value_id_t low;
  // High entry generation bits common to every access in this dispatch.
  loom_value_id_t high;
} loom_amdgpu_sanitizer_race_shadow_entry_base_t;

typedef struct loom_amdgpu_sanitizer_race_lower_state_t {
  // True once entry-block TSAN config values have been materialized.
  bool has_config_values;
  // Entry-dominating TSAN config values loaded from iree_tsan_config.
  loom_amdgpu_sanitizer_race_config_values_t config_values;
  // Entry-dominating TSAN config flags required by observation guards.
  loom_value_id_t required_config_flags;
  // Entry-dominating config.flags value masked by required_config_flags.
  loom_value_id_t active_config_flags;
  // True once entry-body TSAN instrumentation flags have been materialized.
  bool has_instrumentation_flags;
  // Entry-body guard flags. Zero disables observation lowering at runtime.
  loom_value_id_t instrumentation_flags;
  // True once entry-block launch topology values have been materialized.
  bool has_topology_values;
  // Entry-dominating flattened workgroup id for the active dispatch.
  loom_value_id_t workgroup_linear_id;
  // Entry-dominating byte offset of this workgroup inside a dispatch shadow.
  loom_value_id_t workgroup_shadow_offset;
  // Entry-dominating flattened workitem id in the supported TSAN shadow range.
  loom_value_id_t workitem_linear_id;
  // True once entry-block dispatch-slot values have been materialized.
  bool has_dispatch_values;
  // Entry-dominating pointer to the active AQL dispatch packet.
  loom_value_id_t dispatch_ptr;
  // Entry-dominating ID of the active dispatch used as a freshness tag.
  loom_value_id_t dispatch_id;
  // Entry-dominating queue-local TSAN shadow slot ordinal.
  loom_value_id_t shadow_slot;
  // True once the entry-body workgroup shadow record offset is materialized.
  bool has_workgroup_shadow_record_offset;
  // Entry-body byte offset of this workgroup shadow record from shadow_base.
  loom_value_id_t workgroup_shadow_record_offset;
  // True once entry-body shadow entry base bits have been materialized.
  bool has_shadow_entry_base;
  // Entry-body shadow entry bits shared by all access observations.
  loom_amdgpu_sanitizer_race_shadow_entry_base_t shadow_entry_base;
  // True once trap-mode TSAN failures have a shared cold island.
  bool has_trap_island;
  // Shared no-return cold island for trap-mode TSAN failures.
  loom_amdgpu_sanitizer_trap_island_t trap_island;
  // True once TSAN data-race failures have a shared cold report island.
  bool has_report_island;
  // Shared cold island for TSAN data-race reports.
  loom_amdgpu_sanitizer_race_report_island_t report_island;
} loom_amdgpu_sanitizer_race_lower_state_t;

typedef struct loom_amdgpu_sanitizer_race_dispatch_values_t {
  // Device-visible pointer to the current AQL dispatch packet.
  loom_value_id_t dispatch_ptr;
  // Byte offset of this workgroup shadow record from shadow_base.
  loom_value_id_t workgroup_shadow_record_offset;
} loom_amdgpu_sanitizer_race_dispatch_values_t;

typedef struct loom_amdgpu_sanitizer_race_shadow_address_t {
  // Memory-space-relative byte offset observed by the instrumented access.
  loom_value_id_t memory_byte_offset;
  // Per-workgroup shadow header byte offset from shadow_base.
  loom_value_id_t workgroup_offset;
  // Per-access shadow-entry byte offset from shadow_base.
  loom_value_id_t entry_offset;
} loom_amdgpu_sanitizer_race_shadow_address_t;

typedef struct loom_amdgpu_sanitizer_race_shadow_entry_t {
  // Low 32 bits of the compact TSAN shadow entry.
  loom_value_id_t low;
  // High 32 bits of the compact TSAN shadow entry.
  loom_value_id_t high;
  // Combined 64-bit TSAN shadow entry.
  loom_value_id_t value;
} loom_amdgpu_sanitizer_race_shadow_entry_t;

static int loom_amdgpu_sanitizer_race_lower_state_key;

static uint32_t loom_amdgpu_sanitizer_race_shadow_bit_mask(uint32_t bit_shift,
                                                           uint32_t bit_count) {
  if (bit_count == 32) {
    return UINT32_MAX;
  }
  return (((uint32_t)1u << bit_count) - 1u) << bit_shift;
}

static uint32_t loom_amdgpu_sanitizer_race_shadow_low_epoch_generation_mask(
    void) {
  const uint32_t bit_count = 32u - LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_SHIFT;
  return loom_amdgpu_sanitizer_race_shadow_bit_mask(
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_SHIFT, bit_count);
}

static uint32_t loom_amdgpu_sanitizer_race_shadow_low_generation_bit_count(
    void) {
  return 32u - LOOM_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_SHIFT;
}

static uint32_t loom_amdgpu_sanitizer_race_shadow_high_generation_bit_count(
    void) {
  return LOOM_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_BIT_COUNT -
         loom_amdgpu_sanitizer_race_shadow_low_generation_bit_count();
}

static uint32_t loom_amdgpu_sanitizer_race_shadow_high_generation_mask(void) {
  return loom_amdgpu_sanitizer_race_shadow_bit_mask(
      /*bit_shift=*/0,
      loom_amdgpu_sanitizer_race_shadow_high_generation_bit_count());
}

static uint32_t loom_amdgpu_sanitizer_race_shadow_workitem_mask(void) {
  return loom_amdgpu_sanitizer_race_shadow_bit_mask(
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_SHIFT,
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_BIT_COUNT);
}

static iree_status_t loom_amdgpu_sanitizer_race_lower_state(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_race_lower_state_t** out_state) {
  *out_state = NULL;
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_sanitizer_race_lower_state_key, sizeof(*state),
      (void**)&state));
  *out_state = state;
  return iree_ok_status();
}

static iree_host_size_t loom_amdgpu_sanitizer_race_packet_operand_count(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  iree_host_size_t count = 0;
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* operand = &operands[i];
    if (loom_low_operand_role_is_packet_operand(operand->role)) {
      ++count;
    }
  }
  return count;
}

static iree_status_t loom_amdgpu_sanitizer_race_build_descriptor_op(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_location_id_t location,
    loom_op_t** out_op) {
  *out_op = NULL;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, operand_count, attrs,
      result_types, result_count, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, out_op);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_u32_attr(
    loom_builder_t* builder, iree_string_view_t name, uint32_t value,
    loom_named_attr_t* out_attr) {
  *out_attr = (loom_named_attr_t){0};
  IREE_RETURN_IF_ERROR(
      loom_builder_intern_string(builder, name, &out_attr->name_id));
  out_attr->value = loom_attr_i64(value);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_m0_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* consumer_descriptor, uint32_t value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t m0_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_descriptor_implicit_resource_type(
      descriptor_set, consumer_descriptor, &m0_type));
  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM);
  loom_named_attr_t imm32_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_u32_attr(
      builder, IREE_SV("imm32"), value, &imm32_attr));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, descriptor_set, descriptor,
      loom_make_named_attr_slice(&imm32_attr, 1), m0_type, location,
      &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_append_optional_m0(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_location_id_t location,
    loom_value_id_t* operands, iree_host_size_t operand_capacity,
    iree_host_size_t* inout_operand_count) {
  const iree_host_size_t packet_operand_count =
      loom_amdgpu_sanitizer_race_packet_operand_count(descriptor_set,
                                                      descriptor);
  if (packet_operand_count == *inout_operand_count) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(packet_operand_count, *inout_operand_count + 1,
                 "AMDGPU sanitizer race descriptor has an unsupported packet "
                 "operand count");
  IREE_ASSERT_LT(*inout_operand_count, operand_capacity);
  return loom_amdgpu_sanitizer_race_build_m0_const_u32(
      builder, descriptor_set, descriptor, 0, location,
      &operands[(*inout_operand_count)++]);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_global_load(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t register_count,
    loom_value_id_t base_address, loom_value_id_t byte_offset,
    loom_amdgpu_system_memory_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;

  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, register_count,
      &result_type));
  loom_named_attr_t attrs[3] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_offset_attr(
      builder, 0, &attrs[attr_count++]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_load_attrs_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs,
      IREE_ARRAYSIZE(attrs), &attr_count));
  loom_amdgpu_filter_descriptor_optional_attrs(builder, descriptor_set,
                                               descriptor, /*required_count=*/1,
                                               attrs, &attr_count);

  loom_value_id_t operands[3] = {byte_offset, base_address,
                                 LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 2;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_append_optional_m0(
      builder, descriptor_set, descriptor, location, operands,
      IREE_ARRAYSIZE(operands), &operand_count));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  const loom_value_id_t value =
      loom_value_slice_get(loom_low_op_results(op), 0);
  if (iree_any_bit_set(flags, LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_ACQUIRE)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_system_memory_build_acquire_ordering_scoped(
            builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location));
  }
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_global_load_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, loom_value_id_t byte_offset,
    loom_amdgpu_system_memory_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_amdgpu_sanitizer_race_build_global_load(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B32_SADDR,
      /*register_count=*/1, base_address, byte_offset, flags, location,
      out_value);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_global_load_b64(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, loom_value_id_t byte_offset,
    loom_amdgpu_system_memory_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_amdgpu_sanitizer_race_build_global_load(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR,
      /*register_count=*/2, base_address, byte_offset, flags, location,
      out_value);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_global_swap_u64_acq_rel(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, loom_value_id_t byte_offset,
    loom_value_id_t desired_value, loom_location_id_t location,
    loom_value_id_t* out_observed_value) {
  *out_observed_value = LOOM_VALUE_ID_INVALID;

  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_SWAP_U64_RTN_SADDR);

  loom_value_id_t desired_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, desired_value, /*expected_unit_count=*/2,
      location, &desired_vgpr));

  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_system_memory_append_return_atomic_attrs_scoped(
          builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs,
          IREE_ARRAYSIZE(attrs), &attr_count));
  loom_amdgpu_filter_descriptor_optional_attrs(builder, descriptor_set,
                                               descriptor, /*required_count=*/0,
                                               attrs, &attr_count);

  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_release_ordering_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location));

  loom_value_id_t operands[4] = {byte_offset, desired_vgpr, base_address,
                                 LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 3;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_append_optional_m0(
      builder, descriptor_set, descriptor, location, operands,
      IREE_ARRAYSIZE(operands), &operand_count));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &result_type));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  const loom_value_id_t observed =
      loom_value_slice_get(loom_low_op_results(op), 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_acquire_ordering_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location));
  *out_observed_value = observed;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_global_atomic_add(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t register_count,
    loom_value_id_t base_address, loom_value_id_t byte_offset,
    loom_value_id_t value, loom_location_id_t location) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);

  loom_value_id_t value_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, value, register_count, location, &value_vgpr));

  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_system_memory_append_no_return_atomic_attrs_scoped(
          builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs,
          IREE_ARRAYSIZE(attrs), &attr_count));
  loom_amdgpu_filter_descriptor_optional_attrs(builder, descriptor_set,
                                               descriptor, /*required_count=*/0,
                                               attrs, &attr_count);

  loom_value_id_t operands[4] = {byte_offset, value_vgpr, base_address,
                                 LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 3;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_append_optional_m0(
      builder, descriptor_set, descriptor, location, operands,
      IREE_ARRAYSIZE(operands), &operand_count));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  return loom_amdgpu_system_memory_build_release_ordering_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, location);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_config_values(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_symbol_ref_t config_symbol,
    loom_amdgpu_sanitizer_race_config_values_t* out_values) {
  *out_values = (loom_amdgpu_sanitizer_race_config_values_t){0};
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_location_id_t location = source_op->location;

  loom_amdgpu_sanitizer_race_config_values_t values = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_build_data_symbol_address(builder, descriptor_set,
                                            (loom_amdgpu_data_symbol_address_t){
                                                .symbol = config_symbol,
                                                .byte_offset = 0,
                                            },
                                            location, &values.address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b32(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_TSAN_CONFIG_FLAGS_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location, &values.flags));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b32(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_TSAN_CONFIG_MEMORY_GRANULE_SHIFT_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location,
      &values.memory_granule_shift));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b64(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_TSAN_CONFIG_SHADOW_BASE_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location, &values.shadow_base));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b64(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_TSAN_CONFIG_DISPATCH_SHADOW_STRIDE_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location,
      &values.dispatch_shadow_stride));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b64(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_TSAN_CONFIG_WORKGROUP_SHADOW_STRIDE_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location,
      &values.workgroup_shadow_stride));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_uniform_load_b32(
      builder, descriptor_set, values.address,
      LOOM_AMDGPU_TSAN_CONFIG_SHADOW_SLOT_COUNT_OFFSET,
      LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_NONE, location,
      &values.shadow_slot_count));
  *out_values = values;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_emit_shadow_slot(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    loom_value_id_t dispatch_id, loom_value_id_t* out_shadow_slot) {
  *out_shadow_slot = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t dispatch_id_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, dispatch_id, 0, sgpr_type, &dispatch_id_low));
  loom_value_id_t shadow_slot_mask_sgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_SUB_U32,
      config->shadow_slot_count, 1, sgpr_type, &shadow_slot_mask_sgpr));
  loom_value_id_t shadow_slot_sgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32, dispatch_id_low,
      shadow_slot_mask_sgpr, sgpr_type, &shadow_slot_sgpr));

  return loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, shadow_slot_sgpr, out_shadow_slot);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_scaled_shadow_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t ordinal, loom_value_id_t stride,
    loom_value_id_t* out_offset) {
  *out_offset = LOOM_VALUE_ID_INVALID;
  loom_value_id_t ordinal_wide = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_from_u32(
      context, source_op, ordinal, &ordinal_wide));
  loom_value_id_t stride_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      loom_low_lower_context_builder(context),
      loom_low_lower_context_descriptor_set(context), stride,
      /*expected_unit_count=*/2, source_op->location, &stride_vgpr));
  return loom_amdgpu_emit_vgpr64_mul_lo(context, source_op, ordinal_wide,
                                        stride_vgpr, out_offset);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_config_guard_values(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t config_flags, loom_value_id_t* out_required_flags,
    loom_value_id_t* out_active_flags) {
  *out_required_flags = LOOM_VALUE_ID_INVALID;
  *out_active_flags = LOOM_VALUE_ID_INVALID;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  const uint32_t required_flags = LOOM_AMDGPU_TSAN_CONFIG_FLAG_ENABLED |
                                  LOOM_AMDGPU_TSAN_CONFIG_FLAG_QUEUE_STATE;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, required_flags,
      sgpr_type, out_required_flags));
  return loom_amdgpu_emit_sgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32, config_flags,
      *out_required_flags, sgpr_type, out_active_flags);
}

static iree_status_t loom_amdgpu_sanitizer_race_get_config_values(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_race_config_values_t* out_config) {
  *out_config = (loom_amdgpu_sanitizer_race_config_values_t){0};
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(state->has_config_values,
              "AMDGPU TSAN config values were not materialized in entry setup");
  *out_config = state->config_values;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_get_workgroup_shadow_offset(
    loom_low_lower_context_t* context, loom_value_id_t* out_offset) {
  *out_offset = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(
      state->has_topology_values,
      "AMDGPU TSAN topology values were not materialized in entry setup");
  *out_offset = state->workgroup_shadow_offset;
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_sanitizer_race_build_workgroup_shadow_record_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    loom_value_id_t shadow_slot, loom_value_id_t* out_offset) {
  *out_offset = LOOM_VALUE_ID_INVALID;

  loom_value_id_t dispatch_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_scaled_shadow_offset(
      context, source_op, shadow_slot, config->dispatch_shadow_stride,
      &dispatch_offset));

  loom_value_id_t workgroup_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_workgroup_shadow_offset(
      context, &workgroup_offset));
  return loom_amdgpu_emit_vgpr64_add(context, source_op, dispatch_offset,
                                     workgroup_offset, out_offset);
}

static iree_status_t loom_amdgpu_sanitizer_race_get_workitem_linear_id(
    loom_low_lower_context_t* context, loom_value_id_t* out_linear_id) {
  *out_linear_id = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(
      state->has_topology_values,
      "AMDGPU TSAN topology values were not materialized in entry setup");
  *out_linear_id = state->workitem_linear_id;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_get_dispatch_ptr(
    loom_low_lower_context_t* context, loom_value_id_t* out_dispatch_ptr) {
  *out_dispatch_ptr = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(
      state->has_dispatch_values,
      "AMDGPU TSAN dispatch slot values were not materialized in entry setup");
  *out_dispatch_ptr = state->dispatch_ptr;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_get_instrumentation_flags(
    loom_low_lower_context_t* context, loom_value_id_t* out_flags) {
  *out_flags = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(
      state->has_instrumentation_flags,
      "AMDGPU TSAN instrumentation flags were not materialized in entry setup");
  *out_flags = state->instrumentation_flags;
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_sanitizer_race_get_workgroup_shadow_record_offset(
    loom_low_lower_context_t* context, loom_value_id_t* out_offset) {
  *out_offset = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(state->has_workgroup_shadow_record_offset,
              "AMDGPU TSAN workgroup shadow record offset was not "
              "materialized in entry setup");
  *out_offset = state->workgroup_shadow_record_offset;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_get_shadow_entry_base(
    loom_low_lower_context_t* context,
    const loom_amdgpu_sanitizer_race_shadow_entry_base_t** out_base) {
  *out_base = NULL;
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(
      state->has_shadow_entry_base,
      "AMDGPU TSAN shadow entry base was not materialized in entry setup");
  *out_base = &state->shadow_entry_base;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_shadow_entry_base(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t workitem_linear_id, loom_value_id_t generation_low,
    loom_amdgpu_sanitizer_race_shadow_entry_base_t* out_base) {
  *out_base = (loom_amdgpu_sanitizer_race_shadow_entry_base_t){0};
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  loom_value_id_t workitem_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_SHIFT, workitem_linear_id,
      vgpr_type, &workitem_bits));

  loom_value_id_t generation_low_part = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      generation_low,
      (1u << loom_amdgpu_sanitizer_race_shadow_low_generation_bit_count()) - 1u,
      vgpr_type, &generation_low_part));
  loom_value_id_t generation_low_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_SHIFT, generation_low_part,
      vgpr_type, &generation_low_bits));

  loom_value_id_t low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, workitem_bits,
      generation_low_bits, vgpr_type, &low));

  loom_value_id_t generation_high_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      loom_amdgpu_sanitizer_race_shadow_low_generation_bit_count(),
      generation_low, vgpr_type, &generation_high_bits));
  loom_value_id_t high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      generation_high_bits,
      (1u << loom_amdgpu_sanitizer_race_shadow_high_generation_bit_count()) -
          1u,
      vgpr_type, &high));

  *out_base = (loom_amdgpu_sanitizer_race_shadow_entry_base_t){
      .low = low,
      .high = high,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_emit_dispatch_generation_low(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dispatch_id, loom_value_id_t* out_generation_low) {
  *out_generation_low = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t dispatch_id_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, dispatch_id, 0, sgpr_type, &dispatch_id_low));
  return loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, dispatch_id_low, out_generation_low);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_required_config_scc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_scc) {
  *out_scc = LOOM_VALUE_ID_INVALID;

  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(state->has_config_values,
              "AMDGPU TSAN config guard values were not materialized in entry "
              "setup");

  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  const loom_value_id_t operands[] = {state->active_config_flags,
                                      state->required_config_flags};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_I32, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &scc_type, 1,
      &compare_op));
  *out_scc = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_runtime_active_scc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_scc) {
  *out_scc = LOOM_VALUE_ID_INVALID;

  loom_value_id_t instrumentation_flags = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_instrumentation_flags(
      context, &instrumentation_flags));
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  IREE_ASSERT(state->has_config_values,
              "AMDGPU TSAN runtime guard values were not materialized in entry "
              "setup");

  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  const loom_value_id_t operands[] = {instrumentation_flags,
                                      state->required_config_flags};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_I32, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &scc_type, 1,
      &compare_op));
  *out_scc = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_split_enabled_block(
    loom_builder_t* builder, loom_value_id_t enabled_scc,
    loom_location_id_t location, loom_block_t** out_enabled_block,
    loom_block_t** out_continuation_block) {
  *out_enabled_block = NULL;
  *out_continuation_block = NULL;
  IREE_ASSERT(
      builder->ip.before_op == NULL && builder->ip.block->parent_region != NULL,
      "AMDGPU sanitizer race lowering requires block-end insertion in a low "
      "region");
  loom_block_t* config_block = builder->ip.block;
  loom_block_t* enabled_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, config_block->parent_region,
      (uint16_t)(config_block->region_index + 1), &enabled_block));
  loom_block_t* continuation_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, enabled_block->parent_region,
      (uint16_t)(enabled_block->region_index + 1), &continuation_block));
  loom_op_t* branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, enabled_scc,
                                              enabled_block, continuation_block,
                                              location, &branch_op));
  loom_builder_set_block(builder, enabled_block);
  *out_enabled_block = enabled_block;
  *out_continuation_block = continuation_block;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_branch_to_continuation(
    loom_builder_t* builder, loom_block_t* continuation_block,
    loom_location_id_t location) {
  loom_op_t* branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, continuation_block,
                                         /*args=*/NULL, /*args_count=*/0,
                                         location, &branch_op));
  loom_builder_set_block(builder, continuation_block);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_branch_to_entry_body(
    loom_low_lower_context_t* context,
    const loom_low_lower_entry_interposition_t* interposition,
    const loom_value_id_t* target_args, loom_location_id_t location) {
  const uint16_t arg_count =
      interposition->forwarded_arg_count + interposition->target_arg_count;
  loom_value_id_t* args = NULL;
  if (arg_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, arg_count, sizeof(*args), (void**)&args));
  }
  for (uint16_t i = 0; i < interposition->forwarded_arg_count; ++i) {
    args[i] = interposition->forwarded_args[i];
  }
  for (uint16_t i = 0; i < interposition->target_arg_count; ++i) {
    args[interposition->forwarded_arg_count + i] = target_args[i];
  }
  loom_op_t* branch_op = NULL;
  return loom_low_br_build(loom_low_lower_context_builder(context),
                           interposition->body_block, args, arg_count, location,
                           &branch_op);
}

static iree_status_t loom_amdgpu_sanitizer_race_split_entry_guard_blocks(
    loom_builder_t* builder, loom_value_id_t enabled_scc,
    loom_location_id_t location, loom_block_t** out_enabled_block,
    loom_block_t** out_disabled_block) {
  if (out_enabled_block != NULL) {
    *out_enabled_block = NULL;
  }
  if (out_disabled_block != NULL) {
    *out_disabled_block = NULL;
  }
  IREE_ASSERT(
      builder->ip.before_op == NULL && builder->ip.block->parent_region != NULL,
      "AMDGPU sanitizer race entry setup requires block-end insertion in a low "
      "region");
  loom_block_t* setup_block = builder->ip.block;
  loom_block_t* enabled_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, setup_block->parent_region,
      (uint16_t)(setup_block->region_index + 1), &enabled_block));
  loom_block_t* disabled_block = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, enabled_block->parent_region,
      (uint16_t)(enabled_block->region_index + 1), &disabled_block));
  loom_op_t* branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, enabled_scc,
                                              enabled_block, disabled_block,
                                              location, &branch_op));
  loom_builder_set_block(builder, enabled_block);
  if (out_enabled_block != NULL) {
    *out_enabled_block = enabled_block;
  }
  if (out_disabled_block != NULL) {
    *out_disabled_block = disabled_block;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_get_report_island(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_amdgpu_sanitizer_race_report_island_t** out_island) {
  *out_island = NULL;
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  if (!state->has_report_island) {
    loom_symbol_ref_t feedback_config_symbol = loom_symbol_ref_null();
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_feedback_config_symbol(
        context, &feedback_config_symbol));
    loom_builder_ip_t saved_ip = loom_builder_save(builder);
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_race_report_island(
        builder, descriptor_set, saved_ip.block, feedback_config_symbol,
        location, &state->report_island));
    loom_builder_restore(builder, saved_ip);
    state->has_report_island = true;
  }
  *out_island = &state->report_island;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_get_trap_island(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_amdgpu_sanitizer_trap_island_t** out_island) {
  *out_island = NULL;
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  if (!state->has_trap_island) {
    loom_builder_ip_t saved_ip = loom_builder_save(builder);
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_trap_island(
        builder, descriptor_set, saved_ip.block, location,
        &state->trap_island));
    loom_builder_restore(builder, saved_ip);
    state->has_trap_island = true;
  }
  *out_island = &state->trap_island;
  return iree_ok_status();
}

static bool loom_amdgpu_sanitizer_race_workgroup_size_supported(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle) {
  loom_target_workgroup_size_t workgroup_size = {0};
  uint32_t flat_workgroup_size = 0;
  return loom_amdgpu_required_workgroup_size(module, function, bundle,
                                             &workgroup_size) &&
         loom_amdgpu_required_flat_workgroup_size(module, function, bundle,
                                                  &flat_workgroup_size) &&
         workgroup_size.x != 0 && workgroup_size.y != 0 &&
         workgroup_size.z != 0 && flat_workgroup_size <= 1024 &&
         loom_amdgpu_u32_is_power_of_two(workgroup_size.x) &&
         loom_amdgpu_u32_is_power_of_two(workgroup_size.y) &&
         loom_amdgpu_u32_is_power_of_two(workgroup_size.z);
}

static bool loom_amdgpu_sanitizer_race_required_descriptors_present(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_descriptor_ref_t required_refs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B32_SADDR,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_SWAP_U64_RTN_SADDR,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_SUB_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32_RHS_SYMBOL_REL32_LO,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32_RHS_SYMBOL_REL32_HI,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_I32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_U64,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_GETPC_B64,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHR_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_CI_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_HI_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_CO_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_CO_CI_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32,
  };
  if (!loom_amdgpu_descriptor_set_has_all_refs(descriptor_set, required_refs,
                                               IREE_ARRAYSIZE(required_refs))) {
    return false;
  }
  return loom_amdgpu_descriptor_set_has_ref(
             descriptor_set,
             LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U32_SADDR) ||
         loom_amdgpu_descriptor_set_has_ref(
             descriptor_set,
             LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
}

static iree_status_t loom_amdgpu_sanitizer_race_vgpr_u64_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t value, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, (uint32_t)value,
      vgpr_type, &low));
  loom_value_id_t high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      (uint32_t)(value >> 32), vgpr_type, &high));
  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &vgpr_x2_type));
  const loom_value_id_t parts[] = {low, high};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), parts, IREE_ARRAYSIZE(parts),
      vgpr_x2_type, source_op->location, &concat_op));
  *out_low_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_zero_vgpr_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_low_value) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                    vgpr_type, out_low_value);
}

iree_status_t loom_amdgpu_sanitizer_race_emit_entry_setup(
    loom_low_lower_context_t* context) {
  const loom_op_t* first_race_op = NULL;
  const loom_op_t* first_race_access_op = NULL;
  const iree_host_size_t plan_count =
      loom_low_lower_context_selected_plan_count(context);
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    if (selected_plan.elided) {
      continue;
    }
    if (selected_plan.plan.id == LOOM_OP_SANITIZER_RACE_ACCESS) {
      if (first_race_access_op == NULL) {
        first_race_access_op = selected_plan.source_op;
      }
      if (first_race_op == NULL) {
        first_race_op = selected_plan.source_op;
      }
    } else if (selected_plan.plan.id == LOOM_OP_SANITIZER_RACE_SYNC &&
               first_race_op == NULL) {
      first_race_op = selected_plan.source_op;
    }
  }
  if (first_race_op == NULL) {
    return iree_ok_status();
  }

  loom_amdgpu_sanitizer_race_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_lower_state(context, &state));
  if (state->has_config_values) {
    return iree_ok_status();
  }

  loom_symbol_ref_t config_symbol = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_tsan_config_symbol(context, &config_symbol));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_config_values(
      context, first_race_op, config_symbol, &state->config_values));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_config_guard_values(
      context, first_race_op, state->config_values.flags,
      &state->required_config_flags, &state->active_config_flags));

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_workgroup_linear_id(
      context, first_race_op, sgpr_type, &state->workgroup_linear_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_scaled_shadow_offset(
      context, first_race_op, state->workgroup_linear_id,
      state->config_values.workgroup_shadow_stride,
      &state->workgroup_shadow_offset));
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_workitem_linear_id(
      context, first_race_op, vgpr_type, &state->workitem_linear_id));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_lookup_current_dispatch_ptr(context, &state->dispatch_ptr));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_lookup_current_dispatch_id(context, &state->dispatch_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_emit_shadow_slot(
      context, first_race_op, &state->config_values, state->dispatch_id,
      &state->shadow_slot));
  state->has_config_values = true;
  state->has_topology_values = true;
  state->has_dispatch_values = true;

  const bool has_race_access = first_race_access_op != NULL;
  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &vgpr_x2_type));
  loom_type_t entry_state_arg_types[4] = {
      sgpr_type,
      vgpr_x2_type,
      loom_type_none(),
      loom_type_none(),
  };
  uint16_t entry_state_arg_count = 2;
  if (has_race_access) {
    entry_state_arg_types[entry_state_arg_count++] = vgpr_type;
    entry_state_arg_types[entry_state_arg_count++] = vgpr_type;
  }
  loom_low_lower_entry_interposition_t entry = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_interpose_entry_block(
      context, entry_state_arg_types, entry_state_arg_count, &entry));
  IREE_ASSERT_EQ(entry.target_arg_count, entry_state_arg_count);
  state->instrumentation_flags = entry.target_args[0];
  state->workgroup_shadow_record_offset = entry.target_args[1];
  state->has_instrumentation_flags = true;
  state->has_workgroup_shadow_record_offset = true;
  if (has_race_access) {
    state->shadow_entry_base = (loom_amdgpu_sanitizer_race_shadow_entry_base_t){
        .low = entry.target_args[2],
        .high = entry.target_args[3],
    };
    state->has_shadow_entry_base = true;
  }

  loom_value_id_t enabled_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_required_config_scc(
      context, first_race_op, &enabled_scc));
  loom_block_t* config_disabled_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_split_entry_guard_blocks(
      loom_low_lower_context_builder(context), enabled_scc,
      first_race_op->location, /*out_enabled_block=*/NULL,
      &config_disabled_block));

  loom_value_id_t enabled_args[4] = {
      state->required_config_flags, LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  loom_value_id_t generation_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_build_workgroup_shadow_record_offset(
          context, first_race_op, &state->config_values, state->shadow_slot,
          &enabled_args[1]));
  if (has_race_access) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_sanitizer_race_emit_dispatch_generation_low(
            context, first_race_access_op, state->dispatch_id,
            &generation_low));
    loom_amdgpu_sanitizer_race_shadow_entry_base_t enabled_base = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_shadow_entry_base(
        context, first_race_access_op, state->workitem_linear_id,
        generation_low, &enabled_base));
    enabled_args[2] = enabled_base.low;
    enabled_args[3] = enabled_base.high;
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_branch_to_entry_body(
      context, &entry, enabled_args, first_race_op->location));

  loom_builder_set_block(loom_low_lower_context_builder(context),
                         config_disabled_block);
  loom_value_id_t zero_sgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, first_race_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0,
      sgpr_type, &zero_sgpr));
  loom_value_id_t zero_vgpr64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u64_constant(
      context, first_race_op, 0, &zero_vgpr64));
  loom_value_id_t disabled_args[4] = {
      zero_sgpr,
      zero_vgpr64,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  if (has_race_access) {
    loom_value_id_t zero_vgpr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_zero_vgpr_u32(
        context, first_race_op, &zero_vgpr));
    disabled_args[2] = zero_vgpr;
    disabled_args[3] = zero_vgpr;
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_branch_to_entry_body(
      context, &entry, disabled_args, first_race_op->location));
  loom_builder_set_block(loom_low_lower_context_builder(context),
                         entry.body_block);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_vgpr_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t value, loom_value_id_t* out_low_value) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, value,
                                    vgpr_type, out_low_value);
}

static iree_status_t loom_amdgpu_sanitizer_race_concat_vgpr_u64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low, loom_value_id_t high, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &vgpr_x2_type));
  const loom_value_id_t parts[] = {low, high};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), parts, IREE_ARRAYSIZE(parts),
      vgpr_x2_type, source_op->location, &concat_op));
  *out_low_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_vgpr_u32_or_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, uint32_t immediate, loom_value_id_t* out_low_value) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  if (immediate == 0) {
    *out_low_value = value;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32_LIT, value,
      immediate, vgpr_type, out_low_value);
}

static iree_status_t loom_amdgpu_sanitizer_race_vgpr_u32_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t false_value, loom_value_id_t true_value,
    loom_value_id_t condition_mask, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  const loom_value_id_t operands[] = {false_value, true_value, condition_mask};
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &vgpr_type, 1,
      &select_op));
  *out_value = loom_value_slice_get(loom_low_op_results(select_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_mask_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &op));
  *out_mask = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_dispatch_values(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_race_dispatch_values_t* out_values) {
  *out_values = (loom_amdgpu_sanitizer_race_dispatch_values_t){0};

  loom_amdgpu_sanitizer_race_dispatch_values_t values = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_dispatch_ptr(
      context, &values.dispatch_ptr));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_get_workgroup_shadow_record_offset(
          context, &values.workgroup_shadow_record_offset));
  *out_values = values;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_shadow_entry_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    const loom_amdgpu_sanitizer_race_dispatch_values_t* dispatch,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_sanitizer_race_shadow_address_t* out_address) {
  *out_address = (loom_amdgpu_sanitizer_race_shadow_address_t){0};
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  loom_amdgpu_sanitizer_race_shadow_address_t address = {0};
  address.workgroup_offset = dispatch->workgroup_shadow_record_offset;

  loom_value_id_t local_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vaddr(
      context, source_op, access, LOOM_VALUE_ID_INVALID, &local_byte_offset));
  address.memory_byte_offset = local_byte_offset;
  loom_value_id_t granule_shift = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, config->memory_granule_shift, &granule_shift));
  loom_value_id_t entry_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32,
      granule_shift, local_byte_offset, vgpr_type, &entry_index));
  loom_value_id_t entry_data_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 3,
      entry_index, vgpr_type, &entry_data_offset));
  loom_value_id_t entry_workgroup_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      entry_data_offset, LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_HEADER_BYTE_LENGTH,
      vgpr_type, &entry_workgroup_offset));
  loom_value_id_t entry_workgroup_offset_wide = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_from_u32(
      context, source_op, entry_workgroup_offset,
      &entry_workgroup_offset_wide));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_add(
      context, source_op, address.workgroup_offset, entry_workgroup_offset_wide,
      &address.entry_offset));
  *out_address = address;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_shadow_report_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    const loom_amdgpu_sanitizer_race_shadow_address_t* shadow_address,
    loom_value_id_t* out_address) {
  *out_address = LOOM_VALUE_ID_INVALID;
  loom_value_id_t shadow_base_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      loom_low_lower_context_builder(context),
      loom_low_lower_context_descriptor_set(context), config->shadow_base,
      /*expected_unit_count=*/2, source_op->location, &shadow_base_vgpr));
  return loom_amdgpu_emit_vgpr64_add(context, source_op, shadow_base_vgpr,
                                     shadow_address->entry_offset, out_address);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_shadow_entry(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_tsan_shadow_access_kind_t access_kind,
    const loom_amdgpu_sanitizer_race_shadow_entry_base_t* base,
    loom_value_id_t epoch, loom_sanitizer_site_id_t site_id,
    loom_amdgpu_sanitizer_race_shadow_entry_t* out_entry) {
  *out_entry = (loom_amdgpu_sanitizer_race_shadow_entry_t){0};
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  loom_value_id_t epoch_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, epoch,
      (1u << LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_BIT_COUNT) - 1u, vgpr_type,
      &epoch_low));
  loom_value_id_t epoch_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_SHIFT, epoch_low, vgpr_type,
      &epoch_bits));

  loom_value_id_t low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, access_kind, &low));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low, base->low,
      vgpr_type, &low));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low, epoch_bits,
      vgpr_type, &low));

  loom_value_id_t high = base->high;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_or_immediate(
      context, source_op, high,
      (site_id & ((1u << LOOM_AMDGPU_TSAN_SHADOW_ENTRY_SITE_BIT_COUNT) - 1u))
          << (LOOM_AMDGPU_TSAN_SHADOW_ENTRY_SITE_SHIFT - 32u),
      &high));

  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_concat_vgpr_u64(
      context, source_op, low, high, &value));
  *out_entry = (loom_amdgpu_sanitizer_race_shadow_entry_t){
      .low = low,
      .high = high,
      .value = value,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_unpack_shadow_entry(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t entry_value,
    loom_amdgpu_sanitizer_race_shadow_entry_t* out_entry) {
  *out_entry = (loom_amdgpu_sanitizer_race_shadow_entry_t){0};

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, entry_value, /*lane_offset=*/0, vgpr_type, &low));
  loom_value_id_t high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, entry_value, /*lane_offset=*/1, vgpr_type, &high));
  *out_entry = (loom_amdgpu_sanitizer_race_shadow_entry_t){
      .low = low,
      .high = high,
      .value = entry_value,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_decode_shadow_access_kind(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_shadow_entry_t* entry,
    loom_value_id_t* out_access_kind) {
  *out_access_kind = LOOM_VALUE_ID_INVALID;

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, entry->low,
      (1u << LOOM_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_BIT_COUNT) - 1u,
      vgpr_type, out_access_kind);
}

static iree_status_t loom_amdgpu_sanitizer_race_decode_shadow_workitem(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t entry_value, loom_value_id_t* out_workitem) {
  *out_workitem = LOOM_VALUE_ID_INVALID;

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, entry_value, /*lane_offset=*/0, vgpr_type, &low));
  loom_value_id_t shifted_workitem = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_SHIFT, low, vgpr_type,
      &shifted_workitem));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      shifted_workitem,
      (1u << LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_BIT_COUNT) - 1u, vgpr_type,
      out_workitem));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_decode_shadow_site_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t entry_value, loom_value_id_t* out_site_id) {
  *out_site_id = LOOM_VALUE_ID_INVALID;

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, entry_value, /*lane_offset=*/1, vgpr_type, &high));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_SITE_SHIFT - 32u, high, vgpr_type,
      out_site_id);
}

static iree_string_view_t
loom_amdgpu_sanitizer_race_source_memory_rejection_key(
    const loom_low_source_memory_access_diagnostic_t* diagnostic) {
  if (diagnostic->rejection_bits != 0) {
    return loom_low_source_memory_access_rejection_key(
        diagnostic->rejection_bits);
  }
  return IREE_SV("target_contract.sanitizer_race.source_memory_access");
}

static bool loom_amdgpu_sanitizer_race_access_source_kind(
    loom_sanitizer_race_access_kind_t kind,
    loom_low_source_memory_operation_kind_t* out_source_kind) {
  switch (kind) {
    case LOOM_SANITIZER_RACE_ACCESS_KIND_READ:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
      return true;
    case LOOM_SANITIZER_RACE_ACCESS_KIND_WRITE:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
      return true;
    case LOOM_SANITIZER_RACE_ACCESS_KIND_READ_WRITE:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_sanitizer_race_access_report_kind(
    loom_sanitizer_race_access_kind_t kind, bool atomic,
    loom_amdgpu_tsan_access_kind_t* out_report_kind,
    loom_amdgpu_tsan_shadow_access_kind_t* out_shadow_kind) {
  if (atomic) {
    *out_report_kind = LOOM_AMDGPU_TSAN_ACCESS_KIND_ATOMIC;
    *out_shadow_kind = LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC;
    return true;
  }
  switch (kind) {
    case LOOM_SANITIZER_RACE_ACCESS_KIND_READ:
      *out_report_kind = LOOM_AMDGPU_TSAN_ACCESS_KIND_READ;
      *out_shadow_kind = LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ;
      return true;
    case LOOM_SANITIZER_RACE_ACCESS_KIND_WRITE:
      *out_report_kind = LOOM_AMDGPU_TSAN_ACCESS_KIND_WRITE;
      *out_shadow_kind = LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_WRITE;
      return true;
    case LOOM_SANITIZER_RACE_ACCESS_KIND_READ_WRITE:
      *out_report_kind = LOOM_AMDGPU_TSAN_ACCESS_KIND_READ_WRITE;
      *out_shadow_kind = LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ_WRITE;
      return true;
    default:
      *out_report_kind = LOOM_AMDGPU_TSAN_ACCESS_KIND_UNKNOWN;
      *out_shadow_kind = LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_EMPTY;
      return false;
  }
}

static bool loom_amdgpu_sanitizer_race_access_payload_type(
    const loom_module_t* module, loom_value_id_t view_value_id,
    loom_type_t* out_vector_type) {
  *out_vector_type = loom_type_none();
  if (view_value_id >= module->values.count) {
    return false;
  }
  const loom_type_t view_type = loom_module_value_type(module, view_value_id);
  if (!loom_type_is_view(view_type)) {
    return false;
  }
  *out_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(view_type),
                          loom_dim_pack_static(1), /*encoding_id=*/0);
  return true;
}

static iree_status_t loom_amdgpu_sanitizer_race_reject_memory_selection(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  bool handled = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_access_rejection_diagnostic(
      context, op, source, diagnostic, &handled));
  if (handled) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(
      context, op,
      loom_amdgpu_memory_access_rejection_key(diagnostic->rejection_bits));
}

static bool loom_amdgpu_sanitizer_race_access_plan_build(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    const loom_amdgpu_source_alloca_layout_t* alloca_layout,
    const loom_op_t* op, loom_amdgpu_sanitizer_race_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_source_diagnostic,
    loom_amdgpu_memory_access_diagnostic_t* out_memory_diagnostic) {
  *out_plan = (loom_amdgpu_sanitizer_race_access_plan_t){0};
  *out_source_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};
  *out_memory_diagnostic = (loom_amdgpu_memory_access_diagnostic_t){0};

  loom_low_source_memory_operation_kind_t source_kind =
      LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  if (!loom_amdgpu_sanitizer_race_access_source_kind(
          loom_sanitizer_race_access_kind(op), &source_kind)) {
    return false;
  }

  const loom_value_id_t view_value_id = loom_sanitizer_race_access_view(op);
  loom_type_t vector_type = loom_type_none();
  if (!loom_amdgpu_sanitizer_race_access_payload_type(module, view_value_id,
                                                      &vector_type)) {
    out_source_diagnostic->rejection_bits =
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_LAYOUT;
    return false;
  }

  loom_low_source_memory_access_plan_t source = {0};
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_low_source_memory_access_plan_build_indexed_with_view_regions(
          module, fact_table, view_regions, source_kind, view_value_id,
          loom_sanitizer_race_access_indices(op),
          loom_sanitizer_race_access_static_indices(op), vector_type,
          cache_policy, &source, out_source_diagnostic)) {
    return false;
  }
  if (source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    out_memory_diagnostic->rejection_bits =
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE;
    return false;
  }

  loom_amdgpu_memory_access_t access = {0};
  access.source = source;
  out_plan->address.source = source;
  if (!loom_amdgpu_memory_access_select_u32_vaddr_byte_offset(
          module, fact_table, view_regions, alloca_layout, &source, &access,
          out_memory_diagnostic)) {
    return false;
  }

  loom_amdgpu_tsan_access_kind_t report_kind =
      LOOM_AMDGPU_TSAN_ACCESS_KIND_UNKNOWN;
  loom_amdgpu_tsan_shadow_access_kind_t shadow_kind =
      LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_EMPTY;
  if (!loom_amdgpu_sanitizer_race_access_report_kind(
          loom_sanitizer_race_access_kind(op),
          loom_sanitizer_race_access_atomic(op), &report_kind, &shadow_kind)) {
    out_source_diagnostic->rejection_bits =
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_UNSUPPORTED_OP;
    return false;
  }

  *out_plan = (loom_amdgpu_sanitizer_race_access_plan_t){
      .address = access,
      .report_access_kind = report_kind,
      .shadow_access_kind = shadow_kind,
      .access_size = source.element_byte_count * source.vector_lane_count,
      .atomic = loom_sanitizer_race_access_atomic(op),
  };
  return true;
}

static iree_status_t loom_amdgpu_sanitizer_race_verify_access_address(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_view_region_table_t* view_regions =
      loom_target_low_legality_view_regions(context);
  const loom_amdgpu_source_alloca_layout_t* alloca_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_low_legality(
      context, &alloca_layout));
  loom_amdgpu_sanitizer_race_access_plan_t plan = {0};
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t memory_diagnostic = {0};
  if (!loom_amdgpu_sanitizer_race_access_plan_build(
          module, loom_target_low_legality_fact_table(context), view_regions,
          alloca_layout, op, &plan, &source_diagnostic, &memory_diagnostic)) {
    if (source_diagnostic.rejection_bits != 0) {
      if (iree_any_bit_set(source_diagnostic.rejection_bits,
                           LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_LAYOUT)) {
        return loom_amdgpu_low_legality_reject(
            context, op,
            IREE_SV("target_contract.sanitizer_race.payload_type"));
      }
      if (iree_any_bit_set(
              source_diagnostic.rejection_bits,
              LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_UNSUPPORTED_OP)) {
        return loom_amdgpu_low_legality_reject(
            context, op, IREE_SV("target_contract.sanitizer_race.access_kind"));
      }
    }
    if (memory_diagnostic.rejection_bits ==
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE) {
      return loom_amdgpu_low_legality_reject(
          context, op,
          IREE_SV("target_contract.sanitizer_race.workgroup_memory_required"));
    }
    if (memory_diagnostic.rejection_bits != 0) {
      return loom_amdgpu_sanitizer_race_reject_memory_selection(
          context, op, &plan.address.source, &memory_diagnostic);
    }
    return loom_amdgpu_low_legality_reject(
        context, op,
        loom_amdgpu_sanitizer_race_source_memory_rejection_key(
            &source_diagnostic));
  }
  if (!loom_amdgpu_sanitizer_race_workgroup_size_supported(
          module, loom_target_low_legality_function(context),
          loom_target_low_legality_bundle(context))) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        IREE_SV("target_contract.sanitizer_race.fixed_power_of_two_workgroup_"
                "required"));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_sanitizer_race_access_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_race_access_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_sanitizer_race_access_plan_t){0};
  *out_selected = false;
  if (!loom_sanitizer_race_access_isa(source_op)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_sanitizer_race_required_descriptors_present(
          loom_low_lower_context_descriptor_set(context))) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_sanitizer_race_workgroup_size_supported(
          loom_low_lower_context_module(context),
          loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context))) {
    return iree_ok_status();
  }

  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  const loom_amdgpu_source_alloca_layout_t* alloca_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_lower_context(
      context, &alloca_layout));
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t memory_diagnostic = {0};
  if (!loom_amdgpu_sanitizer_race_access_plan_build(
          loom_low_lower_context_module(context),
          loom_low_lower_context_fact_table(context), view_regions,
          alloca_layout, source_op, out_plan, &source_diagnostic,
          &memory_diagnostic)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_site_id_for_op(
      context, source_op, &out_plan->site_id));
  if (out_plan->site_id == LOOM_SANITIZER_SITE_ID_INVALID) {
    return iree_ok_status();
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_config_guard(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_race_config_values_t* out_config,
    loom_block_t** out_continuation_block) {
  *out_config = (loom_amdgpu_sanitizer_race_config_values_t){0};
  *out_continuation_block = NULL;

  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_get_config_values(context, out_config));
  loom_value_id_t enabled_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_runtime_active_scc(
      context, source_op, &enabled_scc));
  loom_block_t* enabled_block = NULL;
  return loom_amdgpu_sanitizer_race_split_enabled_block(
      loom_low_lower_context_builder(context), enabled_scc, source_op->location,
      &enabled_block, out_continuation_block);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_current_epoch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    const loom_amdgpu_sanitizer_race_shadow_address_t* shadow_address,
    loom_value_id_t* out_epoch) {
  *out_epoch = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t epoch_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, shadow_address->workgroup_offset, /*lane_offset=*/0,
      vgpr_type, &epoch_offset));
  return loom_amdgpu_sanitizer_race_build_global_load_b32(
      loom_low_lower_context_builder(context),
      loom_low_lower_context_descriptor_set(context), config->shadow_base,
      epoch_offset, LOOM_AMDGPU_SYSTEM_MEMORY_LOAD_FLAG_ACQUIRE,
      source_op->location, out_epoch);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_shadow_entry_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_shadow_address_t* shadow_address,
    loom_value_id_t* out_offset) {
  *out_offset = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_low_slice(context, source_op,
                                    shadow_address->entry_offset,
                                    /*lane_offset=*/0, vgpr_type, out_offset);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_failure_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_access_plan_t* plan,
    const loom_amdgpu_sanitizer_race_shadow_entry_t* current_entry,
    const loom_amdgpu_sanitizer_race_shadow_entry_t* observed_entry,
    loom_value_id_t prior_access_kind, loom_value_id_t* out_failure_mask) {
  *out_failure_mask = LOOM_VALUE_ID_INVALID;

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_zero_vgpr_u32(context, source_op, &zero));

  loom_value_id_t prior_nonempty = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
      prior_access_kind, zero, &prior_nonempty));

  loom_value_id_t low_difference = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32,
      observed_entry->low, current_entry->low, vgpr_type, &low_difference));

  loom_value_id_t epoch_generation_low_difference = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      low_difference,
      loom_amdgpu_sanitizer_race_shadow_low_epoch_generation_mask(), vgpr_type,
      &epoch_generation_low_difference));
  loom_value_id_t high_difference = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32,
      observed_entry->high, current_entry->high, vgpr_type, &high_difference));
  loom_value_id_t epoch_generation_high_difference = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      high_difference, loom_amdgpu_sanitizer_race_shadow_high_generation_mask(),
      vgpr_type, &epoch_generation_high_difference));
  loom_value_id_t epoch_generation_difference = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
      epoch_generation_low_difference, epoch_generation_high_difference,
      vgpr_type, &epoch_generation_difference));
  loom_value_id_t same_epoch_generation = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      epoch_generation_difference, zero, &same_epoch_generation));

  loom_value_id_t workitem_difference = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      low_difference, loom_amdgpu_sanitizer_race_shadow_workitem_mask(),
      vgpr_type, &workitem_difference));
  loom_value_id_t different_workitem = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
      workitem_difference, zero, &different_workitem));

  loom_value_id_t conflict_kind = LOOM_VALUE_ID_INVALID;
  if (plan->shadow_access_kind == LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ) {
    loom_value_id_t prior_read = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
        context, source_op, LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ,
        &prior_read));
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
        prior_access_kind, prior_read, &conflict_kind));
  } else if (plan->shadow_access_kind ==
             LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC) {
    loom_value_id_t prior_atomic = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
        context, source_op, LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC,
        &prior_atomic));
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
        prior_access_kind, prior_atomic, &conflict_kind));
  } else {
    conflict_kind = LOOM_VALUE_ID_INVALID;
  }

  loom_value_id_t failure_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, prior_nonempty,
      same_epoch_generation, &failure_mask));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, failure_mask,
      different_workitem, &failure_mask));
  loom_value_id_t raw_failure_mask = LOOM_VALUE_ID_INVALID;
  if (conflict_kind == LOOM_VALUE_ID_INVALID) {
    raw_failure_mask = failure_mask;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, failure_mask,
        conflict_kind, &raw_failure_mask));
  }
  const uint32_t wavefront_size =
      loom_amdgpu_target_wavefront_size(loom_low_lower_context_bundle(context));
  return loom_amdgpu_build_feedback_canonical_exec_mask(
      loom_low_lower_context_builder(context),
      loom_low_lower_context_descriptor_set(context), raw_failure_mask,
      wavefront_size, source_op->location, out_failure_mask);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_report(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_access_plan_t* plan,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    const loom_amdgpu_sanitizer_race_dispatch_values_t* dispatch,
    const loom_amdgpu_sanitizer_race_shadow_address_t* shadow_address,
    loom_value_id_t current_workitem_linear_id,
    const loom_amdgpu_sanitizer_race_shadow_entry_t* observed_entry,
    loom_value_id_t prior_access_kind,
    loom_amdgpu_feedback_packet_source_t* out_source,
    loom_amdgpu_sanitizer_race_report_t* out_report) {
  *out_source = (loom_amdgpu_feedback_packet_source_t){0};
  *out_report = (loom_amdgpu_sanitizer_race_report_t){0};

  loom_value_id_t workgroup_id_x = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_current_workgroup_id(
      context, LOOM_KERNEL_DIMENSION_X, &workgroup_id_x));
  loom_value_id_t workgroup_id_y = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_current_workgroup_id(
      context, LOOM_KERNEL_DIMENSION_Y, &workgroup_id_y));
  loom_value_id_t workgroup_id_z = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_current_workgroup_id(
      context, LOOM_KERNEL_DIMENSION_Z, &workgroup_id_z));
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_zero_vgpr_u32(context, source_op, &zero));

  loom_value_id_t prior_atomic_kind = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC,
      &prior_atomic_kind));
  loom_value_id_t prior_atomic_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      prior_access_kind, prior_atomic_kind, &prior_atomic_mask));

  const uint32_t static_flags =
      LOOM_AMDGPU_TSAN_REPORT_FLAG_CURRENT_WORKITEM_LINEAR |
      LOOM_AMDGPU_TSAN_REPORT_FLAG_PRIOR_WORKITEM_LINEAR |
      (plan->atomic ? LOOM_AMDGPU_TSAN_REPORT_FLAG_CURRENT_ATOMIC : 0u);
  loom_value_id_t base_flags = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, static_flags, &base_flags));
  loom_value_id_t atomic_flags = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op,
      static_flags | LOOM_AMDGPU_TSAN_REPORT_FLAG_PRIOR_ATOMIC, &atomic_flags));
  loom_value_id_t flags = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_select(
      context, source_op, base_flags, atomic_flags, prior_atomic_mask, &flags));

  out_source->dispatch_ptr = dispatch->dispatch_ptr;
  out_source->workgroup_id_x = workgroup_id_x;
  out_source->workitem_id_x = current_workitem_linear_id;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, LOOM_AMDGPU_TSAN_CHECK_KIND_DATA_RACE,
      &out_report->check_kind));
  out_report->flags = flags;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, LOOM_AMDGPU_TSAN_MEMORY_SPACE_WORKGROUP,
      &out_report->memory_space));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, plan->report_access_kind,
      &out_report->current_access_kind));
  out_report->prior_access_kind = prior_access_kind;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
      context, source_op, plan->access_size, &out_report->access_size));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u64_constant(
      context, source_op, plan->site_id, &out_report->current_site_id));
  loom_value_id_t prior_site_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_decode_shadow_site_id(
      context, source_op, observed_entry->value, &prior_site_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_from_u32(
      context, source_op, prior_site_id, &out_report->prior_site_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_from_u32(
      context, source_op, shadow_address->memory_byte_offset,
      &out_report->memory_address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_shadow_report_address(
      context, source_op, config, shadow_address, &out_report->shadow_address));
  out_report->shadow_value = observed_entry->value;
  loom_value_id_t prior_workitem = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_decode_shadow_workitem(
      context, source_op, observed_entry->value, &prior_workitem));
  out_report->current_workgroup_id_x = workgroup_id_x;
  out_report->current_workgroup_id_y = workgroup_id_y;
  out_report->current_workgroup_id_z = workgroup_id_z;
  out_report->current_workitem_id_x = current_workitem_linear_id;
  out_report->current_workitem_id_y = zero;
  out_report->current_workitem_id_z = zero;
  out_report->prior_workgroup_id_x = workgroup_id_x;
  out_report->prior_workgroup_id_y = workgroup_id_y;
  out_report->prior_workgroup_id_z = workgroup_id_z;
  out_report->prior_workitem_id_x = prior_workitem;
  out_report->prior_workitem_id_y = zero;
  out_report->prior_workitem_id_z = zero;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_sanitizer_race_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_access_plan_t* plan) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);

  loom_amdgpu_sanitizer_race_config_values_t config = {0};
  loom_block_t* continuation_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_config_guard(
      context, source_op, &config, &continuation_block));

  loom_amdgpu_sanitizer_race_dispatch_values_t dispatch = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_build_dispatch_values(context, &dispatch));

  loom_amdgpu_sanitizer_race_shadow_address_t shadow_address = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_shadow_entry_address(
      context, source_op, &config, &dispatch, &plan->address, &shadow_address));

  loom_value_id_t epoch = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_current_epoch(
      context, source_op, &config, &shadow_address, &epoch));

  loom_value_id_t workitem_linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_workitem_linear_id(
      context, &workitem_linear_id));

  const loom_amdgpu_sanitizer_race_shadow_entry_base_t* shadow_entry_base =
      NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_shadow_entry_base(
      context, &shadow_entry_base));
  loom_amdgpu_sanitizer_race_shadow_entry_t current_entry = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_shadow_entry(
      context, source_op, plan->shadow_access_kind, shadow_entry_base, epoch,
      plan->site_id, &current_entry));

  loom_value_id_t shadow_entry_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_shadow_entry_offset(
      context, source_op, &shadow_address, &shadow_entry_offset));
  loom_value_id_t observed_entry_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_global_swap_u64_acq_rel(
      builder, descriptor_set, config.shadow_base, shadow_entry_offset,
      current_entry.value, source_op->location, &observed_entry_value));

  loom_amdgpu_sanitizer_race_shadow_entry_t observed_entry = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_unpack_shadow_entry(
      context, source_op, observed_entry_value, &observed_entry));
  loom_value_id_t prior_access_kind = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_decode_shadow_access_kind(
      context, source_op, &observed_entry, &prior_access_kind));

  loom_value_id_t failure_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_failure_mask(
      context, source_op, plan, &current_entry, &observed_entry,
      prior_access_kind, &failure_mask));

  const loom_sanitizer_reporting_mode_t reporting_mode =
      loom_low_lower_context_sanitizer_reporting_mode(context);
  switch (reporting_mode) {
    case LOOM_SANITIZER_REPORTING_MODE_TRAP: {
      loom_amdgpu_sanitizer_access_report_failure_branch_t branch = {0};
      const loom_amdgpu_sanitizer_trap_island_t* island = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_trap_island(
          context, source_op->location, &island));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_build_sanitizer_trap_failure_mask_branch_to_island(
              builder, descriptor_set, island, failure_mask,
              source_op->location, &branch));
      break;
    }
    case LOOM_SANITIZER_REPORTING_MODE_DEFAULT:
    case LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY: {
      loom_amdgpu_sanitizer_race_report_failure_branch_t branch = {0};
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_build_sanitizer_race_report_failure_mask_split(
              builder, descriptor_set, failure_mask, source_op->location,
              &branch));

      loom_amdgpu_feedback_packet_source_t source = {0};
      loom_amdgpu_sanitizer_race_report_t report = {0};
      loom_value_id_t report_workitem_linear_id = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
          context, source_op, workitem_linear_id, &report_workitem_linear_id));
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_report(
          context, source_op, plan, &config, &dispatch, &shadow_address,
          report_workitem_linear_id, &observed_entry, prior_access_kind,
          &source, &report));
      const loom_amdgpu_sanitizer_race_report_island_t* island = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_report_island(
          context, source_op->location, &island));
      IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_race_report_branch(
          builder, descriptor_set, island, &source, &report,
          source_op->location));
      loom_builder_set_block(builder, branch.continuation_block);
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unsupported AMDGPU sanitizer reporting mode");
      IREE_BUILTIN_UNREACHABLE();
  }

  return loom_amdgpu_sanitizer_race_branch_to_continuation(
      builder, continuation_block, source_op->location);
}

iree_status_t loom_amdgpu_select_sanitizer_race_sync_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_race_sync_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_sanitizer_race_sync_plan_t){0};
  *out_selected = false;
  if (!loom_sanitizer_race_sync_isa(source_op)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_sanitizer_race_required_descriptors_present(
          loom_low_lower_context_descriptor_set(context))) {
    return iree_ok_status();
  }
  if (loom_sanitizer_race_sync_memory_space(source_op) !=
          LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      loom_sanitizer_race_sync_scope(source_op) !=
          LOOM_ATOMIC_SCOPE_WORKGROUP) {
    return iree_ok_status();
  }
  const loom_atomic_ordering_t ordering =
      loom_sanitizer_race_sync_ordering(source_op);
  if (ordering != LOOM_ATOMIC_ORDERING_ACQ_REL &&
      ordering != LOOM_ATOMIC_ORDERING_SEQ_CST) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_sanitizer_race_workgroup_size_supported(
          loom_low_lower_context_module(context),
          loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context))) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_select_workgroup_barrier_plan(context, &out_plan->barrier));
  if (out_plan->barrier.kind == LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_NONE) {
    return iree_ok_status();
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_exec_narrow_and_save(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lane_mask, loom_location_id_t location,
    loom_value_id_t* out_saved_exec) {
  *out_saved_exec = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, &scc_type));
  const loom_type_t result_types[] = {sgpr_x2_type, scc_type};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
      &lane_mask, /*operand_count=*/1, loom_named_attr_slice_empty(),
      result_types, IREE_ARRAYSIZE(result_types), location, &op));
  *out_saved_exec = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_exec_restore(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t saved_exec, loom_location_id_t location) {
  loom_op_t* op = NULL;
  return loom_amdgpu_sanitizer_race_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, /*operand_count=*/1, loom_named_attr_slice_empty(),
      /*result_types=*/NULL, /*result_count=*/0, location, &op);
}

iree_status_t loom_amdgpu_lower_sanitizer_race_sync(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_sync_plan_t* plan) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);

  loom_amdgpu_sanitizer_race_config_values_t config = {0};
  loom_block_t* continuation_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_config_guard(
      context, source_op, &config, &continuation_block));

  loom_amdgpu_sanitizer_race_dispatch_values_t dispatch = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_build_dispatch_values(context, &dispatch));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t epoch_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, dispatch.workgroup_shadow_record_offset,
      /*lane_offset=*/0, vgpr_type, &epoch_offset));

  loom_value_id_t workitem_linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_workitem_linear_id(
      context, &workitem_linear_id));
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_zero_vgpr_u32(context, source_op, &zero));
  loom_value_id_t leader_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_cmp_mask(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      workitem_linear_id, zero, &leader_mask));

  loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_exec_narrow_and_save(
      builder, descriptor_set, leader_mask, source_op->location, &saved_exec));
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U32_SADDR)) {
    loom_value_id_t one = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u32_constant(
        context, source_op, 1, &one));
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_global_atomic_add(
        builder, descriptor_set,
        LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U32_SADDR,
        /*register_count=*/1, config.shadow_base, epoch_offset, one,
        source_op->location));
  } else {
    loom_value_id_t one = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_vgpr_u64_constant(
        context, source_op, 1, &one));
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_global_atomic_add(
        builder, descriptor_set,
        LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR,
        /*register_count=*/2, config.shadow_base, epoch_offset, one,
        source_op->location));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_exec_restore(
      builder, descriptor_set, saved_exec, source_op->location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_lower_workgroup_barrier_plan(
      context, source_op, &plan->barrier));

  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_branch_to_continuation(
      builder, continuation_block, source_op->location));
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_sanitizer_race_access(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  if (!loom_sanitizer_race_access_isa(op)) {
    return iree_ok_status();
  }
  *out_handled = true;
  return loom_amdgpu_sanitizer_race_verify_access_address(context, op);
}

iree_status_t loom_amdgpu_low_legality_verify_sanitizer_race_sync(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  if (!loom_sanitizer_race_sync_isa(op)) {
    return iree_ok_status();
  }
  *out_handled = true;

  if (loom_sanitizer_race_sync_memory_space(op) !=
      LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        IREE_SV("target_contract.sanitizer_race.workgroup_memory_required"));
  }
  if (loom_sanitizer_race_sync_scope(op) != LOOM_ATOMIC_SCOPE_WORKGROUP) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        IREE_SV("target_contract.sanitizer_race.workgroup_scope_required"));
  }

  switch (loom_sanitizer_race_sync_ordering(op)) {
    case LOOM_ATOMIC_ORDERING_ACQ_REL:
    case LOOM_ATOMIC_ORDERING_SEQ_CST:
      break;
    case LOOM_ATOMIC_ORDERING_ACQUIRE:
    case LOOM_ATOMIC_ORDERING_RELEASE:
    case LOOM_ATOMIC_ORDERING_RELAXED:
    case LOOM_ATOMIC_ORDERING_COUNT_:
    default:
      return loom_amdgpu_low_legality_reject(
          context, op,
          IREE_SV("target_contract.sanitizer_race.acq_rel_sync_required"));
  }
  if (!loom_amdgpu_sanitizer_race_workgroup_size_supported(
          loom_target_low_legality_module(context),
          loom_target_low_legality_function(context),
          loom_target_low_legality_bundle(context))) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        IREE_SV("target_contract.sanitizer_race.fixed_power_of_two_workgroup_"
                "required"));
  }
  if (!loom_amdgpu_workgroup_barrier_lowering_available(
          loom_target_low_legality_descriptor_set(context))) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("descriptor.workgroup_barrier"));
  }
  return iree_ok_status();
}
