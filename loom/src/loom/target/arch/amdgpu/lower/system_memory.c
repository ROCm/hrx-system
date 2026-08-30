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

static iree_status_t loom_amdgpu_system_memory_build_sgpr_u32_binary(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_system_memory_require_register_class(
      builder, descriptor_set, lhs, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  loom_amdgpu_system_memory_require_register_class(
      builder, descriptor_set, rhs, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  const loom_value_id_t operands[] = {lhs, rhs};
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &sgpr_type,
      /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_system_memory_build_m0_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* consumer_descriptor, uint32_t value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t m0_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_descriptor_implicit_resource_type(
      descriptor_set, consumer_descriptor, &m0_type));
  return loom_amdgpu_system_memory_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM,
      value, m0_type, location, out_value);
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

  loom_value_id_t sum_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_sgpr_u32_binary(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32, base_lo,
      offset_lo, location, &sum_lo));
  loom_value_id_t sum_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_sgpr_u32_binary(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32, base_hi,
      zero, location, &sum_hi));

  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));
  const loom_value_id_t parts[] = {sum_lo, sum_hi};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts), sgpr_x2_type,
                            location, &concat_op));
  *out_address = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static void loom_amdgpu_system_memory_global_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor,
    const loom_low_asm_form_t** out_asm_form) {
  *out_descriptor = NULL;
  *out_asm_form = NULL;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, descriptor_ref);
  const uint32_t canonical_asm_form_ordinal =
      loom_low_descriptor_set_descriptor_view(descriptor_set, descriptor)
          ->canonical_asm_form_ordinal;
  IREE_ASSERT_LT(canonical_asm_form_ordinal, descriptor_set->asm_form_count);
  const loom_low_asm_form_t* asm_form =
      &descriptor_set->asm_forms[canonical_asm_form_ordinal];
  IREE_ASSERT(asm_form->operand_index_count == 2 ||
              asm_form->operand_index_count == 3);
  *out_descriptor = descriptor;
  *out_asm_form = asm_form;
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

  const loom_low_descriptor_t* descriptor = NULL;
  const loom_low_asm_form_t* asm_form = NULL;
  loom_amdgpu_system_memory_global_memory_descriptor(
      descriptor_set, descriptor_ref, &descriptor, &asm_form);

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

  loom_value_id_t operands[3] = {zero_vaddr, base_address,
                                 LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 2;
  if (asm_form->operand_index_count == 3) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_m0_const_u32(
        builder, descriptor_set, descriptor, 0, location,
        &operands[operand_count++]));
  }
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type,
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
      builder, descriptor_set, descriptor, &source,
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

typedef uint32_t loom_amdgpu_system_memory_attr_flags_t;

#define LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_GLC ((uint32_t)1u << 0)
#define LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_DLC ((uint32_t)1u << 1)
#define LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SC0 ((uint32_t)1u << 2)
#define LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SC1 ((uint32_t)1u << 3)
#define LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SCOPE ((uint32_t)1u << 4)

typedef struct loom_amdgpu_system_memory_attr_field_t {
  // Presence bit required for this descriptor attribute.
  loom_amdgpu_system_memory_attr_flags_t flag;
  // Descriptor attribute name.
  iree_string_view_t name;
} loom_amdgpu_system_memory_attr_field_t;

static const loom_amdgpu_system_memory_attr_field_t
    kAmdgpuSystemMemoryAttrFields[] = {
        {
            .flag = LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_GLC,
            .name = IREE_SVL("glc"),
        },
        {
            .flag = LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_DLC,
            .name = IREE_SVL("dlc"),
        },
        {
            .flag = LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SC0,
            .name = IREE_SVL("sc0"),
        },
        {
            .flag = LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SC1,
            .name = IREE_SVL("sc1"),
        },
        {
            .flag = LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SCOPE,
            .name = IREE_SVL("scope"),
        },
};

typedef enum loom_amdgpu_system_memory_action_kind_e {
  // No system-memory ordering action.
  LOOM_AMDGPU_SYSTEM_MEMORY_ACTION_NONE = 0,
  // Emit a wait packet selected by VMEM counter mask.
  LOOM_AMDGPU_SYSTEM_MEMORY_ACTION_WAIT_COUNTER = 1,
  // Emit a descriptor-ref packet with table-selected attrs.
  LOOM_AMDGPU_SYSTEM_MEMORY_ACTION_EXPLICIT_PACKET = 2,
} loom_amdgpu_system_memory_action_kind_t;

typedef struct loom_amdgpu_system_memory_action_t {
  // Concrete action emitted for this ordering step.
  loom_amdgpu_system_memory_action_kind_t kind;
  // Wait counter mask when |kind| is WAIT_COUNTER.
  uint32_t counter_mask;
  // Explicit packet descriptor ref when |kind| is EXPLICIT_PACKET.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Descriptor attrs appended when |kind| is EXPLICIT_PACKET.
  loom_amdgpu_system_memory_attr_flags_t attr_flags;
} loom_amdgpu_system_memory_action_t;

#define LOOM_AMDGPU_SYSTEM_MEMORY_WAIT(counter_mask_)        \
  {                                                          \
      .kind = LOOM_AMDGPU_SYSTEM_MEMORY_ACTION_WAIT_COUNTER, \
      .counter_mask = (counter_mask_),                       \
  }

#define LOOM_AMDGPU_SYSTEM_MEMORY_PACKET(descriptor_ref_, attr_flags_) \
  {                                                                    \
      .kind = LOOM_AMDGPU_SYSTEM_MEMORY_ACTION_EXPLICIT_PACKET,        \
      .descriptor_ref = (descriptor_ref_),                             \
      .attr_flags = (attr_flags_),                                     \
  }

#define LOOM_AMDGPU_SYSTEM_MEMORY_ORDERING_ACTION_CAPACITY 3

typedef struct loom_amdgpu_system_memory_policy_t {
  // Descriptor-set cache-policy encoding selected by target info.
  loom_amdgpu_vector_memory_cache_policy_encoding_t encoding;
  // Attrs appended to system-memory load packets.
  loom_amdgpu_system_memory_attr_flags_t load_attrs;
  // Attrs appended to system-release store packets.
  loom_amdgpu_system_memory_attr_flags_t release_store_attrs;
  // Attrs appended to no-return system atomic packets.
  loom_amdgpu_system_memory_attr_flags_t no_return_atomic_attrs;
  // Attrs appended to returning system atomic packets.
  loom_amdgpu_system_memory_attr_flags_t return_atomic_attrs;
  // Packets emitted before a system-release publication.
  loom_amdgpu_system_memory_action_t
      release_actions[LOOM_AMDGPU_SYSTEM_MEMORY_ORDERING_ACTION_CAPACITY];
  // Number of populated release ordering actions.
  iree_host_size_t release_action_count;
  // Packets emitted to wait for system-memory loads.
  loom_amdgpu_system_memory_action_t
      load_wait_actions[LOOM_AMDGPU_SYSTEM_MEMORY_ORDERING_ACTION_CAPACITY];
  // Number of populated load-wait actions.
  iree_host_size_t load_wait_action_count;
  // Packets emitted after a system-acquire operation.
  loom_amdgpu_system_memory_action_t
      acquire_actions[LOOM_AMDGPU_SYSTEM_MEMORY_ORDERING_ACTION_CAPACITY];
  // Number of populated acquire ordering actions.
  iree_host_size_t acquire_action_count;
} loom_amdgpu_system_memory_policy_t;

static const loom_amdgpu_system_memory_policy_t kAmdgpuSystemMemoryPolicies[] = {
    {
        .encoding =
            LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC,
        .load_attrs = LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_GLC |
                      LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_DLC,
        .release_actions =
            {
                LOOM_AMDGPU_SYSTEM_MEMORY_WAIT(
                    LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD),
                LOOM_AMDGPU_SYSTEM_MEMORY_WAIT(
                    LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE),
            },
        .release_action_count = 2,
        .load_wait_actions =
            {
                LOOM_AMDGPU_SYSTEM_MEMORY_WAIT(
                    LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD),
            },
        .load_wait_action_count = 1,
        .acquire_actions =
            {
                LOOM_AMDGPU_SYSTEM_MEMORY_WAIT(
                    LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD),
                LOOM_AMDGPU_SYSTEM_MEMORY_PACKET(
                    LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL1_INV, 0),
                LOOM_AMDGPU_SYSTEM_MEMORY_PACKET(
                    LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL0_INV, 0),
            },
        .acquire_action_count = 3,
    },
    {
        .encoding =
            LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
        .load_attrs = LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SCOPE,
        .release_store_attrs = LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SCOPE,
        .no_return_atomic_attrs = LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SCOPE,
        .return_atomic_attrs = LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SCOPE,
        .release_actions =
            {
                LOOM_AMDGPU_SYSTEM_MEMORY_PACKET(
                    LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_WB,
                    LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SCOPE),
                LOOM_AMDGPU_SYSTEM_MEMORY_WAIT(
                    LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE),
            },
        .release_action_count = 2,
        .load_wait_actions =
            {
                LOOM_AMDGPU_SYSTEM_MEMORY_WAIT(
                    LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD),
            },
        .load_wait_action_count = 1,
        .acquire_actions =
            {
                LOOM_AMDGPU_SYSTEM_MEMORY_WAIT(
                    LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD),
                LOOM_AMDGPU_SYSTEM_MEMORY_PACKET(
                    LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_INV,
                    LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SCOPE),
            },
        .acquire_action_count = 2,
    },
    {
        .encoding =
            LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1,
        .load_attrs = LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SC0 |
                      LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SC1,
        .release_store_attrs = LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SC0 |
                               LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SC1,
        .no_return_atomic_attrs = LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SC1,
        .return_atomic_attrs = LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SC0 |
                               LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SC1,
        .release_actions =
            {
                LOOM_AMDGPU_SYSTEM_MEMORY_PACKET(
                    LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_WBL2,
                    LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SC0 |
                        LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SC1),
            },
        .release_action_count = 1,
        .load_wait_actions =
            {
                LOOM_AMDGPU_SYSTEM_MEMORY_WAIT(
                    LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD),
            },
        .load_wait_action_count = 1,
        .acquire_actions =
            {
                LOOM_AMDGPU_SYSTEM_MEMORY_WAIT(
                    LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD),
                LOOM_AMDGPU_SYSTEM_MEMORY_PACKET(
                    LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_INV,
                    LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SC0 |
                        LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SC1),
            },
        .acquire_action_count = 2,
    },
};

static const loom_amdgpu_system_memory_policy_t*
loom_amdgpu_system_memory_policy_lookup(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_vector_memory_cache_policy_encoding_t encoding =
      loom_amdgpu_memory_cache_policy_descriptor_encoding(descriptor_set);
  if (encoding == LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE) {
    return NULL;
  }
  const iree_host_size_t row_index = (iree_host_size_t)encoding - 1u;
  if (row_index < IREE_ARRAYSIZE(kAmdgpuSystemMemoryPolicies) &&
      kAmdgpuSystemMemoryPolicies[row_index].encoding == encoding) {
    return &kAmdgpuSystemMemoryPolicies[row_index];
  }
  return NULL;
}

static bool loom_amdgpu_system_memory_actions_available(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_system_memory_action_t* actions,
    iree_host_size_t action_count) {
  for (iree_host_size_t i = 0; i < action_count; ++i) {
    const loom_amdgpu_system_memory_action_t* action = &actions[i];
    switch (action->kind) {
      case LOOM_AMDGPU_SYSTEM_MEMORY_ACTION_NONE:
        break;
      case LOOM_AMDGPU_SYSTEM_MEMORY_ACTION_WAIT_COUNTER: {
        loom_amdgpu_wait_packet_selection_t selection = {0};
        if (!loom_amdgpu_wait_packet_try_select_counter_mask(
                descriptor_set, action->counter_mask, /*target_count=*/0,
                &selection)) {
          return false;
        }
        break;
      }
      case LOOM_AMDGPU_SYSTEM_MEMORY_ACTION_EXPLICIT_PACKET:
        if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                                action->descriptor_ref)) {
          return false;
        }
        break;
    }
  }
  return true;
}

bool loom_amdgpu_system_memory_release_ordering_available(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_system_memory_policy_t* policy =
      loom_amdgpu_system_memory_policy_lookup(descriptor_set);
  return policy != NULL && loom_amdgpu_system_memory_actions_available(
                               descriptor_set, policy->release_actions,
                               policy->release_action_count);
}

bool loom_amdgpu_system_memory_acquire_ordering_available(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_system_memory_policy_t* policy =
      loom_amdgpu_system_memory_policy_lookup(descriptor_set);
  return policy != NULL && loom_amdgpu_system_memory_actions_available(
                               descriptor_set, policy->acquire_actions,
                               policy->acquire_action_count);
}

static const loom_amdgpu_system_memory_policy_t*
loom_amdgpu_system_memory_policy_require(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_system_memory_policy_t* policy =
      loom_amdgpu_system_memory_policy_lookup(descriptor_set);
  if (policy == NULL) {
    IREE_ASSERT_UNREACHABLE("validated AMDGPU system-memory policy");
    IREE_BUILTIN_UNREACHABLE();
  }
  return policy;
}

static iree_status_t loom_amdgpu_system_memory_append_attrs(
    loom_builder_t* builder, loom_amdgpu_system_memory_attr_flags_t attr_flags,
    loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuSystemMemoryAttrFields); ++i) {
    const loom_amdgpu_system_memory_attr_field_t* field =
        &kAmdgpuSystemMemoryAttrFields[i];
    if (!iree_any_bit_set(attr_flags, field->flag)) {
      continue;
    }
    const uint32_t value = field->flag == LOOM_AMDGPU_SYSTEM_MEMORY_ATTR_SCOPE
                               ? (uint32_t)scope
                               : 1;
    IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_u32_attr(
        builder, field->name, value, attrs, attr_capacity, inout_attr_count));
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
  const loom_amdgpu_system_memory_policy_t* policy =
      loom_amdgpu_system_memory_policy_require(descriptor_set);
  return loom_amdgpu_system_memory_append_attrs(builder, policy->load_attrs,
                                                scope, attrs, attr_capacity,
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
  const loom_amdgpu_system_memory_policy_t* policy =
      loom_amdgpu_system_memory_policy_require(descriptor_set);
  return loom_amdgpu_system_memory_append_attrs(
      builder, policy->release_store_attrs, scope, attrs, attr_capacity,
      inout_attr_count);
}

iree_status_t loom_amdgpu_system_memory_append_no_return_atomic_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  return loom_amdgpu_system_memory_append_no_return_atomic_attrs_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs, attr_capacity,
      inout_attr_count);
}

iree_status_t loom_amdgpu_system_memory_append_no_return_atomic_attrs_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  const loom_amdgpu_system_memory_policy_t* policy =
      loom_amdgpu_system_memory_policy_require(descriptor_set);
  return loom_amdgpu_system_memory_append_attrs(
      builder, policy->no_return_atomic_attrs, scope, attrs, attr_capacity,
      inout_attr_count);
}

iree_status_t loom_amdgpu_system_memory_append_return_atomic_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  return loom_amdgpu_system_memory_append_return_atomic_attrs_scoped(
      builder, descriptor_set, LOOM_CACHE_SCOPE_SYSTEM, attrs, attr_capacity,
      inout_attr_count);
}

iree_status_t loom_amdgpu_system_memory_append_return_atomic_attrs_scoped(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_cache_scope_t scope, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  const loom_amdgpu_system_memory_policy_t* policy =
      loom_amdgpu_system_memory_policy_require(descriptor_set);
  return loom_amdgpu_system_memory_append_attrs(
      builder, policy->return_atomic_attrs, scope, attrs, attr_capacity,
      inout_attr_count);
}

static iree_status_t loom_amdgpu_system_memory_build_resolved_packet(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_named_attr_slice_t attrs,
    loom_location_id_t location) {
  loom_op_t* op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, /*operands=*/NULL,
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

static iree_status_t loom_amdgpu_system_memory_build_action(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_system_memory_action_t* action, loom_cache_scope_t scope,
    loom_location_id_t location) {
  switch (action->kind) {
    case LOOM_AMDGPU_SYSTEM_MEMORY_ACTION_NONE:
      return iree_ok_status();
    case LOOM_AMDGPU_SYSTEM_MEMORY_ACTION_WAIT_COUNTER:
      return loom_amdgpu_system_memory_build_wait_counter_mask(
          builder, descriptor_set, action->counter_mask, /*target_count=*/0,
          location);
    case LOOM_AMDGPU_SYSTEM_MEMORY_ACTION_EXPLICIT_PACKET:
      break;
  }
  loom_named_attr_t attrs[IREE_ARRAYSIZE(kAmdgpuSystemMemoryAttrFields)] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_attrs(
      builder, action->attr_flags, scope, attrs, IREE_ARRAYSIZE(attrs),
      &attr_count));
  return loom_amdgpu_system_memory_build_explicit_packet(
      builder, descriptor_set, action->descriptor_ref,
      loom_make_named_attr_slice(attrs, attr_count), location);
}

static iree_status_t loom_amdgpu_system_memory_build_actions(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_system_memory_action_t* actions,
    iree_host_size_t action_count, loom_cache_scope_t scope,
    loom_location_id_t location) {
  for (iree_host_size_t i = 0; i < action_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_action(
        builder, descriptor_set, &actions[i], scope, location));
  }
  return iree_ok_status();
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
  const loom_amdgpu_system_memory_policy_t* policy =
      loom_amdgpu_system_memory_policy_require(descriptor_set);
  return loom_amdgpu_system_memory_build_actions(
      builder, descriptor_set, policy->release_actions,
      policy->release_action_count, scope, location);
}

iree_status_t loom_amdgpu_system_memory_build_load_wait(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  const loom_amdgpu_system_memory_policy_t* policy =
      loom_amdgpu_system_memory_policy_require(descriptor_set);
  return loom_amdgpu_system_memory_build_actions(
      builder, descriptor_set, policy->load_wait_actions,
      policy->load_wait_action_count, LOOM_CACHE_SCOPE_SYSTEM, location);
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
  const loom_amdgpu_system_memory_policy_t* policy =
      loom_amdgpu_system_memory_policy_require(descriptor_set);
  return loom_amdgpu_system_memory_build_actions(
      builder, descriptor_set, policy->acquire_actions,
      policy->acquire_action_count, scope, location);
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
