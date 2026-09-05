// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_emit.h"

#include <stdint.h>

#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/fragment.h"
#include "loom/target/arch/amdgpu/lower/candidates/compare_candidates.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/encoding/float16.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_access.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_narrow.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_packet.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/util/fact_table.h"

static uint32_t loom_amdgpu_fragment_memory_packet_element_count(
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet) {
  return ((uint32_t)packet->result_register_count *
          plan->address_layout.payload_elements_per_register) /
         plan->address_layout.payload_registers_per_element;
}

typedef enum loom_amdgpu_fragment_memory_pending_store_payload_form_e {
  LOOM_AMDGPU_FRAGMENT_MEMORY_PENDING_STORE_PAYLOAD_FORM_F32 = 0,
  LOOM_AMDGPU_FRAGMENT_MEMORY_PENDING_STORE_PAYLOAD_FORM_B16 = 1,
} loom_amdgpu_fragment_memory_pending_store_payload_form_t;

typedef struct loom_amdgpu_fragment_memory_pending_store_t {
  // Packet plan copied from the immutable fragment memory plan.
  loom_amdgpu_fragment_memory_packet_plan_t packet;
  // Local result-fragment lane payload before final packing.
  loom_value_id_t low_source_register;
  // Cross-lane result-fragment payload paired with low_source_register.
  loom_value_id_t low_paired_source_register;
} loom_amdgpu_fragment_memory_pending_store_t;

static iree_status_t loom_amdgpu_emit_fragment_memory_packed_b16_load_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    loom_amdgpu_fragment_memory_address_state_t* address_state,
    loom_value_id_t low_packet_resource, loom_type_t vgpr_type,
    loom_value_id_t low_soffset, loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;
  if (plan->packed_b16_high_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    loom_amdgpu_fragment_memory_address_t low_address;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, plan, packet->register_index,
        /*element_index=*/0, packet->descriptor_ref, address_state, vgpr_type,
        &low_address));
    loom_value_id_t low_partial_packet = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_load_packet(
        context, source_op, layout, plan, packet, /*element_index=*/0,
        /*vector_lane_count=*/1, vgpr_type, &low_address, low_packet_resource,
        low_soffset, &low_partial_packet));

    loom_amdgpu_fragment_memory_address_t high_address;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, plan, packet->register_index,
        /*element_index=*/1, plan->packed_b16_high_descriptor_ref,
        address_state, vgpr_type, &high_address));
    return loom_amdgpu_emit_fragment_load_high_half_packet(
        context, source_op, layout, plan, packet, /*element_index=*/1,
        /*vector_lane_count=*/1, vgpr_type, &high_address, low_partial_packet,
        low_packet_resource, low_soffset, out_low_packet);
  }

  loom_value_id_t low_elements[LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT] =
      {0};
  for (uint16_t element_index = 0;
       element_index < LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
       ++element_index) {
    loom_amdgpu_fragment_memory_address_t address;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, plan, packet->register_index, element_index,
        packet->descriptor_ref, address_state, vgpr_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_load_packet(
        context, source_op, layout, plan, packet, element_index,
        /*vector_lane_count=*/1, vgpr_type, &address, low_packet_resource,
        low_soffset, &low_elements[element_index]));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_low_subword_load_packet(
            context, source_op, low_elements[element_index], vgpr_type,
            &low_elements[element_index]));
  }

  loom_value_id_t high_element = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
      low_elements[1], vgpr_type, &high_element));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_elements[0],
      high_element, vgpr_type, out_low_packet);
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_packed_16bit_result_load_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    loom_amdgpu_fragment_memory_address_state_t* address_state,
    loom_value_id_t low_packet_resource, loom_type_t vgpr_type,
    loom_value_id_t low_soffset, loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_elements[LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT] =
      {0};
  for (uint16_t i = 0; i < packet->result_register_count; ++i) {
    const loom_amdgpu_fragment_memory_packet_plan_t element_packet = {
        .flags = packet->flags,
        .register_index = (uint16_t)(packet->register_index + i),
        .result_register_count = 1,
        .packet_register_count = packet->packet_register_count,
        .descriptor_ref = packet->descriptor_ref,
    };
    loom_amdgpu_fragment_memory_address_t address;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, plan, element_packet.register_index,
        /*element_index=*/0, packet->descriptor_ref, address_state, vgpr_type,
        &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_load_packet(
        context, source_op, layout, plan, &element_packet, /*element_index=*/0,
        /*vector_lane_count=*/1, vgpr_type, &address, low_packet_resource,
        low_soffset, &low_elements[i]));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_low_subword_load_packet(
            context, source_op, low_elements[i], vgpr_type, &low_elements[i]));
  }
  if (packet->result_register_count == 1) {
    *out_low_packet = low_elements[0];
    return iree_ok_status();
  }

  loom_value_id_t high_element = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
      low_elements[1], vgpr_type, &high_element));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_elements[0],
      high_element, vgpr_type, out_low_packet);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_packed_b16_store_element(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_payload_register, uint16_t element_index,
    loom_type_t vgpr_type, loom_value_id_t* out_low_element) {
  *out_low_element = LOOM_VALUE_ID_INVALID;
  if (element_index == 0) {
    return loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, low_payload_register, out_low_element);
  }
  if (element_index == 1) {
    return loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
        low_payload_register, vgpr_type, out_low_element);
  }
  IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment packed b16 element");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_saveexec(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_type_t mask_type,
    loom_value_id_t* out_saved_exec) {
  *out_saved_exec = LOOM_VALUE_ID_INVALID;
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  const loom_type_t result_types[] = {mask_type, scc_type};
  loom_op_t* saveexec_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
      &low_condition, 1, loom_named_attr_slice_empty(), result_types,
      IREE_ARRAYSIZE(result_types), &saveexec_op));
  *out_saved_exec = loom_value_slice_get(loom_low_op_results(saveexec_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_restore_exec(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_saved_exec) {
  loom_op_t* low_op = NULL;
  return loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &low_saved_exec, 1, loom_named_attr_slice_empty(),
      /*result_types=*/NULL, /*result_count=*/0, &low_op);
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_crosslane_packed_b16_pair_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    const loom_low_lower_resolved_descriptor_t* crosslane_descriptor,
    loom_value_id_t low_paired_lane_byte_offset, loom_value_id_t low_source,
    loom_type_t vgpr_type, loom_value_id_t* out_paired_source) {
  *out_paired_source = LOOM_VALUE_ID_INVALID;
  if (iree_all_bits_set(
          packet->flags,
          LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE_DPP)) {
    return loom_amdgpu_emit_direct_crosslane_register(
        context, source_op, crosslane_descriptor, LOOM_AMDGPU_CROSSLANE_DPP,
        low_source, LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_1, vgpr_type,
        out_paired_source);
  }
  return loom_amdgpu_emit_subgroup_bpermute_register(
      context, source_op, crosslane_descriptor, low_paired_lane_byte_offset,
      /*static_byte_offset=*/0, low_source, vgpr_type, out_paired_source);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_apply_f32_scale(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, loom_value_id_t low_scale,
    loom_type_t vgpr_type, loom_value_id_t* out_scaled) {
  *out_scaled = low_source;
  if (low_scale == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32, low_source,
      low_scale, vgpr_type, out_scaled);
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_crosslane_packed_b16_prepare_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    const loom_low_lower_resolved_descriptor_t* crosslane_descriptor,
    loom_value_id_t low_paired_lane_byte_offset, loom_value_id_t low_payload,
    const loom_amdgpu_float16_pack_descriptors_t* pre_narrow_bf16_descriptors,
    loom_value_id_t low_scale, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_pending_store_t* out_pending_store) {
  *out_pending_store = (loom_amdgpu_fragment_memory_pending_store_t){
      .packet = *packet,
      .low_source_register = LOOM_VALUE_ID_INVALID,
      .low_paired_source_register = LOOM_VALUE_ID_INVALID,
  };
  loom_value_id_t low_source_register = LOOM_VALUE_ID_INVALID;
  const bool source_is_packed =
      plan->narrowed_result_packed_source != LOOM_VALUE_ID_INVALID;
  if (source_is_packed) {
    loom_value_id_t low_packed_register = low_payload;
    if (plan->payload_register_count != 1) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_payload, packet->register_index / 2u,
          vgpr_type, &low_packed_register));
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_packed_b16_store_element(
            context, source_op, low_packed_register,
            packet->register_index & 1u, vgpr_type, &low_source_register));
  } else {
    low_source_register = low_payload;
    if (plan->register_count != 1) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_payload, packet->register_index, vgpr_type,
          &low_source_register));
    }
  }
  if (pre_narrow_bf16_descriptors != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_apply_f32_scale(
        context, source_op, low_source_register, low_scale, vgpr_type,
        &low_source_register));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
        context, source_op, pre_narrow_bf16_descriptors, low_source_register,
        vgpr_type, &low_source_register));
  }
  out_pending_store->low_source_register = low_source_register;

  return loom_amdgpu_emit_fragment_memory_crosslane_packed_b16_pair_register(
      context, source_op, packet, crosslane_descriptor,
      low_paired_lane_byte_offset, low_source_register, vgpr_type,
      &out_pending_store->low_paired_source_register);
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_flush_crosslane_packed_b16_stores(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_float16_pack_descriptors_t* float16_pack_descriptors,
    loom_amdgpu_fragment_memory_address_state_t* address_state,
    const loom_amdgpu_fragment_memory_pending_store_t* pending_stores,
    iree_host_size_t pending_store_count,
    loom_amdgpu_fragment_memory_pending_store_payload_form_t payload_form,
    loom_value_id_t low_publishing_lane_mask, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t low_packet_resource,
    loom_value_id_t low_scale, loom_value_id_t low_paired_scale,
    loom_value_id_t low_soffset) {
  if (pending_store_count == 0) {
    return iree_ok_status();
  }
  const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor =
      iree_any_bit_set(float16_pack_descriptors->flags,
                       LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_PACK_U16)
          ? &float16_pack_descriptors->pack_u16_descriptor
          : NULL;

  loom_value_id_t low_saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_saveexec(
      context, source_op, low_publishing_lane_mask, mask_type,
      &low_saved_exec));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < pending_store_count && iree_status_is_ok(status); ++i) {
    const loom_amdgpu_fragment_memory_pending_store_t* pending_store =
        &pending_stores[i];
    loom_value_id_t low_source_register = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_paired_source_register = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_payload_packet = LOOM_VALUE_ID_INVALID;
    switch (payload_form) {
      case LOOM_AMDGPU_FRAGMENT_MEMORY_PENDING_STORE_PAYLOAD_FORM_F32:
        status = loom_amdgpu_emit_fragment_memory_apply_f32_scale(
            context, source_op, pending_store->low_source_register, low_scale,
            vgpr_type, &low_source_register);
        if (!iree_status_is_ok(status)) {
          break;
        }
        status = loom_amdgpu_emit_fragment_memory_apply_f32_scale(
            context, source_op, pending_store->low_paired_source_register,
            low_paired_scale, vgpr_type, &low_paired_source_register);
        if (!iree_status_is_ok(status)) {
          break;
        }
        if (plan->payload_form ==
            LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_F16) {
          status = loom_amdgpu_emit_f32_pair_to_packed_f16_with_descriptors(
              context, source_op, float16_pack_descriptors, low_source_register,
              low_paired_source_register, vgpr_type, &low_payload_packet);
        } else {
          IREE_ASSERT_EQ(
              plan->payload_form,
              LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16);
          status = loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
              context, source_op, float16_pack_descriptors, low_source_register,
              low_paired_source_register, vgpr_type, &low_payload_packet);
        }
        break;
      case LOOM_AMDGPU_FRAGMENT_MEMORY_PENDING_STORE_PAYLOAD_FORM_B16:
        status = loom_amdgpu_emit_packed_u16_lane_pair(
            context, source_op, pack_u16_descriptor,
            pending_store->low_source_register,
            pending_store->low_paired_source_register, vgpr_type,
            &low_payload_packet);
        break;
      default:
        IREE_ASSERT_UNREACHABLE(
            "selected AMDGPU fragment pending store payload form");
        IREE_BUILTIN_UNREACHABLE();
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    loom_amdgpu_fragment_memory_address_t address;
    status = loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, plan, pending_store->packet.register_index,
        /*element_index=*/0, pending_store->packet.descriptor_ref,
        address_state, vgpr_type, &address);
    if (!iree_status_is_ok(status)) {
      break;
    }
    status = loom_amdgpu_emit_fragment_store_packet(
        context, source_op, layout, plan, &pending_store->packet,
        /*element_index=*/0, LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT,
        &address, low_payload_packet, low_packet_resource, low_soffset);
  }
  return iree_status_join(status, loom_amdgpu_emit_fragment_memory_restore_exec(
                                      context, source_op, low_saved_exec));
}

static iree_status_t loom_amdgpu_emit_fragment_memory_packed_b16_store_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    loom_amdgpu_fragment_memory_address_state_t* address_state,
    loom_value_id_t low_payload, loom_value_id_t low_packet_resource,
    loom_type_t vgpr_type, loom_value_id_t low_soffset) {
  loom_value_id_t low_payload_register = low_payload;
  if (plan->register_count != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_payload, packet->register_index, vgpr_type,
        &low_payload_register));
  }
  for (uint16_t element_index = 0;
       element_index < LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
       ++element_index) {
    loom_value_id_t low_element = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_packed_b16_store_element(
            context, source_op, low_payload_register, element_index, vgpr_type,
            &low_element));
    loom_amdgpu_fragment_memory_address_t address;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, plan, packet->register_index, element_index,
        packet->descriptor_ref, address_state, vgpr_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_store_packet(
        context, source_op, layout, plan, packet, element_index,
        /*vector_lane_count=*/1, &address, low_element, low_packet_resource,
        low_soffset));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_fragment_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(plan->layout_kind);
  if (layout == NULL) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment memory layout");
    IREE_BUILTIN_UNREACHABLE();
  }
  IREE_ASSERT_GT(plan->packet_count, 0u);

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  const uint16_t lane_divisor = plan->address_layout.primary_lane_divisor;
  loom_amdgpu_matrix_fragment_lane_ids_t lane_ids;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_matrix_fragment_lane_ids(
      context, source_op, lane_divisor, vgpr_type, &lane_ids));
  const bool load_packed_16bit_result =
      plan->payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT;
  const bool load_fp8_to_16bit =
      loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
          plan->payload_form);
  loom_type_t mask_type = loom_type_none();
  if (load_fp8_to_16bit) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
  }

  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_packet_resource = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context,
        loom_low_source_memory_access_base_view_value_id(&plan->source),
        &low_resource));
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_resource(
        context, source_op, plan, low_resource, &low_packet_resource,
        &low_soffset));
  }
  loom_amdgpu_fragment_memory_address_state_t address_state;
  IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_fragment_memory_address_state(
      context, source_op, plan, &lane_ids, vgpr_type, &address_state));

  loom_value_id_t low_packets[LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS] =
      {0};
  for (uint16_t packet_index = 0; packet_index < plan->packet_count;
       ++packet_index) {
    const loom_amdgpu_fragment_memory_packet_plan_t* packet =
        &plan->packets[packet_index];
    if (load_fp8_to_16bit) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_fp8_to_packed_16bit_load_packet(
              context, source_op, layout, plan, packet, &address_state,
              low_packet_resource, vgpr_type, mask_type, low_soffset,
              &low_packets[packet_index]));
      continue;
    }
    if (load_packed_16bit_result) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_packed_16bit_result_load_packet(
              context, source_op, layout, plan, packet, &address_state,
              low_packet_resource, vgpr_type, low_soffset,
              &low_packets[packet_index]));
      continue;
    }
    if (plan->packetization ==
        LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_PACKED_B16) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_packed_b16_load_packet(
              context, source_op, layout, plan, packet, &address_state,
              low_packet_resource, vgpr_type, low_soffset,
              &low_packets[packet_index]));
      continue;
    }
    loom_type_t packet_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_type(
        context, packet->packet_register_count, vgpr_type, &packet_type));
    loom_amdgpu_fragment_memory_address_t address;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, plan, packet->register_index,
        /*element_index=*/0, packet->descriptor_ref, &address_state, vgpr_type,
        &address));
    const uint32_t vector_lane_count =
        plan->packetization ==
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_SCALAR_B16
            ? 1
            : loom_amdgpu_fragment_memory_packet_element_count(plan, packet);
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_load_packet(
        context, source_op, layout, plan, packet, /*element_index=*/0,
        vector_lane_count, packet_type, &address, low_packet_resource,
        low_soffset, &low_packets[packet_index]));
    if (plan->packetization ==
        LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_SCALAR_B16) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_low_subword_load_packet(
              context, source_op, low_packets[packet_index], vgpr_type,
              &low_packets[packet_index]));
    }
  }

  if (plan->packet_count == 1) {
    return loom_low_lower_bind_value(context, plan->payload, low_packets[0]);
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
      context, plan->payload_register_count, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_packets, plan->packet_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->payload,
                                   loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_lower_vector_fragment_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(plan->layout_kind);
  if (layout == NULL) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment memory layout");
    IREE_BUILTIN_UNREACHABLE();
  }
  IREE_ASSERT_GT(plan->packet_count, 0u);

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  const uint16_t lane_divisor = plan->address_layout.primary_lane_divisor;
  loom_amdgpu_matrix_fragment_lane_ids_t lane_ids;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_matrix_fragment_lane_ids(
      context, source_op, lane_divisor, vgpr_type, &lane_ids));
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_packet_resource = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context,
        loom_low_source_memory_access_base_view_value_id(&plan->source),
        &low_resource));
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_resource(
        context, source_op, plan, low_resource, &low_packet_resource,
        &low_soffset));
  }
  loom_amdgpu_fragment_memory_address_state_t address_state;
  IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_fragment_memory_address_state(
      context, source_op, plan, &lane_ids, vgpr_type, &address_state));

  if (loom_amdgpu_fragment_memory_payload_form_is_store_narrow_f32_to_16bit(
          plan->payload_form)) {
    loom_type_t mask_type = loom_type_none();
    const loom_amdgpu_float16_pack_descriptors_t* float16_pack_descriptors =
        NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_float16_pack_descriptors(
        context, &float16_pack_descriptors));
    const bool has_crosslane_packed_b16_store =
        loom_amdgpu_fragment_memory_epilogue_strategy_is_crosslane_packed_b16(
            plan->epilogue_strategy);
    const loom_matrix_fragment_packed_b16_publication_t* publication =
        plan->packed_b16_publication;
    IREE_ASSERT_TRUE(!has_crosslane_packed_b16_store ||
                     (publication != NULL &&
                      publication->publishing_participant_and_mask != 0));
    const bool has_dpp_crosslane_packed_b16_store =
        loom_amdgpu_fragment_memory_epilogue_strategy_uses_dpp(
            plan->epilogue_strategy);
    const bool pre_narrow_crosslane_sources =
        has_crosslane_packed_b16_store &&
        plan->payload_form ==
            LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16 &&
        plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID &&
        !iree_any_bit_set(
            float16_pack_descriptors->flags,
            LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE_BF16);
    const bool copy_packed_crosslane_sources =
        has_crosslane_packed_b16_store &&
        plan->narrowed_result_packed_source != LOOM_VALUE_ID_INVALID;
    const loom_amdgpu_float16_pack_descriptors_t* pre_narrow_bf16_descriptors =
        pre_narrow_crosslane_sources ? float16_pack_descriptors : NULL;
    const loom_amdgpu_fragment_memory_pending_store_payload_form_t
        pending_store_payload_form =
            pre_narrow_crosslane_sources || copy_packed_crosslane_sources
                ? LOOM_AMDGPU_FRAGMENT_MEMORY_PENDING_STORE_PAYLOAD_FORM_B16
                : LOOM_AMDGPU_FRAGMENT_MEMORY_PENDING_STORE_PAYLOAD_FORM_F32;
    if (has_crosslane_packed_b16_store) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
    }
    loom_value_id_t low_paired_lane_byte_offset = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_publishing_lane_mask = LOOM_VALUE_ID_INVALID;
    loom_low_lower_resolved_descriptor_t crosslane_descriptor = {0};
    if (has_crosslane_packed_b16_store) {
      if (has_dpp_crosslane_packed_b16_store) {
        const loom_low_descriptor_set_t* descriptor_set =
            loom_low_lower_context_descriptor_set(context);
        const loom_amdgpu_descriptor_ref_t dpp_ref =
            loom_amdgpu_select_dpp_descriptor_ref(descriptor_set);
        IREE_ASSERT_NE(dpp_ref, LOOM_AMDGPU_DESCRIPTOR_REF_NONE);
        IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
            context, dpp_ref, &crosslane_descriptor));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
            context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
            &crosslane_descriptor));
        loom_value_id_t low_paired_lane = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT,
            lane_ids.lane, publication->paired_participant_xor_mask, vgpr_type,
            &low_paired_lane));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
            context, source_op, low_paired_lane, vgpr_type,
            &low_paired_lane_byte_offset));
      }
      loom_value_id_t low_publishing_participant_selector =
          LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
          lane_ids.lane, publication->publishing_participant_and_mask,
          vgpr_type, &low_publishing_participant_selector));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_compare_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
          kLoomAmdgpuVectorCmpiCompareDescriptorCandidates
              [LOOM_VECTOR_CMPI_PREDICATE_EQ]
                  .src1_inline_descriptor_ref,
          low_publishing_participant_selector,
          publication->publishing_participant_equal_value, vgpr_type, mask_type,
          &low_publishing_lane_mask));
    }

    loom_value_id_t low_payload = LOOM_VALUE_ID_INVALID;
    if (plan->narrowed_result_packed_source != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, plan->narrowed_result_packed_source, &low_payload));
    } else if (plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, plan->narrowed_result_round_source, &low_payload));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->payload, &low_payload));
    }
    loom_value_id_t low_scale = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_paired_scale = LOOM_VALUE_ID_INVALID;
    if (plan->narrowed_result_scale_source != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, plan->narrowed_result_scale_source, &low_scale));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
          context, source_op, low_scale, &low_scale));
      low_paired_scale = low_scale;
    }

    loom_amdgpu_fragment_memory_pending_store_t
        pending_stores[LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS] = {0};
    iree_host_size_t pending_store_count = 0;
    for (uint16_t packet_index = 0; packet_index < plan->packet_count;
         ++packet_index) {
      const loom_amdgpu_fragment_memory_packet_plan_t* packet =
          &plan->packets[packet_index];
      if (iree_all_bits_set(
              packet->flags,
              LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE)) {
        IREE_ASSERT_LT(pending_store_count, IREE_ARRAYSIZE(pending_stores));
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_memory_crosslane_packed_b16_prepare_store(
                context, source_op, plan, packet, &crosslane_descriptor,
                low_paired_lane_byte_offset, low_payload,
                pre_narrow_bf16_descriptors, low_scale, vgpr_type,
                &pending_stores[pending_store_count++]));
        continue;
      }

      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_flush_crosslane_packed_b16_stores(
              context, source_op, layout, plan, float16_pack_descriptors,
              &address_state, pending_stores, pending_store_count,
              pending_store_payload_form, low_publishing_lane_mask, vgpr_type,
              mask_type, low_packet_resource, low_scale, low_paired_scale,
              low_soffset));
      pending_store_count = 0;

      loom_value_id_t low_payload_packet = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_packed_16bit_packet(
          context, source_op, plan, low_payload, float16_pack_descriptors,
          packet->register_index, packet->result_register_count,
          packet->packet_register_count, low_scale, vgpr_type,
          &low_payload_packet));
      loom_amdgpu_fragment_memory_address_t address;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
          context, source_op, plan, packet->register_index,
          /*element_index=*/0, packet->descriptor_ref, &address_state,
          vgpr_type, &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_store_packet(
          context, source_op, layout, plan, packet, /*element_index=*/0,
          packet->result_register_count, &address, low_payload_packet,
          low_packet_resource, low_soffset));
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_flush_crosslane_packed_b16_stores(
            context, source_op, layout, plan, float16_pack_descriptors,
            &address_state, pending_stores, pending_store_count,
            pending_store_payload_form, low_publishing_lane_mask, vgpr_type,
            mask_type, low_packet_resource, low_scale, low_paired_scale,
            low_soffset));
    return iree_ok_status();
  }

  loom_value_id_t low_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->payload, &low_payload));

  if (plan->packetization ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_PACKED_B16) {
    for (uint16_t packet_index = 0; packet_index < plan->packet_count;
         ++packet_index) {
      const loom_amdgpu_fragment_memory_packet_plan_t* packet =
          &plan->packets[packet_index];
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_packed_b16_store_packet(
              context, source_op, layout, plan, packet, &address_state,
              low_payload, low_packet_resource, vgpr_type, low_soffset));
    }
    return iree_ok_status();
  }

  for (uint16_t packet_index = 0; packet_index < plan->packet_count;
       ++packet_index) {
    const loom_amdgpu_fragment_memory_packet_plan_t* packet =
        &plan->packets[packet_index];
    loom_value_id_t low_payload_packet = low_payload;
    if (plan->payload_form ==
        LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_EXTEND_F16_TO_F32) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_f16_to_f32_packet(
          context, source_op, plan, low_payload, packet, vgpr_type,
          &low_payload_packet));
    } else if (packet->result_register_count != plan->register_count) {
      loom_type_t packet_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_type(
          context, packet->packet_register_count, vgpr_type, &packet_type));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_payload, packet->register_index, packet_type,
          &low_payload_packet));
    }
    loom_amdgpu_fragment_memory_address_t address;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, plan, packet->register_index,
        /*element_index=*/0, packet->descriptor_ref, &address_state, vgpr_type,
        &address));
    const uint32_t vector_lane_count =
        loom_amdgpu_fragment_memory_packet_element_count(plan, packet);
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_store_packet(
        context, source_op, layout, plan, packet, /*element_index=*/0,
        vector_lane_count, &address, low_payload_packet, low_packet_resource,
        low_soffset));
  }
  return iree_ok_status();
}

void loom_amdgpu_mark_fragment_memory_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  (void)source_op;
  if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    loom_low_lower_require_source_value_storage(
        context,
        loom_low_source_memory_access_base_view_value_id(&plan->source));
  }
  for (uint8_t i = 0; i < plan->source.dynamic_term_count; ++i) {
    if (loom_amdgpu_fragment_memory_uses_dynamic_view_base_value(plan, i)) {
      loom_low_lower_require_source_value_storage(
          context, plan->source.dynamic_view_base_value_id);
      continue;
    }
    const loom_low_source_memory_dynamic_term_t* term =
        &plan->source.dynamic_terms[i];
    loom_low_lower_require_source_value_storage(context, term->index);
    for (uint8_t j = 0; j < term->stride_value_count; ++j) {
      loom_low_lower_require_source_value_storage(context,
                                                  term->stride_values[j]);
    }
  }
  for (uint8_t view_axis = 0; view_axis < plan->view_rank; ++view_axis) {
    const loom_low_source_memory_axis_byte_stride_t* byte_stride =
        &plan->runtime_axes[view_axis].byte_stride;
    for (uint8_t j = 0; j < byte_stride->dynamic_factor_count; ++j) {
      loom_low_lower_require_source_value_storage(
          context, byte_stride->dynamic_factors[j]);
    }
  }

  if (plan->operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD) {
    if (plan->fp8_load_scale_source != LOOM_VALUE_ID_INVALID) {
      loom_low_lower_require_source_value_storage(context,
                                                  plan->fp8_load_scale_source);
    }
    return;
  }
  if (plan->operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE) {
    if (plan->narrowed_result_packed_source != LOOM_VALUE_ID_INVALID) {
      loom_low_lower_require_source_value_storage(
          context, plan->narrowed_result_packed_source);
    } else if (plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID) {
      loom_low_lower_require_source_value_storage(
          context, plan->narrowed_result_round_source);
    } else {
      loom_low_lower_require_source_value_storage(context, plan->payload);
    }
    return;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU fragment memory operation kind");
}
