// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet.h"

#include <inttypes.h>
#include <string.h>

#include "loom/ops/low/ops.h"

bool loom_low_packet_try_op_attrs(const loom_op_t* op,
                                  loom_named_attr_slice_t* out_attrs,
                                  uint16_t* out_attrs_attr_index) {
  if (out_attrs != NULL) {
    *out_attrs = loom_named_attr_slice_empty();
  }
  if (out_attrs_attr_index != NULL) {
    *out_attrs_attr_index = UINT16_MAX;
  }
  if (loom_low_op_isa(op)) {
    if (out_attrs != NULL) {
      *out_attrs = loom_low_op_attrs(op);
    }
    if (out_attrs_attr_index != NULL) {
      *out_attrs_attr_index = loom_low_op_attrs_ATTR_INDEX;
    }
    return true;
  }
  if (loom_low_const_isa(op)) {
    if (out_attrs != NULL) {
      *out_attrs = loom_low_const_attrs(op);
    }
    if (out_attrs_attr_index != NULL) {
      *out_attrs_attr_index = loom_low_const_attrs_ATTR_INDEX;
    }
    return true;
  }
  return false;
}

loom_named_attr_slice_t loom_low_packet_attrs(
    const loom_low_packet_view_t* packet) {
  if (packet == NULL || packet->node == NULL) {
    return loom_named_attr_slice_empty();
  }
  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  (void)loom_low_packet_try_op_attrs(packet->node->op, &attrs, NULL);
  return attrs;
}

static iree_status_t loom_low_packet_validate_asm_form_ordinal(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal, uint32_t asm_form_ordinal) {
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "selected asm form has no descriptor set");
  }
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_set_asm_form_at(descriptor_set, asm_form_ordinal);
  if (asm_form == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "selected asm form ordinal %" PRIu32
                            " is out of range",
                            asm_form_ordinal);
  }
  if (asm_form->descriptor_ordinal != descriptor_ordinal) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected asm form ordinal %" PRIu32 " belongs to descriptor %" PRIu32
        " instead of descriptor %" PRIu32,
        asm_form_ordinal, asm_form->descriptor_ordinal, descriptor_ordinal);
  }
  return iree_ok_status();
}

iree_status_t loom_low_packet_validate_asm_form_table(
    const loom_low_schedule_table_t* schedule,
    const loom_low_packet_asm_form_table_t* asm_forms) {
  if (schedule->module != asm_forms->module ||
      schedule->function_op != asm_forms->function_op) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected asm-form table must describe the scheduled low function");
  }
  if (schedule->target.descriptor_set != asm_forms->target.descriptor_set) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected asm-form table must use the schedule descriptor set");
  }
  if (asm_forms->asm_form_ordinal_count != schedule->scheduled_node_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "selected asm-form table has %" PRIhsz
                            " packet entries for %" PRIhsz " scheduled packets",
                            asm_forms->asm_form_ordinal_count,
                            schedule->scheduled_node_count);
  }
  if (asm_forms->asm_form_ordinal_count > 0 &&
      asm_forms->asm_form_ordinals == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected asm-form table entries are required for non-empty table");
  }
  for (iree_host_size_t packet_index = 0;
       packet_index < asm_forms->asm_form_ordinal_count; ++packet_index) {
    const uint32_t asm_form_ordinal =
        asm_forms->asm_form_ordinals[packet_index];
    if (asm_form_ordinal == LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
      continue;
    }
    const loom_low_packet_view_t packet =
        loom_low_packet_at(schedule, packet_index);
    const uint32_t descriptor_ordinal = packet.descriptor_ordinal;
    if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "selected asm form ordinal %" PRIu32
                              " was provided for structural "
                              "packet %" PRIhsz,
                              asm_form_ordinal, packet_index);
    }
    IREE_RETURN_IF_ERROR(loom_low_packet_validate_asm_form_ordinal(
        schedule->target.descriptor_set, descriptor_ordinal, asm_form_ordinal));
  }
  return iree_ok_status();
}

iree_status_t loom_low_packet_lookup_asm_form(
    const loom_low_schedule_table_t* schedule,
    const loom_low_packet_asm_form_table_t* asm_forms,
    const loom_low_packet_view_t* packet, uint32_t* out_asm_form_ordinal) {
  *out_asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  if (packet->node == NULL || packet->descriptor == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "packet has no descriptor-backed asm form");
  }
  const uint32_t descriptor_ordinal = packet->descriptor_ordinal;
  if (asm_forms != NULL) {
    if (schedule->module != asm_forms->module ||
        schedule->function_op != asm_forms->function_op ||
        schedule->target.descriptor_set != asm_forms->target.descriptor_set) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "selected asm-form table must match the scheduled low function");
    }
    if (packet->packet_index >= asm_forms->asm_form_ordinal_count ||
        asm_forms->asm_form_ordinals == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "selected asm-form table does not cover packet %" PRIhsz,
          packet->packet_index);
    }
    const uint32_t selected_asm_form_ordinal =
        asm_forms->asm_form_ordinals[packet->packet_index];
    if (selected_asm_form_ordinal != LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
      IREE_RETURN_IF_ERROR(loom_low_packet_validate_asm_form_ordinal(
          schedule->target.descriptor_set, descriptor_ordinal,
          selected_asm_form_ordinal));
      *out_asm_form_ordinal = selected_asm_form_ordinal;
      return iree_ok_status();
    }
  }
  *out_asm_form_ordinal = loom_low_descriptor_set_lookup_canonical_asm_form(
      schedule->target.descriptor_set, descriptor_ordinal);
  return iree_ok_status();
}

uint32_t loom_low_packet_block_index(const loom_low_schedule_table_t* schedule,
                                     const loom_block_t* block) {
  if (!schedule || !block) {
    return LOOM_LOW_PACKET_INDEX_NONE;
  }
  uint16_t block_index = 0;
  if (!loom_region_try_block_index(block->parent_region, block, &block_index) ||
      block_index >= schedule->block_count) {
    return LOOM_LOW_PACKET_INDEX_NONE;
  }
  return schedule->blocks[block_index].block == block
             ? block_index
             : LOOM_LOW_PACKET_INDEX_NONE;
}

uint32_t loom_low_packet_hazard_gap_packet_index(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_hazard_gap_t* hazard_gap,
    uint32_t scheduled_ordinal) {
  if (!schedule || !hazard_gap ||
      hazard_gap->block_index >= schedule->block_count) {
    return LOOM_LOW_PACKET_INDEX_NONE;
  }
  const loom_low_schedule_block_t* block =
      &schedule->blocks[hazard_gap->block_index];
  const uint64_t packet_index =
      (uint64_t)block->scheduled_node_start + scheduled_ordinal;
  return packet_index < LOOM_LOW_PACKET_INDEX_NONE ? (uint32_t)packet_index
                                                   : LOOM_LOW_PACKET_INDEX_NONE;
}
