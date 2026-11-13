// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/function_envelope.h"

#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_encoding.h"

static void loom_aie2p_function_envelope_append_stack(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_aie2p_stack_frame_t* stack, bool entering,
    loom_aie2p_function_envelope_sequence_t* sequence) {
  const bool initialize = entering && stack->storage.byte_length != 0;
  if (!initialize && stack->byte_length == 0) {
    return;
  }
  const uint32_t ordinal = initialize
                               ? AIE2P_CORE_DESCRIPTOR_REF_STATE_STACK_POINTER
                               : AIE2P_CORE_DESCRIPTOR_REF_STACK_ADJUST;
  const int64_t displacement =
      entering ? (int64_t)stack->byte_length : -(int64_t)stack->byte_length;
  sequence->instructions[sequence->count++] =
      (loom_aie2p_function_envelope_instruction_t){
          .slot =
              {
                  .encoded_slot =
                      loom_aie2p_bundle_encoding_encode_state_immediate(
                          descriptor_set, ordinal,
                          initialize ? 0 : displacement),
                  .scheduled_packet_index = LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
                  .flags =
                      initialize
                          ? LOOM_AIE2P_PLANNED_SLOT_FLAG_STACK_INITIALIZATION
                          : 0,
              },
          .realization = {.descriptor_ordinal = ordinal},
      };
}

static void loom_aie2p_function_envelope_append_link(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_aie2p_physical_register_id_t shadow_register, bool save,
    loom_aie2p_function_envelope_sequence_t* sequence) {
  const loom_aie2p_physical_register_id_t link_register =
      loom_aie2p_machine_find_physical_register(IREE_SV("lr"));
  const loom_aie2p_register_move_t move = {
      .source = save ? link_register : shadow_register,
      .destination = save ? shadow_register : link_register,
  };
  const loom_aie2p_move_route_id_t route =
      loom_aie2p_descriptor_resolve_move_route(move.source, move.destination);
  IREE_ASSERT(route != LOOM_AIE2P_MOVE_ROUTE_ID_INVALID &&
              loom_aie2p_descriptor_move_route_step_count(route) == 1 &&
              "AIE2P link transport must have one native route step");
  const loom_aie2p_move_route_step_t step =
      loom_aie2p_descriptor_move_route_step(route, move, 0);
  sequence->instructions[sequence->count++] =
      (loom_aie2p_function_envelope_instruction_t){
          .slot =
              {
                  .encoded_slot = loom_aie2p_descriptor_encode_move_step(
                      descriptor_set, step),
                  .scheduled_packet_index = LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
                  .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_MOVE,
                  .move_step_profile_id = step.profile_id,
              },
          .realization = {.move = step.move},
      };
}

static void loom_aie2p_function_envelope_append_shadow(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_aie2p_physical_register_id_t shadow_register, bool save,
    loom_aie2p_function_envelope_sequence_t* sequence) {
  const uint32_t ordinal = save ? AIE2P_CORE_DESCRIPTOR_REF_STACK_STORE_ER
                                : AIE2P_CORE_DESCRIPTOR_REF_STACK_LOAD_ER;
  const loom_low_allocation_assignment_t assignment = {
      .descriptor_reg_class_id = AIE2P_CORE_REG_CLASS_ID_AIE2P_ER,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = shadow_register,
      .location_count = 1,
      .unit_count = 1,
  };
  const loom_low_allocation_assignment_t* operands[] = {&assignment, NULL};
  const int64_t displacement[] = {-LOOM_AIE2P_RETURN_ADDRESS_FRAME_BYTES};
  sequence->instructions[sequence->count++] =
      (loom_aie2p_function_envelope_instruction_t){
          .slot =
              {
                  .encoded_slot = loom_aie2p_descriptor_encode(
                      descriptor_set, ordinal, operands, displacement),
                  .scheduled_packet_index = LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
                  .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_LINK_SHADOW_STORAGE,
              },
          .realization = {.descriptor_ordinal = ordinal},
      };
}

void loom_aie2p_function_envelope_build(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_aie2p_stack_frame_t* stack,
    loom_aie2p_return_address_t return_address,
    loom_aie2p_function_envelope_t* out_envelope) {
  *out_envelope = (loom_aie2p_function_envelope_t){0};
  loom_aie2p_function_envelope_append_stack(descriptor_set, stack, true,
                                            &out_envelope->entry);
  if (return_address.kind == LOOM_AIE2P_RETURN_ADDRESS_FRAME) {
    loom_aie2p_function_envelope_append_shadow(descriptor_set,
                                               return_address.shadow_register,
                                               true, &out_envelope->entry);
  }
  if (return_address.kind != LOOM_AIE2P_RETURN_ADDRESS_NONE) {
    loom_aie2p_function_envelope_append_link(descriptor_set,
                                             return_address.shadow_register,
                                             true, &out_envelope->entry);
  }
  // Frame-backed preservation must consume both saved values before releasing
  // its frame. Register-only entries retain the ordinary stack-first epilogue.
  if (return_address.kind != LOOM_AIE2P_RETURN_ADDRESS_FRAME) {
    loom_aie2p_function_envelope_append_stack(descriptor_set, stack, false,
                                              &out_envelope->exit);
  }
  if (return_address.kind != LOOM_AIE2P_RETURN_ADDRESS_NONE) {
    loom_aie2p_function_envelope_append_link(descriptor_set,
                                             return_address.shadow_register,
                                             false, &out_envelope->exit);
  }
  if (return_address.kind == LOOM_AIE2P_RETURN_ADDRESS_FRAME) {
    loom_aie2p_function_envelope_append_shadow(descriptor_set,
                                               return_address.shadow_register,
                                               false, &out_envelope->exit);
    loom_aie2p_function_envelope_append_stack(descriptor_set, stack, false,
                                              &out_envelope->exit);
  }
}
