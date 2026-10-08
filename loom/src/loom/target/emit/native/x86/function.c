// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/function.h"

#include "iree/base/internal/math.h"
#include "loom/codegen/low/allocation/unit_location.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/error/x86_error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/register_classes.h"
#include "loom/target/emit/native/x86/transport.h"

enum {
  LOOM_X86_INDIRECT_RESULT_POINTER_REGISTER = 10,
  LOOM_X86_GPR_TRANSPORT_SCRATCH_REGISTER = 11,
  LOOM_X86_VECTOR_TRANSPORT_SCRATCH_REGISTER = 8,
  LOOM_X86_RETAINED_RESULT_POINTER_REGISTER = 15,
};

typedef struct loom_x86_incoming_fixup_t {
  // Saved pre-alignment pointer load, or UINT32_MAX with static alignment.
  uint32_t pointer_instruction;
  // Incoming argument load whose displacement is relative to caller RSP.
  uint32_t value_instruction;
} loom_x86_incoming_fixup_t;

// Function-local projection of generic storage onto the native stack.
typedef struct loom_x86_function_builder_t {
  // Prepared output under construction.
  loom_x86_function_t* function;
  // Source operation being prepared, used only for admission diagnostics.
  const loom_op_t* source_op;
  // Optional destination for source diagnostics; sink failures propagate.
  iree_diagnostic_emitter_t emitter;
  // Rejection is independent of whether a diagnostic sink is installed.
  bool rejected;
  // Retained SysV plan for the function being emitted.
  const loom_x86_function_abi_t* function_abi;
  // Retained SysV plans for callees in this module.
  const loom_x86_module_abi_t* module_abi;
  // Writable call-cleanup rows owned by the prepared function arena.
  uint32_t* upper_vector_call_cleanup_indices;
  // Native RSP-relative base of each generic storage space.
  uint64_t storage_offsets[LOOM_STORAGE_SPACE_COUNT_];
  // RSP-relative byte offsets indexed by final move-storage cell ordinal.
  uint64_t* move_storage_offsets;
  // Combined byte extent before ABI padding and restoration storage.
  uint64_t storage_byte_length;
  // Strongest alignment among the projected spaces.
  uint64_t storage_alignment;
  // Invocation-only loads fixed after final callee-save/frame selection.
  struct {
    // Exact load indices retained as each incoming transfer is constructed.
    loom_x86_incoming_fixup_t* fixups;
    // Number of initialized incoming argument transfers.
    iree_host_size_t count;
  } incoming;
} loom_x86_function_builder_t;

static iree_status_t loom_x86_function_reject(
    loom_x86_function_builder_t* builder, const loom_error_def_t* error,
    iree_string_view_t constraint) {
  builder->rejected = true;
  const loom_diagnostic_param_t params[] = {loom_param_string(constraint)};
  return iree_diagnostic_emit(builder->emitter,
                              &(loom_diagnostic_emission_t){
                                  .op = builder->source_op,
                                  .error = error,
                                  .params = params,
                                  .param_count = IREE_ARRAYSIZE(params),
                              });
}

// Bound each component before arithmetic. The native stack has signed 32-bit
// displacements, so accepted components cannot overflow the 64-bit layout sum.
static iree_status_t loom_x86_function_append_storage(
    uint64_t byte_length, uint64_t byte_alignment,
    loom_x86_function_builder_t* builder, uint64_t* out_offset) {
  if (byte_length > INT32_MAX || byte_alignment > INT32_MAX) {
    return loom_x86_function_reject(
        builder, LOOM_ERR_X86_004,
        IREE_SV("stack storage and alignment within signed 32-bit addressing"));
  }
  const uint64_t offset =
      iree_host_align(builder->storage_byte_length, byte_alignment);
  if (offset + byte_length > INT32_MAX) {
    return loom_x86_function_reject(
        builder, LOOM_ERR_X86_004,
        IREE_SV("stack storage and alignment within signed 32-bit addressing"));
  }
  *out_offset = offset;
  builder->storage_byte_length = offset + byte_length;
  builder->storage_alignment =
      iree_max(builder->storage_alignment, byte_alignment);
  return iree_ok_status();
}

iree_host_size_t loom_x86_function_reserved_ranges(
    const loom_x86_function_abi_t* function_abi,
    loom_low_allocation_reserved_range_t out_ranges[3]) {
  const loom_low_descriptor_set_t* descriptor_set =
      function_abi->target.descriptor_set;
  if (LOOM_X86_REGISTER_CLASS_GPR64 >= descriptor_set->reg_class_count ||
      descriptor_set->reg_classes[LOOM_X86_REGISTER_CLASS_GPR64]
              .allocatable_count <= 4) {
    return 0;
  }
  out_ranges[0] = (loom_low_allocation_reserved_range_t){
      .register_class = IREE_SV("x86.gpr64"),
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = 4,
      .location_count = 1,
  };
  iree_host_size_t count = 1;
  if (function_abi->has_simd_stack_argument &&
      descriptor_set->reg_classes[LOOM_X86_REGISTER_CLASS_GPR64]
              .allocatable_count > 5) {
    out_ranges[count++] = (loom_low_allocation_reserved_range_t){
        .register_class = IREE_SV("x86.gpr64"),
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .location_base = 5,
        .location_count = 1,
    };
  }
  if (function_abi->has_indirect_results &&
      descriptor_set->reg_classes[LOOM_X86_REGISTER_CLASS_GPR64]
              .allocatable_count > 15) {
    out_ranges[count++] = (loom_low_allocation_reserved_range_t){
        .register_class = IREE_SV("x86.gpr64"),
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .location_base = LOOM_X86_RETAINED_RESULT_POINTER_REGISTER,
        .location_count = 1,
    };
  }
  return count;
}

static iree_status_t loom_x86_function_storage_layout(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_x86_function_builder_t* builder) {
  const loom_low_storage_layout_t* layout =
      &frame->schedule.requirements.storage_layout;
  if (layout->space_sizes.workgroup_bytes) {
    return loom_x86_function_reject(
        builder, LOOM_ERR_X86_004,
        IREE_SV("stack, scratch, or private storage; host functions have no "
                "workgroup storage ABI"));
  }
  static const loom_storage_space_t spaces[] = {
      LOOM_STORAGE_SPACE_STACK,
      LOOM_STORAGE_SPACE_SCRATCH,
      LOOM_STORAGE_SPACE_PRIVATE,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(spaces) && !builder->rejected;
       ++i) {
    const loom_low_storage_layout_requirement_t requirement =
        loom_low_storage_layout_requirement(layout, spaces[i]);
    if (requirement.byte_length == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_x86_function_append_storage(
        requirement.byte_length, requirement.minimum_alignment, builder,
        &builder->storage_offsets[spaces[i]]));
  }
  const loom_low_allocation_table_t* allocation = &frame->allocation;
  if (!builder->rejected && allocation->move_storage_count) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, allocation->move_storage_count,
                                  sizeof(*builder->move_storage_offsets),
                                  (void**)&builder->move_storage_offsets));
    for (iree_host_size_t i = 0;
         i < allocation->move_storage_count && !builder->rejected; ++i) {
      const loom_low_move_storage_t* cell = &allocation->move_storage[i];
      IREE_RETURN_IF_ERROR(loom_x86_function_append_storage(
          cell->byte_length, cell->byte_alignment, builder,
          &builder->move_storage_offsets[i]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_function_stack_frame(
    loom_x86_function_builder_t* builder) {
  loom_x86_function_t* function = builder->function;
  if (builder->storage_byte_length == 0 && builder->storage_alignment == 0 &&
      builder->incoming.count == 0) {
    return iree_ok_status();
  }
  const uint64_t alignment = iree_max(builder->storage_alignment, 8u);
  function->stack.alignment = (uint32_t)alignment;
  uint64_t allocation_size = builder->storage_byte_length;
  if (alignment > 16) {
    function->stack.realignment.mask = -(int32_t)alignment;
    if (!function->stack.has_frame_pointer) {
      // Storage admission bounds the extent to INT32_MAX, leaving ample room
      // in this widened calculation for the restoration slot and padding.
      const uint64_t saved_pointer_offset = iree_host_align(allocation_size, 8);
      allocation_size = saved_pointer_offset + sizeof(uint64_t);
      function->stack.realignment.saved_pointer_offset =
          (uint32_t)saved_pointer_offset;
      // SysV integer arguments occupy RDI, RSI, RDX, RCX, R8, and R9. R11 is
      // available before entry transport; no SSA lifetime is pinned to it.
      function->stack.realignment.scratch_register = 11;
    }
    allocation_size = iree_host_align(allocation_size, alignment);
  } else {
    // Entry RSP is 8 modulo 16 because CALL has pushed the return address.
    // Include all callee-save pushes when padding the local allocation.
    const uint64_t incoming_bytes =
        sizeof(uint64_t) *
        (1u + iree_math_count_ones_u32(function->saved_registers));
    allocation_size =
        iree_host_align(allocation_size + incoming_bytes, alignment) -
        incoming_bytes;
  }
  if (allocation_size > INT32_MAX) {
    return loom_x86_function_reject(
        builder, LOOM_ERR_X86_004,
        IREE_SV("a stack adjustment within signed 32 bits"));
  }
  function->stack.allocation_size = (uint32_t)allocation_size;
  const uint32_t incoming_offset =
      8u * (1u + iree_math_count_ones_u32(function->saved_registers));
  for (iree_host_size_t i = 0; i < builder->incoming.count; ++i) {
    const loom_x86_incoming_fixup_t* fixup = &builder->incoming.fixups[i];
    loom_x86_instruction_t* load =
        &function->instructions[fixup->value_instruction];
    if (function->stack.has_frame_pointer) {
      load->operands.immediate += incoming_offset;
    } else if (fixup->pointer_instruction != UINT32_MAX) {
      function->instructions[fixup->pointer_instruction].operands.immediate =
          function->stack.realignment.saved_pointer_offset;
      load->operands.immediate += incoming_offset;
    } else {
      load->operands.immediate +=
          incoming_offset + function->stack.allocation_size;
    }
  }
  return iree_ok_status();
}

static void loom_x86_function_append(loom_x86_function_t* function,
                                     uint16_t encoding_format_id,
                                     uint16_t encoding_id,
                                     loom_x86_encoding_operands_t operands,
                                     uint32_t reference) {
  function->instructions[function->instruction_count++] =
      (loom_x86_instruction_t){
          .operands = operands,
          .reference = reference,
          .encoding_format_id = encoding_format_id,
          .encoding_id = encoding_id,
      };
  // SysV RBX, RBP, R12..R15 are callee-preserved. Reads, unused entry
  // arguments, and coalesced moves require no preservation.
  const uint16_t preserved = (1u << 3) | (1u << 5) | (0xfu << 12);
  if (!(encoding_format_id & LOOM_X86_ENCODING_FORMAT_VECTOR)) {
    function->saved_registers |=
        loom_x86_encoding_gpr_writes(
            (loom_x86_encoding_form_t)encoding_format_id, &operands) &
        preserved;
  }
}

static void loom_x86_function_normalize(loom_x86_function_t* function,
                                        loom_x86_call_abi_value_action_t action,
                                        uint32_t register_location) {
  uint32_t mask = 0;
  switch (action) {
    case LOOM_X86_CALL_ABI_VALUE_ACTION_NONE:
      return;
    case LOOM_X86_CALL_ABI_VALUE_ACTION_NORMALIZE_I1:
      mask = 1;
      break;
    case LOOM_X86_CALL_ABI_VALUE_ACTION_NORMALIZE_I8:
      mask = UINT8_MAX;
      break;
    case LOOM_X86_CALL_ABI_VALUE_ACTION_NORMALIZE_I16:
      mask = UINT16_MAX;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("unknown x86 ABI value action");
      return;
  }
  loom_x86_function_append(
      function, LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE, 0x81 | (4u << 9),
      (loom_x86_encoding_operands_t){.immediate = mask,
                                     .result = (uint8_t)register_location},
      UINT32_MAX);
}

static void loom_x86_function_normalize_arguments(
    loom_x86_function_builder_t* builder) {
  const loom_low_call_contract_t* contract =
      &builder->function_abi->call_contract;
  for (uint16_t i = 0; i < contract->argument_count; ++i) {
    const loom_low_allocation_abi_location_t* location =
        &contract->arguments[i];
    if (location->location_kind ==
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      loom_x86_function_normalize(
          builder->function,
          (loom_x86_call_abi_value_action_t)builder->function_abi->arguments[i]
              .action,
          location->location_base);
    }
  }
}

static uint8_t loom_x86_function_register(
    const loom_low_allocation_assignment_t* assignment) {
  return loom_x86_transport_register(assignment->descriptor_reg_class_id,
                                     assignment->location_base);
}

static void loom_x86_function_note_upper_vector_state(
    loom_x86_function_t* function, uint16_t descriptor_reg_class_id,
    uint32_t location) {
  const loom_x86_register_class_t register_class =
      loom_x86_logical_register_class(descriptor_reg_class_id);
  if ((register_class == LOOM_X86_REGISTER_CLASS_YMM ||
       register_class == LOOM_X86_REGISTER_CLASS_ZMM) &&
      location < 16) {
    function->may_dirty_upper_vector_state = true;
  }
}

static void loom_x86_function_append_transport(
    loom_x86_function_t* function,
    const loom_x86_transport_instruction_t* instruction) {
  if (instruction->encoding_format_id) {
    loom_x86_function_append(function, instruction->encoding_format_id,
                             instruction->encoding_id, instruction->operands,
                             UINT32_MAX);
    // Vector-form transport may write a GPR, which the generic append path
    // cannot derive from a packed vector recipe.
    const uint16_t preserved = (1u << 3) | (1u << 5) | (0xfu << 12);
    function->saved_registers |= instruction->gpr_writes & preserved;
    function->may_dirty_upper_vector_state |=
        instruction->may_dirty_upper_vector_state;
  }
}

static iree_status_t loom_x86_function_move(
    loom_x86_function_builder_t* builder, uint16_t destination_class,
    uint32_t destination, uint16_t source_class, uint32_t source) {
  loom_x86_transport_instruction_t instruction;
  if (!loom_x86_transport_select_register(destination_class, destination,
                                          source_class, source, &instruction)) {
    return loom_x86_function_reject(
        builder, LOOM_ERR_X86_004,
        IREE_SV("compatible physical register representations"));
  }
  loom_x86_function_append_transport(builder->function, &instruction);
  return iree_ok_status();
}

static iree_status_t loom_x86_function_moves(
    const loom_low_allocation_table_t* allocation, loom_low_move_range_t range,
    loom_x86_function_builder_t* builder) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < range.count && !builder->rejected && iree_status_is_ok(status);
       ++i) {
    const loom_low_move_t* move = &allocation->moves[range.start + i];
    const bool is_store =
        move->destination.location_kind ==
            LOOM_LOW_ALLOCATION_LOCATION_MOVE_STORAGE ||
        move->destination.location_kind == LOOM_LOW_ALLOCATION_LOCATION_STORAGE;
    const bool is_load =
        move->source.location_kind ==
            LOOM_LOW_ALLOCATION_LOCATION_MOVE_STORAGE ||
        move->source.location_kind == LOOM_LOW_ALLOCATION_LOCATION_STORAGE;
    if (is_store || is_load) {
      const loom_low_move_location_t* cell =
          is_store ? &move->destination : &move->source;
      const loom_low_move_location_t* reg =
          is_store ? &move->source : &move->destination;
      uint64_t byte_offset = 0;
      if (cell->location_kind == LOOM_LOW_ALLOCATION_LOCATION_STORAGE) {
        const loom_low_storage_transport_binding_t* binding =
            &allocation->storage_transport->bindings[cell->location];
        byte_offset =
            builder->storage_offsets[binding->space] + binding->byte_offset;
      } else {
        byte_offset = builder->move_storage_offsets[cell->location];
      }
      loom_x86_transport_instruction_t instruction;
      loom_x86_transport_select_storage_register(
          is_store ? LOOM_X86_STORAGE_TRANSFER_STORE
                   : LOOM_X86_STORAGE_TRANSFER_LOAD,
          cell->descriptor_reg_class_id, reg->descriptor_reg_class_id,
          reg->location, 4, (int32_t)byte_offset, &instruction);
      loom_x86_function_append_transport(builder->function, &instruction);
      continue;
    }
    status = loom_x86_function_move(
        builder, move->destination.descriptor_reg_class_id,
        move->destination.location, move->source.descriptor_reg_class_id,
        move->source.location);
  }
  return status;
}

static uint32_t loom_x86_function_transport_scratch(
    uint16_t descriptor_reg_class_id) {
  const loom_x86_register_class_t register_class =
      loom_x86_logical_register_class(descriptor_reg_class_id);
  return register_class == LOOM_X86_REGISTER_CLASS_GPR32 ||
                 register_class == LOOM_X86_REGISTER_CLASS_GPR64
             ? LOOM_X86_GPR_TRANSPORT_SCRATCH_REGISTER
             : LOOM_X86_VECTOR_TRANSPORT_SCRATCH_REGISTER;
}

static void loom_x86_function_append_local_storage_transport(
    const loom_low_emission_frame_t* frame,
    const loom_low_allocation_assignment_t* assignment,
    loom_x86_storage_transfer_t transfer, uint32_t register_location,
    loom_x86_function_builder_t* builder) {
  const loom_low_storage_transport_binding_t* binding =
      &frame->allocation.storage_transport->bindings[assignment->location_base];
  loom_x86_transport_instruction_t instruction;
  loom_x86_transport_select_storage(
      transfer, assignment->descriptor_reg_class_id, register_location, 4,
      (int32_t)(builder->storage_offsets[binding->space] +
                binding->byte_offset),
      &instruction);
  loom_x86_function_append_transport(builder->function, &instruction);
}

static void loom_x86_function_append_abi_result_transport(
    const loom_x86_function_abi_t* abi, uint16_t result_index,
    uint16_t descriptor_reg_class_id, loom_x86_storage_transfer_t transfer,
    uint32_t register_location, uint8_t base_register,
    loom_x86_function_builder_t* builder) {
  loom_x86_transport_instruction_t instruction;
  const bool selected = loom_x86_transport_select_abi_storage(
      transfer, descriptor_reg_class_id, abi->results[result_index].byte_length,
      register_location, base_register,
      (int32_t)abi->results[result_index].stack_offset, &instruction);
  IREE_ASSERT_TRUE(selected);
  loom_x86_function_append_transport(builder->function, &instruction);
}

// Incoming memory is anchored at the caller's RSP. With dynamic alignment the
// saved pre-alignment pointer is loaded into the destination itself, so entry
// transport needs no globally reserved address scratch register.
static void loom_x86_function_incoming(const loom_low_emission_frame_t* frame,
                                       loom_low_allocation_location_kind_t kind,
                                       loom_x86_function_builder_t* builder) {
  loom_x86_function_t* function = builder->function;
  const loom_low_call_contract_t* contract =
      &builder->function_abi->call_contract;
  for (uint16_t i = 0; i < contract->argument_count; ++i) {
    if (contract->arguments[i].location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_assignment_for_value_ordinal(&frame->allocation, i,
                                                         NULL);
    if (!assignment || assignment->location_kind != kind) {
      continue;
    }
    const uint32_t destination = kind == LOOM_LOW_ALLOCATION_LOCATION_STORAGE
                                     ? loom_x86_function_transport_scratch(
                                           assignment->descriptor_reg_class_id)
                                     : assignment->location_base;
    loom_x86_incoming_fixup_t* fixup =
        &builder->incoming.fixups[builder->incoming.count++];
    fixup->pointer_instruction = UINT32_MAX;
    uint8_t base = function->stack.has_frame_pointer ? 5 : 4;
    if (!function->stack.has_frame_pointer && builder->storage_alignment > 16) {
      fixup->pointer_instruction = (uint32_t)function->instruction_count;
      loom_x86_function_append(
          function, LOOM_X86_ENCODING_FORM_LOAD, 0x8b | LOOM_X86_ENCODING_REX_W,
          (loom_x86_encoding_operands_t){.result = (uint8_t)destination,
                                         .inputs = {4}},
          UINT32_MAX);
      base = (uint8_t)destination;
    }
    fixup->value_instruction = (uint32_t)function->instruction_count;
    loom_x86_transport_instruction_t load;
    const bool selected = loom_x86_transport_select_abi_storage(
        LOOM_X86_STORAGE_TRANSFER_LOAD, assignment->descriptor_reg_class_id,
        builder->function_abi->arguments[i].byte_length, destination, base,
        (int32_t)builder->function_abi->arguments[i].stack_offset, &load);
    IREE_ASSERT_TRUE(selected);
    loom_x86_function_append_transport(function, &load);
    if (kind == LOOM_LOW_ALLOCATION_LOCATION_STORAGE) {
      loom_x86_function_append_local_storage_transport(
          frame, assignment, LOOM_X86_STORAGE_TRANSFER_STORE, destination,
          builder);
    }
  }
}

static iree_status_t loom_x86_function_call(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet,
    loom_x86_function_builder_t* builder) {
  loom_x86_function_t* function = builder->function;
  const loom_symbol_ref_t callee = loom_low_func_call_callee(packet->node->op);
  const loom_x86_function_abi_t* callee_abi =
      loom_x86_module_abi_lookup(builder->module_abi, callee);
  IREE_ASSERT_NE(callee_abi, NULL);
  const loom_low_allocation_call_moves_t* moves =
      loom_low_allocation_find_call_moves_by_source_ordinal(
          &frame->allocation, packet->node->source_ordinal);
  // Stack operands must be read before argument registers are permuted. The
  // outgoing area is disjoint from all function-local storage and is written
  // only by the call that consumes it.
  for (uint16_t i = 0; i < packet->node->operand_count; ++i) {
    if (callee_abi->call_contract.arguments[i].location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED) {
      continue;
    }
    const loom_low_allocation_assignment_t* argument =
        loom_low_packet_operand_assignment(&frame->allocation, packet, i);
    if (argument->location_kind == LOOM_LOW_ALLOCATION_LOCATION_STORAGE) {
      continue;
    }
    loom_x86_transport_instruction_t store;
    const bool selected = loom_x86_transport_select_abi_storage(
        LOOM_X86_STORAGE_TRANSFER_STORE, argument->descriptor_reg_class_id,
        callee_abi->arguments[i].byte_length, argument->location_base, 4,
        (int32_t)callee_abi->arguments[i].stack_offset, &store);
    IREE_ASSERT_TRUE(selected);
    loom_x86_function_append_transport(function, &store);
  }
  IREE_RETURN_IF_ERROR(
      loom_x86_function_moves(&frame->allocation, moves->arguments, builder));
  // All register sources have been consumed. RAX and XMM8 are caller-clobbered
  // and are not argument destinations, so stack copies can use them here.
  for (uint16_t i = 0; i < packet->node->operand_count; ++i) {
    if (callee_abi->call_contract.arguments[i].location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED) {
      continue;
    }
    const loom_low_allocation_assignment_t* argument =
        loom_low_packet_operand_assignment(&frame->allocation, packet, i);
    if (argument->location_kind != LOOM_LOW_ALLOCATION_LOCATION_STORAGE) {
      continue;
    }
    const loom_low_storage_transport_binding_t* binding =
        &frame->allocation.storage_transport->bindings[argument->location_base];
    const loom_x86_register_class_t register_class =
        loom_x86_logical_register_class(argument->descriptor_reg_class_id);
    const uint32_t scratch =
        register_class == LOOM_X86_REGISTER_CLASS_GPR32 ||
                register_class == LOOM_X86_REGISTER_CLASS_GPR64
            ? 0
            : 8;
    loom_x86_transport_instruction_t load;
    loom_x86_transport_select_storage(
        LOOM_X86_STORAGE_TRANSFER_LOAD, argument->descriptor_reg_class_id,
        scratch, 4,
        (int32_t)(builder->storage_offsets[binding->space] +
                  binding->byte_offset),
        &load);
    loom_x86_function_append_transport(function, &load);
    loom_x86_transport_instruction_t store;
    const bool selected = loom_x86_transport_select_abi_storage(
        LOOM_X86_STORAGE_TRANSFER_STORE, argument->descriptor_reg_class_id,
        callee_abi->arguments[i].byte_length, scratch, 4,
        (int32_t)callee_abi->arguments[i].stack_offset, &store);
    IREE_ASSERT_TRUE(selected);
    loom_x86_function_append_transport(function, &store);
  }
  if (callee_abi->has_indirect_results) {
    IREE_RETURN_IF_ERROR(
        loom_x86_function_move(builder, LOOM_X86_REGISTER_CLASS_GPR64,
                               LOOM_X86_INDIRECT_RESULT_POINTER_REGISTER,
                               LOOM_X86_REGISTER_CLASS_GPR64, 4));
  }
  if (!callee_abi->has_upper_vector_register_argument) {
    builder->upper_vector_call_cleanup_indices
        [function->upper_vector_call_cleanup_count++] =
        (uint32_t)function->instruction_count;
  }
  loom_x86_function_append(function, LOOM_X86_ENCODING_FORM_CALL, 0,
                           (loom_x86_encoding_operands_t){0}, callee.symbol_id);
  ++function->symbol_fixup_count;
  function->may_dirty_upper_vector_state |= callee_abi->has_upper_vector_result;
  for (uint16_t i = 0; i < callee_abi->call_contract.result_count; ++i) {
    const loom_low_allocation_abi_location_t* location =
        &callee_abi->call_contract.results[i];
    if (location->location_kind ==
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      loom_x86_function_normalize(
          function,
          (loom_x86_call_abi_value_action_t)callee_abi->results[i].action,
          location->location_base);
    }
  }
  // Storage destinations use fixed caller-clobbered scratch outside the ABI
  // result prefixes. Register result transport restores either scratch before
  // any indirect result is loaded into its final assigned register.
  for (uint16_t i = 0; i < packet->node->result_count; ++i) {
    if (callee_abi->call_contract.results[i].location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED) {
      continue;
    }
    const loom_low_allocation_assignment_t* result =
        loom_low_packet_result_assignment(&frame->allocation, packet, i);
    if (!result ||
        result->location_kind != LOOM_LOW_ALLOCATION_LOCATION_STORAGE) {
      continue;
    }
    const uint32_t scratch =
        loom_x86_function_transport_scratch(result->descriptor_reg_class_id);
    loom_x86_function_append_abi_result_transport(
        callee_abi, i, result->descriptor_reg_class_id,
        LOOM_X86_STORAGE_TRANSFER_LOAD, scratch, 4, builder);
    loom_x86_function_normalize(
        function,
        (loom_x86_call_abi_value_action_t)callee_abi->results[i].action,
        scratch);
    loom_x86_function_append_local_storage_transport(
        frame, result, LOOM_X86_STORAGE_TRANSFER_STORE, scratch, builder);
  }
  IREE_RETURN_IF_ERROR(
      loom_x86_function_moves(&frame->allocation, moves->results, builder));
  for (uint16_t i = 0; i < packet->node->result_count; ++i) {
    if (callee_abi->call_contract.results[i].location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED) {
      continue;
    }
    const loom_low_allocation_assignment_t* result =
        loom_low_packet_result_assignment(&frame->allocation, packet, i);
    if (!result ||
        result->location_kind == LOOM_LOW_ALLOCATION_LOCATION_STORAGE) {
      continue;
    }
    loom_x86_function_append_abi_result_transport(
        callee_abi, i, result->descriptor_reg_class_id,
        LOOM_X86_STORAGE_TRANSFER_LOAD, result->location_base, 4, builder);
    loom_x86_function_normalize(
        function,
        (loom_x86_call_abi_value_action_t)callee_abi->results[i].action,
        result->location_base);
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_function_return(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet,
    loom_x86_function_builder_t* builder) {
  const loom_x86_function_abi_t* abi = builder->function_abi;
  const loom_low_allocation_exit_moves_t* moves =
      loom_low_allocation_find_exit_moves_by_source_ordinal(
          &frame->allocation, packet->node->source_ordinal);
  for (uint16_t i = 0; i < packet->node->operand_count; ++i) {
    if (abi->call_contract.results[i].location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED) {
      continue;
    }
    const loom_low_allocation_assignment_t* result =
        loom_low_packet_operand_assignment(&frame->allocation, packet, i);
    if (result->location_kind == LOOM_LOW_ALLOCATION_LOCATION_STORAGE) {
      continue;
    }
    loom_x86_function_append_abi_result_transport(
        abi, i, result->descriptor_reg_class_id,
        LOOM_X86_STORAGE_TRANSFER_STORE, result->location_base,
        LOOM_X86_RETAINED_RESULT_POINTER_REGISTER, builder);
  }
  if (moves) {
    IREE_RETURN_IF_ERROR(
        loom_x86_function_moves(&frame->allocation, moves->results, builder));
  }
  for (uint16_t i = 0; i < packet->node->operand_count; ++i) {
    if (abi->call_contract.results[i].location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED) {
      continue;
    }
    const loom_low_allocation_assignment_t* result =
        loom_low_packet_operand_assignment(&frame->allocation, packet, i);
    if (result->location_kind != LOOM_LOW_ALLOCATION_LOCATION_STORAGE) {
      continue;
    }
    const uint32_t scratch =
        loom_x86_function_transport_scratch(result->descriptor_reg_class_id);
    loom_x86_function_append_local_storage_transport(
        frame, result, LOOM_X86_STORAGE_TRANSFER_LOAD, scratch, builder);
    loom_x86_function_append_abi_result_transport(
        abi, i, result->descriptor_reg_class_id,
        LOOM_X86_STORAGE_TRANSFER_STORE, scratch,
        LOOM_X86_RETAINED_RESULT_POINTER_REGISTER, builder);
  }
  return iree_ok_status();
}

static loom_attribute_t loom_x86_function_immediate(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet, uint16_t index) {
  const loom_low_immediate_t* immediate =
      &frame->target.descriptor_set
           ->immediates[packet->descriptor->immediate_start + index];
  loom_attribute_t value = loom_low_packet_immediate_attr(packet, immediate);
  return value.kind == LOOM_ATTR_ABSENT
             ? loom_attr_i64(immediate->default_value)
             : value;
}

static iree_status_t loom_x86_function_packet(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet,
    loom_x86_function_builder_t* builder) {
  loom_x86_function_t* function = builder->function;
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  if (descriptor->encoding_format_id == LOOM_X86_ENCODING_FORM_NONE) {
    iree_string_view_t mnemonic = loom_low_descriptor_set_string(
        frame->target.descriptor_set, descriptor->mnemonic_string_ref);
    return loom_x86_function_reject(builder, LOOM_ERR_X86_005, mnemonic);
  }
  loom_x86_encoding_operands_t operands = {0};
  const bool may_dirty_upper_vector_state =
      (descriptor->encoding_format_id & LOOM_X86_ENCODING_FORMAT_VECTOR) &&
      (descriptor->encoding_id >> 14);
  for (uint16_t i = 0; i < packet->node->operand_count; ++i) {
    const loom_low_allocation_assignment_t* input =
        loom_low_packet_operand_assignment(&frame->allocation, packet, i);
    if (may_dirty_upper_vector_state) {
      loom_x86_function_note_upper_vector_state(
          function, input->descriptor_reg_class_id, input->location_base);
    }
    operands.inputs[i] = loom_x86_function_register(input);
  }
  if (packet->node->result_count) {
    const loom_low_allocation_assignment_t* result =
        loom_low_packet_result_assignment(&frame->allocation, packet, 0);
    if (may_dirty_upper_vector_state) {
      loom_x86_function_note_upper_vector_state(
          function, result->descriptor_reg_class_id, result->location_base);
    }
    operands.result = loom_x86_function_register(result);
    const loom_x86_register_class_t result_class =
        loom_x86_logical_register_class(result->descriptor_reg_class_id);
    if (result_class == LOOM_X86_REGISTER_CLASS_GPR32 ||
        result_class == LOOM_X86_REGISTER_CLASS_GPR64) {
      const uint16_t preserved = (1u << 3) | (1u << 5) | (0xfu << 12);
      function->saved_registers |= (1u << operands.result) & preserved;
    }
  }
  uint32_t reference = UINT32_MAX;
  if (descriptor->immediate_count) {
    const loom_attribute_t immediate =
        loom_x86_function_immediate(frame, packet, 0);
    if (immediate.kind == LOOM_ATTR_SYMBOL) {
      reference = loom_attr_as_symbol(immediate).symbol_id;
      ++function->symbol_fixup_count;
    } else {
      operands.immediate = immediate.i64;
    }
  }
  if (descriptor->immediate_count == 2) {
    operands.scale = (uint8_t)iree_math_count_trailing_zeros_u32(
        (uint32_t)loom_x86_function_immediate(frame, packet, 1).i64);
  }
  loom_x86_function_append(function, descriptor->encoding_format_id,
                           descriptor->encoding_id, operands, reference);
  return iree_ok_status();
}

static iree_status_t loom_x86_function_storage(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet,
    loom_x86_function_builder_t* builder) {
  const loom_op_t* op = packet->node->op;
  const bool is_address = loom_low_storage_address_isa(op);
  const bool is_store = loom_low_spill_isa(op);
  const loom_value_id_t storage_value =
      is_address ? loom_low_storage_address_storage(op)
                 : (is_store ? loom_low_spill_storage(op)
                             : loom_low_reload_storage(op));
  const uint64_t relative_offset =
      is_address
          ? loom_low_storage_address_offset(op)
          : (is_store ? loom_low_spill_offset(op) : loom_low_reload_offset(op));
  const loom_low_storage_layout_t* layout =
      &frame->schedule.requirements.storage_layout;
  loom_low_storage_layout_reference_t reference;
  loom_low_storage_layout_lookup_reference(&layout->index, layout->records,
                                           storage_value, &reference);
  const uint64_t byte_offset =
      builder->storage_offsets[reference.reservation.space] +
      reference.reservation.byte_offset + reference.byte_offset +
      relative_offset;
  const loom_low_allocation_assignment_t* assignment =
      is_store
          ? loom_low_packet_operand_assignment(&frame->allocation, packet, 0)
          : loom_low_packet_result_assignment(&frame->allocation, packet, 0);
  if (assignment->location_kind == LOOM_LOW_ALLOCATION_LOCATION_STORAGE) {
    return iree_ok_status();
  }
  const loom_x86_register_class_t logical_class =
      loom_x86_logical_register_class(assignment->descriptor_reg_class_id);
  if (is_address && (logical_class != LOOM_X86_REGISTER_CLASS_GPR64 ||
                     assignment->location_count != 1)) {
    return loom_x86_function_reject(
        builder, LOOM_ERR_X86_004,
        IREE_SV("one 64-bit GPR for storage addresses"));
  }
  const uint32_t unit_bytes =
      loom_x86_transport_byte_length(assignment->descriptor_reg_class_id);
  if (!is_address && (relative_offset > reference.byte_length ||
                      (uint64_t)assignment->location_count * unit_bytes >
                          reference.byte_length - relative_offset)) {
    return loom_x86_function_reject(
        builder, LOOM_ERR_X86_004,
        IREE_SV("a register transfer that fits within its storage span"));
  }
  // Every current x86 register class owns one physical allocation unit.
  IREE_ASSERT_EQ(assignment->location_count, 1u);
  const loom_low_move_location_t location =
      loom_low_allocation_assignment_unit_location(frame->target.descriptor_set,
                                                   assignment, 0);
  if (is_address) {
    loom_x86_function_append(
        builder->function, LOOM_X86_ENCODING_FORM_ADDRESS_DISPLACEMENT,
        0x8d | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){.immediate = (int64_t)byte_offset,
                                       .result = (uint8_t)location.location,
                                       .inputs = {4}},
        UINT32_MAX);
  } else {
    loom_x86_transport_instruction_t instruction;
    loom_x86_transport_select_storage(is_store ? LOOM_X86_STORAGE_TRANSFER_STORE
                                               : LOOM_X86_STORAGE_TRANSFER_LOAD,
                                      location.descriptor_reg_class_id,
                                      location.location, 4,
                                      (int32_t)byte_offset, &instruction);
    loom_x86_function_append_transport(builder->function, &instruction);
  }
  return iree_ok_status();
}

static void loom_x86_function_jump(loom_x86_function_t* function,
                                   uint32_t target, uint32_t fallthrough) {
  if (target != fallthrough) {
    loom_x86_function_append(function, LOOM_X86_ENCODING_FORM_JUMP, 0,
                             (loom_x86_encoding_operands_t){0}, target);
  }
}

static iree_status_t loom_x86_function_structural(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet,
    loom_x86_function_builder_t* builder) {
  loom_x86_function_t* function = builder->function;
  const loom_low_schedule_node_t* node = packet->node;
  if (loom_low_storage_reserve_isa(node->op) ||
      loom_low_storage_view_isa(node->op)) {
    return iree_ok_status();
  }
  if (loom_low_spill_isa(node->op) || loom_low_reload_isa(node->op) ||
      loom_low_storage_address_isa(node->op)) {
    return loom_x86_function_storage(frame, packet, builder);
  }
  const uint32_t block_index = node->block_index;
  const loom_cfg_graph_t* graph = &frame->schedule.cfg_graph;
  if (loom_low_func_call_isa(node->op)) {
    return loom_x86_function_call(frame, packet, builder);
  }
  if (loom_low_return_isa(node->op)) {
    IREE_RETURN_IF_ERROR(loom_x86_function_return(frame, packet, builder));
    loom_x86_function_jump(function, function->block_count, block_index + 1);
    return iree_ok_status();
  }
  if (loom_low_br_isa(node->op)) {
    const loom_low_allocation_edge_copy_group_t* group =
        loom_low_allocation_find_edge_copy_group_by_source_ordinal(
            &frame->allocation, node->source_ordinal);
    if (group) {
      IREE_RETURN_IF_ERROR(loom_x86_function_moves(
          &frame->allocation, group->move_group.moves, builder));
    }
    loom_x86_function_jump(
        function,
        graph->successor_indices[graph->blocks[block_index].successor_start],
        block_index + 1);
    return iree_ok_status();
  }
  if (loom_low_cond_br_isa(node->op)) {
    const uint16_t* targets =
        graph->successor_indices + graph->blocks[block_index].successor_start;
    const loom_low_allocation_assignment_t* condition =
        loom_low_packet_operand_assignment(&frame->allocation, packet, 0);
    const uint16_t width =
        loom_x86_logical_register_class(condition->descriptor_reg_class_id) ==
                LOOM_X86_REGISTER_CLASS_GPR64
            ? LOOM_X86_ENCODING_REX_W
            : 0;
    const bool true_falls_through = targets[0] == block_index + 1;
    loom_x86_function_append(function,
                             true_falls_through
                                 ? LOOM_X86_ENCODING_FORM_BRANCH_ZERO
                                 : LOOM_X86_ENCODING_FORM_BRANCH_NONZERO,
                             width,
                             (loom_x86_encoding_operands_t){
                                 .inputs = {(uint8_t)condition->location_base}},
                             targets[true_falls_through ? 1 : 0]);
    if (!true_falls_through) {
      loom_x86_function_jump(function, targets[1], block_index + 1);
    }
    return iree_ok_status();
  }
  if (loom_low_copy_isa(node->op) || loom_low_move_isa(node->op) ||
      loom_low_slice_isa(node->op) || loom_low_concat_isa(node->op)) {
    const loom_low_allocation_packet_move_group_t* group =
        loom_low_allocation_find_packet_move_group_by_source_ordinal(
            &frame->allocation, node->source_ordinal);
    return group ? loom_x86_function_moves(&frame->allocation,
                                           group->move_group.moves, builder)
                 : iree_ok_status();
  }
  iree_string_view_t name = loom_op_name(frame->module, node->op);
  return loom_x86_function_reject(builder, LOOM_ERR_X86_005, name);
}

iree_status_t loom_x86_function_prepare(
    const loom_low_emission_frame_t* frame,
    const loom_x86_function_abi_t* function_abi,
    const loom_x86_module_abi_t* module_abi, iree_diagnostic_emitter_t emitter,
    iree_arena_allocator_t* arena, bool* out_accepted,
    loom_x86_function_t* out_function) {
  *out_accepted = false;
  *out_function = (loom_x86_function_t){0};
  const loom_low_schedule_table_t* schedule = &frame->schedule;
  loom_x86_function_t function = {
      .block_count = (uint32_t)schedule->block_count,
      .saved_registers = function_abi->has_simd_stack_argument ? (1u << 5) : 0,
      .may_dirty_upper_vector_state =
          function_abi->has_upper_vector_register_argument,
      .has_upper_vector_result = function_abi->has_upper_vector_result,
      .stack.has_frame_pointer = function_abi->has_simd_stack_argument,
  };
  iree_host_size_t incoming_count = 0;
  iree_host_size_t normalization_count = 0;
  for (uint16_t i = 0; i < function_abi->call_contract.argument_count; ++i) {
    incoming_count += function_abi->call_contract.arguments[i].location_kind ==
                      LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED;
    normalization_count +=
        function_abi->call_contract.arguments[i].location_kind ==
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER &&
        function_abi->arguments[i].action !=
            LOOM_X86_CALL_ABI_VALUE_ACTION_NONE;
  }
  iree_host_size_t outgoing_count = 0;
  iree_host_size_t indirect_call_count = 0;
  iree_host_size_t indirect_call_result_count = 0;
  uint32_t outgoing_bytes = 0;
  uint32_t outgoing_alignment = schedule->call_node_count ? 16 : 0;
  for (iree_host_size_t i = 0; i < schedule->call_node_count; ++i) {
    const loom_low_schedule_node_t* node =
        &schedule->nodes[schedule->call_node_indices[i]];
    const loom_x86_function_abi_t* callee_abi = loom_x86_module_abi_lookup(
        module_abi, loom_low_func_call_callee(node->op));
    IREE_ASSERT_NE(callee_abi, NULL);
    outgoing_bytes = iree_max(outgoing_bytes, callee_abi->call_storage_bytes);
    outgoing_alignment =
        iree_max(outgoing_alignment, callee_abi->call_storage_alignment);
    for (uint16_t j = 0; j < callee_abi->call_contract.argument_count; ++j) {
      outgoing_count += callee_abi->call_contract.arguments[j].location_kind ==
                        LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED;
    }
    indirect_call_count += callee_abi->has_indirect_results;
    for (uint16_t j = 0; j < callee_abi->call_contract.result_count; ++j) {
      normalization_count +=
          callee_abi->results[j].action != LOOM_X86_CALL_ABI_VALUE_ACTION_NONE;
      indirect_call_result_count +=
          callee_abi->call_contract.results[j].location_kind ==
          LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED;
    }
  }
  iree_host_size_t indirect_result_count = 0;
  for (uint16_t i = 0; i < function_abi->call_contract.result_count; ++i) {
    indirect_result_count +=
        function_abi->call_contract.results[i].location_kind ==
        LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED;
  }
  // A descriptor is one encoding record; a return or conditional branch needs
  // at most two. Allocation retains the exact number of final physical moves,
  // including multi-unit transport and cycle scratch.
  const iree_host_size_t capacity =
      schedule->scheduled_node_count + schedule->block_count +
      frame->allocation.move_count + 2u * outgoing_count + 3u * incoming_count +
      normalization_count + indirect_call_count +
      2u * indirect_call_result_count + function_abi->has_indirect_results +
      2u * indirect_result_count * frame->allocation.exit_move_count;
  loom_x86_function_builder_t builder = {
      .function = &function,
      .source_op = frame->function_op,
      .emitter = emitter,
      .function_abi = function_abi,
      .module_abi = module_abi,
      .storage_byte_length = outgoing_bytes,
      .storage_alignment = outgoing_alignment,
  };
  IREE_RETURN_IF_ERROR(
      loom_x86_function_storage_layout(frame, arena, &builder));
  if (builder.rejected) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, capacity, sizeof(*function.instructions),
                                (void**)&function.instructions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, schedule->block_count + 1, sizeof(*function.block_starts),
      (void**)&function.block_starts));
  if (schedule->call_node_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, schedule->call_node_count,
        sizeof(*builder.upper_vector_call_cleanup_indices),
        (void**)&builder.upper_vector_call_cleanup_indices));
    function.upper_vector_call_cleanup_indices =
        builder.upper_vector_call_cleanup_indices;
  }
  if (incoming_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, incoming_count, sizeof(*builder.incoming.fixups),
        (void**)&builder.incoming.fixups));
  }
  // Invocation transport precedes block labels so a body backedge cannot
  // reload original ABI inputs. Its writes participate in frame preservation.
  if (function_abi->has_indirect_results) {
    IREE_RETURN_IF_ERROR(
        loom_x86_function_move(&builder, LOOM_X86_REGISTER_CLASS_GPR64,
                               LOOM_X86_RETAINED_RESULT_POINTER_REGISTER,
                               LOOM_X86_REGISTER_CLASS_GPR64,
                               LOOM_X86_INDIRECT_RESULT_POINTER_REGISTER));
  }
  loom_x86_function_normalize_arguments(&builder);
  loom_x86_function_incoming(frame, LOOM_LOW_ALLOCATION_LOCATION_STORAGE,
                             &builder);
  iree_status_t status = loom_x86_function_moves(
      &frame->allocation, frame->allocation.entry_moves.moves, &builder);
  if (!builder.rejected && iree_status_is_ok(status)) {
    loom_x86_function_incoming(
        frame, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, &builder);
  }
  for (uint32_t b = 0; b < schedule->block_count && !builder.rejected &&
                       iree_status_is_ok(status);
       ++b) {
    function.block_starts[b] = function.instruction_count;
    const loom_low_schedule_block_t* block = &schedule->blocks[b];
    for (uint32_t i = 0; i < block->scheduled_node_count && !builder.rejected &&
                         iree_status_is_ok(status);
         ++i) {
      const loom_low_packet_view_t packet =
          loom_low_packet_at_block_ordinal(schedule, b, i);
      if (loom_low_packet_is_compile_time_only(&packet)) {
        continue;
      }
      builder.source_op = packet.node->op;
      status = packet.descriptor
                   ? loom_x86_function_packet(frame, &packet, &builder)
                   : loom_x86_function_structural(frame, &packet, &builder);
    }
  }
  function.block_starts[schedule->block_count] = function.instruction_count;
  if (!builder.rejected && iree_status_is_ok(status)) {
    builder.source_op = frame->function_op;
    status = loom_x86_function_stack_frame(&builder);
  }
  if (!builder.rejected && iree_status_is_ok(status)) {
    *out_function = function;
    *out_accepted = true;
  }
  return status;
}

typedef struct loom_x86_branch_fixup_t {
  // Output displacement field offset.
  iree_io_stream_pos_t offset;
  // Retained block ordinal; block_count denotes the epilogue.
  uint32_t target;
} loom_x86_branch_fixup_t;

static iree_status_t loom_x86_function_write_encoding(
    iree_io_stream_t* stream, loom_x86_encoding_form_t form,
    uint16_t encoding_id, loom_x86_encoding_operands_t operands) {
  loom_x86_encoded_instruction_t instruction;
  loom_x86_encode_instruction(form, encoding_id, &operands, &instruction);
  return iree_io_stream_write(stream, instruction.length, instruction.bytes);
}

static iree_status_t loom_x86_function_write_stack(
    iree_io_stream_t* stream, loom_x86_encoding_form_t form, uint8_t reg) {
  return loom_x86_function_write_encoding(
      stream, form, 0, (loom_x86_encoding_operands_t){.inputs = {reg}});
}

static iree_status_t loom_x86_function_write_stack_enter(
    const loom_x86_function_t* function, iree_io_stream_t* stream) {
  if (function->stack.has_frame_pointer) {
    IREE_RETURN_IF_ERROR(loom_x86_function_write_encoding(
        stream, LOOM_X86_ENCODING_FORM_MOVE, 0x8b | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){.result = 5, .inputs = {4}}));
  } else if (function->stack.realignment.mask) {
    IREE_RETURN_IF_ERROR(loom_x86_function_write_encoding(
        stream, LOOM_X86_ENCODING_FORM_MOVE, 0x8b | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){
            .result = function->stack.realignment.scratch_register,
            .inputs = {4}}));
  }
  if (function->stack.realignment.mask) {
    IREE_RETURN_IF_ERROR(loom_x86_function_write_encoding(
        stream, LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
        0x81 | (4u << 9) | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){
            .immediate = function->stack.realignment.mask, .result = 4}));
  }
  if (function->stack.allocation_size) {
    IREE_RETURN_IF_ERROR(loom_x86_function_write_encoding(
        stream, LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
        0x81 | (5u << 9) | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){
            .immediate = function->stack.allocation_size, .result = 4}));
  }
  if (function->stack.realignment.mask && !function->stack.has_frame_pointer) {
    IREE_RETURN_IF_ERROR(loom_x86_function_write_encoding(
        stream, LOOM_X86_ENCODING_FORM_STORE, 0x89 | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){
            .immediate = function->stack.realignment.saved_pointer_offset,
            .inputs = {function->stack.realignment.scratch_register, 4}}));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_function_write_stack_leave(
    const loom_x86_function_t* function, iree_io_stream_t* stream) {
  if (function->stack.has_frame_pointer) {
    return loom_x86_function_write_encoding(
        stream, LOOM_X86_ENCODING_FORM_MOVE, 0x8b | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){.result = 4, .inputs = {5}});
  }
  if (function->stack.realignment.mask) {
    return loom_x86_function_write_encoding(
        stream, LOOM_X86_ENCODING_FORM_LOAD, 0x8b | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){
            .immediate = function->stack.realignment.saved_pointer_offset,
            .result = 4,
            .inputs = {4}});
  }
  if (function->stack.allocation_size) {
    return loom_x86_function_write_encoding(
        stream, LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
        0x81 | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){
            .immediate = function->stack.allocation_size, .result = 4});
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_function_write_upper_vector_state_cleanup(
    iree_io_stream_t* stream) {
  static const uint8_t bytes[] = {0xc5, 0xf8, 0x77};
  return iree_io_stream_write(stream, sizeof(bytes), bytes);
}

iree_status_t loom_x86_function_write(const loom_x86_function_t* function,
                                      const uint32_t* symbol_indices,
                                      iree_host_size_t section_index,
                                      loom_native_object_fixup_t* symbol_fixups,
                                      iree_io_stream_t* stream,
                                      iree_arena_allocator_t* arena) {
  const iree_io_stream_pos_t function_start = iree_io_stream_offset(stream);
  iree_io_stream_pos_t* block_offsets = NULL;
  loom_x86_branch_fixup_t* fixups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, function->block_count + 1, sizeof(*block_offsets),
      (void**)&block_offsets));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, 2 * function->block_count, sizeof(*fixups), (void**)&fixups));
  iree_status_t status = iree_ok_status();
  for (uint8_t reg = 0; reg < 16 && iree_status_is_ok(status); ++reg) {
    if (function->saved_registers & (1u << reg)) {
      status = loom_x86_function_write_stack(stream,
                                             LOOM_X86_ENCODING_FORM_PUSH, reg);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_x86_function_write_stack_enter(function, stream);
  }
  iree_host_size_t fixup_count = 0;
  iree_host_size_t symbol_fixup_index = 0;
  iree_host_size_t call_cleanup_index = 0;
  uint32_t block = 0;
  for (iree_host_size_t i = 0;
       i < function->instruction_count && iree_status_is_ok(status); ++i) {
    while (block < function->block_count &&
           function->block_starts[block] == i) {
      block_offsets[block++] = iree_io_stream_offset(stream);
    }
    const loom_x86_instruction_t* prepared = &function->instructions[i];
    if (function->may_dirty_upper_vector_state &&
        call_cleanup_index < function->upper_vector_call_cleanup_count &&
        function->upper_vector_call_cleanup_indices[call_cleanup_index] == i) {
      status = loom_x86_function_write_upper_vector_state_cleanup(stream);
      ++call_cleanup_index;
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
    loom_x86_encoded_instruction_t instruction;
    loom_x86_encode_instruction(prepared->encoding_format_id,
                                prepared->encoding_id, &prepared->operands,
                                &instruction);
    const bool is_call =
        prepared->encoding_format_id == LOOM_X86_ENCODING_FORM_CALL;
    const bool is_symbol_address =
        prepared->encoding_format_id ==
            LOOM_X86_ENCODING_FORM_ADDRESS_PC_RELATIVE ||
        (prepared->encoding_format_id & LOOM_X86_ENCODING_FORMAT_VECTOR &&
         ((prepared->encoding_format_id >> 12) & 7) ==
             LOOM_X86_VECTOR_ENCODING_RIP_LOAD);
    if ((is_call || is_symbol_address) && prepared->reference != UINT32_MAX) {
      symbol_fixups[symbol_fixup_index++] = (loom_native_object_fixup_t){
          .section_contribution_index = section_index,
          .section_offset = iree_io_stream_offset(stream) + instruction.length -
                            4 - function_start,
          .relocation_kind =
              is_call ? LOOM_X86_RELOCATION_CALL : LOOM_X86_RELOCATION_ADDRESS,
          .target_symbol_index = symbol_indices[prepared->reference],
          .addend = -4,
      };
    } else if (prepared->reference != UINT32_MAX) {
      fixups[fixup_count++] = (loom_x86_branch_fixup_t){
          .offset = iree_io_stream_offset(stream) + instruction.length - 4,
          .target = prepared->reference,
      };
    }
    status =
        iree_io_stream_write(stream, instruction.length, instruction.bytes);
  }
  while (block <= function->block_count) {
    block_offsets[block++] = iree_io_stream_offset(stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_x86_function_write_stack_leave(function, stream);
  }
  for (uint8_t i = 16; i > 0 && iree_status_is_ok(status); --i) {
    const uint8_t reg = i - 1;
    if (function->saved_registers & (1u << reg)) {
      status = loom_x86_function_write_stack(stream, LOOM_X86_ENCODING_FORM_POP,
                                             reg);
    }
  }
  if (iree_status_is_ok(status) && function->may_dirty_upper_vector_state &&
      !function->has_upper_vector_result) {
    status = loom_x86_function_write_upper_vector_state_cleanup(stream);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_x86_function_write_stack(stream, LOOM_X86_ENCODING_FORM_RETURN, 0);
  }
  const iree_io_stream_pos_t end = iree_io_stream_offset(stream);
  for (iree_host_size_t i = 0; i < fixup_count && iree_status_is_ok(status);
       ++i) {
    const int64_t displacement =
        block_offsets[fixups[i].target] - fixups[i].offset - 4;
    if (displacement < INT32_MIN || displacement > INT32_MAX) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "x86 branch exceeds signed 32-bit displacement");
    } else {
      uint8_t bytes[4];
      iree_unaligned_store_le_u32(bytes, (uint32_t)displacement);
      status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET,
                                   fixups[i].offset);
      if (iree_status_is_ok(status)) {
        status = iree_io_stream_write(stream, 4, bytes);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, end);
  }
  return status;
}
