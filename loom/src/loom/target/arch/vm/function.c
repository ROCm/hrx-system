// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/function.h"

#include <string.h>

#include "iree/vm/bytecode/wire/core.h"
#include "loom/codegen/low/allocation/move_sequence.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/vm/descriptors/descriptors.h"
#include "loom/target/arch/vm/module.h"

// Branch displacement fields are patched after the single emission walk. Dense
// target indices come directly from the shared CFG, in scheduled block order.
typedef struct loom_vm_branch_fixup_t {
  // Absolute stream offset of the signed word-displacement field.
  iree_io_stream_pos_t offset;
  // Dense target block index in the shared schedule.
  uint16_t target_block;
} loom_vm_branch_fixup_t;

static iree_status_t loom_vm_function_moves(const loom_low_move_t* moves,
                                            loom_low_move_range_t range,
                                            iree_io_stream_t* stream) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < range.count && iree_status_is_ok(status);
       ++i) {
    const loom_low_move_t* move = &moves[range.start + i];
    // Value copies and ref retains share the same register-pair encoding.
    // Ref copies acquire the new owner before replacing their destination.
    const iree_vm_bytecode_value_copy_t instruction = {
        .opcode =
            move->source.descriptor_reg_class_id == VM_CORE_REG_CLASS_ID_REF
                ? IREE_VM_BYTECODE_OPCODE_REF_RETAIN
                : IREE_VM_BYTECODE_OPCODE_VALUE_COPY,
        .destination_v8 = (uint8_t)move->destination.location,
        .source_v8 = (uint8_t)move->source.location,
    };
    status = iree_io_stream_write(stream, sizeof(instruction), &instruction);
  }
  return status;
}

static iree_status_t loom_vm_function_branch(
    iree_io_stream_t* stream, uint8_t opcode, uint8_t condition,
    uint16_t target_block, loom_vm_branch_fixup_t* out_fixup) {
  // The wide branch family shares one encoding. An unconditional branch has
  // zero in the condition byte, which is reserved padding in that form.
  const iree_vm_bytecode_control_branch_if_s32_t instruction = {
      .opcode = opcode,
      .condition_v8 = condition,
  };
  *out_fixup = (loom_vm_branch_fixup_t){
      .offset = iree_io_stream_offset(stream) +
                offsetof(iree_vm_bytecode_control_branch_if_s32_t,
                         target_word_offset_s32),
      .target_block = target_block,
  };
  return iree_io_stream_write(stream, sizeof(instruction), &instruction);
}

typedef struct loom_vm_return_state_t {
  // ABI result registers remain live even when their identity moves are elided.
  uint16_t result_count;
  // Function register high water, including any return-cycle temporary.
  uint16_t register_count;
} loom_vm_return_state_t;

// Cycle scratch excludes the complete result prefix and the move sources. The
// solver's compacted move list omits identity results, which must survive too.
// No temporary is reserved throughout the rest of the function.
static iree_status_t loom_vm_return_temporary(
    void* user_data, const loom_low_move_location_t* storage_class,
    const loom_low_move_t* moves, iree_host_size_t move_count,
    loom_low_move_location_t* out_temporary, bool* out_resolved) {
  loom_vm_return_state_t* states = user_data;
  loom_vm_return_state_t* state =
      &states[storage_class->descriptor_reg_class_id];
  bool occupied[256] = {false};
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    if (moves[i].source.descriptor_reg_class_id !=
        storage_class->descriptor_reg_class_id) {
      continue;
    }
    occupied[moves[i].source.location] = true;
    occupied[moves[i].destination.location] = true;
  }
  uint32_t location = state->result_count;
  while (occupied[location]) {
    ++location;
  }
  *out_temporary = *storage_class;
  out_temporary->location = location;
  state->register_count = iree_max(state->register_count, location + 1);
  *out_resolved = true;
  return iree_ok_status();
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_vm_function_return(
    const loom_low_emission_frame_t* frame,
    const loom_low_schedule_node_t* node,
    loom_low_move_sequence_scratch_t* scratch, iree_io_stream_t* stream,
    const loom_vm_function_signature_t* signature,
    iree_vm_bytecode_v0_function_row_t* out_row) {
  loom_vm_return_state_t states[] = {
      [VM_CORE_REG_CLASS_ID_VALUE] =
          {.result_count = iree_min(16, signature->row.result_value_count_u16),
           .register_count = out_row->value_register_count_u16},
      [VM_CORE_REG_CLASS_ID_REF] =
          {.result_count = iree_min(16, signature->row.result_ref_count_u16),
           .register_count = out_row->ref_register_count_u16},
  };
  uint16_t ordinals_by_bank[IREE_ARRAYSIZE(states)] = {0};
  const loom_value_ordinal_t* ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  uint16_t direct_count = 0;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < node->operand_count && iree_status_is_ok(status);
       ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_assignment_for_value_ordinal(&frame->allocation,
                                                         ordinals[i], NULL);
    const loom_low_move_location_t source = {
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .descriptor_reg_class_id = assignment->descriptor_reg_class_id,
        .location = assignment->location_base,
    };
    const uint16_t ordinal =
        ordinals_by_bank[assignment->descriptor_reg_class_id]++;
    if (ordinal >= 16) {
      if (assignment->descriptor_reg_class_id == VM_CORE_REG_CLASS_ID_REF) {
        // Overflow publication consumes its source, which may also appear in
        // another result. Local refs are dead at return, so slot zero holds
        // one temporary owner without reserving a physical register.
        const iree_vm_bytecode_ref_stack_load_retain_t transfers[] = {
            {.opcode = IREE_VM_BYTECODE_OPCODE_REF_STACK_STORE_RETAIN,
             .destination_r8 = (uint8_t)source.location,
             .slot_u16 = 0},
            {.opcode = IREE_VM_BYTECODE_OPCODE_REF_ABI_RESULT_STORE_MOVE,
             .destination_r8 = (uint8_t)source.location,
             .slot_u16 = ordinal - 16},
            {.opcode = IREE_VM_BYTECODE_OPCODE_REF_STACK_LOAD_MOVE,
             .destination_r8 = (uint8_t)source.location,
             .slot_u16 = 0},
        };
        out_row->local_ref_count_u32 =
            iree_max(out_row->local_ref_count_u32, 1);
        status = iree_io_stream_write(stream, sizeof(transfers), transfers);
        continue;
      }
      const iree_vm_bytecode_value_abi_result_store_t instruction = {
          .opcode = IREE_VM_BYTECODE_OPCODE_VALUE_ABI_RESULT_STORE,
          .source_v8 = (uint8_t)source.location,
          .slot_u16 = ordinal - 16,
      };
      status = iree_io_stream_write(stream, sizeof(instruction), &instruction);
      continue;
    }
    scratch->moves[direct_count] =
        (loom_low_move_t){.source = source, .destination = source};
    scratch->moves[direct_count++].destination.location = ordinal;
  }
  IREE_RETURN_IF_ERROR(status);
  const loom_low_move_sequence_options_t options = {
      .descriptor_set = frame->target.descriptor_set,
      .resolve_temporary = {.fn = loom_vm_return_temporary,
                            .user_data = states},
  };
  // Each direct bank has at most 16 results. Each cycle adds at most one
  // save for two original moves, so this also covers every cyclic permutation.
  loom_low_move_t moves[24 * IREE_ARRAYSIZE(states)];
  iree_host_size_t move_count = 0;
  bool complete = false;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_resolve(
      scratch, direct_count, &options, IREE_ARRAYSIZE(moves), moves,
      &move_count, &complete));
  // Capacity covers the worst permutation and the temporary always resolves.
  IREE_ASSERT(complete);
  out_row->value_register_count_u16 =
      states[VM_CORE_REG_CLASS_ID_VALUE].register_count;
  out_row->ref_register_count_u16 =
      states[VM_CORE_REG_CLASS_ID_REF].register_count;
  IREE_RETURN_IF_ERROR(loom_vm_function_moves(
      moves, (loom_low_move_range_t){.count = move_count}, stream));
  const iree_vm_bytecode_control_return_t instruction = {
      .opcode = IREE_VM_BYTECODE_OPCODE_CONTROL_RETURN,
  };
  return iree_io_stream_write(stream, sizeof(instruction), &instruction);
}

// Bind a data operand while emitting its first consumer, keeping unused and
// non-VM payloads out of this artifact. Authored numeric ordinals name the
// declaration-order table; symbols and ordinals share the same final binding.
static iree_status_t loom_vm_function_rodata(const loom_module_t* module,
                                             loom_attribute_t value,
                                             loom_vm_module_plan_t* plan,
                                             uint64_t* out_ordinal) {
  loom_symbol_ref_t symbol = value.symbol;
  if (value.kind != LOOM_ATTR_SYMBOL) {
    if ((uint64_t)value.i64 >= plan->rodata.symbol_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM rodata ordinal exceeds the data table");
    }
    symbol = (loom_symbol_ref_t){.symbol_id = plan->rodata.symbols[value.i64]};
  }
  const loom_op_t* definition =
      module->symbols.entries[symbol.symbol_id].defining_op;
  if (!loom_global_rodata_def_isa(definition)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM rodata operand requires a data definition");
  }
  uint16_t* ordinal = &plan->rodata.ordinals_by_symbol[symbol.symbol_id];
  if (*ordinal == UINT16_MAX) {
    const int64_t alignment = loom_global_rodata_def_alignment(definition);
    if (alignment > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM rodata alignment exceeds u32");
    }
    *ordinal = (uint16_t)plan->rodata.count;
    plan->rodata.values[plan->rodata.count++] = definition;
    plan->rodata.alignment =
        iree_max(plan->rodata.alignment, (uint32_t)alignment);
  }
  *out_ordinal = *ordinal;
  return iree_ok_status();
}

// Keep field encoding in its own compilation boundary instead of growing the
// register-allocation scope of frame and control-flow emission.
IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_vm_function_packet(
    const loom_low_emission_frame_t* frame,
    const loom_low_schedule_node_t* node, loom_vm_module_plan_t* module_plan,
    iree_io_stream_t* stream) {
  const loom_low_descriptor_t* descriptor = node->descriptor;
  const loom_low_operand_t* operands =
      frame->target.descriptor_set->operands + descriptor->operand_start;
  const loom_value_ordinal_t* results =
      loom_low_schedule_node_const_result_ordinals(node);
  const loom_value_ordinal_t* inputs =
      loom_low_schedule_node_const_operand_ordinals(node);
  uint8_t packet[sizeof(iree_vm_bytecode_constant_i64_t)] = {
      (uint8_t)descriptor->encoding_id,
  };
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand = &operands[i];
    const loom_value_ordinal_t ordinal =
        (operand->role == LOOM_LOW_OPERAND_ROLE_RESULT
             ? results
             : inputs)[operand->source_value_index];
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_assignment_for_value_ordinal(&frame->allocation,
                                                         ordinal, NULL);
    packet[operand->encoding_field_id] = (uint8_t)assignment->location_base;
  }
  iree_host_size_t packet_length = descriptor->encoding_format_id;
  if (descriptor->immediate_count) {
    // Required descriptor immediates and canonical IR dictionaries share name
    // order, established by the projection and Low verifier.
    const loom_named_attr_slice_t attributes =
        loom_low_const_isa(node->op) ? loom_low_const_attrs(node->op)
                                     : loom_low_op_attrs(node->op);
    // Keep constants canonical through Low CSE, including those synthesized by
    // address materializers. Select the shortest encoding of the complete cell
    // here; its high half also matters to untyped register copies and spills.
    if (packet[0] == IREE_VM_BYTECODE_OPCODE_CONSTANT_I32 ||
        packet[0] == IREE_VM_BYTECODE_OPCODE_CONSTANT_I64) {
      // Verified i32 immediates already have a zero high half.
      const loom_attribute_t bits_attr =
          packet[0] == IREE_VM_BYTECODE_OPCODE_CONSTANT_I32
              ? loom_vm_core_constant_i32_bits(attributes)
              : loom_vm_core_constant_i64_bits(attributes);
      const uint64_t bits = (uint64_t)bits_attr.i64;
      if (bits == 0) {
        packet[0] = IREE_VM_BYTECODE_OPCODE_CONSTANT_ZERO;
        packet_length = sizeof(iree_vm_bytecode_constant_zero_t);
      } else if (bits + UINT64_C(32768) <= UINT64_C(65535)) {
        packet[0] = IREE_VM_BYTECODE_OPCODE_CONSTANT_S16;
        memcpy(
            packet + offsetof(iree_vm_bytecode_constant_s16_t, immediate_i16),
            &bits, sizeof(int16_t));
        packet_length = sizeof(iree_vm_bytecode_constant_s16_t);
      } else {
        if (bits <= UINT32_MAX) {
          packet[0] = IREE_VM_BYTECODE_OPCODE_CONSTANT_I32;
          packet_length = sizeof(iree_vm_bytecode_constant_i32_t);
        }
        memcpy(packet + offsetof(iree_vm_bytecode_constant_i32_t, bits_u32),
               &bits,
               packet_length -
                   offsetof(iree_vm_bytecode_constant_i32_t, bits_u32));
      }
    } else {
      for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
        const loom_low_immediate_t* immediate =
            &frame->target.descriptor_set
                 ->immediates[descriptor->immediate_start + i];
        const loom_attribute_t value = attributes.entries[i].value;
        uint64_t bits;
        if (immediate->kind == LOOM_LOW_IMMEDIATE_KIND_ORDINAL) {
          IREE_RETURN_IF_ERROR(loom_vm_function_rodata(frame->module, value,
                                                       module_plan, &bits));
        } else {
          bits = (uint64_t)value.i64;
        }
        if (immediate->bit_width <= 8) {
          // Packed selector components share a zero-initialized packet byte.
          packet[immediate->encoding_field_id] |=
              (uint8_t)(bits << immediate->encoding_id);
        } else {
          memcpy(packet + immediate->encoding_field_id, &bits,
                 immediate->bit_width / 8);
        }
      }
    }
  }
  return iree_io_stream_write(stream, packet_length, packet);
}

// Transfers whole value cells between physical registers and frame-local
// bytes. Core lane groups preserve every scalar format without byte packing.
static iree_status_t loom_vm_function_transfer_stack(iree_io_stream_t* stream,
                                                     uint16_t byte_offset,
                                                     uint16_t register_base,
                                                     uint16_t count,
                                                     bool is_store) {
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < count && iree_status_is_ok(status);) {
    const uint8_t lane_log2 =
        (uint8_t)iree_min(3, 31 - iree_math_count_leading_zeros_u32(count - i));
    const uint8_t format = IREE_VM_BYTECODE_MEMORY_FORMAT_I64_X1 + lane_log2;
    if (is_store) {
      const iree_vm_bytecode_stack_store_t instruction = {
          .opcode = IREE_VM_BYTECODE_OPCODE_STACK_STORE,
          .base_u16 = byte_offset + i * sizeof(uint64_t),
          .source_v8 = (uint8_t)(register_base + i),
          .format_u8 = format,
      };
      status = iree_io_stream_write(stream, sizeof(instruction), &instruction);
    } else {
      const iree_vm_bytecode_stack_load_t instruction = {
          .opcode = IREE_VM_BYTECODE_OPCODE_STACK_LOAD,
          .destination_v8 = (uint8_t)(register_base + i),
          .base_u16 = byte_offset + i * sizeof(uint64_t),
          .format_u8 = format,
      };
      status = iree_io_stream_write(stream, sizeof(instruction), &instruction);
    }
    i += 1u << lane_log2;
  }
  return status;
}

// Ref-slot transfers share a register/slot encoding. The selected opcode states
// whether a transfer retains an owner or moves and clears its source.
static iree_status_t loom_vm_function_transfer_refs(iree_io_stream_t* stream,
                                                    uint8_t opcode,
                                                    uint16_t slot_base,
                                                    uint16_t register_base,
                                                    uint16_t count) {
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const iree_vm_bytecode_ref_stack_load_retain_t instruction = {
        .opcode = opcode,
        .destination_r8 = (uint8_t)(register_base + i),
        .slot_u16 = slot_base + i,
    };
    status = iree_io_stream_write(stream, sizeof(instruction), &instruction);
  }
  return status;
}

typedef struct loom_vm_call_bank_t {
  // Source-ordered physical argument locations, including overflow fields.
  uint8_t* arguments;
  // Source-ordered physical result locations, including overflow fields.
  uint8_t* results;
  // Number of logical arguments carried by this bank.
  uint16_t argument_count;
  // Number of logical results carried by this bank.
  uint16_t result_count;
} loom_vm_call_bank_t;

typedef struct loom_vm_call_scratch_t {
  // Reusable source-ordered physical locations, partitioned by register bank.
  loom_vm_call_bank_t banks[2];
  // Byte base of ordinary local storage after the largest outgoing packet.
  uint16_t local_base;
  // Ref-slot base of caller snapshots after the largest outgoing ref packet.
  uint32_t ref_base;
} loom_vm_call_scratch_t;

// The scheduler retains call sites; module collection retains their signatures.
// Reserve the canonical offset-zero packet before projecting local storage.
static iree_status_t loom_vm_function_prepare_calls(
    const loom_low_emission_frame_t* frame,
    const loom_vm_module_plan_t* functions, iree_arena_allocator_t* arena,
    loom_vm_call_scratch_t* scratch) {
  uint16_t max_arguments = 0, max_results = 0;
  uint32_t packet_bytes = 0;
  uint32_t packet_refs = 0;
  for (iree_host_size_t i = 0; i < frame->schedule.call_node_count; ++i) {
    const loom_low_schedule_node_t* node =
        &frame->schedule.nodes[frame->schedule.call_node_indices[i]];
    const loom_vm_module_callable_t* binding =
        functions
            ->bindings_by_symbol[loom_low_func_call_callee(node->op).symbol_id];
    // Open source declarations remain legal until an executable is requested.
    // Validate that boundary once before emitting calls from this schedule.
    if (!binding) {
      const loom_symbol_t* symbol =
          &frame->module->symbols
               .entries[loom_low_func_call_callee(node->op).symbol_id];
      const iree_string_view_t name =
          loom_string_table_get(&frame->module->strings, symbol->name_id);
      return iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "VM callee '@%.*s' has no definition or runtime import",
          (int)name.size, name.data);
    }
    const iree_vm_bytecode_v0_signature_row_t* signature =
        &binding->signature.row;
    const uint32_t argument_overflow =
        signature->argument_value_count_u16 -
        iree_min(16, signature->argument_value_count_u16);
    const uint32_t result_overflow =
        signature->result_value_count_u16 -
        iree_min(16, signature->result_value_count_u16);
    packet_bytes = iree_max(
        packet_bytes, (argument_overflow + result_overflow) * sizeof(uint64_t));
    packet_refs = iree_max(packet_refs,
                           signature->argument_ref_count_u16 -
                               iree_min(16, signature->argument_ref_count_u16) +
                               signature->result_ref_count_u16 -
                               iree_min(16, signature->result_ref_count_u16));
    max_arguments = iree_max(max_arguments, node->operand_count);
    max_results = iree_max(max_results, node->result_count);
  }
  if (packet_bytes > UINT16_MAX ||
      frame->schedule.requirements.storage_layout.space_sizes.stack_bytes >
          UINT16_MAX - packet_bytes) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM local storage exceeds 65535 bytes");
  }
  scratch->local_base = (uint16_t)packet_bytes;
  scratch->ref_base = packet_refs;
  if (max_arguments || max_results) {
    uint8_t* locations = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, 2u * (max_arguments + max_results),
                                  sizeof(*locations), (void**)&locations));
    for (uint16_t bank = 0; bank < 2; ++bank) {
      scratch->banks[bank].arguments = locations;
      locations += max_arguments;
      scratch->banks[bank].results = locations;
      locations += max_results;
    }
  }
  return iree_ok_status();
}

// Captures each allocated location once while partitioning logical operands
// and results into the independent architectural banks.
static void loom_vm_function_call_bindings(
    const loom_low_emission_frame_t* frame,
    const loom_low_schedule_node_t* node, loom_vm_call_bank_t* banks) {
  for (uint16_t bank = 0; bank < 2; ++bank) {
    banks[bank].argument_count = banks[bank].result_count = 0;
  }
  for (uint16_t side = 0; side < 2; ++side) {
    const uint16_t count = side ? node->result_count : node->operand_count;
    const loom_value_ordinal_t* ordinals =
        side ? loom_low_schedule_node_const_result_ordinals(node)
             : loom_low_schedule_node_const_operand_ordinals(node);
    for (uint16_t i = 0; i < count; ++i) {
      const loom_low_allocation_assignment_t* assignment =
          loom_low_allocation_assignment_for_value_ordinal(&frame->allocation,
                                                           ordinals[i], NULL);
      loom_vm_call_bank_t* bank = &banks[assignment->descriptor_reg_class_id];
      uint16_t* bank_count = side ? &bank->result_count : &bank->argument_count;
      (side ? bank->results : bank->arguments)[(*bank_count)++] =
          (uint8_t)assignment->location_base;
    }
  }
}

// Calls overwrite only the argument/result prefix. Marshal from a snapshot so
// argument permutations cannot clobber their sources, then overlay the results
// at their allocated positions before restoring live caller values. Registers
// outside the prefix survive the call directly. No leaf storage is reserved.
// Keep bank marshalling in its own optimization scope, separate from frame
// construction and the outer block-emission loop.
IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_vm_function_call(
    const loom_low_emission_frame_t* frame,
    const loom_low_schedule_node_t* node,
    const loom_vm_module_plan_t* functions, loom_vm_call_scratch_t* scratch,
    iree_io_stream_t* stream, iree_vm_bytecode_v0_function_row_t* out_row) {
  const loom_vm_module_callable_t* binding =
      functions
          ->bindings_by_symbol[loom_low_func_call_callee(node->op).symbol_id];
  loom_vm_call_bank_t* banks = scratch->banks;
  loom_vm_function_call_bindings(frame, node, banks);
  const loom_vm_call_bank_t* values = &banks[VM_CORE_REG_CLASS_ID_VALUE];
  const loom_vm_call_bank_t* refs = &banks[VM_CORE_REG_CLASS_ID_REF];
  const uint16_t prefix_count =
      iree_min(16, iree_max(values->argument_count, values->result_count));
  const uint16_t ref_prefix_count =
      iree_min(16, iree_max(refs->argument_count, refs->result_count));
  const uint32_t local_ref_count = scratch->ref_base + ref_prefix_count;
  if (local_ref_count > (uint32_t)UINT16_MAX + 1) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM call scratch exceeds the local ref-slot limit");
  }
  const uint16_t ref_base = (uint16_t)scratch->ref_base;
  const uint32_t byte_offset = iree_host_align(
      scratch->local_base +
          frame->schedule.requirements.storage_layout.space_sizes.stack_bytes,
      sizeof(uint64_t));
  const uint32_t local_byte_length =
      byte_offset + prefix_count * sizeof(uint64_t);
  if (local_byte_length > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM call scratch exceeds the local byte limit");
  }
  out_row->flags_u16 |= IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL;
  out_row->value_register_count_u16 =
      iree_max(out_row->value_register_count_u16, prefix_count);
  out_row->local_byte_length_u16 =
      iree_max(out_row->local_byte_length_u16, local_byte_length);
  out_row->ref_register_count_u16 =
      iree_max(out_row->ref_register_count_u16, ref_prefix_count);
  out_row->local_ref_count_u32 =
      iree_max(out_row->local_ref_count_u32, local_ref_count);
  iree_status_t status = loom_vm_function_transfer_stack(stream, byte_offset, 0,
                                                         prefix_count, true);
  if (iree_status_is_ok(status)) {
    status = loom_vm_function_transfer_refs(
        stream, IREE_VM_BYTECODE_OPCODE_REF_STACK_STORE_RETAIN, ref_base, 0,
        ref_prefix_count);
  }
  // Save overflow before rearranging any source in the direct prefix.
  for (uint16_t i = 16; i < values->argument_count && iree_status_is_ok(status);
       ++i) {
    status = loom_vm_function_transfer_stack(
        stream, (i - 16) * sizeof(uint64_t), values->arguments[i], 1, true);
  }
  for (uint16_t i = 0;
       i < iree_min(16, values->argument_count) && iree_status_is_ok(status);
       ++i) {
    const uint8_t source = values->arguments[i];
    if (source == i) {
      continue;
    }
    if (source < prefix_count) {
      status = loom_vm_function_transfer_stack(
          stream, byte_offset + source * sizeof(uint64_t), i, 1, false);
    } else {
      const iree_vm_bytecode_value_copy_t instruction = {
          .opcode = IREE_VM_BYTECODE_OPCODE_VALUE_COPY,
          .destination_v8 = (uint8_t)i,
          .source_v8 = source,
      };
      status = iree_io_stream_write(stream, sizeof(instruction), &instruction);
    }
  }
  for (uint16_t i = 16; i < refs->argument_count && iree_status_is_ok(status);
       ++i) {
    status = loom_vm_function_transfer_refs(
        stream, IREE_VM_BYTECODE_OPCODE_REF_STACK_STORE_RETAIN, i - 16,
        refs->arguments[i], 1);
  }
  for (uint16_t i = 0;
       i < iree_min(16, refs->argument_count) && iree_status_is_ok(status);
       ++i) {
    const uint8_t source = refs->arguments[i];
    if (source == i) {
      continue;
    }
    if (source < ref_prefix_count) {
      status = loom_vm_function_transfer_refs(
          stream, IREE_VM_BYTECODE_OPCODE_REF_STACK_LOAD_RETAIN,
          ref_base + source, i, 1);
    } else {
      const iree_vm_bytecode_ref_retain_t instruction = {
          .opcode = IREE_VM_BYTECODE_OPCODE_REF_RETAIN,
          .destination_r8 = (uint8_t)i,
          .source_r8 = source,
      };
      status = iree_io_stream_write(stream, sizeof(instruction), &instruction);
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_vm_bytecode_control_call_t instruction = {
        .opcode = IREE_VM_BYTECODE_OPCODE_CONTROL_CALL,
        .target_kind_u8 = binding->target_kind,
        .target_ordinal_u16 = binding->ordinal,
        .direct_ref_move_mask_u16 =
            (uint16_t)((1u << iree_min(16, refs->argument_count)) - 1),
    };
    status = iree_io_stream_write(stream, sizeof(instruction), &instruction);
  }
  for (uint16_t i = 0;
       i < iree_min(16, values->result_count) && iree_status_is_ok(status);
       ++i) {
    const uint8_t destination = values->results[i];
    if (destination < prefix_count) {
      status = loom_vm_function_transfer_stack(
          stream, byte_offset + destination * sizeof(uint64_t), i, 1, true);
    } else {
      const iree_vm_bytecode_value_copy_t instruction = {
          .opcode = IREE_VM_BYTECODE_OPCODE_VALUE_COPY,
          .destination_v8 = destination,
          .source_v8 = (uint8_t)i,
      };
      status = iree_io_stream_write(stream, sizeof(instruction), &instruction);
    }
  }
  for (uint16_t i = 0;
       i < iree_min(16, refs->result_count) && iree_status_is_ok(status); ++i) {
    const uint8_t destination = refs->results[i];
    if (destination < ref_prefix_count) {
      status = loom_vm_function_transfer_refs(
          stream, IREE_VM_BYTECODE_OPCODE_REF_STACK_STORE_MOVE,
          ref_base + destination, i, 1);
    } else {
      const iree_vm_bytecode_ref_move_t instruction = {
          .opcode = IREE_VM_BYTECODE_OPCODE_REF_MOVE,
          .destination_r8 = destination,
          .source_r8 = (uint8_t)i,
      };
      status = iree_io_stream_write(stream, sizeof(instruction), &instruction);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_vm_function_transfer_stack(stream, byte_offset, 0,
                                             prefix_count, false);
  }
  if (iree_status_is_ok(status)) {
    status = loom_vm_function_transfer_refs(
        stream, IREE_VM_BYTECODE_OPCODE_REF_STACK_LOAD_MOVE, ref_base, 0,
        ref_prefix_count);
  }
  const uint16_t argument_overflow =
      values->argument_count - iree_min(16, values->argument_count);
  for (uint16_t i = 16; i < values->result_count && iree_status_is_ok(status);
       ++i) {
    status = loom_vm_function_transfer_stack(
        stream, (argument_overflow + i - 16) * sizeof(uint64_t),
        values->results[i], 1, false);
  }
  const uint16_t ref_argument_overflow =
      refs->argument_count - iree_min(16, refs->argument_count);
  for (uint16_t i = 16; i < refs->result_count && iree_status_is_ok(status);
       ++i) {
    status = loom_vm_function_transfer_refs(
        stream, IREE_VM_BYTECODE_OPCODE_REF_STACK_LOAD_MOVE,
        ref_argument_overflow + i - 16, refs->results[i], 1);
  }
  return status;
}

static iree_status_t loom_vm_function_storage(
    const loom_low_emission_frame_t* frame,
    const loom_low_schedule_node_t* node, uint16_t local_base,
    iree_io_stream_t* stream) {
  const bool is_store = loom_low_spill_isa(node->op);
  const loom_value_id_t storage_value = is_store
                                            ? loom_low_spill_storage(node->op)
                                            : loom_low_reload_storage(node->op);
  const uint64_t relative_offset = is_store ? loom_low_spill_offset(node->op)
                                            : loom_low_reload_offset(node->op);
  loom_low_storage_layout_reference_t reference;
  loom_low_storage_layout_lookup_reference(
      &frame->schedule.requirements.storage_layout, frame->module,
      storage_value, &reference);
  const loom_value_ordinal_t value_ordinal =
      (is_store ? loom_low_schedule_node_const_operand_ordinals(node)
                : loom_low_schedule_node_const_result_ordinals(node))[0];
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_assignment_for_value_ordinal(&frame->allocation,
                                                       value_ordinal, NULL);
  // Low verification establishes the byte offset. The target allocation unit
  // determines the transfer width, including padding in narrow scalar cells.
  if (assignment->location_count * sizeof(uint64_t) >
      reference.byte_length - relative_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM register transfer exceeds its storage span");
  }
  return loom_vm_function_transfer_stack(
      stream,
      (uint16_t)(local_base + reference.reservation.byte_offset +
                 reference.byte_offset + relative_offset),
      (uint16_t)assignment->location_base, (uint16_t)assignment->location_count,
      is_store);
}

// Static register parameters occupy the entry prefix of the local value
// domain. Unused overflow arguments have no assignment and need no load.
static iree_status_t loom_vm_function_arguments(
    const loom_low_emission_frame_t* frame,
    const loom_vm_function_signature_t* signature, uint16_t argument_count,
    iree_io_stream_t* stream) {
  uint16_t ordinals_by_bank[2] = {0};
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < argument_count && iree_status_is_ok(status); ++i) {
    const bool is_ref =
        signature->fields[i].kind_u16 == IREE_VM_BYTECODE_SIGNATURE_KIND_REF;
    const uint16_t ordinal = ordinals_by_bank[is_ref]++;
    if (ordinal < 16) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_assignment_for_value_ordinal(&frame->allocation, i,
                                                         NULL);
    if (!assignment) {
      continue;
    }
    const iree_vm_bytecode_value_abi_argument_load_t instruction = {
        .opcode = is_ref ? IREE_VM_BYTECODE_OPCODE_REF_ABI_ARGUMENT_LOAD_MOVE
                         : IREE_VM_BYTECODE_OPCODE_VALUE_ABI_ARGUMENT_LOAD,
        .destination_v8 = (uint8_t)assignment->location_base,
        .slot_u16 = ordinal - 16,
    };
    status = iree_io_stream_write(stream, sizeof(instruction), &instruction);
  }
  return status;
}

iree_status_t loom_vm_function_emit(
    const loom_target_emit_request_t* request, loom_func_like_t function,
    const loom_target_function_version_t* function_version,
    const loom_vm_function_signature_t* signature,
    loom_vm_module_plan_t* functions, iree_io_stream_t* stream,
    bool* out_emitted, iree_vm_bytecode_v0_function_row_t* out_row) {
  *out_emitted = false;
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  loom_low_allocation_fixed_value_t fixed_values[32];
  uint16_t value_ordinal = 0;
  uint16_t ref_ordinal = 0;
  uint16_t fixed_count = 0;
  for (uint16_t i = 0; i < argument_count; ++i) {
    const uint16_t ordinal =
        signature->fields[i].kind_u16 == IREE_VM_BYTECODE_SIGNATURE_KIND_REF
            ? ref_ordinal++
            : value_ordinal++;
    if (ordinal >= 16 ||
        !loom_module_value_has_uses(request->module, arguments[i])) {
      continue;
    }
    fixed_values[fixed_count++] = (loom_low_allocation_fixed_value_t){
        .value_id = arguments[i],
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .location_base = ordinal,
        .location_count = 1,
    };
  }
  const loom_low_emission_frame_options_t options = {
      .descriptor_registry = request->low_descriptor_registry,
      .function_target_facts = function_version != NULL
                                   ? function_version->function_target_facts
                                   : NULL,
      .memory_accesses =
          function_version != NULL ? function_version->memory_accesses : NULL,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .allocation_fixed_values = fixed_values,
      .allocation_fixed_value_count = fixed_count,
      .emitter = request->diagnostic_emitter,
  };
  const loom_low_emission_frame_spill_free_options_t spill_options = {
      .materialization_options =
          {
              .has_supported_storage_spaces = true,
              .supported_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_STACK,
          },
  };
  loom_low_emission_frame_t frame = {0};
  bool frame_accepted = false;
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_build_spill_free(
      request->module, function.op, &options, &spill_options,
      request->scratch_arena, &frame, &frame_accepted));
  if (!frame_accepted) {
    return iree_ok_status();
  }
  const loom_low_storage_layout_space_sizes_t local_storage =
      frame.schedule.requirements.storage_layout.space_sizes;
  if (local_storage.scratch_bytes || local_storage.private_bytes ||
      local_storage.workgroup_bytes) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "VM local storage requires the stack space");
  }
  // Preparation writes the layout and location arrays. Each emitted call
  // initializes its own bank counts; leaves never access the banks.
  loom_vm_call_scratch_t call_scratch;
  IREE_RETURN_IF_ERROR(loom_vm_function_prepare_calls(
      &frame, functions, request->scratch_arena, &call_scratch));
  out_row->local_byte_length_u16 =
      call_scratch.local_base + (uint16_t)local_storage.stack_bytes;
  out_row->value_register_count_u16 = (uint16_t)iree_max(
      iree_min(16, iree_max(signature->row.argument_value_count_u16,
                            signature->row.result_value_count_u16)),
      frame.allocation.physical_extents
          .ends_by_reg_class[VM_CORE_REG_CLASS_ID_VALUE]);
  out_row->ref_register_count_u16 = (uint16_t)iree_max(
      iree_min(16, iree_max(signature->row.argument_ref_count_u16,
                            signature->row.result_ref_count_u16)),
      frame.allocation.physical_extents
          .ends_by_reg_class[VM_CORE_REG_CLASS_ID_REF]);
  out_row->block_count_u32 = (uint32_t)frame.schedule.block_count;
  loom_low_move_sequence_scratch_t return_scratch = {0};
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_initialize(
      request->scratch_arena,
      iree_min(16, signature->row.result_value_count_u16) +
          iree_min(16, signature->row.result_ref_count_u16),
      &return_scratch));
  const loom_cfg_graph_t* graph = &frame.schedule.cfg_graph;
  iree_io_stream_pos_t* block_offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->scratch_arena, graph->block_count, sizeof(*block_offsets),
      (void**)&block_offsets));
  loom_vm_branch_fixup_t* fixups = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(request->scratch_arena, graph->edge_count,
                                sizeof(*fixups), (void**)&fixups));
  iree_host_size_t fixup_count = 0;
  iree_host_size_t edge_copy_index = 0;

  const iree_io_stream_pos_t start = iree_io_stream_offset(stream);
  iree_status_t status = iree_ok_status();
  if (signature->row.argument_value_count_u16 > 16 ||
      signature->row.argument_ref_count_u16 > 16) {
    // The ABI prologue executes once. Branches to the first body block carry
    // their own arguments and must not reload the original invocation inputs.
    const iree_vm_bytecode_control_block_t instruction = {
        .opcode = IREE_VM_BYTECODE_OPCODE_CONTROL_BLOCK,
    };
    status = iree_io_stream_write(stream, sizeof(instruction), &instruction);
    if (iree_status_is_ok(status)) {
      status =
          loom_vm_function_arguments(&frame, signature, argument_count, stream);
    }
    ++out_row->block_count_u32;
  }
  for (iree_host_size_t b = 0;
       b < frame.schedule.block_count && iree_status_is_ok(status); ++b) {
    const loom_low_schedule_block_t* block = &frame.schedule.blocks[b];
    block_offsets[b] = iree_io_stream_offset(stream);
    const iree_vm_bytecode_control_block_t instruction = {
        .opcode = IREE_VM_BYTECODE_OPCODE_CONTROL_BLOCK,
    };
    status = iree_io_stream_write(stream, sizeof(instruction), &instruction);
    for (uint32_t i = 0;
         i < block->scheduled_node_count && iree_status_is_ok(status); ++i) {
      const uint32_t node_index =
          frame.schedule
              .scheduled_node_indices[block->scheduled_node_start + i];
      const loom_low_schedule_node_t* node = &frame.schedule.nodes[node_index];
      if (node->descriptor) {
        status = loom_vm_function_packet(&frame, node, functions, stream);
      } else if (loom_low_func_call_isa(node->op)) {
        status = loom_vm_function_call(&frame, node, functions, &call_scratch,
                                       stream, out_row);
      } else if (loom_low_storage_reserve_isa(node->op) ||
                 loom_low_storage_view_isa(node->op)) {
        // Storage layout is already fixed by the shared scheduler.
      } else if (loom_low_spill_isa(node->op) ||
                 loom_low_reload_isa(node->op)) {
        status = loom_vm_function_storage(&frame, node, call_scratch.local_base,
                                          stream);
      } else if (loom_low_return_isa(node->op)) {
        status = loom_vm_function_return(&frame, node, &return_scratch, stream,
                                         signature, out_row);
      } else if (loom_low_br_isa(node->op)) {
        // Allocation records one group for each payload-bearing branch, in
        // source block order, including groups whose moves all coalesced.
        if (node->operand_count) {
          const loom_low_move_range_t range =
              frame.allocation.edge_copy_groups[edge_copy_index++]
                  .move_group.moves;
          status =
              loom_vm_function_moves(frame.allocation.moves, range, stream);
        }
        const uint16_t target =
            graph->successor_indices[graph->blocks[b].successor_start];
        if (iree_status_is_ok(status) && target != b + 1) {
          status = loom_vm_function_branch(
              stream, IREE_VM_BYTECODE_OPCODE_CONTROL_BRANCH_S32, 0, target,
              &fixups[fixup_count++]);
        }
      } else if (loom_low_cond_br_isa(node->op)) {
        const uint16_t* targets =
            graph->successor_indices + graph->blocks[b].successor_start;
        const loom_value_ordinal_t condition_ordinal =
            loom_low_schedule_node_const_operand_ordinals(node)[0];
        const uint8_t condition =
            (uint8_t)loom_low_allocation_assignment_for_value_ordinal(
                &frame.allocation, condition_ordinal, NULL)
                ->location_base;
        // Invert the condition when the true edge can fall through. Otherwise
        // branch on true and emit the false jump only if it cannot fall
        // through.
        if (targets[0] == b + 1) {
          status = loom_vm_function_branch(
              stream, IREE_VM_BYTECODE_OPCODE_CONTROL_BRANCH_UNLESS_S32,
              condition, targets[1], &fixups[fixup_count++]);
        } else {
          status = loom_vm_function_branch(
              stream, IREE_VM_BYTECODE_OPCODE_CONTROL_BRANCH_IF_S32, condition,
              targets[0], &fixups[fixup_count++]);
          if (iree_status_is_ok(status) && targets[1] != b + 1) {
            status = loom_vm_function_branch(
                stream, IREE_VM_BYTECODE_OPCODE_CONTROL_BRANCH_S32, 0,
                targets[1], &fixups[fixup_count++]);
          }
        }
      } else if (loom_low_copy_isa(node->op) || loom_low_move_isa(node->op) ||
                 loom_low_slice_isa(node->op) ||
                 loom_low_concat_isa(node->op)) {
        const loom_low_allocation_packet_move_group_t* group =
            loom_low_allocation_find_packet_move_group_by_source_ordinal(
                &frame.allocation, node->source_ordinal);
        if (group) {
          const loom_low_move_range_t range = group->move_group.moves;
          status =
              loom_vm_function_moves(frame.allocation.moves, range, stream);
        }
      } else {
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "VM instruction emission for '%.*s' is unavailable",
            (int)loom_op_name(request->module, node->op).size,
            loom_op_name(request->module, node->op).data);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_io_stream_pos_t end = iree_io_stream_offset(stream);
    const iree_io_stream_pos_t length = end - start;
    if (length > INT32_MAX) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "VM function bytecode exceeds signed 32-bit extent");
    } else {
      out_row->bytecode_length_u32 = (uint32_t)length;
    }
    for (iree_host_size_t i = 0; i < fixup_count && iree_status_is_ok(status);
         ++i) {
      const loom_vm_branch_fixup_t fixup = fixups[i];
      const int32_t displacement =
          (int32_t)((block_offsets[fixup.target_block] - fixup.offset -
                     (iree_io_stream_pos_t)sizeof(int32_t)) /
                    4);
      status =
          iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, fixup.offset);
      if (iree_status_is_ok(status)) {
        status =
            iree_io_stream_write(stream, sizeof(displacement), &displacement);
      }
    }
    if (fixup_count && iree_status_is_ok(status)) {
      status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, end);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_emitted = true;
  }
  return status;
}
