// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/function_locals.h"

#include <string.h>

#include "iree/vm/bytecode/wire/core/function.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/core/ref.h"
#include "iree/vm/bytecode/wire/core/selectors.h"
#include "iree/vm/bytecode/wire/core/stack.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/vm/descriptors.h"

enum {
  LOOM_VM_FUNCTION_VALUE_LOCAL_UNIT_BYTE_LENGTH = 8,
  LOOM_VM_FUNCTION_REF_LOCAL_UNIT_BYTE_LENGTH = 16,
  LOOM_VM_FUNCTION_FUNCTION_LOCAL_UNIT_BYTE_LENGTH = 16,
};

// Register and storage fields carried by one structural local transfer.
typedef struct loom_vm_function_local_transfer_t {
  // True for low.reload and false for low.spill.
  bool is_reload;
  // Storage handle referenced by the transfer.
  loom_value_id_t storage_value_id;
  // Byte offset relative to |storage_value_id|.
  uint64_t byte_offset;
  // Physical register assignment transferred to or from storage.
  const loom_low_allocation_assignment_t* assignment;
  // VM frame-local bank selected by |assignment|.
  loom_vm_function_local_bank_t bank;
  // Byte length of one allocation unit in |bank|.
  uint32_t unit_byte_length;
} loom_vm_function_local_transfer_t;

static const loom_low_allocation_assignment_t*
loom_vm_function_local_transfer_assignment(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet) {
  const loom_value_ordinal_t value_ordinal =
      loom_low_spill_isa(packet->node->op)
          ? loom_low_schedule_node_const_operand_ordinals(packet->node)[0]
          : loom_low_schedule_node_const_result_ordinals(packet->node)[0];
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_assignment_for_value_ordinal(&frame->allocation,
                                                       value_ordinal, NULL);
  IREE_ASSERT(assignment != NULL);
  return assignment;
}

static void loom_vm_function_local_classify_assignment(
    const loom_low_allocation_assignment_t* assignment,
    loom_vm_function_local_bank_t* out_bank, uint32_t* out_unit_byte_length) {
  switch (assignment->descriptor_reg_class_id) {
    case VM_CORE_REG_CLASS_ID_VALUE:
      *out_bank = LOOM_VM_FUNCTION_LOCAL_BANK_VALUE;
      *out_unit_byte_length = LOOM_VM_FUNCTION_VALUE_LOCAL_UNIT_BYTE_LENGTH;
      return;
    case VM_CORE_REG_CLASS_ID_REF:
      *out_bank = LOOM_VM_FUNCTION_LOCAL_BANK_REF;
      *out_unit_byte_length = LOOM_VM_FUNCTION_REF_LOCAL_UNIT_BYTE_LENGTH;
      return;
    case VM_CORE_REG_CLASS_ID_FUNCTION:
      *out_bank = LOOM_VM_FUNCTION_LOCAL_BANK_FUNCTION;
      *out_unit_byte_length = LOOM_VM_FUNCTION_FUNCTION_LOCAL_UNIT_BYTE_LENGTH;
      return;
    default:
      IREE_ASSERT_UNREACHABLE("verified VM register class");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static void loom_vm_function_local_transfer_view(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet,
    loom_vm_function_local_transfer_t* out_transfer) {
  const loom_op_t* op = packet->node->op;
  IREE_ASSERT(loom_low_spill_isa(op) || loom_low_reload_isa(op));
  const bool is_reload = loom_low_reload_isa(op);
  const loom_low_allocation_assignment_t* assignment =
      loom_vm_function_local_transfer_assignment(frame, packet);
  loom_vm_function_local_bank_t bank = LOOM_VM_FUNCTION_LOCAL_BANK_NONE;
  uint32_t unit_byte_length = 0;
  loom_vm_function_local_classify_assignment(assignment, &bank,
                                             &unit_byte_length);
  *out_transfer = (loom_vm_function_local_transfer_t){
      .is_reload = is_reload,
      .storage_value_id =
          is_reload ? loom_low_reload_storage(op) : loom_low_spill_storage(op),
      .byte_offset = (uint64_t)(is_reload ? loom_low_reload_offset(op)
                                          : loom_low_spill_offset(op)),
      .assignment = assignment,
      .bank = bank,
      .unit_byte_length = unit_byte_length,
  };
}

bool loom_vm_function_local_transfer_is_packet(
    const loom_low_packet_view_t* packet) {
  return loom_low_spill_isa(packet->node->op) ||
         loom_low_reload_isa(packet->node->op);
}

static iree_status_t loom_vm_function_local_classify_transfer(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet,
    loom_vm_function_local_reservation_t* reservations,
    iree_host_size_t reservation_count) {
  loom_vm_function_local_transfer_t transfer;
  loom_vm_function_local_transfer_view(frame, packet, &transfer);
  IREE_ASSERT_EQ(transfer.assignment->location_kind,
                 LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  IREE_ASSERT_NE(transfer.assignment->location_count, 0u);

  loom_low_storage_layout_reference_t reference;
  loom_low_storage_layout_lookup_reference(
      &frame->schedule.storage_layout, frame->module, transfer.storage_value_id,
      &reference);
  if (reference.reservation.space != LOOM_STORAGE_SPACE_PRIVATE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "VM local transfers require low.storage<private>");
  }

  uint64_t access_byte_length = 0;
  if (!iree_checked_mul_u64(transfer.assignment->location_count,
                            transfer.unit_byte_length, &access_byte_length) ||
      transfer.byte_offset > reference.byte_length ||
      access_byte_length > reference.byte_length - transfer.byte_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM local transfer exceeds its storage view");
  }
  uint64_t root_byte_offset = 0;
  if (!iree_checked_add_u64(reference.byte_offset, transfer.byte_offset,
                            &root_byte_offset) ||
      (transfer.bank != LOOM_VM_FUNCTION_LOCAL_BANK_VALUE &&
       root_byte_offset % transfer.unit_byte_length != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM ref and function local transfers must be cell-aligned");
  }

  IREE_ASSERT_LT(reference.reservation_ordinal, reservation_count);
  loom_vm_function_local_reservation_t* reservation =
      &reservations[reference.reservation_ordinal];
  if (reservation->bank != LOOM_VM_FUNCTION_LOCAL_BANK_NONE &&
      reservation->bank != transfer.bank) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "one Low storage reservation cannot span multiple VM local banks");
  }
  reservation->bank = transfer.bank;
  return iree_ok_status();
}

static iree_status_t loom_vm_function_local_append_value_reservation(
    const loom_low_storage_layout_reservation_t* source,
    loom_vm_function_local_reservation_t* target,
    loom_vm_function_local_counts_t* counts) {
  uint64_t base = 0;
  if (!iree_checked_align_u64(counts->byte_length, source->byte_alignment,
                              &base) ||
      base > UINT16_MAX || source->byte_size > UINT16_MAX - base) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VM function local-byte storage exceeds the u16 format limit");
  }
  target->base = (uint32_t)base;
  counts->byte_length = (uint16_t)(base + source->byte_size);
  return iree_ok_status();
}

static iree_status_t loom_vm_function_local_append_cell_reservation(
    const loom_low_storage_layout_reservation_t* source,
    uint32_t unit_byte_length, loom_vm_function_local_reservation_t* target,
    uint32_t* count) {
  uint64_t padded_byte_size = 0;
  if (!iree_checked_add_u64(source->byte_size, unit_byte_length - 1u,
                            &padded_byte_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM function local-cell storage overflows");
  }
  const uint64_t cell_count = padded_byte_size / unit_byte_length;
  const uint64_t next_count = (uint64_t)*count + cell_count;
  if (next_count > (uint64_t)UINT16_MAX + 1u) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VM function local-cell storage exceeds the u16 ordinal limit");
  }
  target->base = *count;
  *count = (uint32_t)next_count;
  return iree_ok_status();
}

iree_status_t loom_vm_function_local_layout_build(
    const loom_low_emission_frame_t* frame,
    loom_vm_function_local_counts_t call_prefix, iree_arena_allocator_t* arena,
    loom_vm_function_local_layout_t* out_layout) {
  *out_layout = (loom_vm_function_local_layout_t){0};
  const loom_low_storage_layout_t* storage_layout =
      &frame->schedule.storage_layout;
  loom_vm_function_local_reservation_t* reservations = NULL;
  if (storage_layout->record_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, storage_layout->record_count, sizeof(*reservations),
        (void**)&reservations));
    memset(reservations, 0,
           storage_layout->record_count * sizeof(*reservations));
  }

  for (iree_host_size_t packet_index = 0;
       packet_index < frame->schedule.scheduled_node_count; ++packet_index) {
    const loom_low_packet_view_t packet =
        loom_low_packet_at(&frame->schedule, packet_index);
    if (!loom_vm_function_local_transfer_is_packet(&packet)) continue;
    IREE_RETURN_IF_ERROR(loom_vm_function_local_classify_transfer(
        frame, &packet, reservations, storage_layout->record_count));
  }

  loom_vm_function_local_counts_t counts = call_prefix;
  for (iree_host_size_t i = 0; i < storage_layout->record_count; ++i) {
    loom_vm_function_local_reservation_t* target = &reservations[i];
    const loom_low_storage_layout_reservation_t* source =
        &storage_layout->records[i].reservation;
    switch (target->bank) {
      case LOOM_VM_FUNCTION_LOCAL_BANK_NONE:
        break;
      case LOOM_VM_FUNCTION_LOCAL_BANK_VALUE: {
        IREE_RETURN_IF_ERROR(loom_vm_function_local_append_value_reservation(
            source, target, &counts));
        break;
      }
      case LOOM_VM_FUNCTION_LOCAL_BANK_REF: {
        IREE_RETURN_IF_ERROR(loom_vm_function_local_append_cell_reservation(
            source, LOOM_VM_FUNCTION_REF_LOCAL_UNIT_BYTE_LENGTH, target,
            &counts.ref_count));
        break;
      }
      case LOOM_VM_FUNCTION_LOCAL_BANK_FUNCTION: {
        IREE_RETURN_IF_ERROR(loom_vm_function_local_append_cell_reservation(
            source, LOOM_VM_FUNCTION_FUNCTION_LOCAL_UNIT_BYTE_LENGTH, target,
            &counts.function_count));
        break;
      }
      default:
        IREE_ASSERT_UNREACHABLE("classified VM local bank");
        IREE_BUILTIN_UNREACHABLE();
    }
  }

  *out_layout = (loom_vm_function_local_layout_t){
      .counts = counts,
      .reservations = reservations,
      .reservation_count = storage_layout->record_count,
  };
  return iree_ok_status();
}

static uint32_t loom_vm_function_value_transfer_record_count(
    uint32_t unit_count) {
  const uint32_t remainder = unit_count % 8u;
  return unit_count / 8u + ((remainder & 4u) != 0) + ((remainder & 2u) != 0) +
         ((remainder & 1u) != 0);
}

uint32_t loom_vm_function_local_transfer_byte_length(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet) {
  loom_vm_function_local_transfer_t transfer;
  loom_vm_function_local_transfer_view(frame, packet, &transfer);
  switch (transfer.bank) {
    case LOOM_VM_FUNCTION_LOCAL_BANK_VALUE:
      return loom_vm_function_value_transfer_record_count(
                 transfer.assignment->location_count) *
             (transfer.is_reload ? sizeof(iree_vm_isa_stack_load_record_t)
                                 : sizeof(iree_vm_isa_stack_store_record_t));
    case LOOM_VM_FUNCTION_LOCAL_BANK_REF:
      return transfer.assignment->location_count *
             (transfer.is_reload
                  ? sizeof(iree_vm_isa_ref_stack_load_retain_record_t)
                  : sizeof(iree_vm_isa_ref_stack_store_retain_record_t));
    case LOOM_VM_FUNCTION_LOCAL_BANK_FUNCTION:
      return transfer.assignment->location_count *
             (transfer.is_reload
                  ? sizeof(iree_vm_isa_func_stack_load_record_t)
                  : sizeof(iree_vm_isa_func_stack_store_record_t));
    default:
      IREE_ASSERT_UNREACHABLE("classified VM local transfer bank");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static uint32_t loom_vm_function_local_transfer_base(
    const loom_low_emission_frame_t* frame,
    const loom_vm_function_local_layout_t* layout,
    const loom_vm_function_local_transfer_t* transfer) {
  loom_low_storage_layout_reference_t reference;
  loom_low_storage_layout_lookup_reference(
      &frame->schedule.storage_layout, frame->module,
      transfer->storage_value_id, &reference);
  IREE_ASSERT_LT(reference.reservation_ordinal, layout->reservation_count);
  const loom_vm_function_local_reservation_t* reservation =
      &layout->reservations[reference.reservation_ordinal];
  IREE_ASSERT_EQ(reservation->bank, transfer->bank);
  const uint64_t root_byte_offset =
      reference.byte_offset + transfer->byte_offset;
  const uint64_t local_offset =
      transfer->bank == LOOM_VM_FUNCTION_LOCAL_BANK_VALUE
          ? root_byte_offset
          : root_byte_offset / transfer->unit_byte_length;
  IREE_ASSERT_LE((uint64_t)reservation->base + local_offset, UINT16_MAX);
  return reservation->base + (uint32_t)local_offset;
}

static uint8_t loom_vm_function_value_transfer_lane_count(
    uint32_t remaining_unit_count) {
  if (remaining_unit_count >= 8u) return 8u;
  if (remaining_unit_count >= 4u) return 4u;
  if (remaining_unit_count >= 2u) return 2u;
  return 1u;
}

static uint8_t loom_vm_function_value_transfer_format(uint8_t lane_count) {
  switch (lane_count) {
    case 1:
      return IREE_VM_ISA_MEMORY_FORMAT_I64_X1;
    case 2:
      return IREE_VM_ISA_MEMORY_FORMAT_I64_X2;
    case 4:
      return IREE_VM_ISA_MEMORY_FORMAT_I64_X4;
    default:
      IREE_ASSERT_EQ(lane_count, 8u);
      return IREE_VM_ISA_MEMORY_FORMAT_I64_X8;
  }
}

static iree_status_t loom_vm_function_local_value_transfer_encode(
    const loom_vm_function_local_transfer_t* transfer, uint32_t local_base,
    loom_bytecode_page_writer_t* writer) {
  uint32_t processed_unit_count = 0;
  while (processed_unit_count < transfer->assignment->location_count) {
    const uint8_t lane_count = loom_vm_function_value_transfer_lane_count(
        transfer->assignment->location_count - processed_unit_count);
    const uint32_t register_base =
        transfer->assignment->location_base + processed_unit_count;
    const uint32_t byte_base =
        local_base +
        processed_unit_count * LOOM_VM_FUNCTION_VALUE_LOCAL_UNIT_BYTE_LENGTH;
    IREE_ASSERT_LT(register_base, 256u);
    IREE_ASSERT_LE(byte_base, UINT16_MAX);
    if (transfer->is_reload) {
      const iree_vm_isa_stack_load_record_t record = {
          .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_LOAD,
          .dst_v8 = (uint8_t)register_base,
          .base_u16 = (uint16_t)byte_base,
          .format_u8 = loom_vm_function_value_transfer_format(lane_count),
          .zero_padding_u8 = {0, 0, 0},
      };
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write(writer, &record, sizeof(record)));
    } else {
      const iree_vm_isa_stack_store_record_t record = {
          .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_STORE,
          .zero_padding_u8 = 0,
          .base_u16 = (uint16_t)byte_base,
          .src_v8 = (uint8_t)register_base,
          .format_u8 = loom_vm_function_value_transfer_format(lane_count),
          .zero_padding_u16 = 0,
      };
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write(writer, &record, sizeof(record)));
    }
    processed_unit_count += lane_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_function_local_ref_transfer_encode(
    const loom_vm_function_local_transfer_t* transfer, uint32_t local_base,
    loom_bytecode_page_writer_t* writer) {
  for (uint32_t i = 0; i < transfer->assignment->location_count; ++i) {
    const uint32_t register_ordinal = transfer->assignment->location_base + i;
    const uint32_t slot_ordinal = local_base + i;
    IREE_ASSERT_LT(register_ordinal, 256u);
    IREE_ASSERT_LE(slot_ordinal, UINT16_MAX);
    if (transfer->is_reload) {
      const iree_vm_isa_ref_stack_load_retain_record_t record = {
          .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_REF_STACK_LOAD_RETAIN,
          .dst_r8 = (uint8_t)register_ordinal,
          .slot_u16 = (uint16_t)slot_ordinal,
      };
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write(writer, &record, sizeof(record)));
    } else {
      // low.spill is a non-consuming read. The local slot takes an independent
      // owner so arbitrary valid Low remains correct, not only allocator-
      // generated traffic whose store is ordinarily the value's final use.
      const iree_vm_isa_ref_stack_store_retain_record_t record = {
          .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_REF_STACK_STORE_RETAIN,
          .src_r8 = (uint8_t)register_ordinal,
          .slot_u16 = (uint16_t)slot_ordinal,
      };
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write(writer, &record, sizeof(record)));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_function_local_function_transfer_encode(
    const loom_vm_function_local_transfer_t* transfer, uint32_t local_base,
    loom_bytecode_page_writer_t* writer) {
  for (uint32_t i = 0; i < transfer->assignment->location_count; ++i) {
    const uint32_t register_ordinal = transfer->assignment->location_base + i;
    const uint32_t local_ordinal = local_base + i;
    IREE_ASSERT_LT(register_ordinal, 256u);
    IREE_ASSERT_LE(local_ordinal, UINT16_MAX);
    if (transfer->is_reload) {
      const iree_vm_isa_func_stack_load_record_t record = {
          .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_FUNC_STACK_LOAD,
          .dst_f8 = (uint8_t)register_ordinal,
          .local_ordinal_u16 = (uint16_t)local_ordinal,
      };
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write(writer, &record, sizeof(record)));
    } else {
      const iree_vm_isa_func_stack_store_record_t record = {
          .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_FUNC_STACK_STORE,
          .src_f8 = (uint8_t)register_ordinal,
          .local_ordinal_u16 = (uint16_t)local_ordinal,
      };
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write(writer, &record, sizeof(record)));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_vm_function_local_transfer_encode(
    const loom_low_emission_frame_t* frame,
    const loom_vm_function_local_layout_t* layout,
    const loom_low_packet_view_t* packet, loom_bytecode_page_writer_t* writer) {
  loom_vm_function_local_transfer_t transfer;
  loom_vm_function_local_transfer_view(frame, packet, &transfer);
  const uint32_t local_base =
      loom_vm_function_local_transfer_base(frame, layout, &transfer);
  switch (transfer.bank) {
    case LOOM_VM_FUNCTION_LOCAL_BANK_VALUE:
      return loom_vm_function_local_value_transfer_encode(&transfer, local_base,
                                                          writer);
    case LOOM_VM_FUNCTION_LOCAL_BANK_REF:
      return loom_vm_function_local_ref_transfer_encode(&transfer, local_base,
                                                        writer);
    case LOOM_VM_FUNCTION_LOCAL_BANK_FUNCTION:
      return loom_vm_function_local_function_transfer_encode(
          &transfer, local_base, writer);
    default:
      IREE_ASSERT_UNREACHABLE("classified VM local transfer bank");
      IREE_BUILTIN_UNREACHABLE();
  }
}
