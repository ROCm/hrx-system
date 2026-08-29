// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/verification_isa_buffer.h"

#include "iree/vm/bytecode/interpreter_atomic.h"
#include "iree/vm/bytecode/verification_isa_registers.h"
#include "iree/vm/bytecode/wire/core/buffer.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/core/selectors.h"

static iree_status_t iree_vm_bytecode_verify_atomic_carrier(uint8_t carrier) {
  if (!iree_vm_bytecode_atomic_carrier_is_supported(carrier)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "atomic carrier is unsupported on this target");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_atomic_apply_selectors(
    uint8_t selector0, uint8_t selector1, uint8_t minimum_kind) {
  const uint8_t kind = selector0 & 0x0Fu;
  const uint8_t ordering = selector1 & 0x07u;
  const uint8_t scope = (selector1 >> 3) & 0x07u;
  if ((selector0 & 0x70u) != 0 || (selector1 & 0xC0u) != 0 ||
      kind < minimum_kind ||
      ordering > IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_SEQ_CST ||
      scope > IREE_VM_ISA_BUFFER_ATOMIC_SCOPE_SYSTEM) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer atomic selectors are invalid");
  }
  return iree_ok_status();
}

static bool iree_vm_bytecode_atomic_ordering_pair_is_valid(
    uint8_t success_ordering, uint8_t failure_ordering) {
  switch (success_ordering) {
    case IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_RELAXED:
    case IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_RELEASE:
      return failure_ordering == IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_RELAXED;
    case IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_ACQUIRE:
    case IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_ACQ_REL:
      return failure_ordering == IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_RELAXED ||
             failure_ordering == IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_ACQUIRE;
    case IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_SEQ_CST:
      return failure_ordering == IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_RELAXED ||
             failure_ordering == IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_ACQUIRE ||
             failure_ordering == IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_SEQ_CST;
    default:
      return false;
  }
}

static iree_status_t iree_vm_bytecode_verify_atomic_cmpxchg_selectors(
    uint8_t selector0, uint8_t selector1) {
  const uint8_t success_ordering = selector0 & 0x07u;
  const uint8_t failure_ordering = (selector0 >> 3) & 0x07u;
  const uint8_t scope = selector1 & 0x07u;
  if ((selector0 & 0x40u) != 0 || (selector1 & 0xF8u) != 0 ||
      !iree_vm_bytecode_atomic_ordering_pair_is_valid(success_ordering,
                                                      failure_ordering) ||
      scope > IREE_VM_ISA_BUFFER_ATOMIC_SCOPE_SYSTEM) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer atomic cmpxchg selectors are invalid");
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_verify_buffer_record(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_function_row_t* function,
    iree_const_byte_span_t record_data) {
  switch (record_data.data[0]) {
    case IREE_VM_ISA_CORE_OPCODE_BUFFER_ALLOCATE: {
      const iree_vm_isa_buffer_allocate_record_t* record =
          (const iree_vm_isa_buffer_allocate_record_t*)record_data.data;
      if (record->minimum_alignment_log2_u8 > 63) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "buffer.allocate alignment exponent exceeds 63");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->dst_r8, function->ref_register_count_u16));
      return iree_vm_bytecode_verify_value_register(
          record->length_v8, function->value_register_count_u16);
    }
    case IREE_VM_ISA_CORE_OPCODE_BUFFER_LENGTH: {
      const iree_vm_isa_buffer_length_record_t* record =
          (const iree_vm_isa_buffer_length_record_t*)record_data.data;
      if (record->zero_padding_u8 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "buffer.length padding is nonzero");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->dst_v8, function->value_register_count_u16));
      return iree_vm_bytecode_verify_ref_register(
          record->buffer_r8, function->ref_register_count_u16);
    }
    case IREE_VM_ISA_CORE_OPCODE_BUFFER_SUBSPAN: {
      const iree_vm_isa_buffer_subspan_record_t* record =
          (const iree_vm_isa_buffer_subspan_record_t*)record_data.data;
      if (record->zero_padding_u8[0] != 0 || record->zero_padding_u8[1] != 0 ||
          record->zero_padding_u8[2] != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "buffer.subspan padding is nonzero");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->dst_r8, function->ref_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->buffer_r8, function->ref_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->offset_v8, function->value_register_count_u16));
      return iree_vm_bytecode_verify_value_register(
          record->length_v8, function->value_register_count_u16);
    }
    case IREE_VM_ISA_CORE_OPCODE_BUFFER_LOAD: {
      const iree_vm_isa_buffer_load_record_t* record =
          (const iree_vm_isa_buffer_load_record_t*)record_data.data;
      if (record->zero_padding_u8 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "buffer.load padding is nonzero");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->buffer_r8, function->ref_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->base_v8, function->value_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->index_v8, function->value_register_count_u16));
      return iree_vm_bytecode_verify_lane_register_range(
          record->dst_v8, record->format_u8, function);
    }
    case IREE_VM_ISA_CORE_OPCODE_BUFFER_STORE: {
      const iree_vm_isa_buffer_store_record_t* record =
          (const iree_vm_isa_buffer_store_record_t*)record_data.data;
      if (record->zero_padding_u8 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "buffer.store padding is nonzero");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->buffer_r8, function->ref_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->base_v8, function->value_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->index_v8, function->value_register_count_u16));
      return iree_vm_bytecode_verify_lane_register_range(
          record->src_v8, record->format_u8, function);
    }
    case IREE_VM_ISA_CORE_OPCODE_BUFFER_ATOMIC_REDUCE: {
      const iree_vm_isa_buffer_atomic_reduce_record_t* record =
          (const iree_vm_isa_buffer_atomic_reduce_record_t*)record_data.data;
      if (record->zero_padding_u16 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "buffer.atomic.reduce padding is nonzero");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_atomic_apply_selectors(
          record->selector0_u8, record->selector1_u8,
          IREE_VM_ISA_BUFFER_ATOMIC_KIND_ADD_INTEGER));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->buffer_r8, function->ref_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->offset_v8, function->value_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->operand_v8, function->value_register_count_u16));
      return iree_vm_bytecode_verify_atomic_carrier(record->selector0_u8 >> 7);
    }
    case IREE_VM_ISA_CORE_OPCODE_BUFFER_ATOMIC_RMW: {
      const iree_vm_isa_buffer_atomic_rmw_record_t* record =
          (const iree_vm_isa_buffer_atomic_rmw_record_t*)record_data.data;
      if (record->zero_padding_u8 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "buffer.atomic.rmw padding is nonzero");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_atomic_apply_selectors(
          record->selector0_u8, record->selector1_u8,
          IREE_VM_ISA_BUFFER_ATOMIC_KIND_EXCHANGE_INTEGER));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->old_v8, function->value_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->buffer_r8, function->ref_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->offset_v8, function->value_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->operand_v8, function->value_register_count_u16));
      return iree_vm_bytecode_verify_atomic_carrier(record->selector0_u8 >> 7);
    }
    case IREE_VM_ISA_CORE_OPCODE_BUFFER_ATOMIC_CMPXCHG: {
      const iree_vm_isa_buffer_atomic_cmpxchg_record_t* record =
          (const iree_vm_isa_buffer_atomic_cmpxchg_record_t*)record_data.data;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_atomic_cmpxchg_selectors(
          record->selector0_u8, record->selector1_u8));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->old_v8, function->value_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->buffer_r8, function->ref_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->offset_v8, function->value_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->expected_v8, function->value_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->replacement_v8, function->value_register_count_u16));
      return iree_vm_bytecode_verify_atomic_carrier(record->selector0_u8 >> 7);
    }
    case IREE_VM_ISA_CORE_OPCODE_BUFFER_FILL: {
      const iree_vm_isa_buffer_fill_record_t* record =
          (const iree_vm_isa_buffer_fill_record_t*)record_data.data;
      if (record->zero_padding_u16 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "buffer.fill padding is nonzero");
      }
      if (record->pattern_width_u8 != 1 && record->pattern_width_u8 != 2 &&
          record->pattern_width_u8 != 4 && record->pattern_width_u8 != 8) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "buffer.fill pattern width is invalid");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->buffer_r8, function->ref_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->offset_v8, function->value_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->length_v8, function->value_register_count_u16));
      return iree_vm_bytecode_verify_value_register(
          record->pattern_v8, function->value_register_count_u16);
    }
    case IREE_VM_ISA_CORE_OPCODE_BUFFER_COPY: {
      const iree_vm_isa_buffer_copy_record_t* record =
          (const iree_vm_isa_buffer_copy_record_t*)record_data.data;
      if (record->zero_padding_u16 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "buffer.copy padding is nonzero");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->target_r8, function->ref_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->target_offset_v8, function->value_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->source_r8, function->ref_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->source_offset_v8, function->value_register_count_u16));
      return iree_vm_bytecode_verify_value_register(
          record->length_v8, function->value_register_count_u16);
    }
    case IREE_VM_ISA_CORE_OPCODE_BUFFER_COMPARE: {
      const iree_vm_isa_buffer_compare_record_t* record =
          (const iree_vm_isa_buffer_compare_record_t*)record_data.data;
      if (record->zero_padding_u8 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "buffer.compare padding is nonzero");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->dst_v8, function->value_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->lhs_r8, function->ref_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->lhs_offset_v8, function->value_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->rhs_r8, function->ref_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->rhs_offset_v8, function->value_register_count_u16));
      return iree_vm_bytecode_verify_value_register(
          record->length_v8, function->value_register_count_u16);
    }
    case IREE_VM_ISA_CORE_OPCODE_BUFFER_COPY_RODATA: {
      const iree_vm_isa_buffer_copy_rodata_record_t* record =
          (const iree_vm_isa_buffer_copy_rodata_record_t*)record_data.data;
      if (record->zero_padding0_u8 != 0 || record->zero_padding1_u8 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "buffer.copy.rodata padding is nonzero");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->target_r8, function->ref_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->target_offset_v8, function->value_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->length_v8, function->value_register_count_u16));
      if (record->rodata_u16 >= layout->rodata.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "buffer.copy.rodata ordinal is out of range");
      }
      if (record->source_offset_u32 >
          layout->rodata.descriptors[record->rodata_u16].byte_length_u64) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "buffer.copy.rodata source offset is out of range");
      }
      return iree_ok_status();
    }
    case IREE_VM_ISA_CORE_OPCODE_BUFFER_RODATA_LOAD: {
      const iree_vm_isa_buffer_rodata_load_record_t* record =
          (const iree_vm_isa_buffer_rodata_load_record_t*)record_data.data;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->dst_r8, function->ref_register_count_u16));
      if (record->rodata_u16 >= layout->rodata.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "rodata ordinal is out of range");
      }
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "generated buffer verification form is invalid");
  }
}
