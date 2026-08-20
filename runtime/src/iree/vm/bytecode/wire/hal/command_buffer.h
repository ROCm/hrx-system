// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for hal.family.command_buffer.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_HAL_COMMAND_BUFFER_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_HAL_COMMAND_BUFFER_H_

#include <stdint.h>

// Selects whether command dispatch records a fixed phase barrier first.
typedef uint8_t iree_vm_isa_cmd_dispatch_barrier_before_t;
enum {
  // Records only the dispatch.
  IREE_VM_ISA_CMD_DISPATCH_BARRIER_BEFORE_NONE = 0x00,
  // Records a full non-host device execution/memory barrier covering RAW, WAR,
  // and WAW reuse, then dispatches.
  IREE_VM_ISA_CMD_DISPATCH_BARRIER_BEFORE_ALL = 0x01,
};

enum {
  IREE_VM_ISA_CMD_DISPATCH_BARRIER_BEFORE_NONE_SINCE_MINOR = 0,
  IREE_VM_ISA_CMD_DISPATCH_BARRIER_BEFORE_ALL_SINCE_MINOR = 0,
};

// Selects channel communication shape and therefore the contextual send/receive
// packets and element counts required on the current rank.
typedef uint8_t iree_vm_isa_collective_kind_t;
enum {
  // Every rank sends element_count elements and receives
  // channel_count*element_count elements.
  IREE_VM_ISA_COLLECTIVE_KIND_ALL_GATHER = 0x00,
  // Every rank sends and receives element_count elements using the selected
  // reduction.
  IREE_VM_ISA_COLLECTIVE_KIND_ALL_REDUCE = 0x01,
  // Every rank sends and receives element_count elements, which must be
  // divisible by channel_count.
  IREE_VM_ISA_COLLECTIVE_KIND_ALL_TO_ALL = 0x02,
  // Root rank param sends element_count elements; every other rank receives
  // them, and param must be in range.
  IREE_VM_ISA_COLLECTIVE_KIND_BROADCAST = 0x03,
  // Every rank sends element_count elements and root rank param alone receives
  // the selected reduction.
  IREE_VM_ISA_COLLECTIVE_KIND_REDUCE = 0x04,
  // Every rank sends channel_count*element_count elements and receives
  // element_count reduced elements.
  IREE_VM_ISA_COLLECTIVE_KIND_REDUCE_SCATTER = 0x05,
  // Sends element_count elements to peer rank param and uses no receive side.
  IREE_VM_ISA_COLLECTIVE_KIND_SEND = 0x06,
  // Receives element_count elements from peer rank param and uses no send side.
  IREE_VM_ISA_COLLECTIVE_KIND_RECV = 0x07,
  // Uses signed low/high 16-bit target/source ranks; target -1 skips send and
  // source -1 fills receive with zeros.
  IREE_VM_ISA_COLLECTIVE_KIND_SEND_RECV = 0x08,
};

enum {
  IREE_VM_ISA_COLLECTIVE_KIND_ALL_GATHER_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_KIND_ALL_REDUCE_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_KIND_ALL_TO_ALL_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_KIND_BROADCAST_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_KIND_REDUCE_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_KIND_REDUCE_SCATTER_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_KIND_SEND_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_KIND_RECV_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_KIND_SEND_RECV_SINCE_MINOR = 0,
};

// Selects the HAL collective reduction. Reducing kinds require a non-none
// value; nonreducing kinds require none.
typedef uint8_t iree_vm_isa_collective_reduction_t;
enum {
  // Performs no element reduction.
  IREE_VM_ISA_COLLECTIVE_REDUCTION_NONE = 0x00,
  // Reduces corresponding selected-type elements by sum.
  IREE_VM_ISA_COLLECTIVE_REDUCTION_SUM = 0x01,
  // Reduces corresponding selected-type elements by product.
  IREE_VM_ISA_COLLECTIVE_REDUCTION_PRODUCT = 0x02,
  // Reduces corresponding selected-type elements by minimum.
  IREE_VM_ISA_COLLECTIVE_REDUCTION_MINIMUM = 0x03,
  // Reduces corresponding selected-type elements by maximum.
  IREE_VM_ISA_COLLECTIVE_REDUCTION_MAXIMUM = 0x04,
  // Reduces corresponding selected-type elements by average.
  IREE_VM_ISA_COLLECTIVE_REDUCTION_AVERAGE = 0x05,
};

enum {
  IREE_VM_ISA_COLLECTIVE_REDUCTION_NONE_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_REDUCTION_SUM_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_REDUCTION_PRODUCT_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_REDUCTION_MINIMUM_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_REDUCTION_MAXIMUM_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_REDUCTION_AVERAGE_SINCE_MINOR = 0,
};

// Selects the element interpretation and byte width used for checked collective
// range sizing and provider reduction.
typedef uint8_t iree_vm_isa_collective_element_t;
enum {
  // Uses signed 8-bit integer elements of width one byte.
  IREE_VM_ISA_COLLECTIVE_ELEMENT_SI8 = 0x00,
  // Uses unsigned 8-bit integer elements of width one byte.
  IREE_VM_ISA_COLLECTIVE_ELEMENT_UI8 = 0x01,
  // Uses signed 16-bit integer elements of width two bytes.
  IREE_VM_ISA_COLLECTIVE_ELEMENT_SI16 = 0x02,
  // Uses unsigned 16-bit integer elements of width two bytes.
  IREE_VM_ISA_COLLECTIVE_ELEMENT_UI16 = 0x03,
  // Uses signed 32-bit integer elements of width four bytes.
  IREE_VM_ISA_COLLECTIVE_ELEMENT_SI32 = 0x04,
  // Uses unsigned 32-bit integer elements of width four bytes.
  IREE_VM_ISA_COLLECTIVE_ELEMENT_UI32 = 0x05,
  // Uses signed 64-bit integer elements of width eight bytes.
  IREE_VM_ISA_COLLECTIVE_ELEMENT_SI64 = 0x06,
  // Uses unsigned 64-bit integer elements of width eight bytes.
  IREE_VM_ISA_COLLECTIVE_ELEMENT_UI64 = 0x07,
  // Uses IEEE binary16 elements of width two bytes.
  IREE_VM_ISA_COLLECTIVE_ELEMENT_F16 = 0x08,
  // Uses IEEE binary32 elements of width four bytes.
  IREE_VM_ISA_COLLECTIVE_ELEMENT_F32 = 0x09,
  // Uses IEEE binary64 elements of width eight bytes.
  IREE_VM_ISA_COLLECTIVE_ELEMENT_F64 = 0x0A,
  // Uses bfloat16 elements of width two bytes.
  IREE_VM_ISA_COLLECTIVE_ELEMENT_BF16 = 0x0B,
};

enum {
  IREE_VM_ISA_COLLECTIVE_ELEMENT_SI8_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_ELEMENT_UI8_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_ELEMENT_SI16_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_ELEMENT_UI16_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_ELEMENT_SI32_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_ELEMENT_UI32_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_ELEMENT_SI64_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_ELEMENT_UI64_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_ELEMENT_F16_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_ELEMENT_F32_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_ELEMENT_F64_SINCE_MINOR = 0,
  IREE_VM_ISA_COLLECTIVE_ELEMENT_BF16_SINCE_MINOR = 0,
};

// Page 0xF0, opcode 0x17: Creates one command buffer and enters recording
// exactly once.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // result_nonnull exact hal.command_buffer ref.
  uint8_t dst_r8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Architectural flag bits constrained by mask 0x00000191.
  uint32_t mode_u32;
  // Architectural flag bits constrained by mask 0x00000003.
  uint32_t categories_u32;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
  // Number of indirect binding slots addressable at submission.
  uint8_t binding_capacity_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[2];
} iree_vm_isa_hal_cmd_create_record_t;

// Page 0xF0, opcode 0x18: Permanently seals one recording command buffer
// without consuming it.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.command_buffer ref.
  uint8_t command_buffer_r8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_hal_cmd_finalize_record_t;

// Page 0xF0, opcode 0x19: Records independent global-memory and buffer-specific
// dependencies.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.command_buffer ref.
  uint8_t command_buffer_r8;
  // Unsigned source-stage mask.
  uint8_t source_stage_v8;
  // Unsigned target-stage mask.
  uint8_t target_stage_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
  // Canonical zero reserved flags.
  uint32_t flags_u32;
  // Four-byte-aligned local base of u32 memory source scopes.
  uint16_t memory_source_scope_base_u16;
  // Four-byte-aligned local base of u32 memory target scopes.
  uint16_t memory_target_scope_base_u16;
  // Shared global-memory barrier row count.
  uint16_t memory_count_u16;
  // Four-byte-aligned local base of u32 buffer source scopes.
  uint16_t buffer_source_scope_base_u16;
  // Four-byte-aligned local base of u32 buffer target scopes.
  uint16_t buffer_target_scope_base_u16;
  // Base ref slot of nullable direct buffers.
  uint16_t buffer_ref_base_u16;
  // Four-byte-aligned local base of u32 indirect slots.
  uint16_t buffer_slot_base_u16;
  // Eight-byte-aligned local base of u64 buffer offsets.
  uint16_t buffer_offset_base_u16;
  // Eight-byte-aligned local base of u64 buffer lengths.
  uint16_t buffer_length_base_u16;
  // Shared buffer-barrier row count.
  uint16_t buffer_count_u16;
} iree_vm_isa_hal_cmd_execution_barrier_record_t;

// Page 0xF0, opcode 0x1A: Records an implementation-defined hint over one
// complete buffer ref.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.command_buffer ref.
  uint8_t command_buffer_r8;
  // Direct exact hal.buffer or indirect binding slot.
  uint8_t buffer_r8_nullable;
  // Unsigned direct-or-slot binding index.
  uint8_t buffer_slot_v8;
  // Unsigned buffer byte offset.
  uint8_t buffer_offset_v8;
  // Unsigned buffer byte length.
  uint8_t buffer_length_v8;
  // Canonical zero padding.
  uint8_t zero_padding0_u8;
  // Canonical zero reserved flags.
  uint32_t flags_u32;
  // Reserved zero dynamic advice argument zero.
  uint8_t arg0_v8;
  // Reserved zero dynamic advice argument one.
  uint8_t arg1_v8;
  // Canonical zero padding.
  uint8_t zero_padding1_u8[2];
} iree_vm_isa_hal_cmd_advise_buffer_record_t;

// Page 0xF0, opcode 0x1B: Records a repeating little-endian fill over one
// complete buffer ref.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.command_buffer ref.
  uint8_t command_buffer_r8;
  // Direct exact hal.buffer or indirect binding slot.
  uint8_t target_buffer_r8_nullable;
  // Unsigned target binding slot.
  uint8_t target_slot_v8;
  // Unsigned target byte offset.
  uint8_t target_offset_v8;
  // Unsigned target byte length.
  uint8_t target_length_v8;
  // Low pattern bits in little-endian byte order.
  uint8_t pattern_v8;
  // Repeating pattern width of one, two, or four bytes.
  uint8_t pattern_width_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
  // Canonical zero reserved flags.
  uint32_t flags_u32;
} iree_vm_isa_hal_cmd_fill_buffer_record_t;

// Page 0xF0, opcode 0x1C: Copies readable host bytes immediately into recorded
// command state.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.command_buffer ref.
  uint8_t command_buffer_r8;
  // required exact vm.buffer ref.
  uint8_t source_vm_buffer_r8;
  // Unsigned source host-byte offset.
  uint8_t source_offset_v8;
  // Direct exact hal.buffer or indirect binding slot.
  uint8_t target_buffer_r8_nullable;
  // Unsigned target binding slot.
  uint8_t target_slot_v8;
  // Unsigned target device-byte offset.
  uint8_t target_offset_v8;
  // Exact source/target byte length.
  uint8_t target_length_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
  // Canonical zero reserved flags.
  uint32_t flags_u32;
} iree_vm_isa_hal_cmd_update_buffer_record_t;

// Page 0xF0, opcode 0x1D: Records one byte-length copy between complete buffer
// refs.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.command_buffer ref.
  uint8_t command_buffer_r8;
  // Direct exact hal.buffer or indirect binding slot.
  uint8_t source_buffer_r8_nullable;
  // Unsigned source binding slot.
  uint8_t source_slot_v8;
  // Unsigned source device-byte offset.
  uint8_t source_offset_v8;
  // Direct exact hal.buffer or indirect binding slot.
  uint8_t target_buffer_r8_nullable;
  // Unsigned target binding slot.
  uint8_t target_slot_v8;
  // Unsigned target device-byte offset.
  uint8_t target_offset_v8;
  // Exact copy byte length.
  uint8_t length_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[2];
  // Canonical zero reserved flags.
  uint32_t flags_u32;
} iree_vm_isa_hal_cmd_copy_buffer_record_t;

// Page 0xF0, opcode 0x1E: Records one context-sensitive channel collective.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.command_buffer ref.
  uint8_t command_buffer_r8;
  // required exact hal.channel ref.
  uint8_t channel_r8;
  // Packed collective kind, reduction, and element selectors.
  uint32_t op_u32;
  // Collective root/peer packed u32 parameter.
  uint8_t param_v8;
  // Direct exact hal.buffer or indirect binding slot.
  uint8_t send_buffer_r8_nullable;
  // Unsigned send binding slot.
  uint8_t send_slot_v8;
  // Unsigned send device-byte offset.
  uint8_t send_offset_v8;
  // Available send byte length.
  uint8_t send_length_v8;
  // Direct exact hal.buffer or indirect binding slot.
  uint8_t recv_buffer_r8_nullable;
  // Unsigned receive binding slot.
  uint8_t recv_slot_v8;
  // Unsigned receive device-byte offset.
  uint8_t recv_offset_v8;
  // Available receive byte length.
  uint8_t recv_length_v8;
  // Logical collective element count.
  uint8_t element_count_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[2];
} iree_vm_isa_hal_cmd_collective_record_t;

// Page 0xF0, opcode 0x1F: Records a static-count table dispatch with an
// optional fused barrier.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.command_buffer ref.
  uint8_t command_buffer_r8;
  // required exact hal.executable_function_table ref.
  uint8_t function_table_r8;
  // Unsigned selected function ordinal.
  uint8_t function_ordinal_v8;
  // NONE=0 or full device phase barrier ALL=1.
  uint8_t barrier_before_u8;
  // Four-byte-aligned local base of seven u32 launch lanes.
  uint16_t launch_base_u16;
  // Base of function-local dispatch-constant bytes.
  uint16_t constant_base_u16;
  // Exact dispatch-constant byte count.
  uint16_t constant_count_u16;
  // Base ref slot of nullable direct binding buffers.
  uint16_t binding_buffer_base_u16;
  // Four-byte-aligned local base of u32 binding slots.
  uint16_t binding_slot_base_u16;
  // Eight-byte-aligned local base of u64 binding offsets.
  uint16_t binding_offset_base_u16;
  // Eight-byte-aligned local base of u64 binding lengths.
  uint16_t binding_length_base_u16;
  // Shared direct-or-slot binding row count.
  uint16_t binding_count_u16;
  // Canonical zero padding.
  uint8_t zero_padding_u8[2];
  // Architectural flag bits constrained by mask 0x00000020.
  uint32_t flags_u32;
} iree_vm_isa_hal_cmd_dispatch_record_t;

// Page 0xF0, opcode 0x20: Records an indirect-count table dispatch with an
// optional fused barrier.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.command_buffer ref.
  uint8_t command_buffer_r8;
  // required exact hal.executable_function_table ref.
  uint8_t function_table_r8;
  // Unsigned selected function ordinal.
  uint8_t function_ordinal_v8;
  // NONE=0 or full device phase barrier ALL=1.
  uint8_t barrier_before_u8;
  // Four-byte-aligned local base of four u32 launch lanes.
  uint16_t launch_base_u16;
  // Direct exact hal.buffer or indirect binding slot.
  uint8_t workgroup_count_buffer_r8_nullable;
  // Unsigned indirect count binding slot.
  uint8_t workgroup_count_slot_v8;
  // Four-byte-aligned offset of exactly three u32 counts.
  uint8_t workgroup_count_offset_v8;
  // Canonical zero padding.
  uint8_t zero_padding0_u8;
  // Base of function-local dispatch-constant bytes.
  uint16_t constant_base_u16;
  // Exact dispatch-constant byte count.
  uint16_t constant_count_u16;
  // Base ref slot of nullable direct binding buffers.
  uint16_t binding_buffer_base_u16;
  // Four-byte-aligned local base of u32 binding slots.
  uint16_t binding_slot_base_u16;
  // Eight-byte-aligned local base of u64 binding offsets.
  uint16_t binding_offset_base_u16;
  // Eight-byte-aligned local base of u64 binding lengths.
  uint16_t binding_length_base_u16;
  // Shared direct-or-slot binding row count.
  uint16_t binding_count_u16;
  // Canonical zero padding.
  uint8_t zero_padding1_u8[2];
  // Architectural flag bits constrained by mask 0x00000023.
  uint32_t flags_u32;
} iree_vm_isa_hal_cmd_dispatch_indirect_count_record_t;

// Page 0xF0, opcode 0x21: Begins one provider-managed diagnostic label group.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.command_buffer ref.
  uint8_t command_buffer_r8;
  // Canonical zero padding.
  uint8_t zero_padding0_u8;
  // Module string ordinal; empty labels are valid.
  uint16_t label_string_u16;
  // Canonical zero padding.
  uint8_t zero_padding1_u8[2];
  // Little-endian RGBA8 channels; zero means unspecified.
  uint32_t color_u32;
} iree_vm_isa_hal_cmd_debug_group_begin_record_t;

// Page 0xF0, opcode 0x22: Ends the innermost provider-managed diagnostic group.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.command_buffer ref.
  uint8_t command_buffer_r8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_hal_cmd_debug_group_end_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_HAL_COMMAND_BUFFER_H_
// clang-format on
