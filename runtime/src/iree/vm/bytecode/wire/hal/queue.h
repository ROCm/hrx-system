// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for hal.family.queue.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_HAL_QUEUE_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_HAL_QUEUE_H_

#include <stdint.h>

// Page 0xF0, opcode 0x0B: Returns one queue-ordered transient allocation.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // result_nonnull exact hal.buffer ref.
  uint8_t dst_r8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
  // nullable exact hal.pool ref.
  uint8_t pool_r8_nullable;
  // Base ref slot of exact non-null wait semaphores.
  uint16_t wait_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 wait payloads.
  uint16_t wait_payload_base_u16;
  // Shared wait semaphore/payload count.
  uint16_t wait_count_u16;
  // Base ref slot of exact non-null signal semaphores.
  uint16_t signal_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 signal payloads.
  uint16_t signal_payload_base_u16;
  // Shared signal semaphore/payload count.
  uint16_t signal_count_u16;
  // Unsigned HAL buffer-usage flag bits.
  uint8_t usage_v8;
  // Unsigned HAL memory-access flag bits.
  uint8_t access_v8;
  // Unsigned HAL memory-type flag bits.
  uint8_t memory_type_v8;
  // Complete u64 allocation-affinity bitset.
  uint8_t memory_affinity_v8;
  // Minimum device-byte alignment.
  uint8_t min_alignment_v8;
  // Requested device-byte length.
  uint8_t allocation_size_v8;
  // Architectural flag bits constrained by mask 0x00000003.
  uint32_t flags_u32;
} iree_vm_isa_hal_queue_alloca_record_t;

// Page 0xF0, opcode 0x0C: Queues transient-storage reclamation after explicit
// waits.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
  // Base ref slot of exact non-null wait semaphores.
  uint16_t wait_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 wait payloads.
  uint16_t wait_payload_base_u16;
  // Shared wait semaphore/payload count.
  uint16_t wait_count_u16;
  // Base ref slot of exact non-null signal semaphores.
  uint16_t signal_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 signal payloads.
  uint16_t signal_payload_base_u16;
  // Shared signal semaphore/payload count.
  uint16_t signal_count_u16;
  // required exact hal.buffer ref.
  uint8_t buffer_r8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
  // Architectural flag bits constrained by mask 0x00000001.
  uint32_t flags_u32;
} iree_vm_isa_hal_queue_dealloca_record_t;

// Page 0xF0, opcode 0x0D: Queues a repeating one-, two-, or four-byte pattern
// fill.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
  // Base ref slot of exact non-null wait semaphores.
  uint16_t wait_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 wait payloads.
  uint16_t wait_payload_base_u16;
  // Shared wait semaphore/payload count.
  uint16_t wait_count_u16;
  // Base ref slot of exact non-null signal semaphores.
  uint16_t signal_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 signal payloads.
  uint16_t signal_payload_base_u16;
  // Shared signal semaphore/payload count.
  uint16_t signal_count_u16;
  // required exact hal.buffer ref.
  uint8_t target_buffer_r8;
  // Target device-byte offset.
  uint8_t target_offset_v8;
  // Fill length in device bytes.
  uint8_t length_v8;
  // Low pattern bits in little-endian byte order.
  uint8_t pattern_v8;
  // Repeating pattern width of one, two, or four bytes.
  uint8_t pattern_width_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
  // Reserved flags; canonical zero.
  uint32_t flags_u32;
} iree_vm_isa_hal_queue_fill_record_t;

// Page 0xF0, opcode 0x0E: Captures readable host bytes into a direct HAL-buffer
// range.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
  // Base ref slot of exact non-null wait semaphores.
  uint16_t wait_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 wait payloads.
  uint16_t wait_payload_base_u16;
  // Shared wait semaphore/payload count.
  uint16_t wait_count_u16;
  // Base ref slot of exact non-null signal semaphores.
  uint16_t signal_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 signal payloads.
  uint16_t signal_payload_base_u16;
  // Shared signal semaphore/payload count.
  uint16_t signal_count_u16;
  // required exact vm.buffer ref.
  uint8_t source_vm_buffer_r8;
  // Source byte offset.
  uint8_t source_offset_v8;
  // required exact hal.buffer ref.
  uint8_t target_buffer_r8;
  // Target byte offset.
  uint8_t target_offset_v8;
  // Exact transfer byte length.
  uint8_t length_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
  // Reserved flags; canonical zero.
  uint32_t flags_u32;
} iree_vm_isa_hal_queue_update_record_t;

// Page 0xF0, opcode 0x0F: Queues one non-overlapping direct HAL-buffer copy.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
  // Base ref slot of exact non-null wait semaphores.
  uint16_t wait_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 wait payloads.
  uint16_t wait_payload_base_u16;
  // Shared wait semaphore/payload count.
  uint16_t wait_count_u16;
  // Base ref slot of exact non-null signal semaphores.
  uint16_t signal_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 signal payloads.
  uint16_t signal_payload_base_u16;
  // Shared signal semaphore/payload count.
  uint16_t signal_count_u16;
  // required exact hal.buffer ref.
  uint8_t source_buffer_r8;
  // Source byte offset.
  uint8_t source_offset_v8;
  // required exact hal.buffer ref.
  uint8_t target_buffer_r8;
  // Target byte offset.
  uint8_t target_offset_v8;
  // Exact transfer byte length.
  uint8_t length_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
  // Reserved flags; canonical zero.
  uint32_t flags_u32;
} iree_vm_isa_hal_queue_copy_record_t;

// Page 0xF0, opcode 0x10: Queues a readable file range into a direct HAL-buffer
// range.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
  // Base ref slot of exact non-null wait semaphores.
  uint16_t wait_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 wait payloads.
  uint16_t wait_payload_base_u16;
  // Shared wait semaphore/payload count.
  uint16_t wait_count_u16;
  // Base ref slot of exact non-null signal semaphores.
  uint16_t signal_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 signal payloads.
  uint16_t signal_payload_base_u16;
  // Shared signal semaphore/payload count.
  uint16_t signal_count_u16;
  // required exact hal.file ref.
  uint8_t source_file_r8;
  // Source byte offset.
  uint8_t source_offset_v8;
  // required exact hal.buffer ref.
  uint8_t target_buffer_r8;
  // Target byte offset.
  uint8_t target_offset_v8;
  // Exact transfer byte length.
  uint8_t length_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
  // Reserved flags; canonical zero.
  uint32_t flags_u32;
} iree_vm_isa_hal_queue_read_record_t;

// Page 0xF0, opcode 0x11: Queues a direct HAL-buffer range into a writable file
// range.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
  // Base ref slot of exact non-null wait semaphores.
  uint16_t wait_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 wait payloads.
  uint16_t wait_payload_base_u16;
  // Shared wait semaphore/payload count.
  uint16_t wait_count_u16;
  // Base ref slot of exact non-null signal semaphores.
  uint16_t signal_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 signal payloads.
  uint16_t signal_payload_base_u16;
  // Shared signal semaphore/payload count.
  uint16_t signal_count_u16;
  // required exact hal.buffer ref.
  uint8_t source_buffer_r8;
  // Source byte offset.
  uint8_t source_offset_v8;
  // required exact hal.file ref.
  uint8_t target_file_r8;
  // Target byte offset.
  uint8_t target_offset_v8;
  // Exact transfer byte length.
  uint8_t length_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
  // Reserved flags; canonical zero.
  uint32_t flags_u32;
} iree_vm_isa_hal_queue_write_record_t;

// Page 0xF0, opcode 0x12: Queues a static-count dispatch through an executable
// function table.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
  // Base ref slot of exact non-null wait semaphores.
  uint16_t wait_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 wait payloads.
  uint16_t wait_payload_base_u16;
  // Shared wait semaphore/payload count.
  uint16_t wait_count_u16;
  // Base ref slot of exact non-null signal semaphores.
  uint16_t signal_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 signal payloads.
  uint16_t signal_payload_base_u16;
  // Shared signal semaphore/payload count.
  uint16_t signal_count_u16;
  // required exact hal.executable_function_table ref.
  uint8_t function_table_r8;
  // Unsigned selected function ordinal.
  uint8_t function_ordinal_v8;
  // Four-byte-aligned local base of seven u32 launch lanes.
  uint16_t launch_base_u16;
  // Base of a function-local byte range copied as dispatch constants.
  uint16_t constant_base_u16;
  // Exact dispatch-constant byte count.
  uint16_t constant_count_u16;
  // Base ref slot of binding buffers.
  uint16_t binding_buffer_base_u16;
  // Eight-byte-aligned local base of u64 binding offsets.
  uint16_t binding_offset_base_u16;
  // Eight-byte-aligned local base of u64 binding lengths.
  uint16_t binding_length_base_u16;
  // Shared binding-row count.
  uint16_t binding_count_u16;
  // Architectural flag bits constrained by mask 0x00000020.
  uint32_t flags_u32;
} iree_vm_isa_hal_queue_dispatch_record_t;

// Page 0xF0, opcode 0x13: Queues a dispatch whose count is read from a direct
// HAL-buffer range.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
  // Base ref slot of exact non-null wait semaphores.
  uint16_t wait_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 wait payloads.
  uint16_t wait_payload_base_u16;
  // Shared wait semaphore/payload count.
  uint16_t wait_count_u16;
  // Base ref slot of exact non-null signal semaphores.
  uint16_t signal_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 signal payloads.
  uint16_t signal_payload_base_u16;
  // Shared signal semaphore/payload count.
  uint16_t signal_count_u16;
  // required exact hal.executable_function_table ref.
  uint8_t function_table_r8;
  // Unsigned selected function ordinal.
  uint8_t function_ordinal_v8;
  // Four-byte-aligned local base of four u32 launch lanes.
  uint16_t launch_base_u16;
  // required exact hal.buffer ref.
  uint8_t workgroup_count_buffer_r8;
  // Four-byte-aligned device offset of three u32 workgroup counts.
  uint8_t workgroup_count_offset_v8;
  // Base of a function-local byte range copied as dispatch constants.
  uint16_t constant_base_u16;
  // Exact dispatch-constant byte count.
  uint16_t constant_count_u16;
  // Base ref slot of binding buffers.
  uint16_t binding_buffer_base_u16;
  // Eight-byte-aligned local base of u64 binding offsets.
  uint16_t binding_offset_base_u16;
  // Eight-byte-aligned local base of u64 binding lengths.
  uint16_t binding_length_base_u16;
  // Shared binding-row count.
  uint16_t binding_count_u16;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
  // Architectural flag bits constrained by mask 0x00000023.
  uint32_t flags_u32;
} iree_vm_isa_hal_queue_dispatch_indirect_count_record_t;

// Page 0xF0, opcode 0x14: Submits one finalized command buffer with an optional
// binding table.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
  // Base ref slot of exact non-null wait semaphores.
  uint16_t wait_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 wait payloads.
  uint16_t wait_payload_base_u16;
  // Shared wait semaphore/payload count.
  uint16_t wait_count_u16;
  // Base ref slot of exact non-null signal semaphores.
  uint16_t signal_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 signal payloads.
  uint16_t signal_payload_base_u16;
  // Shared signal semaphore/payload count.
  uint16_t signal_count_u16;
  // required exact hal.command_buffer ref.
  uint8_t command_buffer_r8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // Base ref slot of binding buffers.
  uint16_t binding_buffer_base_u16;
  // Eight-byte-aligned local base of u64 binding offsets.
  uint16_t binding_offset_base_u16;
  // Eight-byte-aligned local base of u64 binding lengths.
  uint16_t binding_length_base_u16;
  // Shared binding-row count.
  uint16_t binding_count_u16;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
  // Reserved flags; canonical zero.
  uint32_t flags_u32;
} iree_vm_isa_hal_queue_execute_record_t;

// Page 0xF0, opcode 0x15: Joins explicit waits and publishes explicit signals
// without commands.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
  // Base ref slot of exact non-null wait semaphores.
  uint16_t wait_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 wait payloads.
  uint16_t wait_payload_base_u16;
  // Shared wait semaphore/payload count.
  uint16_t wait_count_u16;
  // Base ref slot of exact non-null signal semaphores.
  uint16_t signal_semaphore_base_u16;
  // Eight-byte-aligned local base of u64 signal payloads.
  uint16_t signal_payload_base_u16;
  // Shared signal semaphore/payload count.
  uint16_t signal_count_u16;
  // Reserved flags; canonical zero.
  uint32_t flags_u32;
} iree_vm_isa_hal_queue_barrier_record_t;

// Page 0xF0, opcode 0x16: Flushes locally pending submissions for one
// device/affinity selection.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
} iree_vm_isa_hal_queue_flush_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_HAL_QUEUE_H_
// clang-format on
