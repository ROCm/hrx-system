// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_FRAME_H_
#define IREE_VM_BYTECODE_INTERPRETER_FRAME_H_

#include <string.h>

#include "iree/vm/bytecode/image.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Caller-owned ref state retired after one suspended child call completes.
typedef struct iree_vm_bytecode_call_cleanup_t {
  // First implementation-private direct argument scratch state.
  iree_vm_ref_t* direct_scratch;
  // Number of direct scratch states to release.
  uint16_t direct_scratch_count;
  // Direct caller register sources to clear after result publication.
  uint16_t direct_source_move_mask;
  // Number of argument states at the start of the local-ref bank to release.
  uint16_t overflow_argument_count;
} iree_vm_bytecode_call_cleanup_t;

// One incoming storage capability. Array contents and objects are borrowed
// from the synchronous caller; no descriptor owns or retires those objects.
typedef struct iree_vm_bytecode_span_region_t {
  // Native storage, never published in guest value registers.
  void* data;
  // Accessible coordinate extent, using the VM stride for reference cells.
  uint64_t byte_length;
  // Exact reference element type, or null for scalar bytes.
  iree_vm_ref_type_t ref_type;
  // Scalar, REF, or PTR element kind; zero marks an unused physical slot.
  uint16_t element_kind;
  // Shallow access admitted by this frame's exact callable signature.
  iree_vm_span_access_t access;
} iree_vm_bytecode_span_region_t;

// Coordinates combine a frame-local region ordinal with its byte offset.
// Region zero selects the ordinary local byte or reference bank. Incoming
// regions use their aligned physical argument pair index plus one, so the
// frame never needs a signature search to resolve an access.
#define IREE_VM_BYTECODE_SPAN_REGION_SHIFT 48u
#define IREE_VM_BYTECODE_SPAN_OFFSET_MASK UINT64_C(0x0000FFFFFFFFFFFF)
// The upper half of region ordinals names buffers pinned for the next call.
#define IREE_VM_BYTECODE_SPAN_BUFFER_REGION_BASE 0x8000u

// Bytecode execution state followed by exact function-owned banks when stored
// in a durable generic frame. The packet is copied by value so the frame has no
// self-relative pointer and all bank bases remain stable through suspension.
typedef struct iree_vm_bytecode_execution_state_t {
  // Physical call packet owned by this execution state.
  iree_vm_call_packet_t call;
  // Next verified instruction record to execute.
  const uint8_t* program_counter;
  // Owning durable frame, or null during synchronous transient execution.
  iree_vm_frame_t* frame;
  // Function value-register bank.
  uint64_t* values;
  // Function ref-register bank followed by local ref slots.
  iree_vm_ref_t* refs;
  // First function-local ref slot.
  iree_vm_ref_t* local_refs;
  // Implementation-private direct argument scratch for mixed call ownership.
  iree_vm_ref_t* call_ref_scratch;
  // Function-register bank followed by local function slots.
  iree_vm_function_ref_t* functions;
  // First function-local function slot.
  iree_vm_function_ref_t* local_functions;
  // Function-local byte storage.
  uint8_t* local_bytes;
  // Private incoming region slots, indexed by physical argument pair.
  iree_vm_bytecode_span_region_t* span_regions;
  // Number of addressable incoming slots; unused slots have a zero kind.
  uint16_t span_region_capacity;
  // Private overflow arguments with native addresses replaced by coordinates.
  uint64_t* span_argument_values;
  // Managed VM buffers pinned until the next child completes or this frame
  // exits.
  iree_vm_ref_t* span_buffers;
  // Initialized prefix of span_buffers; no entry survives a completed child.
  uint16_t span_buffer_count;
  // Maximum pending buffers, bounded by the outgoing span-pair scratch count.
  uint16_t span_buffer_capacity;
  // Number of ref states released by frame cleanup.
  uint32_t ref_count;
  // Ref packet ownership pending one suspended child completion.
  iree_vm_bytecode_call_cleanup_t pending_call_cleanup;
} iree_vm_bytecode_execution_state_t;

// Exact durable payload layout for one verified bytecode function.
typedef struct iree_vm_bytecode_frame_layout_t {
  // Generic frame payload request.
  iree_vm_frame_layout_t frame;
  // Byte offset of the value-register bank.
  iree_host_size_t values_offset;
  // Byte offset of the ref-register bank.
  iree_host_size_t refs_offset;
  // Byte offset of the local ref slots.
  iree_host_size_t local_refs_offset;
  // Byte offset of implementation-private direct call ref scratch.
  iree_host_size_t call_ref_scratch_offset;
  // Byte offset of the function-register bank.
  iree_host_size_t functions_offset;
  // Byte offset of the local function slots.
  iree_host_size_t local_functions_offset;
  // Byte offset of the local byte storage.
  iree_host_size_t local_bytes_offset;
  // Byte offset of private incoming region descriptors.
  iree_host_size_t span_regions_offset;
  // Number of incoming physical-pair slots, or zero for ordinary functions.
  uint16_t span_region_capacity;
  // Byte offset of sanitized incoming overflow values.
  iree_host_size_t span_argument_values_offset;
  // Byte offset of private pending buffer owners within the contiguous ref
  // bank.
  iree_host_size_t span_buffers_offset;
  // Number of pending buffer slots reserved by outgoing call value pairs.
  uint16_t span_buffer_capacity;
} iree_vm_bytecode_frame_layout_t;

// Copies one direct physical value bank in bounded native-sized chunks.
static inline void iree_vm_bytecode_frame_copy_direct_values(
    uint64_t* target, const uint64_t* source, uint16_t count) {
  while (count >= 4) {
    memcpy(target, source, 4 * sizeof(uint64_t));
    target += 4;
    source += 4;
    count -= 4;
  }
  if (count >= 2) {
    memcpy(target, source, 2 * sizeof(uint64_t));
    target += 2;
    source += 2;
    count -= 2;
  }
  if (count != 0) {
    *target = *source;
  }
}

// Initializes the exact register, local, and call-scratch banks for entry.
void iree_vm_bytecode_frame_initialize(
    const iree_vm_bytecode_image_t* image,
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_bytecode_v0_signature_row_t* signature,
    iree_vm_bytecode_execution_state_t* state);

// Calculates the infallible exact durable payload layout for one verified
// function row. Module verification bounds every extent before this call.
iree_vm_bytecode_frame_layout_t iree_vm_bytecode_calculate_frame_layout(
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_bytecode_v0_signature_row_t* signature);

// Releases every ref state owned by a durable bytecode frame.
void iree_vm_bytecode_frame_cleanup(iree_vm_frame_t* frame);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_INTERPRETER_FRAME_H_
