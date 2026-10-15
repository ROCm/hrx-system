// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_KERNARG_H_
#define IREE_HAL_DRIVERS_AMD_GPU_KERNARG_H_

#include "iree/hal/drivers/amd/gpu/image/metadata.h"
#include "iree/hal/executable.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Native services named by hidden arguments. A layout records requirements;
// admission to an executing queue additionally establishes each service's
// backing, initialization, protocol, and lifetime.
enum iree_hal_amd_gpu_kernarg_requirement_bits_e {
  IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_PRINTF = 1u << 0,
  IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_HOSTCALL = 1u << 1,
  IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_DEVICE_ENQUEUE = 1u << 2,
  IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_COMPLETION_ACTION = 1u << 3,
  IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_GRID_SYNC = 1u << 4,
  IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_HEAP = 1u << 5,
  IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_PRIVATE_APERTURE = 1u << 6,
  IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_SHARED_APERTURE = 1u << 7,
  IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_QUEUE_POINTER = 1u << 8,
};
typedef uint32_t iree_hal_amd_gpu_kernarg_requirements_t;

// Runtime values used by the admitted hidden arguments. This is a host input
// record, not a device wire format. Only fields named by the layout are read.
// Values and native services have been admitted before emplacement. A partial
// dispatch counts complete groups separately from its final remainder.
typedef struct iree_hal_amd_gpu_kernarg_inputs_t {
  // Zero-origin HAL dispatches supply zeros for these work-item offsets.
  uint64_t global_offset[3];
  // Complete workgroups in each dimension, excluding any partial group.
  uint32_t block_count[3];
  // Work-items in each full workgroup dimension.
  uint16_t group_size[3];
  // Work-items in each partial group dimension, or zero when divisible.
  uint16_t remainder[3];
  // Dimensionality of the grid, between one and three.
  uint16_t grid_dimensions;
  // Dynamically allocated LDS byte count, excluding the static group segment.
  uint32_t dynamic_lds_size;
  // Native addresses of initialized runtime service objects.
  struct {
    // Buffered printf destination in GPU-visible memory.
    uint64_t printf_buffer;
    // Hostcall service buffer, including its separately admitted protocol.
    uint64_t hostcall_buffer;
    // Default device-enqueue queue object.
    uint64_t default_queue;
    // Device-enqueue ancestor completion state.
    uint64_t completion_action;
    // Independent cooperative grid synchronization state for this invocation.
    uint64_t grid_sync;
    // Device malloc/free V1 heap object.
    uint64_t heap;
    // Native AMD queue ABI object for the older trap-handler ABI.
    uint64_t queue_pointer;
  } services;
  // High address words used by the GFX8 flat-address conversion ABI.
  struct {
    // High 32 bits of the private aperture base.
    uint32_t private_base;
    // High 32 bits of the shared aperture base.
    uint32_t shared_base;
  } apertures;
} iree_hal_amd_gpu_kernarg_inputs_t;

// One immutable kernel argument layout. Its allocation contains a 48-byte
// header on 64-bit hosts, then binding offsets, constant copies, runtime
// copies, and cold HAL parameter reflection. No packet protocol or native
// allocation is owned here. Strings in reflection borrow from the enclosing
// image.
typedef struct iree_hal_amd_gpu_kernarg_layout_t {
  // Complete native kernarg extent, including hidden arguments and padding.
  uint32_t byte_length;
  // Required power-of-two native storage alignment.
  uint32_t alignment;
  // Bytes consumed from the dense HAL dispatch constant stream.
  uint32_t constant_byte_length;
  // Extent through the last explicit argument, including native ABI holes.
  uint32_t native_byte_length;
  // Runtime services that executable/queue admission must provide.
  iree_hal_amd_gpu_kernarg_requirements_t requirements;
  // Number of dense HAL bindings and inline uint16_t destination offsets.
  uint16_t binding_count;
  // Number of coalesced constant copies following the binding offsets.
  uint16_t constant_copy_count;
  // Number of runtime input copies following the constant copies.
  uint16_t runtime_copy_count;
  // Number of HAL-visible parameters; hidden arguments are excluded.
  uint16_t parameter_count;
  // Byte offset from this header to aligned cold parameter reflection.
  uint32_t parameters_offset;
  // Allocator owning the header and all inline records.
  iree_allocator_t host_allocator;
} iree_hal_amd_gpu_kernarg_layout_t;

// Borrows binding_count native byte offsets, indexed by dense HAL binding.
// Recorders use these prepared destinations to relocate an argument template
// without walking parameter reflection when the recording is invoked.
static inline const uint16_t* iree_hal_amd_gpu_kernarg_layout_binding_offsets(
    const iree_hal_amd_gpu_kernarg_layout_t* layout) {
  return (const uint16_t*)(layout + 1);
}

// Constructs a layout from trusted, range-checked image metadata. This cold
// semantic boundary admits argument kinds, widths, and HAL representation;
// it does not repeat the image parser's ordering/range validation. Unsupported
// kinds or missing nonempty argument metadata fail before allocation. All
// tables are copied; only parameter names continue to borrow image storage.
// Success returns one plain owner; failure leaves out_layout NULL.
iree_status_t iree_hal_amd_gpu_kernarg_layout_allocate(
    const iree_hal_amd_gpu_metadata_kernel_t* kernel,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_kernarg_layout_t** out_layout);

// Releases the plain layout allocation. NULL is ignored.
void iree_hal_amd_gpu_kernarg_layout_free(
    iree_hal_amd_gpu_kernarg_layout_t* layout);

// Borrows parameter_count reflection records in native argument order.
static inline const iree_hal_executable_function_parameter_t*
iree_hal_amd_gpu_kernarg_layout_parameters(
    const iree_hal_amd_gpu_kernarg_layout_t* layout) {
  return (
      const iree_hal_executable_function_parameter_t*)((const uint8_t*)layout +
                                                       layout
                                                           ->parameters_offset);
}

// Writes a complete native argument segment from admitted HAL inputs. Source
// constants cover constant_byte_length bytes; binding_addresses contains
// binding_count device addresses, or is NULL to leave zeroed relocation slots
// in a recorded argument template. The caller supplies byte_length writable
// bytes at target, disjoint from every input. Padding is zeroed and hidden
// fields consume the typed runtime values. This hot transform cannot fail,
// allocate, retain resources, or discover pointers inside user bytes.
void iree_hal_amd_gpu_kernarg_layout_emplace(
    const iree_hal_amd_gpu_kernarg_layout_t* layout, const void* constants,
    const uint64_t* binding_addresses,
    const iree_hal_amd_gpu_kernarg_inputs_t* inputs, void* target);

// Writes prepacked native explicit arguments and runtime-owned hidden fields.
// The caller has admitted constants.data_length in [native_byte_length,
// byte_length]. Native explicit offsets and padding are preserved; omitted
// trailing bytes are zeroed before hidden fields are written. No pointer in
// the supplied bytes gains lifetime management. Indirect native arguments
// bypass host emplacement entirely and remain the caller's native ABI object.
void iree_hal_amd_gpu_kernarg_layout_emplace_native(
    const iree_hal_amd_gpu_kernarg_layout_t* layout,
    iree_const_byte_span_t constants,
    const iree_hal_amd_gpu_kernarg_inputs_t* inputs, void* target);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_KERNARG_H_
