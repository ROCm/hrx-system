// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Device-independent GPU image records and loading into caller-owned storage.

#ifndef IREE_HAL_DRIVERS_AMD_GPU_IMAGE_IMAGE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_IMAGE_IMAGE_H_

#include "iree/base/byte_sequence.h"
#include "iree/hal/drivers/amd/gpu/image/metadata.h"
#include "iree/hal/drivers/amd/gpu/target/identity.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum iree_hal_amd_gpu_image_access_bits_e {
  IREE_HAL_AMD_GPU_IMAGE_ACCESS_READ = 1u << 0,
  IREE_HAL_AMD_GPU_IMAGE_ACCESS_WRITE = 1u << 1,
  IREE_HAL_AMD_GPU_IMAGE_ACCESS_EXECUTE = 1u << 2,
} iree_hal_amd_gpu_image_access_bits_t;
typedef uint32_t iree_hal_amd_gpu_image_access_t;

// One independently mappable part of a linked virtual-address extent.
typedef struct iree_hal_amd_gpu_image_segment_t {
  // Byte offset within the retained source sequence.
  uint64_t source_offset;
  // Bytes copied from the source before the zero-filled tail.
  uint64_t source_length;
  // Byte offset from the loaded image base, preserving linked address geometry.
  uint64_t offset;
  // Mapped byte length, including the zero-filled tail.
  uint64_t length;
  // Device access required after loading and publication.
  iree_hal_amd_gpu_image_access_t access;
} iree_hal_amd_gpu_image_segment_t;

typedef enum iree_hal_amd_gpu_image_address_kind_e {
  // Offset from the loaded image base address.
  IREE_HAL_AMD_GPU_IMAGE_ADDRESS_IMAGE = 0,
  // Absolute address, independent of image placement.
  IREE_HAL_AMD_GPU_IMAGE_ADDRESS_ABSOLUTE,
  // Offset from an explicitly supplied import binding.
  IREE_HAL_AMD_GPU_IMAGE_ADDRESS_IMPORT,
} iree_hal_amd_gpu_image_address_kind_t;

// A resolved reference requiring no symbol lookup during materialization.
typedef struct iree_hal_amd_gpu_image_address_t {
  // Interpretation of offset and import_ordinal.
  iree_hal_amd_gpu_image_address_kind_t kind;
  // Dense import ordinal when kind is IMPORT; otherwise zero.
  uint32_t import_ordinal;
  // Unsigned 64-bit address addend, including any signed relocation addend.
  uint64_t offset;
} iree_hal_amd_gpu_image_address_t;

typedef enum iree_hal_amd_gpu_image_symbol_kind_e {
  IREE_HAL_AMD_GPU_IMAGE_SYMBOL_UNSPECIFIED = 0,
  IREE_HAL_AMD_GPU_IMAGE_SYMBOL_DATA,
  IREE_HAL_AMD_GPU_IMAGE_SYMBOL_FUNCTION,
} iree_hal_amd_gpu_image_symbol_kind_t;

typedef enum iree_hal_amd_gpu_image_symbol_visibility_e {
  IREE_HAL_AMD_GPU_IMAGE_SYMBOL_VISIBILITY_DEFAULT = 0,
  IREE_HAL_AMD_GPU_IMAGE_SYMBOL_VISIBILITY_INTERNAL,
  IREE_HAL_AMD_GPU_IMAGE_SYMBOL_VISIBILITY_HIDDEN,
  IREE_HAL_AMD_GPU_IMAGE_SYMBOL_VISIBILITY_PROTECTED,
} iree_hal_amd_gpu_image_symbol_visibility_t;

// A named symbol exposed by the image, sorted lexicographically by name.
typedef struct iree_hal_amd_gpu_image_symbol_t {
  // Name borrowed from image-owned or retained source storage.
  iree_string_view_t name;
  // Address relative to image placement or explicit imports.
  iree_hal_amd_gpu_image_address_t address;
  // Declared byte length, or zero if unspecified.
  uint64_t length;
  // Data, callable code, or an untyped symbol.
  iree_hal_amd_gpu_image_symbol_kind_t kind;
  // Cross-image visibility retained for the caller's linking policy.
  iree_hal_amd_gpu_image_symbol_visibility_t visibility;
  // Permissions of the containing image segment, or zero for absolute/import
  // addresses whose storage contract belongs to their external owner.
  iree_hal_amd_gpu_image_access_t access;
  // Whether a stronger definition may replace this symbol during linking.
  bool weak;
} iree_hal_amd_gpu_image_symbol_t;

typedef struct iree_hal_amd_gpu_image_import_t {
  // Name borrowed from image-owned or retained source storage.
  iree_string_view_t name;
  // Symbol semantics required of a supplied binding.
  iree_hal_amd_gpu_image_symbol_kind_t kind;
  // Whether an unresolved binding is permitted and resolves to address zero.
  bool weak;
} iree_hal_amd_gpu_image_import_t;

typedef enum iree_hal_amd_gpu_image_relocation_kind_e {
  IREE_HAL_AMD_GPU_IMAGE_RELOCATION_ABSOLUTE_64 = 0,
  IREE_HAL_AMD_GPU_IMAGE_RELOCATION_ABSOLUTE_32,
  IREE_HAL_AMD_GPU_IMAGE_RELOCATION_LOW_32,
  IREE_HAL_AMD_GPU_IMAGE_RELOCATION_HIGH_32,
} iree_hal_amd_gpu_image_relocation_kind_t;

typedef struct iree_hal_amd_gpu_image_relocation_t {
  // Dense segment ordinal receiving the relocation.
  uint32_t segment_ordinal;
  // Value encoding written into the target field.
  iree_hal_amd_gpu_image_relocation_kind_t kind;
  // Byte offset within the target segment.
  uint64_t offset;
  // Fully normalized address expression including the relocation addend.
  iree_hal_amd_gpu_image_address_t value;
} iree_hal_amd_gpu_image_relocation_t;

// Native dispatch facts in the same ordinal order as metadata.kernels.
typedef struct iree_hal_amd_gpu_image_kernel_t {
  // Image-relative offset of the 64-byte AMDHSA descriptor used by AQL.
  uint64_t descriptor_offset;
  // Image-relative instruction entry offset used by COMPUTE dispatch.
  uint64_t entry_offset;
  // COMPUTE_PGM_RSRC1 register payload.
  uint32_t compute_pgm_rsrc1;
  // COMPUTE_PGM_RSRC2 register payload.
  uint32_t compute_pgm_rsrc2;
  // COMPUTE_PGM_RSRC3 register payload.
  uint32_t compute_pgm_rsrc3;
  // AMDHSA hidden-SGPR, wave-size and dynamic-stack property bits.
  uint16_t code_properties;
  // AMDHSA kernarg preload descriptor.
  uint16_t kernarg_preload;
} iree_hal_amd_gpu_image_kernel_t;

// One precisely addressable global in an ASAN-instrumented image. The frontend
// verifies eight-byte alignment and a disjoint trailing eight-byte redzone.
typedef struct iree_hal_amd_gpu_image_data_object_t {
  // Byte offset from the loaded image base.
  uint64_t offset;
  // Addressable bytes, excluding alignment padding and the trailing redzone.
  uint64_t length;
} iree_hal_amd_gpu_image_data_object_t;

// Immutable image description initialized by a serialized-format frontend.
//
// This is a plain owner, not a refcounted HAL resource. The enclosing
// executable or compilation cache owns its lifetime. Querying and loading can
// be concurrent while the image is alive; each load requires exclusive access
// to its backing. Fields are read-only after successful initialization. No
// native allocations, mappings, queues, imports, or execution lifetimes are
// retained by the image.
typedef struct iree_hal_amd_gpu_image_t {
  // Allocator owning storage and any independently allocated metadata records.
  iree_allocator_t host_allocator;
  // Retained immutable payload source; callback spans are never saved.
  iree_byte_sequence_t* source;
  // Allocation owning normalized tables and any gathered structural bytes.
  void* storage;
  // Gathered structural bytes backing reflection when the source is segmented.
  void* reflection_storage;
  // Executable target requirements decoded from the serialized artifact.
  iree_hal_amd_gpu_target_identity_t target;
  // AMDHSA code-object ABI version, independent of the serialized frontend.
  uint32_t code_object_version;
  // Minimum linked virtual extent, including gaps. The native storage owner
  // adds any device-specific speculative instruction-fetch/prefetch padding.
  uint64_t byte_length;
  // Required power-of-two alignment of the loaded image base.
  uint64_t alignment;
  // Number of non-overlapping segments, sorted by image-relative offset.
  iree_host_size_t segment_count;
  // Segment records borrowed from storage.
  iree_hal_amd_gpu_image_segment_t* segments;
  // Number of named symbols, sorted lexicographically by name.
  iree_host_size_t symbol_count;
  // Named symbol records borrowed from storage.
  iree_hal_amd_gpu_image_symbol_t* symbols;
  // Number of import bindings required at load time.
  iree_host_size_t import_count;
  // Import requirements borrowed from storage.
  iree_hal_amd_gpu_image_import_t* imports;
  // Number of address patches performed after payload copying.
  iree_host_size_t relocation_count;
  // Normalized relocation records borrowed from storage.
  iree_hal_amd_gpu_image_relocation_t* relocations;
  // Named dependency count; resolving these belongs to the executable owner.
  iree_host_size_t dependency_count;
  // Dependency names borrowed from storage or the retained source.
  iree_string_view_t* dependencies;
  // Number of precisely addressable globals; zero for unmarked images.
  iree_host_size_t asan_data_object_count;
  // Independently owned global ranges sorted by offset, excluding descriptors.
  iree_hal_amd_gpu_image_data_object_t* asan_data_objects;
  // Complete kernel ABI and reflection metadata owned by the image.
  iree_hal_amd_gpu_metadata_t metadata;
  // Dispatch facts borrowed from storage, indexed by metadata kernel ordinal.
  iree_hal_amd_gpu_image_kernel_t* kernels;
} iree_hal_amd_gpu_image_t;

// Releases records and the retained source, performing no device operation.
// Safe for a zero-initialized image. Clears the image before returning.
void iree_hal_amd_gpu_image_deinitialize(iree_hal_amd_gpu_image_t* image);

// Finds an exact symbol name, returning a borrowed record or NULL if absent.
const iree_hal_amd_gpu_image_symbol_t* iree_hal_amd_gpu_image_find_symbol(
    const iree_hal_amd_gpu_image_t* image, iree_string_view_t name);

typedef struct iree_hal_amd_gpu_image_import_binding_t {
  // Address visible to the executing device; zero is a valid resolved value.
  uint64_t address;
  // Whether the caller resolved this import, independently of its address.
  bool resolved;
} iree_hal_amd_gpu_image_import_binding_t;

// Resolves an admitted reference with previously validated placement/bindings.
// Performs no allocation, validation, or symbol lookup.
uint64_t iree_hal_amd_gpu_image_resolve_address(
    iree_hal_amd_gpu_image_address_t reference, uint64_t base_address,
    const iree_hal_amd_gpu_image_import_binding_t* bindings);

// Copies source ranges, zeros declared tails, and applies address relocations.
//
// The caller supplies one writable mapping per segment in image order; each
// mapping starts at that segment's first byte and covers its full length. All
// device mappings must preserve the image's linked offsets from base_address.
// Mapping permissions, native allocation, publication, and execution retirement
// belong to the caller. Import storage stays live through its last execution.
//
// External arguments are validated before writes. The image is already admitted
// trusted state. A source failure may leave partial bytes that cannot be used
// for execution. Source and destination storage must not overlap. Gaps between
// segments are untouched. This operation allocates nothing and enumerates a
// segmented source once for all payload ranges.
iree_status_t iree_hal_amd_gpu_image_load(
    const iree_hal_amd_gpu_image_t* image, uint64_t base_address,
    iree_host_size_t binding_count,
    const iree_hal_amd_gpu_image_import_binding_t* bindings,
    iree_host_size_t mapping_count, const iree_byte_span_t* mappings);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_IMAGE_IMAGE_H_
