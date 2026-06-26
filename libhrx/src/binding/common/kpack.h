// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// In-tree resolver for ROCm "kpack" out-of-band code objects.
//
// kpack-packaged ROCm builds strip device code out of the host .so into sibling
// ".kpack" archives to shrink the shared object. The HIP loader then hands the
// runtime a "HIPK" fat-binary wrapper whose binary pointer is msgpack metadata
// (a kernel name plus a list of candidate .kpack search paths) instead of an
// inline code object. This module resolves that metadata into the real AMDGPU
// code object at registration time, with no dependency on the vendor
// librocm_kpack loader: it parses the msgpack metadata and the .kpack archive
// (header + msgpack table of contents), selects the code object matching the
// requested GPU target (honoring ISA feature-flag subsetting), and decompresses
// it (uncompressed "none" or "zstd-per-kernel" schemes).
//
// The on-disk formats mirror the reference implementation at
// rocm-systems/shared/kpack (runtime/ and python/rocm_kpack/). Only the runtime
// unpacking side is implemented here; HRX is not a compiler and never produces
// kpack archives.
//
// Everything in this header operates on caller-provided memory and is free of
// global state so the individual stages (metadata parse, archive parse, target
// matching, path resolution) can be unit tested in isolation.

#ifndef IREE_HAL_STREAMING_KPACK_H_
#define IREE_HAL_STREAMING_KPACK_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum {
  // Maximum candidate ISA target string length (e.g.
  // "gfx942:sramecc+:xnack-"), including the NUL terminator.
  IREE_HAL_STREAMING_KPACK_TARGET_CAPACITY = 64,
  // Maximum kpack search paths parsed out of a single HIPK metadata blob.
  IREE_HAL_STREAMING_KPACK_MAX_SEARCH_PATHS = 32,
  // Upper bound on metadata blob size walked by the msgpack reader. The HIP ABI
  // carries no length for the wrapper's binary pointer, so the reader is capped
  // here to avoid running off the end of the owning ELF section.
  IREE_HAL_STREAMING_KPACK_MAX_METADATA_SIZE = 64 * 1024,
};

// Archive-wide kernel storage scheme (from the TOC "compression_scheme" field).
typedef enum iree_hal_streaming_kpack_compression_e {
  // Uncompressed: code objects are concatenated and indexed by absolute file
  // offset/size ("blobs" array).
  IREE_HAL_STREAMING_KPACK_COMPRESSION_NONE = 0,
  // Each code object is an independent zstd frame in a single blob, indexed by
  // ordinal.
  IREE_HAL_STREAMING_KPACK_COMPRESSION_ZSTD_PER_KERNEL = 1,
} iree_hal_streaming_kpack_compression_t;

//===----------------------------------------------------------------------===//
// ISA target matching
//===----------------------------------------------------------------------===//
// GPU code objects may be built with a subset of the agent's ISA feature flags
// (e.g. an agent reports "gfx942:sramecc+:xnack-" but a release archive is
// keyed by bare "gfx942"). Omitted features mean "Any", so a requested target
// is compatible with any archive key formed by dropping a subset of its
// features. This mirrors isCompatibleTargetID() across LLVM/comgr/HSA/CLR.

// Strips a leading "amdgcn-amd-amdhsa--" target-triple prefix if present.
// "amdgcn-amd-amdhsa--gfx942:xnack-" -> "gfx942:xnack-"; "gfx1100" ->
// "gfx1100".
iree_string_view_t iree_hal_streaming_kpack_strip_target_prefix(
    iree_string_view_t isa);

// Callback invoked with each compatible target candidate. |target| borrows a
// transient buffer valid only for the duration of the call. Return true to stop
// iteration early (match found).
typedef bool (*iree_hal_streaming_kpack_target_callback_t)(
    iree_string_view_t target, void* user_data);

// Invokes |callback| for each ISA target compatible with |agent_isa|, ordered
// most-specific first (full feature set, then every feature subset by
// descending cardinality, then the bare processor). Stops early and returns
// true if a callback returns true; returns false otherwise (including empty
// input). Targets longer than IREE_HAL_STREAMING_KPACK_TARGET_CAPACITY are
// skipped.
bool iree_hal_streaming_kpack_for_each_compatible_target(
    iree_string_view_t agent_isa,
    iree_hal_streaming_kpack_target_callback_t callback, void* user_data);

//===----------------------------------------------------------------------===//
// HIPK metadata
//===----------------------------------------------------------------------===//

// Parsed view of a HIPK msgpack metadata blob. All string views borrow from the
// input buffer and are valid only as long as it is.
typedef struct iree_hal_streaming_kpack_metadata_t {
  // Base TOC key for the owning binary, e.g. "lib/libfoo.so". The runtime
  // appends "#<co_index>" to form the archive lookup key.
  iree_string_view_t kernel_name;
  // Candidate .kpack paths relative to the owning binary. Entries may contain
  // the "@GFXARCH@" placeholder to be expanded per target architecture.
  iree_string_view_t search_paths[IREE_HAL_STREAMING_KPACK_MAX_SEARCH_PATHS];
  iree_host_size_t search_path_count;
} iree_hal_streaming_kpack_metadata_t;

// Parses HIPK msgpack metadata ({"kernel_name": str, "kpack_search_paths":
// [str, ...]}). The reader self-bounds within |data|; pass a generous upper
// bound when the true length is unknown (see MAX_METADATA_SIZE). Fails with
// IREE_STATUS_INVALID_ARGUMENT if the blob is not the expected shape or carries
// no search paths.
iree_status_t iree_hal_streaming_kpack_parse_metadata(
    iree_const_byte_span_t data,
    iree_hal_streaming_kpack_metadata_t* out_metadata);

//===----------------------------------------------------------------------===//
// Archive parsing
//===----------------------------------------------------------------------===//

// A parsed .kpack archive over an in-memory buffer. All spans borrow from
// |archive| and are valid only as long as it is.
typedef struct iree_hal_streaming_kpack_archive_t {
  iree_const_byte_span_t archive;  // Whole-file bytes.
  uint32_t version;
  iree_hal_streaming_kpack_compression_t compression;
  iree_const_byte_span_t toc_map;  // msgpack bytes of the top-level TOC map.
  // For COMPRESSION_NONE: msgpack bytes of the "blobs" array (indexed by
  // ordinal). For COMPRESSION_ZSTD_PER_KERNEL: empty.
  iree_const_byte_span_t blobs_array;
  // For COMPRESSION_ZSTD_PER_KERNEL: the [num_kernels][frame...] blob bytes.
  // For COMPRESSION_NONE: empty.
  iree_const_byte_span_t zstd_blob;
} iree_hal_streaming_kpack_archive_t;

// Validates the 16-byte fixed header ("KPAK", version, TOC offset) of an
// in-memory archive and parses the top-level msgpack TOC, recording the
// compression scheme and blob locations. Fails with
// IREE_STATUS_INVALID_ARGUMENT on bad magic/structure and
// IREE_STATUS_INCOMPATIBLE on unsupported versions.
iree_status_t iree_hal_streaming_kpack_archive_open(
    iree_const_byte_span_t archive_bytes,
    iree_hal_streaming_kpack_archive_t* out_archive);

// Looks up the code object for (|binary_key|, |arch|) in the archive's TOC and
// returns a freshly-allocated, decompressed copy via |host_allocator| (caller
// frees with iree_allocator_free). |binary_key| is the full indexed key, e.g.
// "lib/libfoo.so#0"; |arch| is an exact TOC architecture key, e.g. "gfx942".
// Fails with IREE_STATUS_NOT_FOUND when the key/arch is absent and
// IREE_STATUS_UNIMPLEMENTED for zstd archives when built without
// HRX_ENABLE_ZSTD.
iree_status_t iree_hal_streaming_kpack_archive_get_kernel(
    const iree_hal_streaming_kpack_archive_t* archive,
    iree_string_view_t binary_key, iree_string_view_t arch,
    iree_allocator_t host_allocator, void** out_kernel,
    iree_host_size_t* out_kernel_size);

//===----------------------------------------------------------------------===//
// Path resolution and discovery
//===----------------------------------------------------------------------===//

// Resolves |relative| against the directory containing |base_path| with lexical
// "." / ".." normalization (no symlink resolution / filesystem access). An
// absolute |relative| is normalized and returned as-is. Writes a NUL-terminated
// path into |out|.
iree_status_t iree_hal_streaming_kpack_resolve_relative_path(
    iree_string_view_t base_path, iree_string_view_t relative, char* out,
    iree_host_size_t out_capacity);

// Replaces the first "@GFXARCH@" placeholder in |pattern| with |arch|, writing
// a NUL-terminated result into |out|. If no placeholder is present |pattern| is
// copied unchanged. |out_had_placeholder| (optional) reports whether a
// placeholder was found.
iree_status_t iree_hal_streaming_kpack_expand_gfxarch(
    iree_string_view_t pattern, iree_string_view_t arch, char* out,
    iree_host_size_t out_capacity, bool* out_had_placeholder);

// Discovers the filesystem path of the loaded binary containing
// |address_in_binary| (e.g. the HIPK wrapper's binary pointer), used to resolve
// archive paths relative to the owning library. On Linux this scans
// /proc/self/maps. |out_offset| (optional) receives the address's offset within
// the file. Fails with IREE_STATUS_NOT_FOUND if the address maps to no file and
// IREE_STATUS_UNIMPLEMENTED on unsupported platforms.
iree_status_t iree_hal_streaming_kpack_discover_binary_path(
    const void* address_in_binary, char* out, iree_host_size_t out_capacity,
    iree_host_size_t* out_offset);

//===----------------------------------------------------------------------===//
// Top-level resolution
//===----------------------------------------------------------------------===//

// Resolves a HIPK out-of-band code object into an owned buffer.
//
// Parses |hipk_metadata| (the wrapper's binary pointer), forms the lookup key
// "<kernel_name>#<co_index>", locates the sibling .kpack archive(s) (relative
// to the binary owning |hipk_metadata|, with ROCM_KPACK_PATH / _PATH_PREFIX /
// _DISABLE / _DEBUG environment overrides honored), and returns the first code
// object matching |target_archs| in priority order. Each target is expanded
// through its compatible ISA feature subsets, and "@GFXARCH@" placeholders in
// search paths are expanded per candidate architecture.
//
// On success |out_code_object| is a freshly-allocated buffer owned by the
// caller (free with iree_allocator_free); it holds a raw AMDGPU code object
// (ELF, a concatenation of ELFs, or a Clang offload bundle) ready to feed back
// through the normal fat-binary extraction path. On failure both outputs are
// left untouched.
iree_status_t iree_hal_streaming_kpack_resolve_code_object(
    const void* hipk_metadata, uint32_t co_index,
    iree_host_size_t target_arch_count, const iree_string_view_t* target_archs,
    iree_allocator_t host_allocator, void** out_code_object,
    iree_host_size_t* out_code_object_size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_STREAMING_KPACK_H_
