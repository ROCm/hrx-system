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
  // Maximum kpack search paths parsed out of a single HIPK metadata blob. The
  // format does not bound the count and the reference kpack packager emits one;
  // this buffers a generous ceiling and rejects a blob that lists more.
  IREE_HAL_STREAMING_KPACK_MAX_SEARCH_PATHS = 32,
  // Upper bound on metadata blob size walked by the msgpack reader. The
  // reference kpack runtime reads the HIPK metadata through a 64 KiB window and
  // lets the msgpack parse self-terminate at the end of the real data; HRX
  // matches that window. The HIP ABI carries no length for the wrapper's binary
  // pointer, so what makes the walk memory-safe is the extent of the mapping
  // holding the blob (see iree_hal_streaming_kpack_query_mapping); this cap
  // sits on top of that extent, clamping the span handed to the reader so a
  // blob in a large mapping is still walked no further than real metadata ever
  // reaches. It bounds what the reader can see rather than validating a size: a
  // value the parse needs that lies past the clamp is invisible and fails as a
  // missing or truncated field, while bytes the parse never reaches are never
  // examined.
  IREE_HAL_STREAMING_KPACK_MAX_METADATA_SIZE = 64 * 1024,
  // Maximum archives opened while resolving one code object. The reference
  // kpack runtime searches candidate archives without a count bound; HRX bounds
  // the ranked set because the search ranks targets across every opened
  // archive, so truncating it could silently select a compatible but
  // lower-ranked code object. Exceeding this is reported as
  // IREE_STATUS_RESOURCE_EXHAUSTED instead.
  IREE_HAL_STREAMING_KPACK_MAX_OPEN_ARCHIVES = 64,
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
// most-specific first: candidates are ranked by descending feature count, and
// within one count by descending feature-bitmask (each feature maps to a bit,
// earlier-listed features to higher bits, so higher-priority features are
// retained longest). The full feature set comes first and the bare processor
// last. A callback returning true stops iteration early. Empty input yields no
// candidates and succeeds.
//
// Fails with IREE_STATUS_OUT_OF_RANGE when |agent_isa| carries more subsettable
// feature flags than the expansion holds, or a processor that does not fit
// IREE_HAL_STREAMING_KPACK_TARGET_CAPACITY: neither can be represented, and
// silently expanding a truncated feature set or dropping a whole target would
// produce a wrong candidate set rather than a diagnosable failure. A single
// feature-bearing candidate that would not fit the capacity is skipped instead,
// since a shorter subset of the same target still resolves.
iree_status_t iree_hal_streaming_kpack_for_each_compatible_target(
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
// [str, ...]}). The reader self-bounds within |data|, which must therefore be a
// span the caller has established is readable: the blob carries no length of
// its own, and every bound the reader enforces is measured against |data|'s
// end (see iree_hal_streaming_kpack_query_mapping). Fails with
// IREE_STATUS_INVALID_ARGUMENT if the blob is not the expected shape or carries
// no search paths, and IREE_STATUS_OUT_OF_RANGE if it lists more than
// IREE_HAL_STREAMING_KPACK_MAX_SEARCH_PATHS paths.
//
// A search-path entry that is not a string is skipped rather than rejected: it
// names no path, so it drops out of a list that is only a set of candidates to
// try. The list itself must still be an array, and one that yields no usable
// entry fails as though it were empty.
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
// IREE_STATUS_INVALID_ARGUMENT on bad magic/structure, including a
// "compression_scheme" field that is not a string or does not name a supported
// scheme, and IREE_STATUS_INCOMPATIBLE on unsupported versions.
iree_status_t iree_hal_streaming_kpack_archive_open(
    iree_const_byte_span_t archive_bytes,
    iree_hal_streaming_kpack_archive_t* out_archive);

// Looks up the code object for (|binary_key|, |arch|) in the archive's TOC and
// returns a freshly-allocated, decompressed copy via |host_allocator| (caller
// frees with iree_allocator_free). |binary_key| is the full indexed key, e.g.
// "lib/libfoo.so#0"; |arch| is an exact TOC architecture key, e.g. "gfx942".
// Fails with IREE_STATUS_NOT_FOUND when the key/arch is absent or its ordinal
// has no backing blob/frame, IREE_STATUS_INVALID_ARGUMENT on a malformed TOC
// entry or zstd blob (missing ordinal, zero-size blob, a blob offset/size
// outside the archive, or a truncated/oversized zstd frame or kernel count),
// IREE_STATUS_DATA_LOSS when zstd decompression fails or the decompressed size
// does not match the recorded original_size, and IREE_STATUS_UNIMPLEMENTED for
// zstd archives in builds without the HIP binding.
iree_status_t iree_hal_streaming_kpack_archive_get_kernel(
    const iree_hal_streaming_kpack_archive_t* archive,
    iree_string_view_t binary_key, iree_string_view_t arch,
    iree_allocator_t host_allocator, void** out_kernel,
    iree_host_size_t* out_kernel_size);

//===----------------------------------------------------------------------===//
// Path resolution and mapping queries
//===----------------------------------------------------------------------===//

// Resolves |relative| against the directory containing |base_path|, simplifying
// "." components, duplicate '/' separators, and a trailing '/' (a leading '/'
// is preserved). ".." is left in place: it does not commute with symlink
// resolution, so collapsing it here could name a different file than the kernel
// resolves at open time. No symlink resolution or filesystem access. An
// absolute |relative| is normalized the same way and returned without joining
// |base_path|. Writes a NUL-terminated path into |out|.
//
// Fails with IREE_STATUS_OUT_OF_RANGE when a path does not fit the buffer
// holding it: the normalized result and its NUL terminator against
// |out_capacity|, or |base_path|'s directory joined with |relative| against the
// buffer that join is staged in.
iree_status_t iree_hal_streaming_kpack_resolve_relative_path(
    iree_string_view_t base_path, iree_string_view_t relative, char* out,
    iree_host_size_t out_capacity);

// Replaces the first "@GFXARCH@" placeholder in |pattern| with |arch|, writing
// a NUL-terminated result into |out|. If no placeholder is present |pattern| is
// copied unchanged. |out_had_placeholder| (optional) reports whether a
// placeholder was found.
//
// Fails with IREE_STATUS_OUT_OF_RANGE if the result and its NUL terminator do
// not fit |out_capacity|, including a zero-capacity |out|.
iree_status_t iree_hal_streaming_kpack_expand_gfxarch(
    iree_string_view_t pattern, iree_string_view_t arch, char* out,
    iree_host_size_t out_capacity, bool* out_had_placeholder);

// Describes the virtual memory mapping containing a queried address.
typedef struct iree_hal_streaming_kpack_mapping_t {
  // Bytes readable at the queried address before the end of its containing
  // mapping. Always nonzero on success. Bounds reads of data that carries no
  // length of its own.
  iree_host_size_t readable_bytes;
  // Filesystem path of the file backing the mapping, NUL-terminated in the
  // caller-provided buffer. Empty when the mapping has a valid extent but no
  // file that can be opened: an anonymous mapping, a special region ([heap],
  // [stack], ...), or one whose backing file has been unlinked.
  iree_string_view_t path;
} iree_hal_streaming_kpack_mapping_t;

// Queries the mapping containing |address| (e.g. the HIPK wrapper's binary
// pointer), reporting both how far it is safe to read from |address| and which
// file backs it, so archive paths can be resolved relative to the owning
// library. On Linux this scans /proc/self/maps; dladdr() cannot reliably
// resolve data segments, and the HIPK wrapper points into a data section.
// |path_buffer| receives the backing path and |out_mapping|->path borrows it.
//
// Fails with IREE_STATUS_INVALID_ARGUMENT on a null address/buffer/output or
// zero |path_capacity|, IREE_STATUS_NOT_FOUND if |address| lies in no mapping
// or in one that cannot be read (a PROT_NONE guard region), including when
// /proc/self/maps cannot be read, IREE_STATUS_OUT_OF_RANGE if the backing path
// does not fit in |path_capacity|, and IREE_STATUS_UNIMPLEMENTED on unsupported
// platforms.
iree_status_t iree_hal_streaming_kpack_query_mapping(
    const void* address, char* path_buffer, iree_host_size_t path_capacity,
    iree_hal_streaming_kpack_mapping_t* out_mapping);

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
// A search path whose open reports it absent (IREE_STATUS_NOT_FOUND) is not
// part of the search space and is skipped: the embedded paths are speculative,
// so a miss is the normal case. Every other open failure, and every failure to
// map or parse what the path names, is a fact about the caller's system that
// the caller cannot otherwise discover, so the candidate is skipped but its
// cause is retained. A path that is malformed rather than absent is therefore
// reported, not skipped: one leading through a component that names a file
// rather than a directory fails the open with
// IREE_STATUS_FAILED_PRECONDITION and is retained. A retained cause is reported
// if the search matches nothing, making such a path diagnosable rather than
// indistinguishable from an absent one, and it surfaces one of two ways:
//
//   If no archive was opened, the search space was empty and a retained cause
//   is the only explanation for the miss, so it is returned as-is and its code
//   is the diagnosis. It may be any code the underlying open, mapping, or
//   archive parse produces (see iree_hal_streaming_kpack_archive_open), or one
//   the resolver produces for a search path it cannot form.
//
//   If archives were opened, the search ran to completion and the miss is a
//   target mismatch, which is reported as IREE_STATUS_NOT_FOUND with any
//   retained cause attached to it as diagnostic context.
//
// Fails with:
//   IREE_STATUS_INVALID_ARGUMENT on a null argument, metadata that is not the
//     expected shape, or no usable |target_archs|.
//   IREE_STATUS_OUT_OF_RANGE when a value exceeds the bound holding it: a
//     lookup key formed from an over-long kernel name, metadata listing more
//     than IREE_HAL_STREAMING_KPACK_MAX_SEARCH_PATHS paths, a requested target
//     that carries more subsettable ISA feature flags than the expansion holds
//     or whose processor does not fit a target candidate buffer, a backing path
//     longer than the resolver's path buffer, or a search path that overruns
//     that buffer once formed and expanded.
//   IREE_STATUS_UNAVAILABLE when ROCM_KPACK_DISABLE is set.
//   IREE_STATUS_NOT_FOUND when |hipk_metadata| lies in no readable mapping, or
//     when no reachable archive holds a code object matching |target_archs|.
//   IREE_STATUS_FAILED_PRECONDITION when a search path cannot be formed or
//     followed: one relative to a binary that no file backs, so there is
//     nothing to resolve it against, or one leading through a component that
//     names a file rather than a directory.
//   IREE_STATUS_RESOURCE_EXHAUSTED when more than
//     IREE_HAL_STREAMING_KPACK_MAX_OPEN_ARCHIVES archives match, as the ranked
//     search cannot be completed and truncating it could silently select a
//     lower-ranked code object. It is also the retained cause when more
//     compatible target candidates are produced than the ranked candidate list
//     holds: the lowest-ranked fallbacks past the cap are not searched, so a
//     miss reports the truncation rather than an unexplained not-found.
//   IREE_STATUS_UNIMPLEMENTED on platforms without a mapping query, since the
//     blob cannot be bounded and so cannot be parsed safely (see
//     iree_hal_streaming_kpack_query_mapping).
//   the retained cause of an unusable candidate, as described above.
//
// On success |out_code_object| is a freshly-allocated buffer owned by the
// caller (free with iree_allocator_free) and |out_code_object_size| is nonzero;
// it holds a raw AMDGPU code object (ELF, a concatenation of ELFs, or a Clang
// offload bundle) ready to feed back through the normal fat-binary extraction
// path. Both outputs are cleared once the arguments are accepted, so on any
// failure past that point they are NULL and zero.
iree_status_t iree_hal_streaming_kpack_resolve_code_object(
    const void* hipk_metadata, uint32_t co_index,
    iree_host_size_t target_arch_count, const iree_string_view_t* target_archs,
    iree_allocator_t host_allocator, void** out_code_object,
    iree_host_size_t* out_code_object_size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_STREAMING_KPACK_H_
