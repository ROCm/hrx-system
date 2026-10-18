// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEMORY_PAIR_H_
#define AMDF_SRC_MEMORY_PAIR_H_

#include "amdf/amdf.h"

// Provider-local capabilities of one memory access site.
typedef uint64_t amdf_memory_site_capabilities_t;
enum amdf_memory_site_capability_bits_e {
  // The exact queue family can read this attachment.
  AMDF_MEMORY_SITE_CAPABILITY_READ = UINT64_C(1) << 0,
  // The exact queue family can write this attachment.
  AMDF_MEMORY_SITE_CAPABILITY_WRITE = UINT64_C(1) << 1,
  // This attachment can supply backing to a compatible VMM target.
  AMDF_MEMORY_SITE_CAPABILITY_MAPPING_SOURCE = UINT64_C(1) << 2,
  // This device can install a compatible VMM source attachment.
  AMDF_MEMORY_SITE_CAPABILITY_MAPPING_TARGET = UINT64_C(1) << 3,
  // The local release cost is known, including a zero-cost no-op.
  AMDF_MEMORY_SITE_CAPABILITY_RELEASE_COST_KNOWN = UINT64_C(1) << 4,
  // The local acquire cost is known, including a zero-cost no-op.
  AMDF_MEMORY_SITE_CAPABILITY_ACQUIRE_COST_KNOWN = UINT64_C(1) << 5,
};

// Opaque provider-local identity of one compatibility domain.
typedef struct amdf_memory_compatibility_domain_t {
  // Provider-defined identity words.
  uint64_t words[2];
} amdf_memory_compatibility_domain_t;

// Returns true when a compatibility-domain identity is available.
static inline bool amdf_memory_compatibility_domain_is_valid(
    const amdf_memory_compatibility_domain_t* domain) {
  return (domain->words[0] | domain->words[1]) != 0;
}

// Returns true when two compatibility-domain identities are equal.
static inline bool amdf_memory_compatibility_domain_is_equal(
    const amdf_memory_compatibility_domain_t* lhs,
    const amdf_memory_compatibility_domain_t* rhs) {
  return lhs->words[0] == rhs->words[0] && lhs->words[1] == rhs->words[1];
}

// Provider-neutral input to one local memory-site description.
typedef struct amdf_memory_site_query_t {
  // Exact permissions established for the selected device access.
  amdf_memory_access_t access;
  // Backing and access properties established by native construction.
  amdf_memory_flags_t flags;
  // Native target operations established for naturally aligned 32-bit words.
  amdf_atomic_operations_t atomic_operations_32;
  // Native target operations established for naturally aligned 64-bit words.
  amdf_atomic_operations_t atomic_operations_64;
  // Borrowed immutable properties of the exact local queue family.
  const amdf_queue_family_info_t* queue_family_info;
} amdf_memory_site_query_t;

// Host visibility policy independent of mapping addresses and native handles.
typedef struct amdf_memory_host_description_t {
  // Host cache behavior established for this view.
  amdf_host_cacheability_t cacheability;
  // Host cache-line length in bytes, or zero when not applicable.
  uint32_t cache_line_size;
  // Available flush operation before considering the exact peer's coherence.
  amdf_cache_transition_t flush;
  // Available invalidate operation before considering peer coherence.
  amdf_cache_transition_t invalidate;
} amdf_memory_host_description_t;

// Exact local facts composed by the common pair query.
typedef struct amdf_memory_site_description_t {
  // Local read, write, mapping, and cost capabilities.
  amdf_memory_site_capabilities_t capabilities;
  // Visibility operation performed after local writes.
  amdf_cache_transition_t release;
  // Visibility operation performed before local reads.
  amdf_cache_transition_t acquire;
  // Provider-local compatibility domain for VMM source/target composition.
  amdf_memory_compatibility_domain_t mapping_domain;
  // Provider-local compatibility domain for atomic reach composition.
  amdf_memory_compatibility_domain_t atomic_domain;
  // Width-specific atomic reach of this site against this attachment.
  amdf_atomic_reach_t atomic_reach;
  // Fixed local release cost in nanoseconds when reported as known.
  uint64_t release_fixed_cost_nanoseconds;
  // Fixed local acquire cost in nanoseconds when reported as known.
  uint64_t acquire_fixed_cost_nanoseconds;
} amdf_memory_site_description_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Describes a host view against one peer's coherence contract. Native API
// publication and unqualified operations cannot become no-ops through
// coherence. An already-described device peer can qualify SYSTEM atomic widths
// for a write-back view of the same backing; device_peer is NULL for host-only
// pairs.
amdf_memory_site_description_t amdf_memory_describe_host_site(
    const amdf_memory_host_description_t* host, amdf_memory_map_flags_t access,
    bool coherent, const amdf_memory_site_description_t* device_peer);

// Describes a previously qualified ordinary native file-buffer view. Native
// I/O completion/submission supplies the file-side publication/consumption
// boundary; GPU/CPU peer operations and caller scheduling remain independent.
amdf_memory_site_description_t amdf_memory_describe_file_site(
    amdf_memory_map_flags_t access);

// Composes two local descriptions of already-established shared backing reach.
// The caller validates output storage; failure leaves it unchanged. Unsupported
// permissions or transitions are ordinary capability misses, not no-ops.
amdf_status_t amdf_memory_pair_compose(
    const amdf_memory_site_description_t* producer,
    const amdf_memory_site_description_t* consumer,
    amdf_memory_pair_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_MEMORY_PAIR_H_
