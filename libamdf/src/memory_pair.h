// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEMORY_PAIR_H_
#define AMDF_SRC_MEMORY_PAIR_H_

#include "amdf/amdf.h"

// Provider-local capabilities of one concrete attachment access site.
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
  // Borrowed immutable facts of the selected device access.
  const amdf_memory_access_info_t* access_info;
  // Borrowed immutable properties of the exact local queue family.
  const amdf_queue_family_info_t* queue_family_info;
} amdf_memory_site_query_t;

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

#endif  // AMDF_SRC_MEMORY_PAIR_H_
