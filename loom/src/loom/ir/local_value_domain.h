// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Function/region-local value domains backed by module ordinal scratch.
//
// A local value domain acquires a dense value_id -> value_ordinal map in the
// module-owned scratch table for one compiler frame. The module owns the large
// reusable scratch storage; the domain owns only the compact ordinal ->
// value_id list needed for cleanup, iteration, diagnostics, and dense phase
// tables.

#ifndef LOOM_IR_LOCAL_VALUE_DOMAIN_H_
#define LOOM_IR_LOCAL_VALUE_DOMAIN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_local_value_domain_flag_bits_e {
  // The domain currently owns the module value-ordinal scratch map.
  LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED = 1u << 0,
  // The domain covers values in recursively nested regions below |region|.
  LOOM_LOCAL_VALUE_DOMAIN_FLAG_REGION_TREE = 1u << 1,
};
typedef uint16_t loom_local_value_domain_flags_t;

typedef struct loom_local_value_domain_t {
  // Module whose value-ordinal scratch map stores this domain.
  loom_module_t* module;
  // Region whose local values are covered by this domain.
  const loom_region_t* region;
  // Function/region-local value IDs indexed by local value ordinal.
  loom_value_id_t* value_ids;
  // Number of initialized value IDs in value_ids.
  loom_value_ordinal_t value_count;
  // Allocated capacity of value_ids.
  iree_host_size_t value_capacity;
  // Domain lifecycle flags.
  loom_local_value_domain_flags_t flags;
} loom_local_value_domain_t;

// Acquires a local value domain for |region| in |module|'s ordinal scratch.
//
// The domain initially registers block arguments, op results, op operands, SSA
// references carried by value types, and values captured by nested regions.
// Rewriting frames that create new values while the domain is active must
// explicitly register those values before indexing ordinal-keyed scratch.
iree_status_t loom_local_value_domain_acquire_for_region(
    loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_local_value_domain_t* out_domain);

// Acquires a local value domain for |region| and every nested region below it.
//
// This is the domain shape used by structured-control analyses and emitters
// that need to assign storage to values defined inside op-owned regions.
iree_status_t loom_local_value_domain_acquire_for_region_tree(
    loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_local_value_domain_t* out_domain);

// Clears all acquired value IDs and releases the module ordinal scratch map.
void loom_local_value_domain_release(loom_local_value_domain_t* domain);

// Registers |value_id| in an acquired domain and returns its local ordinal.
// Existing registrations return their current ordinal. This keeps rewrite
// frames compact while allowing new values to join the same ordinal-keyed
// scratch domain as they are created.
iree_status_t loom_local_value_domain_register_value(
    loom_local_value_domain_t* domain, iree_arena_allocator_t* arena,
    loom_value_id_t value_id, loom_value_ordinal_t* out_ordinal);

// Returns true while the domain owns the module ordinal scratch map.
static inline bool loom_local_value_domain_is_acquired(
    const loom_local_value_domain_t* domain) {
  return iree_any_bit_set(domain->flags, LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED);
}

// Returns the local ordinal for |value_id|, or INVALID when the value is not
// covered by the acquired domain.
static inline loom_value_ordinal_t loom_local_value_domain_try_ordinal(
    const loom_local_value_domain_t* domain, loom_value_id_t value_id) {
  IREE_ASSERT(
      iree_any_bit_set(domain->flags, LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED));
  const loom_value_u32_scratch_t* scratch = &domain->module->scratch.values;
  if (value_id >= domain->module->values.count) {
    return LOOM_VALUE_ORDINAL_INVALID;
  }
  const loom_value_ordinal_t value_ordinal =
      scratch->values_by_value_id[value_id];
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return LOOM_VALUE_ORDINAL_INVALID;
  }
  IREE_ASSERT_LT(value_ordinal, domain->value_count);
  IREE_ASSERT_EQ(domain->value_ids[value_ordinal], value_id);
  return value_ordinal;
}

// Returns the local ordinal for |value_id|. This is an invariant-checked direct
// array lookup: callers must only ask for values covered by the acquired
// domain.
static inline loom_value_ordinal_t loom_local_value_domain_ordinal(
    const loom_local_value_domain_t* domain, loom_value_id_t value_id) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(domain, value_id);
  IREE_ASSERT_NE(value_ordinal, LOOM_VALUE_ORDINAL_INVALID);
  return value_ordinal;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_LOCAL_VALUE_DOMAIN_H_
