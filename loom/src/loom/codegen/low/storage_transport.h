// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Storage traffic incorporated into synchronous callable boundaries.

#ifndef LOOM_CODEGEN_LOW_STORAGE_TRANSPORT_H_
#define LOOM_CODEGEN_LOW_STORAGE_TRANSPORT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/ir/local_value_domain.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_low_function_model_t;

typedef struct loom_low_storage_transport_binding_t {
  // Function-local space projected by the target's invocation frame.
  loom_storage_space_t space;
  // Descriptor class defining the cell width and ownership semantics.
  uint16_t register_class;
  // Byte offset within |space|, including the root reservation and view.
  uint64_t byte_offset;
} loom_low_storage_transport_binding_t;

typedef struct loom_low_storage_transport_effect_t {
  // Immediate-body source ordinal of a call or return with folded traffic.
  uint32_t source_ordinal;
  // Memory effects of the folded traffic, supplementing semantic traits.
  loom_trait_flags_t traits;
} loom_low_storage_transport_effect_t;

typedef struct loom_low_storage_transport_t {
  // Arena-owned bindings, indexed by final STORAGE allocation locations.
  const loom_low_storage_transport_binding_t* bindings;
  // Number of initialized bindings.
  iree_host_size_t binding_count;
  // Binding index by local value ordinal, or UINT32_MAX. NULL without folds.
  const uint32_t* bindings_by_value_ordinal;
  // Physical memory effects at affected boundaries, in source order.
  const loom_low_storage_transport_effect_t* effects;
  // Number of boundaries with additional physical effects.
  iree_host_size_t effect_count;
} loom_low_storage_transport_t;

// Proves storage transfers at callable boundaries in one immutable function
// snapshot. Reloads may join their sole boundary consumer across intervening
// operations that preserve ordinary read ordering. A stored entry argument or
// call result has one store use in the boundary's store prefix and an
// unescaped, single-writer root reservation. All other instruction occurrences
// retain ordinary register materialization. The returned plan borrows the
// model's IR and local value numbering and must be rebuilt after any IR
// mutation. No plan arrays are allocated without folds.
iree_status_t loom_low_storage_transport_build(
    const struct loom_low_function_model_t* model,
    loom_low_storage_space_set_t synchronous_spaces,
    iree_arena_allocator_t* arena,
    const loom_low_storage_transport_t** out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_STORAGE_TRANSPORT_H_
