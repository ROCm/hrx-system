// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// View representations shared by boundary projection and native callers.

#ifndef LOOM_TARGET_VIEW_BOUNDARY_H_
#define LOOM_TARGET_VIEW_BOUNDARY_H_

#include "loom/ir/types.h"

// Physical representation of source views crossing callable and CFG boundaries.
typedef enum loom_target_view_boundary_carrier_e {
  // Preserve one semantic view for target-selected direct lowering.
  LOOM_TARGET_VIEW_BOUNDARY_CARRIER_DIRECT = 0,
  // Transport a scalar-element view as a materializing buffer and byte offset.
  // Reference-element views cannot use byte-addressable buffer storage and
  // retain their semantic carrier for separate target lowering.
  // Shape and element type remain in the source signature; dynamic dimensions
  // are separate scalar parameters or results, not extra view components.
  LOOM_TARGET_VIEW_BOUNDARY_CARRIER_BUFFER_OFFSET = 1,
  // Use BUFFER_OFFSET for owning arguments and results, but preserve explicitly
  // nonescaping view arguments for a target's direct borrowed representation.
  LOOM_TARGET_VIEW_BOUNDARY_CARRIER_BORROWED_DIRECT = 2,
} loom_target_view_boundary_carrier_t;

// Ordered components of the BUFFER_OFFSET representation. The buffer owns the
// storage lifetime; the offset identifies the logical view's byte origin.
typedef enum loom_view_boundary_component_e {
  LOOM_VIEW_BOUNDARY_COMPONENT_BUFFER = 0,
  LOOM_VIEW_BOUNDARY_COMPONENT_OFFSET = 1,
  LOOM_VIEW_BOUNDARY_COMPONENT_COUNT = 2,
} loom_view_boundary_component_t;

// Returns the semantic carrier type lowered by the selected target.
static inline loom_type_t loom_view_boundary_component_type(
    loom_view_boundary_component_t component) {
  return component == LOOM_VIEW_BOUNDARY_COMPONENT_BUFFER
             ? loom_type_buffer()
             : loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
}

// Returns the derived value-name suffix used by boundary projection.
static inline iree_string_view_t loom_view_boundary_component_name(
    loom_view_boundary_component_t component) {
  return component == LOOM_VIEW_BOUNDARY_COMPONENT_BUFFER ? IREE_SV("root")
                                                          : IREE_SV("offset");
}

// Returns the number of carriers contributed by this view representation.
// Other semantic types retain their own boundary contracts.
static inline uint16_t loom_view_boundary_component_count(
    loom_target_view_boundary_carrier_t carrier, loom_type_t type) {
  return loom_type_is_view(type) && !loom_type_has_reference_element(type) &&
                 carrier != LOOM_TARGET_VIEW_BOUNDARY_CARRIER_DIRECT
             ? LOOM_VIEW_BOUNDARY_COMPONENT_COUNT
             : 1;
}

#endif  // LOOM_TARGET_VIEW_BOUNDARY_H_
