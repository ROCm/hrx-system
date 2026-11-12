// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_CONTROL_CALL_ALIASES_H_
#define LOOM_IMPORT_CXX_CONTROL_CALL_ALIASES_H_

#include <cstddef>
#include <cstdint>
#include <span>

#include "iree/base/internal/arena.h"
#include "iree/base/status.h"

namespace loom::cxx_import {

// Identity facts for one indirect source parameter at a retained call edge.
struct CallAliasParameter {
  // Whether the callee may modify the referenced object.
  bool written;
  // Whether the argument retains any exact or opaque object identity.
  bool has_identity;
  // Whether a reference-returning origin has not reached its fixed point.
  bool unresolved_identity;
};

// One exact half-open component interval supplied by a call argument.
struct CallAliasInterval {
  // Stable source-object identity shared by intervals of the same object.
  const void* binding;
  // Inclusive first flattened component within the source object.
  size_t begin;
  // Exclusive final component, with SIZE_MAX denoting the complete object.
  size_t end;
  // Ordinal of the parameter supplied by this interval.
  size_t parameter;
};

// Selects indirect parameters that require one shared storage identity.
//
// Independent value writebacks preserve source semantics unless overlapping
// arguments include at least one writable parameter. Exact intervals are
// indexed by source binding and unresolved identities conservatively overlap
// every other retained identity. The selected byte span is arena-owned and is
// empty when no parameter requires storage.
class CallAliases {
 public:
  CallAliases(std::span<const CallAliasParameter> parameters,
              std::span<const CallAliasInterval> intervals)
      : parameters_(parameters), intervals_(intervals) {}

  iree_status_t select_storage(iree_arena_allocator_t* scratch_arena,
                               std::span<const uint8_t>* out_selected) const;

 private:
  std::span<const CallAliasParameter> parameters_;
  std::span<const CallAliasInterval> intervals_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_CONTROL_CALL_ALIASES_H_
