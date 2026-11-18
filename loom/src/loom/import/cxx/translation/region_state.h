// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_TRANSLATION_REGION_STATE_H_
#define LOOM_IMPORT_CXX_TRANSLATION_REGION_STATE_H_

#include <initializer_list>
#include <vector>

#include "loom/import/cxx/value/partition.h"

namespace loom::cxx_import {

class Types;
class ValueBindings;

// Ordered source subobjects carried across one structured control-flow edge.
// The same projections describe its signature, outgoing operands, and incoming
// bindings. Unwritten and uninitialized siblings stay outside the edge.
class RegionState {
 public:
  void append(cxx::Symbol* binding, const ValueProjection& projection);
  bool empty() const { return groups_.empty(); }
  size_t group_count() const { return groups_.size(); }
  size_t component_count() const { return component_count_; }

  // Source types follow any leading expression or control results.
  void append_types(std::vector<const cxx::Type*>& types) const;
  std::vector<const cxx::Type*> source_types(
      std::initializer_list<const cxx::Type*> prefix = {}) const;
  // Scalar loops use their inferred scheme. A dependent group requires the
  // entire source scheme to bind fresh region arguments and result identities.
  std::vector<const cxx::Type*> dependent_types(Types& types,
                                                cxx::AST* owner) const;

  // Append directly after any leading results. A continuing edge missing a
  // source value fails at its source owner, before the region can be published.
  [[nodiscard]] bool append_values(const ValueBindings& bindings,
                                   ValueArena& arena,
                                   std::vector<loom_value_id_t>& values) const;
  void bind_values(ValueBindings& bindings, ValueArena& arena,
                   loom_module_t* module, const loom_value_id_t* values) const;

 private:
  struct Group {
    // Automatic source binding containing the projected subobject.
    cxx::Symbol* binding;
    // Source type used to bind dependent destination identities.
    const cxx::Type* type;
    // Complete transported SSA group.
    const Partition* partition;
    // First component in the source binding. Physical object offsets and
    // managed ownership flags play no part in a structured region edge.
    size_t offset;
  };
  // Source order selected by the control analysis and current binding state.
  std::vector<Group> groups_;
  // Flattened SSA arity, accumulated as groups are selected.
  size_t component_count_ = 0;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_TRANSLATION_REGION_STATE_H_
