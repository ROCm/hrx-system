// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_DUPLICATION_H_
#define LOOM_IMPORT_CXX_VALUE_DUPLICATION_H_

#include "loom/import/cxx/source/locations.h"
#include "loom/import/cxx/value/representation.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

// Emits a distinct High ownership identity for one complete admitted source
// value. Records and arrays recurse through their source partitions; dependent
// shaped leaves remain closed tuples whose result types refer to their own
// duplicated dimensions and encoding. This component decides representation,
// not source liveness: callers decide when a semantic copy must be separated.
class ValueDuplication {
 public:
  ValueDuplication(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                   Types& types, ValueArena& arena, Locations& locations,
                   loom_builder_t& builder)
      : unit_(unit),
        diagnostics_(diagnostics),
        types_(types),
        arena_(arena),
        locations_(locations),
        builder_(builder) {}

  // Duplicates |source| according to |source_type|'s admitted partition. The
  // returned binding has the same partition and logical value but fresh High
  // identities for every physical component.
  Value build(Value source, const cxx::Type* source_type, cxx::AST* owner);

 private:
  using BuildFunction = iree_status_t (*)(loom_builder_t*,
                                          const loom_value_id_t*,
                                          iree_host_size_t, const loom_type_t*,
                                          iree_host_size_t, loom_location_id_t,
                                          loom_op_t**);

  Value build_leaf(Value source, const cxx::Type* source_type, cxx::AST* owner,
                   BuildFunction build);
  Value build_partition(Value source, const cxx::Type* source_type,
                        cxx::AST* owner);

  // Source type traits and reference-member projection.
  cxx::TranslationUnit& unit_;
  // Admission boundary for source values without structural copy semantics.
  Diagnostics& diagnostics_;
  // Source partitions and destination-dependent High type binding.
  Types& types_;
  // Immutable storage for flattened aggregate results.
  ValueArena& arena_;
  // Retained source provenance for emitted duplicate operations.
  Locations& locations_;
  // Current insertion point owned by the translation driver.
  loom_builder_t& builder_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_DUPLICATION_H_
