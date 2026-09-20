// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_CONFIG_H_
#define LOOM_IMPORT_CXX_BINDING_CONFIG_H_

#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "loom/import/cxx/symbol/names.h"
#include "loom/import/cxx/value/scalar.h"

namespace loom::cxx_import {

// Owns named scalar configuration bindings for one translation unit. Collection
// reconciles source redeclarations and aliases by explicit key before any reads
// are lowered. A definition fixes every alias in this import; declaration-only
// bindings remain symbolic until ordinary Loom configuration specialization.
// Source, diagnostics and scalar builders outlive this owner. Emitted IR
// borrows no frontend storage.
class Configs {
 public:
  Configs(cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
          Scalars& scalars, Locations& locations, SymbolNames& symbol_names)
      : unit_(unit),
        diagnostics_(diagnostics),
        types_(types),
        scalars_(scalars),
        locations_(locations),
        symbol_names_(symbol_names) {}

  // Admits a namespace-scope declaration during the shared declaration walk.
  // Returns whether it is a config binding. Every redeclaration must carry the
  // same leading annotation; unsupported objects and initializers diagnose.
  bool declaration(cxx::Symbol* symbol,
                   cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                   cxx::AST* owner);

  // Completes redeclaration admission and emits one module symbol per key in
  // source order. Called once after collection, before lowering any functions.
  void build(loom_builder_t* builder);

  // Reads a registered binding at the current insertion point, or returns empty
  // for an ordinary source variable. Exact definitions produce constants;
  // unresolved declarations produce typed config.get operations.
  std::optional<loom_value_id_t> read(cxx::VariableSymbol* variable,
                                      loom_builder_t* builder,
                                      loom_location_id_t location);

 private:
  struct Binding {
    // Explicit key borrowed from the source's string literal pool.
    std::string_view name;
    // Unqualified source type retains signedness and nominal enum identity.
    const cxx::Type* source_type;
    // Scalar carrier shared by declarations, definitions and value reads.
    loom_type_t type;
    // Exact source specialization, absent for declaration-only bindings.
    std::optional<loom_attribute_t> value;
    // Declaration or exact definition used as the emitted symbol's location.
    cxx::AST* owner;
    // Module-local identity assigned when collection is complete.
    loom_symbol_ref_t reference;
  };

  // Invocation-owned frontend identities, types and redeclaration chains.
  cxx::TranslationUnit& unit_;
  // Source admission diagnostic boundary.
  Diagnostics& diagnostics_;
  // Scalar source representation projection.
  Types& types_;
  // Shared source constant encoding, including signed storage normalization.
  Scalars& scalars_;
  // Copies source provenance into the output module.
  Locations& locations_;
  // Shared namespace reserves one exact symbol per reconciled config key.
  SymbolNames& symbol_names_;
  // Deterministic first-declaration order, independent of hash-table iteration.
  std::vector<Binding> bindings_;
  // Explicit names reconcile different C++ symbols denoting the same setting.
  std::unordered_map<std::string_view, size_t> names_;
  // Canonical source identities provide direct lookup at every read.
  std::unordered_map<cxx::VariableSymbol*, size_t> variables_;
  // Individual admitted declarations enforce explicit annotations on the full
  // frontend-owned redeclaration chain without indexing ordinary globals.
  std::unordered_set<cxx::VariableSymbol*> declarations_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_CONFIG_H_
