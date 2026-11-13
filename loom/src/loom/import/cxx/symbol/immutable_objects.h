// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SYMBOL_IMMUTABLE_OBJECTS_H_
#define LOOM_IMPORT_CXX_SYMBOL_IMMUTABLE_OBJECTS_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "loom/import/cxx/source/locations.h"
#include "loom/import/cxx/symbol/names.h"
#include "loom/ir/module.h"

namespace loom::cxx_import {

// One reached source-owned immutable object. Its symbol and source layout are
// stable for the translation-unit lifetime; SSA roots remain function-local.
struct ImmutableObject {
  // Symbol naming the emitted read-only declaration or payload.
  loom_symbol_ref_t symbol;
  // Stable module-owned spelling suitable for derived SSA value names.
  iree_string_view_t value_name;
  // Complete source object type represented by the payload bytes.
  const cxx::Type* type;
  // Source ABI alignment, including an explicit stronger declaration.
  int64_t alignment;
};

// Owns immutable C++ object identity and payloads for one output module.
// Strong external definitions are retained as source-module roots; other
// objects are admitted lazily from reached expressions. All objects are keyed
// by canonical source symbol and emitted after callable translation completes.
class ImmutableObjects {
 public:
  ImmutableObjects(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                   Locations& locations, SymbolNames& names,
                   loom_module_t* module)
      : unit_(unit),
        diagnostics_(diagnostics),
        locations_(locations),
        names_(names),
        module_(module) {}

  // Admits or reuses one constant object. Local declarations are validated
  // before translating their function; only reached storage uses retain bytes.
  // Unsupported representation semantics diagnose at the owning source use.
  ImmutableObject reference(cxx::VariableSymbol* variable, cxx::AST* owner);

  // Retains a namespace-scope strong external definition even when no local
  // expression reaches it. Declarations, inline definitions, and private
  // definitions remain lazy.
  void retain_definition(cxx::VariableSymbol* variable, cxx::AST* owner);

  // Validates static declarations and retains their analyzed lexical
  // identities. Payloads are materialized only when translation reaches a
  // storage use. Inline and template function objects share source identity
  // across modules; objects belonging to internal or singly defined functions
  // stay private.
  void retain_locals(cxx::FunctionSymbol* function,
                     std::span<cxx::InitDeclaratorAST* const> declarations);

  // Emits every reached object declaration or definition into the module
  // insertion block.
  void build(loom_builder_t* builder);

 private:
  enum class EmissionKind {
    Declaration,
    PrivateDefinition,
    PublicDefinition,
    LinkOnceDefinition,
  };

  struct Declaration {
    // Defining source symbol, or null for an external declaration.
    cxx::VariableSymbol* definition;
    // Complete source type already admitted for immutable storage.
    const cxx::Type* type;
    // Source ABI alignment, including explicit declaration alignment.
    int64_t alignment;
    // Source linkage projected into the eventual emitted symbol.
    EmissionKind kind;
    // Selected name, made unique for private objects on materialization.
    std::string spelling;
    // Source declaration or use that owns admission diagnostics.
    cxx::AST* source;
  };

  struct Emission {
    // Public reference returned to function translators.
    ImmutableObject object;
    // Source linkage policy projected into the emitted symbol.
    EmissionKind kind;
    // Narrow string initializer whose complete object bytes are valid UTF-8.
    bool is_utf8;
    // Concrete selected-source-ABI representation, empty for declarations.
    std::vector<uint8_t> contents;
    // First source use responsible for retaining this object.
    cxx::AST* source;
  };

  // Validates source storage without allocating output symbols or payloads.
  Declaration declare(cxx::VariableSymbol* variable, cxx::AST* owner,
                      EmissionKind kind, std::string spelling);
  // Materializes the source ABI bytes and retains one reached source object.
  ImmutableObject materialize(cxx::VariableSymbol* variable,
                              Declaration declaration);

  // Source frontend and data model used for semantic admission and layout.
  cxx::TranslationUnit& unit_;
  // Public source rejection boundary.
  Diagnostics& diagnostics_;
  // Retains definition provenance in the output module.
  Locations& locations_;
  // Shared module namespace for functions, configs, and objects.
  SymbolNames& names_;
  // Output symbol and payload owner.
  loom_module_t* module_;
  // Canonical source identity to retained emission ordinal.
  std::unordered_map<cxx::VariableSymbol*, size_t> variables_;
  // Admitted local declarations whose bytes are retained only on demand.
  std::unordered_map<cxx::VariableSymbol*, Declaration> locals_;
  // Encounter-ordered retained objects.
  std::vector<Emission> emissions_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SYMBOL_IMMUTABLE_OBJECTS_H_
