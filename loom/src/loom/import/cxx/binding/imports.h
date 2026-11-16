// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_IMPORTS_H_
#define LOOM_IMPORT_CXX_BINDING_IMPORTS_H_

#include <cxx/ast_fwd.h>
#include <cxx/cxx_fwd.h>
#include <cxx/symbols_fwd.h>

#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {

// Runtime service binding retained from one declaration family.
// An absent symbol delegates the external spelling to the Loom declaration.
struct ImportBinding {
  // Nonempty runtime module or service name.
  std::string module;
  // Explicit external symbol, or no value to reuse the Loom symbol name.
  std::optional<std::string> symbol;
  // Attribute owning diagnostics and source location provenance.
  cxx::AttributeAST* source;
};

// An incomplete source payload naming an exact external object identity.
struct ObjectImport {
  // Canonical source declaration; the payload never acquires C++ layout.
  cxx::ClassSymbol* source;
  // Exact namespace and payload name, with source diagnostic provenance.
  ImportBinding binding;
};

// Owns runtime import annotations independently of callable reachability.
// Identical annotations may repeat across C++ redeclarations; conflicting
// bindings diagnose before any native declaration is emitted.
class Imports {
 public:
  Imports(cxx::TranslationUnit& unit, Diagnostics& diagnostics)
      : unit_(unit), diagnostics_(diagnostics) {}

  // Admits an import on a namespace-scope callable or incomplete payload class.
  void declaration(cxx::Symbol* symbol,
                   cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                   cxx::AST* owner);

  // Returns the runtime binding for this canonical function, when present.
  const ImportBinding* lookup(cxx::FunctionSymbol* function) const;

  // Object declarations in source order, independent of hash-table iteration.
  std::span<const ObjectImport> objects() const { return objects_; }

 private:
  // Source semantic model and string-literal interpretation.
  cxx::TranslationUnit& unit_;
  // Source admission diagnostic boundary.
  Diagnostics& diagnostics_;
  // Contracts indexed by canonical C++ function identity.
  std::unordered_map<cxx::FunctionSymbol*, ImportBinding> imports_;
  // Canonical payload identities index their first declaration.
  std::unordered_map<cxx::ClassSymbol*, size_t> object_indices_;
  // Deterministic order for module-local object declarations.
  std::vector<ObjectImport> objects_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_IMPORTS_H_
