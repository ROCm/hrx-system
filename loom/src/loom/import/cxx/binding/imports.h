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
  // Imported payload supplied as a borrowed implicit object, or null for an
  // ordinary namespace-scope function.
  cxx::ClassSymbol* receiver_payload = nullptr;
};

// Original semantic handle selected by one compile-time protocol
// arrow. The protocol pointer itself never acquires a native value.
struct ObjectMethodReceiver {
  // Source handle expression evaluated exactly once by the ordinary caller.
  cxx::ExpressionAST* handle;
  // Canonical imported payload shared by the handle and method declaration.
  cxx::ClassSymbol* payload;
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

  // Admits an import on a namespace-scope callable, a protocol
  // method, or an incomplete payload class.
  void declaration(cxx::Symbol* symbol,
                   cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                   cxx::AST* owner);

  // Returns the runtime binding for this canonical function, when present.
  const ImportBinding* lookup(cxx::FunctionSymbol* function) const;

  // Returns the import when the function is a method on one
  // protocol<T> specialization, or null for an ordinary import.
  const ImportBinding* object_method(cxx::FunctionSymbol* function) const;

  // Recognizes the frontend-inserted terminal operator-> call for an imported
  // object method and returns its original borrowed handle. Non-method calls
  // return no value; malformed method receivers diagnose.
  std::optional<ObjectMethodReceiver> object_method_receiver(
      cxx::FunctionSymbol* function, cxx::CallExpressionAST* call) const;

  // Whether this bodyless operator-> is the compile-time adapter from a
  // borrowed object handle to its matching protocol.
  bool is_protocol_operator(cxx::FunctionSymbol* function) const;

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
