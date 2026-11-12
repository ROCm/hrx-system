// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_TEMPLATE_DEFINITIONS_H_
#define LOOM_IMPORT_CXX_BINDING_TEMPLATE_DEFINITIONS_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>

#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace loom::cxx_import {

class Diagnostics;
class Types;

// Owns source declarations that provide bodyful implementations of a linked
// Loom template family. Providers remain module definitions independently of
// source roots and ordinary call reachability.
class TemplateDefinitions {
 public:
  struct Provider {
    // Canonical concrete source function supplying the implementation body.
    cxx::FunctionSymbol* function;
    // Canonical LOOM_TEMPLATE_DECL declaration owning the family signature.
    cxx::FunctionSymbol* family = nullptr;
    // First provider annotation, or the priority annotation before a family.
    cxx::AST* source;
    // Source annotation naming the family.
    cxx::AST* family_source = nullptr;
    // Optional linker selection priority.
    std::optional<int64_t> priority;
    // Source annotation owning the priority value.
    cxx::AST* priority_source = nullptr;
  };

  TemplateDefinitions(cxx::TranslationUnit& unit, Diagnostics& diagnostics)
      : unit_(unit), diagnostics_(diagnostics) {}

  // Reconciles provider family and priority annotations across declarations.
  void declaration(cxx::FunctionSymbol* function,
                   cxx::List<cxx::AttributeSpecifierAST*>* attributes);

  // Returns the provider contract for a source function, or null when ordinary.
  const Provider* lookup(cxx::FunctionSymbol* function) const;

  // Providers in deterministic first-declaration order.
  std::span<const Provider> providers() const { return providers_; }

  // Validates the completed provider classification, template input carriers,
  // and exact result signature. Called after the declaration walk has
  // reconciled all attributes.
  void validate(const Provider& provider, cxx::FunctionSymbol* family,
                Types& types) const;

 private:
  Provider& get_or_create(cxx::FunctionSymbol* function, cxx::AST* source);

  // Invocation-owned frontend and diagnostic boundary.
  cxx::TranslationUnit& unit_;
  Diagnostics& diagnostics_;
  // Source-order provider contracts retained at stable vector indices.
  std::vector<Provider> providers_;
  // Canonical implementation identity to |providers_| index.
  std::unordered_map<cxx::FunctionSymbol*, size_t> provider_indices_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_TEMPLATE_DEFINITIONS_H_
