// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_TARGET_DEFINITIONS_H_
#define LOOM_IMPORT_CXX_BINDING_TARGET_DEFINITIONS_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "loom/import/cxx/source/locations.h"
#include "loom/import/cxx/symbol/names.h"
#include "loom/ir/ir.h"

namespace loom::cxx_import {

// Owns constexpr target definitions and function-like target bindings for one
// source translation unit. Facade types select a registered TargetLike op; its
// generated field descriptors remain the only target schema consumed by this
// target-neutral importer.
class TargetDefinitions {
 public:
  TargetDefinitions(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                    Locations& locations, SymbolNames& symbol_names,
                    loom_module_t* module)
      : unit_(unit),
        diagnostics_(diagnostics),
        locations_(locations),
        symbol_names_(symbol_names),
        module_(module) {}

  // Admits a marked namespace-scope constexpr object or a target binding during
  // the shared declaration walk. Returns whether |symbol| is a target
  // definition whose global storage is replaced by module metadata.
  bool declaration(cxx::Symbol* symbol,
                   cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                   cxx::AST* owner);

  // Emits every admitted target definition in source order and resolves
  // function bindings. Called once after declaration collection and before
  // functions.
  void build(loom_builder_t* builder);

  // Returns the retained target symbol for a function-like definition or
  // declaration, or null when the source has no target binding. All bindings
  // have been validated by build().
  loom_symbol_ref_t reference(cxx::FunctionSymbol* function) const;

  // Rejects a target binding on a function consumed as kernel launch
  // configuration instead of emitted as a callable definition.
  void reject_launch_configuration_function(
      cxx::FunctionSymbol* function) const;

 private:
  struct Definition {
    // Canonical source object whose constexpr value supplies target fields.
    cxx::VariableSymbol* variable;
    // Declarator owning source diagnostics and the emitted target location.
    cxx::AST* owner;
    // Registered target operation selected by the facade type marker.
    loom_op_kind_t operation_kind;
    // TargetLike operation schema valid for the context lifetime.
    const loom_op_vtable_t* vtable;
    // Module spelling derived from the qualified source object name.
    std::string name;
    // Module-local identity assigned during build().
    loom_symbol_ref_t reference;
  };

  struct FunctionBinding {
    // Canonical target object named by the function attribute.
    cxx::VariableSymbol* variable;
    // Attribute owning conflict or invalid-target diagnostics.
    cxx::AST* owner;
  };

  void admit_function(cxx::FunctionSymbol* function,
                      cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                      cxx::AST* owner);
  bool admit_definition(cxx::VariableSymbol* variable, cxx::AST* owner);
  void emit_definition(Definition& definition, loom_builder_t* builder);
  std::string qualified_name(cxx::VariableSymbol* variable) const;

  // Invocation-owned frontend and source diagnostic boundary.
  cxx::TranslationUnit& unit_;
  Diagnostics& diagnostics_;
  // Copies source provenance into emitted target definitions.
  Locations& locations_;
  // Functions, configs, and targets share one exact output namespace.
  SymbolNames& symbol_names_;
  // Owns registered metadata, target symbols, attributes, and operations.
  loom_module_t* module_;
  // Target definitions in deterministic source declaration order.
  std::vector<Definition> definitions_;
  // Canonical source target identity to |definitions_| index.
  std::unordered_map<cxx::VariableSymbol*, size_t> variables_;
  // Function-like declarations with an explicit target contract.
  std::unordered_map<cxx::FunctionSymbol*, FunctionBinding> functions_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_TARGET_DEFINITIONS_H_
