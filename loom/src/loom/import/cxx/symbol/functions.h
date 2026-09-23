// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SYMBOL_FUNCTIONS_H_
#define LOOM_IMPORT_CXX_SYMBOL_FUNCTIONS_H_

#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "loom/import/cxx/binding/config.h"
#include "loom/import/cxx/binding/intrinsics.h"
#include "loom/import/cxx/binding/launch.h"
#include "loom/import/cxx/binding/parameter_contracts.h"
#include "loom/import/cxx/source/locations.h"
#include "loom/import/cxx/symbol/names.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

enum class FunctionKind { Ordinary, Kernel, CheckCase };

// Admitted definition ready for recursive body construction. The source owns
// AST/types and the module owns native IR; both outlive this borrowed contract.
struct FunctionBody {
  // Definition whose source range diagnoses implicit-return admission.
  cxx::FunctionDefinitionAST* source;
  // Source statements to translate, without a second function-body lookup.
  cxx::CompoundStatementAST* body;
  // Native function, kernel, or check definition owning the body region.
  loom_op_t* operation;
  // Entry region with projected parameter types, ready for source bindings.
  loom_region_t* region;
  // Resolved source return type, including void.
  const cxx::Type* return_type;
  // Selects body projection, return terminators, and storage admission.
  FunctionKind kind;
  // Incoming pointer contracts in source parameter order, empty when absent.
  std::span<const ParameterContract> parameter_contracts;
};

// Owns root selection, native symbol identities and reachable function order.
// Binding admission sees every concrete declaration once during selection;
// concrete template instances are admitted when reached. The source and output
// module, along with their admission services, outlive this object.
class Functions {
 public:
  Functions(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
            loom_module_t* module, Intrinsics& intrinsics,
            LaunchContracts& launches, Configs& configs, SymbolNames& names)
      : unit_(unit),
        diagnostics_(diagnostics),
        module_(module),
        intrinsics_(intrinsics),
        launches_(launches),
        configs_(configs),
        names_(names),
        parameter_contracts_(unit, diagnostics) {}

  // Selects explicit qualified roots or externally visible concrete
  // definitions. Called once before translating the pending worklist.
  void select(std::span<const iree_string_view_t> roots);
  // Retains one identity and queues a newly reached source function exactly
  // once. The caller has admitted a concrete definition through selection or
  // definition().
  loom_symbol_ref_t declare(cxx::FunctionSymbol* function);
  // Returns the concrete definition retained for a semantic declaration, or
  // the frontend-owned definition of a reached concrete template instance.
  cxx::FunctionSymbol* definition(cxx::FunctionSymbol* function) const;
  // Source case identity retained during declaration admission.
  bool is_check_case(cxx::FunctionSymbol* function) const;
  // Emits benchmark records referencing selected cases after body projection.
  void build_benchmarks(Locations& locations, loom_builder_t* builder);
  // Discovery order grows as declare reaches helpers. Spans are invalidated by
  // growth; consumers obtain the next indexed entry after each body finishes.
  std::span<cxx::FunctionSymbol* const> pending() const { return pending_; }
  // Builds the selected native definition and launch contract at the caller's
  // module insertion point. The caller then enters and fills the returned body.
  FunctionBody define(cxx::FunctionSymbol* symbol, Types& types,
                      Locations& locations, loom_builder_t* builder);

 private:
  enum class DeclarationScope { Namespace, Nested };

  bool admit_declaration(cxx::Symbol* symbol,
                         cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                         cxx::AST* owner, DeclarationScope scope);
  void admit_symbol(cxx::Symbol* symbol,
                    cxx::List<cxx::AttributeSpecifierAST*>* attributes);
  loom_symbol_ref_t create_symbol(cxx::FunctionSymbol* function,
                                  cxx::AST* source);
  void collect(cxx::List<cxx::DeclarationAST*>* declarations,
               DeclarationScope scope,
               std::vector<cxx::FunctionSymbol*>& definitions);
  void collect(cxx::DeclarationAST* declaration, DeclarationScope scope,
               std::vector<cxx::FunctionSymbol*>& definitions);
  const std::string& qualified_name(cxx::FunctionSymbol* symbol);

  // Source declarations, semantic identities and namespace relationships.
  cxx::TranslationUnit& unit_;
  // Root/definition admission diagnostic boundary.
  Diagnostics& diagnostics_;
  // Owns output strings, symbols and definitions.
  loom_module_t* module_;
  // Retains generated operation bindings admitted while visiting declarations.
  Intrinsics& intrinsics_;
  // Retains merged launch contracts for definitions and concrete instances.
  LaunchContracts& launches_;
  // Reconciles named scalar settings before root selection and body lowering.
  Configs& configs_;
  // Exact callable/configuration names and generated private names.
  SymbolNames& names_;
  // Pointer entry preconditions reconciled before body construction.
  ParameterContracts parameter_contracts_;
  struct Benchmark {
    // Semantic declaration supplying the benchmark's name.
    cxx::FunctionSymbol* function;
    // Semantic case declaration named by the source attribute.
    cxx::FunctionSymbol* case_function;
    // Attribute owning diagnostics and the emitted benchmark location.
    cxx::AST* source;
  };
  // Concrete definitions indexed by canonical semantic declaration identity.
  std::unordered_map<cxx::FunctionSymbol*, cxx::FunctionSymbol*> definitions_;
  // Case annotations retained with their original declaration for diagnostics.
  std::unordered_map<cxx::FunctionSymbol*, cxx::AST*> check_cases_;
  // Benchmark declarations in source order, resolved during admission.
  std::vector<Benchmark> benchmarks_;
  // Selected externally visible definitions; other reached helpers stay
  // private.
  std::unordered_set<cxx::FunctionSymbol*> exported_;
  // Each reached source function receives one native identity.
  std::unordered_map<cxx::FunctionSymbol*, loom_symbol_ref_t> callees_;
  // Reachable worklist in deterministic discovery order.
  std::vector<cxx::FunctionSymbol*> pending_;
  // Canonical declarations inherit one exact name borrowed from source text.
  std::unordered_map<cxx::FunctionSymbol*, std::string_view> explicit_names_;
  // Source qualification retained once per reached function.
  std::unordered_map<cxx::FunctionSymbol*, std::string> qualified_names_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SYMBOL_FUNCTIONS_H_
