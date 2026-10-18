// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SYMBOL_FUNCTIONS_H_
#define LOOM_IMPORT_CXX_SYMBOL_FUNCTIONS_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "loom/import/cxx/binding/config.h"
#include "loom/import/cxx/binding/fastmath.h"
#include "loom/import/cxx/binding/function_contracts.h"
#include "loom/import/cxx/binding/function_imports.h"
#include "loom/import/cxx/binding/intrinsics.h"
#include "loom/import/cxx/binding/launch.h"
#include "loom/import/cxx/binding/parameter_contracts.h"
#include "loom/import/cxx/binding/target_definitions.h"
#include "loom/import/cxx/binding/template_definitions.h"
#include "loom/import/cxx/source/locations.h"
#include "loom/import/cxx/symbol/names.h"
#include "loom/import/cxx/value/signature.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

enum class FunctionKind {
  Ordinary,
  Kernel,
  TemplateDefinition,
  LaunchConfiguration,
  ClusteredLaunchConfiguration,
  CheckCase,
  CheckScenario,
};

// Native transport selected for a reached C++ implicit-object function. One
// source method may have both private instances when callers own the object in
// different representations.
enum class ImplicitObjectABI {
  // Namespace, static-member, or explicit-object function.
  None,
  // Flattened record components with functional writeback when mutable.
  Value,
  // Original pointer root and byte offset with in-place storage effects.
  Storage,
};

// One reachable source function under its native implicit-object transport.
// The symbol is the concrete definition identity, including template
// substitution; receiver ABI only specializes non-static member functions.
struct FunctionInstance {
  cxx::FunctionSymbol* symbol;
  ImplicitObjectABI receiver_abi;
};

// Admitted implicit object bound ahead of every explicit source parameter.
struct ImplicitObjectParameter {
  // Native transport selected by the call site's owned representation.
  ImplicitObjectABI abi;
  // Cv-qualified complete source object type governing member observations.
  const cxx::Type* object_type;
  // Whether a value receiver returns its possibly replaced components.
  bool mutable_access;
};

// One source lvalue-reference parameter projected through Loom's value ABI.
// Parameters remain in source order and are indexed by semantic ordinal so
// redeclarations need not share parameter symbol identities.
struct ValueReferenceParameter {
  // Ordinal in the source function parameter list.
  size_t ordinal;
  // Whether the callee returns and the caller writes back the final value.
  bool mutable_access;
};

// Admitted definition ready for recursive body construction. The source owns
// AST/types and the module owns native IR; both outlive this borrowed contract.
struct FunctionBody {
  // Definition whose source range diagnoses implicit-return admission.
  cxx::FunctionDefinitionAST* source;
  // Source statements to translate, without a second function-body lookup.
  cxx::CompoundStatementAST* body;
  // Native function, kernel, template, or check definition owning the region.
  loom_op_t* operation;
  // Entry region with projected parameter types, ready for source bindings.
  loom_region_t* region;
  // Resolved source return type, including void.
  const cxx::Type* return_type;
  // Selects body projection, return terminators, and storage admission.
  FunctionKind kind;
  // Incoming pointer contracts in source parameter order, empty when absent.
  std::span<const ParameterContract> parameter_contracts;
  // Admitted non-escaping value references in source parameter order.
  std::span<const ValueReferenceParameter> reference_parameters;
  // Implicit object transport for a reached non-static member definition.
  std::optional<ImplicitObjectParameter> implicit_object;
};

// Native definition and any source-owned launch configuration body that must
// be projected before its ordinary body.
struct DefinedFunction {
  // Function, kernel, or check body selected by the pending worklist.
  FunctionBody body;
  // Kernel configuration projected into the definition's config region.
  std::optional<FunctionBody> configuration;
};

// Owns root selection, native symbol identities and reachable function order.
// Binding admission sees every concrete declaration once during selection;
// concrete template instances are admitted when reached. The source and output
// module, along with their admission services, outlive this object.
class Functions {
 public:
  Functions(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
            loom_module_t* module, Intrinsics& intrinsics,
            LaunchContracts& launches, Configs& configs,
            FastMathContracts& fastmath, FunctionContracts& function_contracts,
            TargetDefinitions& target_definitions,
            TemplateDefinitions& template_definitions, SymbolNames& names)
      : unit_(unit),
        diagnostics_(diagnostics),
        module_(module),
        intrinsics_(intrinsics),
        launches_(launches),
        configs_(configs),
        fastmath_(fastmath),
        function_contracts_(function_contracts),
        function_imports_(unit, diagnostics),
        target_definitions_(target_definitions),
        template_definitions_(template_definitions),
        names_(names),
        parameter_contracts_(unit, diagnostics) {}

  // Selects explicit qualified roots or externally visible concrete
  // definitions. Called once before translating the pending worklist.
  void select(std::span<const iree_string_view_t> roots);
  // Retains one identity and queues a newly reached source definition or
  // bodyless namespace declaration exactly once.
  loom_symbol_ref_t declare(
      cxx::FunctionSymbol* function, cxx::AST* owner = nullptr,
      ImplicitObjectABI receiver_abi = ImplicitObjectABI::None);
  // Returns the concrete definition retained for a semantic declaration, or
  // the frontend-owned definition of a reached concrete template instance.
  cxx::FunctionSymbol* definition(cxx::FunctionSymbol* function) const;
  // Source case identity retained during declaration admission.
  bool is_check_case(cxx::FunctionSymbol* function) const;
  // Whether the declaration owns a check case or scenario record.
  bool is_check_record(cxx::FunctionSymbol* function) const;
  // Emits reachable bodyless declarations after recursive body discovery.
  void build_declarations(Types& types, Locations& locations,
                          loom_builder_t* builder);
  // Emits benchmark records referencing selected records after body projection.
  void build_benchmarks(Locations& locations, loom_builder_t* builder);
  // Discovery order grows as declare reaches helpers. Spans are invalidated by
  // growth; consumers obtain the next indexed entry after each body finishes.
  std::span<const FunctionInstance> pending() const { return pending_; }
  // Builds the selected native definition and launch contract at the caller's
  // module insertion point. The caller then enters and fills the returned body.
  DefinedFunction define(FunctionInstance instance, Types& types,
                         Locations& locations, loom_builder_t* builder);
  // Returns the retained value-reference ABI for a reached function.
  std::span<const ValueReferenceParameter> reference_parameters(
      cxx::FunctionSymbol* symbol) const;

 private:
  enum class DeclarationScope { Namespace, Nested };

  bool admit_declaration(cxx::Symbol* symbol,
                         cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                         cxx::AST* owner, DeclarationScope scope);
  void admit_symbol(cxx::Symbol* symbol,
                    cxx::List<cxx::AttributeSpecifierAST*>* attributes);
  loom_symbol_ref_t create_symbol(FunctionInstance instance, cxx::AST* source);
  loom_symbol_ref_t retain(FunctionInstance instance, cxx::AST* owner);
  void admit_reference_parameters(cxx::FunctionSymbol* symbol,
                                  cxx::AST* source);
  BoundSignature bind_callable_signature(
      FunctionInstance instance, cxx::AST* source, Types& types,
      loom_builder_t* builder, std::vector<loom_type_t>& arguments,
      std::vector<loom_type_t>& results,
      std::vector<loom_tied_result_t>& tied_results,
      std::vector<loom_predicate_t>& predicates);
  void collect(cxx::List<cxx::DeclarationAST*>* declarations,
               DeclarationScope scope,
               std::vector<cxx::FunctionSymbol*>& definitions);
  void collect(cxx::DeclarationAST* declaration, DeclarationScope scope,
               std::vector<cxx::FunctionSymbol*>& definitions);
  const std::string& qualified_name(cxx::FunctionSymbol* symbol);
  std::optional<ImplicitObjectParameter> implicit_object(
      FunctionInstance instance, cxx::AST* source) const;

  struct FunctionInstanceHash {
    size_t operator()(const FunctionInstance& instance) const {
      return std::hash<cxx::FunctionSymbol*>{}(instance.symbol) ^
             (static_cast<size_t>(instance.receiver_abi) << 1);
    }
  };
  struct FunctionInstanceEqual {
    bool operator()(const FunctionInstance& left,
                    const FunctionInstance& right) const {
      return left.symbol == right.symbol &&
             left.receiver_abi == right.receiver_abi;
    }
  };

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
  // Reconciles lexical floating-point permissions across redeclarations.
  FastMathContracts& fastmath_;
  // Callable predicates reconciled by semantic source identity.
  FunctionContracts& function_contracts_;
  // Runtime service bindings reconciled across bodyless redeclarations.
  FunctionImports function_imports_;
  // Owns source target definitions and function-like bindings to them.
  TargetDefinitions& target_definitions_;
  // Owns target-selected template family provider contracts.
  TemplateDefinitions& template_definitions_;
  // Exact callable/configuration names and generated private names.
  SymbolNames& names_;
  // Pointer entry preconditions reconciled before body construction.
  ParameterContracts parameter_contracts_;
  struct Benchmark {
    // Semantic declaration supplying the benchmark's name.
    cxx::FunctionSymbol* function;
    // Semantic case or scenario declaration named by the source attribute.
    cxx::FunctionSymbol* record_function;
    // Attribute owning diagnostics and the emitted benchmark location.
    cxx::AST* source;
  };
  // Concrete definitions indexed by canonical semantic declaration identity.
  std::unordered_map<cxx::FunctionSymbol*, cxx::FunctionSymbol*> definitions_;
  struct FunctionDeclaration {
    // Semantic declaration supplying the callable signature and parameters.
    cxx::FunctionSymbol* function;
    // Declarator owning source location and late type diagnostics.
    cxx::InitDeclaratorAST* source;
  };
  // Namespace prototypes available to satisfy reachable bodyless calls.
  std::unordered_map<cxx::FunctionSymbol*, FunctionDeclaration> declarations_;
  // Case annotations retained with their original declaration for diagnostics.
  std::unordered_map<cxx::FunctionSymbol*, cxx::AST*> check_cases_;
  struct CheckScenario {
    // Annotation owning diagnostics and source location.
    cxx::AST* source;
    // Finite compile-visible domain, absent for an unconfigured scenario.
    std::optional<int64_t> configuration_count;
  };
  // Scenario annotations retained by canonical semantic declaration identity.
  std::unordered_map<cxx::FunctionSymbol*, CheckScenario> check_scenarios_;
  // Benchmark declarations in source order, resolved during admission.
  std::vector<Benchmark> benchmarks_;
  // Selected externally visible definitions; other reached helpers stay
  // private.
  std::unordered_set<cxx::FunctionSymbol*> exported_;
  // Reached bodyless declarations whose names must remain stable for linking.
  std::unordered_set<cxx::FunctionSymbol*> external_;
  // Nonempty value-reference ABIs indexed by canonical declaration identity.
  std::unordered_map<cxx::FunctionSymbol*, std::vector<ValueReferenceParameter>>
      reference_parameters_;
  // Each reached source function receives one native identity.
  std::unordered_map<FunctionInstance, loom_symbol_ref_t, FunctionInstanceHash,
                     FunctionInstanceEqual>
      callees_;
  // Reachable worklist in deterministic discovery order.
  std::vector<FunctionInstance> pending_;
  // Reachable bodyless declarations in deterministic discovery order.
  std::vector<FunctionDeclaration> pending_declarations_;
  // Canonical declarations inherit one exact name borrowed from source text.
  std::unordered_map<cxx::FunctionSymbol*, std::string_view> explicit_names_;
  // Source qualification retained once per reached function.
  std::unordered_map<cxx::FunctionSymbol*, std::string> qualified_names_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SYMBOL_FUNCTIONS_H_
