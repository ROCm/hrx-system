// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_LAUNCH_H_
#define LOOM_IMPORT_CXX_BINDING_LAUNCH_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <array>
#include <bitset>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "loom/import/cxx/source/source.h"
#include "loom/ops/op_defs.h"

namespace loom::cxx_import {

class SymbolNames;
class Types;

// Shape of the launch aggregate retained after source admission.
enum class LaunchConfigurationKind { Standard, Clustered };

// Admitted source body and native signature for one kernel-owned launch
// configuration. Translation projects the body directly into the kernel's
// config region; it never becomes an independently callable function.
struct LaunchConfiguration {
  // Definition owning source locations and the semantic function identity.
  cxx::FunctionDefinitionAST* source;
  // Compound statement translated into the kernel config region.
  cxx::CompoundStatementAST* body;
  // Aggregate source result retained for ordinary value translation.
  const cxx::Type* result_type;
  // Admitted workload source types in source parameter order. The function
  // builder binds their flattened High signature to config-region identities.
  std::vector<const cxx::Type*> workloads;
  // Selects the launch fields carried by the aggregate result.
  LaunchConfigurationKind kind;
};

// Owns source launch admission and its retained per-function contracts. Exact
// annotations become constants; bounded annotations become required config
// symbols with predicates enforced by ordinary Loom config specialization.
class LaunchContracts {
 public:
  LaunchContracts(cxx::TranslationUnit& unit, Diagnostics& diagnostics)
      : unit_(unit), diagnostics_(diagnostics) {}

  // Admits one declaration, merging consistent contracts across redeclarations.
  void declaration(cxx::FunctionSymbol* function,
                   cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                   cxx::TemplateDeclarationAST* pattern = nullptr);

  // Rejects launch annotations on an ordinary function before lowering its
  // body.
  void reject_ordinary_function(cxx::FunctionSymbol* function);

  // Returns the internal source function whose body defines this kernel's
  // launch configuration, or null when annotations/config values own it.
  cxx::FunctionSymbol* configuration_function(cxx::FunctionSymbol* function);

  // Materializes the selected configuration body for a reached kernel. A
  // specialization retains its substituted workload types and geometry.
  LaunchConfiguration bind_configuration(cxx::FunctionSymbol* selected,
                                         cxx::AST* kernel, Types& types);

  // Returns whether a source function is owned as another kernel's launch
  // configuration and therefore cannot also become an ordinary callable.
  bool is_configuration(cxx::FunctionSymbol* function) const;

  // Emits the launch terminator into the builder's current kernel config
  // region.
  void build(cxx::FunctionSymbol* function, std::string_view symbol,
             SymbolNames& names, loom_builder_t* builder,
             loom_location_id_t location);

 private:
  enum class Form { Exact, Range };

  struct Axis {
    // Inclusive lower bound, equal to upper for exact dimensions.
    int32_t lower;
    // Inclusive upper bound.
    int32_t upper;

    bool operator==(const Axis&) const = default;
  };

  struct Dimensions {
    // Whether the dimensions are fixed or require caller configuration.
    Form form;
    // Source annotation used to diagnose contradictory declarations.
    cxx::AttributeAST* source;
    // Launch extents in x, y, z order.
    std::array<Axis, 3> axes;
  };

  struct Contract {
    // Workgroup count contract; absent dimensions remain unconstrained configs.
    std::optional<Dimensions> count;
    // Workgroup size contract; absent dimensions remain unconstrained configs.
    std::optional<Dimensions> size;
    // Internal source function projected directly into the config region.
    cxx::FunctionSymbol* configuration = nullptr;
    // Kernel attribute selecting |configuration| for conflict diagnostics.
    cxx::AttributeAST* configuration_source = nullptr;
  };

  enum class Field { Configuration, Count, Size };

  struct Annotation {
    // Semantic launch field named by the source attribute.
    Field field;
    // Exact or bounded dimensions; ignored for a configuration function.
    Form form;
    // Original declaration site, including its C++ argument expressions.
    cxx::AttributeAST* source;

    // One configuration function, three exact extents, or six range bounds.
    size_t arity() const {
      if (field == Field::Configuration) {
        return 1;
      }
      return form == Form::Exact ? 3 : 6;
    }
  };

  // Attribute arity is bounded by the six values of a three-dimensional range.
  using Arguments = std::array<cxx::ExpressionAST*, 6>;
  struct DependentAnnotation {
    // Attribute whose arguments depend on this declaration's template
    // parameters.
    Annotation annotation;
    // Original template parameter scope for substituting those arguments.
    cxx::TemplateDeclarationAST* declaration;
    // Expression operands in source order, borrowed from CXX.
    Arguments arguments;
    // Positions that require C++ template substitution.
    std::bitset<6> dependent;
  };

  struct TemplateContract {
    // Facts already known without choosing template arguments.
    Contract fixed;
    // Only attributes requiring substitution for a reached specialization.
    std::vector<DependentAnnotation> dependent;
  };

  Arguments read_arguments(Annotation annotation);
  Dimensions parse(Annotation annotation,
                   std::span<cxx::ExpressionAST* const> arguments);
  cxx::FunctionSymbol* parse_configuration(cxx::AttributeAST* attribute,
                                           cxx::ExpressionAST* expression);
  void apply(Contract& contract, Annotation annotation,
             const Arguments& arguments);
  const Contract* resolve(cxx::FunctionSymbol* function);
  std::array<loom_value_id_t, 3> build_dimensions(
      const std::optional<Dimensions>& dimensions, std::string_view prefix,
      std::string_view name, SymbolNames& names, cxx::AST* source,
      loom_builder_t* builder, loom_location_id_t location);

  // Invocation-owned frontend for constant evaluation and canonical symbols.
  cxx::TranslationUnit& unit_;
  // Source admission diagnostic boundary.
  Diagnostics& diagnostics_;
  // Only annotated functions occupy the index; calls consume admitted facts.
  std::unordered_map<cxx::FunctionSymbol*, Contract> contracts_;
  // Only redeclared function templates with spelled launch geometry need an
  // entry.
  std::unordered_map<cxx::FunctionSymbol*, TemplateContract> templates_;
  // Reverse ownership prevents configuration bodies from becoming callables.
  std::unordered_set<cxx::FunctionSymbol*> configurations_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_LAUNCH_H_
