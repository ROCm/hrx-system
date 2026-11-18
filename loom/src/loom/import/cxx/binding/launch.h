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
#include <optional>
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
                   cxx::List<cxx::AttributeSpecifierAST*>* attributes);

  // Rejects launch annotations on an ordinary function before lowering its
  // body.
  void reject_ordinary_function(cxx::FunctionSymbol* function);

  // Returns the internal source function whose body defines this kernel's
  // launch configuration, or null when annotations/config values own it.
  cxx::FunctionSymbol* configuration_function(
      cxx::FunctionSymbol* function) const;

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

  Dimensions parse(Form form, cxx::AttributeAST* attribute);
  void merge(std::optional<Dimensions>& previous,
             const std::optional<Dimensions>& next);
  cxx::FunctionSymbol* parse_configuration(cxx::AttributeAST* attribute);
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
  // Reverse ownership prevents configuration bodies from becoming callables.
  std::unordered_set<cxx::FunctionSymbol*> configurations_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_LAUNCH_H_
