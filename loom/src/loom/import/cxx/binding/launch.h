// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_LAUNCH_H_
#define LOOM_IMPORT_CXX_BINDING_LAUNCH_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>

#include <array>
#include <optional>
#include <string_view>
#include <unordered_map>

#include "loom/import/cxx/source/source.h"
#include "loom/ops/op_defs.h"

namespace loom::cxx_import {

class SymbolNames;

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
  };

  Dimensions parse(Form form, cxx::AttributeAST* attribute);
  void merge(std::optional<Dimensions>& previous,
             const std::optional<Dimensions>& next);
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
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_LAUNCH_H_
