// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_TEMPLATE_APPLY_H_
#define LOOM_IMPORT_CXX_BINDING_TEMPLATE_APPLY_H_

#include <cxx/ast_fwd.h>
#include <cxx/attributes.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <optional>
#include <span>
#include <string_view>

#include "loom/import/cxx/value/representation.h"
#include "loom/ops/op_defs.h"

namespace loom::cxx_import {

class Diagnostics;
class FunctionContracts;
class Types;

// One admitted call to a link-selected Loom template family. The C++ function
// owns overload resolution and semantic types; linking owns provider selection.
class TemplateApplyIntrinsic {
 public:
  struct Family {
    // Module-interned spelling, reserved before any declaration is needed.
    loom_string_id_t name;
    // Canonical source declaration owning this family's stable signature.
    cxx::FunctionSymbol* function = nullptr;
    // Lazily materialized declaration, absent until a reached call needs it.
    loom_op_t* declaration = nullptr;
    // Borrowed frontend-owned semantic parameter types in source order.
    std::span<const cxx::Type* const> parameter_types;
    // Flattened High argument count established during source admission.
    size_t argument_component_count = 0;
    // Frontend-owned result type, or null for a void declaration.
    const cxx::Type* result_type = nullptr;
    // Source declaration location shared by every application binding.
    loom_location_id_t declaration_location = LOOM_LOCATION_UNKNOWN;
  };

  static bool supports(std::string_view name) {
    return name == "template.apply";
  }

  // Whether the binding names its family through a C++ template argument.
  static bool is_explicit(const cxx::Attribute& attribute);

  // Resolves the concrete helper's first template argument to a source
  // function.
  static cxx::FunctionSymbol* resolve_family_argument(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics,
      cxx::FunctionSymbol* function, cxx::AST* owner);

  // Returns the family spelling after validating the operation arguments.
  static std::optional<std::string_view> admit(cxx::TranslationUnit& unit,
                                               Diagnostics& diagnostics,
                                               const cxx::Attribute& attribute,
                                               cxx::AST* owner);

  // Resolves a concrete source signature against an already interned family.
  static TemplateApplyIntrinsic resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      FunctionContracts& contracts, cxx::FunctionSymbol* function,
      Family* family, loom_location_id_t declaration_location, cxx::AST* owner);

  // Admits one concrete explicit helper against this family's source signature.
  // The returned binding borrows the same family schema without copying it.
  TemplateApplyIntrinsic bind_application(cxx::TranslationUnit& unit,
                                          Diagnostics& diagnostics,
                                          Types& types,
                                          cxx::FunctionSymbol* function,
                                          cxx::AST* owner) const;

  // Emits one family application after the caller has evaluated each operand.
  std::optional<Value> call(std::span<const Value> arguments, Types& types,
                            ValueArena& arena, cxx::AST* owner,
                            loom_builder_t* builder,
                            loom_location_id_t location) const;

  // Materializes the stable family declaration independently of applications
  // and returns its module-local symbol. Providers use this path even when the
  // current source module contains no template.apply operation.
  loom_symbol_ref_t materialize(Types& types, cxx::AST* owner,
                                loom_builder_t* builder) const;

  // Canonical source declaration owning the family's C++ signature.
  cxx::FunctionSymbol* source_function() const { return family_->function; }

  bool equivalent(const TemplateApplyIntrinsic& other) const {
    return family_ == other.family_;
  }

 private:
  TemplateApplyIntrinsic(Family* family, FunctionContracts& contracts)
      : family_(family), contracts_(&contracts) {}

  // Link-selected family retained by the owning Intrinsics registry.
  Family* family_;
  // Invocation-owned callable contracts projected onto the family declaration.
  FunctionContracts* contracts_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_TEMPLATE_APPLY_H_
