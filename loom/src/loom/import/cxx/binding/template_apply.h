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
#include <utility>
#include <vector>

#include "loom/import/cxx/value/representation.h"
#include "loom/ops/op_defs.h"

namespace loom::cxx_import {

class Diagnostics;
class Types;

// One admitted call to a link-selected Loom template family. The C++ function
// owns overload resolution and semantic types; linking owns provider selection.
class TemplateApplyIntrinsic {
 public:
  struct Family {
    // Module-local reference to the unresolved or separately defined family.
    loom_symbol_ref_t reference;
    // Canonical source declaration owning this family's stable signature.
    cxx::FunctionSymbol* function = nullptr;
    // Lazily materialized declaration, absent until a reached call needs it.
    loom_op_t* declaration = nullptr;
  };

  static bool supports(std::string_view name) {
    return name == "template.apply";
  }

  // Returns the family spelling after validating the operation arguments.
  static std::optional<std::string_view> admit(cxx::TranslationUnit& unit,
                                               Diagnostics& diagnostics,
                                               const cxx::Attribute& attribute,
                                               cxx::AST* owner);

  // Resolves a concrete source signature against an already interned family.
  static TemplateApplyIntrinsic resolve(cxx::TranslationUnit& unit,
                                        Diagnostics& diagnostics, Types& types,
                                        cxx::FunctionSymbol* function,
                                        Family* family,
                                        loom_location_id_t declaration_location,
                                        cxx::AST* owner);

  // Emits one family application after the caller has evaluated each operand.
  std::optional<Value> call(std::span<const Value> arguments, Types& types,
                            ValueArena& arena, cxx::AST* owner,
                            loom_builder_t* builder,
                            loom_location_id_t location) const;

  bool equivalent(const TemplateApplyIntrinsic& other) const {
    return family_ == other.family_;
  }

 private:
  TemplateApplyIntrinsic(Family* family,
                         std::vector<const cxx::Type*> parameter_types,
                         const cxx::Type* result_type,
                         loom_location_id_t declaration_location)
      : family_(family),
        parameter_types_(std::move(parameter_types)),
        result_type_(result_type),
        declaration_location_(declaration_location) {}

  // Link-selected family retained by the owning Intrinsics registry.
  Family* family_;
  // Frontend-owned semantic parameter types in source order.
  std::vector<const cxx::Type*> parameter_types_;
  // Frontend-owned semantic result type, or null for a void declaration.
  const cxx::Type* result_type_;
  // Source declaration location retained for linker-facing diagnostics.
  loom_location_id_t declaration_location_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_TEMPLATE_APPLY_H_
