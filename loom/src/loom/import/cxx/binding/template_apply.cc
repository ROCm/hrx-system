// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/template_apply.h"

#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <array>
#include <utility>
#include <vector>

#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/symbol/names.h"
#include "loom/import/cxx/value/signature.h"
#include "loom/import/cxx/value/types.h"
#include "loom/ir/module.h"
#include "loom/ops/template/ops.h"

namespace loom::cxx_import {

std::optional<std::string_view> TemplateApplyIntrinsic::admit(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    const cxx::Attribute& attribute, cxx::AST* owner) {
  if (!supports(attribute.arguments[0]->name())) {
    return std::nullopt;
  }
  if (attribute.arguments.size() != 2 ||
      !is_symbol_name(attribute.arguments[1]->name())) {
    diagnostics.reject(unit, owner,
                       "template.apply requires one valid family symbol");
  }
  return attribute.arguments[1]->name();
}

TemplateApplyIntrinsic TemplateApplyIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, Family* family,
    loom_location_id_t declaration_location, cxx::AST* owner) {
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  if (signature->isVariadic()) {
    diagnostics.reject(unit, owner,
                       "template.apply declarations cannot be variadic");
  }
  auto* canonical = function->canonical();
  if (family->function && family->function != canonical) {
    diagnostics.reject(unit, owner,
                       "template family cannot name multiple C++ functions");
  }
  family->function = canonical;
  std::vector<const cxx::Type*> parameters;
  parameters.reserve(signature->parameterTypes().size());
  for (const auto* parameter : signature->parameterTypes()) {
    types.partition(parameter, owner);
    parameters.push_back(parameter);
  }
  const auto* result = types.unqualified(signature->returnType());
  if (result->kind() == cxx::TypeKind::kVoid) {
    return TemplateApplyIntrinsic(family, std::move(parameters), nullptr,
                                  declaration_location);
  }
  types.partition(result, owner);
  return TemplateApplyIntrinsic(family, std::move(parameters), result,
                                declaration_location);
}

std::optional<Value> TemplateApplyIntrinsic::call(
    std::span<const Value> arguments, Types& types, ValueArena& arena,
    cxx::AST* owner, loom_builder_t* builder,
    loom_location_id_t location) const {
  if (!family_->declaration) {
    loom_builder_t declaration_builder;
    loom_builder_initialize(builder->module, &builder->module->arena,
                            loom_module_block(builder->module),
                            &declaration_builder);
    std::vector<const cxx::Type*> sources(parameter_types_.begin(),
                                          parameter_types_.end());
    if (result_type_) {
      sources.push_back(result_type_);
    }
    auto signature =
        bind_signature(types, sources, owner, &declaration_builder);
    size_t argument_count = 0;
    for (const auto* parameter : parameter_types_) {
      argument_count += types.partition(parameter, owner).component_count;
    }
    const loom_type_t* argument_types =
        argument_count ? signature.types.data() : nullptr;
    const size_t result_count = signature.types.size() - argument_count;
    const loom_type_t* result_types =
        result_count ? signature.types.data() + argument_count : nullptr;
    check(loom_template_decl_build(
        &declaration_builder, /*build_flags=*/0, /*visibility=*/0,
        /*retain=*/0, /*cc=*/0, /*purity=*/0, /*temperature=*/0,
        loom_symbol_ref_null(), loom_parameterized_attr_array_empty(),
        family_->reference, argument_types, argument_count, result_types,
        result_count, /*tied_results=*/nullptr,
        /*tied_result_count=*/0, /*predicates=*/nullptr,
        /*predicates_count=*/0, declaration_location_, &family_->declaration));
  }

  std::vector<loom_value_id_t> operands;
  operands.reserve(parameter_types_.size());
  for (const auto argument : arguments) {
    argument.append_to(operands);
  }

  BoundSignature results;
  if (result_type_) {
    std::array<const cxx::Type*, 1> sources = {result_type_};
    results = bind_signature(types, sources, owner, builder);
  }
  loom_op_t* op;
  check(loom_template_apply_build(
      builder, /*build_flags=*/0, family_->reference, operands.data(),
      operands.size(), /*purity=*/0, /*temperature=*/0, results.types.data(),
      results.types.size(), /*tied_results=*/nullptr, /*tied_result_count=*/0,
      location, &op));
  if (!result_type_) {
    return std::nullopt;
  }
  return arena.capture(types.partition(result_type_, owner),
                       {loom_op_results(op), results.types.size()});
}

}  // namespace loom::cxx_import
