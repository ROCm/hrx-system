// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/template_apply.h"

#include <cxx/const_value.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <array>
#include <memory>
#include <variant>
#include <vector>

#include "loom/import/cxx/binding/function_contracts.h"
#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/symbol/names.h"
#include "loom/import/cxx/value/signature.h"
#include "loom/import/cxx/value/types.h"
#include "loom/ir/module.h"
#include "loom/ops/template/ops.h"

namespace loom::cxx_import {

bool TemplateApplyIntrinsic::is_explicit(const cxx::Attribute& attribute) {
  return attribute.arguments.size() == 1 &&
         supports(attribute.arguments[0]->name());
}

cxx::FunctionSymbol* TemplateApplyIntrinsic::resolve_family_argument(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::FunctionSymbol* function, cxx::AST* owner) {
  auto arguments = function->templateArguments();
  auto selected = arguments.empty()
                      ? std::nullopt
                      : cxx::template_argument_value(arguments[0]);
  auto* address =
      selected ? std::get_if<std::shared_ptr<cxx::ConstAddress>>(&*selected)
               : nullptr;
  auto* family =
      address && *address
          ? cxx::symbol_cast<cxx::FunctionSymbol>((*address)->symbol())
          : nullptr;
  if (!family) {
    diagnostics.reject(unit, owner,
                       "template.apply requires a family function as its first "
                       "template argument");
  }
  return family;
}

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
    FunctionContracts& contracts, cxx::FunctionSymbol* function, Family* family,
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
  if (family->function) {
    return TemplateApplyIntrinsic(family, contracts);
  }
  family->function = canonical;
  family->declaration_location = declaration_location;
  family->parameter_types = signature->parameterTypes();
  for (const auto* parameter : family->parameter_types) {
    family->argument_component_count +=
        types.partition(parameter, owner).component_count;
  }
  const auto* result = types.unqualified(signature->returnType());
  if (result->kind() != cxx::TypeKind::kVoid) {
    types.partition(result, owner);
    family->result_type = result;
  }
  return TemplateApplyIntrinsic(family, contracts);
}

TemplateApplyIntrinsic TemplateApplyIntrinsic::bind_application(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, cxx::AST* owner) const {
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  const auto& parameters = signature->parameterTypes();
  if (signature->isVariadic() ||
      parameters.size() != family_->parameter_types.size()) {
    diagnostics.reject(unit, owner,
                       "template.apply arguments must match the family arity");
  }
  const auto* result = types.unqualified(signature->returnType());
  if (result->kind() == cxx::TypeKind::kVoid) {
    result = nullptr;
  }
  if (result != family_->result_type) {
    diagnostics.reject(unit, owner,
                       "template.apply result type must match its family");
  }
  for (size_t i = 0; i < parameters.size(); ++i) {
    if (types.unqualified(parameters[i]) !=
        types.unqualified(family_->parameter_types[i])) {
      diagnostics.reject(unit, owner,
                         "template.apply argument type must match its family");
    }
  }
  return *this;
}

loom_symbol_ref_t TemplateApplyIntrinsic::materialize(
    Types& types, cxx::AST* owner, loom_builder_t* builder) const {
  if (!family_->declaration) {
    loom_builder_t declaration_builder;
    loom_builder_initialize(builder->module, &builder->module->arena,
                            loom_module_block(builder->module),
                            &declaration_builder);
    std::vector<const cxx::Type*> sources(family_->parameter_types.begin(),
                                          family_->parameter_types.end());
    if (family_->result_type) {
      sources.push_back(family_->result_type);
    }
    const bool has_predicates = contracts_->has_predicates(family_->function);
    auto signature = bind_signature(
        types, sources, owner, &declaration_builder,
        has_predicates ? SignatureIdentityRequirement::Required
                       : SignatureIdentityRequirement::DependentTypes);
    const size_t argument_count = family_->argument_component_count;
    const loom_type_t* argument_types =
        argument_count ? signature.types.data() : nullptr;
    const size_t result_count = signature.types.size() - argument_count;
    const loom_type_t* result_types =
        result_count ? signature.types.data() + argument_count : nullptr;
    auto build_flags = annotated(family_->function, "device")
                           ? LOOM_TEMPLATE_DECL_BUILD_FLAG_HAS_CC
                           : 0;
    auto predicates =
        contracts_->bind(family_->function, types, signature.identities,
                         FunctionContractSignature::Flattened, owner);
    auto requirements =
        contracts_->bind_requirements(family_->function, builder->module);
    if (!predicates.empty()) {
      build_flags |= LOOM_TEMPLATE_DECL_BUILD_FLAG_HAS_PREDICATES;
    }
    if (!requirements.empty()) {
      build_flags |= LOOM_TEMPLATE_DECL_BUILD_FLAG_HAS_REQUIRES;
    }
    loom_symbol_ref_t reference = {};
    check(loom_module_add_symbol(builder->module, family_->name,
                                 &reference.symbol_id));
    check(loom_template_decl_build(
        &declaration_builder, build_flags, /*visibility=*/0,
        /*retain=*/0,
        annotated(family_->function, "device") ? LOOM_TEMPLATE_CC_DEVICE : 0,
        /*effect_contract=*/0, /*temperature=*/0, loom_symbol_ref_null(),
        loom_make_parameterized_attr_array(requirements.data(),
                                           requirements.size()),
        reference, argument_types, argument_count, result_types, result_count,
        /*tied_results=*/nullptr,
        /*tied_result_count=*/0, predicates.data(), predicates.size(),
        family_->declaration_location, &family_->declaration));
  }
  return loom_template_decl_family(family_->declaration);
}

std::optional<Value> TemplateApplyIntrinsic::call(
    std::span<const Value> arguments, Types& types, ValueArena& arena,
    cxx::AST* owner, loom_builder_t* builder,
    loom_location_id_t location) const {
  auto family = materialize(types, owner, builder);

  std::vector<loom_value_id_t> operands;
  operands.reserve(family_->argument_component_count);
  for (const auto argument : arguments) {
    argument.append_to(operands);
  }

  BoundSignature results;
  if (family_->result_type) {
    std::array<const cxx::Type*, 1> sources = {family_->result_type};
    results = bind_signature(types, sources, owner, builder);
  }
  loom_op_t* op;
  check(loom_template_apply_build(
      builder, /*build_flags=*/0, family, operands.data(), operands.size(),
      /*effect_contract=*/0, /*temperature=*/0, results.types.data(),
      results.types.size(), /*tied_results=*/nullptr, /*tied_result_count=*/0,
      location, &op));
  if (!family_->result_type) {
    return std::nullopt;
  }
  return arena.capture(types.partition(family_->result_type, owner),
                       {loom_op_results(op), results.types.size()});
}

}  // namespace loom::cxx_import
