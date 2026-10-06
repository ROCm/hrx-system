// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/intrinsics.h"

#include <cxx/ast.h>
#include <cxx/attributes.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <array>
#include <type_traits>
#include <utility>
#include <vector>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/locations.h"
#include "loom/import/cxx/symbol/names.h"
#include "loom/ir/module.h"

namespace loom::cxx_import {
namespace {

const cxx::Attribute* operation_attribute(const cxx::AttributeMap* attributes) {
  if (!attributes) {
    return nullptr;
  }
  for (const auto& attribute : *attributes) {
    if (attribute.attributeNamespace && attribute.name &&
        attribute.attributeNamespace->name() == "loom" &&
        attribute.name->name() == "op") {
      return &attribute;
    }
  }
  return nullptr;
}

std::span<const loom_value_id_t> flatten(
    std::span<const Value> arguments,
    std::array<loom_value_id_t, 8>& inline_values,
    std::vector<loom_value_id_t>& overflow) {
  size_t count = 0;
  for (auto argument : arguments) {
    count += argument.partition().component_count;
  }
  if (count <= inline_values.size()) {
    size_t index = 0;
    for (auto argument : arguments) {
      for (auto component : argument.components()) {
        inline_values[index++] = component;
      }
    }
    return {inline_values.data(), count};
  }
  overflow.reserve(count);
  for (auto argument : arguments) {
    argument.append_to(overflow);
  }
  return overflow;
}

}  // namespace

void Intrinsics::declaration(cxx::FunctionSymbol* function,
                             cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                             cxx::AST* owner) {
  bool found = false;
  visit_loom_attributes(
      unit_, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name != "op") {
          return;
        }
        if (found || function->declaration()) {
          diagnostics_.reject(unit_, owner,
                              "an intrinsic requires one operation binding and "
                              "no function body");
        }
        found = true;
        auto* clause = attribute->attributeArgumentClause;
        if (!clause || !clause->expressionList) {
          diagnostics_.reject(unit_, owner,
                              "loom::op requires a string operation name");
        }
        for (auto* argument : cxx::ListView{clause->expressionList}) {
          if (!cxx::ast_cast<cxx::StringLiteralExpressionAST>(argument)) {
            diagnostics_.reject(
                unit_, argument,
                "operation bindings accept only string arguments");
          }
        }
      });
  if (!found) {
    if (operation_attribute(function->attributes())) {
      diagnostics_.reject(unit_, owner,
                          "operation bindings require leading attributes on a "
                          "plain function declaration");
    }
    return;
  }

  // Specifiers retain their own resolved arguments. A symbol's merged map
  // cannot distinguish conflicting source redeclarations.
  const cxx::Attribute* selected = nullptr;
  for (auto* specifier : cxx::ListView{attributes}) {
    if (auto* attribute = operation_attribute(specifier->attributes)) {
      selected = attribute;
      break;
    }
  }
  if (function->isTemplatePattern()) {
    TemplateBinding binding{selected, {}};
    if (auto* scalar = loom_cxx_scalar_binding_find(
            view(selected->arguments[0]->name()))) {
      binding.operation = resolve_scalar_operation(scalar, *selected, owner);
    } else if (auto shaped = ShapedIntrinsic::admit(unit_, diagnostics_,
                                                    *selected, owner)) {
      binding.operation = *shaped;
    } else if (auto encoding = EncodingIntrinsic::admit(
                   unit_, diagnostics_, module_->context, *selected, owner)) {
      binding.operation = *encoding;
    } else if (TemplateApplyIntrinsic::admit(unit_, diagnostics_, *selected,
                                             owner)) {
      diagnostics_.reject(unit_, owner,
                          "template.apply requires an ordinary function");
    } else if (selected->arguments.size() != 1 ||
               (!ViewIntrinsic::supports(selected->arguments[0]->name()) &&
                !DecodeIntrinsic::supports(selected->arguments[0]->name()) &&
                !AtomicIntrinsic::supports(selected->arguments[0]->name()) &&
                !FenceIntrinsic::supports(selected->arguments[0]->name()) &&
                !SubgroupIntrinsic::supports(selected->arguments[0]->name()) &&
                !BarrierIntrinsic::supports(selected->arguments[0]->name()) &&
                !CheckIntrinsic::parse_operation(
                    selected->arguments[0]->name()) &&
                !AssemblyIntrinsic::supports(selected->arguments[0]->name()) &&
                !TemplateApplyIntrinsic::supports(
                    selected->arguments[0]->name()))) {
      diagnostics_.reject(unit_, owner,
                          "function template operation has no C++ projection");
    }
    auto [entry, inserted] = template_bindings_.try_emplace(
        function->canonical(), std::move(binding));
    if (!inserted && *entry->second.attribute != *selected) {
      diagnostics_.reject(unit_, owner,
                          "conflicting intrinsic template redeclarations");
    }
    return;
  }
  auto binding = resolve(function, *selected, owner);
  auto [entry, inserted] =
      bindings_.try_emplace(function->canonical(), binding);
  bool equivalent =
      inserted || std::visit(
                      [&](const auto& previous) {
                        using T = std::decay_t<decltype(previous)>;
                        auto* current = std::get_if<T>(&binding);
                        return current && previous.equivalent(*current);
                      },
                      entry->second);
  if (!equivalent) {
    diagnostics_.reject(unit_, owner, "conflicting intrinsic redeclarations");
  }
}

Intrinsics::Binding Intrinsics::resolve(cxx::FunctionSymbol* function,
                                        const cxx::Attribute& attribute,
                                        cxx::AST* owner) {
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  if (auto family = TemplateApplyIntrinsic::admit(unit_, diagnostics_,
                                                  attribute, owner)) {
    return TemplateApplyIntrinsic::resolve(
        unit_, diagnostics_, types_, function, template_family(*family, owner),
        locations_.get(owner), owner);
  }
  if (auto decode = DecodeIntrinsic::resolve(
          unit_, diagnostics_, types_, signature, attribute, module_, owner)) {
    return *decode;
  }
  if (auto encoding = EncodingIntrinsic::admit(
          unit_, diagnostics_, module_->context, attribute, owner)) {
    return EncodingIntrinsic::resolve(*encoding, unit_, diagnostics_, types_,
                                      function, module_, owner);
  }
  if (auto binding = CheckIntrinsic::resolve(unit_, diagnostics_, types_,
                                             function, attribute, owner)) {
    return *binding;
  }
  if (auto* scalar =
          loom_cxx_scalar_binding_find(view(attribute.arguments[0]->name()))) {
    return resolve_scalar(resolve_scalar_operation(scalar, attribute, owner),
                          signature, owner);
  }
  if (auto shaped =
          ShapedIntrinsic::admit(unit_, diagnostics_, attribute, owner)) {
    return ShapedIntrinsic::resolve(*shaped, unit_, diagnostics_, types_,
                                    signature, owner);
  }
  if (auto assembly = AssemblyIntrinsic::resolve(unit_, diagnostics_, types_,
                                                 function, attribute, owner)) {
    return *assembly;
  }
  if (auto atomic = AtomicIntrinsic::resolve(unit_, diagnostics_, types_,
                                             function, attribute, owner)) {
    return *atomic;
  }
  if (auto fence = FenceIntrinsic::resolve(unit_, diagnostics_, function,
                                           attribute, owner)) {
    return *fence;
  }
  if (auto view = ViewIntrinsic::resolve(unit_, diagnostics_, types_, signature,
                                         attribute, owner)) {
    return *view;
  }
  if (auto subgroup = SubgroupIntrinsic::resolve(unit_, diagnostics_, types_,
                                                 function, attribute, owner)) {
    return *subgroup;
  }
  if (auto barrier = BarrierIntrinsic::resolve(unit_, diagnostics_, function,
                                               attribute, owner)) {
    return *barrier;
  }
  diagnostics_.reject(unit_, owner, "operation has no C++ projection");
}

Intrinsics::ScalarOperation Intrinsics::resolve_scalar_operation(
    const loom_cxx_scalar_binding_t* scalar, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  ScalarOperation result{scalar};
  for (size_t index = 1; index < attribute.arguments.size(); ++index) {
    uint8_t flag;
    if (!scalar->has_fastmath ||
        !loom_cxx_scalar_flag_parse(view(attribute.arguments[index]->name()),
                                    &flag)) {
      diagnostics_.reject(unit_, owner,
                          "intrinsic has an unsupported fast-math flag");
    }
    result.flags |= flag;
  }
  return result;
}

Intrinsics::ScalarBinding Intrinsics::resolve_scalar(
    ScalarOperation operation, const cxx::FunctionType* signature,
    cxx::AST* owner) {
  const auto& traits = unit_.typeTraits();
  const auto* return_type = traits.remove_cv(signature->returnType());
  if (operation.scalar->category == LOOM_CXX_SCALAR_CATEGORY_INTEGER) {
    if (!traits.is_integral(return_type) ||
        return_type->kind() == cxx::TypeKind::kBool) {
      diagnostics_.reject(
          unit_, owner,
          "scalar intrinsic result must be a non-boolean integer");
    }
  } else if (!types_.is_float(return_type)) {
    diagnostics_.reject(unit_, owner,
                        "scalar intrinsic result must be _Float16, __bf16, "
                        "float, or double");
  }
  if (signature->isVariadic() ||
      signature->parameterTypes().size() != operation.scalar->operand_count) {
    diagnostics_.reject(unit_, owner,
                        "intrinsic declaration has the wrong operand count");
  }
  for (const auto* parameter : signature->parameterTypes()) {
    if (traits.remove_cv(parameter) != return_type) {
      diagnostics_.reject(unit_, owner,
                          "intrinsic operands must have the result's type");
    }
  }
  return {operation, types_.get(return_type, owner)};
}

TemplateApplyIntrinsic::Family* Intrinsics::template_family(
    std::string_view spelling, cxx::AST* owner) {
  if (auto found = template_families_.find(std::string(spelling));
      found != template_families_.end()) {
    return &found->second;
  }
  names_.reserve(spelling, owner);
  loom_string_id_t name;
  check(loom_module_intern_string(module_, view(spelling), &name));
  TemplateApplyIntrinsic::Family family = {};
  check(loom_module_add_symbol(module_, name, &family.reference.symbol_id));
  auto [entry, inserted] =
      template_families_.emplace(std::string(spelling), family);
  (void)inserted;
  return &entry->second;
}

Intrinsics::Binding* Intrinsics::concrete_binding(cxx::FunctionSymbol* function,
                                                  cxx::AST* owner) {
  auto entry = bindings_.find(function->canonical());
  if (entry != bindings_.end()) {
    return &entry->second;
  }
  if (!function->isSpecialization()) {
    return nullptr;
  }
  auto* primary =
      cxx::symbol_cast<cxx::FunctionSymbol>(function->primaryTemplateSymbol());
  auto pattern = primary ? template_bindings_.find(primary->canonical())
                         : template_bindings_.end();
  if (pattern == template_bindings_.end()) {
    return nullptr;
  }
  auto* attribute = operation_attribute(function->attributes());
  if (!attribute || *attribute != *pattern->second.attribute) {
    diagnostics_.reject(
        unit_, owner,
        "intrinsic specialization does not preserve its template binding");
  }
  Binding binding = std::visit(
      [&](auto operation) -> Binding {
        using T = decltype(operation);
        auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
        if constexpr (std::is_same_v<T, ScalarOperation>) {
          return resolve_scalar(operation, signature, owner);
        } else if constexpr (std::is_same_v<T, ShapedIntrinsic::Operation>) {
          return ShapedIntrinsic::resolve(operation, unit_, diagnostics_,
                                          types_, signature, owner);
        } else if constexpr (std::is_same_v<T, EncodingIntrinsic::Family>) {
          return EncodingIntrinsic::resolve(operation, unit_, diagnostics_,
                                            types_, function, module_, owner);
        } else {
          return resolve(function, *attribute, owner);
        }
      },
      pattern->second.operation);
  auto inserted =
      bindings_.try_emplace(function->canonical(), std::move(binding));
  return &inserted.first->second;
}

Intrinsics::Binding* Intrinsics::lookup(cxx::FunctionSymbol* function,
                                        cxx::AST* owner) {
  return concrete_binding(function, owner);
}

const CheckIntrinsic* Intrinsics::check_binding(cxx::FunctionSymbol* function,
                                                cxx::AST* owner) {
  auto* binding = lookup(function, owner);
  return binding ? std::get_if<CheckIntrinsic>(binding) : nullptr;
}

IntrinsicCallResult Intrinsics::call(const Binding& admitted,
                                     std::span<const Value> arguments,
                                     ValueArena& arena, Storage& storage,
                                     cxx::AST* owner, uint8_t math_flags,
                                     loom_builder_t* builder,
                                     loom_location_id_t location) {
  const auto* binding = &admitted;
  if (auto* decode = std::get_if<DecodeIntrinsic>(binding)) {
    return {decode->call(arguments, builder, location)};
  }
  if (auto* encoding = std::get_if<EncodingIntrinsic>(binding)) {
    return {encoding->call(arena, builder, location)};
  }
  if (auto* view = std::get_if<ViewIntrinsic>(binding)) {
    return {view->call(arguments, types_, arena, storage, owner, builder,
                       location)};
  }
  if (auto* atomic = std::get_if<AtomicIntrinsic>(binding)) {
    return {atomic->call(arguments, storage, owner, builder, location)};
  }
  if (auto* fence = std::get_if<FenceIntrinsic>(binding)) {
    fence->call(builder, location);
    return {std::nullopt};
  }
  if (auto* subgroup = std::get_if<SubgroupIntrinsic>(binding)) {
    return {Value(subgroup->call(arguments, builder, location))};
  }
  if (auto* barrier = std::get_if<BarrierIntrinsic>(binding)) {
    barrier->call(builder, location);
    return {std::nullopt};
  }
  if (auto* application = std::get_if<TemplateApplyIntrinsic>(binding)) {
    return {
        application->call(arguments, types_, arena, owner, builder, location)};
  }
  std::array<loom_value_id_t, 8> inline_values;
  std::vector<loom_value_id_t> overflow;
  auto flattened = flatten(arguments, inline_values, overflow);
  if (auto* scalar = std::get_if<ScalarBinding>(binding)) {
    loom_op_t* op;
    check(scalar->operation.scalar->build(
        builder, scalar->operation.flags | math_flags, flattened.data(),
        scalar->type, location, &op));
    return {Value(loom_op_results(op)[0])};
  }
  const auto& shaped = std::get<ShapedIntrinsic>(admitted);
  return {Value(shaped.call(flattened, math_flags, builder, location))};
}

}  // namespace loom::cxx_import
