// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/atomic.h"

#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <array>
#include <cstdint>
#include <utility>
#include <variant>

#include "loom/import/cxx/source/error.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/view/ops.h"

namespace loom::cxx_import {
namespace {

void read_selectors(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                    cxx::FunctionSymbol* function,
                    const cxx::Attribute& attribute, cxx::AST* owner,
                    std::span<uint8_t> selectors) {
  auto arguments = function->templateArguments();
  if (attribute.arguments.size() != 1 || arguments.size() < selectors.size()) {
    diagnostics.reject(
        unit, owner,
        selectors.size() == 2
            ? "atomic binding requires two leading constant template arguments"
            : "atomic binding requires three leading constant template "
              "arguments");
  }
  for (size_t index = 0; index < selectors.size(); ++index) {
    auto value = cxx::template_argument_value(arguments[index]);
    auto* number = value ? std::get_if<std::intmax_t>(&*value) : nullptr;
    if (!number || *number < 0 || *number > UINT8_MAX) {
      diagnostics.reject(unit, owner,
                         "atomic selectors require constant enum values");
    }
    selectors[index] = static_cast<uint8_t>(*number);
  }
}

void require_ordering(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                      cxx::AST* owner, const loom_op_vtable_t* operation,
                      loom_diagnostic_field_ref_t field, uint8_t ordering) {
  // Import constructs verified High, so reject source values using the owning
  // operation's schema before constructing IR. The allowed cases stay in High.
  if (!loom_attr_descriptor_has_enum_case(
          &operation->attr_descriptors[field.index], ordering)) {
    diagnostics.reject(unit, owner,
                       "ordering is not permitted for this operation");
  }
}

}  // namespace

bool AtomicIntrinsic::supports(std::string_view name) {
  return name == "view.atomic.load" || name == "view.atomic.store" ||
         name == "view.atomic.rmw" || name == "view.atomic.reduce" ||
         name == "view.atomic.cmpxchg";
}

std::optional<AtomicIntrinsic> AtomicIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  auto name = attribute.arguments[0]->name();
  if (!supports(name)) {
    return std::nullopt;
  }
  auto operation = name == "view.atomic.load"     ? Operation::Load
                   : name == "view.atomic.store"  ? Operation::Store
                   : name == "view.atomic.rmw"    ? Operation::Rmw
                   : name == "view.atomic.reduce" ? Operation::Reduce
                                                  : Operation::CompareExchange;
  size_t selector_count =
      operation == Operation::Load || operation == Operation::Store ? 2 : 3;
  std::array<uint8_t, 3> selectors;
  read_selectors(unit, diagnostics, function, attribute, owner,
                 std::span(selectors).first(selector_count));
  auto kind = LOOM_ATOMIC_KIND_XCHGI;
  auto ordering = LOOM_ATOMIC_ORDERING_RELAXED;
  auto failure_ordering = LOOM_ATOMIC_ORDERING_RELAXED;
  if (operation == Operation::CompareExchange) {
    if (!loom_atomic_ordering_is_valid(selectors[0]) ||
        !loom_atomic_ordering_is_valid(selectors[1])) {
      diagnostics.reject(unit, owner, "unknown atomic ordering");
    }
    ordering = static_cast<loom_atomic_ordering_t>(selectors[0]);
    failure_ordering = static_cast<loom_atomic_ordering_t>(selectors[1]);
    auto error =
        loom_atomic_cmpxchg_ordering_validate(ordering, failure_ordering);
    if (error != LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_NONE) {
      diagnostics.reject(
          unit, owner,
          string(
              loom_atomic_cmpxchg_ordering_error_expected_constraint(error)));
    }
  } else {
    if (operation == Operation::Rmw || operation == Operation::Reduce) {
      if (!loom_atomic_kind_is_valid(selectors[0]) ||
          !loom_atomic_kind_accepts_integer(selectors[0])) {
        diagnostics.reject(unit, owner,
                           "atomic binding requires an integer kind");
      }
      kind = static_cast<loom_atomic_kind_t>(selectors[0]);
    }
    auto selected_ordering = selectors[selector_count - 2];
    if (!loom_atomic_ordering_is_valid(selected_ordering)) {
      diagnostics.reject(unit, owner, "unknown atomic ordering");
    }
    ordering = static_cast<loom_atomic_ordering_t>(selected_ordering);
  }
  if (operation == Operation::Load || operation == Operation::Store) {
    auto kind = operation == Operation::Load ? LOOM_OP_VIEW_ATOMIC_LOAD
                                             : LOOM_OP_VIEW_ATOMIC_STORE;
    auto field = operation == Operation::Load
                     ? loom_view_atomic_load_ordering_diagnostic_ref()
                     : loom_view_atomic_store_ordering_diagnostic_ref();
    require_ordering(
        unit, diagnostics, owner,
        loom_view_dialect_vtables(nullptr)[loom_op_dialect_index(kind)], field,
        ordering);
  }
  auto selected_scope = selectors[selector_count - 1];
  if (!loom_atomic_scope_is_valid(selected_scope)) {
    diagnostics.reject(unit, owner, "unknown atomic scope");
  }
  auto scope = static_cast<loom_atomic_scope_t>(selected_scope);

  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  auto parameters = signature->parameterTypes();
  size_t operand_count = operation == Operation::Load              ? 1
                         : operation == Operation::CompareExchange ? 3
                                                                   : 2;
  if (signature->isVariadic() || parameters.size() != operand_count) {
    diagnostics.reject(unit, owner,
                       "atomic declaration has the wrong operand count");
  }
  auto* pointer =
      cxx::type_cast<cxx::PointerType>(types.unqualified(parameters.back()));
  if (!pointer) {
    diagnostics.reject(unit, owner, "atomic access requires a pointer");
  }
  if (operation != Operation::Load &&
      unit.typeTraits().is_const(pointer->elementType())) {
    diagnostics.reject(unit, owner,
                       "atomic destination requires a mutable pointer");
  }
  auto* element = types.unqualified(pointer->elementType());
  if (!unit.typeTraits().is_integral(element) ||
      element->kind() == cxx::TypeKind::kBool) {
    diagnostics.reject(unit, owner,
                       "atomic destination requires a non-boolean integer");
  }
  for (size_t index = 0; index + 1 < parameters.size(); ++index) {
    if (types.unqualified(parameters[index]) != element) {
      diagnostics.reject(unit, owner,
                         "atomic operands must match the destination element "
                         "type");
    }
  }
  auto* result = types.unqualified(signature->returnType());
  bool returns_void =
      operation == Operation::Reduce || operation == Operation::Store;
  if (returns_void ? result->kind() != cxx::TypeKind::kVoid
                   : result != element) {
    diagnostics.reject(unit, owner,
                       "atomic result must be the destination element type, "
                       "or void for store/reduction");
  }
  if (((kind == LOOM_ATOMIC_KIND_MINSI || kind == LOOM_ATOMIC_KIND_MAXSI) &&
       types.is_unsigned(element)) ||
      ((kind == LOOM_ATOMIC_KIND_MINUI || kind == LOOM_ATOMIC_KIND_MAXUI) &&
       !types.is_unsigned(element))) {
    diagnostics.reject(unit, owner,
                       "atomic minimum/maximum signedness must match the "
                       "destination element type");
  }
  return AtomicIntrinsic(operation, pointer->elementType(),
                         types.get(element, owner), kind, ordering,
                         failure_ordering, scope);
}

std::optional<Value> AtomicIntrinsic::call(std::span<const Value> arguments,
                                           Storage& storage, cxx::AST* owner,
                                           loom_builder_t* builder,
                                           loom_location_id_t location) const {
  auto access =
      storage.dereference(arguments.back().pointer(), element_type_, owner);
  const int64_t index = 0;
  loom_op_t* op;
  switch (operation_) {
    case Operation::Load:
      check(loom_view_atomic_load_build(builder, 0, access.view, nullptr, 0,
                                        &index, 1, ordering_, scope_, 0, 0,
                                        type_, location, &op));
      return Value(loom_op_results(op)[0]);
    case Operation::Store:
      check(loom_view_atomic_store_build(
          builder, 0, arguments[0].ssa(), access.view, nullptr, 0, &index, 1,
          ordering_, scope_, 0, 0, location, &op));
      return std::nullopt;
    case Operation::Rmw:
      check(loom_view_atomic_rmw_build(
          builder, 0, kind_, arguments[0].ssa(), access.view, nullptr, 0,
          &index, 1, ordering_, scope_, 0, 0, type_, location, &op));
      return Value(loom_op_results(op)[0]);
    case Operation::Reduce:
      check(loom_view_atomic_reduce_build(
          builder, 0, kind_, arguments[0].ssa(), access.view, nullptr, 0,
          &index, 1, ordering_, scope_, 0, 0, location, &op));
      return std::nullopt;
    case Operation::CompareExchange:
      check(loom_view_atomic_cmpxchg_build(
          builder, 0, arguments[0].ssa(), arguments[1].ssa(), access.view,
          nullptr, 0, &index, 1, ordering_, failure_ordering_, scope_, 0, 0,
          type_, location, &op));
      return Value(loom_op_results(op)[0]);
  }
  std::unreachable();
}

bool AtomicIntrinsic::equivalent(const AtomicIntrinsic& other) const {
  return operation_ == other.operation_ &&
         element_type_ == other.element_type_ && kind_ == other.kind_ &&
         ordering_ == other.ordering_ &&
         failure_ordering_ == other.failure_ordering_ && scope_ == other.scope_;
}

bool FenceIntrinsic::supports(std::string_view name) {
  return name == "buffer.fence";
}

std::optional<FenceIntrinsic> FenceIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  if (!supports(attribute.arguments[0]->name())) {
    return std::nullopt;
  }
  std::array<uint8_t, 2> selectors;
  read_selectors(unit, diagnostics, function, attribute, owner, selectors);
  if (!loom_atomic_ordering_is_valid(selectors[0])) {
    diagnostics.reject(unit, owner, "unknown atomic ordering");
  }
  if (!loom_atomic_scope_is_valid(selectors[1])) {
    diagnostics.reject(unit, owner, "unknown atomic scope");
  }
  require_ordering(unit, diagnostics, owner,
                   loom_buffer_dialect_vtables(
                       nullptr)[loom_op_dialect_index(LOOM_OP_BUFFER_FENCE)],
                   loom_buffer_fence_ordering_diagnostic_ref(), selectors[0]);
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  if (signature->isVariadic() || !signature->parameterTypes().empty() ||
      signature->returnType()->kind() != cxx::TypeKind::kVoid) {
    diagnostics.reject(unit, owner, "buffer.fence requires void()");
  }
  return FenceIntrinsic(static_cast<loom_atomic_ordering_t>(selectors[0]),
                        static_cast<loom_atomic_scope_t>(selectors[1]));
}

void FenceIntrinsic::call(loom_builder_t* builder,
                          loom_location_id_t location) const {
  loom_op_t* op;
  check(loom_buffer_fence_build(builder, scope_, ordering_, location, &op));
}

bool FenceIntrinsic::equivalent(const FenceIntrinsic& other) const {
  return ordering_ == other.ordering_ && scope_ == other.scope_;
}

}  // namespace loom::cxx_import
