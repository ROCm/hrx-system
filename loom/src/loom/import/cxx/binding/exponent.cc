// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/exponent.h"

#include <cxx/types.h>

#include "loom/import/cxx/source/error.h"
#include "loom/ops/scalar/ops.h"

namespace loom::cxx_import {

ExponentIntrinsic ExponentIntrinsic::resolve(Types& types,
                                             const cxx::FunctionType* signature,
                                             cxx::AST* owner) {
  const auto* exponent_type =
      cxx::type_cast<cxx::PointerType>(signature->parameterTypes()[1])
          ->elementType();
  return ExponentIntrinsic(types.get(signature->returnType(), owner),
                           exponent_type);
}

Value ExponentIntrinsic::call(std::span<const Value> arguments,
                              Storage& storage, cxx::AST* owner,
                              loom_builder_t* builder,
                              loom_location_id_t location) const {
  loom_op_t* op;
  check(loom_scalar_frexpf_build(builder, arguments[0].ssa(), type_, location,
                                 &op));
  auto access = storage.dereference(
      storage.project(arguments[1].pointer(), exponent_type_, owner),
      exponent_type_, owner);
  storage.store(access, loom_op_results(op)[1], exponent_type_, owner);
  return Value(loom_op_results(op)[0]);
}

}  // namespace loom::cxx_import
