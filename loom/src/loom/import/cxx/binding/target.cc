// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/target.h"

#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include "loom/import/cxx/source/diagnostics.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/target/ops.h"

namespace loom::cxx_import {

std::optional<TargetIntrinsic> TargetIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  if (attribute.arguments.empty() ||
      attribute.arguments[0]->name() != "target.subgroup.size") {
    return std::nullopt;
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  auto* result = types.unqualified(signature->returnType());
  auto result_type = types.get(result, owner);
  if (attribute.arguments.size() != 1 || signature->isVariadic() ||
      !signature->parameterTypes().empty() ||
      !loom_type_equal(result_type, loom_type_scalar(LOOM_SCALAR_TYPE_I32)) ||
      !types.is_unsigned(result)) {
    diagnostics.reject(
        unit, owner,
        "target.subgroup.size requires unsigned() with one operation binding");
  }
  return TargetIntrinsic(result_type);
}

loom_value_id_t TargetIntrinsic::call(loom_builder_t* builder,
                                      loom_location_id_t location) const {
  loom_op_t* op;
  auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  check(loom_target_subgroup_size_build(builder, index_type, location, &op));
  auto value = loom_op_results(op)[0];
  check(loom_index_cast_build(builder, value, index_type, result_type_,
                              location, &op));
  return loom_op_results(op)[0];
}

}  // namespace loom::cxx_import
