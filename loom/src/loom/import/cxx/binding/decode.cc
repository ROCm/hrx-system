// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/decode.h"

#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <array>

#include "loom/import/cxx/source/error.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/auxiliary.h"
#include "loom/ops/vector/ops.h"

namespace loom::cxx_import {

std::optional<DecodeIntrinsic> DecodeIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    const cxx::FunctionType* signature, const cxx::Attribute& attribute,
    loom_module_t* module, cxx::AST* owner) {
  if (!supports(attribute.arguments[0]->name())) {
    return std::nullopt;
  }
  auto parameters = signature->parameterTypes();
  if (attribute.arguments.size() != 1 || signature->isVariadic() ||
      (parameters.size() != 2 && parameters.size() != 3)) {
    diagnostics.reject(unit, owner,
                       "vector.decode requires payload, schema, and an "
                       "optional auxiliary record");
  }
  if (!types.native_vector(parameters[0]) ||
      !types.native_vector(signature->returnType())) {
    diagnostics.reject(unit, owner,
                       "vector.decode payload and result must be vectors");
  }
  const auto& schema = types.partition(parameters[1], owner);
  if (schema.kind != ValueKind::Encoding ||
      static_cast<const EncodingPartition&>(schema).role !=
          LOOM_ENCODING_ROLE_STORAGE_SCHEMA) {
    diagnostics.reject(unit, owner,
                       "vector.decode requires a schema encoding value");
  }
  std::vector<loom_string_id_t> names;
  if (parameters.size() == 3) {
    auto* record = types.record(parameters[2], owner);
    if (!record) {
      diagnostics.reject(unit, owner,
                         "vector.decode auxiliaries must be a flat record "
                         "of named vectors");
    }
    if (record->members.size() > LOOM_ENCODING_AUXILIARY_KEY_COUNT_) {
      diagnostics.reject(unit, owner, "too many vector.decode auxiliaries");
    }
    names.reserve(record->members.size());
    for (const auto& member : record->members) {
      if (!types.native_vector(member.type())) {
        diagnostics.reject(unit, owner,
                           "vector.decode auxiliaries must be a flat record "
                           "of named vectors");
      }
      auto name = cxx::to_string(member.source->name());
      loom_encoding_auxiliary_key_t key;
      if (!loom_encoding_auxiliary_key_lookup(view(name), &key)) {
        diagnostics.reject(unit, owner,
                           "unknown vector.decode auxiliary '" + name + "'");
      }
      loom_string_id_t name_id;
      check(loom_module_intern_string(module, view(name), &name_id));
      names.push_back(name_id);
    }
  }
  return DecodeIntrinsic(types.get(signature->returnType(), owner),
                         std::move(names));
}

Value DecodeIntrinsic::call(std::span<const Value> arguments,
                            loom_builder_t* builder,
                            loom_location_id_t location) const {
  std::array<loom_named_value_t, LOOM_ENCODING_AUXILIARY_KEY_COUNT_> auxiliary;
  for (size_t i = 0; i < auxiliary_names_.size(); ++i) {
    auxiliary[i] = {auxiliary_names_[i], 0, arguments[2].components()[i]};
  }
  loom_op_t* op;
  check(loom_vector_decode_build(
      builder, arguments[0].ssa(), arguments[1].components()[0],
      auxiliary.data(), auxiliary_names_.size(), result_type_, location, &op));
  return Value(loom_op_results(op)[0]);
}

}  // namespace loom::cxx_import
