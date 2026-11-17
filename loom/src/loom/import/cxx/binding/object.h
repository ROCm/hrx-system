// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_OBJECT_H_
#define LOOM_IMPORT_CXX_BINDING_OBJECT_H_

#include <cxx/attributes.h>
#include <cxx/symbols_fwd.h>

#include <optional>
#include <span>
#include <string_view>

#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

// Admitted managed-handle operations. Source special members express copies,
// moves, and destruction; these bindings emit their explicit High primitives.
class ObjectIntrinsic {
 public:
  static bool supports(std::string_view name);
  static std::optional<ObjectIntrinsic> resolve(cxx::TranslationUnit& unit,
                                                Diagnostics& diagnostics,
                                                Types& types,
                                                cxx::FunctionSymbol* function,
                                                const cxx::Attribute& attribute,
                                                cxx::AST* owner);

  std::optional<Value> call(std::span<const Value> arguments, ValueArena& arena,
                            loom_builder_t* builder,
                            loom_location_id_t location) const;

  bool equivalent(const ObjectIntrinsic& other) const {
    return operation_ == other.operation_ && result_ == other.result_ &&
           loom_type_equal(result_type_, other.result_type_);
  }

 private:
  enum class Operation {
    Null,
    Duplicate,
    Borrow,
    Release,
    CompareNull,
    CompareEqual
  };
  static std::optional<Operation> find(std::string_view name);

  ObjectIntrinsic(Operation operation, const Partition* result,
                  loom_type_t result_type)
      : operation_(operation), result_(result), result_type_(result_type) {}

  // Operation selected once at declaration admission.
  Operation operation_;
  // Source result structure, absent for release.
  const Partition* result_;
  // Exact High result identity and ownership, absent for release.
  loom_type_t result_type_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_OBJECT_H_
