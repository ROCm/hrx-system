// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_VIEW_H_
#define LOOM_IMPORT_CXX_BINDING_VIEW_H_

#include <cxx/attributes.h>
#include <cxx/types_fwd.h>

#include <optional>
#include <span>
#include <string_view>

#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/value/representation.h"
#include "loom/import/cxx/value/types.h"
#include "loom/ops/op_defs.h"

namespace loom::cxx_import {

class Storage;

// One concrete specialization of a typed encoding or view operation. Template
// declarations establish the public operation spelling; concrete source types
// establish the exact rank, extent, element, and constness contract retained
// here. Calls consume that contract without reexamining source declarations.
class ViewIntrinsic {
 public:
  // Returns whether |name| belongs to this binding family.
  static bool supports(std::string_view name);

  // Resolves a concrete specialization. Unknown operation names return no
  // value; malformed recognized declarations diagnose at |owner|.
  static std::optional<ViewIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      const cxx::FunctionType* signature, const cxx::Attribute& attribute,
      cxx::AST* owner);

  // Emits the admitted operation. Value absence is the result of a handled
  // void store, not an unrecognized intrinsic.
  std::optional<Value> call(std::span<const Value> arguments, Types& types,
                            ValueArena& arena, Storage& storage,
                            cxx::AST* owner, loom_builder_t* builder,
                            loom_location_id_t location) const;

  // Compares the complete concrete contract for one canonical declaration.
  bool equivalent(const ViewIntrinsic& other) const;

 private:
  enum class Operation {
    LayoutDense,
    LayoutStrided,
    BufferView,
    Subview,
    Load,
    Store,
  };

  static std::optional<Operation> parse_operation(std::string_view name);

  explicit ViewIntrinsic(Operation operation) : operation_(operation) {}

  // Selected operation kind.
  Operation operation_;
  // Concrete source return type used to bind dependent result identities.
  const cxx::Type* result_source_type_ = nullptr;
  // Admitted encoding result for layout factories.
  const EncodingPartition* result_encoding_ = nullptr;
  // Admitted source view for subview/load/store.
  const ViewPartition* source_view_ = nullptr;
  // Admitted dependent result view for buffer.view/subview.
  const ViewPartition* result_view_ = nullptr;
  // Native scalar result for view.load.
  loom_type_t scalar_result_ = {};
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_VIEW_H_
