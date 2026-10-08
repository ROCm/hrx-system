// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_ENCODING_H_
#define LOOM_IMPORT_CXX_BINDING_ENCODING_H_

#include <cxx/attributes.h>
#include <cxx/symbols_fwd.h>

#include <optional>
#include <span>
#include <string_view>

#include "loom/import/cxx/value/types.h"
#include "loom/ir/encoding.h"

namespace loom::cxx_import {

// Projects one constant aggregate template argument into an encoding schema.
// The declaration selects a registered family; aggregate field names and enum
// spellings use that family's public parameter vocabulary. Runtime values stay
// outside this static boundary. Each used specialization retains a module-owned
// encoding identity, so emission never reevaluates its source parameters.
class EncodingIntrinsic {
 public:
  enum class Operation {
    StaticSchema,
    PhysicalStorage,
  };

  struct Family {
    // Operation selected by the declaration's attribute spelling.
    Operation operation;
    // Public family or alias spelling, borrowed from the source declaration.
    std::string_view name;
    // Registered static parameter contract, valid for the context lifetime.
    const loom_encoding_family_descriptor_t* descriptor;
    // Canonical alias supplying fixed parameters, or null for a base family.
    const loom_encoding_alias_descriptor_t* alias;
  };

  // Resolves the family once when admitting the declaration, including an
  // unused template. Other operation names remain unclaimed.
  static std::optional<Family> admit(cxx::TranslationUnit& unit,
                                     Diagnostics& diagnostics,
                                     const loom_context_t* context,
                                     cxx::FunctionSymbol* function,
                                     const cxx::Attribute& attribute,
                                     cxx::AST* owner);

  // Admits either a static schema definition or dynamic physical-storage
  // composition. The module interner owns encoding normalization and the
  // normal IR verifier owns family semantics.
  static EncodingIntrinsic resolve(Family family, cxx::TranslationUnit& unit,
                                   Diagnostics& diagnostics, Types& types,
                                   cxx::FunctionSymbol* function,
                                   loom_module_t* module, cxx::AST* owner);

  // Emits the retained definition without source inspection.
  Value call(std::span<const Value> arguments, ValueArena& arena,
             loom_builder_t* builder, loom_location_id_t location) const;

  bool equivalent(const EncodingIntrinsic& other) const {
    return operation_ == other.operation_ &&
           encoding_id_ == other.encoding_id_ && result_ == other.result_ &&
           layout_name_id_ == other.layout_name_id_ &&
           schema_name_id_ == other.schema_name_id_;
  }

 private:
  EncodingIntrinsic(Operation operation, uint16_t encoding_id,
                    const EncodingPartition* result)
      : operation_(operation), encoding_id_(encoding_id), result_(result) {}

  // Selected definition kind.
  Operation operation_;

  // Encoding definition interned in the invocation's output module.
  uint16_t encoding_id_;
  // Source result partition owned by this invocation's Types projection.
  const EncodingPartition* result_;
  // Physical-storage layout operand key, absent for static schemas.
  loom_string_id_t layout_name_id_ = LOOM_STRING_ID_INVALID;
  // Physical-storage schema operand key, absent for static schemas.
  loom_string_id_t schema_name_id_ = LOOM_STRING_ID_INVALID;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_ENCODING_H_
