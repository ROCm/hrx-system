// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/signature.h"

#include "loom/import/cxx/source/error.h"

namespace loom::cxx_import {

BoundSignature bind_signature(Types& types,
                              std::span<const cxx::Type* const> sources,
                              cxx::AST* owner, loom_builder_t* builder,
                              SignatureIdentityRequirement identities) {
  BoundSignature result;
  size_t component_count = 0;
  bool requires_binding = identities == SignatureIdentityRequirement::Required;
  for (const auto* source : sources) {
    component_count += types.partition(source, owner).component_count;
    requires_binding |= types.requires_binding(source, owner);
  }
  result.types.reserve(component_count);
  if (!requires_binding) {
    for (const auto* source : sources) {
      types.append(source, owner, result.types);
    }
    return result;
  }

  result.identities.resize(component_count);
  check(loom_builder_reserve_values(builder, result.identities.size(),
                                    result.identities.data()));
  size_t component_offset = 0;
  for (const auto* source : sources) {
    auto count = types.partition(source, owner).component_count;
    types.append_bound(source, owner,
                       std::span<const loom_value_id_t>(result.identities)
                           .subspan(component_offset, count),
                       result.type_storage, result.types);
    component_offset += count;
  }
  return result;
}

}  // namespace loom::cxx_import
