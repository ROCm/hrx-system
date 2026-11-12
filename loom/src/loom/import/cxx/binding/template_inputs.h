// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_TEMPLATE_INPUTS_H_
#define LOOM_IMPORT_CXX_BINDING_TEMPLATE_INPUTS_H_

#include "loom/import/cxx/value/representation.h"
#include "loom/import/cxx/value/signature.h"

namespace loom::cxx_import {

// Template inputs preserve nominal element/access semantics while permitting
// different extents and complete layouts within one shaped carrier.
bool template_input_compatible(Types& types, const cxx::Type* family,
                               const cxx::Type* source, cxx::AST* owner);

// Appends family-coordinate dependencies and the original actual payload.
// Only template selection may prove and refine that payload's type.
void append_template_input(Value value, const cxx::Type* family, Types& types,
                           cxx::AST* owner, loom_builder_t* builder,
                           loom_location_id_t location,
                           std::vector<loom_value_id_t>& operands);

// Reconstructs a provider's source value from its admitted family argument
// group. Explicit refinements preserve exact typing inside the source body.
Value bind_template_input(Value incoming, const cxx::Type* source, Types& types,
                          ValueArena& arena, cxx::AST* owner,
                          loom_builder_t* builder, loom_location_id_t location);

// Native family layout and source scalar-predicate correspondence. Missing
// source-only dependencies have no native identity; predicates cannot name
// those shaped-value implementation components.
struct BoundTemplateSignature {
  // Family argument/result identities consumed by the definition builder.
  BoundSignature native;
  // Provider source component order for its scalar predicate binding.
  std::vector<loom_value_id_t> source_identities;
  // Number of native argument components before the result suffix.
  size_t argument_count = 0;
};

// Binds a provider to the family's native component positions, specializing
// only shaped payload types. Source and family result types are already exact.
BoundTemplateSignature bind_template_signature(
    Types& types, const cxx::FunctionType* family,
    const cxx::FunctionType* provider, cxx::AST* owner, loom_builder_t* builder,
    bool has_predicates);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_TEMPLATE_INPUTS_H_
