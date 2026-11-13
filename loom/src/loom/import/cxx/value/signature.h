// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_SIGNATURE_H_
#define LOOM_IMPORT_CXX_VALUE_SIGNATURE_H_

#include <cxx/types_fwd.h>

#include <cstddef>
#include <span>
#include <vector>

#include "loom/import/cxx/value/types.h"
#include "loom/ops/op_defs.h"

namespace loom::cxx_import {

// A flattened High signature and, when any component type is dependent, the
// destination identities reserved for the immediately following op build.
// The object must remain alive until that build consumes every identity.
struct BoundSignature {
  BoundSignature() = default;
  BoundSignature(const BoundSignature&) = delete;
  BoundSignature& operator=(const BoundSignature&) = delete;
  BoundSignature(BoundSignature&&) = default;
  BoundSignature& operator=(BoundSignature&&) = default;

  // High component types in source-value and partition order.
  std::vector<loom_type_t> types;
  // Reserved identities in the same order; empty for a static signature.
  std::vector<loom_value_id_t> identities;
  // Overflow dimension payloads retained until the signature's builder call.
  BoundTypeStorage type_storage;
};

// Independently bound header and result schemes for one condition-controlled
// loop. The generated scf.while builder consumes reserved header identities
// before result identities, so |identities| must remain alive through that
// builder call.
struct BoundConditionLoopSignatures {
  BoundConditionLoopSignatures() = default;
  BoundConditionLoopSignatures(const BoundConditionLoopSignatures&) = delete;
  BoundConditionLoopSignatures& operator=(const BoundConditionLoopSignatures&) =
      delete;
  BoundConditionLoopSignatures(BoundConditionLoopSignatures&&) = default;
  BoundConditionLoopSignatures& operator=(BoundConditionLoopSignatures&&) =
      default;

  // Storage backing the builder's pending reservation.
  std::vector<loom_value_id_t> identities;
  // Before-region recurring state, including any independent suffix types.
  std::vector<loom_type_t> header_types;
  // Overflow dimension payloads retained through the loop builder call.
  BoundTypeStorage header_type_storage;
  // After-region and operation-result recurring state.
  std::vector<loom_type_t> result_types;
  // Overflow dimension payloads retained through the loop builder call.
  BoundTypeStorage result_type_storage;
};

// Controls whether a static signature needs its destination identities before
// its owning operation is built. Dependent types always require identities;
// callable predicates additionally reference otherwise static arguments and
// results.
enum class SignatureIdentityRequirement {
  DependentTypes,
  Required,
};

// Retained placement of one source parameter in a staged native signature.
// Source components are permuted into one leading specialization group and
// one trailing binding group without losing the parameter's source identity.
struct StagedParameterProjection {
  // Offset in source-component order across all source parameters.
  size_t source_component_offset;
  // Total flattened source components reconstructed for the C++ parameter.
  size_t component_count;
  // First native specialization component contributed by this parameter.
  size_t specialization_offset;
  // Number of leading source components transported as specializations.
  size_t specialization_count;
  // First native binding component contributed by this parameter.
  size_t binding_offset;
  // Number of trailing source components transported as bindings.
  size_t binding_count;
};

// Immutable source-to-native projection shared by command and pipeline
// signatures. Admission owners select each parameter's specialization/binding
// split; binding and call translation consume only the retained mapping.
struct StagedSignatureProjection {
  // Appends one source parameter and its leading specialization component
  // count, retaining all remaining components as launch bindings.
  void append(const cxx::Type* source_type, size_t component_count,
              size_t specialization_count);

  // Source ordinal of the first parameter after the launch boundary.
  size_t launch_parameter_ordinal = 0;
  // Flattened specialization count before all native bindings.
  size_t specialization_count = 0;
  // Flattened native binding count after all specializations.
  size_t binding_count = 0;
  // Whether any native component type references another component identity.
  bool requires_binding = false;
  // Canonical unqualified source types in source parameter order.
  std::vector<const cxx::Type*> source_types;
  // Source parameter order, retained independently of native argument order.
  std::vector<StagedParameterProjection> parameters;
};

// Types and reserved identities consumed by one staged callable builder.
// Destination identities follow native order while source identities follow
// flattened C++ parameter order for type and predicate binding.
struct BoundStagedSignature {
  BoundStagedSignature() = default;
  BoundStagedSignature(const BoundStagedSignature&) = delete;
  BoundStagedSignature& operator=(const BoundStagedSignature&) = delete;
  BoundStagedSignature(BoundStagedSignature&&) = default;
  BoundStagedSignature& operator=(BoundStagedSignature&&) = default;

  // Retained projection shared with body and call translation.
  const StagedSignatureProjection* projection = nullptr;
  // Native specialization types in callable signature order.
  std::vector<loom_type_t> specialization_types;
  // Native binding types in callable signature order.
  std::vector<loom_type_t> binding_types;
  // Builder-reserved identities in specialization-then-binding order.
  std::vector<loom_value_id_t> destination_identities;
  // The same identities permuted into flattened source parameter order.
  std::vector<loom_value_id_t> source_identities;
  // Overflow type payloads retained through the owning builder call.
  BoundTypeStorage type_storage;
};

// Projects |sources| into one callable or structured-result signature. If a
// source view occurs directly or inside a record, reserves every destination
// identity before binding its dependent shape/layout references. Static-only
// signatures preserve the ordinary builder path and allocate no identities.
BoundSignature bind_signature(Types& types,
                              std::span<const cxx::Type* const> sources,
                              cxx::AST* owner, loom_builder_t* builder,
                              SignatureIdentityRequirement identities =
                                  SignatureIdentityRequirement::DependentTypes);

// Projects |sources| onto caller-reserved destination identities. The span
// contains exactly one identity per flattened component in source order and
// remains available on the returned signature for predicate substitution.
BoundSignature bind_signature_to(Types& types,
                                 std::span<const cxx::Type* const> sources,
                                 cxx::AST* owner,
                                 std::span<const loom_value_id_t> identities);

// Binds a retained staged projection to the next command or pipeline builder.
// Every source component must belong to exactly one specialization or binding
// range; the projection's native permutation is applied without rediscovery.
BoundStagedSignature bind_staged_signature(
    const StagedSignatureProjection& projection, Types& types, cxx::AST* owner,
    loom_builder_t* builder,
    SignatureIdentityRequirement identities =
        SignatureIdentityRequirement::DependentTypes);

// Binds the independent header and result schemes required by scf.while when
// any source component has a dependent type. |header_suffix_types| describes
// independent synthetic state, such as the active bit used for structured
// break. Static state leaves both schemes empty unless
// |require_result_types| is true, which supports different header/result
// arities without changing the ordinary inferred path.
BoundConditionLoopSignatures bind_condition_loop_signatures(
    Types& types, std::span<const cxx::Type* const> header_sources,
    std::span<const cxx::Type* const> result_sources, cxx::AST* owner,
    loom_builder_t* builder,
    std::span<const loom_type_t> header_suffix_types = {},
    bool require_result_types = false);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_SIGNATURE_H_
