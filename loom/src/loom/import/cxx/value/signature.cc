// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/signature.h"

#include <algorithm>
#include <utility>

#include "loom/import/cxx/source/error.h"

namespace loom::cxx_import {
namespace {

size_t component_count(Types& types, std::span<const cxx::Type* const> sources,
                       cxx::AST* owner) {
  size_t result = 0;
  for (const auto* source : sources) {
    result += types.partition(source, owner).component_count;
  }
  return result;
}

bool signature_requires_binding(Types& types,
                                std::span<const cxx::Type* const> sources,
                                cxx::AST* owner) {
  for (const auto* source : sources) {
    if (types.requires_binding(source, owner)) {
      return true;
    }
  }
  return false;
}

void append_bound(Types& types, std::span<const cxx::Type* const> sources,
                  cxx::AST* owner, std::span<const loom_value_id_t> identities,
                  BoundTypeStorage& type_storage,
                  std::vector<loom_type_t>& output_types) {
  size_t component_offset = 0;
  for (const auto* source : sources) {
    auto count = types.partition(source, owner).component_count;
    types.append_bound(source, owner,
                       identities.subspan(component_offset, count),
                       type_storage, output_types);
    component_offset += count;
  }
}

// The caller writes and the callee reads these same two native slices.
template <typename T>
auto argument_slices(const StagedParameterProjection& parameter,
                     std::span<T> specializations, std::span<T> bindings) {
  return std::pair{
      specializations.subspan(parameter.specialization_offset,
                              parameter.specialization_count),
      bindings.subspan(parameter.binding_offset, parameter.binding_count)};
}

}  // namespace

void StagedParameterProjection::write_argument(
    std::span<const loom_value_id_t> components,
    std::span<loom_value_id_t> specializations,
    std::span<loom_value_id_t> bindings) const {
  IREE_ASSERT_EQ(components.size(), component_count);
  const auto [first, last] = argument_slices(*this, specializations, bindings);
  const auto split = components.begin() + specialization_count;
  std::copy(components.begin(), split, first.begin());
  std::copy(split, components.end(), last.begin());
}

Value StagedSignatureProjection::entry_argument(size_t ordinal,
                                                loom_region_t* region,
                                                const Partition& partition,
                                                ValueArena& arena) const {
  auto* block = loom_region_entry_block(region);
  auto native =
      std::span<const loom_value_id_t>(block->arg_ids, block->arg_count);
  const auto [first, last] =
      argument_slices(parameters[ordinal], native.first(specialization_count),
                      native.subspan(specialization_count));
  return arena.capture(partition, first, last);
}

void StagedSignatureProjection::append(const cxx::Type* source_type,
                                       size_t component_count,
                                       size_t parameter_specialization_count) {
  IREE_ASSERT_LE(parameter_specialization_count, component_count);
  source_types.push_back(source_type);
  parameters.push_back({
      specialization_count + binding_count,
      component_count,
      specialization_count,
      parameter_specialization_count,
      binding_count,
      component_count - parameter_specialization_count,
  });
  specialization_count += parameter_specialization_count;
  binding_count += component_count - parameter_specialization_count;
}

BoundSignature bind_signature(Types& types,
                              std::span<const cxx::Type* const> sources,
                              cxx::AST* owner, loom_builder_t* builder,
                              SignatureIdentityRequirement identities) {
  BoundSignature result;
  bool requires_binding = identities == SignatureIdentityRequirement::Required;
  requires_binding |= signature_requires_binding(types, sources, owner);
  const size_t count = component_count(types, sources, owner);
  result.types.reserve(count);
  if (!requires_binding) {
    for (const auto* source : sources) {
      types.append(source, owner, result.types);
    }
    return result;
  }

  result.identities.resize(count);
  check(loom_builder_reserve_values(builder, result.identities.size(),
                                    result.identities.data()));
  append_bound(types, sources, owner, result.identities, result.type_storage,
               result.types);
  return result;
}

BoundSignature bind_signature_to(Types& types,
                                 std::span<const cxx::Type* const> sources,
                                 cxx::AST* owner,
                                 std::span<const loom_value_id_t> identities) {
  BoundSignature result;
  const size_t count = component_count(types, sources, owner);
  IREE_ASSERT_EQ(identities.size(), count);
  result.types.reserve(count);
  result.identities.assign(identities.begin(), identities.end());
  append_bound(types, sources, owner, result.identities, result.type_storage,
               result.types);
  return result;
}

BoundStagedSignature bind_staged_signature(
    const StagedSignatureProjection& projection, Types& types, cxx::AST* owner,
    loom_builder_t* builder, SignatureIdentityRequirement identities) {
  BoundStagedSignature result;
  result.projection = &projection;
  const size_t component_count =
      projection.specialization_count + projection.binding_count;
  const bool requires_identities =
      identities == SignatureIdentityRequirement::Required ||
      projection.requires_binding;
  if (requires_identities) {
    result.destination_identities.resize(component_count);
    check(loom_builder_reserve_values(builder, component_count,
                                      result.destination_identities.data()));
    result.source_identities.resize(component_count);
    auto native =
        std::span<const loom_value_id_t>(result.destination_identities);
    for (const auto& parameter : projection.parameters) {
      const auto [specializations, bindings] = argument_slices(
          parameter, native.first(projection.specialization_count),
          native.subspan(projection.specialization_count));
      auto output = std::span(result.source_identities)
                        .subspan(parameter.source_component_offset,
                                 parameter.component_count)
                        .begin();
      output =
          std::copy(specializations.begin(), specializations.end(), output);
      std::copy(bindings.begin(), bindings.end(), output);
    }
  }

  result.specialization_types.reserve(projection.specialization_count);
  result.binding_types.reserve(projection.binding_count);
  std::vector<loom_type_t> source_types;
  source_types.reserve(component_count);
  for (size_t ordinal = 0; ordinal < projection.source_types.size();
       ++ordinal) {
    const auto& parameter = projection.parameters[ordinal];
    const size_t previous_count = source_types.size();
    if (requires_identities) {
      types.append_bound(
          projection.source_types[ordinal], owner,
          std::span<const loom_value_id_t>(result.source_identities)
              .subspan(parameter.source_component_offset,
                       parameter.component_count),
          result.type_storage, source_types);
    } else {
      types.append(projection.source_types[ordinal], owner, source_types);
    }
    IREE_ASSERT_EQ(source_types.size() - previous_count,
                   parameter.component_count);
  }
  for (const auto& parameter : projection.parameters) {
    const auto parameter_types = std::span<const loom_type_t>(source_types)
                                     .subspan(parameter.source_component_offset,
                                              parameter.component_count);
    result.specialization_types.insert(
        result.specialization_types.end(), parameter_types.begin(),
        parameter_types.begin() + parameter.specialization_count);
    result.binding_types.insert(
        result.binding_types.end(),
        parameter_types.begin() + parameter.specialization_count,
        parameter_types.end());
  }
  return result;
}

BoundConditionLoopSignatures bind_condition_loop_signatures(
    Types& types, std::span<const cxx::Type* const> header_sources,
    std::span<const cxx::Type* const> result_sources, cxx::AST* owner,
    loom_builder_t* builder, std::span<const loom_type_t> header_suffix_types,
    bool require_result_types) {
  BoundConditionLoopSignatures result;
  if (!signature_requires_binding(types, header_sources, owner) &&
      !signature_requires_binding(types, result_sources, owner)) {
    if (require_result_types) {
      const size_t count = component_count(types, result_sources, owner);
      result.result_types.reserve(count);
      for (const auto* source : result_sources) {
        types.append(source, owner, result.result_types);
      }
    }
    return result;
  }

  const size_t source_component_count =
      component_count(types, header_sources, owner);
  const size_t result_component_count =
      component_count(types, result_sources, owner);
  const size_t header_component_count =
      source_component_count + header_suffix_types.size();
  result.identities.resize(header_component_count + result_component_count);
  check(loom_builder_reserve_values(builder, result.identities.size(),
                                    result.identities.data()));
  result.header_types.reserve(header_component_count);
  append_bound(types, header_sources, owner,
               std::span<const loom_value_id_t>(result.identities)
                   .first(source_component_count),
               result.header_type_storage, result.header_types);
  result.header_types.insert(result.header_types.end(),
                             header_suffix_types.begin(),
                             header_suffix_types.end());
  result.result_types.reserve(result_component_count);
  append_bound(types, result_sources, owner,
               std::span<const loom_value_id_t>(result.identities)
                   .subspan(header_component_count, result_component_count),
               result.result_type_storage, result.result_types);
  return result;
}

}  // namespace loom::cxx_import
