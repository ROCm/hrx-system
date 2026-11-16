// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_PIPELINE_H_
#define LOOM_IMPORT_CXX_BINDING_PIPELINE_H_

#include <cxx/attributes.h>

#include <optional>
#include <string_view>

#include "loom/import/cxx/binding/pipeline_contracts.h"
#include "loom/import/cxx/value/types.h"
#include "loom/ir/types.h"

namespace loom::cxx_import {

// Selects an invocation-owned memory pool. Call translation retains the direct
// source index list so static coordinates remain attributes instead of
// becoming temporary SSA constants.
struct PipelineMemoryIntrinsic {
  static bool supports(std::string_view name);
  static std::optional<PipelineMemoryIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
      cxx::AST* owner);

  bool equivalent(const PipelineMemoryIntrinsic& other) const {
    return memory_space == other.memory_space && result == other.result;
  }

  loom_value_fact_memory_space_t memory_space;
  const Partition* result;
};

// Composes one statically selected child using the same staged signature that
// owns its declaration or definition. Composition constructs graph state; it
// is intentionally distinct from an ordinary C++ call or a kernel launch.
struct PipelineComposeIntrinsic {
  static bool supports(std::string_view name);
  static std::optional<PipelineComposeIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      PipelineContracts& pipelines, cxx::FunctionSymbol* function,
      const cxx::Attribute& attribute, cxx::AST* owner);

  bool equivalent(const PipelineComposeIntrinsic& other) const {
    return pipeline == other.pipeline && projection == other.projection;
  }

  cxx::FunctionSymbol* pipeline;
  const StagedSignatureProjection* projection;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_PIPELINE_H_
