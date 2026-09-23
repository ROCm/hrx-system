// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_PARAMETER_CONTRACTS_H_
#define LOOM_IMPORT_CXX_BINDING_PARAMETER_CONTRACTS_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "loom/import/cxx/source/locations.h"
#include "loom/import/cxx/source/source.h"
#include "loom/ops/op_defs.h"

namespace loom::cxx_import {

// Entry precondition on the address supplied by one source pointer parameter.
struct ParameterAlignment {
  // Positive power-of-two byte alignment, or zero for an unannotated parameter.
  int64_t minimum_alignment = 0;
  // Original annotation owning diagnostics and the emitted assumption location.
  cxx::AttributeAST* source = nullptr;
};

// Independent entry promises on an incoming kernel buffer root. Neither
// promise changes its storage identity or applies to a reassigned pointer.
struct ParameterContract {
  // Alignment of the incoming pointer address, absent when unannotated.
  ParameterAlignment alignment;
  // Root noalias promise; null when absent. Owns diagnostics and IR location.
  cxx::AttributeAST* noalias_source = nullptr;
};

// Emits admitted entry promises while preserving buffer identity. The caller
// has established that the source pointer is a zero-origin kernel binding.
loom_value_id_t apply_parameter_contract(const ParameterContract& contract,
                                         loom_value_id_t buffer,
                                         Locations& locations,
                                         loom_builder_t* builder);

// Admits pointer-parameter contracts once and reconciles redeclarations by
// canonical function identity and parameter ordinal. Unannotated functions
// allocate no records. The source outlives this invocation-owned index.
class ParameterContracts {
 public:
  ParameterContracts(cxx::TranslationUnit& unit, Diagnostics& diagnostics)
      : unit_(unit), diagnostics_(diagnostics) {}

  // Admits a concrete declaration whose parameter attribute positions have
  // already been checked. Template patterns are admitted after substitution.
  void declaration(cxx::FunctionSymbol* function,
                   cxx::FunctionDeclaratorChunkAST* prototype);

  // Returns contracts in source parameter order, or an empty span when none
  // were declared. Records remain valid through body construction.
  std::span<const ParameterContract> get(cxx::FunctionSymbol* function) const;

 private:
  // Frontend semantic types and constant expressions.
  cxx::TranslationUnit& unit_;
  // Source admission failure boundary.
  Diagnostics& diagnostics_;
  // Only functions with explicit parameter contracts occupy the index.
  std::unordered_map<cxx::FunctionSymbol*, std::vector<ParameterContract>>
      contracts_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_PARAMETER_CONTRACTS_H_
