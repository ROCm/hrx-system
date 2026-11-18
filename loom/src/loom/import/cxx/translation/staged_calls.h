// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_TRANSLATION_STAGED_CALLS_H_
#define LOOM_IMPORT_CXX_TRANSLATION_STAGED_CALLS_H_

#include <cxx/ast_fwd.h>
#include <cxx/types_fwd.h>

#include <span>
#include <string_view>
#include <vector>

#include "loom/import/cxx/value/representation.h"

struct loom_builder_t;

namespace cxx {
class TranslationUnit;
}

namespace loom::cxx_import {

class Diagnostics;
class ExpressionEvaluator;
class Functions;
class IndexList;
class Locations;
class Scalars;
class Storage;
class Types;
enum class FunctionKind;
struct CommandLaunchIntrinsic;
struct CommandParameterIntrinsic;
struct KernelLaunchIntrinsic;
struct PipelineComposeIntrinsic;
struct PipelineMemoryIntrinsic;
struct StagedSignatureProjection;

// Emits the explicit kernel, command, and pipeline calls in a staged C++ body.
// Each operand is evaluated once in source order before its components are
// placed in the called program's admitted native signature.
class StagedCalls {
 public:
  StagedCalls(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
              ExpressionEvaluator& evaluator, Types& types, Scalars& scalars,
              Storage& storage, Functions& functions, Locations& locations,
              ValueArena& arena, loom_builder_t& builder, FunctionKind kind,
              bool in_strand)
      : unit_(unit),
        diagnostics_(diagnostics),
        evaluator_(evaluator),
        types_(types),
        scalars_(scalars),
        storage_(storage),
        functions_(functions),
        locations_(locations),
        value_arena_(arena),
        builder_(builder),
        kind_(kind),
        in_strand_(in_strand) {}

  void kernel_launch(cxx::CallExpressionAST* call,
                     const KernelLaunchIntrinsic& launch);
  Value pipeline_memory(cxx::CallExpressionAST* call,
                        const PipelineMemoryIntrinsic& binding);
  void pipeline_compose(cxx::CallExpressionAST* call,
                        const PipelineComposeIntrinsic& binding);
  void command_launch(cxx::CallExpressionAST* call,
                      const CommandLaunchIntrinsic& launch);
  Value command_parameter(cxx::CallExpressionAST* call,
                          const CommandParameterIntrinsic& binding);
  std::vector<loom_value_id_t> dynamic_indices(const IndexList& list);

 private:
  [[noreturn]] void fail(cxx::AST* owner, std::string_view message);
  const cxx::Type* storage_element_type(const cxx::Type* source_type,
                                        const Partition& partition);
  bool storage_elements_compatible(const cxx::Type* expected,
                                   const cxx::Type* actual);
  void append_kernel_workloads(cxx::ExpressionAST* source,
                               const KernelLaunchIntrinsic& launch,
                               std::vector<loom_value_id_t>& workloads);
  void append_staged_arguments(cxx::List<cxx::ExpressionAST*>* source,
                               const StagedSignatureProjection& projection,
                               std::span<const uint8_t> buffer_materializations,
                               std::vector<loom_value_id_t>& specializations,
                               std::vector<loom_value_id_t>& bindings);

  // Semantic source of canonical types and qualifiers.
  cxx::TranslationUnit& unit_;
  // Source diagnostics for invalid layouts and invocation context.
  Diagnostics& diagnostics_;
  // Surrounding evaluation owns control flow and operand side effects.
  ExpressionEvaluator& evaluator_;
  // Retained source-to-IR types of parameters and results.
  Types& types_;
  // Numeric materialization of zero-origin buffer displacements.
  Scalars& scalars_;
  // Source pointer materialization for launch bindings.
  Storage& storage_;
  // Reached callable declarations in this module.
  Functions& functions_;
  // IR locations retained from the source calls and operands.
  Locations& locations_;
  // Owns composite results for the enclosing function.
  ValueArena& value_arena_;
  // Active block in the enclosing program.
  loom_builder_t& builder_;
  // Callable kind determines legal staged operations.
  FunctionKind kind_;
  // Whether this insertion point belongs to a pipeline strand.
  bool in_strand_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_TRANSLATION_STAGED_CALLS_H_
