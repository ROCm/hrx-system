// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_ASSEMBLY_H_
#define LOOM_IMPORT_CXX_BINDING_ASSEMBLY_H_

#include <cxx/ast_fwd.h>
#include <cxx/attributes.h>
#include <cxx/symbols_fwd.h>

#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "loom/error/source.h"
#include "loom/import/cxx/source/options.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

class SymbolNames;

// Retains admitted fragments and their borrowed source snapshots. Verification
// runs once after translation, before importer-owned IR invariants are checked.
class AssemblyFragments {
 public:
  void record(loom_func_like_t function, loom_source_entry_t source);
  void verify(loom_module_t* module, loom_diagnostic_sink_t sink) const;

 private:
  // Module-owned functions appended by literal admission, in source order.
  std::vector<loom_func_like_t> functions_;
  // Borrowed source bytes indexed by module source identity; gaps are empty.
  std::vector<loom_source_entry_t> sources_;
};

// One admitted semantic signature for embedded descriptor-backed assembly.
// The C++ signature owns semantic types; each literal owns physical register
// types. Both survive as the ordinary low.invoke/low.func.def boundary.
class AssemblyIntrinsic {
 public:
  static bool supports(std::string_view name) { return name == "low.assembly"; }

  static std::optional<AssemblyIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
      cxx::AST* owner);

  // Admits a raw literal once, owning its parsed IR in |module|. The input
  // AST/source remain borrowed only for this import invocation. Retains the
  // function in |fragments| for semantic verification after translation.
  // Source errors diagnose and throw SourceRejected; infrastructure failures
  // throw StatusError.
  loom_symbol_ref_t fragment(cxx::TranslationUnit& unit,
                             Diagnostics& diagnostics,
                             cxx::ExpressionAST* source, SymbolNames& names,
                             AssemblyFragments& fragments,
                             loom_module_t* module,
                             const loom_cxx_import_options_t& options);

  // Emits the already-admitted fragment after flattening each static source
  // value through its ordinary call-boundary representation.
  std::optional<Value> call(loom_symbol_ref_t fragment,
                            std::span<const Value> arguments, ValueArena& arena,
                            loom_builder_t* builder,
                            loom_location_id_t location) const;

  bool equivalent(const AssemblyIntrinsic& other) const;

 private:
  AssemblyIntrinsic(std::string_view contract, size_t argument_count,
                    const Partition* result_partition,
                    std::vector<loom_type_t> result_types)
      : contract_(contract),
        argument_count_(argument_count),
        result_partition_(result_partition),
        result_types_(std::move(result_types)) {}

  // Frontend-owned stable contract key, valid for this import invocation.
  std::string_view contract_;
  // Flattened semantic operands following the source literal.
  size_t argument_count_;
  // Admitted static source result, or null for a void signature.
  const Partition* result_partition_;
  // Flattened semantic result types in source transport order.
  std::vector<loom_type_t> result_types_;
  // Admitted literal identities and their module-owned private functions.
  std::unordered_map<cxx::StringLiteralExpressionAST*, loom_symbol_ref_t>
      fragments_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_ASSEMBLY_H_
