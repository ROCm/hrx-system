// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_FUNCTION_CONTRACTS_H_
#define LOOM_IMPORT_CXX_BINDING_FUNCTION_CONTRACTS_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "loom/import/cxx/binding/target_requirements.h"
#include "loom/import/cxx/source/source.h"
#include "loom/ir/attribute.h"

namespace loom::cxx_import {

class Types;

// Selects the native transport shape whose identities back one source
// function contract. Ordinary callables flatten source values; kernel launch
// bindings retain one native value per source parameter.
enum class FunctionContractSignature {
  Flattened,
  Kernel,
};

// Owns source callable preconditions independently of parameter spellings.
// Declaration admission resolves semantic bindings to source ordinals once;
// definition construction substitutes the operation's reserved SSA identities
// without rescanning source or native IR.
class FunctionContracts {
 public:
  FunctionContracts(cxx::TranslationUnit& unit, Diagnostics& diagnostics)
      : unit_(unit), diagnostics_(diagnostics) {}

  // Admits a selected trailing function-prototype attribute and reconciles it
  // with any previous constrained redeclaration. Template patterns are
  // admitted after frontend substitution produces a concrete declaration.
  void declaration(cxx::FunctionSymbol* function,
                   cxx::FunctionDeclaratorChunkAST* prototype);

  // Returns whether this callable has a nonempty native predicate contract.
  bool has_predicates(cxx::FunctionSymbol* function) const;

  // Returns whether this callable has target applicability requirements.
  bool has_requirements(cxx::FunctionSymbol* function) const;

  // Substitutes flattened argument/result or kernel argument identities into
  // the admitted predicate templates. The identities must be reserved for the
  // immediately following callable build and remain owned by the builder.
  std::vector<loom_predicate_t> bind(
      cxx::FunctionSymbol* function, Types& types,
      std::span<const loom_value_id_t> identities,
      FunctionContractSignature signature, cxx::AST* owner) const;

  // Materializes admitted target applicability requirements in |module|.
  std::vector<loom_attribute_t> bind_requirements(cxx::FunctionSymbol* function,
                                                  loom_module_t* module) const;

 private:
  enum class RootKind {
    Parameter,
    Result,
  };

  struct Value {
    // Source-root category independent of any particular redeclaration name.
    RootKind root;
    // Parameter ordinal for Parameter; ignored for Result.
    size_t parameter_ordinal;
    // Direct semantic member path from the root to one scalar component.
    std::vector<cxx::Symbol*> members;
    // Root source type used to validate typed result placeholders.
    const cxx::Type* root_type;
    // Scalar leaf type before frontend-selected implicit conversions.
    const cxx::Type* source_type;
    // Scalar operand type after frontend-selected implicit conversions.
    const cxx::Type* converted_type;
    // Original scalar leaf owning late representation diagnostics.
    cxx::ExpressionAST* source;
  };

  struct Predicate {
    // Semantic values referenced by VALUE arguments, in ordinal order.
    std::array<Value, 3> values;
    // Number of active entries in values.
    uint8_t value_count;
    // Predicate template whose VALUE payloads index values.
    loom_predicate_t predicate;
  };

  struct Contract {
    // Conjunction in the first constrained declaration's source order.
    std::vector<Predicate> predicates;
    // Target conditions in the first constrained declaration's source order.
    std::vector<ProjectedTargetRequirement> requirements;
  };

  static bool predicate_equal(const Predicate& left, const Predicate& right);
  static bool contract_equal(const Contract& left, const Contract& right);

  // Frontend semantic identities and type relationships.
  cxx::TranslationUnit& unit_;
  // Source admission and late representation failure boundary.
  Diagnostics& diagnostics_;
  // Explicit contracts indexed by canonical function identity.
  std::unordered_map<cxx::FunctionSymbol*, Contract> contracts_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_FUNCTION_CONTRACTS_H_
