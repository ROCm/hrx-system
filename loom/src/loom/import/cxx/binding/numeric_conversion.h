// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_NUMERIC_CONVERSION_H_
#define LOOM_IMPORT_CXX_BINDING_NUMERIC_CONVERSION_H_

#include <cxx/ast_interpreter.h>
#include <cxx/attributes.h>
#include <cxx/types_fwd.h>

#include <optional>
#include <span>

#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/value/representation.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

class Scalars;
class Vectors;

// Provides C++ constant semantics for declaration-only numeric.convert
// specializations. Runtime calls are resolved independently by the intrinsic.
class NumericConversionConstantEvaluator final
    : public cxx::ConstexprCallProvider {
 public:
  std::optional<cxx::ConstValue> evaluate(
      cxx::ASTInterpreter& interpreter, cxx::FunctionSymbol* function,
      std::span<const cxx::ConstValue> arguments) override;
};

// One explicit numeric conversion over a source scalar, native fixed vector,
// or rank-generic semantic vector. Resolved C++ types retain integer
// signedness after their signless High carriers have been selected.
class NumericConversionIntrinsic {
 public:
  struct Operation {
    bool operator==(const Operation&) const = default;
  };

  // Admits numeric.convert independently of a dependent specialization.
  static std::optional<Operation> admit(cxx::TranslationUnit& unit,
                                        Diagnostics& diagnostics,
                                        const cxx::Attribute& attribute,
                                        cxx::AST* owner);

  // Validates one concrete specialization and retains its source numeric
  // interpretation and value representation.
  static NumericConversionIntrinsic resolve(
      Operation operation, cxx::TranslationUnit& unit, Diagnostics& diagnostics,
      Types& types, const cxx::FunctionType* signature, cxx::AST* owner);

  // Emits the direct High conversion.
  Value call(std::span<const Value> arguments, ValueArena& arena,
             Scalars& scalars, Vectors& vectors, cxx::AST* owner) const;

  bool equivalent(const NumericConversionIntrinsic& other) const {
    return representation_ == other.representation_ &&
           input_type_ == other.input_type_ &&
           output_type_ == other.output_type_ &&
           input_vector_ == other.input_vector_ &&
           output_vector_ == other.output_vector_;
  }

 private:
  enum class Representation { Scalar, NativeVector, SemanticVector };

  NumericConversionIntrinsic(Representation representation,
                             const cxx::Type* input_type,
                             const cxx::Type* output_type,
                             const VectorPartition* input_vector = nullptr,
                             const VectorPartition* output_vector = nullptr)
      : representation_(representation),
        input_type_(input_type),
        output_type_(output_type),
        input_vector_(input_vector),
        output_vector_(output_vector) {}

  // Source value representation selected at specialization admission.
  Representation representation_;
  // Unqualified source type preserving scalar/lane signedness.
  const cxx::Type* input_type_;
  // Unqualified result type preserving scalar/lane signedness.
  const cxx::Type* output_type_;
  // Rank-generic source partition, present only for SemanticVector.
  const VectorPartition* input_vector_;
  // Rank-generic result partition, present only for SemanticVector.
  const VectorPartition* output_vector_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_NUMERIC_CONVERSION_H_
