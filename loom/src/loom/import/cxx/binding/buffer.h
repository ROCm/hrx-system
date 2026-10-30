// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_BUFFER_H_
#define LOOM_IMPORT_CXX_BINDING_BUFFER_H_

#include <cxx/ast_fwd.h>
#include <cxx/attributes.h>
#include <cxx/symbols_fwd.h>

#include <optional>
#include <span>
#include <string_view>

#include "loom/import/cxx/value/storage.h"

namespace loom::cxx_import {

// An admitted buffer allocation or scoped pointer contract. Allocations retain
// their logical or physical count domain, memory space, and minimum alignment.
// Assumptions refine the buffer root while preserving the source pointer's byte
// origin.
class BufferIntrinsic {
 public:
  static bool supports(std::string_view name);

  // Unknown operations return nullopt. Invalid signatures and selectors
  // diagnose at owner. The translation unit owns the retained source types.
  static std::optional<BufferIntrinsic> resolve(cxx::TranslationUnit& unit,
                                                Diagnostics& diagnostics,
                                                Types& types,
                                                cxx::FunctionSymbol* function,
                                                const cxx::Attribute& attribute,
                                                cxx::AST* owner);

  // Emits the selected buffer operation after arguments have been evaluated
  // using normal C++ rules.
  Value call(std::span<const Value> arguments, ValueArena& arena,
             Storage& storage, cxx::AST* owner) const;

  // Literal bindings consume source bytes before runtime argument evaluation.
  bool has_literal() const {
    return operation_ == Operation::Constant ||
           operation_ == Operation::AllocaInit;
  }
  Value call_literal(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                     cxx::ExpressionAST* argument, Storage& storage,
                     cxx::AST* owner) const;

  bool equivalent(const BufferIntrinsic& other) const;

 private:
  enum class Operation {
    Constant,
    AllocaInit,
    AllocaBytes,
    AllocaElements,
    AssumeAlignment,
    AssumeMemorySpace,
    AssumeNoalias,
    Cast,
    Root,
  };

  BufferIntrinsic(Operation operation,
                  loom_value_fact_memory_space_t memory_space,
                  int64_t alignment, bool is_utf8)
      : operation_(operation),
        is_utf8_(is_utf8),
        memory_space_(memory_space),
        alignment_(alignment) {}
  BufferIntrinsic(const cxx::Type* element_type,
                  const cxx::Type* element_count_type,
                  loom_value_fact_memory_space_t memory_space,
                  int64_t alignment, bool has_pool)
      : operation_(Operation::AllocaElements),
        element_type_(element_type),
        element_count_type_(element_count_type),
        memory_space_(memory_space),
        alignment_(alignment),
        has_pool_(has_pool) {}
  BufferIntrinsic(loom_value_fact_memory_space_t memory_space,
                  int64_t alignment, bool has_pool)
      : operation_(Operation::AllocaBytes),
        memory_space_(memory_space),
        alignment_(alignment),
        has_pool_(has_pool) {}
  explicit BufferIntrinsic(const cxx::Type* element_type)
      : operation_(Operation::AssumeNoalias), element_type_(element_type) {}
  BufferIntrinsic(const cxx::Type* element_type, int64_t alignment)
      : operation_(Operation::AssumeAlignment),
        element_type_(element_type),
        alignment_(alignment) {}
  BufferIntrinsic(const cxx::Type* element_type,
                  loom_value_fact_memory_space_t memory_space)
      : operation_(Operation::AssumeMemorySpace),
        element_type_(element_type),
        memory_space_(memory_space) {}
  BufferIntrinsic(Operation operation, const BufferPartition* result_buffer)
      : operation_(operation), result_buffer_(result_buffer) {}

  // Buffer operation selected by the source declaration.
  Operation operation_;
  // Literal payload form selected by the source binding, never inferred.
  bool is_utf8_ = false;

  // Source object type named by T and by the result pointee.
  const cxx::Type* element_type_ = nullptr;
  // Source size or native index type accepted by the runtime operand.
  const cxx::Type* element_count_type_ = nullptr;
  // Source-only element interpretation produced by buffer.cast.
  const BufferPartition* result_buffer_ = nullptr;
  // High storage domain selected for allocation or refinement.
  loom_value_fact_memory_space_t memory_space_ =
      LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  // Positive power-of-two minimum byte alignment.
  int64_t alignment_ = 0;
  // Whether buffer.alloca takes an explicit allocation pool operand.
  bool has_pool_ = false;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_BUFFER_H_
