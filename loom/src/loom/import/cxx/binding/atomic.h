// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_ATOMIC_H_
#define LOOM_IMPORT_CXX_BINDING_ATOMIC_H_

#include <cxx/ast_fwd.h>
#include <cxx/attributes.h>
#include <cxx/symbols_fwd.h>

#include <optional>
#include <span>
#include <string_view>

#include "loom/import/cxx/value/storage.h"
#include "loom/ops/atomic.h"

namespace loom::cxx_import {

// An admitted scalar integer atomic through a source pointer. Leading function
// template arguments select ordering/scope for load/store, kind/ordering/scope
// for RMW/reduction, or success/failure/scope for compare-exchange. The
// concrete signature supplies the integer type. Calls retain the pointer's
// allocation and byte origin.
class AtomicIntrinsic {
 public:
  static bool supports(std::string_view name);

  // Unknown operations return nullopt. Invalid signatures and selectors
  // diagnose at owner and throw SourceRejected. The translation unit owns the
  // retained element type and must outlive this binding.
  static std::optional<AtomicIntrinsic> resolve(cxx::TranslationUnit& unit,
                                                Diagnostics& diagnostics,
                                                Types& types,
                                                cxx::FunctionSymbol* function,
                                                const cxx::Attribute& attribute,
                                                cxx::AST* owner);

  // Emits one atomic access using already evaluated source arguments and the
  // ordinary storage projection. Store and reduction have no result; load,
  // RMW, and CAS return the observed memory value. Target lowering owns width,
  // alignment, and scope support; no target policy is inferred from the pointer
  // here.
  std::optional<Value> call(std::span<const Value> arguments, Storage& storage,
                            cxx::AST* owner, loom_builder_t* builder,
                            loom_location_id_t location) const;

  bool equivalent(const AtomicIntrinsic& other) const;

 private:
  friend class AtomicBuiltin;

  enum class Operation { Load, Store, Rmw, Reduce, CompareExchange };

  AtomicIntrinsic(Operation operation, const cxx::Type* element_type,
                  loom_type_t type, loom_atomic_kind_t kind,
                  loom_atomic_ordering_t ordering,
                  loom_atomic_ordering_t failure_ordering,
                  loom_atomic_scope_t scope)
      : operation_(operation),
        element_type_(element_type),
        type_(type),
        kind_(kind),
        ordering_(ordering),
        failure_ordering_(failure_ordering),
        scope_(scope) {}

  // Admitted operation family.
  Operation operation_;
  // Source-owned pointee type, including const/volatile qualification.
  const cxx::Type* element_type_;
  // Signless High representation of the integer payload.
  loom_type_t type_;
  // Integer combining operation; unused for load, store, and compare-exchange.
  loom_atomic_kind_t kind_;
  // Access ordering or successful compare-exchange ordering.
  loom_atomic_ordering_t ordering_;
  // Failed compare-exchange ordering; relaxed for the other operations.
  loom_atomic_ordering_t failure_ordering_;
  // Explicit source synchronization scope, preserved without narrowing.
  loom_atomic_scope_t scope_;
};

// GCC argument and result conventions over the same typed High projection.
// The parser owns the concrete signature and argument conversions. Admission
// consumes constant orderings; the driver evaluates all remaining arguments.
class AtomicBuiltin {
 public:
  // Admits integer load_n/store_n/exchange_n/compare_exchange_n and the
  // add/sub/and/or/xor fetch families. Other builtins return nullopt.
  static std::optional<AtomicBuiltin> resolve(cxx::TranslationUnit& unit,
                                              Diagnostics& diagnostics,
                                              Types& types,
                                              cxx::CallExpressionAST* call);

  // Number of leading value operands, including CAS's evaluated weak operand.
  size_t argument_count() const;

  // Arguments are evaluated once before the operation. CAS returns success
  // and writes observed memory into expected storage only on failure. A strong
  // High CAS implements either weak value without spurious failure.
  std::optional<Value> call(std::span<const Value> arguments, Storage& storage,
                            Scalars& scalars, cxx::AST* owner,
                            loom_builder_t* builder,
                            loom_location_id_t location) const;

 private:
  enum class Result { Observed, Updated, Success };

  AtomicBuiltin(AtomicIntrinsic intrinsic, Result result,
                const cxx::Type* expected_type)
      : intrinsic_(intrinsic), result_(result), expected_type_(expected_type) {}

  // The source-selected memory operation, ordering, scope, and payload type.
  AtomicIntrinsic intrinsic_;
  // Whether the builtin returns the observed value, updated value, or success.
  Result result_;
  // Source-owned CAS expected-storage element type; null for other operations.
  const cxx::Type* expected_type_;
};

// A standalone memory fence has only ordering and scope. It carries neither a
// storage operand nor an element type and is valid in ordinary functions.
class FenceIntrinsic {
 public:
  static bool supports(std::string_view name);

  // Admits a void() declaration with leading ordering/scope template arguments.
  // Invalid source signatures and selectors diagnose at owner.
  static std::optional<FenceIntrinsic> resolve(cxx::TranslationUnit& unit,
                                               Diagnostics& diagnostics,
                                               cxx::FunctionSymbol* function,
                                               const cxx::Attribute& attribute,
                                               cxx::AST* owner);

  // Admits GCC thread fences with constant ordering and system scope.
  // A relaxed fence emits no operation; other orderings use buffer.fence.
  static std::optional<FenceIntrinsic> resolve_builtin(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics,
      cxx::CallExpressionAST* call);

  void call(loom_builder_t* builder, loom_location_id_t location) const;

  bool equivalent(const FenceIntrinsic& other) const;

 private:
  FenceIntrinsic(loom_atomic_ordering_t ordering, loom_atomic_scope_t scope)
      : ordering_(ordering), scope_(scope) {}

  // Thread memory ordering, independent of collective execution.
  loom_atomic_ordering_t ordering_;
  // Explicit source synchronization scope, preserved without narrowing.
  loom_atomic_scope_t scope_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_ATOMIC_H_
