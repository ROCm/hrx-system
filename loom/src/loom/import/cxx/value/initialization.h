// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_INITIALIZATION_H_
#define LOOM_IMPORT_CXX_VALUE_INITIALIZATION_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>

#include <array>
#include <optional>
#include <span>

#include "loom/import/cxx/source/constants.h"
#include "loom/import/cxx/value/objects.h"
#include "loom/import/cxx/value/vector.h"

namespace loom::cxx_import {

// Source aggregate initialization has two destinations: immutable components
// and addressable objects. In-place initialization publishes each completed
// subobject before evaluating the next clause; value copies capture all source
// components before any destination store. The resolved AST supplies clause
// ordering, defaults and conversions, independently of the target ABI.
class Initialization {
 public:
  // Borrowed source-expression evaluator. Initializer traversal controls when
  // each clause executes; the surrounding translator owns its expression and
  // conversion semantics. This interface owns no source or output state.
  class Evaluation {
   public:
    virtual Value expression(cxx::ExpressionAST* expression) = 0;
    virtual Value convert(cxx::ExpressionAST* expression, const cxx::Type* type,
                          cxx::AST* owner) = 0;
    // Captures the durable storage identity selected by one reference
    // initializer as the existing buffer-plus-offset pointer value.
    virtual Value reference(cxx::ExpressionAST* expression,
                            cxx::AST* owner) = 0;
    virtual void effect(cxx::ExpressionAST* expression) = 0;
    // Installs one source current-object scope. Value receivers publish each
    // completed field into the ordinary symbol environment; storage receivers
    // retain the destination identity used by field lvalues.
    virtual void enter_value_receiver(const RecordPartition& record) = 0;
    virtual void enter_storage_receiver(const RecordPartition& record,
                                        StorageProjection destination,
                                        const cxx::Type* object_type) = 0;
    virtual void publish_receiver_member(cxx::FieldSymbol* field,
                                         Value value) = 0;
    virtual void complete_storage_receiver_member(cxx::FieldSymbol* field) = 0;
    virtual Value receiver_member(cxx::FieldSymbol* field, cxx::AST* owner) = 0;
    virtual void leave_receiver() = 0;
    // Branch construction retains the translator's automatic-binding joins.
    virtual void conditional_object(StorageProjection destination,
                                    const cxx::Type* type,
                                    cxx::ConditionalExpressionAST* expression,
                                    bool retained) = 0;
    // Invokes one user-provided constructor with the exact destination object
    // and explicit source arguments. The constructor definition owns member
    // initialization followed by ordinary body translation.
    virtual void call_constructor(
        StorageProjection destination, const RecordPartition& record,
        const cxx::Type* object_type, cxx::FunctionSymbol* constructor,
        std::span<cxx::ExpressionAST* const> arguments, cxx::AST* owner) = 0;

   protected:
    ~Evaluation() = default;
  };

  Initialization(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                 Types& types, Scalars& scalars, Vectors& vectors,
                 Storage& storage, Objects& objects, ValueArena& arena,
                 Locations& locations, loom_builder_t& builder,
                 Evaluation& evaluation)
      : unit_(unit),
        diagnostics_(diagnostics),
        types_(types),
        scalars_(scalars),
        vectors_(vectors),
        storage_(storage),
        objects_(objects),
        arena_(arena),
        locations_(locations),
        builder_(builder),
        evaluation_(evaluation) {
    zero_name_ids_.fill(LOOM_STRING_ID_INVALID);
  }

  // Builds one source value from its complete normalized initializer. The
  // selected constructor is retained separately on declarations and implicit
  // array elements that do not use a construction expression.
  Value value(const cxx::Type* type, cxx::ExpressionAST* initializer,
              cxx::FunctionSymbol* constructor, cxx::AST* owner,
              std::optional<Value> retained = std::nullopt);
  // Captures a character-array string initializer using the source ABI's code
  // units, terminator, and zero fill. Other initializer forms return no value.
  std::optional<ObjectRepresentation> literal(const cxx::Type* type,
                                              cxx::ExpressionAST* initializer,
                                              cxx::AST* owner);
  // Initializes an already allocated object from a present source initializer.
  // Array scalar elements keep the allocation's retained typed view.
  void storage(StorageAllocation allocation, const cxx::Type* type,
               cxx::ExpressionAST* initializer,
               cxx::FunctionSymbol* constructor, cxx::AST* owner);
  // Constructs a projected object, including one selected initializer arm.
  // The caller has already allocated and admitted the destination storage.
  void object(StorageProjection destination, const cxx::Type* type,
              cxx::ExpressionAST* initializer, cxx::FunctionSymbol* constructor,
              cxx::AST* owner, bool retained = false);
  // Applies the frontend-owned member initialization plan at the entry of a
  // retained user-provided constructor. The ordinary function translator then
  // executes its compound body against the same storage receiver.
  void constructor_receiver(StorageProjection destination,
                            const RecordPartition& record,
                            const cxx::Type* object_type,
                            cxx::FunctionSymbol* constructor, cxx::AST* owner);

 private:
  Value from_elements(const cxx::Type* type,
                      cxx::List<cxx::ExpressionAST*>* elements,
                      cxx::VariableSymbol* implicit_element,
                      cxx::FunctionSymbol* constructor, cxx::AST* owner,
                      std::optional<Value> retained);
  // Constructs a native vector from its complete source initializer. Pure
  // constant lanes are retained as frontend values so the producer can choose
  // between one typed literal with static patches and from_elements without
  // changing live-expression evaluation order.
  loom_value_id_t initialize_vector(const cxx::Type* type,
                                    cxx::List<cxx::ExpressionAST*>* elements,
                                    cxx::AST* owner);
  // Returns the existing one-operation literal path when every source lane is
  // a pure frontend constant. Mixed analysis is kept off this common path.
  std::optional<loom_value_id_t> constant_vector(
      const cxx::Type* type, cxx::List<cxx::ExpressionAST*>* elements,
      cxx::AST* owner);
  loom_string_id_t zero_name(loom_scalar_type_t scalar_type);
  Value zero(const cxx::Type* type, cxx::AST* owner);
  Value default_record(const RecordPartition& record,
                       cxx::FunctionSymbol* constructor, bool zero_initialize,
                       cxx::AST* owner, std::optional<Value> retained);
  Value finish_value_receiver(const RecordPartition& record, cxx::AST* owner);
  void default_storage_record(StorageProjection destination,
                              const RecordPartition& record,
                              const cxx::Type* object_type,
                              cxx::FunctionSymbol* constructor,
                              bool zero_initialize, bool retained,
                              cxx::AST* owner);

  // Source type traits retain qualifiers on memory subobjects.
  cxx::TranslationUnit& unit_;
  // Source admission failures retain their owning clause.
  Diagnostics& diagnostics_;
  // Canonical array and record component schemas.
  Types& types_;
  // Scalar zero and array index construction.
  Scalars& scalars_;
  // Explicit vector construction from admitted element clauses.
  Vectors& vectors_;
  // Source-layout memory operations and allocation projections.
  Storage& storage_;
  // Whole-object value snapshots and subobject projections.
  Objects& objects_;
  // Immutable component storage scoped to the translated function.
  ValueArena& arena_;
  // Source provenance attached to initialization operations.
  Locations& locations_;
  // Borrowed insertion point owned by the surrounding translator.
  loom_builder_t& builder_;
  // Borrowed evaluator outlives all initializer traversal.
  Evaluation& evaluation_;
  // Lazily interned compiler names distinguish simultaneous scalar zero types.
  std::array<loom_string_id_t, LOOM_SCALAR_TYPE_COUNT_> zero_name_ids_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_INITIALIZATION_H_
