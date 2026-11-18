// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_CONTROL_OBJECT_IDENTITY_H_
#define LOOM_IMPORT_CXX_CONTROL_OBJECT_IDENTITY_H_

#include <cxx/ast_fwd.h>
#include <cxx/cxx_fwd.h>
#include <cxx/symbols_fwd.h>

#include <cstddef>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "loom/import/cxx/value/bindings.h"
#include "loom/import/cxx/value/partition.h"

namespace loom::cxx_import {

class Types;

// A subobject selected from one evaluated complete prvalue.
struct TemporaryMember {
  // Expression producing the complete object, including its side effects.
  cxx::ExpressionAST* object;
  // Member components relative to that complete object's value.
  ValueProjection group;
};

// Storage requirements of owned source objects. The owning AST walk joins
// prvalues with their direct initialization destinations and containing
// subobjects. A group shares storage demand, not an address: member offsets
// still distinguish subobjects. Copies keep distinct groups. Callable
// propagation only sets a group's monotone bit; it never revisits expressions.
class ObjectIdentity {
 public:
  ObjectIdentity(cxx::TranslationUnit& unit, Types& types)
      : unit_(unit), types_(types) {}

  // Retains one evaluated expression after its children. Non-object values
  // create no nodes; transparent syntax and subobject access reuse a group.
  void retain(cxx::ExpressionAST* expression);
  // Declares a complete owned object, optionally sharing its prvalue
  // initializer. The source constructor, when present, decides whether
  // initialization copies a distinct object or constructs this one directly.
  void bind(cxx::Symbol* binding, cxx::ExpressionAST* initializer,
            bool storage);
  // The reference declaration owns the complete lifetime-extended temporary,
  // even when the reference itself denotes one of its subobjects.
  void own_members(cxx::Symbol* binding, cxx::ExpressionAST* origin);
  // Complete prvalue and member geometry for a retained xvalue selection.
  const TemporaryMember* temporary_member(cxx::ExpressionAST* expression) const;
  // A same-type prvalue return initializes this function's result object.
  void returned(cxx::FunctionSymbol* function, cxx::ExpressionAST* expression);
  // Compresses construction groups before interprocedural demand propagation.
  void finish();

  struct MemberConstruction {
    // Local aggregate whose lifetime owns the constructed member.
    cxx::Symbol* binding;
    // First component relative to the complete aggregate.
    size_t offset;
    // Prvalue that directly initializes this numeric subobject.
    cxx::ExpressionAST* initializer;
  };
  std::span<const MemberConstruction> member_constructions() const;
  const Destination* aggregate_destination(
      cxx::ExpressionAST* initializer) const;
  const MemberConstruction* member_construction(
      cxx::ExpressionAST* initializer) const;

  // Keys are live source AST or symbol identities from this owning function.
  // Unknown keys have no owned source object and therefore no storage demand.
  bool requires_storage(const void* key) const;
  bool require_storage(const void* key);

 private:
  struct Group {
    // Canonical construction group; self for a representative.
    size_t parent;
    // Union-by-size weight, meaningful only on a representative.
    size_t size = 1;
    // Whether the complete object must retain physical source identity.
    bool storage = false;
  };

  struct Element {
    // Construction group of the enclosing aggregate.
    size_t parent;
    // Construction group of the direct-initialized element.
    size_t child;
    // First source component relative to the enclosing aggregate.
    size_t offset;
    // Exact direct-initialization clause.
    cxx::ExpressionAST* initializer;
  };
  struct AggregateMembers {
    // Containment edges independent of complete-object identity.
    std::vector<Element> elements;
    // Named owners of directly initialized value-transported aggregates.
    std::vector<std::pair<size_t, cxx::Symbol*>> bindings;
    // Resolved numeric construction destinations, in source traversal order.
    std::vector<MemberConstruction> constructions;
    // Owning source binding and component origin of each mixed construction
    // group.
    std::unordered_map<size_t, Destination> destinations;
    // Exact initializer-to-destination index for in-place construction.
    std::unordered_map<cxx::ExpressionAST*, size_t> by_initializer;
  };
  void finish_members();

  size_t add(const void* key);
  size_t root(size_t index) const;
  void join(size_t first, size_t second);
  void join(const void* destination, cxx::ExpressionAST* source);
  void alias(cxx::ExpressionAST* expression, const void* source);
  bool retain_temporary_member(cxx::MemberExpressionAST* member);
  void elements(cxx::ExpressionAST* object,
                cxx::List<cxx::ExpressionAST*>* initializers);

  // Source semantic categories and complete object types.
  cxx::TranslationUnit& unit_;
  // Canonical source types and subobject geometry for member projections.
  Types& types_;
  // Source nodes name static construction sites, not dynamic instances.
  std::unordered_map<const void*, size_t> identities_;
  // Disjoint construction groups with retained storage demand.
  std::vector<Group> groups_;
  using TemporaryMembers =
      std::unordered_map<cxx::ExpressionAST*, TemporaryMember>;
  // Only member xvalues selected from otherwise value-transported prvalues.
  std::unique_ptr<TemporaryMembers> temporary_members_;
  // Present only when aggregate initialization crosses a numeric storage
  // boundary.
  std::unique_ptr<AggregateMembers> members_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_CONTROL_OBJECT_IDENTITY_H_
