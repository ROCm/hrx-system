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
#include <unordered_map>
#include <vector>

namespace loom::cxx_import {

class Types;

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
  // A same-type prvalue return initializes this function's result object.
  void returned(cxx::FunctionSymbol* function, cxx::ExpressionAST* expression);
  // Compresses construction groups before interprocedural demand propagation.
  void finish();

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

  size_t add(const void* key);
  size_t root(size_t index) const;
  void join(size_t first, size_t second);
  void join(const void* destination, cxx::ExpressionAST* source);
  void alias(cxx::ExpressionAST* expression, const void* source);
  void elements(cxx::ExpressionAST* object,
                cxx::List<cxx::ExpressionAST*>* initializers);

  // Source semantic categories and complete object types.
  cxx::TranslationUnit& unit_;
  // Canonical source type normalization; retaining identities needs no layout.
  Types& types_;
  // Source nodes name static construction sites, not dynamic instances.
  std::unordered_map<const void*, size_t> identities_;
  // Disjoint construction groups with retained storage demand.
  std::vector<Group> groups_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_CONTROL_OBJECT_IDENTITY_H_
