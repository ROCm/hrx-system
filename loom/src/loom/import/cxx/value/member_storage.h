// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_MEMBER_STORAGE_H_
#define LOOM_IMPORT_CXX_VALUE_MEMBER_STORAGE_H_

#include <span>
#include <unordered_map>
#include <vector>

#include "loom/import/cxx/value/bindings.h"
#include "loom/import/cxx/value/objects.h"

namespace loom::cxx_import {

// Physical homes of numeric subobjects in otherwise value-transported records.
// A source binding owns its homes for its lifetime. Copying a source value
// reads their contents; passing a source reference borrows the same homes.
class MemberStorage {
 public:
  MemberStorage(Types& types, Objects& objects, Storage& storage,
                ValueArena& arena)
      : types_(types), objects_(objects), storage_(storage), arena_(arena) {}

  struct Member {
    // Source subobject including its enclosing binding's component interval.
    ValueProjection group;
    // Physical home for this activation of the source binding.
    StorageProjection address;
    // Whether a direct initializer has already populated this member at its
    // final address.
    bool initialized;
  };

  std::span<Member> lookup(cxx::Symbol* binding);
  bool empty() const { return bindings_.empty(); }
  void clear() { bindings_.clear(); }

  void allocate(cxx::Symbol* binding, std::span<const ValueProjection> groups,
                cxx::AST* source);
  void bind(cxx::Symbol* binding, ValueProjection group,
            StorageProjection address);

  std::optional<StorageProjection> address(std::span<const Member> members,
                                           size_t offset, const cxx::Type* type,
                                           cxx::AST* source);
  // Reads only the requested source subobject, preserving independent
  // initialization of siblings and using each member's current contents.
  std::optional<Value> read(std::span<const Member> members,
                            ValueBindings& values, Destination destination,
                            const cxx::Type* type, cxx::AST* source);
  std::optional<StorageProjection> begin_construction(Destination destination,
                                                      const cxx::Type* type);
  void write(std::span<Member> members, ValueBindings& values,
             Destination destination, Value value, const cxx::Type* type,
             cxx::AST* source, bool initializing = false);

 private:
  // Source layout and projection catalog for this translation.
  Types& types_;
  // Typed aggregate loads and stores for numeric subobjects.
  Objects& objects_;
  // Physical pointer projection and local allocation.
  Storage& storage_;
  // Immutable source values captured at reads.
  ValueArena& arena_;
  // Present only for source bindings with borrowed numeric members.
  std::unordered_map<cxx::Symbol*, std::vector<Member>> bindings_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_MEMBER_STORAGE_H_
