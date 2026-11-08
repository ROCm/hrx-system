// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/objects.h"

#include <cxx/symbols.h>
#include <cxx/types.h>

#include "loom/import/cxx/source/objects.h"

namespace loom::cxx_import {

const cxx::Type* Objects::member_type(const cxx::Type* object_type,
                                      cxx::Symbol* field) const {
  auto traits = unit_.typeTraits();
  return traits.is_volatile(object_type)
             ? traits.add_volatile(source_object_type(field))
             : source_object_type(field);
}

void Objects::load_components(StorageProjection object, const cxx::Type* type,
                              const Partition& partition, cxx::AST* owner,
                              std::vector<loom_value_id_t>& components) {
  if (partition.kind == ValueKind::Record) {
    const auto& record = static_cast<const RecordPartition&>(partition);
    for (const auto& member : record.members) {
      load_components(storage_.member(object, member.source, owner),
                      member_type(type, member.source), *member.partition,
                      owner, components);
    }
  } else if (partition.kind == ValueKind::Array) {
    const auto& array = static_cast<const ArrayPartition&>(partition);
    for (size_t index = 0; index < array.source->size(); ++index) {
      load_components(storage_.element(object, array.source, index, owner),
                      unit_.typeTraits().get_element_type(type), *array.element,
                      owner, components);
    }
  } else {
    components.push_back(
        storage_.load(storage_.dereference(object, type, owner), type, owner));
  }
}

Value Objects::load(StorageProjection object, const cxx::Type* type,
                    cxx::AST* owner) {
  const auto& partition = types_.partition(types_.unqualified(type), owner);
  if (partition.kind == ValueKind::SSA) {
    return storage_.load(storage_.dereference(object, type, owner), type,
                         owner);
  }
  std::vector<loom_value_id_t> components;
  components.reserve(partition.component_count);
  load_components(object, type, partition, owner, components);
  return arena_.capture(partition, components);
}

void Objects::store(StorageProjection object, Value value,
                    const cxx::Type* type, cxx::AST* owner) {
  const auto& partition = value.partition();
  if (partition.kind == ValueKind::Record) {
    const auto& record = static_cast<const RecordPartition&>(partition);
    for (const auto& member : record.members) {
      store(storage_.member(object, member.source, owner),
            value.project(*member.partition, member.component_offset),
            member_type(type, member.source), owner);
    }
  } else if (partition.kind == ValueKind::Array) {
    const auto& array = static_cast<const ArrayPartition&>(partition);
    for (size_t index = 0; index < array.source->size(); ++index) {
      store(
          storage_.element(object, array.source, index, owner),
          value.project(*array.element, index * array.element->component_count),
          unit_.typeTraits().get_element_type(type), owner);
    }
  } else {
    storage_.store(storage_.dereference(object, type, owner), value.ssa(), type,
                   owner);
  }
}

}  // namespace loom::cxx_import
