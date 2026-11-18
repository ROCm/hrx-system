// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/objects.h"

#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/locations.h"
#include "loom/import/cxx/source/objects.h"
#include "loom/ops/ref/ops.h"
#include "loom/ops/view/ops.h"

namespace loom::cxx_import {
namespace {

bool is_owned_reference(loom_type_t type) {
  return loom_type_reference_ownership(type) == LOOM_REFERENCE_OWNERSHIP_OWNED;
}

}  // namespace

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

ReferenceStorageProjection Objects::allocate_reference(
    const ReferenceStorageLayout& layout, Value initial, cxx::AST* owner) {
  IREE_ASSERT_EQ(&initial.partition(), layout.value_partition);
  auto storage = std::make_shared<ReferenceStorage>();
  storage->layout = &layout;
  storage->cells.reserve(layout.cell_types.size());
  for (auto cell_type : layout.cell_types) {
    loom_op_t* op;
    check(loom_view_alloca_build(
        &builder_, 0, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE,
        LOOM_VALUE_ID_INVALID, 0, cell_type, locations_.get(owner), &op));
    storage->cells.push_back(loom_view_alloca_result(op));
  }
  ReferenceStorageProjection result{storage, layout.value_partition, 0};
  store(result, initial, owner);
  return result;
}

ReferenceStorageProjection Objects::admit_reference(
    const ReferenceStorageLayout& layout,
    std::span<const loom_value_id_t> cells) {
  IREE_ASSERT_EQ(cells.size(), layout.cell_types.size());
  auto storage = std::make_shared<ReferenceStorage>();
  storage->layout = &layout;
  storage->cells.assign(cells.begin(), cells.end());
  return {storage, layout.value_partition, 0};
}

ReferenceStorageProjection Objects::project_reference(
    ReferenceStorageProjection object, const Partition& partition,
    size_t component_offset) const {
  IREE_ASSERT_LE(component_offset + partition.component_count,
                 object.partition->component_count);
  object.component_offset += component_offset;
  object.partition = &partition;
  return object;
}

Value Objects::load(ReferenceStorageProjection object, cxx::AST* owner) {
  const auto& layout = *object.storage->layout;
  std::vector<loom_value_id_t> components;
  components.reserve(object.partition->component_count);
  const int64_t static_index = 0;
  for (size_t index = 0; index < object.partition->component_count; ++index) {
    const size_t component = object.component_offset + index;
    loom_op_t* op;
    check(loom_view_load_build(
        &builder_, 0, 0, object.storage->cells[component], nullptr, 0,
        &static_index, 1, 0, 0, layout.element_types[component],
        locations_.get(owner), &op));
    components.push_back(loom_view_load_result(op));
  }
  return arena_.capture(*object.partition, components);
}

void Objects::store(ReferenceStorageProjection object, Value value,
                    cxx::AST* owner) {
  IREE_ASSERT_EQ(&value.partition(), object.partition);
  auto components = value.components();
  const int64_t static_index = 0;
  for (size_t index = 0; index < components.size(); ++index) {
    const size_t component = object.component_offset + index;
    loom_op_t* op;
    check(loom_view_store_build(
        &builder_, 0, 0, components[index], object.storage->cells[component],
        nullptr, 0, &static_index, 1, 0, 0, locations_.get(owner), &op));
    if (is_owned_reference(object.storage->layout->element_types[component])) {
      check(loom_ref_release_build(&builder_, components[index],
                                   locations_.get(owner), &op));
    }
  }
}

Value Objects::take(ReferenceStorageProjection object, cxx::AST* owner) {
  auto value = load(object, owner);
  const auto& layout = *object.storage->layout;
  const int64_t static_index = 0;
  for (size_t index = 0; index < object.partition->component_count; ++index) {
    const size_t component = object.component_offset + index;
    loom_op_t* op;
    check(loom_ref_null_build(&builder_, layout.element_types[component],
                              locations_.get(owner), &op));
    const auto empty = loom_ref_null_result(op);
    check(loom_view_store_build(
        &builder_, 0, 0, empty, object.storage->cells[component], nullptr, 0,
        &static_index, 1, 0, 0, locations_.get(owner), &op));
    if (is_owned_reference(layout.element_types[component])) {
      check(
          loom_ref_release_build(&builder_, empty, locations_.get(owner), &op));
    }
  }
  return value;
}

}  // namespace loom::cxx_import
