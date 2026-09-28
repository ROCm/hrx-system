// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer/type_index.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/parameterized_type.h"

namespace loom {
namespace {

class TypeIndexTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("types"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_type_id_t Intern(loom_type_t type) {
    loom_type_id_t id = LOOM_TYPE_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_type_id(module_, type, &id));
    return id;
  }

  // Canonical repeated children model the DAGs produced by bytecode reading.
  loom_type_id_t InternFunction(loom_type_id_t child, uint16_t fanout) {
    alignas(loom_func_type_data_t) unsigned char
        storage[sizeof(loom_func_type_data_t) + 2 * sizeof(loom_type_t)] = {};
    auto* data = reinterpret_cast<loom_func_type_data_t*>(storage);
    data->arg_count = fanout;
    for (uint16_t i = 0; i < fanout; ++i) {
      data->types[i] = loom_type_table_get(&module_->types, child);
    }
    const loom_type_id_t children[] = {child, child};
    loom_type_id_t id = LOOM_TYPE_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_topological_type_id(
        module_, loom_type_function(data), children, fanout, &id));
    return id;
  }

  // Pool owning source and invocation scratch allocations.
  iree_arena_block_pool_t pool_;
  // Invocation scratch kept alive through all queries.
  iree_arena_allocator_t arena_;
  // Context shared by the source module.
  loom_context_t context_;
  // Immutable source module after index initialization.
  loom_module_t* module_ = nullptr;
};

TEST_F(TypeIndexTest, EmptyModuleAllocatesNothing) {
  const auto storage_before = arena_.used_allocation_size;
  loom_bytecode_type_index_t index;
  IREE_ASSERT_OK(loom_bytecode_type_index_initialize(module_, &arena_, &index));
  EXPECT_EQ(index.nodes, nullptr);
  EXPECT_EQ(index.slots, nullptr);
  EXPECT_EQ(index.dependencies, nullptr);
  EXPECT_EQ(index.stack, nullptr);
  EXPECT_EQ(arena_.used_allocation_size, storage_before);
  EXPECT_EQ(loom_bytecode_type_index_lookup(
                &index, loom_type_scalar(LOOM_SCALAR_TYPE_F32)),
            LOOM_TYPE_ID_INVALID);
}

TEST_F(TypeIndexTest, SharedDependenciesRetainCanonicalIds) {
  std::vector<loom_type_id_t> types = {
      Intern(loom_type_scalar(LOOM_SCALAR_TYPE_F32))};
  for (int level = 0; level < 256; ++level) {
    types.push_back(InternFunction(types.back(), 2));
  }
  const auto source_storage = module_->arena.used_allocation_size;
  loom_bytecode_type_index_t index;
  IREE_ASSERT_OK(loom_bytecode_type_index_initialize(module_, &arena_, &index));
  for (size_t i = 1; i < types.size(); ++i) {
    EXPECT_EQ(loom_bytecode_type_index_lookup(
                  &index, loom_type_table_get(&module_->types, types[i])),
              types[i]);
    const auto& node = index.nodes[types[i]];
    ASSERT_EQ(node.dependencies.count, 2u);
    EXPECT_EQ(node.dependencies.explicit_count, 2u);
    EXPECT_FALSE(node.has_bindings);
    EXPECT_EQ(index.dependencies[node.dependencies.begin], types[i - 1]);
    EXPECT_EQ(index.dependencies[node.dependencies.begin + 1], types[i - 1]);
  }
  EXPECT_EQ(loom_bytecode_type_index_lookup(
                &index, loom_type_scalar(LOOM_SCALAR_TYPE_F16)),
            LOOM_TYPE_ID_INVALID);
  EXPECT_EQ(module_->arena.used_allocation_size, source_storage);
  EXPECT_EQ(module_->types.count, types.size());
}

TEST_F(TypeIndexTest, ShapedTypesRetainTheirScalarDependency) {
  const loom_type_kind_t kinds[] = {LOOM_TYPE_TILE, LOOM_TYPE_TENSOR,
                                    LOOM_TYPE_VECTOR, LOOM_TYPE_VIEW};
  loom_type_id_t types[IREE_ARRAYSIZE(kinds)];
  for (size_t i = 0; i < IREE_ARRAYSIZE(kinds); ++i) {
    types[i] = Intern(loom_type_shaped_1d(kinds[i], LOOM_SCALAR_TYPE_F32, 8,
                                          /*encoding_id=*/0));
  }
  loom_bytecode_type_index_t index;
  IREE_ASSERT_OK(loom_bytecode_type_index_initialize(module_, &arena_, &index));
  const auto scalar = loom_module_lookup_type_id(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_F32));
  ASSERT_NE(scalar, LOOM_TYPE_ID_INVALID);
  for (auto type : types) {
    const auto* node = &index.nodes[type];
    ASSERT_EQ(node->dependencies.count, 1u);
    EXPECT_EQ(node->dependencies.explicit_count, 0u);
    EXPECT_EQ(index.dependencies[node->dependencies.begin], scalar);
  }
}

TEST_F(TypeIndexTest, CanonicalIdentityPreservesScopedBindingsThroughChildren) {
  const loom_type_t dimension_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t dimensions[2];
  IREE_ASSERT_OK(
      loom_module_define_value(module_, dimension_type, &dimensions[0]));
  IREE_ASSERT_OK(
      loom_module_define_value(module_, dimension_type, &dimensions[1]));
  std::vector<loom_type_id_t> first;
  std::vector<loom_type_id_t> second;
  for (uint32_t i = 0; i < 2; ++i) {
    auto& types = i == 0 ? first : second;
    types.push_back(
        Intern(loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                                   loom_dim_pack_dynamic(dimensions[i]),
                                   /*encoding_id=*/0)));
    for (int level = 0; level < 64; ++level) {
      types.push_back(InternFunction(types.back(), 2));
    }
  }
  loom_bytecode_type_index_t index;
  IREE_ASSERT_OK(loom_bytecode_type_index_initialize(module_, &arena_, &index));
  for (size_t i = 0; i < first.size(); ++i) {
    EXPECT_NE(first[i], second[i]);
    EXPECT_EQ(loom_bytecode_type_index_lookup(
                  &index, loom_type_table_get(&module_->types, first[i])),
              first[i]);
    EXPECT_EQ(loom_bytecode_type_index_lookup(
                  &index, loom_type_table_get(&module_->types, second[i])),
              second[i]);
    const auto& first_node = index.nodes[first[i]];
    const auto& second_node = index.nodes[second[i]];
    EXPECT_TRUE(first_node.has_bindings);
    EXPECT_TRUE(second_node.has_bindings);
    if (i > 0) {
      ASSERT_EQ(first_node.dependencies.explicit_count, 2u);
      ASSERT_EQ(second_node.dependencies.explicit_count, 2u);
      EXPECT_EQ(index.dependencies[first_node.dependencies.begin],
                first[i - 1]);
      EXPECT_EQ(index.dependencies[second_node.dependencies.begin],
                second[i - 1]);
    }
  }
}

TEST_F(TypeIndexTest, GeneralConstructionRetainsCanonicalChildren) {
  loom_type_id_t child = Intern(loom_type_scalar(LOOM_SCALAR_TYPE_F32));
  for (int level = 0; level < 6; ++level) {
    child = InternFunction(child, 2);
  }
  const loom_type_t argument = loom_type_table_get(&module_->types, child);
  const loom_type_t result = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t parent;
  IREE_ASSERT_OK(loom_module_intern_function_type(module_, &argument, 1,
                                                  &result, 1, &parent));
  const loom_type_t retained_child = loom_type_func_data(parent)->types[0];
  ASSERT_EQ(loom_type_func_data(retained_child), loom_type_func_data(argument));

  loom_bytecode_type_index_t index;
  IREE_ASSERT_OK(loom_bytecode_type_index_initialize(module_, &arena_, &index));
  EXPECT_EQ(loom_bytecode_type_index_lookup(&index, retained_child), child);
  EXPECT_EQ(loom_bytecode_type_index_lookup(&index, argument), child);
  const auto parent_id = loom_module_lookup_type_id(module_, parent);
  const auto& node = index.nodes[parent_id];
  ASSERT_EQ(node.dependencies.explicit_count, 2u);
  EXPECT_EQ(index.dependencies[node.dependencies.begin], child);
  EXPECT_EQ(index.dependencies[node.dependencies.begin + 1],
            loom_module_lookup_type_id(module_, result));
}

TEST_F(TypeIndexTest, ParameterAttributesRetainTypeDependencyIdentity) {
  static const loom_attr_descriptor_t parameters[] = {{
      /*.name=*/LOOM_BSTRING_REF(8, "metadata"),
      /*.attr_kind=*/LOOM_ATTR_DICT,
  }};
  static const loom_parameterized_type_descriptor_t descriptor = {
      /*.name=*/LOOM_BSTRING_REF(13, "test.metadata"),
      /*.parameter_descriptors=*/parameters,
      /*.ir_kind=*/LOOM_TYPE_PARAMETERIZED,
      /*.type_flags=*/0,
      /*.parameter_count=*/1,
  };
  loom_string_id_t key = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("shape"), &key));
  loom_type_id_t parent_ids[2];
  loom_type_id_t child_ids[2];
  for (size_t i = 0; i < IREE_ARRAYSIZE(parent_ids); ++i) {
    loom_value_id_t dimension = LOOM_VALUE_ID_INVALID;
    IREE_ASSERT_OK(loom_module_define_value(
        module_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &dimension));
    child_ids[i] =
        Intern(loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                                   loom_dim_pack_dynamic(dimension),
                                   /*encoding_id=*/0));
    const loom_named_attr_t entry = {key, {}, loom_attr_type(child_ids[i])};
    loom_attribute_t metadata;
    IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
        module_, loom_make_named_attr_slice(&entry, 1), &metadata));
    parent_ids[i] = Intern(loom_type_parameterized(&descriptor, 1, &metadata));
  }
  ASSERT_NE(parent_ids[0], parent_ids[1]);
  loom_bytecode_type_index_t index;
  IREE_ASSERT_OK(loom_bytecode_type_index_initialize(module_, &arena_, &index));
  for (size_t i = 0; i < IREE_ARRAYSIZE(parent_ids); ++i) {
    EXPECT_EQ(loom_bytecode_type_index_lookup(
                  &index, loom_type_table_get(&module_->types, parent_ids[i])),
              parent_ids[i]);
    const auto& node = index.nodes[parent_ids[i]];
    ASSERT_EQ(node.dependencies.explicit_count, 1u);
    EXPECT_TRUE(node.has_bindings);
    EXPECT_EQ(index.dependencies[node.dependencies.begin], child_ids[i]);
  }
}

}  // namespace
}  // namespace loom
