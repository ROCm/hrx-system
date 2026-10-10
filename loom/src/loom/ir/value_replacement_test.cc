// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/value_replacement.h"

#include <array>
#include <cstring>
#include <set>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/ops/test/types.h"

namespace loom {
namespace {

static const loom_attr_descriptor_t kLookupParameters[] = {{
    .name = LOOM_BSTRING_REF(8, "metadata"),
    .attr_kind = LOOM_ATTR_DICT,
}};

static const loom_parameterized_type_descriptor_t kLookupDescriptor = {
    .name = LOOM_BSTRING_REF(11, "test.lookup"),
    .parameter_descriptors = kLookupParameters,
    .fact_domain = nullptr,
    .ir_kind = LOOM_TYPE_PARAMETERIZED,
    .type_flags = 0,
    .parameter_count = IREE_ARRAYSIZE(kLookupParameters),
};

class ScopedReplacement {
 public:
  ScopedReplacement(loom_module_t* module, loom_value_id_t old_id,
                    loom_value_id_t new_id) {
    loom_value_replacement_initialize(module, old_id, new_id, &value);
  }
  ~ScopedReplacement() { loom_value_replacement_deinitialize(&value); }
  ScopedReplacement(const ScopedReplacement&) = delete;
  ScopedReplacement& operator=(const ScopedReplacement&) = delete;

  // Stack-owned production context, released even after a fatal test assertion.
  loom_value_replacement_t value;
};

class ScopedLookup {
 public:
  ScopedLookup(const loom_module_t* module,
               const loom_type_value_remap_t* remap) {
    loom_type_remap_lookup_initialize(module, remap, &value);
  }
  ~ScopedLookup() { loom_type_remap_lookup_deinitialize(&value); }
  ScopedLookup(const ScopedLookup&) = delete;
  ScopedLookup& operator=(const ScopedLookup&) = delete;

  // Stack-owned production context, released even after a fatal assertion.
  loom_type_remap_lookup_t value;
};

class ValueReplacementTest : public ::testing::Test {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<ValueReplacementTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(1024, {this, Allocate}, &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    ResetModule();
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void ResetModule() {
    failure_index_ = SIZE_MAX;
    loom_module_free(module_);
    module_ = nullptr;
    iree_arena_block_pool_trim(&pool_);
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("replacement"),
                                        &pool_, nullptr, {this, Allocate},
                                        &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    old_id_ = Constant(1);
    new_id_ = Constant(2);
  }

  loom_value_id_t Constant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_test_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_NONE, &op));
    return loom_test_constant_result(op);
  }

  loom_type_id_t Intern(loom_type_t type) {
    loom_type_id_t id;
    IREE_CHECK_OK(loom_module_intern_type_id(module_, type, &id));
    return id;
  }

  loom_value_id_t Carrier(loom_type_t type) {
    loom_value_id_t id;
    IREE_CHECK_OK(loom_module_define_value(module_, type, &id));
    IREE_CHECK_OK(loom_block_add_arg(module_, loom_module_block(module_), id));
    return id;
  }

  static loom_type_t Vector(loom_value_id_t width) {
    return loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                               loom_dim_pack_dynamic(width), 0);
  }

  loom_type_t Pair(loom_type_t first, loom_type_t second) {
    const loom_type_t children[] = {first, second};
    loom_type_t result;
    IREE_CHECK_OK(loom_module_intern_function_type(module_, children, 2,
                                                   nullptr, 0, &result));
    return result;
  }

  loom_type_t Array(loom_type_t element) {
    loom_type_t result;
    IREE_CHECK_OK(loom_test_array_type_make(module_, 0, Intern(element), 0,
                                            loom_named_attr_slice_empty(),
                                            &result));
    return result;
  }

  void ExpectDependencies(loom_value_id_t carrier,
                          std::initializer_list<loom_value_id_t> expected) {
    loom_type_use_iterator_t iterator;
    loom_module_value_type_dependencies(module_, carrier, &iterator);
    for (const auto provider : expected) {
      EXPECT_EQ(loom_type_dependencies_next(&iterator), provider);
    }
    EXPECT_EQ(loom_type_dependencies_next(&iterator), LOOM_VALUE_ID_INVALID);
  }

  static void ExpectSame(loom_type_t first, loom_type_t second) {
    EXPECT_EQ(std::memcmp(&first, &second, sizeof(first)), 0);
  }

  // Backing allocation ordinal to fail, or SIZE_MAX when disabled.
  size_t failure_index_ = SIZE_MAX;
  // Backing allocations attempted since the current fault test began.
  size_t allocation_count_ = 0;
  // Small shared blocks expose both reconstruction and publication failures.
  iree_arena_block_pool_t pool_ = {};
  // Real dialect registrations needed by parameterized attributes and owners.
  loom_context_t context_ = {};
  // Module owning canonical types and active use records under test.
  loom_module_t* module_ = nullptr;
  // Builder for real reference providers and operation attribute owners.
  loom_builder_t builder_ = {};
  // Original index-valued provider.
  loom_value_id_t old_id_;
  // Replacement index-valued provider.
  loom_value_id_t new_id_;
};

TEST_F(ValueReplacementTest, DeepSharedGraphPreservesCanonicalChildren) {
  constexpr int kDepth = 2048;
  loom_type_t original = Vector(old_id_);
  for (int i = 0; i < kDepth; ++i) {
    original = Pair(original, original);
  }
  const auto first = Carrier(original);
  const auto second = Carrier(original);
  const auto count = module_->types.count;
  IREE_ASSERT_OK(loom_value_replace_all_uses_with(module_, old_id_, new_id_));
  EXPECT_EQ(module_->types.count, count + kDepth + 1);
  auto type = loom_module_value_type(module_, first);
  ExpectSame(type, loom_module_value_type(module_, second));
  for (int i = 0; i < kDepth; ++i) {
    const auto* data = loom_type_func_data(type);
    ASSERT_EQ(data->arg_count, 2);
    ExpectSame(data->types[0], data->types[1]);
    type = data->types[0];
  }
  EXPECT_EQ(loom_type_dim_value_id_at(type, 0), new_id_);
  ExpectDependencies(first, {new_id_});
  ExpectDependencies(second, {new_id_});
  EXPECT_FALSE(loom_module_value_has_type_uses(module_, old_id_));
}

TEST_F(ValueReplacementTest, MixedAttributeTypeGraphUsesAnExplicitStack) {
  constexpr int kDepth = 2048;
  auto original = Vector(old_id_);
  for (int i = 0; i < kDepth; ++i) {
    if (i % 2) {
      original = Pair(original, original);
    } else {
      loom_attribute_t options;
      IREE_ASSERT_OK(loom_test_options_attr_make(
          module_, LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_ELEMENT_TYPE,
          LOOM_TEST_OPTIONS_MODE_FAST, loom_enum_array_empty(),
          Intern(original), loom_attr_absent(), loom_symbol_ref_null(),
          loom_parameterized_attr_array_empty(), &options));
      const loom_attribute_t variants[] = {options, options};
      IREE_ASSERT_OK(loom_test_variant_set_type_make(
          module_, 0, loom_make_parameterized_attr_array(variants, 2),
          loom_parameterized_attr_array_empty(), &original));
    }
  }
  const auto carrier = Carrier(original);
  IREE_ASSERT_OK(
      loom_module_replace_value_type_uses(module_, old_id_, new_id_));
  auto type = loom_module_value_type(module_, carrier);
  for (int i = kDepth - 1; i >= 0; --i) {
    if (i % 2) {
      const auto* data = loom_type_func_data(type);
      ExpectSame(data->types[0], data->types[1]);
      type = data->types[0];
    } else {
      const auto variants = loom_test_variant_set_type_values(type);
      ASSERT_EQ(variants.count, 2);
      const auto element =
          loom_test_options_attr_element_type(variants.values[0]);
      EXPECT_EQ(element,
                loom_test_options_attr_element_type(variants.values[1]));
      type = loom_type_table_get(&module_->types, element);
    }
  }
  EXPECT_EQ(loom_type_dim_value_id_at(type, 0), new_id_);
  ExpectDependencies(carrier, {new_id_});
}

TEST_F(ValueReplacementTest,
       LookupNormalizesCanonicalGraphsWithoutPublishingTypes) {
  const loom_value_id_t other_id = Constant(3);
  const loom_value_id_t absent_target_id = Constant(4);
  const loom_value_id_t other_target_id = Constant(5);
  const loom_overflow_dim_t source_dimensions[] = {
      loom_dim_pack_dynamic(old_id_), loom_dim_pack_static(7),
      loom_dim_pack_dynamic(other_id), loom_dim_pack_dynamic(old_id_)};
  const loom_overflow_dim_t target_dimensions[] = {
      loom_dim_pack_dynamic(new_id_), loom_dim_pack_static(7),
      loom_dim_pack_dynamic(other_target_id), loom_dim_pack_dynamic(new_id_)};
  loom_type_t source_leaf = {
      .header =
          loom_type_make_header(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, 4, 0),
  };
  source_leaf.dims[0] = reinterpret_cast<uintptr_t>(source_dimensions);
  loom_type_t target_leaf = source_leaf;
  target_leaf.dims[0] = reinterpret_cast<uintptr_t>(target_dimensions);
  const loom_type_id_t source_leaf_id = Intern(source_leaf);
  const loom_type_id_t target_leaf_id = Intern(target_leaf);

  loom_string_id_t element_name = LOOM_STRING_ID_INVALID;
  loom_string_id_t predicate_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("element"), &element_name));
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("predicate"),
                                           &predicate_name));
  loom_predicate_t source_predicate = {
      LOOM_PREDICATE_EQ,   2, {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE}, {},
      {old_id_, other_id},
  };
  loom_predicate_t target_predicate = source_predicate;
  target_predicate.args[0] = new_id_;
  target_predicate.args[1] = other_target_id;
  const loom_named_attr_t source_entries[] = {
      {element_name, {}, loom_attr_type(source_leaf_id)},
      {predicate_name, {}, loom_attr_predicate_list(&source_predicate, 1)},
  };
  const loom_named_attr_t target_entries[] = {
      {element_name, {}, loom_attr_type(target_leaf_id)},
      {predicate_name, {}, loom_attr_predicate_list(&target_predicate, 1)},
  };
  loom_attribute_t source_metadata;
  loom_attribute_t target_metadata;
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module_,
      loom_make_named_attr_slice(source_entries,
                                 IREE_ARRAYSIZE(source_entries)),
      &source_metadata));
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module_,
      loom_make_named_attr_slice(target_entries,
                                 IREE_ARRAYSIZE(target_entries)),
      &target_metadata));
  const loom_attribute_t source_parameters[] = {source_metadata};
  const loom_attribute_t target_parameters[] = {target_metadata};
  loom_type_t source_parameterized;
  loom_type_t target_parameterized;
  loom_type_id_t source_parameterized_id = LOOM_TYPE_ID_INVALID;
  loom_type_id_t target_parameterized_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_make_parameterized_type(
      module_, &kLookupDescriptor, source_parameters,
      IREE_ARRAYSIZE(source_parameters), &source_parameterized,
      &source_parameterized_id));
  IREE_ASSERT_OK(loom_module_make_parameterized_type(
      module_, &kLookupDescriptor, target_parameters,
      IREE_ARRAYSIZE(target_parameters), &target_parameterized,
      &target_parameterized_id));

  loom_string_id_t dialect_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("test.lookup_box"),
                                           &dialect_name));
  loom_type_t source_dialect;
  loom_type_t target_dialect;
  IREE_ASSERT_OK(loom_module_intern_type(
      module_, loom_type_dialect(dialect_name, 1, &source_parameterized),
      &source_dialect));
  IREE_ASSERT_OK(loom_module_intern_type(
      module_, loom_type_dialect(dialect_name, 1, &target_parameterized),
      &target_dialect));
  loom_type_t source_register;
  loom_type_t target_register;
  IREE_ASSERT_OK(loom_module_intern_register_type(
      module_, 17, 29, source_dialect, &source_register));
  IREE_ASSERT_OK(loom_module_intern_register_type(
      module_, 17, 29, target_dialect, &target_register));
  const loom_type_t source = Pair(source_register, source_register);
  const loom_type_t target = Pair(target_register, target_register);
  const loom_type_value_remap_t other_remap = {
      .source_values = &other_id,
      .target_values = &other_target_id,
      .count = 1,
  };
  const loom_type_value_remap_t remap = {
      .source_values = &old_id_,
      .target_values = &new_id_,
      .count = 1,
      .flags = 0,
      .next = &other_remap,
  };
  ASSERT_TRUE(
      loom_type_equal_after_value_remap(module_, source, target, &remap));

  const iree_host_size_t type_count = module_->types.count;
  const iree_host_size_t retained_bytes = module_->arena.used_allocation_size;
  const iree_host_size_t interner_count = module_->type_intern.count;
  const std::array<uint32_t, 2> recent_types = {
      module_->recent_exact_type_ordinals[0],
      module_->recent_exact_type_ordinals[1],
  };
  {
    ScopedLookup lookup(module_, &remap);
    bool equal = false;
    IREE_ASSERT_OK(
        loom_type_remap_lookup_equal(&lookup.value, source, target, &equal));
    EXPECT_TRUE(equal);
    const iree_host_size_t scratch_bytes =
        lookup.value.state.scratch.used_allocation_size;
    EXPECT_GT(scratch_bytes, 0u);
    IREE_ASSERT_OK(
        loom_type_remap_lookup_equal(&lookup.value, source, target, &equal));
    EXPECT_TRUE(equal);
    EXPECT_EQ(lookup.value.state.scratch.used_allocation_size, scratch_bytes);
  }

  const loom_type_value_remap_t absent_remap = {
      .source_values = &old_id_,
      .target_values = &absent_target_id,
      .count = 1,
      .flags = 0,
      .next = &other_remap,
  };
  {
    ScopedLookup lookup(module_, &absent_remap);
    bool equal = true;
    IREE_ASSERT_OK(
        loom_type_remap_lookup_equal(&lookup.value, source, target, &equal));
    EXPECT_FALSE(equal);
    const iree_host_size_t scratch_bytes =
        lookup.value.state.scratch.used_allocation_size;
    IREE_ASSERT_OK(
        loom_type_remap_lookup_equal(&lookup.value, source, target, &equal));
    EXPECT_FALSE(equal);
    EXPECT_EQ(lookup.value.state.scratch.used_allocation_size, scratch_bytes);
  }

  EXPECT_EQ(module_->types.count, type_count);
  EXPECT_EQ(module_->arena.used_allocation_size, retained_bytes);
  EXPECT_EQ(module_->type_intern.count, interner_count);
  EXPECT_EQ(module_->recent_exact_type_ordinals[0], recent_types[0]);
  EXPECT_EQ(module_->recent_exact_type_ordinals[1], recent_types[1]);
}

TEST_F(ValueReplacementTest, LookupLeavesNeedNoTraversalState) {
  const loom_type_id_t source_id = Intern(Vector(old_id_));
  const loom_type_id_t target_id = Intern(Vector(new_id_));
  const loom_type_t source = loom_type_table_get(&module_->types, source_id);
  const loom_type_t target = loom_type_table_get(&module_->types, target_id);
  const loom_type_value_remap_t remap = {
      .source_values = &old_id_,
      .target_values = &new_id_,
      .count = 1,
  };
  const iree_host_size_t type_count = module_->types.count;
  const iree_host_size_t retained_bytes = module_->arena.used_allocation_size;
  const size_t allocation_count = allocation_count_;
  ScopedLookup lookup(module_, &remap);
  EXPECT_FALSE(lookup.value.state_initialized);
  bool equal = false;
  IREE_ASSERT_OK(
      loom_type_remap_lookup_equal(&lookup.value, source, target, &equal));
  EXPECT_TRUE(equal);
  EXPECT_FALSE(lookup.value.state_initialized);
  EXPECT_EQ(allocation_count_, allocation_count);
  EXPECT_EQ(module_->types.count, type_count);
  EXPECT_EQ(module_->arena.used_allocation_size, retained_bytes);
}

TEST_F(ValueReplacementTest,
       LookupAllocationFailuresLeaveCanonicalStateUnchanged) {
  loom_type_t source = Vector(old_id_);
  loom_type_t target = Vector(new_id_);
  for (int i = 0; i < 32; ++i) {
    source = Pair(source, source);
    target = Pair(target, target);
  }
  const loom_type_value_remap_t remap = {
      .source_values = &old_id_,
      .target_values = &new_id_,
      .count = 1,
  };
  const iree_host_size_t type_count = module_->types.count;
  const iree_host_size_t interner_count = module_->type_intern.count;
  const iree_host_size_t retained_bytes = module_->arena.used_allocation_size;
  for (size_t failure = 0;; ++failure) {
    SCOPED_TRACE(failure);
    iree_arena_block_pool_trim(&pool_);
    failure_index_ = failure;
    allocation_count_ = 0;
    bool equal = false;
    iree_status_t status = iree_ok_status();
    {
      ScopedLookup lookup(module_, &remap);
      status =
          loom_type_remap_lookup_equal(&lookup.value, source, target, &equal);
    }
    failure_index_ = SIZE_MAX;
    const bool succeeded = iree_status_is_ok(status);
    if (succeeded) {
      EXPECT_TRUE(equal);
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      EXPECT_FALSE(equal);
      EXPECT_EQ(allocation_count_, failure + 1);
    }
    EXPECT_EQ(module_->types.count, type_count);
    EXPECT_EQ(module_->type_intern.count, interner_count);
    EXPECT_EQ(module_->arena.used_allocation_size, retained_bytes);
    if (succeeded) {
      EXPECT_LE(allocation_count_, failure);
      break;
    }
  }
}

TEST_F(ValueReplacementTest, ReusesOneMemoAcrossTypesAndAttributes) {
  auto original = Vector(old_id_);
  for (int i = 0; i < 64; ++i) {
    original = Pair(original, original);
  }
  ScopedReplacement replacement(module_, old_id_, new_id_);
  loom_type_t result;
  bool changed;
  IREE_ASSERT_OK(loom_value_replacement_type(&replacement.value, original,
                                             &result, &changed));
  ASSERT_TRUE(changed);
  const auto used = replacement.value.state.scratch.used_allocation_size;
  const auto type_count = module_->types.count;
  const auto result_id = Intern(result);
  loom_attribute_t attribute;
  IREE_ASSERT_OK(loom_value_replacement_attribute(
      &replacement.value, loom_attr_type(Intern(original)), &attribute,
      &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(attribute.type_id, result_id);
  EXPECT_EQ(replacement.value.state.scratch.used_allocation_size, used);
  EXPECT_EQ(module_->types.count, type_count);

  const auto parent = Pair(original, Vector(new_id_));
  IREE_ASSERT_OK(loom_value_replacement_type(&replacement.value, parent,
                                             &result, &changed));
  ASSERT_TRUE(changed);
  ExpectSame(loom_type_func_data(result)->types[0],
             loom_type_table_get(&module_->types, result_id));
}

TEST_F(ValueReplacementTest, UnaffectedNonemptyGraphAllocatesNoScratch) {
  auto original = Vector(new_id_);
  for (int i = 0; i < 2048; ++i) {
    original = Pair(original, original);
  }
  ScopedReplacement replacement(module_, old_id_, new_id_);
  const auto used = module_->arena.used_allocation_size;
  const auto type_count = module_->types.count;
  loom_type_t result;
  bool changed;
  IREE_ASSERT_OK(loom_value_replacement_type(&replacement.value, original,
                                             &result, &changed));
  EXPECT_FALSE(changed);
  ExpectSame(result, original);
  EXPECT_EQ(replacement.value.state.scratch.used_allocation_size, 0u);
  EXPECT_EQ(module_->arena.used_allocation_size, used);
  EXPECT_EQ(module_->types.count, type_count);

  const auto parent = Pair(Vector(old_id_), original);
  const auto parent_count = module_->types.count;
  IREE_ASSERT_OK(loom_value_replacement_type(&replacement.value, parent,
                                             &result, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(module_->types.count, parent_count + 1);
  ExpectSame(loom_type_func_data(result)->types[1], original);
}

TEST_F(ValueReplacementTest, PublishedAttributesSurviveScratchBlockReuse) {
  const auto source = Array(Pair(Vector(old_id_), Vector(old_id_)));
  loom_attribute_t options;
  IREE_ASSERT_OK(loom_test_options_attr_make(
      module_, LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_ELEMENT_TYPE,
      LOOM_TEST_OPTIONS_MODE_PRECISE, loom_enum_array_empty(), Intern(source),
      loom_attr_absent(), loom_symbol_ref_null(),
      loom_parameterized_attr_array_empty(), &options));
  loom_string_id_t type_key;
  loom_string_id_t predicate_key;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("options"), &type_key));
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("constraints"),
                                           &predicate_key));
  loom_predicate_t predicate = {LOOM_PREDICATE_EQ,
                                2,
                                {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
                                {},
                                {old_id_, old_id_}};
  const loom_named_attr_t entries[] = {
      {predicate_key, {}, loom_attr_predicate_list(&predicate, 1)},
      {type_key, {}, options}};
  loom_op_t* owner;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &builder_, LOOM_TEST_ATTRS_BUILD_FLAG_HAS_DICT, old_id_,
      loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_NONE, &owner));

  // Starting without free blocks makes every returned block belong to the
  // completed replacement. Reacquire and overwrite them before reading the IR.
  iree_arena_block_pool_trim(&pool_);
  IREE_ASSERT_OK(loom_value_replace_all_uses_with(module_, old_id_, new_id_));
  iree_arena_allocator_t reused;
  iree_arena_initialize(&pool_, &reused);
  const auto allocations_before_reuse = allocation_count_;
  const auto block_size = iree_arena_block_pool_max_allocation_size(&pool_);
  // The first backing allocation proves all previously free blocks were used.
  while (allocation_count_ == allocations_before_reuse) {
    void* memory;
    IREE_ASSERT_OK(iree_arena_allocate(&reused, block_size, &memory));
    std::memset(memory, 0xA5, block_size);
  }
  iree_arena_deinitialize(&reused);
  const auto result = loom_test_attrs_dict(owner);
  ASSERT_EQ(result.count, 2);
  EXPECT_EQ(result.entries[0].name_id, predicate_key);
  EXPECT_EQ(result.entries[1].name_id, type_key);
  ASSERT_EQ(result.entries[1].value.kind, LOOM_ATTR_PARAMETERIZED);
  const auto element =
      loom_test_options_attr_element_type(result.entries[1].value);
  const auto array = loom_type_table_get(&module_->types, element);
  const auto pair = loom_type_table_get(
      &module_->types, loom_test_array_type_element_type(array));
  const auto* children = loom_type_func_data(pair)->types;
  EXPECT_EQ(loom_type_dim_value_id_at(children[0], 0), new_id_);
  ExpectSame(children[0], children[1]);
  ASSERT_EQ(result.entries[0].value.kind, LOOM_ATTR_PREDICATE_LIST);
  const auto& updated = result.entries[0].value.predicate_list[0];
  EXPECT_EQ(updated.args[0], new_id_);
  EXPECT_EQ(updated.args[1], new_id_);
  EXPECT_EQ(loom_test_attrs_input(owner), new_id_);
  EXPECT_FALSE(
      loom_value_has_attribute_uses(loom_module_value(module_, old_id_)));
}

TEST_F(ValueReplacementTest, TemporaryChildrenAreNotRetainedByTheMemo) {
  auto child = Vector(old_id_);
  loom_string_id_t name;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("test.box"), &name));
  const auto original = loom_type_dialect(name, 1, &child);
  ScopedReplacement replacement(module_, old_id_, new_id_);
  loom_type_t first;
  bool changed;
  IREE_ASSERT_OK(loom_value_replacement_type(&replacement.value, original,
                                             &first, &changed));
  ASSERT_TRUE(changed);
  child = loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                              loom_dim_pack_static(7),
                              loom_dim_pack_dynamic(old_id_), 0);
  loom_type_t second;
  IREE_ASSERT_OK(loom_value_replacement_type(&replacement.value, original,
                                             &second, &changed));
  ASSERT_TRUE(changed);
  EXPECT_EQ(loom_type_rank(loom_type_dialect_params(first)[0]), 1);
  const auto result = loom_type_dialect_params(second)[0];
  EXPECT_EQ(loom_type_rank(result), 2);
  EXPECT_EQ(loom_type_dim_value_id_at(result, 1), new_id_);
}

TEST_F(ValueReplacementTest, OverflowDimensionsPreserveOtherProviders) {
  const auto other = Constant(3);
  const loom_overflow_dim_t dimensions[] = {
      loom_dim_pack_dynamic(old_id_), loom_dim_pack_static(7),
      loom_dim_pack_dynamic(other), loom_dim_pack_dynamic(old_id_)};
  loom_type_t original = {
      .header =
          loom_type_make_header(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, 4, 0),
  };
  original.dims[0] = reinterpret_cast<uintptr_t>(dimensions);
  const auto carrier = Carrier(original);
  IREE_ASSERT_OK(loom_value_replace_all_uses_with(module_, old_id_, new_id_));
  const auto result = loom_module_value_type(module_, carrier);
  EXPECT_EQ(loom_type_dim_value_id_at(result, 0), new_id_);
  EXPECT_EQ(loom_type_dim(result, 1), loom_dim_pack_static(7));
  EXPECT_EQ(loom_type_dim_value_id_at(result, 2), other);
  EXPECT_EQ(loom_type_dim_value_id_at(result, 3), new_id_);
  ExpectDependencies(carrier, {new_id_, other});
}

TEST_F(ValueReplacementTest, RegisterPayloadAndExistingCanonicalResultSurvive) {
  auto source = Pair(Vector(old_id_), Vector(old_id_));
  auto expected = Pair(Vector(new_id_), Vector(new_id_));
  IREE_ASSERT_OK(
      loom_module_intern_register_type(module_, 17, 29, source, &source));
  IREE_ASSERT_OK(
      loom_module_intern_register_type(module_, 17, 29, expected, &expected));
  const auto count = module_->types.count;
  loom_type_t result;
  bool changed;
  IREE_ASSERT_OK(loom_module_replace_type_value_references(
      module_, source, old_id_, new_id_, &result, &changed));
  EXPECT_TRUE(changed);
  ExpectSame(result, expected);
  EXPECT_EQ(loom_type_register_payload0(result), 17u);
  EXPECT_EQ(loom_type_register_payload1(result), 29u);
  EXPECT_EQ(module_->types.count, count);
}

TEST_F(ValueReplacementTest, EncodingOverflowDoesNotChangeTheCarrier) {
  const auto old_encoding = Carrier(loom_type_encoding());
  loom_value_id_t new_encoding = old_encoding;
  while (new_encoding <= UINT16_MAX) {
    IREE_ASSERT_OK(
        loom_module_define_value(module_, loom_type_encoding(), &new_encoding));
  }
  auto original = loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                                      loom_dim_pack_static(4), old_encoding);
  original.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  const auto carrier = Carrier(original);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_module_replace_value_type_uses(module_, old_encoding, new_encoding));
  ExpectSame(loom_module_value_type(module_, carrier), original);
  ExpectDependencies(carrier, {old_encoding});

  const loom_type_value_remap_t remap = {
      .source_values = &old_encoding,
      .target_values = &new_encoding,
      .count = 1,
  };
  ScopedLookup lookup(module_, &remap);
  bool equal = true;
  IREE_ASSERT_OK(
      loom_type_remap_lookup_equal(&lookup.value, original, original, &equal));
  EXPECT_FALSE(equal);
  EXPECT_FALSE(lookup.value.state_initialized);
}

TEST_F(ValueReplacementTest, MembershipUsesFullSparseProviderIdentities) {
  loom_type_dependency_id_t root = 0;
  EXPECT_FALSE(loom_type_dependencies_contains(&module_->type_uses, root, 0));
  std::set<loom_value_id_t> providers;
  uint32_t random = 0x91e10da5;
  for (int i = 0; i < 256; ++i) {
    random ^= random << 13;
    random ^= random >> 17;
    random ^= random << 5;
    const auto provider = random & ~uint32_t{1};
    providers.insert(provider);
    IREE_ASSERT_OK(
        loom_type_dependencies_add(&module_->type_uses, root, provider, &root));
    for (const auto member : providers) {
      EXPECT_TRUE(
          loom_type_dependencies_contains(&module_->type_uses, root, member));
      EXPECT_FALSE(loom_type_dependencies_contains(&module_->type_uses, root,
                                                   member | 1));
    }
  }
  // Declared sets include unavailable providers without activating any owner.
  EXPECT_FALSE(loom_module_has_active_type_uses(module_));
}

TEST_F(ValueReplacementTest,
       AllocationFailureKeepsEveryPublishedOwnerConsistent) {
  for (size_t failure = 0;; ++failure) {
    SCOPED_TRACE(failure);
    ASSERT_NO_FATAL_FAILURE(ResetModule());
    auto original = Vector(old_id_);
    for (int i = 0; i < 32; ++i) {
      original = Array(original);
    }
    const std::array<loom_value_id_t, 2> carriers = {Carrier(original),
                                                     Carrier(original)};
    loom_string_id_t key;
    IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("shape"), &key));
    const loom_named_attr_t attributes[] = {
        {key, {}, loom_attr_type(Intern(original))}};
    std::array<loom_op_t*, 2> owners;
    std::array<std::vector<loom_attribute_owner_id_t>, 2> owner_ids;
    for (size_t i = 0; i < owners.size(); ++i) {
      IREE_ASSERT_OK(loom_test_attrs_build(
          &builder_, LOOM_TEST_ATTRS_BUILD_FLAG_HAS_DICT, old_id_,
          loom_make_named_attr_slice(attributes, 1),
          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_NONE,
          &owners[i]));
      const auto* ids = loom_op_attribute_owners(owners[i]);
      owner_ids[i].assign(ids, ids + owners[i]->attribute_count);
    }
    failure_index_ = failure;
    allocation_count_ = 0;
    iree_status_t status =
        loom_value_replace_all_uses_with(module_, old_id_, new_id_);
    failure_index_ = SIZE_MAX;
    const bool succeeded = iree_status_is_ok(status);
    if (!succeeded) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      EXPECT_EQ(allocation_count_, failure + 1);
    }
    for (const auto carrier : carriers) {
      auto type = loom_module_value_type(module_, carrier);
      while (loom_test_array_type_isa(type)) {
        type = loom_type_table_get(&module_->types,
                                   loom_test_array_type_element_type(type));
      }
      const auto provider = loom_type_dim_value_id_at(type, 0);
      EXPECT_TRUE(provider == old_id_ || provider == new_id_);
      ExpectDependencies(carrier, {provider});
    }
    for (size_t i = 0; i < owners.size(); ++i) {
      auto type = loom_type_table_get(
          &module_->types,
          loom_test_attrs_dict(owners[i]).entries[0].value.type_id);
      while (loom_test_array_type_isa(type)) {
        type = loom_type_table_get(&module_->types,
                                   loom_test_array_type_element_type(type));
      }
      const auto provider = loom_type_dim_value_id_at(type, 0);
      uint8_t reference_owner_count = 0;
      for (uint8_t index = 0; index < owners[i]->attribute_count; ++index) {
        EXPECT_EQ(loom_op_attribute_owners(owners[i])[index],
                  owner_ids[i][index]);
        loom_type_use_iterator_t dependencies;
        loom_attribute_dependencies_begin(&module_->type_uses, owners[i], index,
                                          &dependencies);
        const auto first = loom_type_dependencies_next(&dependencies);
        if (first != LOOM_VALUE_ID_INVALID) {
          EXPECT_EQ(first, provider);
          ++reference_owner_count;
        }
        EXPECT_EQ(loom_type_dependencies_next(&dependencies),
                  LOOM_VALUE_ID_INVALID);
      }
      EXPECT_EQ(reference_owner_count, 1);
      EXPECT_EQ(loom_test_attrs_input(owners[i]),
                succeeded ? new_id_ : old_id_);
    }
    IREE_ASSERT_OK(loom_value_replace_all_uses_with(module_, old_id_, new_id_));
    EXPECT_FALSE(loom_module_value_has_type_uses(module_, old_id_));
    EXPECT_FALSE(
        loom_value_has_attribute_uses(loom_module_value(module_, old_id_)));
    for (const auto carrier : carriers) {
      ExpectDependencies(carrier, {new_id_});
    }
    if (succeeded) {
      EXPECT_LE(allocation_count_, failure);
      break;
    }
  }
}

}  // namespace
}  // namespace loom
