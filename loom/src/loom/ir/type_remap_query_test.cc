// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/type_remap_query.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/attribute_schema.h"
#include "loom/ir/context.h"
#include "loom/ir/parameterized_attr.h"
#include "loom/ir/parameterized_type.h"

namespace loom {
namespace {

static const loom_attr_descriptor_t kFamilyParameters[] = {
    {
        /*.name=*/LOOM_BSTRING_REF(4, "type"),
        /*.attr_kind=*/LOOM_ATTR_TYPE,
    },
    {
        /*.name=*/LOOM_BSTRING_REF(9, "predicate"),
        /*.attr_kind=*/LOOM_ATTR_PREDICATE_LIST,
    },
};

static const loom_parameterized_attr_descriptor_t kAttributeFamilies[] = {
    {
        /*.name=*/LOOM_BSTRING_REF(17, "test.remap_family"),
        /*.kind=*/LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_TEST, 0),
        /*.parameter_count=*/IREE_ARRAYSIZE(kFamilyParameters),
        /*.primary_parameter_index=*/
        LOOM_PARAMETERIZED_ATTR_NO_PRIMARY_PARAMETER,
        /*.parameter_descriptors=*/kFamilyParameters,
    },
};

static const loom_bstring_t kEnumCases[] = {
    LOOM_BSTRING_REF(2, "c0"),  LOOM_BSTRING_REF(2, "c1"),
    LOOM_BSTRING_REF(2, "c2"),  LOOM_BSTRING_REF(2, "c3"),
    LOOM_BSTRING_REF(2, "c4"),  LOOM_BSTRING_REF(2, "c5"),
    LOOM_BSTRING_REF(2, "c6"),  LOOM_BSTRING_REF(2, "c7"),
    LOOM_BSTRING_REF(2, "c8"),  LOOM_BSTRING_REF(2, "c9"),
    LOOM_BSTRING_REF(3, "c10"), LOOM_BSTRING_REF(3, "c11"),
    LOOM_BSTRING_REF(3, "c12"), LOOM_BSTRING_REF(3, "c13"),
    LOOM_BSTRING_REF(3, "c14"), LOOM_BSTRING_REF(3, "c15"),
    LOOM_BSTRING_REF(3, "c16"),
};

static const loom_symbol_reference_descriptor_t kSymbolReference = {
    /*.name=*/LOOM_BSTRING_REF(6, "symbol"),
    /*.interfaces=*/0,
    /*.role=*/LOOM_SYMBOL_REFERENCE_ROLE_DEPENDENCY,
};

#define REMAP_TEST_PARAMETER(kind)        \
  {                                       \
      /*.name=*/LOOM_BSTRING_REF(1, "p"), \
      /*.attr_kind=*/kind,                \
  }

static loom_attr_descriptor_t MakeParameterizedParameter(
    loom_attr_kind_t kind) {
  loom_attr_descriptor_t descriptor = {
      /*.name=*/LOOM_BSTRING_REF(1, "p"),
      /*.attr_kind=*/kind,
  };
  descriptor.reference.parameterized_attr_kind = kAttributeFamilies[0].kind;
  return descriptor;
}

static const loom_attr_descriptor_t kTypeParameters[] = {
    {
        /*.name=*/LOOM_BSTRING_REF(1, "p"),
        /*.attr_kind=*/LOOM_ATTR_I64,
        /*.flags=*/LOOM_ATTR_OPTIONAL,
    },
    REMAP_TEST_PARAMETER(LOOM_ATTR_I64),
    REMAP_TEST_PARAMETER(LOOM_ATTR_F64),
    REMAP_TEST_PARAMETER(LOOM_ATTR_STRING),
    REMAP_TEST_PARAMETER(LOOM_ATTR_BOOL),
    {
        /*.name=*/LOOM_BSTRING_REF(1, "p"),
        /*.attr_kind=*/LOOM_ATTR_ENUM,
        /*.flags=*/0,
        /*.enum_max_value=*/IREE_ARRAYSIZE(kEnumCases) - 1,
        /*.enum_case_names=*/kEnumCases,
    },
    {
        /*.name=*/LOOM_BSTRING_REF(1, "p"),
        /*.attr_kind=*/LOOM_ATTR_SYMBOL,
        /*.flags=*/0,
        /*.enum_max_value=*/0,
        /*.enum_case_names=*/nullptr,
        /*.reference=*/{&kSymbolReference},
    },
    {
        /*.name=*/LOOM_BSTRING_REF(1, "p"),
        /*.attr_kind=*/LOOM_ATTR_SYMBOL_ARRAY,
        /*.flags=*/0,
        /*.enum_max_value=*/0,
        /*.enum_case_names=*/nullptr,
        /*.reference=*/{&kSymbolReference},
    },
    {
        /*.name=*/LOOM_BSTRING_REF(1, "p"),
        /*.attr_kind=*/LOOM_ATTR_SYMBOL_SET,
        /*.flags=*/0,
        /*.enum_max_value=*/0,
        /*.enum_case_names=*/nullptr,
        /*.reference=*/{&kSymbolReference},
    },
    REMAP_TEST_PARAMETER(LOOM_ATTR_TYPE),
    REMAP_TEST_PARAMETER(LOOM_ATTR_ENCODING),
    REMAP_TEST_PARAMETER(LOOM_ATTR_BYTES),
    REMAP_TEST_PARAMETER(LOOM_ATTR_I64_ARRAY),
    {
        /*.name=*/LOOM_BSTRING_REF(1, "p"),
        /*.attr_kind=*/LOOM_ATTR_ENUM_ARRAY,
        /*.flags=*/0,
        /*.enum_max_value=*/IREE_ARRAYSIZE(kEnumCases) - 1,
        /*.enum_case_names=*/kEnumCases,
    },
    {
        /*.name=*/LOOM_BSTRING_REF(1, "p"),
        /*.attr_kind=*/LOOM_ATTR_SIGNED_ENUM_SET,
        /*.flags=*/0,
        /*.enum_max_value=*/IREE_ARRAYSIZE(kEnumCases) - 1,
        /*.enum_case_names=*/kEnumCases,
    },
    REMAP_TEST_PARAMETER(LOOM_ATTR_PREDICATE_LIST),
    REMAP_TEST_PARAMETER(LOOM_ATTR_DICT),
    MakeParameterizedParameter(LOOM_ATTR_PARAMETERIZED),
    MakeParameterizedParameter(LOOM_ATTR_PARAMETERIZED_ARRAY),
};

#undef REMAP_TEST_PARAMETER

static const loom_parameterized_type_descriptor_t kTypeDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(20, "test.remap_semantics"),
    /*.parameter_descriptors=*/kTypeParameters,
    /*.ir_kind=*/LOOM_TYPE_PARAMETERIZED,
    /*.type_flags=*/0,
    /*.parameter_count=*/IREE_ARRAYSIZE(kTypeParameters),
};

class TypeRemapQueryTest : public ::testing::Test {
 protected:
  struct FailingAllocatorState {
    // System allocator delegated to before the selected failure.
    iree_allocator_t system_allocator;
    // Zero-based allocation command to fail.
    uint32_t failure_index;
    // Number of allocation commands observed.
    uint32_t allocation_count;
  };

  static iree_status_t FailSelectedAllocation(void* self,
                                              iree_allocator_command_t command,
                                              const void* parameters,
                                              void** pointer) {
    FailingAllocatorState* state = static_cast<FailingAllocatorState*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        state->allocation_count++ == state->failure_index) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected allocation failure");
    }
    return state->system_allocator.ctl(state->system_allocator.self, command,
                                       parameters, pointer);
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_register_parameterized_attrs(
        &context_, LOOM_DIALECT_TEST, kAttributeFamilies,
        IREE_ARRAYSIZE(kAttributeFamilies)));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));

    const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    IREE_ASSERT_OK(loom_module_define_value(module_, index, &source_value_));
    IREE_ASSERT_OK(loom_module_define_value(module_, index, &target_value_));
    IREE_ASSERT_OK(
        loom_block_add_arg(module_, loom_module_block(module_), source_value_));
    IREE_ASSERT_OK(
        loom_block_add_arg(module_, loom_module_block(module_), target_value_));
    remap_ = {
        /*.source_values=*/&source_value_,
        /*.target_values=*/&target_value_,
        /*.count=*/1,
        /*.flags=*/LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE,
    };
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_type_t InternFunction(const loom_type_t* arguments,
                             iree_host_size_t argument_count) {
    loom_type_t type = {};
    IREE_CHECK_OK(loom_module_intern_function_type(
        module_, arguments, argument_count, nullptr, 0, &type));
    return type;
  }

  iree_arena_block_pool_t pool_ = {};
  loom_context_t context_ = {};
  loom_module_t* module_ = nullptr;
  loom_value_id_t source_value_ = LOOM_VALUE_ID_INVALID;
  loom_value_id_t target_value_ = LOOM_VALUE_ID_INVALID;
  loom_type_value_remap_t remap_ = {};
};

TEST_F(TypeRemapQueryTest, InlineComparisonsDoNotAcquireScratch) {
  loom_type_remap_query_t query;
  loom_type_remap_query_initialize(module_, &remap_, &query);
  const iree_host_size_t retained_bytes = module_->arena.used_allocation_size;

  for (uint32_t i = 0; i < 1024; ++i) {
    const loom_type_t source =
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                            loom_dim_pack_dynamic(source_value_), 0);
    const loom_type_t target =
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                            loom_dim_pack_dynamic(target_value_), 0);
    bool equal = false;
    IREE_ASSERT_OK(loom_type_remap_query_equal(&query, source, target, &equal));
    EXPECT_TRUE(equal);
  }
  bool equal = true;
  IREE_ASSERT_OK(loom_type_remap_query_equal(
      &query, loom_type_scalar(LOOM_SCALAR_TYPE_F32),
      loom_type_scalar(LOOM_SCALAR_TYPE_F64), &equal));
  EXPECT_FALSE(equal);
  EXPECT_FALSE(query.scratch_initialized);
  EXPECT_EQ(module_->arena.used_allocation_size, retained_bytes);
  loom_type_remap_query_deinitialize(&query);
}

TEST_F(TypeRemapQueryTest, CompletedFunctionProofIsReused) {
  const loom_type_t source_leaf =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(source_value_), 0);
  const loom_type_t target_leaf =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(target_value_), 0);
  const std::array<loom_type_t, 8> source_children = {
      source_leaf, source_leaf, source_leaf, source_leaf,
      source_leaf, source_leaf, source_leaf, source_leaf,
  };
  const std::array<loom_type_t, 8> target_children = {
      target_leaf, target_leaf, target_leaf, target_leaf,
      target_leaf, target_leaf, target_leaf, target_leaf,
  };
  const loom_type_t source_child =
      InternFunction(source_children.data(), source_children.size());
  const loom_type_t target_child =
      InternFunction(target_children.data(), target_children.size());
  std::array<loom_type_t, 64> source_roots;
  std::array<loom_type_t, 64> target_roots;
  source_roots.fill(source_child);
  target_roots.fill(target_child);
  const loom_type_t source =
      InternFunction(source_roots.data(), source_roots.size());
  const loom_type_t target =
      InternFunction(target_roots.data(), target_roots.size());
  ASSERT_TRUE(
      loom_type_equal_after_value_remap(module_, source, target, &remap_));

  loom_type_remap_query_t query;
  loom_type_remap_query_initialize(module_, &remap_, &query);
  bool equal = false;
  IREE_ASSERT_OK(loom_type_remap_query_equal(&query, source, target, &equal));
  ASSERT_TRUE(equal);
  ASSERT_TRUE(query.scratch_initialized);
  const iree_host_size_t proof_bytes = query.scratch_arena.used_allocation_size;
  for (uint32_t i = 0; i < 1024; ++i) {
    IREE_ASSERT_OK(loom_type_remap_query_equal(&query, source, target, &equal));
    ASSERT_TRUE(equal);
  }
  EXPECT_EQ(query.scratch_arena.used_allocation_size, proof_bytes);
  EXPECT_EQ(query.scratch_arena.allocation_head, nullptr);
  loom_type_remap_query_deinitialize(&query);
}

struct ComplexTypeStorage {
  std::array<int64_t, 4> integers = {1, -2, 3, INT64_MAX};
  std::array<uint8_t, 5> enums = {1, 3, 5, 7, 9};
  std::array<uint64_t, 2> signed_enums = {
      (UINT64_C(1) << 3) | (UINT64_C(1) << 8),
      (UINT64_C(1) << 4) | (UINT64_C(1) << 16),
  };
  std::array<uint8_t, 13> bytes = {0,  1,  2,  3,  5,   8,  13,
                                   21, 34, 55, 89, 144, 233};
  std::array<loom_predicate_t, 2> predicates = {};
  std::array<loom_attribute_t, 2> family_slots = {};
  std::array<loom_attribute_t, 2> family_values = {};
  std::array<loom_named_attr_t, 3> dictionary = {};
  std::array<loom_attribute_t, IREE_ARRAYSIZE(kTypeParameters)> parameters = {};
};

static loom_type_t BuildComplexType(
    loom_module_t* module, loom_value_id_t value_id,
    loom_type_id_t referenced_type, loom_string_id_t string_id,
    uint16_t encoding_id, const std::array<loom_symbol_ref_t, 3>& symbols,
    const std::array<loom_string_id_t, 3>& dictionary_names,
    bool different_bytes, ComplexTypeStorage* storage) {
  storage->predicates[0].kind = LOOM_PREDICATE_EQ;
  storage->predicates[0].arg_count = 2;
  storage->predicates[0].arg_tags[0] = LOOM_PRED_ARG_VALUE;
  storage->predicates[0].arg_tags[1] = LOOM_PRED_ARG_CONST;
  storage->predicates[0].args[0] = value_id;
  storage->predicates[0].args[1] = 16;
  storage->predicates[1].kind = LOOM_PREDICATE_RANGE;
  storage->predicates[1].arg_count = 3;
  storage->predicates[1].arg_tags[0] = LOOM_PRED_ARG_VALUE;
  storage->predicates[1].arg_tags[1] = LOOM_PRED_ARG_CONST;
  storage->predicates[1].arg_tags[2] = LOOM_PRED_ARG_CONST;
  storage->predicates[1].args[0] = value_id;
  storage->predicates[1].args[1] = 1;
  storage->predicates[1].args[2] = 128;
  storage->family_slots = {
      loom_attr_type(referenced_type),
      loom_attr_predicate_list(storage->predicates.data(),
                               storage->predicates.size()),
  };
  storage->family_values.fill(loom_make_parameterized_attr(
      kAttributeFamilies[0].kind, storage->family_slots.data(),
      storage->family_slots.size()));
  storage->dictionary = {
      loom_named_attr_t{dictionary_names[0], 0,
                        loom_attr_type(referenced_type)},
      loom_named_attr_t{dictionary_names[1], 0, storage->family_values[0]},
      loom_named_attr_t{dictionary_names[2], 0,
                        loom_attr_predicate_list(storage->predicates.data(),
                                                 storage->predicates.size())},
  };
  if (different_bytes) {
    ++storage->bytes.back();
  }
  storage->parameters = {
      loom_attr_absent(),
      loom_attr_i64(-17),
      loom_attr_f64(3.25),
      loom_attr_string(string_id),
      loom_attr_bool(true),
      loom_attr_enum(7),
      loom_attr_symbol(symbols[0]),
      loom_attr_symbol_array(symbols.data(), symbols.size()),
      loom_attr_symbol_set(symbols.data(), symbols.size()),
      loom_attr_type(referenced_type),
      loom_attr_encoding(encoding_id),
      loom_attr_bytes(storage->bytes.data(), storage->bytes.size()),
      loom_attr_i64_array(storage->integers.data(), storage->integers.size()),
      loom_attr_enum_array(storage->enums.data(), storage->enums.size()),
      loom_attr_signed_enum_set(storage->signed_enums.data(), 1),
      loom_attr_predicate_list(storage->predicates.data(),
                               storage->predicates.size()),
      loom_make_canonical_attr_dict(storage->dictionary.data(),
                                    storage->dictionary.size()),
      storage->family_values[0],
      loom_attr_parameterized_array(storage->family_values.data(),
                                    storage->family_values.size()),
  };
  loom_type_t type = {};
  IREE_CHECK_OK(loom_module_make_parameterized_type(
      module, &kTypeDescriptor, storage->parameters.data(),
      storage->parameters.size(), &type, nullptr));
  return type;
}

TEST_F(TypeRemapQueryTest, ExactFallbackMatchesAllCanonicalConstructors) {
  loom_string_id_t payload_string = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("payload"), &payload_string));
  std::array<loom_symbol_ref_t, 3> symbols;
  for (uint16_t i = 0; i < symbols.size(); ++i) {
    const char names[] = {'a', 'b', 'c'};
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(
        module_, iree_make_string_view(&names[i], 1), &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    symbols[i] = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  }
  std::array<loom_string_id_t, 3> dictionary_names;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("a_type"),
                                           &dictionary_names[0]));
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("b_family"),
                                           &dictionary_names[1]));
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("c_predicate"),
                                           &dictionary_names[2]));
  loom_string_id_t encoding_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("test.encoding"),
                                           &encoding_name));
  const loom_encoding_t encoding = {
      /*.name_id=*/encoding_name,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
  };
  uint16_t encoding_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module_, &encoding, &encoding_id));
  loom_string_id_t alternate_string = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("alternate"),
                                           &alternate_string));
  const loom_encoding_t alternate_encoding = {
      /*.name_id=*/alternate_string,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
  };
  uint16_t alternate_encoding_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module_, &alternate_encoding,
                                          &alternate_encoding_id));

  loom_type_id_t source_reference = LOOM_TYPE_ID_INVALID;
  loom_type_id_t target_reference = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module_,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(source_value_), 0),
      &source_reference));
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module_,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(target_value_), 0),
      &target_reference));
  loom_type_id_t alternate_reference = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_I16), &alternate_reference));

  ComplexTypeStorage source_storage;
  ComplexTypeStorage target_storage;
  ComplexTypeStorage different_storage;
  const loom_type_t source_parameterized = BuildComplexType(
      module_, source_value_, source_reference, payload_string, encoding_id,
      symbols, dictionary_names, /*different_bytes=*/false, &source_storage);
  const loom_type_t target_parameterized = BuildComplexType(
      module_, target_value_, target_reference, payload_string, encoding_id,
      symbols, dictionary_names, /*different_bytes=*/false, &target_storage);
  const loom_type_t different_parameterized = BuildComplexType(
      module_, target_value_, target_reference, payload_string, encoding_id,
      symbols, dictionary_names, /*different_bytes=*/true, &different_storage);

  auto build_root = [&](loom_value_id_t value_id,
                        loom_type_t parameterized) -> loom_type_t {
    const std::array<loom_overflow_dim_t, 4> dimensions = {
        loom_dim_pack_static(2),
        loom_dim_pack_dynamic(value_id),
        loom_dim_pack_static(5),
        loom_dim_pack_dynamic(value_id),
    };
    loom_type_t overflow = {};
    overflow.header = loom_type_make_header(
        LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, dimensions.size(), 0);
    overflow.dims[0] = reinterpret_cast<uintptr_t>(dimensions.data());
    loom_type_t canonical_overflow = {};
    IREE_CHECK_OK(
        loom_module_intern_type(module_, overflow, &canonical_overflow));
    const loom_type_t dialect_children[] = {parameterized, canonical_overflow};
    loom_type_t dialect = {};
    IREE_CHECK_OK(loom_module_intern_type(
        module_,
        loom_type_dialect(payload_string, IREE_ARRAYSIZE(dialect_children),
                          dialect_children),
        &dialect));
    loom_type_t register_type = {};
    IREE_CHECK_OK(loom_module_intern_register_type(
        module_, /*carrier_payload0=*/17, /*carrier_payload1=*/23, dialect,
        &register_type));
    const loom_type_t children[] = {
        register_type,
        parameterized,
        canonical_overflow,
        loom_type_pool(loom_dim_pack_dynamic(value_id)),
    };
    loom_type_t root = {};
    IREE_CHECK_OK(loom_module_intern_function_type(module_, children, 3,
                                                   &children[3], 1, &root));
    return root;
  };
  const loom_type_t source = build_root(source_value_, source_parameterized);
  const loom_type_t target = build_root(target_value_, target_parameterized);
  const loom_type_t different =
      build_root(target_value_, different_parameterized);
  ASSERT_TRUE(
      loom_type_equal_after_value_remap(module_, source, target, &remap_));
  ASSERT_FALSE(
      loom_type_equal_after_value_remap(module_, source, different, &remap_));

  loom_type_remap_query_t query;
  loom_type_remap_query_initialize(module_, &remap_, &query);
  bool equal = false;
  IREE_ASSERT_OK(loom_type_remap_query_equal(&query, source, target, &equal));
  EXPECT_TRUE(equal);
  IREE_ASSERT_OK(
      loom_type_remap_query_equal(&query, source, different, &equal));
  EXPECT_FALSE(equal);

  std::array<int64_t, 2> alternate_integers = {4, 5};
  const std::array<uint8_t, 2> alternate_enums = {2, 4};
  const std::array<uint64_t, 2> alternate_signed_enums = {
      UINT64_C(1) << 2,
      UINT64_C(1) << 6,
  };
  const std::array<uint8_t, 3> alternate_bytes = {3, 2, 1};
  std::array<loom_predicate_t, 2> alternate_predicates =
      target_storage.predicates;
  alternate_predicates[0].args[1] = 17;
  std::array<loom_attribute_t, 2> alternate_family_slots =
      target_storage.family_slots;
  alternate_family_slots[0] = loom_attr_type(alternate_reference);
  const loom_attribute_t alternate_family = loom_make_parameterized_attr(
      kAttributeFamilies[0].kind, alternate_family_slots.data(),
      alternate_family_slots.size());
  std::array<loom_named_attr_t, 3> alternate_dictionary =
      target_storage.dictionary;
  alternate_dictionary[0].value = loom_attr_type(alternate_reference);
  struct AttributeVariant {
    iree_host_size_t index;
    loom_attribute_t value;
  };
  const std::array<AttributeVariant, IREE_ARRAYSIZE(kTypeParameters)> variants =
      {{
          {0, loom_attr_i64(1)},
          {1, loom_attr_i64(-18)},
          {2, loom_attr_f64(4.25)},
          {3, loom_attr_string(alternate_string)},
          {4, loom_attr_bool(false)},
          {5, loom_attr_enum(8)},
          {6, loom_attr_symbol(symbols[1])},
          {7, loom_attr_symbol_array(symbols.data(), 2)},
          {8, loom_attr_symbol_set(symbols.data(), 2)},
          {9, loom_attr_type(alternate_reference)},
          {10, loom_attr_encoding(alternate_encoding_id)},
          {11, loom_attr_bytes(alternate_bytes.data(), alternate_bytes.size())},
          {12, loom_attr_i64_array(alternate_integers.data(),
                                   alternate_integers.size())},
          {13, loom_attr_enum_array(alternate_enums.data(),
                                    alternate_enums.size())},
          {14, loom_attr_signed_enum_set(alternate_signed_enums.data(), 1)},
          {15, loom_attr_predicate_list(alternate_predicates.data(),
                                        alternate_predicates.size())},
          {16, loom_make_canonical_attr_dict(alternate_dictionary.data(),
                                             alternate_dictionary.size())},
          {17, alternate_family},
          {18, loom_attr_parameterized_array(&alternate_family, 1)},
      }};
  for (const AttributeVariant& variant : variants) {
    std::array<loom_attribute_t, IREE_ARRAYSIZE(kTypeParameters)>
        variant_parameters = target_storage.parameters;
    variant_parameters[variant.index] = variant.value;
    loom_type_t variant_parameterized = {};
    IREE_ASSERT_OK(loom_module_make_parameterized_type(
        module_, &kTypeDescriptor, variant_parameters.data(),
        variant_parameters.size(), &variant_parameterized, nullptr));
    const loom_type_t variant_root =
        build_root(target_value_, variant_parameterized);
    ASSERT_FALSE(loom_type_equal_after_value_remap(module_, source,
                                                   variant_root, &remap_))
        << "attribute parameter " << variant.index;
    IREE_ASSERT_OK(
        loom_type_remap_query_equal(&query, source, variant_root, &equal));
    EXPECT_FALSE(equal) << "attribute parameter " << variant.index;
  }
  EXPECT_NE(query.exact_state, nullptr);
  EXPECT_EQ(query.scratch_arena.allocation_head, nullptr);
  loom_type_remap_query_deinitialize(&query);
}

TEST_F(TypeRemapQueryTest, ExactFallbackDistinguishesCompoundMetadata) {
  loom_type_remap_query_t query;
  loom_type_remap_query_initialize(module_, &remap_, &query);
  bool equal = true;
  IREE_ASSERT_OK(loom_type_remap_query_equal(
      &query, loom_type_scalar(LOOM_SCALAR_TYPE_F32),
      loom_type_scalar(LOOM_SCALAR_TYPE_F64), &equal));
  ASSERT_FALSE(equal);

  auto expect_unequal = [&](loom_type_t source, loom_type_t target) {
    ASSERT_FALSE(
        loom_type_equal_after_value_remap(module_, source, target, &remap_));
    IREE_ASSERT_OK(loom_type_remap_query_equal(&query, source, target, &equal));
    EXPECT_FALSE(equal);
  };

  const loom_type_t scalar = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t one_argument = InternFunction(&scalar, 1);
  const loom_type_t function_arguments[] = {scalar, scalar};
  const loom_type_t two_arguments =
      InternFunction(function_arguments, IREE_ARRAYSIZE(function_arguments));
  expect_unequal(one_argument, two_arguments);

  loom_string_id_t first_name = LOOM_STRING_ID_INVALID;
  loom_string_id_t second_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("first"), &first_name));
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("second"), &second_name));
  loom_type_t first_dialect = {};
  loom_type_t second_dialect = {};
  IREE_ASSERT_OK(loom_module_intern_type(
      module_, loom_type_dialect(first_name, 1, &scalar), &first_dialect));
  IREE_ASSERT_OK(loom_module_intern_type(
      module_, loom_type_dialect(second_name, 1, &scalar), &second_dialect));
  expect_unequal(first_dialect, second_dialect);

  loom_type_t first_register = {};
  loom_type_t second_register = {};
  IREE_ASSERT_OK(loom_module_intern_register_type(
      module_, /*carrier_payload0=*/1, /*carrier_payload1=*/2, scalar,
      &first_register));
  IREE_ASSERT_OK(loom_module_intern_register_type(
      module_, /*carrier_payload0=*/1, /*carrier_payload1=*/3, scalar,
      &second_register));
  expect_unequal(first_register, second_register);

  const std::array<loom_overflow_dim_t, 3> first_dimensions = {
      loom_dim_pack_static(2), loom_dim_pack_dynamic(source_value_),
      loom_dim_pack_static(4)};
  const std::array<loom_overflow_dim_t, 3> second_dimensions = {
      loom_dim_pack_static(3), loom_dim_pack_dynamic(target_value_),
      loom_dim_pack_static(4)};
  auto intern_overflow = [&](const auto& dimensions) {
    loom_type_t type = {};
    type.header = loom_type_make_header(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                                        dimensions.size(), 0);
    type.dims[0] = reinterpret_cast<uintptr_t>(dimensions.data());
    loom_type_t canonical = {};
    IREE_CHECK_OK(loom_module_intern_type(module_, type, &canonical));
    return canonical;
  };
  expect_unequal(intern_overflow(first_dimensions),
                 intern_overflow(second_dimensions));

  const loom_parameterized_type_descriptor_t first_descriptor = {
      /*.name=*/LOOM_BSTRING_REF(10, "test.first"),
      /*.parameter_descriptors=*/nullptr,
      /*.ir_kind=*/LOOM_TYPE_PARAMETERIZED,
      /*.type_flags=*/0,
      /*.parameter_count=*/0,
  };
  const loom_parameterized_type_descriptor_t second_descriptor = {
      /*.name=*/LOOM_BSTRING_REF(11, "test.second"),
      /*.parameter_descriptors=*/nullptr,
      /*.ir_kind=*/LOOM_TYPE_PARAMETERIZED,
      /*.type_flags=*/0,
      /*.parameter_count=*/0,
  };
  loom_type_t first_parameterized = {};
  loom_type_t second_parameterized = {};
  IREE_ASSERT_OK(loom_module_make_parameterized_type(
      module_, &first_descriptor, nullptr, 0, &first_parameterized, nullptr));
  IREE_ASSERT_OK(loom_module_make_parameterized_type(
      module_, &second_descriptor, nullptr, 0, &second_parameterized, nullptr));
  expect_unequal(first_parameterized, second_parameterized);

  EXPECT_NE(query.exact_state, nullptr);
  loom_type_remap_query_deinitialize(&query);
}

TEST_F(TypeRemapQueryTest, ShiftedProductUsesLinearScratch) {
  constexpr uint32_t kDepth = 512;
  std::vector<loom_type_t> chain(kDepth + 2);
  chain[0] = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  for (uint32_t i = 0; i <= kDepth; ++i) {
    chain[i + 1] = InternFunction(&chain[i], 1);
  }

  auto run_product = [&](uint32_t depth) -> std::array<iree_host_size_t, 2> {
    loom_type_remap_query_t query;
    loom_type_remap_query_initialize(module_, &remap_, &query);
    bool equal = true;
    IREE_CHECK_OK(loom_type_remap_query_equal(
        &query, loom_type_scalar(LOOM_SCALAR_TYPE_F32),
        loom_type_scalar(LOOM_SCALAR_TYPE_F64), &equal));
    EXPECT_FALSE(equal);
    for (uint32_t i = 1; i <= depth; ++i) {
      IREE_CHECK_OK(
          loom_type_remap_query_equal(&query, chain[i], chain[i + 1], &equal));
      EXPECT_FALSE(equal);
    }
    EXPECT_EQ(query.scratch_arena.allocation_head, nullptr);
    const std::array<iree_host_size_t, 2> sizes = {
        query.scratch_arena.used_allocation_size,
        query.scratch_arena.total_allocation_size,
    };
    loom_type_remap_query_deinitialize(&query);
    return sizes;
  };

  const auto size_256 = run_product(256);
  const auto size_512 = run_product(512);
  EXPECT_LE(size_512[0], size_256[0] * 2 + 4096);
  EXPECT_LE(size_512[1], size_256[1] * 2 + 4096);
}

TEST_F(TypeRemapQueryTest, EveryScratchBlockFailurePropagates) {
  constexpr uint32_t kDepth = 256;
  std::vector<loom_type_t> source_chain(kDepth + 1);
  std::vector<loom_type_t> target_chain(kDepth + 1);
  source_chain[0] =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(source_value_), 0);
  target_chain[0] =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(target_value_), 0);
  for (uint32_t i = 0; i < kDepth; ++i) {
    source_chain[i + 1] = InternFunction(&source_chain[i], 1);
    target_chain[i + 1] = InternFunction(&target_chain[i], 1);
  }

  auto exercise_failures = [&](bool enter_exact_mode) {
    const iree_allocator_t system_allocator = pool_.block_allocator;
    bool reached_success = false;
    uint32_t successful_allocation_count = 0;
    for (uint32_t failure_index = 0; failure_index < 64; ++failure_index) {
      iree_arena_block_pool_trim(&pool_);
      FailingAllocatorState state = {
          /*.system_allocator=*/system_allocator,
          /*.failure_index=*/failure_index,
          /*.allocation_count=*/0,
      };
      pool_.block_allocator = {/*.self=*/&state,
                               /*.ctl=*/FailSelectedAllocation};
      loom_type_remap_query_t query;
      loom_type_remap_query_initialize(module_, &remap_, &query);
      bool equal = true;
      if (enter_exact_mode) {
        IREE_CHECK_OK(loom_type_remap_query_equal(
            &query, loom_type_scalar(LOOM_SCALAR_TYPE_F32),
            loom_type_scalar(LOOM_SCALAR_TYPE_F64), &equal));
        EXPECT_FALSE(equal);
      }
      iree_status_t status = loom_type_remap_query_equal(
          &query, source_chain.back(), target_chain.back(), &equal);
      loom_type_remap_query_deinitialize(&query);
      pool_.block_allocator = system_allocator;
      iree_arena_block_pool_trim(&pool_);

      if (iree_status_is_ok(status)) {
        EXPECT_TRUE(equal);
        reached_success = true;
        successful_allocation_count = state.allocation_count;
        break;
      }
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      EXPECT_EQ(state.allocation_count, failure_index + 1);
    }
    EXPECT_TRUE(reached_success);
    EXPECT_GT(successful_allocation_count, 2u);
  };

  exercise_failures(/*enter_exact_mode=*/false);
  exercise_failures(/*enter_exact_mode=*/true);
}

}  // namespace
}  // namespace loom
