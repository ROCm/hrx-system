// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer/catalog.h"

#include <cstdio>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/parameterized_attr.h"
#include "loom/ir/parameterized_type.h"

namespace loom {
namespace {

static const loom_attr_descriptor_t kParameters[] = {
    {/*.name=*/LOOM_BSTRING_REF(5, "block"),
     /*.attr_kind=*/LOOM_ATTR_I64},
    {/*.name=*/LOOM_BSTRING_REF(4, "left"),
     /*.attr_kind=*/LOOM_ATTR_ENCODING,
     /*.flags=*/LOOM_ATTR_OPTIONAL},
    {/*.name=*/LOOM_BSTRING_REF(8, "metadata"),
     /*.attr_kind=*/LOOM_ATTR_DICT,
     /*.flags=*/LOOM_ATTR_OPTIONAL},
    {/*.name=*/LOOM_BSTRING_REF(5, "right"),
     /*.attr_kind=*/LOOM_ATTR_ENCODING,
     /*.flags=*/LOOM_ATTR_OPTIONAL},
};
static const loom_encoding_family_descriptor_t kDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(12, "test.catalog"),
    /*.role=*/LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
    /*.family_flags=*/{},
    /*.parameter_count=*/IREE_ARRAYSIZE(kParameters),
    /*.parameter_descriptors=*/kParameters,
};
static const loom_encoding_vtable_t kVtable = {/*.descriptor=*/&kDescriptor};

static const loom_attr_descriptor_t kPayloadParameters[] = {
    {/*.name=*/LOOM_BSTRING_REF(7, "element"),
     /*.attr_kind=*/LOOM_ATTR_TYPE},
    {/*.name=*/LOOM_BSTRING_REF(5, "label"),
     /*.attr_kind=*/LOOM_ATTR_STRING,
     /*.flags=*/LOOM_ATTR_OPTIONAL},
};
static const loom_parameterized_attr_descriptor_t kPayloadDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(20, "test.catalog_payload"),
    /*.kind=*/LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_TEST, 0),
    /*.parameter_count=*/IREE_ARRAYSIZE(kPayloadParameters),
    /*.primary_parameter_index=*/LOOM_PARAMETERIZED_ATTR_NO_PRIMARY_PARAMETER,
    /*.parameter_descriptors=*/kPayloadParameters,
};

// Writer arena whose backing allocation attempts can fail deterministically.
class FailingWriterArena {
 public:
  FailingWriterArena() {
    iree_arena_block_pool_initialize(128, {this, Allocate}, &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }

  ~FailingWriterArena() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void FailAt(iree_host_size_t failure_index) {
    allocation_count_ = 0;
    failure_index_ = failure_index;
  }

  void DisableFailures() { failure_index_ = SIZE_MAX; }

  iree_host_size_t allocation_count() const { return allocation_count_; }

  iree_arena_allocator_t* arena() { return &arena_; }

 private:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* arena = static_cast<FailingWriterArena*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        arena->allocation_count_++ == arena->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected writer allocation failure");
    }
    const iree_allocator_t allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  // Number of backing allocation attempts since the last failure selection.
  iree_host_size_t allocation_count_ = 0;
  // Backing allocation ordinal to fail, or SIZE_MAX when disabled.
  iree_host_size_t failure_index_ = SIZE_MAX;
  // Tiny blocks expose every catalog segment and directory allocation.
  iree_arena_block_pool_t pool_ = {};
  // Invocation-owned writer storage under test.
  iree_arena_allocator_t arena_ = {};
};

class CatalogTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_register_encoding_vtable(&context_, &kVtable));
    IREE_ASSERT_OK(loom_context_register_parameterized_attrs(
        &context_, LOOM_DIALECT_TEST, &kPayloadDescriptor, 1));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("catalog"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_string_id_t Intern(iree_string_view_t name) {
    loom_string_id_t id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &id));
    return id;
  }

  loom_type_id_t InternType(loom_type_t type) {
    loom_type_id_t id = LOOM_TYPE_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_type_id(module_, type, &id));
    return id;
  }

  loom_type_id_t Pair(loom_type_id_t child) {
    alignas(loom_func_type_data_t) unsigned char
        storage[sizeof(loom_func_type_data_t) + 2 * sizeof(loom_type_t)] = {};
    auto* data = reinterpret_cast<loom_func_type_data_t*>(storage);
    data->arg_count = 2;
    const loom_type_id_t children[] = {child, child};
    loom_type_id_t id = LOOM_TYPE_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_topological_type_id(
        module_, loom_type_function(data), children, 2, &id));
    return id;
  }

  uint16_t AddEncoding(loom_named_attr_slice_t parameters,
                       loom_string_id_t alias = LOOM_STRING_ID_INVALID) {
    const loom_encoding_t encoding = {
        /*.name_id=*/Intern(IREE_SV("test.catalog")),
        /*.alias_id=*/alias,
        /*.attribute_count=*/static_cast<uint8_t>(parameters.count),
        /*.family=*/{},
        /*.attributes=*/parameters.entries,
    };
    uint16_t id = 0;
    IREE_CHECK_OK(loom_module_add_encoding(module_, &encoding, &id));
    EXPECT_TRUE(
        loom_encoding_static_is_valid(loom_module_encoding(module_, id)));
    return id;
  }

  uint16_t AddDependentEncoding(
      uint16_t child, uint8_t fanout, int64_t block_size,
      loom_string_id_t alias = LOOM_STRING_ID_INVALID) {
    const loom_named_attr_t parameters[] = {
        {Intern(IREE_SV("block")), {}, loom_attr_i64(block_size)},
        {Intern(IREE_SV("left")), {}, loom_attr_encoding(child)},
        {Intern(IREE_SV("right")), {}, loom_attr_encoding(child)},
    };
    return AddEncoding(loom_make_named_attr_slice(parameters, 1 + fanout),
                       alias);
  }

  iree_status_t NumberEncodings(loom_bytecode_numbering_t* numbering) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_numbering_initialize(numbering, module_, &arena_));
    for (iree_host_size_t i = 0; i < module_->encodings.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_bytecode_number_encoding(
          numbering, static_cast<uint16_t>(i + 1)));
    }
    return iree_ok_status();
  }

  // Blocks shared by source IR and invocation scratch.
  iree_arena_block_pool_t pool_;
  // Invocation-owned catalog storage.
  iree_arena_allocator_t arena_;
  // Registry containing the encoding family used by the source module.
  loom_context_t context_;
  // Immutable source module after catalog initialization.
  loom_module_t* module_ = nullptr;
};

TEST_F(CatalogTest, EmptyEncodingTable) {
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(NumberEncodings(&numbering));
  EXPECT_EQ(loom_bytecode_numbering_string_count(&numbering), 1u);
  EXPECT_EQ(numbering.types.count, 0u);
}

TEST_F(CatalogTest, EncodingOrderPreservesFirstUseAndAliases) {
  uint16_t child = AddDependentEncoding(0, 0, 4, Intern(IREE_SV("root")));
  child = AddDependentEncoding(child, 1, 8, Intern(IREE_SV("tail")));
  child = AddDependentEncoding(child, 2, 16);
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(NumberEncodings(&numbering));
  const iree_string_view_t expected[] = {
      IREE_SV(""),      IREE_SV("test.catalog"), IREE_SV("root"),
      IREE_SV("block"), IREE_SV("tail"),         IREE_SV("left"),
      IREE_SV("right"),
  };
  ASSERT_EQ(loom_bytecode_numbering_string_count(&numbering),
            IREE_ARRAYSIZE(expected));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    EXPECT_TRUE(iree_string_view_equal(
        loom_bytecode_numbering_string(&numbering, static_cast<uint32_t>(i)),
        expected[i]));
  }
  IREE_ASSERT_OK(loom_bytecode_number_attr_value(
      &numbering, loom_attr_encoding(child), nullptr));
  EXPECT_EQ(loom_bytecode_numbering_string_count(&numbering),
            IREE_ARRAYSIZE(expected));
  EXPECT_EQ(numbering.types.count, 0u);
}

TEST_F(CatalogTest, SharedEncodingDependenciesStayBounded) {
  uint16_t child = AddDependentEncoding(0, 0, 1);
  for (int64_t block_size = 2; block_size <= 257; ++block_size) {
    child = AddDependentEncoding(child, 2, block_size);
  }
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(NumberEncodings(&numbering));
  EXPECT_EQ(module_->encodings.count, 257u);
  EXPECT_EQ(loom_bytecode_numbering_string_count(&numbering), 5u);
  EXPECT_EQ(numbering.types.count, 0u);
}

TEST_F(CatalogTest, DeepEncodingDependenciesNeedNoRecursiveStack) {
  uint16_t child = AddDependentEncoding(0, 0, 1);
  for (int64_t block_size = 2; block_size <= 8192; ++block_size) {
    child = AddDependentEncoding(child, 1, block_size);
  }
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(NumberEncodings(&numbering));
  EXPECT_EQ(module_->encodings.count, 8192u);
  EXPECT_EQ(loom_bytecode_numbering_string_count(&numbering), 4u);
}

TEST_F(CatalogTest, EncodingPayloadsNumberNestedTypesAndStrings) {
  const uint16_t base = AddDependentEncoding(0, 0, 4);
  loom_type_id_t element = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_F32), &element));
  const loom_string_id_t label = Intern(IREE_SV("element_label"));
  const loom_named_attr_t metadata_entries[] = {
      {Intern(IREE_SV("element")), {}, loom_attr_type(element)},
      {Intern(IREE_SV("label")), {}, loom_attr_string(label)},
  };
  loom_attribute_t metadata;
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module_, loom_make_named_attr_slice(metadata_entries, 2), &metadata));
  const loom_named_attr_t parameters[] = {
      {Intern(IREE_SV("block")), {}, loom_attr_i64(8)},
      {Intern(IREE_SV("left")), {}, loom_attr_encoding(base)},
      {Intern(IREE_SV("metadata")), {}, metadata},
  };
  AddEncoding(
      loom_make_named_attr_slice(parameters, IREE_ARRAYSIZE(parameters)));
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(NumberEncodings(&numbering));
  ASSERT_EQ(numbering.types.count, 1u);
  EXPECT_EQ(numbering.types.module_indices_by_writer_id[0], element);
  uint32_t label_id = 0;
  IREE_ASSERT_OK(loom_bytecode_numbering_intern_module_string(&numbering, label,
                                                              &label_id));
  ASSERT_LT(label_id, loom_bytecode_numbering_string_count(&numbering));
  EXPECT_TRUE(iree_string_view_equal(
      loom_bytecode_numbering_string(&numbering, label_id),
      IREE_SV("element_label")));
}

TEST_F(CatalogTest, SharedTypeNumberingRetainsCompletedResults) {
  std::vector<loom_type_id_t> types = {
      InternType(loom_type_scalar(LOOM_SCALAR_TYPE_F32))};
  for (int level = 0; level < 8192; ++level) {
    types.push_back(Pair(types.back()));
  }
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(
      loom_bytecode_numbering_initialize(&numbering, module_, &arena_));
  uint32_t writer_id = 0;
  IREE_ASSERT_OK(loom_bytecode_numbering_intern_type(
      &numbering, loom_type_table_get(&module_->types, types.back()),
      &writer_id, nullptr));
  ASSERT_EQ(numbering.types.count, types.size());
  EXPECT_EQ(writer_id, types.size() - 1);
  const auto completed_storage = arena_.used_allocation_size;
  for (size_t i = 0; i < types.size(); ++i) {
    EXPECT_EQ(numbering.types.module_indices_by_writer_id[i], types[i]);
    IREE_ASSERT_OK(loom_bytecode_numbering_intern_type(
        &numbering, loom_type_table_get(&module_->types, types[i]), &writer_id,
        nullptr));
    EXPECT_EQ(writer_id, i);
  }
  EXPECT_EQ(numbering.types.count, types.size());
  EXPECT_EQ(arena_.used_allocation_size, completed_storage);
}

TEST_F(CatalogTest, KnownModuleIdsPreserveFirstUseOrder) {
  const auto leaf = InternType(loom_type_scalar(LOOM_SCALAR_TYPE_F32));
  const auto other = InternType(loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  const auto parent = Pair(leaf);
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(
      loom_bytecode_numbering_initialize(&numbering, module_, &arena_));
  uint32_t writer_id = 0;
  IREE_ASSERT_OK(loom_bytecode_numbering_intern_module_type(&numbering, other,
                                                            &writer_id));
  EXPECT_EQ(writer_id, 0u);
  IREE_ASSERT_OK(loom_bytecode_numbering_intern_module_type(&numbering, parent,
                                                            &writer_id));
  EXPECT_EQ(writer_id, 2u);
  loom_type_id_t module_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_bytecode_numbering_intern_type(
      &numbering, loom_type_table_get(&module_->types, leaf), &writer_id,
      &module_id));
  EXPECT_EQ(writer_id, 1u);
  EXPECT_EQ(module_id, leaf);
  EXPECT_EQ(numbering.types.count, 3u);
}

TEST_F(CatalogTest, TypeAllocationFailureLeavesSourceReusable) {
  auto root = InternType(loom_type_scalar(LOOM_SCALAR_TYPE_F32));
  for (size_t i = 0; i < 64; ++i) {
    root = Pair(root);
  }
  const auto source_storage = module_->arena.used_allocation_size;
  const auto source_type_count = module_->types.count;
  for (iree_host_size_t failure_index = 0;; ++failure_index) {
    SCOPED_TRACE(failure_index);
    bool completed = false;
    {
      FailingWriterArena writer_arena;
      writer_arena.FailAt(failure_index);
      loom_bytecode_numbering_t numbering;
      iree_status_t status = loom_bytecode_numbering_initialize(
          &numbering, module_, writer_arena.arena());
      uint32_t writer_id = 0;
      if (iree_status_is_ok(status)) {
        status = loom_bytecode_numbering_intern_module_type(&numbering, root,
                                                            &writer_id);
      }
      completed = iree_status_is_ok(status);
      if (completed) {
        EXPECT_EQ(numbering.types.count, source_type_count);
      } else {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      }
    }
    EXPECT_EQ(module_->arena.used_allocation_size, source_storage);
    EXPECT_EQ(module_->types.count, source_type_count);
    if (completed) {
      break;
    }

    // A new invocation can number the same immutable source after every
    // possible failure boundary and release of the failed invocation's arena.
    FailingWriterArena retry_arena;
    loom_bytecode_numbering_t numbering;
    IREE_ASSERT_OK(loom_bytecode_numbering_initialize(&numbering, module_,
                                                      retry_arena.arena()));
    uint32_t writer_id = 0;
    IREE_ASSERT_OK(loom_bytecode_numbering_intern_module_type(&numbering, root,
                                                              &writer_id));
    EXPECT_EQ(writer_id, source_type_count - 1);
  }
}

TEST_F(CatalogTest, NumberingRetainsDistinctScopedTypeIdentities) {
  const loom_type_t dimension_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_id_t types[2];
  for (size_t i = 0; i < 2; ++i) {
    loom_value_id_t dimension = LOOM_VALUE_ID_INVALID;
    IREE_ASSERT_OK(
        loom_module_define_value(module_, dimension_type, &dimension));
    types[i] = Pair(InternType(loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
        loom_dim_pack_dynamic(dimension), /*encoding_id=*/0)));
  }
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(
      loom_bytecode_numbering_initialize(&numbering, module_, &arena_));
  uint32_t writer_ids[2] = {};
  loom_type_id_t type_ids[2] = {};
  for (size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(loom_bytecode_numbering_intern_type(
        &numbering, loom_type_table_get(&module_->types, types[i]),
        &writer_ids[i], &type_ids[i]));
    EXPECT_EQ(type_ids[i], types[i]);
    EXPECT_TRUE(numbering.types.index.nodes[type_ids[i]].has_bindings);
  }
  EXPECT_EQ(writer_ids[0], writer_ids[1]);
  EXPECT_NE(type_ids[0], type_ids[1]);
  // Bound types have no global wire entry; only their scalar leaf is global.
  EXPECT_EQ(numbering.types.count, 1u);
}

TEST_F(CatalogTest, TypeAndAttributeMetadataKeepFirstUseOrder) {
  static const loom_attr_descriptor_t variants_parameter = [] {
    loom_attr_descriptor_t descriptor = {
        /*.name=*/LOOM_BSTRING_REF(8, "variants"),
        /*.attr_kind=*/LOOM_ATTR_PARAMETERIZED_ARRAY,
    };
    descriptor.reference.parameterized_attr_kind =
        LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_TEST, 0);
    return descriptor;
  }();
  static const loom_attr_descriptor_t parameters[] = {
      {/*.name=*/LOOM_BSTRING_REF(6, "before"),
       /*.attr_kind=*/LOOM_ATTR_STRING},
      {/*.name=*/LOOM_BSTRING_REF(8, "metadata"),
       /*.attr_kind=*/LOOM_ATTR_DICT},
      variants_parameter,
      {/*.name=*/LOOM_BSTRING_REF(5, "after"),
       /*.attr_kind=*/LOOM_ATTR_STRING},
      {/*.name=*/LOOM_BSTRING_REF(8, "optional"),
       /*.attr_kind=*/LOOM_ATTR_TYPE,
       /*.flags=*/LOOM_ATTR_OPTIONAL},
  };
  static const loom_parameterized_type_descriptor_t descriptor = {
      /*.name=*/LOOM_BSTRING_REF(17, "test.catalog_type"),
      /*.parameter_descriptors=*/parameters,
      /*.ir_kind=*/LOOM_TYPE_PARAMETERIZED,
      /*.type_flags=*/{},
      /*.parameter_count=*/IREE_ARRAYSIZE(parameters),
  };
  // Module insertion order intentionally differs from first-use wire order.
  const auto late = Intern(IREE_SV("late"));
  const auto inner_label = Intern(IREE_SV("inner_label"));
  const auto early = Intern(IREE_SV("early"));
  const auto scalar = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const auto element = InternType(scalar);
  const auto child =
      InternType(loom_type_dialect(Intern(IREE_SV("test.child")), 1, &scalar));
  const loom_named_attr_t entry = {
      Intern(IREE_SV("shape")), {}, loom_attr_type(child)};
  loom_attribute_t metadata;
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module_, loom_make_named_attr_slice(&entry, 1), &metadata));
  const loom_attribute_t payload_slots[] = {loom_attr_type(child),
                                            loom_attr_string(inner_label)};
  loom_attribute_t payload;
  IREE_ASSERT_OK(loom_module_make_parameterized_attr(
      module_, kPayloadDescriptor.kind, payload_slots, 2, &payload));
  const loom_attribute_t payloads[] = {payload, payload};
  loom_attribute_t variants;
  IREE_ASSERT_OK(loom_module_make_parameterized_attr_array(
      module_, loom_make_parameterized_attr_array(payloads, 2), &variants));
  const loom_attribute_t slots[] = {loom_attr_string(early), metadata, variants,
                                    loom_attr_string(late), loom_attr_absent()};
  loom_type_t type;
  loom_type_id_t type_id;
  IREE_ASSERT_OK(loom_module_make_parameterized_type(
      module_, &descriptor, slots, IREE_ARRAYSIZE(slots), &type, &type_id));
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(
      loom_bytecode_numbering_initialize(&numbering, module_, &arena_));
  uint32_t writer_id = 0;
  IREE_ASSERT_OK(loom_bytecode_numbering_intern_type(&numbering, type,
                                                     &writer_id, nullptr));
  const loom_type_id_t expected_types[] = {element, child, type_id};
  ASSERT_EQ(numbering.types.count, IREE_ARRAYSIZE(expected_types));
  for (size_t i = 0; i < IREE_ARRAYSIZE(expected_types); ++i) {
    EXPECT_EQ(numbering.types.module_indices_by_writer_id[i],
              expected_types[i]);
  }
  const iree_string_view_t expected_strings[] = {
      IREE_SV(""),
      IREE_SV("test.catalog_type"),
      IREE_SV("before"),
      IREE_SV("early"),
      IREE_SV("metadata"),
      IREE_SV("shape"),
      IREE_SV("test.child"),
      IREE_SV("variants"),
      IREE_SV("test.catalog_payload"),
      IREE_SV("element"),
      IREE_SV("label"),
      IREE_SV("inner_label"),
      IREE_SV("after"),
      IREE_SV("late"),
  };
  ASSERT_EQ(loom_bytecode_numbering_string_count(&numbering),
            IREE_ARRAYSIZE(expected_strings));
  for (size_t i = 0; i < IREE_ARRAYSIZE(expected_strings); ++i) {
    EXPECT_TRUE(iree_string_view_equal(
        loom_bytecode_numbering_string(&numbering, static_cast<uint32_t>(i)),
        expected_strings[i]));
  }
  const auto completed_storage = arena_.used_allocation_size;
  IREE_ASSERT_OK(
      loom_bytecode_number_attr_value(&numbering, variants, &parameters[2]));
  EXPECT_EQ(loom_bytecode_numbering_string_count(&numbering),
            IREE_ARRAYSIZE(expected_strings));
  EXPECT_EQ(numbering.types.count, IREE_ARRAYSIZE(expected_types));
  EXPECT_EQ(arena_.used_allocation_size, completed_storage);
}

TEST_F(CatalogTest, ParameterizedTypesResumeAfterNestedTypes) {
  static const loom_parameterized_type_descriptor_t descriptor = {
      /*.name=*/LOOM_BSTRING_REF(10, "test.chain"),
      /*.parameter_descriptors=*/kPayloadParameters,
      /*.ir_kind=*/LOOM_TYPE_PARAMETERIZED,
      /*.type_flags=*/{},
      /*.parameter_count=*/IREE_ARRAYSIZE(kPayloadParameters),
  };
  const auto label = Intern(IREE_SV("chain_label"));
  std::vector<loom_type_id_t> types = {
      InternType(loom_type_scalar(LOOM_SCALAR_TYPE_F32))};
  loom_type_t type;
  for (int level = 0; level < 512; ++level) {
    const loom_attribute_t slots[] = {loom_attr_type(types.back()),
                                      loom_attr_string(label)};
    loom_type_id_t id;
    IREE_ASSERT_OK(loom_module_make_parameterized_type(module_, &descriptor,
                                                       slots, 2, &type, &id));
    types.push_back(id);
  }
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(
      loom_bytecode_numbering_initialize(&numbering, module_, &arena_));
  uint32_t writer_id = 0;
  IREE_ASSERT_OK(loom_bytecode_numbering_intern_type(&numbering, type,
                                                     &writer_id, nullptr));
  ASSERT_EQ(numbering.types.count, types.size());
  for (size_t i = 0; i < types.size(); ++i) {
    EXPECT_EQ(numbering.types.module_indices_by_writer_id[i], types[i]);
  }
  const iree_string_view_t expected[] = {
      IREE_SV(""),      IREE_SV("test.chain"),  IREE_SV("element"),
      IREE_SV("label"), IREE_SV("chain_label"),
  };
  ASSERT_EQ(loom_bytecode_numbering_string_count(&numbering),
            IREE_ARRAYSIZE(expected));
  for (size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    EXPECT_TRUE(iree_string_view_equal(
        loom_bytecode_numbering_string(&numbering, static_cast<uint32_t>(i)),
        expected[i]));
  }
}

TEST_F(CatalogTest, SymbolOrderRetainsOrdinaryModuleOrder) {
  constexpr size_t kSymbolCount = LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_CAPACITY;
  std::vector<loom_symbol_id_t> symbol_ids;
  symbol_ids.reserve(kSymbolCount);
  for (size_t i = 0; i < kSymbolCount; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "symbol_%08zu", i);
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(
        module_, Intern(iree_make_cstring_view(name)), &symbol_id));
    symbol_ids.push_back(symbol_id);
  }

  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(
      loom_bytecode_numbering_initialize(&numbering, module_, &arena_));
  for (size_t i = 0; i < kSymbolCount; ++i) {
    EXPECT_EQ(loom_bytecode_module_symbol_id(&numbering,
                                             static_cast<loom_symbol_id_t>(i)),
              symbol_ids[i]);
    EXPECT_EQ(loom_bytecode_wire_symbol_ordinal(&numbering, symbol_ids[i]), i);
  }
}

TEST_F(CatalogTest, SymbolOrderUsesPoolSizedStorage) {
  constexpr size_t kSymbolCount =
      4 * LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_CAPACITY + 1;
  std::vector<loom_symbol_id_t> symbol_ids;
  symbol_ids.reserve(kSymbolCount);
  for (size_t i = 0; i < kSymbolCount; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "symbol_%08zu", i);
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(
        module_, Intern(iree_make_cstring_view(name)), &symbol_id));
    symbol_ids.push_back(symbol_id);
  }

  iree_arena_block_pool_statistics_t before;
  iree_arena_block_pool_query_statistics(&pool_, &before);
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(
      loom_bytecode_numbering_initialize(&numbering, module_, &arena_));
  iree_arena_block_pool_statistics_t after;
  iree_arena_block_pool_query_statistics(&pool_, &after);
  EXPECT_EQ(after.oversized_allocation_count,
            before.oversized_allocation_count);
  EXPECT_EQ(after.oversized_allocation_bytes,
            before.oversized_allocation_bytes);

  constexpr size_t kSegmentCapacity =
      LOOM_BYTECODE_SYMBOL_ORDER_SEGMENT_CAPACITY;
  const size_t indices[] = {0,
                            kSegmentCapacity - 1,
                            kSegmentCapacity,
                            2 * kSegmentCapacity - 1,
                            2 * kSegmentCapacity,
                            4 * kSegmentCapacity};
  for (size_t index : indices) {
    EXPECT_EQ(loom_bytecode_module_symbol_id(
                  &numbering, static_cast<loom_symbol_id_t>(index)),
              symbol_ids[index]);
    EXPECT_EQ(loom_bytecode_wire_symbol_ordinal(&numbering, symbol_ids[index]),
              index);
  }
}

TEST_F(CatalogTest, StringCatalogRetainsFirstUseOrderAtScale) {
  constexpr size_t kStringCount = 257;
  std::vector<std::string> names;
  std::vector<loom_string_id_t> module_ids;
  names.reserve(kStringCount);
  module_ids.reserve(kStringCount);
  for (size_t i = 0; i < kStringCount; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "catalog_name_%08zu", i);
    names.emplace_back(name);
    module_ids.push_back(Intern(
        iree_make_string_view(names.back().data(), names.back().size())));
  }

  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(
      loom_bytecode_numbering_initialize(&numbering, module_, &arena_));
  for (size_t ordinal = kStringCount; ordinal > 0; --ordinal) {
    uint32_t writer_id = 0;
    IREE_ASSERT_OK(loom_bytecode_numbering_intern_module_string(
        &numbering, module_ids[ordinal - 1], &writer_id));
    EXPECT_EQ(writer_id, kStringCount - ordinal + 1);
  }

  ASSERT_EQ(loom_bytecode_numbering_string_count(&numbering), kStringCount + 1);
  EXPECT_TRUE(
      iree_string_view_is_empty(loom_bytecode_numbering_string(&numbering, 0)));
  for (uint32_t writer_id = 1; writer_id <= kStringCount; ++writer_id) {
    const std::string& expected = names[kStringCount - writer_id];
    EXPECT_TRUE(iree_string_view_equal(
        loom_bytecode_numbering_string(&numbering, writer_id),
        iree_make_string_view(expected.data(), expected.size())));
  }

  for (size_t i = 0; i < kStringCount; ++i) {
    uint32_t writer_id = 0;
    IREE_ASSERT_OK(loom_bytecode_numbering_intern_module_string(
        &numbering, module_ids[i], &writer_id));
    EXPECT_EQ(writer_id, kStringCount - i);
  }
  EXPECT_EQ(loom_bytecode_numbering_string_count(&numbering), kStringCount + 1);
}

TEST_F(CatalogTest, StringViewsReuseModuleAndExternalIdentities) {
  const loom_string_id_t module_id = Intern(IREE_SV("module_owned"));
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(
      loom_bytecode_numbering_initialize(&numbering, module_, &arena_));

  std::string module_spelling = "module_owned";
  uint32_t module_view_writer_id = 0;
  IREE_ASSERT_OK(loom_bytecode_numbering_intern_string_view(
      &numbering,
      iree_make_string_view(module_spelling.data(), module_spelling.size()),
      &module_view_writer_id));
  uint32_t module_id_writer_id = 0;
  IREE_ASSERT_OK(loom_bytecode_numbering_intern_module_string(
      &numbering, module_id, &module_id_writer_id));
  EXPECT_EQ(module_view_writer_id, module_id_writer_id);
  EXPECT_EQ(loom_bytecode_numbering_string_count(&numbering), 2u);

  std::string first_external_spelling = "external_only";
  std::string second_external_spelling = first_external_spelling;
  uint32_t first_external_writer_id = 0;
  IREE_ASSERT_OK(loom_bytecode_numbering_intern_string_view(
      &numbering,
      iree_make_string_view(first_external_spelling.data(),
                            first_external_spelling.size()),
      &first_external_writer_id));
  uint32_t second_external_writer_id = 0;
  IREE_ASSERT_OK(loom_bytecode_numbering_intern_string_view(
      &numbering,
      iree_make_string_view(second_external_spelling.data(),
                            second_external_spelling.size()),
      &second_external_writer_id));
  EXPECT_EQ(first_external_writer_id, second_external_writer_id);
  EXPECT_EQ(loom_bytecode_numbering_string_count(&numbering), 3u);
  EXPECT_TRUE(iree_string_view_equal(
      loom_bytecode_numbering_string(&numbering, first_external_writer_id),
      IREE_SV("external_only")));
}

TEST_F(CatalogTest, StringCatalogUsesPoolSizedStorage) {
  constexpr size_t kStringCount = 513;
  std::vector<loom_string_id_t> module_ids;
  std::vector<std::string> external_names;
  module_ids.reserve(kStringCount);
  external_names.reserve(kStringCount);
  for (size_t i = 0; i < kStringCount; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "module_string_%08zu", i);
    module_ids.push_back(Intern(iree_make_cstring_view(name)));
    std::snprintf(name, sizeof(name), "external_name_%08zu", i);
    external_names.emplace_back(name);
  }

  iree_arena_block_pool_statistics_t before;
  iree_arena_block_pool_query_statistics(&pool_, &before);
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(
      loom_bytecode_numbering_initialize(&numbering, module_, &arena_));
  uint32_t writer_id = 0;
  for (loom_string_id_t module_id : module_ids) {
    IREE_ASSERT_OK(loom_bytecode_numbering_intern_module_string(
        &numbering, module_id, &writer_id));
  }
  const size_t module_indices[] = {512, 0, 511, 128, 127};
  for (size_t module_index : module_indices) {
    IREE_ASSERT_OK(loom_bytecode_numbering_intern_module_string(
        &numbering, module_ids[module_index], &writer_id));
    EXPECT_EQ(writer_id, module_index + 1);
  }
  for (const std::string& name : external_names) {
    IREE_ASSERT_OK(loom_bytecode_numbering_intern_string_view(
        &numbering, iree_make_string_view(name.data(), name.size()),
        &writer_id));
  }
  iree_arena_block_pool_statistics_t after;
  iree_arena_block_pool_query_statistics(&pool_, &after);
  EXPECT_EQ(after.oversized_allocation_count,
            before.oversized_allocation_count);
  EXPECT_EQ(after.oversized_allocation_bytes,
            before.oversized_allocation_bytes);
  EXPECT_EQ(loom_bytecode_numbering_string_count(&numbering),
            2 * kStringCount + 1);
}

TEST_F(CatalogTest, ExternalStringAllocationFailurePreservesCatalog) {
  constexpr size_t kModuleStringCount = 127;
  constexpr size_t kExternalStringCount = 1537;
  std::vector<loom_string_id_t> module_ids;
  std::vector<std::string> external_names;
  module_ids.reserve(kModuleStringCount);
  external_names.reserve(kExternalStringCount);
  for (size_t i = 0; i < kModuleStringCount; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "module_name_%08zu", i);
    module_ids.push_back(Intern(iree_make_cstring_view(name)));
  }
  for (size_t i = 0; i < kExternalStringCount; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "external_name_%08zu", i);
    external_names.emplace_back(name);
  }

  // Each boundary makes the next external insertion grow both the ordered
  // catalog and its content index. The larger case also crosses the intern
  // table's inline segment directory.
  const struct {
    size_t module_count;
    size_t external_count;
  } cases[] = {
      {63, 192},
      {127, 1536},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.external_count);
    for (iree_host_size_t failure_index = 0;; ++failure_index) {
      SCOPED_TRACE(failure_index);
      FailingWriterArena writer_arena;
      loom_bytecode_numbering_t numbering;
      IREE_ASSERT_OK(loom_bytecode_numbering_initialize(&numbering, module_,
                                                        writer_arena.arena()));
      uint32_t writer_id = 0;
      for (size_t i = 0; i < test_case.module_count; ++i) {
        IREE_ASSERT_OK(loom_bytecode_numbering_intern_module_string(
            &numbering, module_ids[i], &writer_id));
      }
      for (size_t i = 0; i < test_case.external_count; ++i) {
        const std::string& name = external_names[i];
        IREE_ASSERT_OK(loom_bytecode_numbering_intern_string_view(
            &numbering, iree_make_string_view(name.data(), name.size()),
            &writer_id));
      }

      const iree_host_size_t count =
          loom_bytecode_numbering_string_count(&numbering);
      writer_arena.FailAt(failure_index);
      const std::string& name = external_names[test_case.external_count];
      const iree_string_view_t view =
          iree_make_string_view(name.data(), name.size());
      writer_id = UINT32_MAX;
      iree_status_t status = loom_bytecode_numbering_intern_string_view(
          &numbering, view, &writer_id);
      writer_arena.DisableFailures();
      if (iree_status_is_ok(status)) {
        EXPECT_LE(writer_arena.allocation_count(), failure_index);
        EXPECT_EQ(writer_id, count);
        EXPECT_EQ(loom_bytecode_numbering_string_count(&numbering), count + 1);
        break;
      }

      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      EXPECT_EQ(writer_arena.allocation_count(), failure_index + 1);
      EXPECT_EQ(writer_id, UINT32_MAX);
      EXPECT_EQ(loom_bytecode_numbering_string_count(&numbering), count);

      // Existing rows remain queryable and retry publishes exactly one new ID.
      const std::string& prior_name =
          external_names[test_case.external_count - 1];
      IREE_ASSERT_OK(loom_bytecode_numbering_intern_string_view(
          &numbering,
          iree_make_string_view(prior_name.data(), prior_name.size()),
          &writer_id));
      EXPECT_EQ(writer_id, count - 1);
      IREE_ASSERT_OK(loom_bytecode_numbering_intern_string_view(
          &numbering, view, &writer_id));
      EXPECT_EQ(writer_id, count);
      EXPECT_EQ(loom_bytecode_numbering_string_count(&numbering), count + 1);
      EXPECT_TRUE(iree_string_view_equal(
          loom_bytecode_numbering_string(&numbering, writer_id), view));
    }
  }
}

}  // namespace
}  // namespace loom
