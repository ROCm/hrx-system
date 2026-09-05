// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/storage.h"

#include <cstdint>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/auxiliary.h"
#include "loom/ops/encoding/families.h"
#include "loom/ops/encoding/operand.h"
#include "loom/ops/encoding/ops.h"

namespace loom {
namespace {

static const loom_encoding_record_field_t* FindRecordField(
    const loom_encoding_record_layout_t* layout,
    loom_encoding_record_field_role_t role, uint8_t hierarchy_level) {
  for (uint8_t i = 0; i < layout->field_count; ++i) {
    const loom_encoding_record_field_t* field = &layout->fields[i];
    if (field->role == role && field->hierarchy_level == hierarchy_level) {
      return field;
    }
  }
  return nullptr;
}

static void ExpectRecordField(const loom_encoding_record_layout_t* layout,
                              loom_encoding_record_field_role_t role,
                              uint8_t hierarchy_level,
                              loom_encoding_numeric_format_t numeric_format,
                              const std::vector<uint8_t>& record,
                              const std::vector<uint64_t>& expected) {
  const loom_encoding_record_field_t* field =
      FindRecordField(layout, role, hierarchy_level);
  ASSERT_NE(field, nullptr);
  EXPECT_EQ(field->numeric_format, numeric_format);
  ASSERT_EQ(field->element_count, expected.size());

  std::vector<uint64_t> actual(field->element_count, 0);
  for (uint8_t i = 0; i < field->mapping_count; ++i) {
    const loom_encoding_record_mapping_t* mapping =
        &layout->mappings[field->first_mapping_index + i];
    ASSERT_LE(mapping->bit_count, 64u);
    ASSERT_LE(mapping->field_bit_offset + mapping->bit_count, 64u);
    for (uint16_t element = 0; element < mapping->element_count; ++element) {
      uint64_t mapped_bits = 0;
      const uint32_t source_bit_offset =
          mapping->record_bit_offset + element * mapping->record_bit_stride;
      for (uint8_t bit = 0; bit < mapping->bit_count; ++bit) {
        const uint32_t source_bit = source_bit_offset + bit;
        ASSERT_LT(source_bit, record.size() * 8);
        mapped_bits |=
            uint64_t{(record[source_bit / 8] >> (source_bit % 8)) & 1u} << bit;
      }
      actual[mapping->field_element_offset + element] |=
          mapped_bits << mapping->field_bit_offset;
    }
  }
  EXPECT_EQ(actual, expected);
}

static void ExpectGgmlRecordTopology(
    const loom_encoding_family_descriptor_t* descriptor,
    const loom_encoding_record_layout_t* layout) {
  std::vector<uint8_t> record(layout->geometry.storage_byte_count);
  for (iree_host_size_t i = 0; i < record.size(); ++i) {
    record[i] = static_cast<uint8_t>(i * 37 + 11);
  }
  auto u16_at = [&](iree_host_size_t byte_offset) {
    return uint64_t{record[byte_offset]} |
           (uint64_t{record[byte_offset + 1]} << 8);
  };

  if (descriptor == &loom_encoding_ggml_q4_0_family_descriptor) {
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SCALE, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_F16, record, {u16_at(0)});
    std::vector<uint64_t> payload(32);
    for (iree_host_size_t i = 0; i < payload.size(); ++i) {
      const uint8_t packed = record[2 + i % 16];
      payload[i] = (packed >> (i / 16 * 4)) & 0xFu;
    }
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_PAYLOAD, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_QUANT_I4, record, payload);
    return;
  }

  if (descriptor == &loom_encoding_ggml_q8_0_family_descriptor) {
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SCALE, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_F16, record, {u16_at(0)});
    std::vector<uint64_t> payload(record.begin() + 2, record.end());
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_PAYLOAD, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_QUANT_I8, record, payload);
    return;
  }

  auto expect_k_scale_minimum_fields = [&]() {
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SCALE, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_F16, record, {u16_at(0)});
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_MINIMUM, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_F16, record, {u16_at(2)});
    std::vector<uint64_t> scales(8);
    std::vector<uint64_t> minimums(8);
    for (iree_host_size_t i = 0; i < 4; ++i) {
      scales[i] = record[4 + i] & 0x3Fu;
      minimums[i] = record[8 + i] & 0x3Fu;
      scales[4 + i] = (record[12 + i] & 0xFu) | ((record[4 + i] >> 6) << 4);
      minimums[4 + i] = (record[12 + i] >> 4) | ((record[8 + i] >> 6) << 4);
    }
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SCALE, 1,
                      LOOM_ENCODING_NUMERIC_FORMAT_U6, record, scales);
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_MINIMUM, 1,
                      LOOM_ENCODING_NUMERIC_FORMAT_U6, record, minimums);
  };

  if (descriptor == &loom_encoding_ggml_q4_k_family_descriptor) {
    expect_k_scale_minimum_fields();
    std::vector<uint64_t> payload(256);
    for (iree_host_size_t i = 0; i < payload.size(); ++i) {
      const iree_host_size_t group = i / 32;
      const uint8_t packed = record[16 + group / 2 * 32 + i % 32];
      payload[i] = (packed >> (group % 2 * 4)) & 0xFu;
    }
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_PAYLOAD, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_U4, record, payload);
    return;
  }

  if (descriptor == &loom_encoding_ggml_q5_k_family_descriptor) {
    expect_k_scale_minimum_fields();
    std::vector<uint64_t> payload(256);
    for (iree_host_size_t i = 0; i < payload.size(); ++i) {
      const iree_host_size_t group = i / 32;
      const uint8_t packed = record[48 + group / 2 * 32 + i % 32];
      const uint8_t high_bit = (record[16 + i % 32] >> group) & 1u;
      payload[i] = ((packed >> (group % 2 * 4)) & 0xFu) | (high_bit << 4);
    }
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_PAYLOAD, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_U5, record, payload);
    return;
  }

  if (descriptor == &loom_encoding_ggml_q6_k_family_descriptor) {
    std::vector<uint64_t> payload(256);
    for (iree_host_size_t i = 0; i < payload.size(); ++i) {
      const iree_host_size_t half = i / 128;
      const iree_host_size_t quarter = i % 128 / 32;
      const iree_host_size_t lane = i % 32;
      const uint8_t low = (record[half * 64 + quarter % 2 * 32 + lane] >>
                           (quarter < 2 ? 0 : 4)) &
                          0xFu;
      const uint8_t high =
          (record[128 + half * 32 + lane] >> (quarter * 2)) & 0x3u;
      payload[i] = low | (high << 4);
    }
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_PAYLOAD, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_QUANT_I6, record, payload);
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SCALE, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_F16, record, {u16_at(208)});
    std::vector<uint64_t> scales(record.begin() + 192, record.begin() + 208);
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SCALE, 1,
                      LOOM_ENCODING_NUMERIC_FORMAT_I8, record, scales);
    return;
  }

  ASSERT_EQ(descriptor, &loom_encoding_ggml_q8_1_x4_family_descriptor);
  std::vector<uint64_t> scales(4);
  std::vector<uint64_t> sums(4);
  for (iree_host_size_t i = 0; i < 4; ++i) {
    scales[i] = u16_at(i * 4);
    sums[i] = u16_at(i * 4 + 2);
  }
  ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SCALE, 0,
                    LOOM_ENCODING_NUMERIC_FORMAT_F16, record, scales);
  ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SUM_CORRECTION, 0,
                    LOOM_ENCODING_NUMERIC_FORMAT_F16, record, sums);
  std::vector<uint64_t> payload(record.begin() + 16, record.end());
  ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_PAYLOAD, 0,
                    LOOM_ENCODING_NUMERIC_FORMAT_QUANT_I8, record, payload);
}

static const loom_encoding_family_fixed_metadata_t kFixedRecordMetadata = {
    /*.operand_summary=*/{},
    /*.required_auxiliary_keys=*/{},
    /*.record=*/
    {
        /*.geometry=*/
        {
            /*.logical_element_count=*/32,
            /*.storage_byte_count=*/18,
            /*.required_alignment=*/2,
        },
    },
};
static const loom_encoding_family_descriptor_t kFixedRecordDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(17, "test.fixed_record"),
    /*.role=*/LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
    /*.family_flags=*/{},
    /*.parameter_count=*/{},
    /*.parameter_descriptors=*/{},
    /*.dynamic_parameter_count=*/{},
    /*.dynamic_parameter_descriptors=*/{},
    /*.fixed_metadata=*/&kFixedRecordMetadata,
};
static const loom_encoding_vtable_t kFixedRecordVtable = {
    /*.descriptor=*/&kFixedRecordDescriptor,
};

class EncodingStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_encoding_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_ENCODING, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_register_builtin_encoding_vtables(&context_));
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kFixedRecordVtable));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(iree_string_view_t source) {
    loom_text_parse_options_t options = {};
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(source, IREE_SV("storage_test.loom"),
                                   &context_, &block_pool_, &options, &module));
    return module;
  }

  uint16_t FirstSpecId(loom_module_t* module) {
    const loom_block_t* body = loom_module_block(module);
    const loom_op_t* op = loom_block_const_op(body, 0);
    EXPECT_TRUE(loom_encoding_define_isa(op));
    return loom_encoding_define_spec(op);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST(EncodingStorageQueryTest, AbsentShapedAttachmentIsDense) {
  const loom_type_kind_t shaped_kinds[] = {
      LOOM_TYPE_TILE,
      LOOM_TYPE_TENSOR,
      LOOM_TYPE_VIEW,
  };
  for (loom_type_kind_t shaped_kind : shaped_kinds) {
    const loom_type_t type = loom_type_shaped_1d(
        shaped_kind, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
    loom_value_fact_address_layout_t layout = {};
    EXPECT_TRUE(loom_encoding_query_type_address_layout(
        /*context=*/nullptr, /*module=*/nullptr, type,
        /*stride_storage=*/nullptr, /*stride_capacity=*/0, &layout));
    EXPECT_EQ(layout.kind, LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE);

    loom_value_fact_storage_schema_t storage_schema = {};
    EXPECT_FALSE(loom_encoding_query_type_storage_schema(
        /*context=*/nullptr, /*module=*/nullptr, type, &storage_schema));
  }

  const loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
  loom_value_fact_address_layout_t vector_layout = {};
  EXPECT_FALSE(loom_encoding_query_type_address_layout(
      /*context=*/nullptr, /*module=*/nullptr, vector_type,
      /*stride_storage=*/nullptr, /*stride_capacity=*/0, &vector_layout));
}

TEST_F(EncodingStorageTest, FixedRecordGeometry) {
  loom_module_t* module =
      Parse(IREE_SV("%schema = encoding.define #test.fixed_record : "
                    "encoding<schema>\n"));
  ASSERT_NE(module, nullptr);

  loom_encoding_record_geometry_t geometry;
  ASSERT_TRUE(loom_encoding_query_static_record_geometry(
      module, FirstSpecId(module), &geometry));
  EXPECT_EQ(geometry.logical_element_count, 32u);
  EXPECT_EQ(geometry.storage_byte_count, 18u);
  EXPECT_EQ(geometry.required_alignment, 2u);

  loom_module_free(module);
}

TEST_F(EncodingStorageTest, FixedGgmlSchemasExposeCanonicalContracts) {
  struct SchemaExpectation {
    // Complete source for one exact provider schema.
    iree_string_view_t source;

    // Generated descriptor carrying the fixed provider metadata.
    const loom_encoding_family_descriptor_t* descriptor;

    // Exact serialized record geometry.
    loom_encoding_record_geometry_t record;

    // Target-independent logical operand semantics.
    loom_encoding_operand_summary_t operand;

    // Explicit values needed after extracting the packed payload.
    loom_encoding_auxiliary_key_flags_t required_auxiliary_keys;
  };

  const SchemaExpectation expectations[] = {
      {
          /*.source=*/IREE_SV(
              "%schema = encoding.define #ggml.q4_0 : encoding<schema>\n"),
          /*.descriptor=*/&loom_encoding_ggml_q4_0_family_descriptor,
          /*.record=*/{32, 18, 2},
          /*.operand=*/
          {
              /*.element_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I4,
              /*.scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_F16,
              /*.secondary_scale_format=*/{},
              /*.payload_packing=*/
              LOOM_VALUE_FACT_PAYLOAD_PACKING_LITTLE_ENDIAN_NIBBLES,
              /*.scale_topology=*/LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D,
              /*.affine_policy=*/LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY,
              /*.rounding_policy=*/{},
              /*.codebook_policy=*/{},
              /*.sparsity_policy=*/{},
              /*.flags=*/{},
              /*.sparsity_group=*/{},
              /*.payload_register_count=*/{},
              /*.payload_element_count=*/32,
              /*.scale_group=*/{32, {32}},
              /*.scale_operand_count=*/1,
          },
          /*.required_auxiliary_keys=*/
          1ull << LOOM_ENCODING_AUXILIARY_KEY_SCALE,
      },
      {
          /*.source=*/IREE_SV(
              "%schema = encoding.define #ggml.q8_0 : encoding<schema>\n"),
          /*.descriptor=*/&loom_encoding_ggml_q8_0_family_descriptor,
          /*.record=*/{32, 34, 2},
          /*.operand=*/
          {
              /*.element_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8,
              /*.scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_F16,
              /*.secondary_scale_format=*/{},
              /*.payload_packing=*/LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES,
              /*.scale_topology=*/LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D,
              /*.affine_policy=*/LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY,
              /*.rounding_policy=*/{},
              /*.codebook_policy=*/{},
              /*.sparsity_policy=*/{},
              /*.flags=*/{},
              /*.sparsity_group=*/{},
              /*.payload_register_count=*/{},
              /*.payload_element_count=*/32,
              /*.scale_group=*/{32, {32}},
              /*.scale_operand_count=*/1,
          },
          /*.required_auxiliary_keys=*/
          1ull << LOOM_ENCODING_AUXILIARY_KEY_SCALE,
      },
      {
          /*.source=*/IREE_SV(
              "%schema = encoding.define #ggml.q4_k : encoding<schema>\n"),
          /*.descriptor=*/&loom_encoding_ggml_q4_k_family_descriptor,
          /*.record=*/{256, 144, 2},
          /*.operand=*/
          {
              /*.element_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_U4,
              /*.scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_F16,
              /*.secondary_scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_U6,
              /*.payload_packing=*/LOOM_VALUE_FACT_PAYLOAD_PACKING_MULTI_STREAM,
              /*.scale_topology=*/LOOM_VALUE_FACT_SCALE_TOPOLOGY_HIERARCHICAL,
              /*.affine_policy=*/LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_MIN,
              /*.rounding_policy=*/{},
              /*.codebook_policy=*/{},
              /*.sparsity_policy=*/{},
              /*.flags=*/{},
              /*.sparsity_group=*/{},
              /*.payload_register_count=*/{},
              /*.payload_element_count=*/256,
              /*.scale_group=*/{32, {32}},
              /*.scale_operand_count=*/2,
          },
          /*.required_auxiliary_keys=*/
          (1ull << LOOM_ENCODING_AUXILIARY_KEY_SCALE) |
              (1ull << LOOM_ENCODING_AUXILIARY_KEY_SECONDARY_SCALE) |
              (1ull << LOOM_ENCODING_AUXILIARY_KEY_MINIMUM),
      },
      {
          /*.source=*/IREE_SV(
              "%schema = encoding.define #ggml.q5_k : encoding<schema>\n"),
          /*.descriptor=*/&loom_encoding_ggml_q5_k_family_descriptor,
          /*.record=*/{256, 176, 2},
          /*.operand=*/
          {
              /*.element_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_U5,
              /*.scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_F16,
              /*.secondary_scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_U6,
              /*.payload_packing=*/LOOM_VALUE_FACT_PAYLOAD_PACKING_MULTI_STREAM,
              /*.scale_topology=*/LOOM_VALUE_FACT_SCALE_TOPOLOGY_HIERARCHICAL,
              /*.affine_policy=*/LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_MIN,
              /*.rounding_policy=*/{},
              /*.codebook_policy=*/{},
              /*.sparsity_policy=*/{},
              /*.flags=*/{},
              /*.sparsity_group=*/{},
              /*.payload_register_count=*/{},
              /*.payload_element_count=*/256,
              /*.scale_group=*/{32, {32}},
              /*.scale_operand_count=*/2,
          },
          /*.required_auxiliary_keys=*/
          (1ull << LOOM_ENCODING_AUXILIARY_KEY_SCALE) |
              (1ull << LOOM_ENCODING_AUXILIARY_KEY_SECONDARY_SCALE) |
              (1ull << LOOM_ENCODING_AUXILIARY_KEY_MINIMUM),
      },
      {
          /*.source=*/IREE_SV(
              "%schema = encoding.define #ggml.q6_k : encoding<schema>\n"),
          /*.descriptor=*/&loom_encoding_ggml_q6_k_family_descriptor,
          /*.record=*/{256, 210, 2},
          /*.operand=*/
          {
              /*.element_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I6,
              /*.scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_F16,
              /*.secondary_scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_I8,
              /*.payload_packing=*/LOOM_VALUE_FACT_PAYLOAD_PACKING_MULTI_STREAM,
              /*.scale_topology=*/LOOM_VALUE_FACT_SCALE_TOPOLOGY_HIERARCHICAL,
              /*.affine_policy=*/
              LOOM_VALUE_FACT_AFFINE_POLICY_SUPER_SCALE_TIMES_SUBSCALE,
              /*.rounding_policy=*/{},
              /*.codebook_policy=*/{},
              /*.sparsity_policy=*/{},
              /*.flags=*/{},
              /*.sparsity_group=*/{},
              /*.payload_register_count=*/{},
              /*.payload_element_count=*/256,
              /*.scale_group=*/{16, {16}},
              /*.scale_operand_count=*/2,
          },
          /*.required_auxiliary_keys=*/
          (1ull << LOOM_ENCODING_AUXILIARY_KEY_SCALE) |
              (1ull << LOOM_ENCODING_AUXILIARY_KEY_SECONDARY_SCALE),
      },
      {
          /*.source=*/IREE_SV(
              "%schema = encoding.define #ggml.q8_1_x4 : encoding<schema>\n"),
          /*.descriptor=*/&loom_encoding_ggml_q8_1_x4_family_descriptor,
          /*.record=*/{128, 144, 16},
          /*.operand=*/
          {
              /*.element_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8,
              /*.scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_F16,
              /*.secondary_scale_format=*/{},
              /*.payload_packing=*/
              LOOM_VALUE_FACT_PAYLOAD_PACKING_SEPARATE_SCALE_PAYLOAD,
              /*.scale_topology=*/LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D,
              /*.affine_policy=*/LOOM_VALUE_FACT_AFFINE_POLICY_SUM_CORRECTION,
              /*.rounding_policy=*/{},
              /*.codebook_policy=*/{},
              /*.sparsity_policy=*/{},
              /*.flags=*/{},
              /*.sparsity_group=*/{},
              /*.payload_register_count=*/{},
              /*.payload_element_count=*/128,
              /*.scale_group=*/{32, {32}},
              /*.scale_operand_count=*/1,
          },
          /*.required_auxiliary_keys=*/
          (1ull << LOOM_ENCODING_AUXILIARY_KEY_SCALE) |
              (1ull << LOOM_ENCODING_AUXILIARY_KEY_SUM_CORRECTION),
      },
  };

  for (const SchemaExpectation& expectation : expectations) {
    loom_module_t* module = Parse(expectation.source);
    ASSERT_NE(module, nullptr);
    const uint16_t encoding_id = FirstSpecId(module);

    const loom_encoding_record_layout_t* layout = nullptr;
    ASSERT_TRUE(
        loom_encoding_query_static_record_layout(module, encoding_id, &layout));
    ASSERT_NE(layout, nullptr);
    ExpectGgmlRecordTopology(expectation.descriptor, layout);

    loom_encoding_record_geometry_t record;
    ASSERT_TRUE(loom_encoding_query_static_record_geometry(module, encoding_id,
                                                           &record));
    EXPECT_EQ(record.logical_element_count,
              expectation.record.logical_element_count);
    EXPECT_EQ(record.storage_byte_count, expectation.record.storage_byte_count);
    EXPECT_EQ(record.required_alignment, expectation.record.required_alignment);

    loom_value_fact_storage_schema_t schema;
    ASSERT_TRUE(loom_encoding_query_static_storage_schema(module, encoding_id,
                                                          &schema));
    EXPECT_TRUE(loom_value_fact_encoded_operand_schema_equal(
        schema.encoded_operand, expectation.operand));

    loom_encoding_auxiliary_key_flags_t required_auxiliary_keys = 0;
    ASSERT_TRUE(loom_encoding_auxiliary_required_keys_from_schema(
        schema.encoded_operand, &required_auxiliary_keys, nullptr));
    EXPECT_EQ(required_auxiliary_keys, expectation.required_auxiliary_keys);
    EXPECT_EQ(loom_encoding_record_embedded_auxiliary_keys(layout),
              expectation.required_auxiliary_keys);
    ASSERT_NE(expectation.descriptor->fixed_metadata, nullptr);
    EXPECT_EQ(expectation.descriptor->fixed_metadata->required_auxiliary_keys,
              expectation.required_auxiliary_keys);

    loom_module_free(module);
  }
}

TEST_F(EncodingStorageTest, ShapedTypeResolvesFixedRecordLayout) {
  loom_module_t* module =
      Parse(IREE_SV("%schema = encoding.define #ggml.q5_k : "
                    "encoding<schema>\n"));
  ASSERT_NE(module, nullptr);
  const uint16_t encoding_id = FirstSpecId(module);
  const loom_type_t view_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8,
                          loom_dim_pack_static(256), encoding_id);

  const loom_encoding_record_layout_t* layout = nullptr;
  ASSERT_TRUE(loom_encoding_query_type_record_layout(
      /*context=*/nullptr, module, view_type, &layout));
  ASSERT_NE(layout, nullptr);
  EXPECT_EQ(layout->geometry.logical_element_count, 256u);
  EXPECT_EQ(layout->geometry.storage_byte_count, 176u);

  loom_module_free(module);
}

TEST_F(EncodingStorageTest, OperandSummaryHasNoFixedGeometry) {
  loom_module_t* module =
      Parse(IREE_SV("%schema = encoding.define "
                    "#encoding.operand<element_format=f8e4m3fn, "
                    "payload_elements=1, payload_packing=dense_lanes, "
                    "rounding=finite_only> : encoding<schema>\n"));
  ASSERT_NE(module, nullptr);
  const uint16_t encoding_id = FirstSpecId(module);

  loom_encoding_record_geometry_t geometry = {1, 1, 1};
  EXPECT_FALSE(loom_encoding_query_static_record_geometry(module, encoding_id,
                                                          &geometry));
  EXPECT_EQ(geometry.logical_element_count, 0u);
  EXPECT_EQ(geometry.storage_byte_count, 0u);
  EXPECT_EQ(geometry.required_alignment, 0u);

  loom_value_fact_storage_schema_t schema;
  ASSERT_TRUE(
      loom_encoding_query_static_storage_schema(module, encoding_id, &schema));
  EXPECT_EQ(schema.encoded_operand.element_format,
            LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN);
  EXPECT_EQ(schema.encoded_operand.payload_packing,
            LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES);
  EXPECT_EQ(schema.encoded_operand.payload_element_count, 1u);
  EXPECT_EQ(schema.encoded_operand.rounding_policy,
            LOOM_VALUE_FACT_ROUNDING_POLICY_FINITE_ONLY);

  loom_module_free(module);
}

TEST_F(EncodingStorageTest, ParameterizedOperandSummary) {
  loom_module_t* module = Parse(IREE_SV(
      "%schema = encoding.define #encoding.operand<affine=scale_only, "
      "codebook=static_builtin_table, element_format=f4e2m1, "
      "payload_elements=128, payload_packing=multi_stream, "
      "payload_registers=16, rounding=finite_only, scale_format=f8e4m3, "
      "scale_group_shape=[8, 16], scale_operands=1, "
      "scale_topology=block_2d, secondary_scale_format=f32, "
      "sparsity=n_m_structured, sparsity_group_elements=4, "
      "sparsity_group_nonzero_elements=2, zero_scale_fallback=true> : "
      "encoding<schema>\n"));
  ASSERT_NE(module, nullptr);

  loom_value_fact_storage_schema_t schema;
  ASSERT_TRUE(loom_encoding_query_static_storage_schema(
      module, FirstSpecId(module), &schema));
  const loom_value_fact_encoded_operand_schema_t& operand =
      schema.encoded_operand;
  EXPECT_EQ(operand.element_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1);
  EXPECT_EQ(operand.payload_packing,
            LOOM_VALUE_FACT_PAYLOAD_PACKING_MULTI_STREAM);
  EXPECT_EQ(operand.payload_element_count, 128u);
  EXPECT_EQ(operand.payload_register_count, 16u);
  EXPECT_EQ(operand.rounding_policy,
            LOOM_VALUE_FACT_ROUNDING_POLICY_FINITE_ONLY);
  EXPECT_EQ(operand.scale_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3);
  EXPECT_EQ(operand.secondary_scale_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_F32);
  EXPECT_EQ(operand.scale_topology, LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_2D);
  EXPECT_EQ(operand.scale_group.element_count, 128u);
  EXPECT_EQ(operand.scale_group.shape[0], 8u);
  EXPECT_EQ(operand.scale_group.shape[1], 16u);
  EXPECT_EQ(operand.scale_operand_count, 1u);
  EXPECT_EQ(operand.affine_policy, LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY);
  EXPECT_EQ(operand.codebook_policy,
            LOOM_VALUE_FACT_CODEBOOK_POLICY_STATIC_BUILTIN_TABLE);
  EXPECT_EQ(operand.sparsity_policy,
            LOOM_VALUE_FACT_SPARSITY_POLICY_N_M_STRUCTURED);
  EXPECT_EQ(operand.sparsity_group.element_count, 4u);
  EXPECT_EQ(operand.sparsity_group.nonzero_element_count, 2u);
  EXPECT_TRUE(iree_all_bits_set(
      operand.flags, LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK));

  loom_module_free(module);
}

}  // namespace
}  // namespace loom
