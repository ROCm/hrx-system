// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/input_projection.h"

#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/reader/type_validator.h"
#include "loom/format/bytecode/varint.h"
#include "loom/ops/type_registry.h"

namespace loom {
namespace {

static const loom_attr_descriptor_t kParameters[] = {{
    .name = LOOM_BSTRING_REF(7, "element"),
    .attr_kind = LOOM_ATTR_TYPE,
}};
extern const loom_parameterized_type_descriptor_t kParameterizedType;
static const loom_type_descriptor_t kType = {
    .name = LOOM_BSTRING_REF(10, "wire.array"),
    .ir_kind = LOOM_TYPE_PARAMETERIZED,
    .param_count = IREE_ARRAYSIZE(kParameters),
    .parameterized = &kParameterizedType,
};
const loom_parameterized_type_descriptor_t kParameterizedType = {
    .type_descriptor = &kType,
    .parameter_descriptors = kParameters,
    .ir_kind = LOOM_TYPE_PARAMETERIZED,
    .parameter_count = IREE_ARRAYSIZE(kParameters),
};
static const loom_type_registry_entry_t kRegistry[] = {
    {IREE_SV("wire.array"), &kType},
};

static void Append(std::vector<uint8_t>& bytes, uint64_t value) {
  uint8_t encoded[LOOM_VARINT_MAX_LENGTH];
  iree_host_size_t length = 0;
  IREE_CHECK_OK(loom_uvarint_encode(
      value, iree_make_byte_span(encoded, sizeof(encoded)), &length));
  bytes.insert(bytes.end(), encoded, encoded + length);
}

class BytecodeInputProjectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_type_registry_register_types(
        &context_, kRegistry, IREE_ARRAYSIZE(kRegistry)));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_bytecode_reader_decoder_initialize(
        {
            .fn = [](void*,
                     const loom_diagnostic_t*) { return iree_ok_status(); },
        },
        IREE_SV("inputs.loombc"), &errors_, &decoder_);
    view_.strings = {strings_, IREE_ARRAYSIZE(strings_)};
    view_.encodings.count = 1;
    loom_bytecode_input_projection_initialize(&decoder_, &context_, &view_,
                                              nullptr, &arena_, &projection_);
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_bytecode_reader_consume_status(&decoder_));
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  bool Read(const std::vector<uint8_t>& records, uint64_t count, uint64_t root,
            uint32_t* out_shape) {
    std::vector<uint8_t> payload;
    Append(payload, count);
    payload.insert(payload.end(), records.begin(), records.end());
    Append(payload, root);
    std::vector<uint8_t> bytes;
    Append(bytes, payload.size());
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    loom_bytecode_reader_cursor_t cursor;
    loom_bytecode_reader_cursor_initialize(bytes.data(), bytes.size(), 0,
                                           IREE_SV("SYMBOLS"), &cursor);
    const loom_bytecode_attribute_ssa_validation_scope_t scope = {
        .symbol_name = IREE_SV("family"),
        .value_count = 3,
    };
    return loom_bytecode_input_projection_read(&projection_, &cursor, 1, &scope,
                                               out_shape);
  }

  // Registry shared by static and scoped parameter decoders.
  loom_context_t context_;
  // Physical block owner for the admission arena.
  iree_arena_block_pool_t pool_;
  // Admission scratch lifetime; projection APIs allocate no native module.
  iree_arena_allocator_t arena_;
  // Structured decoder and diagnostic count for each fixture.
  loom_bytecode_reader_decoder_t decoder_;
  // Number of diagnosed source errors.
  uint32_t errors_ = 0;
  // Module-local descriptor and parameter names.
  iree_string_view_t strings_[2] = {IREE_SV("wire.array"), IREE_SV("element")};
  // Tables already established before signature admission.
  loom_bytecode_reader_module_view_t view_ = {};
  // Projection shared by every signature in the module.
  loom_bytecode_input_projection_t projection_;
};

TEST_F(BytecodeInputProjectionTest, PreservesEveryCarrierRankAndAttachment) {
  const std::pair<uint8_t, loom_type_kind_t> kinds[] = {
      {LOOM_BYTECODE_TYPE_GROUP, LOOM_TYPE_GROUP},
      {LOOM_BYTECODE_TYPE_TENSOR, LOOM_TYPE_TENSOR},
      {LOOM_BYTECODE_TYPE_TILE, LOOM_TYPE_TILE},
      {LOOM_BYTECODE_TYPE_VECTOR, LOOM_TYPE_VECTOR},
      {LOOM_BYTECODE_TYPE_VIEW, LOOM_TYPE_VIEW},
  };
  for (const auto& [wire_kind, kind] : kinds) {
    const bool has_encoding =
        kind != LOOM_TYPE_GROUP && kind != LOOM_TYPE_VECTOR;
    const uint8_t minimum_rank = has_encoding ? 0 : 1;
    for (uint8_t rank = minimum_rank; rank <= LOOM_TYPE_MAX_RANK; ++rank) {
      for (uint8_t attachment = 0; attachment < (has_encoding ? 3 : 1);
           ++attachment) {
        SCOPED_TRACE(::testing::Message()
                     << "kind=" << kind << " rank=" << unsigned(rank)
                     << " attachment=" << unsigned(attachment));
        loom_bytecode_input_projection_begin_signature(&projection_);
        std::vector<uint8_t> records = {wire_kind};
        if (kind != LOOM_TYPE_GROUP) {
          Append(records, LOOM_SCALAR_TYPE_F32);
        }
        Append(records, rank);
        if (kind != LOOM_TYPE_GROUP) {
          Append(records, attachment);
          Append(records, attachment);
          if (kind == LOOM_TYPE_VIEW) {
            Append(records, 2);
          }
        }
        uint16_t dynamic_axes = 0;
        for (uint8_t axis = 0; axis < rank; ++axis) {
          const bool dynamic = (axis % 2) != 0;
          dynamic_axes |= uint16_t(dynamic) << axis;
          Append(records, dynamic);
          Append(records, dynamic ? 2 : axis + 7);
        }
        uint32_t id = 0;
        ASSERT_TRUE(Read(records, 1, 1, &id));
        ASSERT_NE(id, 0u);
        const auto* shape =
            loom_bytecode_input_projection_shape(&projection_, id);
        EXPECT_EQ(shape->kind, kind);
        EXPECT_EQ(shape->rank, rank);
        EXPECT_EQ(shape->element_type, kind == LOOM_TYPE_GROUP
                                           ? LOOM_SCALAR_TYPE_NONE
                                           : LOOM_SCALAR_TYPE_F32);
        EXPECT_EQ(shape->attachment, attachment);
        EXPECT_EQ(shape->encoding_ordinal, attachment);
        EXPECT_EQ(shape->dynamic_axes, dynamic_axes);
        EXPECT_EQ(shape->access_alignment, kind == LOOM_TYPE_VIEW ? 2 : 0);
        for (uint8_t axis = 0; axis < rank; ++axis) {
          EXPECT_EQ(shape->dimensions[axis], axis % 2 ? 2u : axis + 7u);
        }
        const auto used = arena_.used_allocation_size;
        loom_bytecode_input_projection_begin_signature(&projection_);
        uint32_t repeated = 0;
        ASSERT_TRUE(Read(records, 1, 1, &repeated));
        EXPECT_EQ(repeated, id);
        EXPECT_EQ(arena_.used_allocation_size, used);
      }
    }
  }
  EXPECT_EQ(errors_, 0u);
}

TEST_F(BytecodeInputProjectionTest, CompositeNodesDoNotHideEarlierShapeRoots) {
  const std::vector<uint8_t> records = {
      LOOM_BYTECODE_TYPE_VECTOR,
      LOOM_SCALAR_TYPE_F32,
      1,
      0,
      0,
      1,
      2,
      LOOM_BYTECODE_TYPE_FUNCTION,
      1,
      0,
      1,
      LOOM_BYTECODE_TYPE_PARAMETERIZED,
      0,
      1,
      1,
      LOOM_BYTECODE_ATTR_TYPE,
      3,
      LOOM_BYTECODE_TYPE_DIALECT,
      0,
      1,
      5,
  };
  uint32_t shape_id = 0;
  ASSERT_TRUE(Read(records, 4, 1, &shape_id));
  ASSERT_NE(shape_id, 0u);
  EXPECT_EQ(projection_.binding_count, 4u);
  EXPECT_EQ(loom_bytecode_input_projection_shape(&projection_, shape_id)
                ->dimensions[0],
            2u);
  for (uint64_t root : {2, 3, 4}) {
    uint32_t composite = 123;
    ASSERT_TRUE(Read({}, 0, root, &composite));
    EXPECT_EQ(composite, 0u);
  }
  uint32_t earlier = 0;
  ASSERT_TRUE(Read({}, 0, 1, &earlier));
  EXPECT_EQ(earlier, shape_id);
  EXPECT_EQ(errors_, 0u);
}

TEST_F(BytecodeInputProjectionTest, RejectsIncompleteCompositeReferences) {
  const std::vector<std::vector<uint8_t>> records = {
      {LOOM_BYTECODE_TYPE_FUNCTION, 1, 0, 1},
      {LOOM_BYTECODE_TYPE_FUNCTION, 1, 0, 0},
      {LOOM_BYTECODE_TYPE_DIALECT, 0, 1, 1},
      {LOOM_BYTECODE_TYPE_PARAMETERIZED, 0, 1, 1, LOOM_BYTECODE_ATTR_TYPE, 1},
      {LOOM_BYTECODE_TYPE_PARAMETERIZED, 0, 1, 1, LOOM_BYTECODE_ATTR_TYPE, 0},
  };
  for (const auto& record : records) {
    loom_bytecode_input_projection_begin_signature(&projection_);
    errors_ = 0;
    uint32_t id = 0;
    EXPECT_FALSE(Read(record, 1, 1, &id));
    EXPECT_EQ(errors_, 1u);
    EXPECT_EQ(projection_.binding_count, 0u);
  }
}

TEST_F(BytecodeInputProjectionTest, ReusesStaticRootAndOwnsLargeShape) {
  const std::vector<uint8_t> bytes = {
      1, LOOM_BYTECODE_TYPE_GROUP, 3, 0, 7, 0, 11, 0, 13,
  };
  const loom_bytecode_reader_section_t section = {
      .absolute_offset = 64,
      .bytes = iree_make_const_byte_span(bytes.data(), bytes.size()),
  };
  view_.sections.types = &section;
  loom_bytecode_type_validation_t validation;
  ASSERT_TRUE(loom_bytecode_type_validation_begin(
      &decoder_, &context_, &view_, section.bytes, section.absolute_offset,
      LOOM_BYTECODE_TYPE_RETAIN_RANGES, &arena_, &validation));
  ASSERT_TRUE(loom_bytecode_type_validation_finish(&validation));
  projection_.type_entries = validation.entries;
  const uint8_t extension = 0;
  const loom_bytecode_attribute_ssa_validation_scope_t scope = {};
  uint32_t first_id = 0;
  iree_host_size_t used = 0;
  for (unsigned i = 0; i < 256; ++i) {
    loom_bytecode_input_projection_begin_signature(&projection_);
    loom_bytecode_reader_cursor_t cursor;
    loom_bytecode_reader_cursor_initialize(&extension, 1, 0, IREE_SV("SYMBOLS"),
                                           &cursor);
    uint32_t id = 0;
    ASSERT_TRUE(loom_bytecode_input_projection_read(&projection_, &cursor, 0,
                                                    &scope, &id));
    if (i == 0) {
      first_id = id;
      used = arena_.used_allocation_size;
    }
    EXPECT_EQ(id, first_id);
    EXPECT_EQ(arena_.used_allocation_size, used);
  }
  const auto* shape =
      loom_bytecode_input_projection_shape(&projection_, first_id);
  EXPECT_EQ(shape->kind, LOOM_TYPE_GROUP);
  EXPECT_EQ(shape->rank, 3);
  EXPECT_EQ(shape->dynamic_axes, 0);
  EXPECT_EQ(shape->dimensions[0], 7u);
  EXPECT_EQ(shape->dimensions[1], 11u);
  EXPECT_EQ(shape->dimensions[2], 13u);
  EXPECT_EQ(projection_.static_types.buckets.count, 1u);
  EXPECT_EQ(projection_.shape_index.count, 1u);
}

TEST_F(BytecodeInputProjectionTest, NaturalAlignmentHasOneSourceIdentity) {
  uint32_t natural = 0, explicit_natural = 0;
  ASSERT_TRUE(
      Read({LOOM_BYTECODE_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, 1, 0, 0, 0, 0, 16},
           1, 1, &natural));
  loom_bytecode_input_projection_begin_signature(&projection_);
  ASSERT_TRUE(
      Read({LOOM_BYTECODE_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, 1, 0, 0, 4, 0, 16},
           1, 1, &explicit_natural));
  EXPECT_EQ(natural, explicit_natural);
}

TEST_F(BytecodeInputProjectionTest,
       KeepsCompleteEncodingSubjectsAndIdentities) {
  view_.encodings.count = 2;
  loom_bytecode_symbol_metadata_t symbols[4] = {};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(symbols); ++i) {
    const bool is_family = i == 3;
    loom_bytecode_input_projection_begin_signature(&projection_);
    const uint8_t attachment = is_family ? LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA
                               : i == 0
                                   ? LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE
                                   : LOOM_BYTECODE_ENCODING_ATTACHMENT_STATIC;
    std::vector<uint8_t> records = {
        LOOM_BYTECODE_TYPE_VIEW,
        LOOM_SCALAR_TYPE_I8,
        1,
        attachment,
        uint8_t(is_family ? 0 : i),
        0,
        1,
        2,
    };
    uint32_t shape = 0;
    ASSERT_TRUE(Read(records, 1, 1, &shape));
    IREE_ASSERT_OK(
        loom_bytecode_input_projection_add_argument(&projection_, 2, shape));
    IREE_ASSERT_OK(loom_bytecode_input_projection_end_signature(
        &projection_, i, &symbols[i].input_contract_ordinal_plus_one));
    symbols[i].kind = is_family ? LOOM_BYTECODE_SYMBOL_TEMPLATE_DECL
                                : LOOM_BYTECODE_SYMBOL_TEMPLATE_DEF;
    symbols[i].template_family_symbol_ordinal = 3;
  }
  iree_arena_allocator_t retained;
  iree_arena_initialize(&pool_, &retained);
  loom_bytecode_module_metadata_t metadata = {
      .symbol_count = IREE_ARRAYSIZE(symbols),
      .symbols = symbols,
  };
  ASSERT_TRUE(loom_bytecode_input_projection_finish_module(
      &projection_, &retained, &metadata));
  ASSERT_EQ(metadata.input_contract_count, 4u);
  for (uint32_t i = 0; i < 3; ++i) {
    const auto* contract =
        metadata
            .input_contracts[symbols[i].input_contract_ordinal_plus_one - 1];
    ASSERT_EQ(contract->count, 1u);
    EXPECT_EQ(contract->values[0].argument, 2);
    EXPECT_EQ(contract->values[0].axis, UINT8_MAX);
    EXPECT_FALSE(contract->values[0].expected_is_argument);
    EXPECT_EQ(contract->values[0].expected, i);
  }
  const auto* family =
      metadata.input_contracts[symbols[3].input_contract_ordinal_plus_one - 1];
  ASSERT_EQ(family->count, 2u);
  EXPECT_EQ(family->values[0].argument, 2);
  EXPECT_EQ(family->values[0].axis, 0);
  EXPECT_TRUE(family->values[0].expected_is_argument);
  EXPECT_EQ(family->values[0].expected, 1u);
  EXPECT_EQ(family->values[1].argument, 2);
  EXPECT_EQ(family->values[1].axis, UINT8_MAX);
  EXPECT_TRUE(family->values[1].expected_is_argument);
  EXPECT_EQ(family->values[1].expected, 0u);
  iree_arena_deinitialize(&retained);
}

TEST_F(BytecodeInputProjectionTest,
       RetainsOnlyChangedInputsAcrossForwardFamilies) {
  constexpr uint32_t kProviders = 4096;
  // Ordinary symbols are not entered into the template projection. Their
  // source ordinals may precede, follow, or separate templates arbitrarily.
  constexpr uint32_t kFirstProvider = 16384;
  constexpr uint32_t kFamily = kFirstProvider + kProviders;
  std::vector<loom_bytecode_symbol_metadata_t> symbols(kFamily + 1);
  for (uint32_t i = 0; i <= kProviders; ++i) {
    const uint32_t ordinal = kFirstProvider + i;
    const bool is_family = i == kProviders;
    const bool exact = !is_family && i % 2 == 0;
    loom_bytecode_input_projection_begin_signature(&projection_);
    std::vector<uint8_t> records = {
        LOOM_BYTECODE_TYPE_VECTOR,   LOOM_SCALAR_TYPE_F32, 1, 0, 0,
        uint8_t(is_family || exact),
    };
    Append(records, is_family || exact ? 1 : 16 + (i / 2) % 32);
    uint32_t shape_id = 0;
    ASSERT_TRUE(Read(records, 1, 1, &shape_id));
    IREE_ASSERT_OK(
        loom_bytecode_input_projection_add_argument(&projection_, 1, shape_id));
    IREE_ASSERT_OK(loom_bytecode_input_projection_end_signature(
        &projection_, ordinal,
        &symbols[ordinal].input_contract_ordinal_plus_one));
    symbols[ordinal].kind = is_family ? LOOM_BYTECODE_SYMBOL_TEMPLATE_DECL
                                      : LOOM_BYTECODE_SYMBOL_TEMPLATE_DEF;
    symbols[ordinal].template_family_symbol_ordinal = kFamily;
  }
  EXPECT_EQ(projection_.signature_index.count, 33u);
  iree_arena_allocator_t retained;
  iree_arena_initialize(&pool_, &retained);
  loom_bytecode_module_metadata_t metadata = {
      .symbol_count = symbols.size(),
      .symbols = symbols.data(),
  };
  ASSERT_TRUE(loom_bytecode_input_projection_finish_module(
      &projection_, &retained, &metadata));
  ASSERT_EQ(metadata.input_contract_count, 33u);
  // Retained contracts have no dependency on the decoder's temporary rows.
  iree_arena_reset(&arena_);
  const auto* family =
      metadata
          .input_contracts[symbols[kFamily].input_contract_ordinal_plus_one -
                           1];
  ASSERT_EQ(family->count, 1u);
  EXPECT_EQ(family->values[0].argument, 1);
  EXPECT_EQ(family->values[0].axis, 0);
  EXPECT_TRUE(family->values[0].expected_is_argument);
  EXPECT_EQ(family->values[0].expected, 0u);
  for (uint32_t i = 0; i < kProviders; ++i) {
    const uint32_t id =
        symbols[kFirstProvider + i].input_contract_ordinal_plus_one;
    if (i % 2 == 0) {
      EXPECT_EQ(id, 0u);
    } else {
      ASSERT_NE(id, 0u);
      const auto* provider = metadata.input_contracts[id - 1];
      ASSERT_EQ(provider->count, 1u);
      EXPECT_EQ(provider->values[0].argument, 1);
      EXPECT_EQ(provider->values[0].axis, 0);
      EXPECT_FALSE(provider->values[0].expected_is_argument);
      EXPECT_EQ(provider->values[0].expected, 16 + (i / 2) % 32);
      if (i >= 64) {
        EXPECT_EQ(
            id,
            symbols[kFirstProvider + i - 64].input_contract_ordinal_plus_one);
      }
    }
  }
  iree_arena_deinitialize(&retained);
}

}  // namespace
}  // namespace loom
