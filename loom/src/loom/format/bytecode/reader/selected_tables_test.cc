// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/selected_tables.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/format.h"

namespace loom {
namespace {

static const loom_attr_descriptor_t kEncodingParameters[] = {
    {
        .name = LOOM_BSTRING_REF(5, "block"),
        .attr_kind = LOOM_ATTR_I64,
        .flags = LOOM_ATTR_OPTIONAL,
    },
    {
        .name = LOOM_BSTRING_REF(4, "base"),
        .attr_kind = LOOM_ATTR_ENCODING,
        .flags = LOOM_ATTR_OPTIONAL,
    },
};
static const loom_encoding_family_descriptor_t kEncodingDescriptor = {
    .name = LOOM_BSTRING_REF(4, "q8_0"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
    .family_flags = {},
    .parameter_count = IREE_ARRAYSIZE(kEncodingParameters),
    .parameter_descriptors = kEncodingParameters,
};
static const loom_encoding_vtable_t kEncodingVtable = {
    .descriptor = &kEncodingDescriptor,
};

static iree_status_t AcceptDiagnostic(void* user_data,
                                      const loom_diagnostic_t* diagnostic) {
  (void)user_data;
  (void)diagnostic;
  return iree_ok_status();
}

struct ExternalSymbolResolver {
  // Indexed source symbol accepted by this resolver.
  uint32_t expected_source_ordinal;
  // Predeclared output symbol returned for the source symbol.
  loom_symbol_ref_t target_ref;
  // Number of resolver invocations, excluding memoized resolutions.
  iree_host_size_t invocation_count;
};

static iree_status_t ResolveExternalSymbol(
    void* user_data, uint32_t source_symbol_ordinal,
    loom_symbol_ref_t* out_target_symbol_ref) {
  auto* resolver = static_cast<ExternalSymbolResolver*>(user_data);
  if (source_symbol_ordinal != resolver->expected_source_ordinal) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected source symbol ordinal");
  }
  ++resolver->invocation_count;
  *out_target_symbol_ref = resolver->target_ref;
  return iree_ok_status();
}

static void AppendUVarint(uint64_t value, std::vector<uint8_t>* bytes) {
  do {
    uint8_t byte = value & 0x7Fu;
    value >>= 7;
    if (value != 0) {
      byte |= 0x80u;
    }
    bytes->push_back(byte);
  } while (value != 0);
}

class BytecodeSelectedTablesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kEncodingVtable));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, iree_string_view_empty(),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{AcceptDiagnostic, nullptr},
        IREE_SV("selected_tables_test.loombc"), &error_count_, &decoder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void InitializeMaterializer(
      const std::vector<uint8_t>& bytecode,
      const loom_bytecode_module_metadata_t* metadata,
      loom_bytecode_selected_table_materializer_t* out_materializer) {
    loom_bytecode_selected_table_materializer_initialize(
        &decoder_, iree_make_const_byte_span(bytecode.data(), bytecode.size()),
        &context_, metadata, &scratch_arena_, module_,
        loom_bytecode_selected_symbol_resolver_empty(), out_materializer);
  }

  // Malformed-bytecode diagnostics emitted by the table decoder.
  uint32_t error_count_ = 0;
  // Backing blocks shared by output IR and materialization scratch.
  iree_arena_block_pool_t block_pool_;
  // Resettable per-frame decoding storage.
  iree_arena_allocator_t scratch_arena_;
  // Registered dialect and encoding descriptors used by the fixture.
  loom_context_t context_;
  // Output module receiving projected table entries.
  loom_module_t* module_ = nullptr;
  // Bytecode decoder reporting into error_count_.
  loom_bytecode_reader_decoder_t decoder_ = {};
};

TEST_F(BytecodeSelectedTablesTest, MaterializesOnlyReachedMixedTableFacts) {
  iree_string_view_t strings[] = {
      iree_string_view_empty(),
      IREE_SV("q8_0"),
      IREE_SV("block"),
      IREE_SV("base"),
  };
  iree_string_view_t sources[] = {IREE_SV("model.loom")};
  std::vector<uint8_t> bytecode;

  loom_bytecode_encoding_metadata_t encodings[2] = {};
  encodings[0].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(),
                  {
                      0x00,  // Family ordinal.
                      0x00,  // No alias.
                      0x01,  // Parameter count.
                      0x02,  // "block".
                      LOOM_BYTECODE_ATTR_I64,
                      0x0E,  // Signed value 7.
                  });
  encodings[0].entry_length = bytecode.size() - encodings[0].entry_offset;
  encodings[0].name_string_index = 1;
  encodings[1].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(),
                  {
                      0x00,  // Family ordinal.
                      0x00,  // No alias.
                      0x01,  // Parameter count.
                      0x03,  // "base".
                      LOOM_BYTECODE_ATTR_ENCODING,
                      0x01,  // Prior encoding ID.
                  });
  encodings[1].entry_length = bytecode.size() - encodings[1].entry_offset;
  encodings[1].name_string_index = 1;

  loom_bytecode_table_entry_metadata_t types[2] = {};
  types[0].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(),
                  {LOOM_BYTECODE_TYPE_SCALAR, LOOM_SCALAR_TYPE_F32});
  types[0].entry_length = bytecode.size() - types[0].entry_offset;
  types[1].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(), {
                                      LOOM_BYTECODE_TYPE_TENSOR,
                                      LOOM_SCALAR_TYPE_F32,
                                      0x01,  // Rank.
                                      LOOM_BYTECODE_ENCODING_ATTACHMENT_STATIC,
                                      0x02,  // Encoding ID.
                                      0x00,  // Static dimension.
                                      0x04,  // Dimension size.
                                  });
  types[1].entry_length = bytecode.size() - types[1].entry_offset;

  loom_bytecode_table_entry_metadata_t locations[2] = {};
  locations[0].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(), {LOOM_LOCATION_NONE, 0x00});
  locations[0].entry_length = bytecode.size() - locations[0].entry_offset;
  locations[1].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(), {
                                      LOOM_LOCATION_FILE,
                                      0x00,  // Flags.
                                      0x00,  // Source ordinal.
                                      0x01,
                                      0x02,
                                      0x03,
                                      0x04,  // Coordinates.
                                  });
  locations[1].entry_length = bytecode.size() - locations[1].entry_offset;

  loom_bytecode_module_metadata_t metadata = {
      .strings = {IREE_ARRAYSIZE(strings), strings},
      .sources = {IREE_ARRAYSIZE(sources), sources},
      .types = {IREE_ARRAYSIZE(types), types},
      .encodings = {IREE_ARRAYSIZE(encodings), encodings},
      .locations = {IREE_ARRAYSIZE(locations), locations}};
  loom_bytecode_selected_table_materializer_t materializer;
  InitializeMaterializer(bytecode, &metadata, &materializer);

  loom_type_id_t target_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_type(
      &materializer, /*source_type_id=*/1, &target_type_id));
  EXPECT_EQ(target_type_id, 1u);
  ASSERT_EQ(module_->types.count, 2u);
  EXPECT_TRUE(loom_type_equal(loom_type_table_get(&module_->types, 0),
                              loom_type_scalar(LOOM_SCALAR_TYPE_F32)));
  EXPECT_EQ(
      loom_type_kind(loom_type_table_get(&module_->types, target_type_id)),
      LOOM_TYPE_TENSOR);
  EXPECT_EQ(loom_type_table_get(&module_->types, target_type_id).encoding_id,
            2u);
  ASSERT_EQ(module_->encodings.count, 2u);
  ASSERT_EQ(module_->encodings.entries[0].attribute_count, 1u);
  EXPECT_EQ(module_->encodings.entries[0].attributes[0].value.i64, 7);
  ASSERT_EQ(module_->encodings.entries[1].attribute_count, 1u);
  EXPECT_EQ(module_->encodings.entries[1].attributes[0].value.encoding_id, 1u);

  loom_location_id_t target_location_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_location(
      &materializer, /*source_location_id=*/1, &target_location_id));
  EXPECT_EQ(target_location_id, 1u);
  ASSERT_EQ(module_->locations.count, 2u);
  EXPECT_EQ(
      loom_location_table_const_entry(&module_->locations, 1)->file.source_id,
      0u);
  ASSERT_EQ(module_->sources.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(module_->sources.entries[0], sources[0]));

  EXPECT_EQ(materializer.projection.buckets.count, 5u);
  // The scalar dependency is already retained even though its source table
  // entry has not been projected yet.
  loom_type_id_t target_element_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_type(
      &materializer, /*source_type_id=*/0, &target_element_id));
  EXPECT_EQ(target_element_id, 0u);
  EXPECT_EQ(module_->types.count, 2u);
  EXPECT_EQ(materializer.projection.buckets.count, 6u);
  EXPECT_EQ(error_count_, 0u);
  loom_bytecode_selected_table_materializer_deinitialize(&materializer);
}

TEST_F(BytecodeSelectedTablesTest, ReusesInheritedSourceWhenComposingLocation) {
  iree_string_view_t sources[] = {IREE_SV("model.loom")};
  std::vector<uint8_t> bytecode;
  loom_bytecode_table_entry_metadata_t locations[2] = {};
  locations[0].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(), {LOOM_LOCATION_NONE, 0x00});
  locations[0].entry_length = bytecode.size() - locations[0].entry_offset;
  locations[1].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(), {
                                      LOOM_LOCATION_FILE,
                                      0x00,  // Flags.
                                      0x00,  // Source ordinal.
                                      0x01,
                                      0x02,
                                      0x03,
                                      0x04,  // Coordinates.
                                  });
  locations[1].entry_length = bytecode.size() - locations[1].entry_offset;
  loom_bytecode_module_metadata_t metadata = {
      .sources = {IREE_ARRAYSIZE(sources), sources},
      .locations = {IREE_ARRAYSIZE(locations), locations}};

  loom_source_id_t inherited_source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_append_source(module_, sources[0], &inherited_source_id));
  ASSERT_EQ(inherited_source_id, 0u);
  loom_bytecode_selected_table_materializer_t materializer;
  InitializeMaterializer(bytecode, &metadata, &materializer);

  loom_location_id_t target_location_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_location(
      &materializer, /*source_location_id=*/1, &target_location_id));
  EXPECT_EQ(target_location_id, 1u);
  ASSERT_EQ(module_->locations.count, 2u);
  EXPECT_EQ(
      loom_location_table_const_entry(&module_->locations, 1)->file.source_id,
      inherited_source_id);
  ASSERT_EQ(module_->sources.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(module_->sources.entries[0], sources[0]));

  loom_bytecode_selected_table_materializer_deinitialize(&materializer);
}

TEST_F(BytecodeSelectedTablesTest, InterleavedReadersShareCurrentSourceNames) {
  iree_string_view_t sources[] = {IREE_SV("first.loom"), IREE_SV("shared.loom"),
                                  IREE_SV("last.loom")};
  std::vector<uint8_t> bytecode;
  loom_bytecode_table_entry_metadata_t locations[4] = {};
  bytecode.insert(bytecode.end(), {LOOM_LOCATION_NONE, 0x00});
  locations[0].entry_length = bytecode.size();
  for (uint8_t i = 0; i < IREE_ARRAYSIZE(sources); ++i) {
    locations[i + 1].entry_offset = bytecode.size();
    bytecode.insert(bytecode.end(), {LOOM_LOCATION_FILE, 0x00, i, 1, 2, 3, 4});
    locations[i + 1].entry_length =
        bytecode.size() - locations[i + 1].entry_offset;
  }
  loom_bytecode_module_metadata_t metadata = {
      .sources = {IREE_ARRAYSIZE(sources), sources},
      .locations = {IREE_ARRAYSIZE(locations), locations}};
  iree_string_view_t other_sources[] = {sources[1], sources[2], sources[0]};
  loom_bytecode_module_metadata_t other_metadata = metadata;
  other_metadata.sources = {IREE_ARRAYSIZE(other_sources), other_sources};
  loom_bytecode_selected_table_materializer_t readers[2];
  InitializeMaterializer(bytecode, &metadata, &readers[0]);
  InitializeMaterializer(bytecode, &other_metadata, &readers[1]);

  // Both readers start with an empty output. Each then reaches names added
  // after initialization by the other reader, as retained contract readers do.
  const uint32_t reader_ordinals[] = {0, 1, 0, 1, 0, 1};
  const loom_location_id_t source_locations[] = {1, 1, 2, 3, 3, 2};
  const loom_source_id_t expected_sources[] = {0, 1, 1, 0, 2, 2};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(source_locations); ++i) {
    loom_location_id_t location_id = LOOM_LOCATION_UNKNOWN;
    IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_location(
        &readers[reader_ordinals[i]], source_locations[i], &location_id));
    const auto* location =
        loom_location_table_const_entry(&module_->locations, location_id);
    EXPECT_EQ(location->file.source_id, expected_sources[i]);
    EXPECT_EQ(location->file.start_line, 1u);
    EXPECT_EQ(location->file.start_col, 2u);
    EXPECT_EQ(location->file.end_line, 3u);
    EXPECT_EQ(location->file.end_col, 4u);
  }
  EXPECT_EQ(module_->sources.count, IREE_ARRAYSIZE(sources));
  loom_bytecode_selected_table_materializer_deinitialize(&readers[1]);
  loom_bytecode_selected_table_materializer_deinitialize(&readers[0]);
}

TEST_F(BytecodeSelectedTablesTest, StandaloneSourcesStayUnindexedUntilShared) {
  constexpr uint32_t kSourceCount = 128;
  std::vector<std::string> names;
  names.reserve(kSourceCount);
  std::vector<iree_string_view_t> sources;
  std::vector<loom_bytecode_table_entry_metadata_t> locations(1);
  std::vector<uint8_t> bytecode = {LOOM_LOCATION_NONE, 0x00};
  locations[0].entry_length = bytecode.size();
  for (uint32_t i = 0; i < kSourceCount; ++i) {
    names.push_back("source_" + std::to_string(i));
    sources.push_back(
        iree_make_string_view(names.back().data(), names.back().size()));
    const auto offset = bytecode.size();
    bytecode.insert(bytecode.end(), {LOOM_LOCATION_FILE, 0x00});
    AppendUVarint(i, &bytecode);
    bytecode.insert(bytecode.end(), {1, 2, 3, 4});
    locations.push_back({offset, bytecode.size() - offset});
  }
  loom_bytecode_module_metadata_t metadata = {
      .sources = {sources.size(), sources.data()},
      .locations = {locations.size(), locations.data()}};
  for (uint32_t reader = 0; reader < 3; ++reader) {
    loom_bytecode_selected_table_materializer_t materializer;
    InitializeMaterializer(bytecode, &metadata, &materializer);
    const auto* previous_index = module_->sources.name_index;
    for (uint32_t i = 0; i < kSourceCount; ++i) {
      loom_location_id_t location_id = LOOM_LOCATION_UNKNOWN;
      IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_location(
          &materializer, i + 1, &location_id));
      EXPECT_EQ(
          loom_location_table_const_entry(&module_->locations, location_id)
              ->file.source_id,
          i);
    }
    EXPECT_EQ(module_->sources.count, kSourceCount);
    if (reader == 0) {
      EXPECT_EQ(module_->sources.name_index, nullptr);
    } else {
      ASSERT_NE(module_->sources.name_index, nullptr);
      EXPECT_EQ(module_->sources.name_index->count, kSourceCount);
      if (reader == 2) {
        EXPECT_EQ(module_->sources.name_index, previous_index);
      }
    }
    loom_bytecode_selected_table_materializer_deinitialize(&materializer);
  }
}

TEST_F(BytecodeSelectedTablesTest, ResolvesExternalSymbolsByDenseSourceIndex) {
  iree_string_view_t strings[] = {
      IREE_SV("unused"),
      IREE_SV("external"),
  };
  loom_bytecode_symbol_metadata_t symbols[1] = {};
  symbols[0].name = strings[1];
  symbols[0].name_string_index = 1;
  uint32_t symbol_ordinal_by_string_index[] = {UINT32_MAX, 0};
  loom_bytecode_module_metadata_t metadata = {
      .strings = {IREE_ARRAYSIZE(strings), strings},
      .symbol_count = IREE_ARRAYSIZE(symbols),
      .symbols = symbols,
      .symbol_ordinal_by_string_index = symbol_ordinal_by_string_index};

  loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("projected"),
                                           &target_name_id));
  loom_symbol_id_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module_, target_name_id, &target_symbol_id));
  ExternalSymbolResolver resolver = {
      .expected_source_ordinal = 0,
      .target_ref = {.module_id = 0, .symbol_id = target_symbol_id},
      .invocation_count = 0,
  };
  loom_bytecode_selected_table_materializer_t materializer;
  loom_bytecode_selected_table_materializer_initialize(
      &decoder_, iree_const_byte_span_empty(), &context_, &metadata,
      &scratch_arena_, module_,
      loom_bytecode_selected_symbol_resolver_make(ResolveExternalSymbol,
                                                  &resolver),
      &materializer);

  loom_symbol_ref_t resolved = loom_symbol_ref_null();
  bool found = false;
  IREE_ASSERT_OK(loom_bytecode_selected_table_resolve_symbol(
      &materializer, /*source_name_ordinal=*/1, &resolved, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ(resolved.symbol_id, target_symbol_id);
  EXPECT_EQ(resolver.invocation_count, 1u);

  resolved = loom_symbol_ref_null();
  found = false;
  IREE_ASSERT_OK(loom_bytecode_selected_table_resolve_symbol(
      &materializer, /*source_name_ordinal=*/1, &resolved, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ(resolved.symbol_id, target_symbol_id);
  EXPECT_EQ(resolver.invocation_count, 1u);

  resolved = loom_symbol_ref_null();
  found = true;
  IREE_ASSERT_OK(loom_bytecode_selected_table_resolve_symbol(
      &materializer, /*source_name_ordinal=*/0, &resolved, &found));
  EXPECT_FALSE(found);
  EXPECT_FALSE(loom_symbol_ref_is_valid(resolved));

  loom_bytecode_selected_table_materializer_deinitialize(&materializer);
}

TEST_F(BytecodeSelectedTablesTest, ProjectsMixedStructuralPayloads) {
  loom_type_id_t unused_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_F64), &unused_type_id));
  loom_type_id_t element_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_F32), &element_type_id));
  ASSERT_NE(element_type_id, 0u);
  loom_string_id_t unused_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("target_only"),
                                           &unused_name_id));
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("test.box"), &name_id));
  ASSERT_NE(name_id, 1u);

  const uint64_t carrier_payload0 = UINT64_C(0x123456789ABCDEF0);
  const uint64_t carrier_payload1 = (UINT64_C(9) << 16) | UINT64_C(0x1234);
  // Generic construction copies nested payloads independently of their
  // canonical table entries. Projected candidates must remain complete so the
  // interner can compare equal children whose storage pointers differ.
  const loom_register_type_data_t register_data = {
      carrier_payload0, carrier_payload1,
      loom_type_scalar(LOOM_SCALAR_TYPE_F32)};
  const loom_type_t register_type =
      loom_type_register_payload_with_value_type(&register_data);
  const loom_type_t parameters[] = {register_type, register_data.value_type};
  const loom_type_t argument_type =
      loom_type_dialect(name_id, IREE_ARRAYSIZE(parameters), parameters);
  loom_type_t expected_type = {};
  IREE_ASSERT_OK(loom_module_intern_function_type(
      module_, &argument_type, 1, &register_type, 1, &expected_type));
  loom_type_id_t expected_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_type_id(module_, expected_type, &expected_type_id));
  ASSERT_EQ(module_->types.count, 5u);
  iree_string_view_t strings[] = {iree_string_view_empty(),
                                  IREE_SV("test.box")};
  loom_bytecode_table_entry_metadata_t entries[4] = {};
  std::vector<uint8_t> bytecode = {LOOM_BYTECODE_TYPE_SCALAR,
                                   LOOM_SCALAR_TYPE_F32};
  entries[0].entry_length = bytecode.size();
  entries[1].entry_offset = bytecode.size();
  bytecode.push_back(LOOM_BYTECODE_TYPE_REGISTER);
  AppendUVarint(carrier_payload0, &bytecode);
  AppendUVarint(carrier_payload1, &bytecode);
  bytecode.insert(bytecode.end(), {/*has_value_type=*/1, /*value_type_id=*/0});
  entries[1].entry_length = bytecode.size() - entries[1].entry_offset;
  entries[2].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(),
                  {LOOM_BYTECODE_TYPE_DIALECT,
                   /*name_id=*/1, /*parameter_count=*/2,
                   /*register_type_id=*/1, /*element_type_id=*/0});
  entries[2].entry_length = bytecode.size() - entries[2].entry_offset;
  entries[3].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(),
                  {LOOM_BYTECODE_TYPE_FUNCTION,
                   /*argument_count=*/1, /*result_count=*/1,
                   /*dialect_type_id=*/2, /*register_type_id=*/1});
  entries[3].entry_length = bytecode.size() - entries[3].entry_offset;
  loom_bytecode_module_metadata_t metadata = {
      .strings = {IREE_ARRAYSIZE(strings), strings},
      .types = {IREE_ARRAYSIZE(entries), entries}};
  loom_bytecode_selected_table_materializer_t materializer;
  InitializeMaterializer(bytecode, &metadata, &materializer);
  loom_type_id_t target_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_type(
      &materializer, /*source_type_id=*/3, &target_type_id));
  EXPECT_EQ(target_type_id, expected_type_id);
  loom_type_id_t repeated_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_type(
      &materializer, /*source_type_id=*/3, &repeated_type_id));
  EXPECT_EQ(repeated_type_id, target_type_id);
  loom_bytecode_selected_table_materializer_deinitialize(&materializer);
  iree_arena_reset(&scratch_arena_);

  ASSERT_EQ(module_->types.count, 5u);
  const loom_func_type_data_t* function =
      loom_type_func_data(loom_type_table_get(&module_->types, target_type_id));
  ASSERT_NE(function, nullptr);
  ASSERT_EQ(function->arg_count, 1u);
  ASSERT_EQ(function->result_count, 1u);
  const loom_type_t dialect_type = function->types[0];
  ASSERT_EQ(loom_type_kind(dialect_type), LOOM_TYPE_DIALECT);
  EXPECT_EQ(loom_type_dialect_name_id(dialect_type), name_id);
  ASSERT_EQ(loom_type_dialect_param_count(dialect_type), 2u);
  const loom_type_t* children = loom_type_dialect_params(dialect_type);
  EXPECT_TRUE(loom_type_equal(
      children[1], loom_type_table_get(&module_->types, element_type_id)));
  EXPECT_TRUE(loom_type_equal(children[0], function->types[1]));
  const loom_register_type_data_t* data = loom_type_register_data(children[0]);
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(data->carrier_payload0, carrier_payload0);
  EXPECT_EQ(data->carrier_payload1, carrier_payload1);
  EXPECT_TRUE(loom_type_equal(data->value_type, children[1]));
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeSelectedTablesTest, MaterializesDeepTypeChainIteratively) {
  constexpr uint32_t kTypeCount = 4096;
  std::vector<uint8_t> bytecode;
  std::vector<loom_bytecode_table_entry_metadata_t> entries(kTypeCount);
  entries[0].entry_offset = bytecode.size();
  bytecode.push_back(LOOM_BYTECODE_TYPE_NONE);
  entries[0].entry_length = bytecode.size() - entries[0].entry_offset;
  for (uint32_t i = 1; i < kTypeCount; ++i) {
    entries[i].entry_offset = bytecode.size();
    bytecode.push_back(LOOM_BYTECODE_TYPE_FUNCTION);
    AppendUVarint(/*argument_count=*/1, &bytecode);
    AppendUVarint(/*result_count=*/0, &bytecode);
    AppendUVarint(/*prior_type_id=*/i - 1, &bytecode);
    entries[i].entry_length = bytecode.size() - entries[i].entry_offset;
  }
  loom_bytecode_module_metadata_t metadata = {
      .types = {entries.size(), entries.data()}};
  loom_bytecode_selected_table_materializer_t materializer;
  InitializeMaterializer(bytecode, &metadata, &materializer);

  // A projection reader can retain header payloads in this same arena while
  // independently materializing another root. Only root-local scratch ends.
  uint32_t* header_payload = nullptr;
  IREE_ASSERT_OK(
      iree_arena_allocate_array(&scratch_arena_, 64, sizeof(*header_payload),
                                reinterpret_cast<void**>(&header_payload)));
  for (uint32_t i = 0; i < 64; ++i) {
    header_payload[i] = i + 7;
  }
  const auto header_checkpoint = iree_arena_checkpoint_save(&scratch_arena_);
  loom_type_id_t target_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_type(
      &materializer, kTypeCount - 1, &target_type_id));
  EXPECT_EQ(target_type_id, kTypeCount - 1);
  EXPECT_EQ(module_->types.count, kTypeCount);
  EXPECT_EQ(materializer.projection.buckets.count, kTypeCount);
  EXPECT_EQ(materializer.worklist.count, 0u);
  EXPECT_EQ(scratch_arena_.used_allocation_size,
            header_checkpoint.used_allocation_size);
  EXPECT_EQ(scratch_arena_.total_allocation_size,
            header_checkpoint.total_allocation_size);
  EXPECT_EQ(scratch_arena_.block_head, header_checkpoint.block_head);
  for (uint32_t i = 0; i < 64; ++i) {
    EXPECT_EQ(header_payload[i], i + 7);
  }
  EXPECT_EQ(error_count_, 0u);
  loom_bytecode_selected_table_materializer_deinitialize(&materializer);
  iree_arena_reset(&scratch_arena_);
  bytecode.clear();
  bytecode.shrink_to_fit();

  loom_type_t type = loom_type_table_get(&module_->types, target_type_id);
  for (uint32_t i = 1; i < kTypeCount; ++i) {
    const auto* function = loom_type_func_data(type);
    ASSERT_NE(function, nullptr);
    ASSERT_EQ(function->arg_count, 1u);
    ASSERT_EQ(function->result_count, 0u);
    type = function->types[0];
  }
  EXPECT_EQ(loom_type_kind(type), LOOM_TYPE_NONE);
}

TEST_F(BytecodeSelectedTablesTest, ReusesWorklistAcrossWideTypeRoots) {
  constexpr uint32_t kArgumentCount = 4096;
  std::vector<uint8_t> bytecode;
  loom_bytecode_table_entry_metadata_t entries[4] = {};
  entries[0].entry_offset = bytecode.size();
  bytecode.push_back(LOOM_BYTECODE_TYPE_NONE);
  entries[0].entry_length = bytecode.size() - entries[0].entry_offset;
  entries[1].entry_offset = bytecode.size();
  bytecode.push_back(LOOM_BYTECODE_TYPE_FUNCTION);
  AppendUVarint(kArgumentCount, &bytecode);
  AppendUVarint(/*result_count=*/0, &bytecode);
  for (uint32_t i = 0; i < kArgumentCount; ++i) {
    AppendUVarint(/*type_id=*/0, &bytecode);
  }
  entries[1].entry_length = bytecode.size() - entries[1].entry_offset;
  entries[2].entry_offset = bytecode.size();
  bytecode.push_back(LOOM_BYTECODE_TYPE_SCALAR);
  bytecode.push_back(LOOM_SCALAR_TYPE_F32);
  entries[2].entry_length = bytecode.size() - entries[2].entry_offset;
  entries[3].entry_offset = bytecode.size();
  bytecode.push_back(LOOM_BYTECODE_TYPE_FUNCTION);
  AppendUVarint(kArgumentCount - 1, &bytecode);
  AppendUVarint(/*result_count=*/0, &bytecode);
  for (uint32_t i = 0; i < kArgumentCount - 1; ++i) {
    AppendUVarint(/*type_id=*/2, &bytecode);
  }
  entries[3].entry_length = bytecode.size() - entries[3].entry_offset;
  loom_bytecode_module_metadata_t metadata = {
      .types = {IREE_ARRAYSIZE(entries), entries}};
  loom_bytecode_selected_table_materializer_t materializer;
  InitializeMaterializer(bytecode, &metadata, &materializer);

  loom_type_id_t target_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_type(
      &materializer, /*source_type_id=*/1, &target_type_id));
  EXPECT_EQ(target_type_id, 1u);
  ASSERT_EQ(module_->types.count, 2u);
  EXPECT_EQ(loom_type_func_arg_count(loom_type_table_get(&module_->types, 1)),
            kArgumentCount);
  EXPECT_EQ(materializer.projection.buckets.count, 2u);
  EXPECT_EQ(materializer.worklist.count, 0u);
  EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
  // The frontier is wide but the projection contains only four identities.
  // Stack storage stays pooled and serves the next independent root unchanged.
  EXPECT_EQ(materializer.retained_arena.allocation_head, nullptr);
  const auto retained_bytes = materializer.retained_arena.used_allocation_size;
  IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_type(
      &materializer, /*source_type_id=*/3, &target_type_id));
  EXPECT_EQ(target_type_id, 3u);
  EXPECT_EQ(module_->types.count, 4u);
  EXPECT_EQ(materializer.projection.buckets.count, 4u);
  EXPECT_EQ(materializer.worklist.count, 0u);
  EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
  EXPECT_EQ(materializer.retained_arena.allocation_head, nullptr);
  EXPECT_EQ(materializer.retained_arena.used_allocation_size, retained_bytes);
  EXPECT_EQ(error_count_, 0u);
  loom_bytecode_selected_table_materializer_deinitialize(&materializer);
  iree_arena_reset(&scratch_arena_);

  const auto* function =
      loom_type_func_data(loom_type_table_get(&module_->types, 1));
  ASSERT_NE(function, nullptr);
  for (uint32_t i = 0; i < kArgumentCount; ++i) {
    EXPECT_EQ(loom_type_kind(function->types[i]), LOOM_TYPE_NONE);
  }
  function = loom_type_func_data(loom_type_table_get(&module_->types, 3));
  ASSERT_NE(function, nullptr);
  ASSERT_EQ(function->arg_count, kArgumentCount - 1);
  for (uint32_t i = 0; i < kArgumentCount - 1; ++i) {
    EXPECT_EQ(loom_type_element_type(function->types[i]), LOOM_SCALAR_TYPE_F32);
  }
}

TEST_F(BytecodeSelectedTablesTest, MaterializesDeepLocationChainIteratively) {
  constexpr uint32_t kLocationCount = 4096;
  std::vector<uint8_t> bytecode;
  std::vector<loom_bytecode_table_entry_metadata_t> entries(kLocationCount);
  entries[0].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(), {LOOM_LOCATION_NONE, 0x00});
  entries[0].entry_length = bytecode.size() - entries[0].entry_offset;
  for (uint32_t i = 1; i < kLocationCount; ++i) {
    entries[i].entry_offset = bytecode.size();
    bytecode.insert(bytecode.end(), {
                                        LOOM_LOCATION_TAGGED,
                                        0x00,  // Flags.
                                        0x01,  // Tag.
                                    });
    AppendUVarint(/*prior_location_id=*/i - 1, &bytecode);
    AppendUVarint(/*data_length=*/0, &bytecode);
    entries[i].entry_length = bytecode.size() - entries[i].entry_offset;
  }
  loom_bytecode_module_metadata_t metadata = {
      .locations = {entries.size(), entries.data()}};
  loom_bytecode_selected_table_materializer_t materializer;
  InitializeMaterializer(bytecode, &metadata, &materializer);

  loom_location_id_t target_location_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_location(
      &materializer, kLocationCount - 1, &target_location_id));
  EXPECT_EQ(target_location_id, kLocationCount - 1);
  EXPECT_EQ(module_->locations.count, kLocationCount);
  EXPECT_EQ(materializer.projection.buckets.count, kLocationCount - 1);
  EXPECT_EQ(materializer.worklist.count, 0u);
  EXPECT_EQ(error_count_, 0u);
  loom_bytecode_selected_table_materializer_deinitialize(&materializer);
}

}  // namespace
}  // namespace loom
