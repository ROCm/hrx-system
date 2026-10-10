// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/source.h"

#include <string>
#include <utility>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"

namespace {

static iree_host_size_t ReferenceAsciiByteOffset(iree_string_view_t source,
                                                 uint32_t line,
                                                 uint32_t column) {
  if (line == 0) {
    return 0;
  }
  uint32_t current_line = 1;
  iree_host_size_t offset = 0;
  while (current_line < line && offset < source.size) {
    current_line += source.data[offset++] == '\n';
  }
  if (current_line < line) {
    return source.size;
  }
  uint32_t current_column = 1;
  while (current_column < column && offset < source.size &&
         source.data[offset] != '\n') {
    ++current_column;
    ++offset;
  }
  return offset;
}

TEST(SourceTest, HighlightOffsetsClampWhileCountingCodePoints) {
  const auto source = IREE_SV("\tαb\nlast\n");
  EXPECT_EQ(loom_source_byte_offset(source, 1, 1), 0u);
  EXPECT_EQ(loom_source_byte_offset(source, 1, 2), 1u);
  EXPECT_EQ(loom_source_byte_offset(source, 1, 3), 3u);
  EXPECT_EQ(loom_source_byte_offset(source, 1, 4), 4u);
  EXPECT_EQ(loom_source_byte_offset(source, 1, 99), 4u);
  EXPECT_EQ(loom_source_byte_offset(source, 2, 0), 5u);
  EXPECT_EQ(loom_source_byte_offset(source, 3, 1), source.size);
  EXPECT_EQ(loom_source_byte_offset(source, 99, 1), source.size);
  EXPECT_EQ(loom_source_byte_offset(source, 0, 99), 0u);
}

TEST(SourceTest, BlockScanningMatchesBytewiseOracle) {
  std::string regular_source;
  for (int line = 0; line < 600; ++line) {
    regular_source.append(1 + (line * 17) % 97, 'a' + line % 26);
    regular_source.push_back('\n');
  }
  const std::string dense_source(2048, '\n');
  std::string long_line_source(4097, 'x');
  long_line_source.append("\ny\n");

  const std::string* sources[] = {&regular_source, &dense_source,
                                  &long_line_source};
  for (const std::string* source : sources) {
    const auto source_view =
        iree_make_string_view(source->data(), source->size());
    uint32_t line_count = 1;
    for (char value : *source) {
      line_count += value == '\n';
    }
    for (uint32_t line = 0; line <= line_count + 2; ++line) {
      for (uint32_t column : {0u, 1u, 2u, 17u, 257u, 4098u}) {
        EXPECT_EQ(loom_source_byte_offset(source_view, line, column),
                  ReferenceAsciiByteOffset(source_view, line, column))
            << "source size " << source->size() << ", line " << line
            << ", column " << column;
      }
    }
  }
}

TEST(SourceTest, BlockScanningAndRangeAnchorsPreserveCoordinates) {
  std::string source(300, 'a');
  source.push_back('\n');
  source.append(300, 'b');
  source.append("\nlast\n");
  const auto source_view = iree_make_string_view(source.data(), source.size());

  EXPECT_EQ(loom_source_byte_offset(source_view, 2, 1), 301u);
  EXPECT_EQ(loom_source_byte_offset(source_view, 2, 101), 401u);
  EXPECT_EQ(loom_source_byte_offset(source_view, 3, 2), 603u);
  EXPECT_EQ(loom_source_byte_offset(source_view, 4, 1), source.size());

  const loom_source_range_t range = {
      .provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
      .filename = iree_string_view_empty(),
      .source = source_view,
      .start = 401,
      .end = source.size(),
      .start_line = 2,
      .start_column = 101,
      .end_line = 4,
      .end_column = 1,
  };
  for (const auto& position : {std::pair<uint32_t, uint32_t>{1, 250},
                               {2, 1},
                               {2, 101},
                               {2, 250},
                               {3, 2},
                               {4, 1},
                               {99, 1}}) {
    EXPECT_EQ(
        loom_source_range_byte_offset(&range, position.first, position.second),
        loom_source_byte_offset(source_view, position.first, position.second));
  }

  std::string dense_source(1024, '\n');
  EXPECT_EQ(loom_source_byte_offset(
                iree_make_string_view(dense_source.data(), dense_source.size()),
                513, 1),
            512u);
}

class SourceResolverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool);
    loom_context_initialize(iree_allocator_system(), &context);
    IREE_ASSERT_OK(loom_context_finalize(&context));
    IREE_ASSERT_OK(loom_module_allocate(&context, IREE_SV("source"), &pool,
                                        nullptr, iree_allocator_system(),
                                        &module));
  }
  void TearDown() override {
    loom_module_free(module);
    loom_context_deinitialize(&context);
    iree_arena_block_pool_deinitialize(&pool);
  }
  loom_location_id_t FileLocation(iree_string_view_t filename,
                                  uint16_t start_line, uint16_t start_column,
                                  uint16_t end_line, uint16_t end_column) {
    loom_source_id_t source_id;
    IREE_CHECK_OK(loom_module_register_source(module, filename, &source_id));
    loom_location_id_t location;
    IREE_CHECK_OK(loom_module_add_location(
        module,
        loom_location_file_range(source_id, start_line, start_column, end_line,
                                 end_column),
        &location));
    return location;
  }
  // Owns module allocations for each test.
  iree_arena_block_pool_t pool;
  // Minimal context: location resolution requires no registered dialects.
  loom_context_t context;
  // Owner of the source and location namespaces under test.
  loom_module_t* module = nullptr;
};

TEST_F(SourceResolverTest,
       ExactResolutionRejectsUnavailableAndReversedCoordinates) {
  loom_source_id_t source_id;
  IREE_ASSERT_OK(
      loom_module_register_source(module, IREE_SV("file"), &source_id));
  const loom_source_entry_t source = {source_id, IREE_SV("αb\nlast\n"),
                                      IREE_SV("file")};
  loom_source_table_resolver_t table = {module, &source, 1};
  const loom_source_resolver_t resolver = {loom_source_table_resolve, &table};
  auto resolve = [&](uint16_t first_line, uint16_t first_column,
                     uint16_t last_line, uint16_t last_column,
                     loom_source_range_t* out_range) {
    loom_location_id_t location;
    IREE_CHECK_OK(loom_module_add_location(
        module,
        loom_location_file_range(source_id, first_line, first_column, last_line,
                                 last_column),
        &location));
    return loom_source_table_resolve(&table, module, location, out_range);
  };
  loom_source_range_t range;
  ASSERT_TRUE(resolve(1, 2, 2, 1, &range));
  EXPECT_EQ(range.start, 2u);
  EXPECT_EQ(range.end, 4u);
  EXPECT_EQ(range.source.data, source.source.data);
  EXPECT_EQ(range.provenance, LOOM_SOURCE_PROVENANCE_EXACT_SOURCE);
  EXPECT_TRUE(resolve(3, 1, 3, 1, &range));
  EXPECT_EQ(range.start, source.source.size);
  EXPECT_EQ(range.end, source.source.size);
  EXPECT_FALSE(resolve(0, 1, 1, 1, &range));
  EXPECT_FALSE(resolve(1, 0, 1, 1, &range));
  EXPECT_FALSE(resolve(4, 1, 4, 1, &range));
  EXPECT_FALSE(resolve(1, 4, 1, 4, &range));
  EXPECT_FALSE(resolve(1, 3, 1, 2, &range));
  EXPECT_FALSE(resolve(2, 1, 1, 1, &range));
  EXPECT_FALSE(
      loom_source_resolve(resolver, module, LOOM_LOCATION_UNKNOWN, &range));
}

TEST_F(SourceResolverTest, MissingSnapshotsRetainRecordedCoordinates) {
  auto location = FileLocation(IREE_SV("kernel.cxx"), 3, 15, 3, 34);
  loom_source_range_t range = {};
  EXPECT_FALSE(loom_source_resolve({}, module, LOOM_LOCATION_UNKNOWN, &range));
  ASSERT_TRUE(loom_source_resolve({}, module, location, &range));
  EXPECT_TRUE(iree_string_view_equal(range.filename, IREE_SV("kernel.cxx")));
  EXPECT_EQ(range.start_line, 3u);
  EXPECT_EQ(range.start_column, 15u);
  EXPECT_EQ(range.end_line, 3u);
  EXPECT_EQ(range.end_column, 34u);
  EXPECT_EQ(range.provenance, LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE);
  EXPECT_EQ(range.source.size, 0u);
  EXPECT_EQ(range.start, 0u);
  EXPECT_EQ(range.end, 0u);

  loom_source_table_resolver_t table = {module, nullptr, 0};
  const loom_source_resolver_t resolver = {loom_source_table_resolve, &table};
  ASSERT_TRUE(loom_source_resolve(resolver, module, location, &range));
  EXPECT_EQ(range.start_column, 15u);
  EXPECT_EQ(range.provenance, LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE);
  // A present snapshot with incompatible coordinates cannot supply spelling.
  const loom_source_entry_t source = {0, IREE_SV("short"),
                                      IREE_SV("kernel.cxx")};
  table.entries = &source;
  table.count = 1;
  ASSERT_TRUE(loom_source_resolve(resolver, module, location, &range));
  EXPECT_EQ(range.end_column, 34u);
  EXPECT_EQ(range.source.size, 0u);
  EXPECT_EQ(range.provenance, LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE);
}

TEST_F(SourceResolverTest, SnapshotsAreQualifiedByModuleEvenWithMatchingNames) {
  auto location = FileLocation(IREE_SV("kernel.cxx"), 1, 1, 1, 4);
  const loom_source_entry_t source = {0, IREE_SV("old"), IREE_SV("kernel.cxx")};
  loom_source_table_resolver_t table = {module, &source, 1};
  const loom_source_resolver_t resolver = {loom_source_table_resolve, &table};
  loom_source_range_t range = {};
  ASSERT_TRUE(loom_source_resolve(resolver, module, location, &range));
  EXPECT_EQ(range.provenance, LOOM_SOURCE_PROVENANCE_EXACT_SOURCE);
  EXPECT_EQ(range.source.data, source.source.data);

  loom_module_t* other = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context, IREE_SV("other"), &pool,
                                      nullptr, iree_allocator_system(),
                                      &other));
  loom_source_id_t source_id;
  IREE_ASSERT_OK(
      loom_module_register_source(other, IREE_SV("kernel.cxx"), &source_id));
  EXPECT_EQ(source_id, source.source_id);
  IREE_ASSERT_OK(loom_module_add_location(
      other, loom_location_file_range(source_id, 1, 1, 1, 4), &location));
  EXPECT_FALSE(loom_source_table_resolve(&table, other, location, &range));
  EXPECT_TRUE(loom_source_resolve(resolver, other, location, &range));
  EXPECT_TRUE(iree_string_view_equal(range.filename, IREE_SV("kernel.cxx")));
  EXPECT_EQ(range.provenance, LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE);
  EXPECT_EQ(range.source.size, 0u);
  EXPECT_EQ(range.end_column, 4u);
  loom_module_free(other);
}

TEST_F(SourceResolverTest, SparseSnapshotTablesResolveBySourceIdentity) {
  loom_source_id_t unused_source_id;
  IREE_ASSERT_OK(loom_module_register_source(module, IREE_SV("unused.h"),
                                             &unused_source_id));
  EXPECT_EQ(unused_source_id, 0u);
  loom_source_id_t source_id;
  IREE_ASSERT_OK(
      loom_module_register_source(module, IREE_SV("kernel.h"), &source_id));
  ASSERT_EQ(source_id, 1u);
  const loom_source_entry_t sources[] = {
      {.source_id = LOOM_SOURCE_ID_INVALID},
      {
          .source_id = source_id,
          .source = IREE_SV("kernel"),
          .filename = IREE_SV("kernel.h"),
      },
  };
  loom_source_table_resolver_t table = {module, sources,
                                        IREE_ARRAYSIZE(sources)};
  loom_location_id_t location;
  IREE_ASSERT_OK(loom_module_add_location(
      module, loom_location_file_range(source_id, 1, 1, 1, 7), &location));
  loom_source_range_t range = {};
  ASSERT_TRUE(loom_source_table_resolve(&table, module, location, &range));
  EXPECT_TRUE(iree_string_view_equal(range.source, IREE_SV("kernel")));
}

TEST_F(SourceResolverTest, TaggedOriginsResolveWithAndWithoutText) {
  auto location = FileLocation(IREE_SV("empty.cxx"), 1, 1, 1, 1);
  IREE_ASSERT_OK(loom_module_add_location(
      module,
      loom_location_tagged(LOOM_LOCATION_TAG_USER_BASE, location, nullptr, 0),
      &location));
  loom_source_range_t range = {};
  ASSERT_TRUE(loom_source_resolve({}, module, location, &range));
  EXPECT_TRUE(iree_string_view_equal(range.filename, IREE_SV("empty.cxx")));
  EXPECT_EQ(range.start_line, 1u);
  EXPECT_EQ(range.provenance, LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE);
  const loom_source_entry_t source = {0, IREE_SV(""), IREE_SV("empty.cxx")};
  loom_source_table_resolver_t table = {module, &source, 1};
  ASSERT_TRUE(loom_source_resolve({loom_source_table_resolve, &table}, module,
                                  location, &range));
  EXPECT_EQ(range.provenance, LOOM_SOURCE_PROVENANCE_EXACT_SOURCE);
  EXPECT_EQ(range.source.size, 0u);
}

class SourceStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_source_storage_initialize(iree_allocator_system(), &sources_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }
  void TearDown() override {
    loom_source_storage_deinitialize(&sources_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  loom_module_t* Module() {
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("input"), &pool_,
                                       nullptr, iree_allocator_system(),
                                       &module));
    return module;
  }

  // Pool backing the modules under test.
  iree_arena_block_pool_t pool_;
  // Owns snapshots independently of source modules and caller buffers.
  loom_source_storage_t sources_;
  // Minimal context for source-table and linker API calls.
  loom_context_t context_;
};

TEST_F(SourceStorageTest, CopiesEmptyAndSparseSnapshotsAndRejectsConflicts) {
  std::string filename = "header.h";
  std::string source = "const int value = 5;\n";
  IREE_ASSERT_OK(loom_source_storage_insert(
      &sources_, 3, iree_make_cstring_view(filename.c_str()),
      iree_make_cstring_view(source.c_str())));
  filename.assign("changed");
  source.assign("changed");
  EXPECT_EQ(sources_.table.count, 4u);
  EXPECT_EQ(sources_.table.entries[0].source_id, LOOM_SOURCE_ID_INVALID);
  EXPECT_TRUE(iree_string_view_equal(sources_.table.entries[3].source,
                                     IREE_SV("const int value = 5;\n")));
  IREE_ASSERT_OK(loom_source_storage_insert(&sources_, 0, IREE_SV("empty.h"),
                                            iree_string_view_empty()));
  IREE_ASSERT_OK(loom_source_storage_insert(&sources_, 0, IREE_SV("empty.h"),
                                            iree_string_view_empty()));
  EXPECT_EQ(sources_.table.entries[0].source_id, 0u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_source_storage_insert(&sources_, 0, IREE_SV("empty.h"),
                                 IREE_SV("different")));
}

TEST_F(SourceStorageTest, LinkMappingRetainsBytesAfterInputTeardown) {
  loom_source_table_resolver_t input_sources = {};
  loom_source_storage_projection_t projection = {
      &input_sources,
      &sources_,
  };
  loom_linker_options_t options = {
      .source_callback = {loom_source_storage_project, &projection},
  };
  loom_linker_t* linker = nullptr;
  IREE_ASSERT_OK(loom_linker_allocate(&context_, &options, &pool_,
                                      iree_allocator_system(), &linker));
  for (int i = 0; i < 2; ++i) {
    loom_module_t* input = Module();
    const auto filename = i == 0 ? IREE_SV("main.cc") : IREE_SV("library.h");
    loom_source_id_t source_id;
    IREE_ASSERT_OK(loom_module_register_source(input, filename, &source_id));
    EXPECT_EQ(source_id, 0u);
    std::string bytes = i == 0 ? "first\nmain\n" : "second\nlibrary\n";
    loom_source_entry_t entry = {
        source_id, iree_make_cstring_view(bytes.c_str()), filename};
    input_sources = {input, &entry, 1};
    IREE_ASSERT_OK(loom_linker_add_module(linker, input, nullptr));
    loom_module_free(input);
    bytes.assign("released");
  }
  loom_module_t* linked = nullptr;
  IREE_ASSERT_OK(loom_linker_finish(linker, &linked));
  loom_linker_free(linker);
  ASSERT_EQ(linked->sources.count, 2u);
  loom_location_id_t location;
  IREE_ASSERT_OK(loom_module_add_location(
      linked, loom_location_file_range(1, 2, 1, 2, 8), &location));
  loom_source_range_t range = {};
  ASSERT_TRUE(loom_source_resolve(loom_source_storage_resolver(&sources_),
                                  linked, location, &range));
  EXPECT_TRUE(iree_string_view_equal(range.filename, IREE_SV("library.h")));
  EXPECT_TRUE(
      iree_string_view_equal(range.source, IREE_SV("second\nlibrary\n")));
  EXPECT_EQ(range.start_line, 2u);
  EXPECT_EQ(range.start, 7u);
  loom_module_free(linked);
}

TEST_F(SourceStorageTest, LinkProjectionBorrowsBytesAndReindexesSources) {
  loom_module_t* input = Module();
  loom_source_id_t input_id;
  IREE_ASSERT_OK(
      loom_module_register_source(input, IREE_SV("input.h"), &input_id));
  IREE_ASSERT_OK(loom_source_storage_insert(
      &sources_, input_id, IREE_SV("input.h"), IREE_SV("original")));
  sources_.table.module = input;
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool_, &arena);
  loom_source_table_projection_t projection = {sources_.table, &arena};
  loom_linker_options_t options = {
      .source_callback = {loom_source_table_project, &projection},
  };
  loom_linker_t* linker = nullptr;
  IREE_ASSERT_OK(loom_linker_allocate(&context_, &options, &pool_,
                                      iree_allocator_system(), &linker));
  // Seed a different identity so the producer must move input.h from ID 0.
  loom_module_t* prefix = Module();
  loom_source_id_t prefix_id;
  IREE_ASSERT_OK(
      loom_module_register_source(prefix, IREE_SV("prefix.h"), &prefix_id));
  projection.table = {prefix, nullptr, 0};
  IREE_ASSERT_OK(loom_linker_add_module(linker, prefix, nullptr));
  EXPECT_EQ(projection.table.entries, nullptr);
  projection.table = sources_.table;
  IREE_ASSERT_OK(loom_linker_add_module(linker, input, nullptr));
  loom_module_t* linked = nullptr;
  IREE_ASSERT_OK(loom_linker_finish(linker, &linked));
  loom_linker_free(linker);
  loom_module_free(input);
  loom_module_free(prefix);

  EXPECT_EQ(projection.table.module, linked);
  ASSERT_EQ(projection.table.count, 2u);
  EXPECT_EQ(projection.table.entries[0].source_id, LOOM_SOURCE_ID_INVALID);
  EXPECT_EQ(projection.table.entries[1].source_id, 1u);
  EXPECT_EQ(projection.table.entries[1].source.data,
            sources_.table.entries[input_id].source.data);
  EXPECT_EQ(projection.table.entries[1].filename.data,
            sources_.table.entries[input_id].filename.data);
  loom_location_id_t location;
  IREE_ASSERT_OK(loom_module_add_location(
      linked, loom_location_file_range(1, 1, 1, 1, 9), &location));
  loom_source_range_t range = {};
  ASSERT_TRUE(
      loom_source_table_resolve(&projection.table, linked, location, &range));
  EXPECT_TRUE(iree_string_view_equal(range.source, IREE_SV("original")));
  loom_module_free(linked);
  iree_arena_deinitialize(&arena);
}

}  // namespace
