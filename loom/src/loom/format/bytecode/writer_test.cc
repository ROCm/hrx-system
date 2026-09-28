// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/varint.h"
#include "loom/format/bytecode/writer/body.h"
#include "loom/format/bytecode/writer/encoder.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class WriterTest : public ::testing::Test {
 protected:
  static iree_status_t AllocateContext(void* self,
                                       iree_allocator_command_t command,
                                       const void* parameters, void** pointer) {
    auto* test = static_cast<WriterTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      ++test->context_allocation_count_;
    }
    const iree_allocator_t system = iree_allocator_system();
    return system.ctl(system.self, command, parameters, pointer);
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize({this, AllocateContext}, &context_);

    // Register dialects so the writer can resolve op names.
    iree_host_size_t func_op_count = 0;
    const loom_op_vtable_t* const* func_vtables =
        loom_func_dialect_vtables(&func_op_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_FUNC, func_vtables, (uint16_t)func_op_count));

    iree_host_size_t global_op_count = 0;
    const loom_op_vtable_t* const* global_vtables =
        loom_global_dialect_vtables(&global_op_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_GLOBAL,
                                                 global_vtables,
                                                 (uint16_t)global_op_count));

    iree_host_size_t op_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&op_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 vtables, (uint16_t)op_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Creates a module with the given name.
  loom_module_t* CreateModule(const char* name) {
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context_, iree_make_cstring_view(name),
                                       &block_pool_, nullptr,
                                       iree_allocator_system(), &module));
    return module;
  }

  // Creates a module containing one test.func with a nested test.attrs dict.
  // If reverse_attr_order is true, the module's string interning and source
  // dict entry order are intentionally reversed relative to canonical key
  // spelling order.
  loom_module_t* CreateAttrsModule(bool reverse_attr_order) {
    loom_module_t* module = CreateModule("attrs");

    loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
    IREE_CHECK_OK(loom_module_intern_type(module, f32_type, &f32_type));

    loom_builder_t module_builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &module_builder);

    loom_string_id_t func_name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&module_builder, IREE_SV("f"),
                                             &func_name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, func_name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};

    loom_type_t arg_types[1] = {f32_type};
    loom_type_t result_types[1] = {f32_type};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_test_func_build(&module_builder, 0, /*visibility=*/0,
                                       /*cc=*/0, callee, arg_types, 1,
                                       result_types, 1, nullptr, 0, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &func_op));
    module->symbols.entries[symbol_id].flags = LOOM_SYMBOL_FLAG_PUBLIC;

    loom_func_like_t func_like = loom_func_like_cast(module, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* arg_ids =
        loom_func_like_arg_ids(func_like, &arg_count);
    EXPECT_EQ(arg_count, 1);
    if (arg_count == 0) {
      return module;
    }

    loom_region_t* body = loom_func_like_body(func_like);
    loom_builder_t body_builder;
    loom_builder_initialize(module, &module->arena,
                            loom_region_entry_block(body), &body_builder);

    loom_string_id_t axis_id = LOOM_STRING_ID_INVALID;
    loom_string_id_t meta_id = LOOM_STRING_ID_INVALID;
    loom_string_id_t opt_id = LOOM_STRING_ID_INVALID;
    loom_string_id_t phase_id = LOOM_STRING_ID_INVALID;
    loom_string_id_t link_id = LOOM_STRING_ID_INVALID;
    if (reverse_attr_order) {
      IREE_CHECK_OK(
          loom_module_intern_string(module, IREE_SV("meta"), &meta_id));
      IREE_CHECK_OK(
          loom_module_intern_string(module, IREE_SV("phase"), &phase_id));
      IREE_CHECK_OK(
          loom_module_intern_string(module, IREE_SV("link"), &link_id));
      IREE_CHECK_OK(loom_module_intern_string(module, IREE_SV("opt"), &opt_id));
      IREE_CHECK_OK(
          loom_module_intern_string(module, IREE_SV("axis"), &axis_id));
    } else {
      IREE_CHECK_OK(
          loom_module_intern_string(module, IREE_SV("axis"), &axis_id));
      IREE_CHECK_OK(loom_module_intern_string(module, IREE_SV("opt"), &opt_id));
      IREE_CHECK_OK(
          loom_module_intern_string(module, IREE_SV("phase"), &phase_id));
      IREE_CHECK_OK(
          loom_module_intern_string(module, IREE_SV("link"), &link_id));
      IREE_CHECK_OK(
          loom_module_intern_string(module, IREE_SV("meta"), &meta_id));
    }

    loom_named_attr_t meta_entries[2] = {
        reverse_attr_order ? loom_named_attr_t{
                                 /*.name_id=*/phase_id,
                                 /*.reserved=*/{},
                                 /*.value=*/loom_attr_string(link_id),
                             }
                           : loom_named_attr_t{
                                 /*.name_id=*/opt_id,
                                 /*.reserved=*/{},
                                 /*.value=*/loom_attr_i64(3),
                             },
        reverse_attr_order ? loom_named_attr_t{
                                 /*.name_id=*/opt_id,
                                 /*.reserved=*/{},
                                 /*.value=*/loom_attr_i64(3),
                             }
                           : loom_named_attr_t{
                                 /*.name_id=*/phase_id,
                                 /*.reserved=*/{},
                                 /*.value=*/loom_attr_string(link_id),
                             },
    };
    loom_attribute_t meta_attr = {0};
    IREE_CHECK_OK(loom_module_make_canonical_attr_dict(
        module,
        loom_make_named_attr_slice(meta_entries, IREE_ARRAYSIZE(meta_entries)),
        &meta_attr));

    loom_named_attr_t entries[2] = {
        reverse_attr_order ? loom_named_attr_t{
                                 /*.name_id=*/meta_id,
                                 /*.reserved=*/{},
                                 /*.value=*/meta_attr,
                             }
                           : loom_named_attr_t{
                                 /*.name_id=*/axis_id,
                                 /*.reserved=*/{},
                                 /*.value=*/loom_attr_i64(0),
                             },
        reverse_attr_order ? loom_named_attr_t{
                                 /*.name_id=*/axis_id,
                                 /*.reserved=*/{},
                                 /*.value=*/loom_attr_i64(0),
                             }
                           : loom_named_attr_t{
                                 /*.name_id=*/meta_id,
                                 /*.reserved=*/{},
                                 /*.value=*/meta_attr,
                             },
    };
    loom_op_t* attrs_op = nullptr;
    IREE_CHECK_OK(loom_test_attrs_build(
        &body_builder, LOOM_TEST_ATTRS_BUILD_FLAG_HAS_DICT, arg_ids[0],
        loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)), f32_type,
        LOOM_LOCATION_UNKNOWN, &attrs_op));

    const loom_value_id_t result_ids[1] = {loom_op_results(attrs_op)[0]};
    loom_op_t* yield_op = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(&body_builder, result_ids, 1,
                                        LOOM_LOCATION_UNKNOWN, &yield_op));
    return module;
  }

  // Writes a module to bytecode and returns the raw bytes.
  std::vector<uint8_t> WriteModule(
      const loom_module_t* module,
      const loom_bytecode_write_options_t* options = nullptr) {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));

    const iree_host_size_t context_allocations = context_allocation_count_;
    IREE_CHECK_OK(
        loom_bytecode_write_module(module, stream, options, &block_pool_));
    EXPECT_EQ(context_allocation_count_, context_allocations);

    // Read the bytes back from the stream.
    iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes(length);
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  // Expects a write attempt to fail with the given status code.
  void ExpectWriteModuleStatus(
      iree_status_code_t expected_status, const loom_module_t* module,
      const loom_bytecode_write_options_t* options = nullptr) {
    iree_io_stream_t* stream = nullptr;
    IREE_ASSERT_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));

    IREE_EXPECT_STATUS_IS(
        expected_status,
        loom_bytecode_write_module(module, stream, options, &block_pool_));

    iree_io_stream_release(stream);
  }

  // Reads a little-endian u16 from raw bytes at the given offset.
  uint16_t ReadU16LE(const std::vector<uint8_t>& bytes, size_t offset) {
    return (uint16_t)bytes[offset] | ((uint16_t)bytes[offset + 1] << 8);
  }

  // Reads a little-endian u32 from raw bytes at the given offset.
  uint32_t ReadU32LE(const std::vector<uint8_t>& bytes, size_t offset) {
    return (uint32_t)bytes[offset] | ((uint32_t)bytes[offset + 1] << 8) |
           ((uint32_t)bytes[offset + 2] << 16) |
           ((uint32_t)bytes[offset + 3] << 24);
  }

  // Reads a little-endian u64 from raw bytes at the given offset.
  uint64_t ReadU64LE(const std::vector<uint8_t>& bytes, size_t offset) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
      value |= (uint64_t)bytes[offset + i] << (i * 8);
    }
    return value;
  }

  // Reads an unsigned varint from raw bytes and advances |offset|.
  uint64_t ReadUVarint(const std::vector<uint8_t>& bytes, size_t* offset) {
    uint64_t value = 0;
    uint32_t shift = 0;
    while (*offset < bytes.size()) {
      uint8_t byte = bytes[(*offset)++];
      value |= (uint64_t)(byte & 0x7F) << shift;
      if ((byte & 0x80) == 0) {
        return value;
      }
      shift += 7;
    }
    return value;
  }

  void SkipSourceTrivia(const std::vector<uint8_t>& bytes, size_t* offset) {
    uint64_t source_trivia = ReadUVarint(bytes, offset);
    uint64_t comment_count =
        source_trivia >> LOOM_BYTECODE_SOURCE_TRIVIA_COMMENT_COUNT_SHIFT;
    for (uint64_t i = 0; i < comment_count; ++i) {
      uint64_t comment_length = ReadUVarint(bytes, offset);
      *offset += (size_t)comment_length;
    }
  }

  void SkipValueDef(const std::vector<uint8_t>& bytes, size_t* offset) {
    ReadUVarint(bytes, offset);  // name_id
    ReadUVarint(bytes, offset);  // Tagged type reference.
    const uint64_t length = ReadUVarint(bytes, offset);
    *offset += length;
  }

  struct SectionEntry {
    // Section kind from loom_bytecode_section_kind_t.
    uint16_t kind;
    // Byte offset from the start of the module.
    uint64_t offset;
    // Byte length of the section payload.
    uint64_t length;
  };

  // Reads the module section directory after the module allocation summary.
  std::vector<SectionEntry> ReadSectionDirectory(
      const std::vector<uint8_t>& bytes, uint64_t module_offset) {
    size_t section_offset = (size_t)module_offset;
    uint64_t section_count = ReadUVarint(bytes, &section_offset);
    // Module allocation summary: value, region, block, and op counts.
    ReadUVarint(bytes, &section_offset);
    ReadUVarint(bytes, &section_offset);
    ReadUVarint(bytes, &section_offset);
    ReadUVarint(bytes, &section_offset);

    std::vector<SectionEntry> entries;
    entries.reserve((size_t)section_count);
    for (uint64_t i = 0; i < section_count; ++i) {
      entries.push_back(SectionEntry{
          /*.kind=*/ReadU16LE(bytes, section_offset),
          /*.offset=*/ReadU64LE(bytes, section_offset + 8),
          /*.length=*/ReadU64LE(bytes, section_offset + 16),
      });
      section_offset += sizeof(loom_bytecode_section_dir_entry_t);
    }
    return entries;
  }

  // Finds a section entry by kind.
  bool FindSection(const std::vector<SectionEntry>& entries, uint16_t kind,
                   SectionEntry* out_entry) {
    for (const SectionEntry& entry : entries) {
      if (entry.kind == kind) {
        *out_entry = entry;
        return true;
      }
    }
    return false;
  }

  size_t SectionPayloadOffset(const std::vector<uint8_t>& bytes,
                              uint16_t section_kind) {
    size_t dir_offset = 24;
    uint64_t module_offset = ReadU64LE(bytes, dir_offset + 8);
    auto entries = ReadSectionDirectory(bytes, module_offset);
    SectionEntry section_entry = {};
    if (!FindSection(entries, section_kind, &section_entry)) {
      return 0;
    }
    return (size_t)module_offset + (size_t)section_entry.offset;
  }

  // Counts context-owned requests independently of module and writer arenas.
  iree_host_size_t context_allocation_count_ = 0;
  // Shared pool for source IR and invocation-local writer scratch.
  iree_arena_block_pool_t block_pool_;
  // Durable compiler descriptors whose allocator must not own writer scratch.
  loom_context_t context_;
};

//===----------------------------------------------------------------------===//
// File header tests
//===----------------------------------------------------------------------===//

TEST_F(WriterTest, FileHeaderMagic) {
  loom_module_t* module = CreateModule("test");
  auto bytes = WriteModule(module);

  ASSERT_GE(bytes.size(), 16u);
  EXPECT_EQ(bytes[0], 'L');
  EXPECT_EQ(bytes[1], 'O');
  EXPECT_EQ(bytes[2], 'O');
  EXPECT_EQ(bytes[3], 'M');

  loom_module_free(module);
}

TEST_F(WriterTest, FileHeaderVersion) {
  loom_module_t* module = CreateModule("test");
  auto bytes = WriteModule(module);

  EXPECT_EQ(bytes[4], LOOM_BYTECODE_FORMAT_VERSION);

  loom_module_free(module);
}

TEST_F(WriterTest, FileHeaderLocationMode) {
  loom_module_t* module = CreateModule("test");
  auto bytes = WriteModule(module);

  EXPECT_EQ(bytes[5], LOOM_BYTECODE_LOCATION_MODE_SOURCE_LOCATIONS);

  loom_module_free(module);
}

TEST_F(WriterTest, NoLocationsModeOmitsLocationsSection) {
  loom_module_t* module = CreateModule("test");
  loom_bytecode_write_options_t options = {{0}};
  options.location_mode = LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS;
  auto bytes = WriteModule(module, &options);

  EXPECT_EQ(bytes[5], LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS);

  size_t dir_offset = 24;
  uint64_t module_offset = ReadU64LE(bytes, dir_offset + 8);
  auto entries = ReadSectionDirectory(bytes, module_offset);
  SectionEntry locations_entry = {};
  EXPECT_FALSE(
      FindSection(entries, LOOM_BYTECODE_SECTION_LOCATIONS, &locations_entry));

  loom_module_free(module);
}

TEST_F(WriterTest, FullLocationsModeFailsUntilFieldSpansExist) {
  loom_module_t* module = CreateModule("test");
  loom_bytecode_write_options_t options = {{0}};
  options.location_mode = LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS;

  ExpectWriteModuleStatus(IREE_STATUS_UNIMPLEMENTED, module, &options);

  loom_module_free(module);
}

TEST_F(WriterTest, FileHeaderModuleCount) {
  loom_module_t* module = CreateModule("test");
  auto bytes = WriteModule(module);

  EXPECT_EQ(ReadU16LE(bytes, 6), 1);

  loom_module_free(module);
}

TEST_F(WriterTest, FileHeaderStringPoolLength) {
  loom_module_t* module = CreateModule("my_module");
  auto bytes = WriteModule(module);

  // String pool length should be the module name length.
  EXPECT_EQ(ReadU32LE(bytes, 8), 9u);  // "my_module" = 9 bytes

  loom_module_free(module);
}

TEST_F(WriterTest, FileHeaderProducer) {
  loom_module_t* module = CreateModule("test");
  auto bytes = WriteModule(module);

  // Default producer is "loom-c", starting at offset 16.
  std::string producer(reinterpret_cast<const char*>(&bytes[16]));
  EXPECT_EQ(producer, "loom-c");

  loom_module_free(module);
}

TEST_F(WriterTest, FileHeaderAlignment) {
  loom_module_t* module = CreateModule("test");
  auto bytes = WriteModule(module);

  // Header (16 bytes) + "loom-c\0" (7 bytes) = 23 bytes, padded to 24.
  // Module directory starts at offset 24.
  size_t header_end = 16 + strlen("loom-c") + 1;
  size_t padded = (header_end + 7) & ~(size_t)7;
  EXPECT_EQ(padded, 24u);

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Module directory tests
//===----------------------------------------------------------------------===//

TEST_F(WriterTest, ModuleDirectoryEntry) {
  loom_module_t* module = CreateModule("test");
  auto bytes = WriteModule(module);

  // Module directory starts after header (padded to 8).
  // Header: 16 + 7 (loom-c\0) = 23 -> padded to 24.
  size_t dir_offset = 24;
  ASSERT_GE(bytes.size(), dir_offset + 24);

  // name_offset should be 0 (first string in pool).
  EXPECT_EQ(ReadU32LE(bytes, dir_offset), 0u);
  // name_length should be 4 ("test").
  EXPECT_EQ(ReadU16LE(bytes, dir_offset + 4), 4u);
  // module_flags should be 0.
  EXPECT_EQ(ReadU16LE(bytes, dir_offset + 6), 0u);

  // module_offset should point past the directory + string pool.
  uint64_t module_offset = ReadU64LE(bytes, dir_offset + 8);
  EXPECT_GT(module_offset, 0u);
  // module_length should be nonzero.
  uint64_t module_length = ReadU64LE(bytes, dir_offset + 16);
  EXPECT_GT(module_length, 0u);
  // Module data should not extend past the file.
  EXPECT_LE(module_offset + module_length, bytes.size());

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Section directory tests
//===----------------------------------------------------------------------===//

TEST_F(WriterTest, SectionDirectoryHasWrittenSections) {
  loom_module_t* module = CreateModule("test");
  auto bytes = WriteModule(module);

  // Find module_offset from the module directory.
  size_t dir_offset = 24;  // After header padding.
  uint64_t module_offset = ReadU64LE(bytes, dir_offset + 8);

  auto entries = ReadSectionDirectory(bytes, module_offset);
  ASSERT_EQ(entries.size(), 9u);

  bool found_kinds[LOOM_BYTECODE_SECTION_COUNT] = {false};
  uint64_t previous_end = 0;
  for (const SectionEntry& entry : entries) {
    uint16_t kind = entry.kind;
    EXPECT_LT(kind, (uint16_t)LOOM_BYTECODE_SECTION_COUNT)
        << "section has invalid kind " << kind;
    EXPECT_GE(entry.offset, previous_end);
    previous_end = entry.offset + entry.length;
    found_kinds[kind] = true;
  }

  for (int i = LOOM_BYTECODE_SECTION_STRINGS; i <= LOOM_BYTECODE_SECTION_IR;
       ++i) {
    EXPECT_TRUE(found_kinds[i]) << "missing section kind " << i;
  }
  EXPECT_FALSE(found_kinds[LOOM_BYTECODE_SECTION_RESOURCES]);
  EXPECT_FALSE(found_kinds[LOOM_BYTECODE_SECTION_SOURCE_TRIVIA]);
  EXPECT_TRUE(found_kinds[LOOM_BYTECODE_SECTION_SYMBOL_REFERENCES]);

  loom_module_free(module);
}

TEST_F(WriterTest, SectionOffsetsAreWithinModule) {
  loom_module_t* module = CreateModule("test");
  auto bytes = WriteModule(module);

  size_t dir_offset = 24;
  uint64_t module_offset = ReadU64LE(bytes, dir_offset + 8);
  uint64_t module_length = ReadU64LE(bytes, dir_offset + 16);

  auto entries = ReadSectionDirectory(bytes, module_offset);
  for (const SectionEntry& entry : entries) {
    EXPECT_LE(entry.offset + entry.length, module_length)
        << "section " << entry.kind << " extends past module boundary";
  }

  loom_module_free(module);
}

TEST_F(WriterTest, SymbolWithoutSerializableKindFails) {
  loom_module_t* module = CreateModule("unresolved");
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("missing"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));

  ExpectWriteModuleStatus(IREE_STATUS_INVALID_ARGUMENT, module);

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Strings section test
//===----------------------------------------------------------------------===//

TEST_F(WriterTest, StringsSectionContainsModuleName) {
  loom_module_t* module = CreateModule("hello");
  auto bytes = WriteModule(module);

  // Find the STRINGS section data.
  size_t dir_offset = 24;
  uint64_t module_offset = ReadU64LE(bytes, dir_offset + 8);

  auto entries = ReadSectionDirectory(bytes, module_offset);
  SectionEntry strings_entry = {};
  ASSERT_TRUE(
      FindSection(entries, LOOM_BYTECODE_SECTION_STRINGS, &strings_entry));
  ASSERT_GT(strings_entry.length, 0u);

  // Parse the strings section using a cursor.
  const uint8_t* section_data =
      bytes.data() + (size_t)module_offset + (size_t)strings_entry.offset;
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(
      section_data, (iree_host_size_t)strings_entry.length, &cursor);

  uint64_t string_count = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &string_count));
  EXPECT_GE(string_count, 2u);

  // The first string is reserved for the anonymous SSA value-name sentinel.
  uint64_t first_length = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &first_length));
  EXPECT_EQ(first_length, 0u);

  // The next string should be the module name "hello".
  uint64_t second_length = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &second_length));
  EXPECT_EQ(second_length, 5u);
  iree_const_byte_span_t span = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_bytecode_cursor_read_span(&cursor, 5, &span));
  EXPECT_EQ(memcmp(span.data, "hello", 5), 0);

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Empty module (no symbols)
//===----------------------------------------------------------------------===//

TEST_F(WriterTest, EmptyModuleRoundTrips) {
  loom_module_t* module = CreateModule("empty");
  auto bytes = WriteModule(module);

  // Should produce a valid file with nonzero size.
  EXPECT_GT(bytes.size(), 100u);

  // Magic is valid.
  EXPECT_EQ(bytes[0], 'L');
  EXPECT_EQ(bytes[1], 'O');
  EXPECT_EQ(bytes[2], 'O');
  EXPECT_EQ(bytes[3], 'M');

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Module with a function
//===----------------------------------------------------------------------===//

TEST_F(WriterTest, ModuleWithFunction) {
  loom_module_t* module = CreateModule("func_test");

  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  IREE_ASSERT_OK(loom_module_intern_type(module, i32_type, &i32_type));

  // Build a test.func op on the module body block with two i32 args and one
  // i32 result. The builder wires the defining_op pointer on the symbol so
  // the writer can find and serialize the function metadata.
  loom_builder_t module_builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &module_builder);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&module_builder, IREE_SV("add"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
  loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  loom_type_t arg_types[2] = {i32_type, i32_type};
  loom_type_t result_types[1] = {i32_type};
  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(&module_builder, 0, /*visibility=*/0,
                                      /*cc=*/0, callee, arg_types, 2,
                                      result_types, 1, nullptr, 0, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &func_op));
  module->symbols.entries[symbol_id].flags = LOOM_SYMBOL_FLAG_PUBLIC;

  // Name the entry-block args.
  loom_func_like_t func_like = loom_func_like_cast(module, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids =
      loom_func_like_arg_ids(func_like, &arg_count);
  loom_string_id_t x_name = LOOM_STRING_ID_INVALID;
  loom_string_id_t y_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("x"), &x_name));
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("y"), &y_name));
  loom_module_value(module, arg_ids[0])->name_id = x_name;
  loom_module_value(module, arg_ids[1])->name_id = y_name;

  // Build an addi op in the function body.
  loom_region_t* body = loom_func_like_body(func_like);
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_region_entry_block(body),
                          &builder);
  loom_op_t* addi_op = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder, arg_ids[0], arg_ids[1],
                                      i32_type, LOOM_LOCATION_UNKNOWN,
                                      &addi_op));

  auto bytes = WriteModule(module);
  EXPECT_GT(bytes.size(), 200u);

  // Verify the magic is still valid (sanity check that writing completed).
  EXPECT_EQ(bytes[0], 'L');
  EXPECT_EQ(bytes[1], 'O');
  EXPECT_EQ(bytes[2], 'O');
  EXPECT_EQ(bytes[3], 'M');

  // Find the OPS section and verify it contains "test.addi".
  size_t header_dir_offset = 24;
  uint64_t module_offset = ReadU64LE(bytes, header_dir_offset + 8);
  auto entries = ReadSectionDirectory(bytes, module_offset);
  SectionEntry ops_entry = {};
  ASSERT_TRUE(FindSection(entries, LOOM_BYTECODE_SECTION_OPS, &ops_entry));
  EXPECT_GT(ops_entry.length, 0u);

  // Parse the OPS section: [count: uvarint], per-op [name_id: uvarint].
  const uint8_t* ops_data =
      bytes.data() + (size_t)module_offset + (size_t)ops_entry.offset;
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(ops_data, (iree_host_size_t)ops_entry.length,
                                  &cursor);
  uint64_t op_count = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &op_count));
  EXPECT_GE(op_count, 1u);

  loom_module_free(module);
}

TEST_F(WriterTest, FunctionSymbolKindUsesDenseWireEnum) {
  loom_module_t* module = CreateAttrsModule(/*reverse_attr_order=*/false);
  auto bytes = WriteModule(module);

  size_t dir_offset = 24;
  uint64_t module_offset = ReadU64LE(bytes, dir_offset + 8);
  auto entries = ReadSectionDirectory(bytes, module_offset);
  SectionEntry symbols_entry = {};
  ASSERT_TRUE(
      FindSection(entries, LOOM_BYTECODE_SECTION_SYMBOLS, &symbols_entry));

  const uint8_t* symbols_data =
      bytes.data() + (size_t)module_offset + (size_t)symbols_entry.offset;
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(
      symbols_data, (iree_host_size_t)symbols_entry.length, &cursor);
  uint64_t symbol_count = 0;
  uint64_t import_count = 0;
  uint64_t export_count = 0;
  uint64_t root_region_payload_count = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &symbol_count));
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &import_count));
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &export_count));
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &root_region_payload_count));
  ASSERT_EQ(symbol_count, 1u);
  ASSERT_EQ(import_count, 0u);
  ASSERT_EQ(export_count, 1u);
  ASSERT_EQ(root_region_payload_count, 1u);

  iree_const_byte_span_t export_table = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_bytecode_cursor_read_span(&cursor, 8, &export_table));
  EXPECT_EQ(export_table.data_length, 8u);

  uint64_t name_id = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &name_id));
  uint8_t kind = 0;
  IREE_ASSERT_OK(loom_bytecode_cursor_read_u8(&cursor, &kind));
  EXPECT_EQ(kind, LOOM_BYTECODE_SYMBOL_FUNC_DEF);

  loom_module_free(module);
}

TEST_F(WriterTest, ExportOffsetsCrossWriterPageBoundary) {
  constexpr uint32_t kSymbolCount =
      LOOM_BYTECODE_WRITER_PAGE_SIZE / sizeof(uint64_t) + 3;
  loom_module_t* module = CreateModule("exports");
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);

  for (uint32_t i = 0; i < kSymbolCount; ++i) {
    const std::string name = "export_" + std::to_string(i);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(
        &builder, iree_make_string_view(name.data(), name.size()), &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_op_t* function_op = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(
        &builder, LOOM_TEST_FUNC_BUILD_FLAG_HAS_VISIBILITY,
        LOOM_TEST_VISIBILITY_PUBLIC, /*cc=*/0,
        {/*.module_id=*/0, /*.symbol_id=*/symbol_id}, /*arg_types=*/nullptr,
        /*arg_count=*/0, /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicate_count=*/0, LOOM_LOCATION_NONE,
        &function_op));
  }

  const std::vector<uint8_t> bytes = WriteModule(module);
  const size_t module_directory_offset = 24;
  const uint64_t module_offset = ReadU64LE(bytes, module_directory_offset + 8);
  const std::vector<SectionEntry> sections =
      ReadSectionDirectory(bytes, module_offset);
  SectionEntry symbols = {};
  ASSERT_TRUE(FindSection(sections, LOOM_BYTECODE_SECTION_SYMBOLS, &symbols));
  const size_t section_start = (size_t)module_offset + symbols.offset;
  const size_t section_end = section_start + symbols.length;
  size_t cursor = section_start;
  ASSERT_EQ(ReadUVarint(bytes, &cursor), kSymbolCount);
  ASSERT_EQ(ReadUVarint(bytes, &cursor), 0u);  // import_count
  ASSERT_EQ(ReadUVarint(bytes, &cursor), kSymbolCount);
  ReadUVarint(bytes, &cursor);  // root_region_payload_count

  const size_t export_table_start = cursor;
  const size_t entries_start =
      export_table_start + kSymbolCount * sizeof(uint64_t);
  ASSERT_GT(entries_start - export_table_start, LOOM_BYTECODE_WRITER_PAGE_SIZE);
  uint64_t previous_entry_offset = 0;
  for (uint32_t i = 0; i < kSymbolCount; ++i) {
    const uint64_t entry_offset =
        ReadU64LE(bytes, export_table_start + i * sizeof(uint64_t));
    if (i == 0) {
      EXPECT_EQ(entry_offset, 0u);
    } else {
      EXPECT_GT(entry_offset, previous_entry_offset);
    }
    previous_entry_offset = entry_offset;

    size_t entry_cursor = entries_start + (size_t)entry_offset;
    ASSERT_LT(entry_cursor, section_end);
    ReadUVarint(bytes, &entry_cursor);  // name_id
    ASSERT_EQ(bytes[entry_cursor++], LOOM_BYTECODE_SYMBOL_FUNC_DEF);
    ASSERT_EQ(bytes[entry_cursor++], LOOM_BYTECODE_SYMBOL_VISIBILITY_PUBLIC);
    EXPECT_NE(ReadU16LE(bytes, entry_cursor) & LOOM_BYTECODE_SYMBOL_FLAG_EXPORT,
              0u);
  }

  loom_module_free(module);
}

TEST_F(WriterTest, ImportAndExportOffsetsAddressMatchingEntries) {
  loom_module_t* module = CreateModule("linkage");
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);

  loom_string_id_t import_name = LOOM_STRING_ID_INVALID;
  loom_string_id_t import_module = LOOM_STRING_ID_INVALID;
  loom_string_id_t import_symbol = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder, IREE_SV("local"), &import_name));
  IREE_ASSERT_OK(loom_builder_intern_string(&builder, IREE_SV("provider"),
                                            &import_module));
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder, IREE_SV("remote"), &import_symbol));
  loom_symbol_id_t import_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module, import_name, &import_symbol_id));
  loom_op_t* import_op = nullptr;
  IREE_ASSERT_OK(loom_func_decl_build(
      &builder,
      LOOM_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_MODULE |
          LOOM_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_SYMBOL,
      /*visibility=*/0, /*retain=*/0, import_module, import_symbol,
      /*cc=*/0, /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
      loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(),
      {/*.module_id=*/0, /*.symbol_id=*/import_symbol_id},
      /*arg_types=*/nullptr, /*arg_types_count=*/0, /*result_types=*/nullptr,
      /*result_count=*/0, /*tied_results=*/nullptr, /*tied_result_count=*/0,
      /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_NONE,
      &import_op));

  loom_string_id_t export_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder, IREE_SV("exported"), &export_name));
  loom_symbol_id_t export_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module, export_name, &export_symbol_id));
  loom_op_t* export_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder, LOOM_TEST_FUNC_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_TEST_VISIBILITY_PUBLIC, /*cc=*/0,
      {/*.module_id=*/0, /*.symbol_id=*/export_symbol_id},
      /*arg_types=*/nullptr, /*arg_count=*/0, /*result_types=*/nullptr,
      /*result_count=*/0, /*tied_results=*/nullptr, /*tied_result_count=*/0,
      /*predicates=*/nullptr, /*predicate_count=*/0, LOOM_LOCATION_NONE,
      &export_op));

  const std::vector<uint8_t> bytes = WriteModule(module);
  const size_t module_directory_offset = 24;
  const uint64_t module_offset = ReadU64LE(bytes, module_directory_offset + 8);
  const std::vector<SectionEntry> sections =
      ReadSectionDirectory(bytes, module_offset);
  SectionEntry symbols = {};
  ASSERT_TRUE(FindSection(sections, LOOM_BYTECODE_SECTION_SYMBOLS, &symbols));
  const size_t section_start = (size_t)module_offset + symbols.offset;
  const size_t section_end = section_start + symbols.length;
  size_t cursor = section_start;
  ASSERT_EQ(ReadUVarint(bytes, &cursor), 2u);  // symbol_count
  ASSERT_EQ(ReadUVarint(bytes, &cursor), 1u);  // import_count
  ASSERT_EQ(ReadUVarint(bytes, &cursor), 1u);  // export_count
  ReadUVarint(bytes, &cursor);                 // root_region_payload_count

  const uint64_t import_offset = ReadU64LE(bytes, cursor);
  const uint64_t export_offset = ReadU64LE(bytes, cursor + sizeof(uint64_t));
  const size_t entries_start = cursor + 2 * sizeof(uint64_t);

  size_t import_cursor = entries_start + (size_t)import_offset;
  ASSERT_LT(import_cursor, section_end);
  ReadUVarint(bytes, &import_cursor);  // name_id
  EXPECT_EQ(bytes[import_cursor++], LOOM_BYTECODE_SYMBOL_FUNC_DECL);
  EXPECT_EQ(bytes[import_cursor++], LOOM_BYTECODE_SYMBOL_VISIBILITY_PRIVATE);
  const uint16_t import_flags = ReadU16LE(bytes, import_cursor);
  EXPECT_NE(import_flags & LOOM_BYTECODE_SYMBOL_FLAG_IMPORT, 0u);
  EXPECT_NE(import_flags & LOOM_BYTECODE_SYMBOL_FLAG_IMPORT_SYMBOL, 0u);
  EXPECT_EQ(import_flags & LOOM_BYTECODE_SYMBOL_FLAG_EXPORT, 0u);

  size_t export_cursor = entries_start + (size_t)export_offset;
  ASSERT_LT(export_cursor, section_end);
  ReadUVarint(bytes, &export_cursor);  // name_id
  EXPECT_EQ(bytes[export_cursor++], LOOM_BYTECODE_SYMBOL_FUNC_DEF);
  EXPECT_EQ(bytes[export_cursor++], LOOM_BYTECODE_SYMBOL_VISIBILITY_PUBLIC);
  const uint16_t export_flags = ReadU16LE(bytes, export_cursor);
  EXPECT_NE(export_flags & LOOM_BYTECODE_SYMBOL_FLAG_EXPORT, 0u);
  EXPECT_EQ(export_flags & LOOM_BYTECODE_SYMBOL_FLAG_IMPORT, 0u);

  loom_module_free(module);
}

TEST_F(WriterTest, UnsupportedRegionSourceFlagsFailLoudly) {
  loom_module_t* module = CreateAttrsModule(/*reverse_attr_order=*/false);
  loom_op_t* func_op = module->symbols.entries[0].defining_op;
  loom_region_t* body =
      loom_func_like_body(loom_func_like_cast(module, func_op));
  ASSERT_NE(body, nullptr);
  body->source_flags = 1u << 1;

  ExpectWriteModuleStatus(IREE_STATUS_INVALID_ARGUMENT, module);

  loom_module_free(module);
}

TEST_F(WriterTest, FunctionBodySummaryAndOpTableRefsUseNewWireShape) {
  loom_module_t* module = CreateModule("test");

  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  IREE_ASSERT_OK(loom_module_intern_type(module, i32_type, &i32_type));

  loom_builder_t module_builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &module_builder);
  loom_string_id_t func_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&module_builder, IREE_SV("f"), &func_name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, func_name_id, &symbol_id));
  loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};

  loom_type_t arg_types[2] = {i32_type, i32_type};
  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &module_builder, 0, /*visibility=*/0, /*cc=*/0, callee, arg_types, 2,
      /*result_types=*/nullptr, 0, /*arg_names=*/nullptr, 0,
      /*result_names=*/nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));

  loom_func_like_t func_like = loom_func_like_cast(module, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids =
      loom_func_like_arg_ids(func_like, &arg_count);
  ASSERT_EQ(arg_count, 2);

  loom_block_t* entry_block =
      loom_region_entry_block(loom_func_like_body(func_like));
  loom_builder_t body_builder;
  loom_builder_initialize(module, &module->arena, entry_block, &body_builder);
  loom_op_t* addi_op = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&body_builder, arg_ids[0], arg_ids[1],
                                      i32_type, LOOM_LOCATION_UNKNOWN,
                                      &addi_op));
  entry_block->flags |= LOOM_BLOCK_FLAG_LEADING_BLANK_LINE;
  addi_op->flags |= LOOM_OP_FLAG_LEADING_BLANK_LINE;
  const iree_string_view_t block_comments[] = {IREE_SV("block")};
  const iree_string_view_t op_comments[] = {IREE_SV("operation")};
  IREE_ASSERT_OK(loom_module_attach_block_comments(
      module, entry_block, block_comments, IREE_ARRAYSIZE(block_comments)));
  IREE_ASSERT_OK(loom_module_attach_op_comments(module, addi_op, op_comments,
                                                IREE_ARRAYSIZE(op_comments)));

  auto bytes = WriteModule(module);
  size_t dir_offset = 24;
  uint64_t module_offset = ReadU64LE(bytes, dir_offset + 8);
  auto entries = ReadSectionDirectory(bytes, module_offset);
  SectionEntry ir_entry = {};
  ASSERT_TRUE(FindSection(entries, LOOM_BYTECODE_SECTION_IR, &ir_entry));

  const uint8_t* ir_data =
      bytes.data() + (size_t)module_offset + (size_t)ir_entry.offset;
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(ir_data, (iree_host_size_t)ir_entry.length,
                                  &cursor);

  uint64_t value_count = 0;
  uint64_t region_count = 0;
  uint64_t block_count = 0;
  uint64_t op_count = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &value_count));
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &region_count));
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &block_count));
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &op_count));
  EXPECT_EQ(value_count, 3u);
  EXPECT_EQ(region_count, 1u);
  EXPECT_EQ(block_count, 1u);
  EXPECT_EQ(op_count, 1u);

  uint64_t root_source_flags = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &root_source_flags));
  ASSERT_EQ(root_source_flags, 0u);
  uint64_t root_block_count = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &root_block_count));
  ASSERT_EQ(root_block_count, 1u);
  uint8_t has_label = 0;
  IREE_ASSERT_OK(loom_bytecode_cursor_read_u8(&cursor, &has_label));
  ASSERT_EQ(has_label, 0u);
  uint64_t block_source_trivia = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &block_source_trivia));
  ASSERT_EQ(block_source_trivia, 3u);
  uint64_t block_comment_length = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &block_comment_length));
  ASSERT_EQ(block_comment_length, block_comments[0].size + 1);
  iree_const_byte_span_t block_comment = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_bytecode_cursor_read_span(
      &cursor, (iree_host_size_t)block_comment_length, &block_comment));
  ASSERT_EQ(block_comment.data[0], ' ');
  EXPECT_EQ(memcmp(block_comment.data + 1, block_comments[0].data,
                   block_comments[0].size),
            0);
  uint64_t block_arg_count = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &block_arg_count));
  ASSERT_EQ(block_arg_count, 2u);
  for (uint64_t i = 0; i < block_arg_count; ++i) {
    uint64_t unused = 0;
    IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &unused));  // name_id
    IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &unused));  // Type reference.
    ASSERT_EQ(unused & 1u, 0u);                             // Static type.
    IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &unused));  // Record length.
    ASSERT_EQ(unused, 0u);
  }
  uint64_t block_op_count = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &block_op_count));
  ASSERT_EQ(block_op_count, 1u);
  uint64_t op_table_index_plus1 = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &op_table_index_plus1));
  EXPECT_GT(op_table_index_plus1, 0u);
  uint8_t instance_flags = 0;
  IREE_ASSERT_OK(loom_bytecode_cursor_read_u8(&cursor, &instance_flags));
  ASSERT_EQ(instance_flags, 0u);
  uint64_t location_id = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &location_id));
  ASSERT_EQ(location_id, 0u);
  uint64_t op_source_trivia = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &op_source_trivia));
  ASSERT_EQ(op_source_trivia, 3u);
  uint64_t op_comment_length = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &op_comment_length));
  ASSERT_EQ(op_comment_length, op_comments[0].size + 1);
  iree_const_byte_span_t op_comment = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_bytecode_cursor_read_span(
      &cursor, (iree_host_size_t)op_comment_length, &op_comment));
  ASSERT_EQ(op_comment.data[0], ' ');
  EXPECT_EQ(
      memcmp(op_comment.data + 1, op_comments[0].data, op_comments[0].size), 0);

  loom_module_free(module);
}

TEST_F(WriterTest, FunctionBodySuccessorsUseRegionBlockOrdinals) {
  loom_module_t* module = CreateModule("successors");

  loom_builder_t module_builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &module_builder);
  loom_string_id_t func_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_intern_string(&module_builder, IREE_SV("cfg"),
                                            &func_name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, func_name_id, &symbol_id));
  loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};

  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &module_builder, 0, /*visibility=*/0, /*cc=*/0, callee,
      /*arg_types=*/nullptr, 0, /*result_types=*/nullptr, 0,
      /*arg_names=*/nullptr, 0, /*result_names=*/nullptr, 0,
      LOOM_LOCATION_UNKNOWN, &func_op));

  loom_region_t* body = loom_test_func_body(func_op);
  loom_block_t* entry_block = loom_region_entry_block(body);
  loom_block_t* exit_block = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module, body, &exit_block));

  loom_builder_t body_builder;
  loom_builder_initialize(module, &module->arena, entry_block, &body_builder);
  loom_op_t* br_op = nullptr;
  IREE_ASSERT_OK(loom_test_br_build(&body_builder, exit_block,
                                    LOOM_LOCATION_UNKNOWN, &br_op));

  auto bytes = WriteModule(module);
  size_t dir_offset = 24;
  uint64_t module_offset = ReadU64LE(bytes, dir_offset + 8);
  auto entries = ReadSectionDirectory(bytes, module_offset);
  SectionEntry ir_entry = {};
  ASSERT_TRUE(FindSection(entries, LOOM_BYTECODE_SECTION_IR, &ir_entry));

  const uint8_t* ir_data =
      bytes.data() + (size_t)module_offset + (size_t)ir_entry.offset;
  loom_bytecode_cursor_t cursor;
  loom_bytecode_cursor_initialize(ir_data, (iree_host_size_t)ir_entry.length,
                                  &cursor);

  uint64_t value_count = 0;
  uint64_t region_count = 0;
  uint64_t block_count = 0;
  uint64_t op_count = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &value_count));
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &region_count));
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &block_count));
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &op_count));
  EXPECT_EQ(value_count, 0u);
  EXPECT_EQ(region_count, 1u);
  EXPECT_EQ(block_count, 2u);
  EXPECT_EQ(op_count, 1u);

  uint64_t root_source_flags = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &root_source_flags));
  ASSERT_EQ(root_source_flags, 0u);
  uint64_t root_block_count = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &root_block_count));
  ASSERT_EQ(root_block_count, 2u);
  uint8_t has_label = 0;
  IREE_ASSERT_OK(loom_bytecode_cursor_read_u8(&cursor, &has_label));
  ASSERT_EQ(has_label, 0u);
  uint64_t block_source_trivia = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &block_source_trivia));
  ASSERT_EQ(block_source_trivia, 0u);
  uint64_t block_arg_count = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &block_arg_count));
  ASSERT_EQ(block_arg_count, 0u);
  uint64_t block_op_count = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &block_op_count));
  ASSERT_EQ(block_op_count, 1u);

  uint64_t unused = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &unused));  // op_table_index
  uint8_t flags = 0;
  IREE_ASSERT_OK(loom_bytecode_cursor_read_u8(&cursor, &flags));
  EXPECT_EQ(flags, 0u);
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &unused));  // location_id
  uint64_t op_source_trivia = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &op_source_trivia));
  ASSERT_EQ(op_source_trivia, 0u);
  uint64_t operand_count = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &operand_count));
  ASSERT_EQ(operand_count, 0u);
  uint64_t successor_count = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &successor_count));
  ASSERT_EQ(successor_count, 1u);
  uint64_t successor_block_index = 0;
  IREE_ASSERT_OK(loom_uvarint_decode(&cursor, &successor_block_index));
  EXPECT_EQ(successor_block_index, 1u);

  loom_module_free(module);
}

TEST_F(WriterTest, CanonicalAttrDictInputOrderDoesNotAffectBytes) {
  loom_module_t* module_a = CreateAttrsModule(/*reverse_attr_order=*/false);
  loom_module_t* module_b = CreateAttrsModule(/*reverse_attr_order=*/true);

  EXPECT_EQ(WriteModule(module_a), WriteModule(module_b));

  loom_module_free(module_a);
  loom_module_free(module_b);
}

TEST_F(WriterTest, TypeCatalogBytesDoNotDependOnTypeConstructionOrder) {
  std::vector<uint8_t> canonical_bytes;
  for (int variant = 0; variant < 2; ++variant) {
    loom_module_t* module = CreateModule("type_catalog");
    loom_type_id_t result = LOOM_TYPE_ID_INVALID;
    if (variant == 1) {
      IREE_ASSERT_OK(loom_module_intern_type_id(
          module, loom_type_scalar(LOOM_SCALAR_TYPE_I32), &result));
    }
    loom_type_id_t child = LOOM_TYPE_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_type_id(
        module, loom_type_scalar(LOOM_SCALAR_TYPE_F32), &child));
    for (int level = 0; level < 8; ++level) {
      alignas(loom_func_type_data_t) unsigned char
          storage[sizeof(loom_func_type_data_t) + 2 * sizeof(loom_type_t)] = {};
      auto* data = reinterpret_cast<loom_func_type_data_t*>(storage);
      data->arg_count = 2;
      data->types[0] = data->types[1] =
          loom_type_table_get(&module->types, child);
      const loom_type_id_t dependencies[] = {child, child};
      IREE_ASSERT_OK(loom_module_intern_topological_type_id(
          module, loom_type_function(data), dependencies, 2, &child));
    }

    // The two public construction paths retain identical canonical types in
    // different module order. Wire IDs follow first use through the graph.
    IREE_ASSERT_OK(loom_module_intern_type_id(
        module, loom_type_scalar(LOOM_SCALAR_TYPE_I32), &result));
    loom_type_id_t parent = LOOM_TYPE_ID_INVALID;
    if (variant == 0) {
      alignas(loom_func_type_data_t) unsigned char
          storage[sizeof(loom_func_type_data_t) + 2 * sizeof(loom_type_t)] = {};
      auto* data = reinterpret_cast<loom_func_type_data_t*>(storage);
      data->arg_count = data->result_count = 1;
      data->types[0] = loom_type_table_get(&module->types, child);
      data->types[1] = loom_type_table_get(&module->types, result);
      const loom_type_id_t dependencies[] = {child, result};
      IREE_ASSERT_OK(loom_module_intern_topological_type_id(
          module, loom_type_function(data), dependencies, 2, &parent));
    } else {
      loom_type_t argument_type = loom_type_table_get(&module->types, child);
      loom_type_t result_type = loom_type_table_get(&module->types, result);
      loom_type_t parent_type;
      IREE_ASSERT_OK(loom_module_intern_function_type(
          module, &argument_type, 1, &result_type, 1, &parent_type));
      IREE_ASSERT_OK(loom_module_intern_type_id(module, parent_type, &parent));
    }

    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    loom_string_id_t key = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_module_intern_string(module, IREE_SV("catalog"), &name));
    IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("type"), &key));
    loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module, name, &symbol));
    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    const loom_named_attr_t entry = {key, {}, loom_attr_type(parent)};
    loom_op_t* record = nullptr;
    IREE_ASSERT_OK(loom_test_record_build(
        &builder, LOOM_TEST_RECORD_BUILD_FLAG_HAS_DICT, 0, {0, symbol},
        loom_make_named_attr_slice(&entry, 1), LOOM_LOCATION_NONE, &record));

    const auto source_storage = module->arena.used_allocation_size;
    const auto source_type_count = module->types.count;
    auto bytes = WriteModule(module);
    EXPECT_EQ(WriteModule(module), bytes);
    EXPECT_EQ(module->arena.used_allocation_size, source_storage);
    EXPECT_EQ(module->types.count, source_type_count);
    size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_TYPES);
    ASSERT_NE(offset, 0u);
    EXPECT_EQ(ReadUVarint(bytes, &offset), 11u);
    if (variant == 0) {
      canonical_bytes = bytes;
    } else {
      EXPECT_EQ(bytes, canonical_bytes);
    }
    loom_module_free(module);
  }
}

TEST_F(WriterTest, OptionalAbsentBodyAttrWrites) {
  loom_module_t* module = CreateModule("optional_absent_attr");

  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_builder_t module_builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &module_builder);

  loom_string_id_t func_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&module_builder, IREE_SV("f"), &func_name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, func_name_id, &symbol_id));
  loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};

  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &module_builder, 0, /*visibility=*/0, /*cc=*/0, callee, &f32_type, 1,
      &f32_type, 1, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  module->symbols.entries[symbol_id].flags = LOOM_SYMBOL_FLAG_PUBLIC;

  loom_func_like_t func_like = loom_func_like_cast(module, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids =
      loom_func_like_arg_ids(func_like, &arg_count);
  ASSERT_EQ(arg_count, 1);

  loom_builder_t body_builder;
  loom_builder_initialize(
      module, &module->arena,
      loom_region_entry_block(loom_func_like_body(func_like)), &body_builder);
  loom_op_t* attrs_op = nullptr;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &body_builder, 0, arg_ids[0], loom_make_named_attr_slice(nullptr, 0),
      f32_type, LOOM_LOCATION_UNKNOWN, &attrs_op));
  const loom_value_id_t result_ids[1] = {loom_op_results(attrs_op)[0]};
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&body_builder, result_ids, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));

  auto bytes = WriteModule(module);
  EXPECT_GT(bytes.size(), 0u);

  loom_module_free(module);
}

TEST_F(WriterTest, ZeroExtentVectorTypeWrites) {
  loom_module_t* module = CreateModule("test");

  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(0), 0);

  loom_builder_t module_builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &module_builder);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&module_builder, IREE_SV("empty"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
  loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &module_builder, 0, /*visibility=*/0, /*cc=*/0, callee, &vector_type, 1,
      /*result_types=*/nullptr, 0, /*arg_names=*/nullptr, 0,
      /*result_names=*/nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func_like = loom_func_like_cast(module, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids =
      loom_func_like_arg_ids(func_like, &arg_count);
  ASSERT_EQ(arg_count, 1);
  ASSERT_EQ(module->types.count, 2u);
  EXPECT_TRUE(
      loom_type_equal(vector_type, loom_module_value_type(module, arg_ids[0])));

  auto bytes = WriteModule(module);
  EXPECT_GT(bytes.size(), 0u);

  loom_module_free(module);
}

TEST_F(WriterTest, ProjectsModuleSymbolsIntoPresentationOrder) {
  loom_module_t* module = CreateModule("test");
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);

  loom_string_id_t first_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder, IREE_SV("first"), &first_name_id));
  loom_symbol_id_t first_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module, first_name_id, &first_symbol_id));
  loom_string_id_t second_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder, IREE_SV("second"), &second_name_id));
  loom_symbol_id_t second_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module, second_name_id, &second_symbol_id));

  loom_op_t* second_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder, 0, /*visibility=*/0, /*cc=*/0,
      loom_symbol_ref_t{/*.module_id=*/0, second_symbol_id},
      /*arg_types=*/nullptr, 0, /*result_types=*/nullptr, 0,
      /*arg_names=*/nullptr, 0, /*result_names=*/nullptr, 0,
      LOOM_LOCATION_UNKNOWN, &second_op));
  loom_op_t* first_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder, 0, /*visibility=*/0, /*cc=*/0,
      loom_symbol_ref_t{/*.module_id=*/0, first_symbol_id},
      /*arg_types=*/nullptr, 0, /*result_types=*/nullptr, 0,
      /*arg_names=*/nullptr, 0, /*result_names=*/nullptr, 0,
      LOOM_LOCATION_UNKNOWN, &first_op));

  const loom_symbol_id_t module_symbol_ids[] = {first_symbol_id,
                                                second_symbol_id};
  loom_symbol_id_t wire_symbol_ordinals[] = {LOOM_SYMBOL_ID_INVALID,
                                             LOOM_SYMBOL_ID_INVALID};
  loom_bytecode_write_options_t options = {
      /*.producer=*/{},
      /*.location_mode=*/{},
      /*.low_repr_environment=*/{},
      /*.symbol_projection=*/
      {
          /*.module_symbol_ids=*/module_symbol_ids,
          /*.wire_symbol_ordinals=*/wire_symbol_ordinals,
          /*.count=*/IREE_ARRAYSIZE(module_symbol_ids),
      },
  };
  auto bytes = WriteModule(module, &options);
  EXPECT_GT(bytes.size(), 0u);
  EXPECT_EQ(wire_symbol_ordinals[0], 1u);
  EXPECT_EQ(wire_symbol_ordinals[1], 0u);

  loom_module_free(module);
}

TEST_F(WriterTest, BodyPayloadIndexUsesSparsePoolSizedPages) {
  constexpr iree_host_size_t kPageCapacity =
      LOOM_BYTECODE_IR_REGION_PAGE_CAPACITY;
  constexpr iree_host_size_t kSymbolCount = 2 * kPageCapacity + 1;
  const loom_symbol_id_t body_symbol_ids[] = {
      0,
      (loom_symbol_id_t)(kPageCapacity - 1),
      (loom_symbol_id_t)(2 * kPageCapacity),
  };

  loom_module_t* module = CreateModule("body_payload_index");
  loom_builder_t module_builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &module_builder);
  for (iree_host_size_t i = 0; i < kSymbolCount; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "function_%08zu", i);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(
        module, iree_make_cstring_view(name), &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    ASSERT_EQ(symbol_id, i);

    const bool has_body = symbol_id == body_symbol_ids[0] ||
                          symbol_id == body_symbol_ids[1] ||
                          symbol_id == body_symbol_ids[2];
    if (has_body) {
      loom_op_t* function_op = nullptr;
      IREE_ASSERT_OK(loom_test_func_build(
          &module_builder, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
          {/*.module_id=*/0, /*.symbol_id=*/symbol_id}, /*arg_types=*/nullptr,
          /*arg_count=*/0, /*result_types=*/nullptr, /*result_count=*/0,
          /*tied_results=*/nullptr, /*tied_result_count=*/0,
          /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_NONE,
          &function_op));
      loom_builder_t body_builder;
      loom_builder_initialize(
          module, &module->arena,
          loom_region_entry_block(loom_test_func_body(function_op)),
          &body_builder);
      loom_op_t* yield_op = nullptr;
      IREE_ASSERT_OK(loom_test_yield_build(&body_builder, /*values=*/nullptr,
                                           /*values_count=*/0,
                                           LOOM_LOCATION_NONE, &yield_op));
    } else {
      loom_op_t* declaration_op = nullptr;
      IREE_ASSERT_OK(loom_test_decl_build(
          &module_builder, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
          {/*.module_id=*/0, /*.symbol_id=*/symbol_id}, /*arg_types=*/nullptr,
          /*arg_types_count=*/0, /*result_types=*/nullptr, /*result_count=*/0,
          /*tied_results=*/nullptr, /*tied_result_count=*/0, LOOM_LOCATION_NONE,
          &declaration_op));
    }
  }

  iree_arena_allocator_t writer_arena;
  iree_arena_initialize(&block_pool_, &writer_arena);
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(
      loom_bytecode_numbering_initialize(&numbering, module, &writer_arena));
  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
          IREE_IO_STREAM_MODE_RESIZABLE,
      4096, iree_allocator_system(), &stream));
  loom_bytecode_page_writer_t page_writer;
  loom_bytecode_page_writer_initialize(&page_writer, stream);
  loom_bytecode_ir_region_index_t index = {};
  IREE_ASSERT_OK(
      loom_bytecode_write_ir_section(&page_writer, &numbering, &index));
  IREE_ASSERT_OK(loom_bytecode_page_writer_flush(&page_writer));

  ASSERT_NE(index.pages, nullptr);
  EXPECT_NE(index.pages[0], nullptr);
  EXPECT_EQ(index.pages[1], nullptr);
  EXPECT_NE(index.pages[2], nullptr);
  EXPECT_EQ(index.payload_count, IREE_ARRAYSIZE(body_symbol_ids));
  uint64_t previous_end = 0;
  for (loom_symbol_id_t symbol_id : body_symbol_ids) {
    const loom_bytecode_ir_region_list_t list =
        loom_bytecode_ir_region_index_list(&index, symbol_id);
    ASSERT_EQ(list.count, 1u);
    ASSERT_NE(list.values, nullptr);
    EXPECT_EQ(list.values[0].region_index, 0u);
    EXPECT_GT(list.values[0].length, 0u);
    EXPECT_GE(list.values[0].offset, previous_end);
    previous_end = list.values[0].offset + list.values[0].length;
  }
  const loom_symbol_id_t empty_symbol_ids[] = {
      1,
      (loom_symbol_id_t)(kPageCapacity - 2),
      (loom_symbol_id_t)kPageCapacity,
      (loom_symbol_id_t)(2 * kPageCapacity - 1),
  };
  for (loom_symbol_id_t symbol_id : empty_symbol_ids) {
    const loom_bytecode_ir_region_list_t list =
        loom_bytecode_ir_region_index_list(&index, symbol_id);
    EXPECT_EQ(list.values, nullptr);
    EXPECT_EQ(list.count, 0u);
  }
  EXPECT_EQ(writer_arena.allocation_head, nullptr);

  iree_io_stream_release(stream);
  iree_arena_deinitialize(&writer_arena);

  iree_arena_block_pool_statistics_t before;
  iree_arena_block_pool_query_statistics(&block_pool_, &before);
  const std::vector<uint8_t> bytes = WriteModule(module);
  iree_arena_block_pool_statistics_t after;
  iree_arena_block_pool_query_statistics(&block_pool_, &after);
  EXPECT_EQ(after.oversized_allocation_count,
            before.oversized_allocation_count);
  EXPECT_EQ(after.oversized_allocation_bytes,
            before.oversized_allocation_bytes);

  size_t symbols_offset =
      SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_SYMBOLS);
  ASSERT_NE(symbols_offset, 0u);
  EXPECT_EQ(ReadUVarint(bytes, &symbols_offset), kSymbolCount);
  ReadUVarint(bytes, &symbols_offset);  // Import count.
  ReadUVarint(bytes, &symbols_offset);  // Export count.
  EXPECT_EQ(ReadUVarint(bytes, &symbols_offset),
            IREE_ARRAYSIZE(body_symbol_ids));

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

TEST_F(WriterTest, NullContextFails) {
  // Create module without context.
  loom_module_t* module = CreateModule("test");
  loom_context_t* saved_context = module->context;
  module->context = nullptr;

  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
          IREE_IO_STREAM_MODE_RESIZABLE,
      4096, iree_allocator_system(), &stream));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_bytecode_write_module(module, stream, nullptr, &block_pool_));

  iree_io_stream_release(stream);
  module->context = saved_context;
  loom_module_free(module);
}

TEST_F(WriterTest, NonSeekableStreamFails) {
  loom_module_t* module = CreateModule("test");

  // Create a writable but non-seekable stream.
  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_RESIZABLE, 4096,
      iree_allocator_system(), &stream));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      loom_bytecode_write_module(module, stream, nullptr, &block_pool_));

  iree_io_stream_release(stream);
  loom_module_free(module);
}

TEST_F(WriterTest, ClosedEnumAttributeRejectsFutureOrdinal) {
  loom_module_t* module = CreateModule("closed_enum");
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  IREE_ASSERT_OK(loom_module_intern_type(module, i32_type, &i32_type));

  loom_builder_t module_builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &module_builder);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&module_builder, IREE_SV("f"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
  loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &module_builder, 0, /*visibility=*/0, /*cc=*/0, callee, &i32_type, 1,
      &i32_type, 1, /*tied_results=*/nullptr, 0, /*predicates=*/nullptr, 0,
      LOOM_LOCATION_UNKNOWN, &func_op));

  loom_func_like_t func_like = loom_func_like_cast(module, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids =
      loom_func_like_arg_ids(func_like, &arg_count);
  ASSERT_EQ(arg_count, 1);
  loom_builder_t body_builder;
  loom_builder_initialize(
      module, &module->arena,
      loom_region_entry_block(loom_func_like_body(func_like)), &body_builder);
  loom_op_t* cmp_op = nullptr;
  IREE_ASSERT_OK(loom_test_cmp_build(&body_builder, LOOM_TEST_CMP_PREDICATE_EQ,
                                     arg_ids[0], arg_ids[0],
                                     LOOM_LOCATION_UNKNOWN, &cmp_op));
  IREE_ASSERT_OK(
      loom_test_cmp_set_predicate(module, cmp_op, loom_attr_enum(250)));
  loom_value_id_t result_id = loom_test_cmp_result(cmp_op);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&body_builder, &result_id, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));

  ExpectWriteModuleStatus(IREE_STATUS_INVALID_ARGUMENT, module);
  loom_module_free(module);
}

TEST_F(WriterTest, RankZeroVectorTypeFails) {
  loom_module_t* module = CreateModule("test");

  loom_type_t vector_type =
      loom_type_shaped_0d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, 0);
  IREE_ASSERT_OK(loom_module_intern_type(module, vector_type, &vector_type));

  loom_builder_t module_builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &module_builder);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&module_builder, IREE_SV("bad"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
  loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &module_builder, 0, /*visibility=*/0, /*cc=*/0, callee, &vector_type, 1,
      /*result_types=*/nullptr, 0, /*arg_names=*/nullptr, 0,
      /*result_names=*/nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));

  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
          IREE_IO_STREAM_MODE_RESIZABLE,
      4096, iree_allocator_system(), &stream));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_bytecode_write_module(module, stream, nullptr, &block_pool_));

  iree_io_stream_release(stream);
  loom_module_free(module);
}

TEST_F(WriterTest, VectorEncodingAttachmentFails) {
  loom_module_t* module = CreateModule("test");

  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 7);
  IREE_ASSERT_OK(loom_module_intern_type(module, vector_type, &vector_type));

  loom_builder_t module_builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &module_builder);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&module_builder, IREE_SV("bad"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
  loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &module_builder, 0, /*visibility=*/0, /*cc=*/0, callee, &vector_type, 1,
      /*result_types=*/nullptr, 0, /*arg_names=*/nullptr, 0,
      /*result_names=*/nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));

  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
          IREE_IO_STREAM_MODE_RESIZABLE,
      4096, iree_allocator_system(), &stream));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_bytecode_write_module(module, stream, nullptr, &block_pool_));

  iree_io_stream_release(stream);
  loom_module_free(module);
}

TEST_F(WriterTest, InvalidEncodingRoleFails) {
  loom_module_t* module = CreateModule("test");

  loom_type_t encoding_type =
      loom_type_encoding_with_role((loom_encoding_role_t)99);
  IREE_ASSERT_OK(
      loom_module_intern_type(module, encoding_type, &encoding_type));

  loom_builder_t module_builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &module_builder);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&module_builder, IREE_SV("bad"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
  loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &module_builder, 0, /*visibility=*/0, /*cc=*/0, callee, &encoding_type, 1,
      /*result_types=*/nullptr, 0, /*arg_names=*/nullptr, 0,
      /*result_names=*/nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));

  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
          IREE_IO_STREAM_MODE_RESIZABLE,
      4096, iree_allocator_system(), &stream));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_bytecode_write_module(module, stream, nullptr, &block_pool_));

  iree_io_stream_release(stream);
  loom_module_free(module);
}

TEST_F(WriterTest, DanglingGlobalSymbolFailsLoudly) {
  loom_module_t* module = CreateModule("test");

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("g"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
  module->symbols.entries[symbol_id].kind = LOOM_SYMBOL_GLOBAL;

  ExpectWriteModuleStatus(IREE_STATUS_INVALID_ARGUMENT, module);

  loom_module_free(module);
}

TEST_F(WriterTest, GlobalSymbolWritesDefiningOpPayload) {
  loom_module_t* module = CreateModule("test");
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  IREE_ASSERT_OK(loom_module_intern_type(module, f32_type, &f32_type));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_intern_string(&builder, IREE_SV("pi"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
  loom_symbol_ref_t symbol = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  loom_op_t* global_op = nullptr;
  IREE_ASSERT_OK(loom_global_constant_build(
      &builder, 0, symbol, f32_type, /*predicates=*/nullptr,
      /*predicates_count=*/0, loom_attr_f64(3.25), LOOM_LOCATION_UNKNOWN,
      &global_op));

  std::vector<uint8_t> bytes = WriteModule(module);
  size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_SYMBOLS);
  ASSERT_GT(offset, 0u);
  uint64_t symbol_count = ReadUVarint(bytes, &offset);
  ASSERT_EQ(symbol_count, 1u);
  uint64_t import_count = ReadUVarint(bytes, &offset);
  uint64_t export_count = ReadUVarint(bytes, &offset);
  ReadUVarint(bytes, &offset);  // root_region_payload_count
  offset += (import_count + export_count) * sizeof(uint64_t);
  ReadUVarint(bytes, &offset);  // name_id
  EXPECT_EQ(bytes[offset++], LOOM_BYTECODE_SYMBOL_GLOBAL);
  offset += 1;                 // visibility
  offset += sizeof(uint16_t);  // flags
  EXPECT_EQ(ReadUVarint(bytes, &offset), LOOM_LOCATION_UNKNOWN);
  EXPECT_NE(ReadUVarint(bytes, &offset), 0u);
  SkipSourceTrivia(bytes, &offset);
  EXPECT_EQ(ReadUVarint(bytes, &offset), 1u);
  EXPECT_EQ(ReadUVarint(bytes, &offset), 1u);
  SkipValueDef(bytes, &offset);
  EXPECT_EQ(ReadUVarint(bytes, &offset), 1u);
  ReadUVarint(bytes, &offset);  // initializer key id
  EXPECT_EQ(bytes[offset++], 1u);

  loom_module_free(module);
}

TEST_F(WriterTest, GlobalSymbolWritesDeclarationLocalValues) {
  loom_module_t* module = CreateModule("test");
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  IREE_ASSERT_OK(loom_module_intern_type(module, index_type, &index_type));

  loom_value_id_t dim_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, index_type, &dim_id));
  loom_string_id_t dim_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("n"), &dim_name_id));
  IREE_ASSERT_OK(loom_module_set_value_name(module, dim_id, dim_name_id));

  loom_type_t tile_type =
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(dim_id), /*encoding_id=*/0);
  loom_predicate_t* predicates = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate_array(
      &module->arena, 1, sizeof(loom_predicate_t), (void**)&predicates));
  predicates[0] = loom_predicate_t{
      /*.kind=*/LOOM_PREDICATE_MUL,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
      /*.reserved=*/{},
      /*.args=*/{(int64_t)dim_id, 16},
  };

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder, IREE_SV("weights"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
  loom_symbol_ref_t symbol = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  loom_op_t* global_op = nullptr;
  IREE_ASSERT_OK(loom_global_constant_build(
      &builder, LOOM_GLOBAL_CONSTANT_BUILD_FLAG_HAS_PREDICATES, symbol,
      tile_type, predicates, 1, loom_attr_absent(), LOOM_LOCATION_UNKNOWN,
      &global_op));

  std::vector<uint8_t> bytes = WriteModule(module);
  size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_SYMBOLS);
  ASSERT_GT(offset, 0u);
  uint64_t symbol_count = ReadUVarint(bytes, &offset);
  ASSERT_EQ(symbol_count, 1u);
  uint64_t import_count = ReadUVarint(bytes, &offset);
  uint64_t export_count = ReadUVarint(bytes, &offset);
  ReadUVarint(bytes, &offset);  // root_region_payload_count
  offset += (import_count + export_count) * sizeof(uint64_t);
  ReadUVarint(bytes, &offset);  // name_id
  EXPECT_EQ(bytes[offset++], LOOM_BYTECODE_SYMBOL_GLOBAL);
  offset += 1;                 // visibility
  offset += sizeof(uint16_t);  // flags
  EXPECT_EQ(ReadUVarint(bytes, &offset), LOOM_LOCATION_UNKNOWN);
  EXPECT_NE(ReadUVarint(bytes, &offset), 0u);
  SkipSourceTrivia(bytes, &offset);
  EXPECT_EQ(ReadUVarint(bytes, &offset), 1u);
  EXPECT_EQ(ReadUVarint(bytes, &offset), 2u);
  SkipValueDef(bytes, &offset);  // Global value.
  SkipValueDef(bytes, &offset);  // Declaration-local dynamic dim.
  EXPECT_EQ(ReadUVarint(bytes, &offset), 1u);
  ReadUVarint(bytes, &offset);  // predicates key id
  EXPECT_EQ(bytes[offset++], 8u);

  loom_module_free(module);
}

TEST_F(WriterTest, ValueNumberingFollowsPhysicalDefinitionOrder) {
  constexpr uint32_t kValueCount = 513;
  std::vector<uint8_t> canonical_bytes;
  for (bool reverse_allocation_order : {false, true}) {
    loom_module_t* module = CreateModule("value_order");
    loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    IREE_ASSERT_OK(loom_module_intern_type(module, i32_type, &i32_type));

    loom_builder_t module_builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(
        &module_builder, IREE_SV("ordered_values"), &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_op_t* func_op = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(
        &module_builder, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
        {/*.module_id=*/0, /*.symbol_id=*/symbol_id}, /*arg_types=*/nullptr,
        /*arg_count=*/0, /*result_types=*/nullptr, /*result_count=*/0,
        /*arg_names=*/nullptr, /*arg_name_count=*/0, /*result_names=*/nullptr,
        /*result_name_count=*/0, LOOM_LOCATION_NONE, &func_op));

    loom_block_t* body = loom_region_entry_block(
        loom_func_like_body(loom_func_like_cast(module, func_op)));
    loom_builder_t body_builder;
    loom_builder_initialize(module, &module->arena, body, &body_builder);
    std::vector<loom_op_t*> constants(kValueCount);
    std::vector<loom_value_id_t> values(kValueCount);
    for (uint32_t i = 0; i < kValueCount; ++i) {
      const uint32_t logical_index =
          reverse_allocation_order ? kValueCount - i - 1 : i;
      IREE_ASSERT_OK(loom_test_constant_build(
          &body_builder, loom_attr_i64(logical_index), i32_type,
          LOOM_LOCATION_NONE, &constants[logical_index]));
      values[logical_index] = loom_op_results(constants[logical_index])[0];
    }
    loom_op_t* yield_op = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&body_builder, values.data(),
                                         values.size(), LOOM_LOCATION_NONE,
                                         &yield_op));

    if (reverse_allocation_order) {
      for (loom_op_t* constant : constants) {
        loom_block_unlink_op(module, constant);
        IREE_ASSERT_OK(
            loom_block_insert_before_op(module, body, yield_op, constant));
      }
    }

    std::vector<uint8_t> bytes = WriteModule(module);
    if (canonical_bytes.empty()) {
      canonical_bytes = std::move(bytes);
    } else {
      EXPECT_EQ(bytes, canonical_bytes);
    }
    loom_module_free(module);
  }
}

TEST_F(WriterTest, GlobalValueClosureRetainsFirstDiscoveryOrder) {
  constexpr uint32_t kLocalValueCount = 300;
  std::vector<uint8_t> canonical_bytes;
  for (bool reverse_allocation_order : {false, true}) {
    loom_module_t* module = CreateModule("global_value_order");
    loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    IREE_ASSERT_OK(loom_module_intern_type(module, index_type, &index_type));

    std::vector<loom_value_id_t> values(kLocalValueCount);
    for (uint32_t i = 0; i < kLocalValueCount; ++i) {
      const uint32_t logical_index =
          reverse_allocation_order ? kLocalValueCount - i - 1 : i;
      IREE_ASSERT_OK(
          loom_module_define_value(module, index_type, &values[logical_index]));
    }
    std::vector<loom_predicate_t> predicates(2 * kLocalValueCount);
    for (iree_host_size_t i = 0; i < predicates.size(); ++i) {
      const uint32_t logical_index = (uint32_t)i % kLocalValueCount;
      predicates[i] = loom_predicate_t{
          /*.kind=*/LOOM_PREDICATE_MUL,
          /*.arg_count=*/2,
          /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
          /*.reserved=*/{},
          /*.args=*/{(int64_t)values[logical_index], 1},
      };
    }

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(
        &builder, IREE_SV("captured_values"), &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    loom_op_t* global_op = nullptr;
    IREE_ASSERT_OK(loom_global_constant_build(
        &builder, LOOM_GLOBAL_CONSTANT_BUILD_FLAG_HAS_PREDICATES,
        {/*.module_id=*/0, /*.symbol_id=*/symbol_id}, index_type,
        predicates.data(), predicates.size(), loom_attr_i64(0),
        LOOM_LOCATION_NONE, &global_op));

    std::vector<uint8_t> bytes = WriteModule(module);
    size_t offset = SectionPayloadOffset(bytes, LOOM_BYTECODE_SECTION_SYMBOLS);
    ASSERT_GT(offset, 0u);
    ASSERT_EQ(ReadUVarint(bytes, &offset), 1u);  // symbol_count
    const uint64_t import_count = ReadUVarint(bytes, &offset);
    const uint64_t export_count = ReadUVarint(bytes, &offset);
    ReadUVarint(bytes, &offset);  // root_region_payload_count
    offset += (import_count + export_count) * sizeof(uint64_t);
    ReadUVarint(bytes, &offset);  // name_id
    ASSERT_EQ(bytes[offset++], LOOM_BYTECODE_SYMBOL_GLOBAL);
    offset += 1;                  // visibility
    offset += sizeof(uint16_t);   // flags
    ReadUVarint(bytes, &offset);  // location
    ReadUVarint(bytes, &offset);  // defining op kind
    SkipSourceTrivia(bytes, &offset);
    ASSERT_EQ(ReadUVarint(bytes, &offset), 1u);  // result_count
    EXPECT_EQ(ReadUVarint(bytes, &offset), kLocalValueCount + 1);

    if (canonical_bytes.empty()) {
      canonical_bytes = std::move(bytes);
    } else {
      EXPECT_EQ(bytes, canonical_bytes);
    }
    loom_module_free(module);
  }
}

TEST_F(WriterTest, ExecutableSymbolFailsLoudly) {
  loom_module_t* module = CreateModule("test");

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("exe"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
  module->symbols.entries[symbol_id].kind = LOOM_SYMBOL_EXECUTABLE;

  ExpectWriteModuleStatus(IREE_STATUS_UNIMPLEMENTED, module);

  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Custom producer string
//===----------------------------------------------------------------------===//

TEST_F(WriterTest, CustomProducer) {
  loom_module_t* module = CreateModule("test");

  loom_bytecode_write_options_t options = {{0}};
  options.producer = IREE_SV("my-tool 2.0");

  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
          IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      4096, iree_allocator_system(), &stream));

  IREE_ASSERT_OK(
      loom_bytecode_write_module(module, stream, &options, &block_pool_));

  // Read back and check producer string.
  iree_io_stream_pos_t length = iree_io_stream_length(stream);
  std::vector<uint8_t> bytes(length);
  IREE_ASSERT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(
      iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));

  std::string producer(reinterpret_cast<const char*>(&bytes[16]));
  EXPECT_EQ(producer, "my-tool 2.0");

  iree_io_stream_release(stream);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
