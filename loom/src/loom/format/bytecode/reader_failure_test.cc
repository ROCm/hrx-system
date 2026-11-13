// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <vector>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/function_projection_reader.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/selected_reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/ir/module.h"
#include "loom/ops/test/registry.h"
#include "loom/testing/gen.h"

namespace {

struct ReaderAllocator {
  // Request ordinal to fail, or SIZE_MAX for ordinary allocation.
  size_t fail_at = SIZE_MAX;
  // Allocation requests made by this invocation and its products.
  size_t request_count = 0;
  // Live allocations, including backing blocks retained until pool shutdown.
  size_t live_count = 0;

  static iree_status_t Control(void* user_data,
                               iree_allocator_command_t command,
                               const void* parameters, void** pointer) {
    auto& self = *static_cast<ReaderAllocator*>(user_data);
    const bool was_live = *pointer != nullptr;
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        self.request_count++ == self.fail_at) {
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    const auto system = iree_allocator_system();
    auto status = system.ctl(system.self, command, parameters, pointer);
    if (iree_status_is_ok(status)) {
      if (command == IREE_ALLOCATOR_COMMAND_FREE) {
        self.live_count -= was_live;
      } else if (!was_live && *pointer) {
        ++self.live_count;
      }
    }
    return status;
  }
};

enum class ReadPath { kMetadata, kIndex, kModule, kSelected };

class ReaderFailureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &index_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_gen_t generator;
    loom_test_gen_initialize_seeded(42, &generator);
    auto config = loom_test_gen_module_config_fuzz_preset(0, 1);
    loom_module_t* source = nullptr;
    IREE_ASSERT_OK(
        loom_test_gen_module(&generator, &config, &context_, &pool_, &source));
    ASSERT_NE(source, nullptr);
    iree_io_stream_t* stream = nullptr;
    IREE_ASSERT_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
            IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_ASSERT_OK(loom_bytecode_write_module(source, stream, nullptr, &pool_));
    bytes_.resize(iree_io_stream_length(stream));
    IREE_ASSERT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_ASSERT_OK(
        iree_io_stream_read(stream, bytes_.size(), bytes_.data(), nullptr));
    iree_io_stream_release(stream);
    loom_module_free(source);
    loom_bytecode_read_result_t result = {};
    IREE_ASSERT_OK(loom_bytecode_read_index(
        Bytes(bytes_), IREE_SV("failure.loombc"), &context_, &pool_,
        &index_arena_, nullptr, &result, &metadata_));
    ASSERT_EQ(result.error_count, 0u);
    ASSERT_EQ(metadata_.module_count, 1u);
    ASSERT_GT(metadata_.modules[0].types.count, 0u);
    ASSERT_GT(metadata_.modules[0].region_payload_count, 0u);
    const auto& body = metadata_.modules[0].region_payloads[0];
    ASSERT_GE(body.length, 10u);
    malformed_body_ = bytes_;
    std::fill_n(malformed_body_.begin() + body.absolute_offset, 10, 0x80);
    // Body payloads are deliberately lazy. Build the index over the actual
    // malformed input, preserving the immutable-index contract during reads.
    IREE_ASSERT_OK(loom_bytecode_read_index(
        Bytes(malformed_body_), IREE_SV("failure.loombc"), &context_, &pool_,
        &index_arena_, nullptr, &result, &malformed_metadata_));
    ASSERT_EQ(result.error_count, 0u);
    for (iree_host_size_t i = 0; i < metadata_.modules[0].symbol_count; ++i) {
      ordinals_.push_back(i);
    }
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&index_arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  static iree_const_byte_span_t Bytes(const std::vector<uint8_t>& bytes) {
    return iree_make_const_byte_span(bytes.data(), bytes.size());
  }

  iree_status_t Read(ReadPath path, const std::vector<uint8_t>& bytes,
                     const loom_bytecode_file_metadata_t& input_metadata,
                     iree_allocator_t allocator, iree_arena_block_pool_t* pool,
                     iree_arena_allocator_t* index_arena,
                     loom_diagnostic_sink_t sink,
                     loom_bytecode_read_result_t* result,
                     loom_bytecode_file_metadata_t* output_index,
                     loom_module_t** output_module) {
    const loom_bytecode_index_options_t index_options = {
        .diagnostic_sink = sink,
    };
    const loom_bytecode_read_options_t read_options = {
        .diagnostic_sink = sink,
    };
    switch (path) {
      case ReadPath::kMetadata:
        return loom_bytecode_read_metadata(Bytes(bytes),
                                           IREE_SV("failure.loombc"), &context_,
                                           pool, &index_options, result);
      case ReadPath::kIndex:
        return loom_bytecode_read_index(Bytes(bytes), IREE_SV("failure.loombc"),
                                        &context_, pool, index_arena,
                                        &index_options, result, output_index);
      case ReadPath::kModule:
        return loom_bytecode_read_module(
            Bytes(bytes), IREE_SV("failure.loombc"), &context_, pool,
            &read_options, allocator, result, output_module);
      case ReadPath::kSelected:
        return loom_bytecode_materialize_module_symbols(
            Bytes(bytes), IREE_SV("failure.loombc"), &context_, pool,
            &input_metadata, 0, {ordinals_.size(), ordinals_.data()},
            &read_options, allocator, result, output_module);
    }
    IREE_CHECK_UNREACHABLE("read path");
  }

  // Source fixture and immutable valid/lazy-invalid indexes.
  iree_arena_block_pool_t pool_;
  // Storage retaining both indexes until all public invocations finish.
  iree_arena_allocator_t index_arena_;
  // Finalized dialect registry shared by all readers.
  loom_context_t context_;
  // Serialized representative module, including types, attributes, and bodies.
  std::vector<uint8_t> bytes_;
  // Same module with an invalid lazy root-region summary.
  std::vector<uint8_t> malformed_body_;
  // Valid index over bytes_.
  loom_bytecode_file_metadata_t metadata_ = {};
  // Valid index over malformed_body_; its body is not yet admitted.
  loom_bytecode_file_metadata_t malformed_metadata_ = {};
  // Complete source-order symbol selection for selected materialization.
  std::vector<iree_host_size_t> ordinals_;
};

TEST_F(ReaderFailureTest, AllocationFailureNeverPublishesAProduct) {
  for (auto path : {ReadPath::kMetadata, ReadPath::kIndex, ReadPath::kModule,
                    ReadPath::kSelected}) {
    SCOPED_TRACE(static_cast<int>(path));
    for (bool malformed_body : {false, true}) {
      SCOPED_TRACE(malformed_body);
      const auto& bytes = malformed_body ? malformed_body_ : bytes_;
      const auto& metadata = malformed_body ? malformed_metadata_ : metadata_;
      size_t allocation_count = 0;
      // First measure this invocation's allocations, then fail each request.
      for (size_t attempt = 0; attempt <= allocation_count; ++attempt) {
        SCOPED_TRACE(attempt);
        ReaderAllocator tracking;
        tracking.fail_at = attempt ? attempt - 1 : SIZE_MAX;
        const iree_allocator_t allocator = {&tracking,
                                            ReaderAllocator::Control};
        iree_arena_block_pool_t pool;
        iree_arena_block_pool_initialize(4096, allocator, &pool);
        iree_arena_allocator_t index_arena;
        iree_arena_initialize(&pool, &index_arena);
        loom_bytecode_read_result_t result = {};
        loom_bytecode_file_metadata_t output_index = {};
        loom_module_t* output_module = nullptr;
        auto status =
            Read(path, bytes, metadata, allocator, &pool, &index_arena, {},
                 &result, &output_index, &output_module);
        if (attempt == 0) {
          IREE_EXPECT_OK(status);
          allocation_count = tracking.request_count;
          EXPECT_GT(allocation_count, 0u);
          const bool reads_body =
              path == ReadPath::kModule || path == ReadPath::kSelected;
          EXPECT_EQ(result.error_count, malformed_body && reads_body ? 1u : 0u);
          if (reads_body) {
            EXPECT_EQ(output_module != nullptr, !malformed_body);
          }
        } else {
          IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
          EXPECT_EQ(output_module, nullptr);
          EXPECT_EQ(output_index.module_count, 0u);
          EXPECT_EQ(output_index.modules, nullptr);
        }
        loom_module_free(output_module);
        iree_arena_deinitialize(&index_arena);
        iree_arena_block_pool_deinitialize(&pool);
        EXPECT_EQ(tracking.live_count, 0u);
      }
    }
  }
}

TEST_F(ReaderFailureTest, RejectionAndSinkFailureKeepSeparateOwnership) {
  auto malformed_types = bytes_;
  const auto type_offset = metadata_.modules[0].types.entries[0].entry_offset;
  malformed_types[type_offset] = UINT8_MAX;
  for (auto path : {ReadPath::kMetadata, ReadPath::kIndex, ReadPath::kModule,
                    ReadPath::kSelected}) {
    SCOPED_TRACE(static_cast<int>(path));
    const bool selected = path == ReadPath::kSelected;
    const auto& bytes = selected ? malformed_body_ : malformed_types;
    for (auto code :
         {IREE_STATUS_OK, IREE_STATUS_DATA_LOSS, IREE_STATUS_DEFERRED}) {
      SCOPED_TRACE(code);
      auto expected = code == IREE_STATUS_OK
                          ? iree_ok_status()
                          : iree_make_status(code, "original sink failure");
      loom_diagnostic_sink_t sink = {};
      if (code != IREE_STATUS_OK) {
        sink = {
            [](void* data, const loom_diagnostic_t*) {
              return *static_cast<iree_status_t*>(data);
            },
            &expected,
        };
      }
      iree_arena_allocator_t arena;
      iree_arena_initialize(&pool_, &arena);
      loom_bytecode_read_result_t result = {};
      loom_bytecode_file_metadata_t output_index = {};
      loom_module_t* output_module = nullptr;
      auto status =
          Read(path, bytes, malformed_metadata_, iree_allocator_system(),
               &pool_, &arena, sink, &result, &output_index, &output_module);
      EXPECT_EQ(status, expected);
      iree_status_ignore(status);
      if (code == IREE_STATUS_OK) {
        EXPECT_EQ(result.error_count, 1u);
      }
      EXPECT_EQ(output_module, nullptr);
      EXPECT_EQ(output_index.module_count, 0u);
      EXPECT_EQ(output_index.modules, nullptr);
      loom_module_free(output_module);
      iree_arena_deinitialize(&arena);
    }
  }
}

TEST_F(ReaderFailureTest, PersistentProjectionTransfersResolverFailureOnce) {
  const auto& metadata = metadata_.modules[0];
  uint32_t ordinal = 0;
  while (ordinal < metadata.symbol_count &&
         metadata.symbols[ordinal].region_payload_count == 0) {
    ++ordinal;
  }
  ASSERT_LT(ordinal, metadata.symbol_count);

  for (auto code : {IREE_STATUS_RESOURCE_EXHAUSTED, IREE_STATUS_DATA_LOSS,
                    IREE_STATUS_DEFERRED}) {
    SCOPED_TRACE(code);
    loom_module_t* module = nullptr;
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("projected"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module));
    struct ResolverState {
      // Caller-owned destination for projected symbol identities.
      loom_module_t* module;
      // Admitted metadata retaining source symbol names.
      const loom_bytecode_module_metadata_t* metadata;
      // Failure transferred by the first resolver call only.
      iree_status_t failure;
      // Number of source identities requested through the callback.
      size_t call_count;
    } state = {module, &metadata, iree_make_status(code, "resolver failure"),
               0};
    const auto expected = state.failure;
    const loom_bytecode_function_projection_reader_options_t options = {
        .symbol_resolver =
            [](void* user_data, uint32_t source_ordinal,
               loom_symbol_ref_t* out_reference) -> iree_status_t {
          auto& state = *static_cast<ResolverState*>(user_data);
          ++state.call_count;
          if (!iree_status_is_ok(state.failure)) {
            auto failure = state.failure;
            state.failure = iree_ok_status();
            return failure;
          }
          const auto name_ordinal =
              state.metadata->symbols[source_ordinal].name_string_index;
          loom_string_id_t name = LOOM_STRING_ID_INVALID;
          IREE_RETURN_IF_ERROR(loom_module_intern_string(
              state.module, state.metadata->strings.values[name_ordinal],
              &name));
          loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
          IREE_RETURN_IF_ERROR(
              loom_module_add_symbol(state.module, name, &symbol));
          *out_reference = {0, symbol};
          return iree_ok_status();
        },
        .symbol_resolver_user_data = &state,
    };
    iree_arena_allocator_t arena;
    iree_arena_initialize(&pool_, &arena);
    loom_bytecode_function_projection_reader_t* reader = nullptr;
    IREE_ASSERT_OK(loom_bytecode_function_projection_reader_allocate(
        Bytes(bytes_), IREE_SV("failure.loombc"), &metadata, module, &options,
        &arena, &reader));
    loom_bytecode_function_header_t header = {};
    auto status = loom_bytecode_function_projection_reader_read_header(
        reader, ordinal, &header);
    EXPECT_EQ(status, expected);
    iree_status_ignore(status);
    IREE_EXPECT_OK(loom_bytecode_function_projection_reader_read_header(
        reader, ordinal, &header));
    EXPECT_GT(header.region_payload_count, 0u);
    IREE_EXPECT_OK(loom_bytecode_function_projection_reader_read_header(
        reader, ordinal, &header));
    EXPECT_EQ(state.call_count, 2u);
    loom_bytecode_function_projection_reader_deinitialize(reader);
    iree_arena_deinitialize(&arena);
    loom_module_free(module);
  }
}

}  // namespace
