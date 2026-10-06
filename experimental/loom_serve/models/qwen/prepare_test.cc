// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <map>
#include <string>
#include <vector>

#include "experimental/loom_serve/models/qwen/model.h"
#include "experimental/loom_serve/models/qwen/schedule.h"
#include "experimental/loom_serve/runtime/preparation.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"

IREE_FLAG(string, prepare_source, "", "Production model bootstrap source.");

namespace {

std::string String(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class PrepareTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_vm_environment_allocate(allocator, &environment));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types));
  }

  void TearDown() override {
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    loom_serve_preparation_destroy(preparation);
    iree_vm_environment_free(environment);
  }

  void Prepare(iree_host_size_t rows, iree_host_size_t pool, bool mtp,
               const std::vector<loom_serve_packing_shape_t>& shapes,
               iree_host_size_t context = 16384,
               iree_host_size_t prefill = 512) {
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    loom_serve_preparation_destroy(preparation);
    preparation = nullptr;
    std::vector<uint8_t> storage(shapes.size() * 16);
    for (size_t i = 0; i < shapes.size(); ++i) {
      iree_unaligned_store_le_u64(storage.data() + i * 16,
                                  shapes[i].token_capacity);
      iree_unaligned_store_le_u64(storage.data() + i * 16 + 8,
                                  shapes[i].span_capacity);
    }
    iree_vm_buffer_t* shape_buffer = nullptr;
    IREE_ASSERT_OK(iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(storage.data(), storage.size()), 8, allocator,
        &shape_buffer));
    iree_vm_buffer_t* weights = nullptr;
    const auto weight_path = IREE_SV("/models/target.gguf");
    IREE_ASSERT_OK(iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(weight_path.data, weight_path.size), 1,
        allocator, &weights));
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i64(prefill),
        iree_vm_variant_from_i64(context),
        iree_vm_variant_from_i64(pool),
        iree_vm_variant_from_i64(rows),
        iree_vm_variant_from_i32(mtp),
        iree_vm_buffer_variant_from_ptr_move(&types, &shape_buffer),
        iree_vm_variant_from_i64(shapes.size()),
        iree_vm_buffer_variant_from_ptr_move(&types, &weights),
    };
    iree_status_t status = loom_serve_preparation_create(
        environment, iree_make_cstring_view(FLAG_prepare_source),
        IREE_SV("prepare"), iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_from_array(results), iree_vm_module_span_empty(),
        &preparation, allocator);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    IREE_ASSERT_OK(status);
  }

  void CheckStorage(uint64_t rows, uint64_t context, uint64_t pool, bool packed,
                    bool mtp) {
    // These address equations are the preceding production layout. Source
    // descriptors must preserve both dense and pooled kernel bindings.
    const uint64_t attention = pool ? 0 : context * 65536;
    const uint64_t stride = 156895744 + attention;
    const uint64_t row_bytes = rows * stride;
    const uint64_t table = pool ? 1024 + rows * ((context + 63) / 64) * 4 : 256;
    const uint64_t lengths[] = {
        10485760,
        row_bytes + pool * 65536,
        packed ? 2064u : 0,
        packed ? table : 0,
        packed ? 2048u : 0,
        packed ? 64u : 0,
        mtp ? rows * 20480 : 0,
        mtp ? 1548u : 0,
        mtp ? 384u : 0,
        mtp ? (pool ? pool : rows * context) * 4096 : 0,
        mtp ? table : 0,
    };
    const uint64_t clear[] = {
        10485760, row_bytes, 0, 0, 0, 0, mtp ? rows * 20480 : 0, 0, 0, 0, 0};
    std::vector<std::vector<uint64_t>> expected(5);
    for (size_t i = 0; i < IREE_ARRAYSIZE(lengths); ++i) {
      expected[0].insert(expected[0].end(), {lengths[i], 256, clear[i]});
    }
    expected[2].resize(32);
    expected[3].resize(32);
    for (uint64_t row = 0; row < rows; ++row) {
      const uint64_t base = row * stride;
      expected[1].insert(expected[1].end(),
                         {base, 12, base + 256, 156893184, base + 156893440,
                          attention, base + 156893440 + attention, 2048,
                          base + 156895488 + attention, 32});
      expected[2][row * 2] = base + 256;
      expected[2][row * 2 + 1] = pool ? row_bytes : base + 156893440;
      expected[3][row * 2 + 1] = pool ? 0 : row * context * 4096;
    }
    expected[4] = {64, 1024, 20480};
    for (size_t i = 0; i < expected.size(); ++i) {
      SCOPED_TRACE(i);
      iree_vm_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_borrowed(
          &types, results[4 + i], &buffer));
      ASSERT_NE(buffer, nullptr);
      iree_const_byte_span_t bytes = {};
      IREE_ASSERT_OK(iree_vm_buffer_map_read(
          buffer, 0, iree_vm_buffer_length(buffer), &bytes));
      ASSERT_EQ(bytes.data_length, expected[i].size() * 8);
      for (size_t j = 0; j < expected[i].size(); ++j) {
        SCOPED_TRACE(j);
        EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data + j * 8),
                  expected[i][j]);
      }
    }
  }

  // Host allocator for the real VM and copied declarations.
  const iree_allocator_t allocator = iree_allocator_system();
  // Environment outliving source result references.
  iree_vm_environment_t* environment = nullptr;
  // Canonical VM buffer type borrowed from the environment.
  iree_vm_ref_types_t types = {};
  // Independently owned declarations after bootstrap teardown.
  loom_serve_preparation_t* preparation = nullptr;
  // Returned capacities, control state, terminal marker and storage records.
  iree_vm_variant_t results[9] = {};
};

TEST_F(PrepareTest, DeclaresCompleteProductionCatalogAndOpaqueState) {
  struct Stage {
    // Command export selected by source.
    const char* root;
    // Active token specialization.
    uint64_t tokens;
    // Independent selected-span capacity.
    uint64_t spans;
    // Quantized projection input bound.
    uint64_t capacity;
    // Q8 projection input bound.
    uint64_t q8_capacity;
    // Number of reflected fixed parameter roots.
    uint32_t roots;
  };
  for (const size_t rows : {1u, 3u, 16u}) {
    for (int mode = 0; mode < 9; ++mode) {
      SCOPED_TRACE(rows);
      SCOPED_TRACE(mode);
      const bool mtp = mode == 3 || mode == 4 || mode >= 7;
      const size_t pool =
          mode == 2 || mode == 4 || mode == 6 || mode == 8 ? 2048 : 0;
      std::vector<loom_serve_packing_shape_t> shapes =
          mode == 0
              ? std::vector<loom_serve_packing_shape_t>{}
              : std::vector<loom_serve_packing_shape_t>{{32, 1}, {128, rows}};
      if (mode >= 5) {
        shapes.resize(LOOM_SERVE_QWEN_DEFAULT_SHAPE_CAPACITY);
        shapes.resize(loom_serve_qwen_default_shapes(rows, 512, shapes.data()));
      }
      ASSERT_NO_FATAL_FAILURE(Prepare(rows, pool, mtp, shapes));
      ASSERT_NE(preparation, nullptr);
      CheckStorage(rows, 16384, pool, mode != 0, mtp);
      std::vector<Stage> expected = {
          {"qwen38_prefill", 512, rows, 512, 1, 1},
          {"qwen38_text_decode_greedy", 512, rows, 1, 1, 1},
      };
      for (const auto& shape : shapes) {
        expected.push_back({"qwen38_epoch", shape.token_capacity,
                            shape.span_capacity, 512, 1, 1});
      }
      if (mtp) {
        expected.push_back({"qwen38_mtp_propose", 32, rows, 512, 512, 5});
        for (const auto& shape : shapes) {
          expected.push_back({"qwen38_mtp_warm", shape.token_capacity,
                              shape.span_capacity, 512, 512, 7});
        }
        for (const auto& shape : shapes) {
          expected.push_back({"qwen38_mtp_verify", shape.token_capacity,
                              shape.span_capacity, 512, 512, 1});
        }
      }
      ASSERT_EQ(loom_serve_preparation_stage_count(preparation),
                expected.size());
      for (size_t i = 0; i < expected.size(); ++i) {
        const auto* stage = loom_serve_preparation_stage(preparation, i);
        const auto& value = expected[i];
        EXPECT_EQ(String(stage->root), value.root);
        EXPECT_EQ(stage->tag, value.tokens);
        const std::map<std::string, std::string> expected_config = {
            {"runner.qwen38.prefill_token_count", std::to_string(value.tokens)},
            {"runner.qwen38.span_capacity", std::to_string(value.spans)},
            {"ggml.linear_q4k_q8_1_x4.token_capacity",
             std::to_string(value.capacity)},
            {"ggml.linear_q5k_q8_1_x4.token_capacity",
             std::to_string(value.capacity)},
            {"ggml.linear_q6k_q8_1_x4.token_capacity",
             std::to_string(value.capacity)},
            {"ggml.linear_q8_0_q8_1_x4.token_capacity",
             std::to_string(value.q8_capacity)},
            {"ggml.linear_q8_0_q8_1_x4.output_capacity",
             value.q8_capacity == 1 ? "1024" : "5120"},
            {"ggml.quantize_q8_1_x4.group_capacity",
             std::to_string(136 * value.capacity)},
            {"qwen38.attention.cache_capacity", "16384"},
            {"qwen38.attention.pool_capacity", std::to_string(pool)},
        };
        std::map<std::string, std::string> actual_config;
        ASSERT_EQ(stage->config.binding_count, expected_config.size());
        for (size_t j = 0; j < stage->config.binding_count; ++j) {
          const auto& binding = stage->config.bindings[j];
          actual_config.emplace(
              std::string(binding.key.data, binding.key.size),
              std::string(binding.value.data, binding.value.size));
        }
        EXPECT_EQ(actual_config, expected_config);
        ASSERT_EQ(stage->parameter_count, value.roots);
        for (uint32_t j = 0; j < value.roots; ++j) {
          EXPECT_EQ(stage->parameters[j].binding, j);
          EXPECT_EQ(String(stage->parameters[j].path), "/models/target.gguf");
          EXPECT_EQ(String(stage->parameters[j].policy), "weights.loom");
        }
      }
      int64_t capacity = 0;
      int32_t shared = 0;
      IREE_ASSERT_OK(iree_vm_i64_from_variant(results[0], &capacity));
      IREE_ASSERT_OK(iree_vm_i32_from_variant(results[1], &shared));
      EXPECT_EQ(capacity, (pool || mtp) && mode < 5 ? 128 : 512);
      EXPECT_EQ(shared, shapes.size() + 2);
      for (int i = 2; i < 4; ++i) {
        iree_vm_buffer_t* buffer = nullptr;
        IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_borrowed(
            &types, results[i], &buffer));
        iree_const_byte_span_t bytes = {};
        IREE_ASSERT_OK(iree_vm_buffer_map_read(
            buffer, 0, iree_vm_buffer_length(buffer), &bytes));
        if (i == 2) {
          ASSERT_EQ(bytes.data_length, 16);
          EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data), 0);
          EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data + 4), 1);
          EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data + 8), shapes.size());
          EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data + 12),
                    mtp ? shared : 0);
        } else {
          EXPECT_EQ(std::string((const char*)bytes.data, bytes.data_length),
                    "<|im_end|>");
        }
      }
    }
  }
}

TEST_F(PrepareTest, StoragePreservesPageBoundariesAndWideByteOrigins) {
  for (const uint64_t context : {63u, 64u, 65u, 262144u}) {
    for (const uint64_t pool : {0u, 64u, 4194304u}) {
      SCOPED_TRACE(context);
      SCOPED_TRACE(pool);
      // The maximum logical/dense case has byte origins beyond 32 bits; the
      // tiny shared pool also proves that logical map stride is independent.
      ASSERT_NO_FATAL_FAILURE(Prepare(16, pool, true, {{32, 16}}, context, 32));
      CheckStorage(16, context, pool, true, true);
    }
  }
}

TEST(PrepareModelTest, InvalidPoolAndMissingTokenizerRetireSourceResults) {
  const std::string source = FLAG_prepare_source;
  const std::string directory = source.substr(0, source.find_last_of('/'));
  const loom_serve_packing_shape_t shapes[] = {{32, 1}, {128, 3}};
  const loom_serve_qwen_options_t options = {
      .source_directory = iree_make_cstring_view(directory.c_str()),
      .prefill_capacity = 512,
      .context_capacity = 16384,
      .pool_capacity = 2048,
      .epoch_count = 2,
      .epoch_shapes = shapes,
      .enable_mtp = true,
      .weights_path = IREE_SV("/models/target.gguf"),
      .tokenizer_path = IREE_SV("missing-qwen-tokenizer.json"),
      .row_count = 3,
  };
  loom_serve_qwen_model_t* model = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_serve_qwen_model_create(&options, iree_allocator_system(), &model));
  EXPECT_EQ(model, nullptr);
  auto invalid = options;
  invalid.pool_capacity = 65;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_qwen_model_create(&invalid, iree_allocator_system(), &model));
  EXPECT_EQ(model, nullptr);
}

}  // namespace
