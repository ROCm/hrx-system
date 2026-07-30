// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/parameters.h"

#include <stdint.h>

#include <string>
#include <unordered_set>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct LayerParameterSpec {
  const char* suffix;
  uint64_t q4_length;
  uint64_t q6_length;
};

constexpr LayerParameterSpec kLayerParameterSpecs[] = {
    {"attn_norm.weight", 8192, 0},
    {"attn_q.weight", 4718592, 0},
    {"attn_k.weight", 589824, 0},
    {"attn_v.weight", 589824, 860160},
    {"attn_q_norm.weight", 512, 0},
    {"attn_k_norm.weight", 512, 0},
    {"attn_output.weight", 4718592, 0},
    {"ffn_norm.weight", 8192, 0},
    {"ffn_gate_inp.weight", 1048576, 0},
    {"ffn_gate_exps.weight", 113246208, 0},
    {"ffn_up_exps.weight", 113246208, 0},
    {"ffn_down_exps.weight", 113246208, 165150720},
};

class ParameterIndex {
 public:
  ParameterIndex() {
    IREE_CHECK_OK(
        iree_io_parameter_index_create(iree_allocator_system(), &index_));
    IREE_CHECK_OK(
        iree_io_parameter_index_reserve(index_, QWEN_PARAMETER_COUNT + 1));
  }

  ~ParameterIndex() { iree_io_parameter_index_release(index_); }

  iree_io_parameter_index_t* get() const { return index_; }

  void Add(const std::string& key, uint64_t length) {
    iree_io_parameter_index_entry_t entry = {};
    entry.key = iree_make_string_view(key.data(), key.size());
    entry.length = length;
    entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT;
    entry.storage.splat.pattern_length = 1;
    entry.storage.splat.pattern[0] = 0;
    IREE_CHECK_OK(iree_io_parameter_index_add(index_, &entry));
  }

 private:
  iree_io_parameter_index_t* index_ = nullptr;
};

struct SchemaMutation {
  std::string omitted_key;
  std::string wrong_length_key;
  uint64_t wrong_length = 0;
  bool mismatch_layer_zero_pair = false;
  bool add_extra_entry = false;
};

static std::string LayerKey(iree_host_size_t layer, const char* suffix) {
  return "blk." + std::to_string(layer) + "." + suffix;
}

static uint64_t MutateLength(const SchemaMutation& mutation,
                             const std::string& key, uint64_t length) {
  return key == mutation.wrong_length_key ? mutation.wrong_length : length;
}

static void AddEntryUnlessOmitted(ParameterIndex& index,
                                  const SchemaMutation& mutation,
                                  const std::string& key, uint64_t length) {
  if (key != mutation.omitted_key) {
    index.Add(key, MutateLength(mutation, key, length));
  }
}

static void PopulateFixedSchema(ParameterIndex& index,
                                const SchemaMutation& mutation = {}) {
  // Insert in GGUF file order to prove packing is independent of index order.
  AddEntryUnlessOmitted(index, mutation, "output.weight", 255252480);
  AddEntryUnlessOmitted(index, mutation, "output_norm.weight", 8192);
  AddEntryUnlessOmitted(index, mutation, "token_embd.weight", 175030272);

  for (iree_host_size_t layer = 0; layer < QWEN_MODEL_LAYER_COUNT; ++layer) {
    const bool use_q6 = layer % 2 == 0;
    for (const LayerParameterSpec& spec : kLayerParameterSpecs) {
      const std::string key = LayerKey(layer, spec.suffix);
      uint64_t length =
          use_q6 && spec.q6_length != 0 ? spec.q6_length : spec.q4_length;
      if (mutation.mismatch_layer_zero_pair && layer == 0 &&
          std::string(spec.suffix) == "ffn_down_exps.weight") {
        length = spec.q4_length;
      }
      AddEntryUnlessOmitted(index, mutation, key, length);
    }
  }

  if (!mutation.omitted_key.empty() || mutation.add_extra_entry) {
    index.Add("unexpected.weight", 512);
  }
}

static std::vector<std::string> ExpectedPackingOrder() {
  std::vector<std::string> keys;
  keys.reserve(QWEN_PARAMETER_COUNT);
  keys.push_back("token_embd.weight");
  for (iree_host_size_t layer = 0; layer < QWEN_MODEL_LAYER_COUNT; ++layer) {
    for (const LayerParameterSpec& spec : kLayerParameterSpecs) {
      keys.push_back(LayerKey(layer, spec.suffix));
    }
  }
  keys.push_back("output_norm.weight");
  keys.push_back("output.weight");
  return keys;
}

TEST(QwenParameterLayoutTest, ValidatesAndPacksFixedSchema) {
  ParameterIndex index;
  PopulateFixedSchema(index);

  qwen_parameter_layout_t layout;
  IREE_ASSERT_OK(qwen_parameter_layout_build(index.get(), &layout));

  EXPECT_EQ(layout.statistics.encoded_parameter_bytes, 18550716416ull);
  EXPECT_EQ(layout.statistics.parameter_padding_bytes, 0);
  EXPECT_EQ(layout.statistics.immutable_auxiliary_bytes, 64u * sizeof(float));
  EXPECT_EQ(layout.statistics.allocation_bytes, 18550716672ull);
  EXPECT_GT(layout.statistics.allocation_bytes, UINT32_MAX);

  EXPECT_EQ(layout.token_embedding.offset, 0);
  EXPECT_EQ(layout.token_embedding.length, 175030272);
  EXPECT_EQ(layout.layers[0].value_and_down_storage,
            QWEN_QUANTIZED_STORAGE_Q6_K);
  EXPECT_EQ(layout.layers[1].value_and_down_storage,
            QWEN_QUANTIZED_STORAGE_Q4_K);
  EXPECT_EQ(layout.output.offset + layout.output.length,
            layout.rope_inverse_frequencies.offset);

  const std::vector<std::string> expected_keys = ExpectedPackingOrder();
  ASSERT_EQ(expected_keys.size(), QWEN_PARAMETER_COUNT);
  std::unordered_set<std::string> observed_keys;
  iree_device_size_t previous_end = 0;
  iree_device_size_t enumerated_bytes = 0;
  for (iree_host_size_t i = 0; i < QWEN_PARAMETER_COUNT; ++i) {
    char key_storage[QWEN_PARAMETER_KEY_CAPACITY];
    iree_string_view_t key = iree_string_view_empty();
    iree_io_parameter_span_t span;
    IREE_ASSERT_OK(
        qwen_parameter_layout_enumerate(&layout, i, key_storage, &key, &span));
    const std::string key_string(key.data, key.size);
    EXPECT_EQ(key_string, expected_keys[i]);
    EXPECT_TRUE(observed_keys.insert(key_string).second);
    EXPECT_EQ(span.parameter_offset, 0);
    EXPECT_EQ(span.buffer_offset % 256, 0);
    EXPECT_GE(span.buffer_offset, previous_end);
    previous_end = span.buffer_offset + span.length;
    enumerated_bytes += span.length;
  }
  EXPECT_EQ(observed_keys.size(), QWEN_PARAMETER_COUNT);
  EXPECT_EQ(enumerated_bytes, layout.statistics.encoded_parameter_bytes);
  EXPECT_EQ(previous_end, layout.output.offset + layout.output.length);
}

TEST(QwenParameterLayoutTest, RejectsMissingEntryAtExactCount) {
  ParameterIndex index;
  PopulateFixedSchema(index, {.omitted_key = "blk.17.ffn_gate_exps.weight"});

  qwen_parameter_layout_t layout;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        qwen_parameter_layout_build(index.get(), &layout));
}

TEST(QwenParameterLayoutTest, RejectsUnexpectedEntryCount) {
  ParameterIndex index;
  PopulateFixedSchema(index, {.add_extra_entry = true});

  qwen_parameter_layout_t layout;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        qwen_parameter_layout_build(index.get(), &layout));
}

TEST(QwenParameterLayoutTest, RejectsWrongFixedLength) {
  ParameterIndex index;
  PopulateFixedSchema(index, {
                                 .wrong_length_key = "blk.9.attn_q.weight",
                                 .wrong_length = 4718591,
                             });

  qwen_parameter_layout_t layout;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        qwen_parameter_layout_build(index.get(), &layout));
}

TEST(QwenParameterLayoutTest, RejectsWrongQuantizedLength) {
  ParameterIndex index;
  PopulateFixedSchema(index, {
                                 .wrong_length_key = "blk.9.attn_v.weight",
                                 .wrong_length = 860159,
                             });

  qwen_parameter_layout_t layout;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        qwen_parameter_layout_build(index.get(), &layout));
}

TEST(QwenParameterLayoutTest, RejectsMismatchedValueAndDownStorage) {
  ParameterIndex index;
  PopulateFixedSchema(index, {.mismatch_layer_zero_pair = true});

  qwen_parameter_layout_t layout;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        qwen_parameter_layout_build(index.get(), &layout));
}

TEST(QwenParameterLayoutTest, EnumerateRejectsOutOfRangeOrdinal) {
  ParameterIndex index;
  PopulateFixedSchema(index);
  qwen_parameter_layout_t layout;
  IREE_ASSERT_OK(qwen_parameter_layout_build(index.get(), &layout));

  char key_storage[QWEN_PARAMETER_KEY_CAPACITY];
  iree_string_view_t key = iree_string_view_empty();
  iree_io_parameter_span_t span;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      qwen_parameter_layout_enumerate(&layout, QWEN_PARAMETER_COUNT,
                                      key_storage, &key, &span));
}

TEST(QwenParameterLayoutTest, ComputesImmutableRopeFrequencies) {
  float values[QWEN_MODEL_ROPE_FREQUENCY_COUNT];
  qwen_parameter_calculate_rope_inverse_frequencies(values);

  EXPECT_FLOAT_EQ(values[0], 1.0f);
  EXPECT_NEAR(values[32], 0.001f, 1e-9f);
  for (iree_host_size_t i = 1; i < QWEN_MODEL_ROPE_FREQUENCY_COUNT; ++i) {
    EXPECT_GT(values[i - 1], values[i]);
  }
}

}  // namespace
