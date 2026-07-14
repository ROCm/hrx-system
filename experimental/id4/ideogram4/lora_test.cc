// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/lora.h"

#include <cstdio>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static id4_ideogram4_dit_model_config_t MakeModelConfig() {
  id4_ideogram4_dit_model_config_t model;
  memset(&model, 0, sizeof(model));
  model.layer_count = 2;
  model.hidden_size = 8;
  model.intermediate_size = 12;
  model.adaln_size = 4;
  return model;
}

class LoraImportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_io_parameter_index_create(iree_allocator_system(),
                                                  &parameter_index_));
  }

  void TearDown() override {
    iree_io_parameter_index_release(parameter_index_);
  }

  iree_status_t AddTensor(iree_string_view_t key, iree_string_view_t dtype,
                          uint32_t rows, uint32_t columns) {
    char metadata_buffer[128];
    int metadata_length =
        std::snprintf(metadata_buffer, sizeof(metadata_buffer),
                      "{\"dtype\":\"%.*s\",\"shape\":[%" PRIu32 ",%" PRIu32
                      "],\"data_offsets\":[0,%" PRIu64 "]}",
                      (int)dtype.size, dtype.data, rows, columns,
                      (uint64_t)rows * columns * 2);
    if (metadata_length < 0 ||
        (iree_host_size_t)metadata_length >= sizeof(metadata_buffer)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "test tensor metadata overflow");
    }
    iree_io_parameter_index_entry_t entry;
    memset(&entry, 0, sizeof(entry));
    entry.key = key;
    entry.metadata =
        iree_make_const_byte_span(metadata_buffer, metadata_length);
    entry.length = (uint64_t)rows * columns * 2;
    entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT;
    entry.storage.splat.pattern_length = 1;
    return iree_io_parameter_index_add(parameter_index_, &entry);
  }

  void AddPair(iree_string_view_t target_prefix, uint32_t rank,
               uint32_t input_size, uint32_t output_size) {
    std::string prefix(target_prefix.data, target_prefix.size);
    std::string down_key = prefix + ".lora_A.weight";
    std::string up_key = prefix + ".lora_B.weight";
    IREE_ASSERT_OK(
        AddTensor(iree_make_string_view(down_key.data(), down_key.size()),
                  IREE_SV("BF16"), rank, input_size));
    IREE_ASSERT_OK(
        AddTensor(iree_make_string_view(up_key.data(), up_key.size()),
                  IREE_SV("BF16"), output_size, rank));
  }

  iree_status_t Import(id4_ideogram4_lora_t** out_lora) {
    id4_ideogram4_lora_import_options_t options;
    memset(&options, 0, sizeof(options));
    options.structure_size = sizeof(options);
    options.model = MakeModelConfig();
    options.parameter_index = parameter_index_;
    options.source_scope = IREE_SV("archer");
    return id4_ideogram4_lora_import(&options, iree_allocator_system(),
                                     out_lora);
  }

  iree_io_parameter_index_t* parameter_index_ = NULL;
};

TEST_F(LoraImportTest, ImportsValidatedTargets) {
  AddPair(IREE_SV("diffusion_model.layers.1.attention.qkv"), 3, 8, 24);
  AddPair(IREE_SV("diffusion_model.layers.0.feed_forward.w2"), 2, 12, 8);

  id4_ideogram4_lora_t* lora = NULL;
  IREE_ASSERT_OK(Import(&lora));
  EXPECT_TRUE(iree_string_view_equal(id4_ideogram4_lora_source_scope(lora),
                                     IREE_SV("archer")));
  EXPECT_EQ(id4_ideogram4_lora_target_count(lora), 2u);

  const id4_ideogram4_lora_target_t* qkv = id4_ideogram4_lora_lookup_target(
      lora, IREE_SV("layers.1.attention.qkv.weight"));
  ASSERT_NE(qkv, nullptr);
  EXPECT_EQ(qkv->input_size, 8u);
  EXPECT_EQ(qkv->output_size, 24u);
  EXPECT_EQ(qkv->rank, 3u);
  EXPECT_TRUE(iree_string_view_equal(
      qkv->down_parameter_key,
      IREE_SV("diffusion_model.layers.1.attention.qkv.lora_A.weight")));
  EXPECT_TRUE(iree_string_view_equal(
      qkv->up_parameter_key,
      IREE_SV("diffusion_model.layers.1.attention.qkv.lora_B.weight")));

  EXPECT_EQ(id4_ideogram4_lora_lookup_target(
                lora, IREE_SV("layers.0.attention.o.weight")),
            nullptr);
  EXPECT_EQ(id4_ideogram4_lora_target_at(lora, 2), nullptr);
  id4_ideogram4_lora_release(lora);
}

TEST_F(LoraImportTest, RejectsUnknownTarget) {
  IREE_ASSERT_OK(
      AddTensor(IREE_SV("diffusion_model.layers.0.attention.k.lora_A.weight"),
                IREE_SV("BF16"), 2, 8));
  id4_ideogram4_lora_t* lora = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Import(&lora));
  EXPECT_EQ(lora, nullptr);
}

TEST_F(LoraImportTest, RejectsDuplicateTargetHalf) {
  IREE_ASSERT_OK(
      AddTensor(IREE_SV("diffusion_model.layers.0.attention.o.lora_A.weight"),
                IREE_SV("BF16"), 2, 8));
  IREE_ASSERT_OK(
      AddTensor(IREE_SV("diffusion_model.layers.0.attention.o.lora_A.weight"),
                IREE_SV("BF16"), 2, 8));
  id4_ideogram4_lora_t* lora = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS, Import(&lora));
  EXPECT_EQ(lora, nullptr);
}

TEST_F(LoraImportTest, RejectsIncompleteTarget) {
  IREE_ASSERT_OK(
      AddTensor(IREE_SV("diffusion_model.layers.0.attention.o.lora_A.weight"),
                IREE_SV("BF16"), 2, 8));
  id4_ideogram4_lora_t* lora = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Import(&lora));
  EXPECT_EQ(lora, nullptr);
}

TEST_F(LoraImportTest, RejectsWrongDtype) {
  IREE_ASSERT_OK(
      AddTensor(IREE_SV("diffusion_model.layers.0.attention.o.lora_A.weight"),
                IREE_SV("F32"), 2, 8));
  IREE_ASSERT_OK(
      AddTensor(IREE_SV("diffusion_model.layers.0.attention.o.lora_B.weight"),
                IREE_SV("BF16"), 8, 2));
  id4_ideogram4_lora_t* lora = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Import(&lora));
  EXPECT_EQ(lora, nullptr);
}

TEST_F(LoraImportTest, RejectsShapeMismatch) {
  AddPair(IREE_SV("diffusion_model.layers.0.feed_forward.w1"), 2, 9, 12);
  id4_ideogram4_lora_t* lora = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Import(&lora));
  EXPECT_EQ(lora, nullptr);
}

}  // namespace
