// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/binding/cli/lora_set.h"

#include <cstdint>
#include <string>

#include "iree/io/file_contents.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

static id4_ideogram4_dit_model_config_t MakeModelConfig() {
  id4_ideogram4_dit_model_config_t model = {};
  model.layer_count = 1;
  model.hidden_size = 8;
  model.intermediate_size = 12;
  model.adaln_size = 4;
  return model;
}

static std::string MakeLoraSafetensorsContents() {
  constexpr uint64_t kDownByteLength = 2 * 8 * sizeof(uint16_t);
  constexpr uint64_t kUpByteLength = 24 * 2 * sizeof(uint16_t);
  const std::string prefix = "diffusion_model.layers.0.attention.qkv.lora_";
  const std::string header =
      std::string("{\"") + prefix +
      "A.weight\":{\"dtype\":\"BF16\",\"shape\":[2,8],"
      "\"data_offsets\":[0," +
      std::to_string(kDownByteLength) + " ]},\"" + prefix +
      "B.weight\":{\"dtype\":\"BF16\",\"shape\":[24,2],"
      "\"data_offsets\":[" +
      std::to_string(kDownByteLength) + "," +
      std::to_string(kDownByteLength + kUpByteLength) + "]}}";

  std::string contents;
  const uint64_t header_length = header.size();
  for (int i = 0; i < 8; ++i) {
    contents.push_back(static_cast<char>((header_length >> (8 * i)) & 0xFF));
  }
  contents += header;
  contents.resize(contents.size() + kDownByteLength + kUpByteLength, '\0');
  return contents;
}

static void WriteLora(iree_string_view_t path) {
  const std::string contents = MakeLoraSafetensorsContents();
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path, iree_make_const_byte_span(contents.data(), contents.size()),
      iree_allocator_system()));
}

static iree_io_parameter_provider_t* CreateEmptyProvider(
    iree_string_view_t scope) {
  iree_io_parameter_index_t* index = nullptr;
  IREE_CHECK_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));
  iree_io_parameter_provider_t* provider = nullptr;
  IREE_CHECK_OK(iree_io_parameter_index_provider_create(
      scope, index,
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), &provider));
  iree_io_parameter_index_release(index);
  return provider;
}

TEST(LoraSetTest, ComposesOrderedAdaptersAndProviders) {
  iree::testing::TempFilePath lora_path("id4_cli_lora", ".safetensors");
  WriteLora(lora_path.path_view());
  const iree_string_view_t paths[] = {lora_path.path_view(),
                                      lora_path.path_view()};
  const iree_string_view_t strengths[] = {IREE_SV("0.6"), IREE_SV("-0.25")};
  const id4_ideogram4_dit_model_config_t model = MakeModelConfig();
  id4_cli_lora_set_t* lora_set = nullptr;
  IREE_ASSERT_OK(id4_cli_lora_set_create(
      &model, {/*.count=*/IREE_ARRAYSIZE(paths), /*.values=*/paths},
      {/*.count=*/IREE_ARRAYSIZE(strengths), /*.values=*/strengths},
      iree_allocator_system(), &lora_set));

  ASSERT_NE(lora_set, nullptr);
  EXPECT_EQ(id4_cli_lora_set_adapter_count(lora_set), 2u);
  ASSERT_NE(id4_cli_lora_set_strengths(lora_set), nullptr);
  EXPECT_FLOAT_EQ(id4_cli_lora_set_strengths(lora_set)[0], 0.6f);
  EXPECT_FLOAT_EQ(id4_cli_lora_set_strengths(lora_set)[1], -0.25f);
  const id4_ideogram4_lora_topology_t* topology =
      id4_cli_lora_set_topology(lora_set);
  ASSERT_NE(topology, nullptr);
  EXPECT_EQ(id4_ideogram4_lora_topology_adapter_count(topology), 2u);
  EXPECT_EQ(id4_ideogram4_lora_topology_target_count(topology), 1u);
  EXPECT_TRUE(iree_string_view_equal(
      id4_ideogram4_lora_topology_adapter_source_scope(topology, 0),
      IREE_SV("lora_0")));
  EXPECT_TRUE(iree_string_view_equal(
      id4_ideogram4_lora_topology_adapter_source_scope(topology, 1),
      IREE_SV("lora_1")));

  iree_io_parameter_provider_t* base_provider =
      CreateEmptyProvider(IREE_SV("dit_cond_fp8"));
  iree_io_parameter_provider_t* composed_provider = nullptr;
  IREE_ASSERT_OK(id4_cli_lora_set_create_conditioned_provider(
      lora_set, IREE_SV("dit_cond_fp8"), base_provider, iree_allocator_system(),
      &composed_provider));
  EXPECT_TRUE(iree_io_parameter_provider_query_support(
      composed_provider, IREE_SV("dit_cond_fp8")));
  EXPECT_TRUE(iree_io_parameter_provider_query_support(composed_provider,
                                                       IREE_SV("lora_0")));
  EXPECT_TRUE(iree_io_parameter_provider_query_support(composed_provider,
                                                       IREE_SV("lora_1")));
  iree_io_parameter_provider_release(composed_provider);
  iree_io_parameter_provider_release(base_provider);
  id4_cli_lora_set_release(lora_set);
}

TEST(LoraSetTest, DefaultsStrengthsToOne) {
  iree::testing::TempFilePath lora_path("id4_cli_lora_default", ".safetensors");
  WriteLora(lora_path.path_view());
  const iree_string_view_t path = lora_path.path_view();
  const id4_ideogram4_dit_model_config_t model = MakeModelConfig();
  id4_cli_lora_set_t* lora_set = nullptr;
  IREE_ASSERT_OK(id4_cli_lora_set_create(
      &model, {/*.count=*/1, /*.values=*/&path}, iree_string_view_list_empty(),
      iree_allocator_system(), &lora_set));
  ASSERT_NE(lora_set, nullptr);
  EXPECT_FLOAT_EQ(id4_cli_lora_set_strengths(lora_set)[0], 1.0f);
  id4_cli_lora_set_release(lora_set);
}

TEST(LoraSetTest, ValidatesStrengthCardinalityAndValuesBeforeOpeningFiles) {
  const iree_string_view_t path = IREE_SV("does-not-exist.safetensors");
  const iree_string_view_t two_strengths[] = {IREE_SV("0.5"), IREE_SV("0.6")};
  const iree_string_view_t nonfinite_strength = IREE_SV("nan");
  const iree_string_view_t malformed_strength = IREE_SV("0.5oops");
  const id4_ideogram4_dit_model_config_t model = MakeModelConfig();
  id4_cli_lora_set_t* lora_set = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_cli_lora_set_create(&model, {/*.count=*/1, /*.values=*/&path},
                              {/*.count=*/IREE_ARRAYSIZE(two_strengths),
                               /*.values=*/two_strengths},
                              iree_allocator_system(), &lora_set));
  EXPECT_EQ(lora_set, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_cli_lora_set_create(&model, {/*.count=*/1, /*.values=*/&path},
                              {/*.count=*/1, /*.values=*/&nonfinite_strength},
                              iree_allocator_system(), &lora_set));
  EXPECT_EQ(lora_set, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_cli_lora_set_create(&model, {/*.count=*/1, /*.values=*/&path},
                              {/*.count=*/1, /*.values=*/&malformed_strength},
                              iree_allocator_system(), &lora_set));
  EXPECT_EQ(lora_set, nullptr);
}

}  // namespace
