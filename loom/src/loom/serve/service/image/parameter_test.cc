// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fstream>
#include <iterator>
#include <memory>

#include "iree/base/tooling/flags.h"
#include "loom/model/card.h"
#include "loom/serve/runtime/parameter/weights_test_base.h"
#include "loom/serve/runtime/program/package.h"
#include "loom/serve/service/image/model.h"

IREE_FLAG(string, parameter_image_source, "",
          "Parameter image source fixture.");
IREE_FLAG(string, parameter_image_tokenizer, "", "Parameter image tokenizer.");

namespace loom_serve::testing {

class ImageParameterTest : public WeightsTest {
 protected:
  void SetUpDevice() override {
    ASSERT_NO_FATAL_FAILURE(WeightsTest::SetUpDevice());
    const uint64_t root_bytes =
        loom_serve_device_weight_geometry(loom_serve_device_group_at(group_, 0))
            .minimum_size;
    // Three parameter ranges plus one shared activation/execution workspace.
    IREE_ASSERT_OK(
        AcquireWeightDeviceGroup(true, kSlabBytes, 4 * root_bytes, &group_));
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_image_model_free(model_));
    WeightsTest::TearDown();
  }

  void Create(bool card_backed = false,
              iree_string_view_t source_options = iree_string_view_empty(),
              iree_status_code_t expected_code = IREE_STATUS_OK) {
    const auto checkpoint_directory =
        iree::testing::MakeTempFilePath("loom_image_parameters");
    ASSERT_TRUE(std::filesystem::create_directory(checkpoint_directory));
    WriteCheckpoint(checkpoint_directory + "/first.safetensors", 0);
    WriteCheckpoint(checkpoint_directory + "/second.safetensors", 100);
    ASSERT_TRUE(
        std::filesystem::copy_file(FLAG_parameter_image_tokenizer,
                                   checkpoint_directory + "/tokenizer.json"));
    std::unique_ptr<loom_serve_package_t, decltype(&loom_serve_package_release)>
        package(nullptr, loom_serve_package_release);
    std::unique_ptr<loom_model_card_t, decltype(&loom_model_card_release)> card(
        nullptr, loom_model_card_release);
    if (card_backed) {
      const std::filesystem::path paths[] = {
          FLAG_parameter_image_source, WeightSourcePath(),
          std::filesystem::path(WeightSourcePath()).parent_path() /
              "policy.loom",
          FLAG_parameter_image_tokenizer};
      const iree_string_view_t names[] = {
          IREE_SV("parameters.loom"), IREE_SV("weights.loom"),
          IREE_SV("loom/src/loom/serve/runtime/parameter/testdata/weights/"
                  "policy.loom"),
          IREE_SV("tokenizer.json")};
      std::string contents[IREE_ARRAYSIZE(paths)];
      loom_serve_package_file_t files[IREE_ARRAYSIZE(paths)];
      for (size_t i = 0; i < IREE_ARRAYSIZE(paths); ++i) {
        std::ifstream input(paths[i], std::ios::binary);
        ASSERT_TRUE(input.good()) << paths[i];
        contents[i].assign(std::istreambuf_iterator<char>(input), {});
        files[i] = {names[i], iree_make_const_byte_span(contents[i].data(),
                                                        contents[i].size())};
      }
      loom_serve_package_t* value = nullptr;
      IREE_ASSERT_OK(loom_serve_package_create(IREE_SV("model.json"),
                                               IREE_ARRAYSIZE(files), files,
                                               host_allocator_, &value));
      package.reset(value);
      std::ifstream input(std::filesystem::path(FLAG_parameter_image_source)
                              .replace_extension(".json"));
      ASSERT_TRUE(input.good());
      const std::string json(std::istreambuf_iterator<char>(input), {});
      loom_model_card_t* parsed = nullptr;
      IREE_ASSERT_OK(
          loom_model_card_parse(iree_make_string_view(json.data(), json.size()),
                                nullptr, host_allocator_, &parsed));
      card.reset(parsed);
    }
    const auto source =
        card_backed ? IREE_SV("parameters.loom")
                    : iree_make_cstring_view(FLAG_parameter_image_source);
    const iree_string_view_t providers[] = {
        iree_make_cstring_view(WeightSourcePath()), source};
    const loom_serve_image_model_options_t options = {
        .sources =
            {
                .base_directory = IREE_SV("."),
                .prepare = source,
                .control = source,
                .count = card_backed ? 0 : IREE_ARRAYSIZE(providers),
                .values = card_backed ? nullptr : providers,
                .package = package.get(),
                .card = card.get(),
            },
        .checkpoint_directory = iree_make_string_view(
            checkpoint_directory.data(), checkpoint_directory.size()),
        .options = source_options,
        .height = 1,
        .width = 1,
        .text_tokens = 8,
        .specialization_cache = {4, 16 * 1024 * 1024},
    };
    iree::Status status = loom_serve_image_model_allocate(
        device_owner_, &options, host_allocator_, &model_);
    if (expected_code != IREE_STATUS_OK) {
      EXPECT_THAT(status.ToString(),
                  ::testing::HasSubstr(
                      "image parameter arenas require a binding entry"));
    }
    IREE_ASSERT_STATUS_IS(expected_code, status);
  }

  void CheckImages() {
    for (uint64_t seed : {0u, 7u, 1u}) {
      loom_serve_image_request_t request = {
          .seed = seed,
          .width = 1,
          .height = 1,
      };
      loom_serve_image_prepared_t prepared = {};
      iree_const_byte_span_t rgb = iree_const_byte_span_empty();
      IREE_ASSERT_OK(
          loom_serve_image_model_generate(model_, &request, &prepared, &rgb));
      ASSERT_EQ(rgb.data_length, 3 * sizeof(float));
      const std::array<float, 3> expected = {
          static_cast<float>(14 + seed) / 256,
          static_cast<float>(114 + seed) / 256,
          static_cast<float>(127 + seed) / 256};
      std::array<float, 3> actual;
      memcpy(actual.data(), rgb.data, rgb.data_length);
      EXPECT_EQ(actual, expected);
    }
    const auto activity = loom_serve_image_model_parameters(model_).activity;
    EXPECT_EQ(activity.activations, 3u);
    EXPECT_EQ(activity.prepared_bytes, 3u * 8 * sizeof(int32_t));
  }

  // Real model owner, freed before the fixture's device and backing domain.
  loom_serve_image_model_t* model_ = nullptr;
};

TEST_P(ImageParameterTest, ResolvedOffsetsBindOnceBeforeImageUpload) {
  ASSERT_NO_FATAL_FAILURE(Create());
  ASSERT_NO_FATAL_FAILURE(CheckImages());
}

TEST_P(ImageParameterTest, CardCallbacksKeepExactNames) {
  ASSERT_NO_FATAL_FAILURE(Create(true));
  ASSERT_NO_FATAL_FAILURE(CheckImages());
}

TEST_P(ImageParameterTest, ArenaRequiresSourceBindingDeclaration) {
  ASSERT_NO_FATAL_FAILURE(Create(false, IREE_SV("{\"bind_parameters\":false}"),
                                 IREE_STATUS_INVALID_ARGUMENT));
  EXPECT_EQ(model_, nullptr);
}

INSTANTIATE_TEST_SUITE_P(Elastic, ImageParameterTest, ::testing::Values(true));

}  // namespace loom_serve::testing
