// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/preparation.h"

#include <map>
#include <string>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"

IREE_FLAG(string, model_source, "", "Production Krea bootstrap source.");
IREE_FLAG(string, boundary_source, "", "Preparation boundary callers.");

namespace {

std::string String(iree_string_view_t value) {
  return std::string(value.data ? value.data : "", value.size);
}

class PreparationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_vm_environment_allocate(allocator, &environment));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types));
  }

  void TearDown() override {
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_environment_free(environment);
    loom_serve_preparation_destroy(preparation);
  }

  iree_status_t Prepare(uint32_t height, uint32_t width, uint32_t text_tokens,
                        iree_string_view_t adapter) {
    arguments[0] = iree_vm_variant_from_i64(height);
    arguments[1] = iree_vm_variant_from_i64(width);
    arguments[2] = iree_vm_variant_from_i64(text_tokens);
    const iree_string_view_t paths[] = {IREE_SVL("/checkpoints/model"),
                                        adapter};
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(paths); ++i) {
      iree_vm_buffer_t* buffer = nullptr;
      IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
          IREE_VM_BUFFER_ACCESS_FLAG_READ,
          iree_make_byte_span(const_cast<char*>(paths[i].data), paths[i].size),
          iree_vm_buffer_release_callback_null(), allocator, &buffer));
      arguments[3 + i] = iree_vm_buffer_variant_from_ptr_move(&types, &buffer);
    }
    return loom_serve_preparation_create(
        environment, iree_make_cstring_view(FLAG_model_source),
        IREE_SV("prepare"), iree_vm_variant_span_from_array(arguments),
        &preparation, allocator);
  }

  // Host declarations outlive all temporary VM reference and argument storage.
  const iree_allocator_t allocator = iree_allocator_system();
  // Environment owns the reference type table used while bootstrap executes.
  iree_vm_environment_t* environment = nullptr;
  // Resolved VM buffer type, borrowed from the environment.
  iree_vm_ref_types_t types = {};
  // Arguments own the borrowed-content VM buffer wrappers.
  iree_vm_variant_t arguments[5] = {};
  // Independently owned result of the source bootstrap.
  loom_serve_preparation_t* preparation = nullptr;
};

TEST_F(PreparationTest, KreaDeclaresShapesAndSharedDomains) {
  for (uint32_t maximum : {16u, 80u, 128u, 144u, 192u, 512u}) {
    for (bool adapted : {false, true}) {
      SCOPED_TRACE(maximum);
      SCOPED_TRACE(adapted);
      IREE_ASSERT_OK(Prepare(768, 1024, maximum,
                             adapted ? IREE_SV("/adapters/style.safetensors")
                                     : iree_string_view_empty()));
      const iree_host_size_t count = maximum > 128 && maximum % 64 == 0 ? 2 : 1;
      ASSERT_EQ(loom_serve_preparation_stage_count(preparation), count);
      for (iree_host_size_t i = 0; i < count; ++i) {
        const auto* stage = loom_serve_preparation_stage(preparation, i);
        const uint32_t text = i ? 128 : maximum;
        EXPECT_EQ(stage->tag, text);
        EXPECT_EQ(String(stage->root),
                  adapted ? "sample_image_adapted" : "sample_image");
        std::map<std::string, std::string> config;
        for (iree_host_size_t j = 0; j < stage->config.binding_count; ++j) {
          const auto& item = stage->config.bindings[j];
          config.emplace(std::string(item.key.data, item.key.size),
                         std::string(item.value.data, item.value.size));
        }
        EXPECT_EQ(config.size(), 6);
        EXPECT_EQ(config["krea2.image_tokens"], "3072");
        EXPECT_EQ(config["krea2.block_tokens"], std::to_string(3072 + text));
        EXPECT_EQ(config["krea2.text_tokens"], std::to_string(text));
        EXPECT_EQ(config["krea2.time_count"], "8");
        EXPECT_EQ(config["krea2.latent_height"], "96");
        EXPECT_EQ(config["krea2.latent_width"], "128");
        EXPECT_EQ(stage->config.flags,
                  LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED);
        ASSERT_EQ(stage->parameter_count, adapted ? 4 : 3);
        for (iree_host_size_t j = 0; j < stage->parameter_count; ++j) {
          EXPECT_EQ(stage->parameters[j].binding, j);
          EXPECT_EQ(String(stage->parameters[j].policy), "weights.loom");
        }
        EXPECT_EQ(String(stage->parameters[0].path),
                  "/checkpoints/model/text_encoder/model.safetensors");
        EXPECT_EQ(String(stage->parameters[1].path),
                  "/checkpoints/model/turbo.safetensors");
        if (adapted) {
          EXPECT_EQ(String(stage->parameters[2].path),
                    "/adapters/style.safetensors");
        }
        EXPECT_EQ(String(stage->parameters[stage->parameter_count - 1].path),
                  "/checkpoints/model/vae/diffusion_pytorch_model.safetensors");
      }
      loom_serve_preparation_destroy(preparation);
      preparation = nullptr;
      iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    }
  }
}

TEST_F(PreparationTest, DeclarationsOutliveArgumentsAndEnvironment) {
  std::string path = "/adapters/borrowed.safetensors";
  IREE_ASSERT_OK(
      Prepare(1024, 1024, 512, iree_make_cstring_view(path.c_str())));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_environment_free(environment);
  environment = nullptr;
  path.assign(path.size(), 'x');
  const auto* stage = loom_serve_preparation_stage(preparation, 0);
  EXPECT_EQ(String(stage->root), "sample_image_adapted");
  EXPECT_EQ(String(stage->parameters[2].path),
            "/adapters/borrowed.safetensors");
  EXPECT_EQ(String(stage->parameters[2].policy), "weights.loom");
}

TEST_F(PreparationTest, TextConfigurationCopiesValueSpelling) {
  IREE_ASSERT_OK(loom_serve_preparation_create(
      environment, iree_make_cstring_view(FLAG_boundary_source),
      IREE_SV("text_config"), iree_vm_variant_span_empty(), &preparation,
      allocator));
  const auto* stage = loom_serve_preparation_stage(preparation, 0);
  ASSERT_EQ(stage->config.binding_count, 1);
  const auto& value = stage->config.bindings[0].value;
  EXPECT_EQ(std::string(value.data, value.size), "true");
}

TEST_F(PreparationTest, InvalidDeclarationsReleasePartialOwnership) {
  struct Case {
    // Source entry that reaches a public declaration failure.
    const char* entry;
    // Terminal error from the declaration boundary.
    iree_status_code_t code;
  };
  const Case cases[] = {{"missing_stage", IREE_STATUS_OUT_OF_RANGE},
                        {"duplicate_parameter", IREE_STATUS_ALREADY_EXISTS},
                        {"no_stages", IREE_STATUS_INVALID_ARGUMENT}};
  for (const auto& test : cases) {
    SCOPED_TRACE(test.entry);
    IREE_EXPECT_STATUS_IS(
        test.code,
        loom_serve_preparation_create(
            environment, iree_make_cstring_view(FLAG_boundary_source),
            iree_make_cstring_view(test.entry), iree_vm_variant_span_empty(),
            &preparation, allocator));
    EXPECT_EQ(preparation, nullptr);
  }
}

}  // namespace
