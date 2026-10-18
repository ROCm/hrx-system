// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/registry.h"

#include <filesystem>
#include <string>

#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/service/recognition/request.h"
#include "loom/serve/service/service.h"
#include "loom/serve/service/speech/request.h"

namespace {

TEST(RegistryTimeTest, BorrowedServicesShareRegistrationTimestamp) {
  const loom_serve_service_t services[] = {{.name = IREE_SV("first")},
                                           {.name = IREE_SV("second")}};
  loom_serve_registry_t* registry = nullptr;
  IREE_ASSERT_OK(
      loom_serve_registry_allocate(IREE_ARRAYSIZE(services), services, 0, {},
                                   iree_allocator_system(), &registry));
  const int64_t registered =
      loom_serve_registry_at(registry, 0)->registered_unix_seconds;
  for (size_t i = 0; i < IREE_ARRAYSIZE(services); ++i) {
    const auto* model = loom_serve_registry_at(registry, i);
    EXPECT_EQ(model->registered_unix_seconds, registered);
  }
  IREE_EXPECT_OK(loom_serve_registry_free(registry));
}

class RegistryTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    ASSERT_TRUE(path_);
    const loom_serve_registration_factory_t factory = {
        .kinds = 1u << LOOM_SERVE_MODEL_KIND_EMBEDDING,
        .device_count = 2,
        .create =
            [](void* self,
               const loom_serve_model_configuration_t* configuration,
               loom_serve_wake_callback_t observer,
               loom_serve_telemetry_stream_t* stream,
               iree_allocator_t host_allocator,
               loom_serve_registration_t* out_registration) {
              ADD_FAILURE() << "Drained metadata must not prepare a model";
              return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                      "unexpected model preparation");
            },
    };
    IREE_ASSERT_OK(loom_serve_registry_allocate(
        0, nullptr, 1, factory, iree_allocator_system(), &registry_));
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_registry_free(registry_));
    loom_serve_model_configuration_deinitialize(&configuration_);
    if (path_.Exists()) {
      EXPECT_TRUE(path_.Remove());
    }
  }

  static std::string String(iree_string_view_t value) {
    return std::string(value.data, value.size);
  }

  // Real model document in the runner-owned temporary directory.
  iree::testing::TempFilePath path_{"model", ".json"};
  // Loaded declaration until ownership transfers to a registered slot.
  loom_serve_model_configuration_t configuration_ = {};
  // Registry owning moved metadata; no model resources are activated.
  loom_serve_registry_t* registry_ = nullptr;
};

TEST_P(RegistryTest, FileRegistrationOwnsAuthoredPathsAndSelectedBase) {
  const bool explicit_base = GetParam();
  const std::string expected_base =
      explicit_base
          ? "//files/models/vectors"
          : std::filesystem::path(path_.path()).parent_path().string();
  const std::string document =
      R"({"name":"vectors","admission":"drained","device":1,)" +
      std::string(explicit_base
                      ? R"("base_directory":"//files/models/vectors",)"
                      : "") +
      R"("definition":{"kind":"embedding","assets":"weights",)"
      R"("prepare":"boot.loombc","control":"run.loom",)"
      R"("sources":["library/config.loom","library/kernel.loombc"],)"
      R"("options":{"batch_units":64}}})";
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path_.path_view(),
      iree_make_const_byte_span(document.data(), document.size()),
      iree_allocator_system()));
  IREE_ASSERT_OK(loom_serve_model_configuration_load(
      path_.path_view(), iree_allocator_system(), &configuration_));
  ASSERT_TRUE(path_.Remove());

  iree_host_size_t index = IREE_HOST_SIZE_MAX;
  IREE_ASSERT_OK(
      loom_serve_registry_register(registry_, &configuration_, &index));
  EXPECT_EQ(index, 0u);
  EXPECT_EQ(loom_serve_registry_count(registry_), 1u);
  EXPECT_EQ(configuration_.name.data, nullptr);
  EXPECT_EQ(configuration_.sources.values, nullptr);
  const auto* model = loom_serve_registry_at(registry_, index);
  const int64_t registered = model->registered_unix_seconds;
  EXPECT_EQ(model->phase, LOOM_SERVE_REGISTRATION_DECLARED);
  EXPECT_TRUE(model->closed);
  EXPECT_EQ(String(model->service.name), "vectors");
  EXPECT_STREQ(model->service.endpoint, "/v1/embeddings");
  EXPECT_EQ(model->configuration.device_index, 1u);
  const auto& sources = model->configuration.sources;
  EXPECT_EQ(String(sources.base_directory), expected_base);
  EXPECT_EQ(String(sources.prepare), "boot.loombc");
  EXPECT_EQ(String(sources.control), "run.loom");
  ASSERT_EQ(sources.count, 2u);
  EXPECT_EQ(String(sources.values[0]), "library/config.loom");
  EXPECT_EQ(String(sources.values[1]), "library/kernel.loombc");
  EXPECT_EQ(String(model->configuration.assets), "weights");
  EXPECT_EQ(String(model->configuration.options), R"({"batch_units":64})");

  bool progress = true;
  IREE_ASSERT_OK(loom_serve_registry_advance(registry_, index, {}, &progress));
  EXPECT_FALSE(progress);
  EXPECT_EQ(model->registered_unix_seconds, registered);
  IREE_ASSERT_OK(loom_serve_registry_remove(
      loom_serve_registry_at(registry_, index), LOOM_SERVE_REMOVAL_DRAIN));
  IREE_ASSERT_OK(loom_serve_registry_advance(registry_, index, {}, &progress));
  EXPECT_TRUE(progress);
  EXPECT_EQ(loom_serve_registry_count(registry_), 0u);
  EXPECT_EQ(loom_serve_registry_at(registry_, index)->service.name.size, 0u);
}

INSTANTIATE_TEST_SUITE_P(DirectoryBase, RegistryTest, ::testing::Bool());

TEST(RegistryAdmissionTest, NativeSpeechRejectionsPreserveOwnedInput) {
  const loom_serve_service_t services[] = {
      {.name = IREE_SV("vectors"), .kind = LOOM_SERVE_MODEL_KIND_EMBEDDING},
      {.name = IREE_SV("http-speech"), .kind = LOOM_SERVE_MODEL_KIND_SPEECH},
  };
  loom_serve_registry_t* registry = nullptr;
  IREE_ASSERT_OK(
      loom_serve_registry_allocate(IREE_ARRAYSIZE(services), services, 0, {},
                                   iree_allocator_system(), &registry));
  loom_serve_speech_request_t request;
  IREE_ASSERT_OK(loom_serve_speech_request_initialize_text(
      IREE_SV("Keep this request."), IREE_SV("voice"), IREE_SV("Quietly."), 73,
      1.0f, iree_allocator_system(), &request));
  const char* input = iree_string_builder_buffer(&request.input);
  const char* voice = iree_string_builder_buffer(&request.voice);
  const char* instructions = iree_string_builder_buffer(&request.instructions);
  const struct {
    // Name resolved through the actual registry's borrowed service metadata.
    const char* name;
    // Failure at the lookup, modality, or native-capability boundary.
    iree_status_code_t code;
  } cases[] = {{"missing", IREE_STATUS_NOT_FOUND},
               {"vectors", IREE_STATUS_INVALID_ARGUMENT},
               {"http-speech", IREE_STATUS_UNIMPLEMENTED}};
  for (const auto& test : cases) {
    SCOPED_TRACE(test.name);
    loom_serve_speech_stream_t* stream = nullptr;
    IREE_EXPECT_STATUS_IS(
        test.code,
        loom_serve_registry_try_open_speech(
            registry, iree_make_cstring_view(test.name), &request, &stream));
    EXPECT_EQ(stream, nullptr);
    EXPECT_EQ(iree_string_builder_buffer(&request.input), input);
    EXPECT_EQ(iree_string_builder_buffer(&request.voice), voice);
    EXPECT_EQ(iree_string_builder_buffer(&request.instructions), instructions);
    EXPECT_STREQ(input, "Keep this request.");
    EXPECT_STREQ(voice, "voice");
    EXPECT_STREQ(instructions, "Quietly.");
    EXPECT_EQ(request.seed, 73u);
    EXPECT_EQ(request.speed, 1.0f);
  }
  loom_serve_speech_request_deinitialize(&request);
  IREE_EXPECT_OK(loom_serve_registry_free(registry));
}

TEST(RegistryAdmissionTest, NativeRecognitionUsesTypedAdmissionPolicy) {
  const loom_serve_service_t services[] = {
      {.name = IREE_SV("vectors"), .kind = LOOM_SERVE_MODEL_KIND_EMBEDDING},
      {
          .name = IREE_SV("http-recognition"),
          .kind = LOOM_SERVE_MODEL_KIND_RECOGNITION,
      },
      {.name = IREE_SV("activity"), .kind = LOOM_SERVE_MODEL_KIND_ACTIVITY},
  };
  loom_serve_registry_t* registry = nullptr;
  IREE_ASSERT_OK(
      loom_serve_registry_allocate(IREE_ARRAYSIZE(services), services, 0, {},
                                   iree_allocator_system(), &registry));
  const loom_serve_recognition_request_t request = {
      .language = IREE_SV("en"),
      .prompt = IREE_SV("Keep this context."),
      .outputs = LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT,
  };
  const struct {
    // Name resolved through the actual registry's borrowed service metadata.
    const char* name;
    // Failure at the lookup, modality, or native-capability boundary.
    iree_status_code_t code;
  } cases[] = {{"missing", IREE_STATUS_NOT_FOUND},
               {"vectors", IREE_STATUS_INVALID_ARGUMENT},
               {"http-recognition", IREE_STATUS_UNIMPLEMENTED},
               {"activity", IREE_STATUS_UNIMPLEMENTED}};
  for (const auto& test : cases) {
    SCOPED_TRACE(test.name);
    loom_serve_recognition_stream_t* stream = nullptr;
    IREE_EXPECT_STATUS_IS(
        test.code,
        loom_serve_registry_try_open_recognition(
            registry, iree_make_cstring_view(test.name), &request, &stream));
    EXPECT_EQ(stream, nullptr);
  }
  IREE_EXPECT_OK(loom_serve_registry_free(registry));
}

}  // namespace
