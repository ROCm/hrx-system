// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/registry.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <string>

#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/runtime/telemetry/stream.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/service/recognition/request.h"
#include "loom/serve/service/service.h"
#include "loom/serve/service/speech/request.h"
#include "loom/service/telemetry/record.h"

namespace {

const iree_string_view_t kDeviceNames[] = {IREE_SV("first"), IREE_SV("second")};
const loom_serve_pool_binding_t kPools[] = {{IREE_SV("pool"), 3}};

TEST(RegistryTimeTest, BorrowedServicesShareRegistrationTimestamp) {
  const loom_serve_service_t services[] = {{.name = IREE_SV("first")},
                                           {.name = IREE_SV("second")}};
  loom_serve_registry_t* registry = nullptr;
  IREE_ASSERT_OK(
      loom_serve_registry_allocate(IREE_ARRAYSIZE(services), services, 0, {},
                                   iree_allocator_system(), &registry));
  const int64_t registered =
      loom_serve_registry_at(registry, 0)->registered_unix_seconds;
  EXPECT_EQ(loom_serve_registry_counters_since_ns(registry), 0);
  for (size_t i = 0; i < IREE_ARRAYSIZE(services); ++i) {
    const auto* model = loom_serve_registry_at(registry, i);
    EXPECT_EQ(model->registered_unix_seconds, registered);
  }
  IREE_EXPECT_OK(loom_serve_registry_free(registry));
}

TEST(RegistryTimeTest, CounterEpochIncludesBorrowedHistory) {
  loom_runtime_telemetry_stream_t* streams[2] = {};
  for (auto& stream : streams) {
    IREE_ASSERT_OK(loom_runtime_telemetry_stream_allocate(
        4, iree_allocator_system(), &stream));
  }
  // Reverse order ensures the aggregate does not simply select the first slot.
  const loom_service_telemetry_counters_t counters[2] = {};
  const loom_serve_service_t services[] = {
      {
          .name = IREE_SV("newer"),
          .telemetry = {.stream = streams[1], .counters = &counters[1]},
      },
      {
          .name = IREE_SV("older"),
          .telemetry = {.stream = streams[0], .counters = &counters[0]},
      },
  };
  loom_serve_registry_t* registry = nullptr;
  IREE_ASSERT_OK(
      loom_serve_registry_allocate(IREE_ARRAYSIZE(services), services, 0, {},
                                   iree_allocator_system(), &registry));
  EXPECT_EQ(loom_serve_registry_counters_since_ns(registry),
            loom_runtime_telemetry_stream_identity(streams[0]).created_ns);
  IREE_EXPECT_OK(loom_serve_registry_free(registry));
  for (auto* stream : streams) {
    loom_runtime_telemetry_stream_free(stream);
  }
}

TEST(RegistryTimeTest, EmptyRegistryHasAnEpochBeforeAnyRegistration) {
  const auto before = iree_time_now();
  loom_serve_registry_t* registry = nullptr;
  IREE_ASSERT_OK(loom_serve_registry_allocate(
      0, nullptr, 2, {}, iree_allocator_system(), &registry));
  const auto since = loom_serve_registry_counters_since_ns(registry);
  EXPECT_GE(since, before);
  EXPECT_LE(since, iree_time_now());
  IREE_EXPECT_OK(loom_serve_registry_free(registry));
}

class RegistryTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    ASSERT_TRUE(path_);
    const loom_serve_registration_factory_t factory = {
        .kinds = 1u << LOOM_SERVE_MODEL_KIND_EMBEDDING,
        .device_names = {2, kDeviceNames},
        .pool_count = 1,
        .pools = kPools,
        .create =
            [](void* self,
               const loom_serve_model_configuration_t* configuration,
               loom_runtime_wake_callback_t observer,
               loom_runtime_telemetry_stream_t* stream,
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
      R"({"name":"vectors","admission":"drained","device":"second",)" +
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
      path_.path_view(), iree_string_view_empty(), iree_allocator_system(),
      &configuration_));
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
  EXPECT_EQ(model->configuration.pool_index, 0u);
  EXPECT_EQ(String(model->configuration.pool), "pool");
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

class RegistryAllocationTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  struct AllocationFault {
    // Allocation attempt rejected after admission fault injection is enabled.
    size_t fail_at = SIZE_MAX;
    // Allocation attempts across both independently owned allocators.
    size_t attempts = 0;
  };

  struct AllocatorState {
    // Shared fault selection across declaration and registry ownership.
    AllocationFault* fault;
    // Live allocations owned by this allocator, including setup metadata.
    size_t live = 0;

    iree_allocator_t allocator() {
      return {this, [](void* self, iree_allocator_command_t command,
                       const void* parameters, void** pointer) {
                auto& state = *static_cast<AllocatorState*>(self);
                const bool had_storage = *pointer != nullptr;
                if (command != IREE_ALLOCATOR_COMMAND_FREE &&
                    state.fault->attempts++ == state.fault->fail_at) {
                  return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
                }
                const auto system = iree_allocator_system();
                iree_status_t status =
                    system.ctl(system.self, command, parameters, pointer);
                if (iree_status_is_ok(status)) {
                  if (command == IREE_ALLOCATOR_COMMAND_FREE) {
                    state.live -= had_storage;
                  } else if (!had_storage) {
                    ++state.live;
                  }
                }
                return status;
              }};
    }
  };
};

TEST_P(RegistryAllocationTest, FailurePreservesDeclarationAndSlotForRetry) {
  const bool explicit_device = (GetParam() & 1) != 0;
  const bool explicit_pool = (GetParam() & 2) != 0;
  const std::string document =
      R"({"name":"vectors","admission":"drained",)" +
      std::string(explicit_device ? R"("device":"first",)" : "") +
      std::string(explicit_pool ? R"("pool":"pool",)" : "") +
      R"("definition":{"kind":"embedding","assets":"/weights",)"
      R"("prepare":"/boot.loom","control":"/run.loom",)"
      R"("sources":["/kernel.loom"]}})";
  // Each missing binding is one owned string; registration also owns history.
  const size_t allocation_count = !explicit_device + !explicit_pool + 1;
  for (size_t fail_at = 0; fail_at < allocation_count; ++fail_at) {
    SCOPED_TRACE(fail_at);
    AllocationFault fault;
    AllocatorState declaration_allocator{&fault};
    AllocatorState registry_allocator{&fault};
    loom_serve_model_configuration_t configuration = {};
    IREE_ASSERT_OK(loom_serve_model_configuration_initialize(
        iree_make_string_view(document.data(), document.size()),
        iree_string_view_empty(), nullptr, declaration_allocator.allocator(),
        &configuration));
    const loom_serve_registration_factory_t factory = {
        .kinds = 1u << LOOM_SERVE_MODEL_KIND_EMBEDDING,
        .device_names = {1, kDeviceNames},
        .pool_count = 1,
        .pools = kPools,
    };
    loom_serve_registry_t* registry = nullptr;
    IREE_ASSERT_OK(loom_serve_registry_allocate(
        0, nullptr, 1, factory, registry_allocator.allocator(), &registry));
    const auto declaration_live = declaration_allocator.live;
    const auto registry_live = registry_allocator.live;
    std::array<unsigned char, sizeof(configuration)> before;
    std::memcpy(before.data(), &configuration, before.size());
    fault = {.fail_at = fail_at};
    iree_host_size_t index = IREE_HOST_SIZE_MAX;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_serve_registry_register(registry, &configuration, &index));
    EXPECT_EQ(fault.attempts, fail_at + 1);
    EXPECT_EQ(std::memcmp(&configuration, before.data(), before.size()), 0);
    EXPECT_EQ(index, IREE_HOST_SIZE_MAX);
    EXPECT_EQ(loom_serve_registry_count(registry), 0u);
    const auto* model = loom_serve_registry_at(registry, 0);
    EXPECT_EQ(model->service.name.size, 0u);
    EXPECT_EQ(model->stream, nullptr);
    EXPECT_EQ(model->incarnation, 0u);
    EXPECT_EQ(declaration_allocator.live, declaration_live);
    EXPECT_EQ(registry_allocator.live, registry_live);

    fault = {};
    IREE_ASSERT_OK(
        loom_serve_registry_register(registry, &configuration, &index));
    EXPECT_EQ(index, 0u);
    EXPECT_EQ(loom_serve_registry_count(registry), 1u);
    EXPECT_EQ(model->incarnation, 1u);
    EXPECT_TRUE(
        iree_string_view_equal(model->configuration.device, IREE_SV("first")));
    EXPECT_TRUE(
        iree_string_view_equal(model->configuration.pool, IREE_SV("pool")));
    EXPECT_EQ(configuration.name.data, nullptr);
    EXPECT_EQ(configuration.device.data, nullptr);
    EXPECT_EQ(configuration.pool.data, nullptr);
    IREE_EXPECT_OK(loom_serve_registry_free(registry));
    loom_serve_model_configuration_deinitialize(&configuration);
    EXPECT_EQ(declaration_allocator.live, 0u);
    EXPECT_EQ(registry_allocator.live, 0u);
  }
}

INSTANTIATE_TEST_SUITE_P(ExplicitBindings, RegistryAllocationTest,
                         ::testing::Range(uint32_t{0}, uint32_t{4}));

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
