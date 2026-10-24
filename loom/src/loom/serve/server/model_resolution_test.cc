// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/model_resolution.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/program/package.h"
#include "loom/serve/server/registry.h"

namespace {

constexpr char kCard[] = R"json({
  "schema_version": 1,
  "model": {"id": "fixture/text", "name": "Text fixture"},
  "programs": {"source": {"inputs": ["prepare.loom", "control.loom"],
                          "libraries": []}},
  "services": {"chat": {"kind": "text", "version": 1}},
  "variants": {"tiny": {"services": {"chat": {
    "programs": ["source"],
    "parameters": {"weights": {"files": ["model.safetensors"]}},
    "tokenizers": {"text": {"path": "vocabulary.data", "format": "huggingface-json"}}
  }}}},
  "presets": {"check": {"service": "chat", "variant": "tiny",
    "programs": "source", "execution": {"rows": 3, "context_capacity": 96,
    "pool_capacity": 256, "prefill_capacity": 32, "max_tokens": 4}}}
})json";

constexpr char kDeployment[] = R"json({
  "name": "fixture-text",
  "model": {"path": "cards/model.json", "preset": "check"},
  "definition": {"kind": "text", "rows": 2, "context_capacity": 64},
  "base_directory": "//host/models"
})json";

class ModelResolutionTest : public ::testing::Test {
 protected:
  void SetUp() override { Initialize(kCard); }

  void Initialize(const std::string& card,
                  const std::string& deployment = kDeployment) {
    loom_serve_model_configuration_deinitialize(&configuration_);
    const loom_serve_package_file_t file = {
        IREE_SV("deployment/cards/model.json"),
        iree_make_const_byte_span(card.data(), card.size())};
    loom_serve_package_t* package = nullptr;
    IREE_ASSERT_OK(loom_serve_package_create(IREE_SV("deployment/serve.json"),
                                             1, &file, allocator_, &package));
    auto status = loom_serve_model_configuration_initialize(
        iree_make_cstring_view(deployment.c_str()), {}, package, allocator_,
        &configuration_);
    loom_serve_package_release(package);
    IREE_ASSERT_OK(status);
  }

  void TearDown() override {
    EXPECT_EQ(task_, nullptr);
    loom_serve_model_configuration_deinitialize(&configuration_);
  }

  loomc_task_sink_t Sink() {
    return {+[](void* self, loomc_task_t* task) {
              auto* test = static_cast<ModelResolutionTest*>(self);
              EXPECT_EQ(test->task_, nullptr);
              test->task_ = task;
              return loomc_ok_status();
            },
            this};
  }

  loom_serve_wake_callback_t Observer() {
    return {
        +[](void* self) { ++static_cast<ModelResolutionTest*>(self)->wakes_; },
        this};
  }

  void Execute() {
    ASSERT_NE(task_, nullptr);
    auto* task = task_;
    task_ = nullptr;
    loomc_task_execute(task, 0);
  }

  // Allocator owning each declaration and its asynchronous result.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Authored declaration retained until the resolution task is joined.
  loom_serve_model_configuration_t configuration_ = {};
  // Accepted task borrowed from the resolution owner until execution.
  loomc_task_t* task_ = nullptr;
  // Number of completion notifications received by the owner.
  int wakes_ = 0;
};

TEST_F(ModelResolutionTest, ResolvesDefaultsOnWorkerAndOwnsResult) {
  loom_serve_model_resolution_t* resolution = nullptr;
  IREE_ASSERT_OK(loom_serve_model_resolution_start(
      Sink(), &configuration_, Observer(), allocator_, &resolution));
  loom_serve_model_configuration_t resolved = {};
  iree_status_t status = iree_ok_status();
  EXPECT_FALSE(
      loom_serve_model_resolution_take(&resolution, &status, &resolved));
  EXPECT_EQ(wakes_, 0);
  Execute();
  EXPECT_EQ(wakes_, 1);
  ASSERT_TRUE(
      loom_serve_model_resolution_take(&resolution, &status, &resolved));
  IREE_ASSERT_OK(status);
  EXPECT_EQ(resolution, nullptr);
  loom_serve_model_configuration_deinitialize(&configuration_);
  EXPECT_EQ(resolved.text.rows, 2u);
  EXPECT_EQ(resolved.text.context_capacity, 64u);
  EXPECT_EQ(resolved.text.prefill_capacity, 32u);
  EXPECT_EQ(resolved.text.max_tokens, 4u);
  EXPECT_EQ(resolved.text.tokenizer.size, 0u);
  EXPECT_EQ(std::string(resolved.sources.package_directory.data,
                        resolved.sources.package_directory.size),
            "cards");
  EXPECT_EQ(resolved.sources.reference.path.size, 0u);
  ASSERT_NE(resolved.sources.card, nullptr);
  EXPECT_EQ(loom_model_card_selection(resolved.sources.card)->inputs.count, 2u);
  loom_serve_model_configuration_deinitialize(&resolved);
}

class ServiceCardResolutionTest
    : public ModelResolutionTest,
      public ::testing::WithParamInterface<const char*> {
 protected:
  void InitializeService(int version) {
    const std::string kind = GetParam();
    const std::string card =
        R"({"schema_version":1,"model":{"id":"fixture","name":"Fixture"},
          "programs":{"source":{"inputs":["unopened.loom"],"libraries":[]}},
          "services":{"serve":{"kind":")" +
        kind + R"(","version":)" + std::to_string(version) +
        R"(}},"variants":{"fixture":{"services":{"serve":{
          "programs":["source"]}}}}})";
    const std::string deployment =
        R"({"name":"fixture","model":{"path":"cards/model.json"},
          "definition":{"kind":")" +
        kind + R"("}})";
    Initialize(card, deployment);
  }
};

TEST_P(ServiceCardResolutionTest, SupportedVersionResolvesWithoutProgramIo) {
  InitializeService(1);
  loom_serve_model_resolution_t* resolution = nullptr;
  IREE_ASSERT_OK(loom_serve_model_resolution_start(
      Sink(), &configuration_, Observer(), allocator_, &resolution));
  Execute();
  loom_serve_model_configuration_t resolved = {};
  iree_status_t status = iree_ok_status();
  ASSERT_TRUE(
      loom_serve_model_resolution_take(&resolution, &status, &resolved));
  IREE_EXPECT_OK(status);
  EXPECT_NE(resolved.sources.card, nullptr);
  EXPECT_EQ(resolved.kind, configuration_.kind);
  loom_serve_model_configuration_deinitialize(&resolved);
}

TEST_P(ServiceCardResolutionTest, UnsupportedVersionFailsBeforeProgramIo) {
  InitializeService(2);
  loom_serve_model_resolution_t* resolution = nullptr;
  IREE_ASSERT_OK(loom_serve_model_resolution_start(
      Sink(), &configuration_, Observer(), allocator_, &resolution));
  Execute();
  loom_serve_model_configuration_t resolved = {};
  iree_status_t status = iree_ok_status();
  ASSERT_TRUE(
      loom_serve_model_resolution_take(&resolution, &status, &resolved));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
  EXPECT_EQ(resolved.sources.card, nullptr);
  loom_serve_model_configuration_deinitialize(&resolved);
}

INSTANTIATE_TEST_SUITE_P(NativeServices, ServiceCardResolutionTest,
                         ::testing::Values("text", "sequence", "image",
                                           "embedding", "decision", "speech",
                                           "recognition", "activity"));

TEST_F(ModelResolutionTest, RejectedSubmissionOwesNoWakeOrBorrow) {
  const loomc_task_sink_t sink = {
      +[](void*, loomc_task_t*) {
        return loomc_make_status(LOOMC_STATUS_UNAVAILABLE, "queue closed");
      },
      nullptr};
  loom_serve_model_resolution_t* resolution = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      loom_serve_model_resolution_start(sink, &configuration_, Observer(),
                                        allocator_, &resolution));
  EXPECT_EQ(resolution, nullptr);
  EXPECT_EQ(wakes_, 0);
}

TEST_F(ModelResolutionTest, RemovalJoinsResolutionBeforeReleasingDeclaration) {
  const loom_serve_registration_factory_t factory = {
      .kinds = 1u << LOOM_SERVE_MODEL_KIND_TEXT,
      .device_count = 1,
      .resolution_sink = Sink(),
      .create =
          +[](void*, const loom_serve_model_configuration_t*,
              loom_serve_wake_callback_t, loom_serve_telemetry_stream_t*,
              iree_allocator_t, loom_serve_registration_t*) {
            ADD_FAILURE() << "cancelled card resolution constructed a model";
            return iree_make_status(IREE_STATUS_INTERNAL,
                                    "unexpected construction");
          },
  };
  loom_serve_registry_t* registry = nullptr;
  IREE_ASSERT_OK(loom_serve_registry_allocate(0, nullptr, 1, factory,
                                              allocator_, &registry));
  iree_host_size_t index = 0;
  IREE_ASSERT_OK(
      loom_serve_registry_register(registry, &configuration_, &index));
  bool progress = false;
  IREE_ASSERT_OK(
      loom_serve_registry_advance(registry, index, Observer(), &progress));
  ASSERT_TRUE(progress);
  ASSERT_NE(task_, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_registry_free(registry));
  IREE_ASSERT_OK(loom_serve_registry_remove(
      loom_serve_registry_at(registry, index), LOOM_SERVE_REMOVAL_ABORT));
  IREE_ASSERT_OK(
      loom_serve_registry_advance(registry, index, Observer(), &progress));
  EXPECT_FALSE(progress);
  EXPECT_EQ(loom_serve_registry_count(registry), 1u);
  Execute();
  IREE_ASSERT_OK(
      loom_serve_registry_advance(registry, index, Observer(), &progress));
  EXPECT_TRUE(progress);
  EXPECT_EQ(loom_serve_registry_count(registry), 0u);
  IREE_ASSERT_OK(loom_serve_registry_free(registry));
}

}  // namespace
