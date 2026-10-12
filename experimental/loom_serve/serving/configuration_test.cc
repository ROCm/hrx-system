// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/serving/configuration.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(ConfigurationTest, IndividualDeclarationOwnsItsAllocatorAndAdmissionMode) {
  std::string json =
      R"({"kind":"embedding","name":"sc\u006fut","source":"model","assets":"assets","admission":"drained"})";
  loom_serve_model_configuration_t model;
  IREE_ASSERT_OK(loom_serve_model_configuration_initialize(
      iree_make_string_view(json.data(), json.size()), &model,
      iree_allocator_system()));
  json.assign(json.size(), 'x');
  EXPECT_EQ(std::string(model.name.data, model.name.size), "scout");
  EXPECT_TRUE(model.start_drained);
  EXPECT_EQ(model.pending_requests, 32u);
  loom_serve_model_configuration_deinitialize(&model);
  EXPECT_EQ(model.name.data, nullptr);
  loom_serve_model_configuration_deinitialize(&model);
}

TEST(ConfigurationTest, AdmissionRejectsUnknownDuplicateAndWronglyTypedValues) {
  for (const char* field : {R"("admission":"paused")", R"("admission":true)",
                            R"("admission":"open","admission":"drained")"}) {
    const std::string json =
        std::string(
            R"({"kind":"embedding","name":"scout","source":"model","assets":"assets",)") +
        field + "}";
    loom_serve_model_configuration_t model;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_model_configuration_initialize(
                              iree_make_string_view(json.data(), json.size()),
                              &model, iree_allocator_system()));
    EXPECT_EQ(model.name.data, nullptr);
    loom_serve_model_configuration_deinitialize(&model);
  }
}

std::string Model(const std::string& name, const std::string& options = "") {
  return R"({"kind":"text","name":")" + name +
         R"(","source":"models/text","weights":"weights.gguf","tokenizer":"tokenizer.json")" +
         options + "}";
}

std::string Image(const std::string& name, const std::string& options = "") {
  return R"({"kind":"image","name":")" + name +
         R"(","source":"models/image","checkpoint":"assets/image")" + options +
         "}";
}

TEST(ConfigurationTest, IndependentModelsOwnDecodedStringsAndOptions) {
  std::string json =
      "{\"models\":[" + Model("scout") + "," +
      Model(
          R"(research\u0065r)",
          R"(,"rows":8,"context_capacity":262144,"pool_capacity":4096,"checkpoint_capacity":2,"prefill_capacity":128,"draft_depth":3,"continuation_epochs":2,"pending_requests":17,"max_tokens":128)") +
      "]}";
  loom_serve_configuration_t configuration;
  IREE_ASSERT_OK(loom_serve_configuration_initialize(
      iree_make_cstring_view(json.c_str()), &configuration,
      iree_allocator_system()));
  json.assign(json.size(), 'x');
  ASSERT_EQ(configuration.model_count, 2u);
  const auto& a = configuration.models[0];
  const auto& b = configuration.models[1];
  EXPECT_EQ(std::string(a.name.data, a.name.size), "scout");
  EXPECT_EQ(std::string(b.name.data, b.name.size), "researcher");
  EXPECT_EQ(a.kind, LOOM_SERVE_MODEL_KIND_TEXT);
  EXPECT_EQ(b.kind, LOOM_SERVE_MODEL_KIND_TEXT);
  EXPECT_EQ(a.text.rows, 4u);
  EXPECT_EQ(a.text.admission_overtakes, 4u);
  EXPECT_EQ(a.text.context_capacity, 16384u);
  EXPECT_EQ(a.text.draft_depth, 0u);
  EXPECT_EQ(a.text.session_cache_capacity, 0u);
  EXPECT_EQ(a.text.prefix_cache_capacity, 0u);
  EXPECT_EQ(b.text.rows, 8u);
  EXPECT_EQ(b.text.admission_overtakes, 8u);
  EXPECT_EQ(b.text.context_capacity, 262144u);
  EXPECT_EQ(b.text.pool_capacity, 4096u);
  EXPECT_EQ(b.text.checkpoint_capacity, 2u);
  EXPECT_EQ(b.text.prefill_capacity, 128u);
  EXPECT_EQ(b.text.draft_depth, 3u);
  EXPECT_EQ(b.text.continuation_epochs, 2u);
  EXPECT_EQ(b.pending_requests, 17u);
  EXPECT_EQ(b.text.max_tokens, 128u);
  loom_serve_configuration_deinitialize(&configuration);
}

TEST(ConfigurationTest, EmptyCatalogHasNoModelStorage) {
  loom_serve_configuration_t configuration;
  IREE_ASSERT_OK(loom_serve_configuration_initialize(
      IREE_SV(R"({"models":[]})"), &configuration, iree_allocator_system()));
  EXPECT_EQ(configuration.model_count, 0u);
  EXPECT_EQ(configuration.models, nullptr);
  loom_serve_configuration_deinitialize(&configuration);
}

TEST(ConfigurationTest, DraftDepthIsModelPolicyWithinNativePacketCapacity) {
  for (unsigned depth = 0; depth <= 7; ++depth) {
    const std::string json = Model(
        "draft", ",\"draft_depth\":" + std::to_string(depth) +
                     ",\"prefill_capacity\":" + std::to_string(depth + 1) +
                     ",\"continuation_epochs\":" + (depth ? "2" : "1"));
    loom_serve_model_configuration_t model;
    IREE_ASSERT_OK(loom_serve_model_configuration_initialize(
        iree_make_string_view(json.data(), json.size()), &model,
        iree_allocator_system()));
    EXPECT_EQ(model.text.draft_depth, depth);
    loom_serve_model_configuration_deinitialize(&model);
  }
}

TEST(ConfigurationTest, TextBootstrapOptionsRemainOpaqueSourcePolicy) {
  const std::string json =
      Model("draft", R"(,"options":{"draft_weights":"/models/draft.gguf"})");
  loom_serve_model_configuration_t model;
  IREE_ASSERT_OK(loom_serve_model_configuration_initialize(
      iree_make_string_view(json.data(), json.size()), &model,
      iree_allocator_system()));
  EXPECT_EQ(std::string(model.options.data, model.options.size),
            R"({"draft_weights":"/models/draft.gguf"})");
  loom_serve_model_configuration_deinitialize(&model);
}

TEST(ConfigurationTest, TextAdmissionBudgetIsIndependentOfRowCount) {
  for (int budget : {0, 1, 16}) {
    const auto json = Model("bounded", ",\"rows\":2,\"admission_overtakes\":" +
                                           std::to_string(budget));
    loom_serve_model_configuration_t model;
    IREE_ASSERT_OK(loom_serve_model_configuration_initialize(
        iree_make_cstring_view(json.c_str()), &model, iree_allocator_system()));
    EXPECT_EQ(model.text.admission_overtakes, static_cast<size_t>(budget));
    loom_serve_model_configuration_deinitialize(&model);
  }
  for (const char* value : {"-1", "1.5", "2147483648"}) {
    const auto json =
        Model("invalid", std::string(",\"admission_overtakes\":") + value);
    loom_serve_model_configuration_t model;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_model_configuration_initialize(
                              iree_make_cstring_view(json.c_str()), &model,
                              iree_allocator_system()));
    loom_serve_model_configuration_deinitialize(&model);
  }
}

TEST(ConfigurationTest, AutomaticCacheAndExplicitPinsHaveIndependentQuotas) {
  loom_serve_model_configuration_t model;
  const auto json =
      Model("cached",
            ",\"checkpoint_capacity\":3,\"session_cache_capacity\":24,"
            "\"prefix_cache_capacity\":4");
  IREE_ASSERT_OK(loom_serve_model_configuration_initialize(
      iree_make_cstring_view(json.c_str()), &model, iree_allocator_system()));
  EXPECT_EQ(model.text.checkpoint_capacity, 3u);
  EXPECT_EQ(model.text.session_cache_capacity, 24u);
  EXPECT_EQ(model.text.prefix_cache_capacity, 4u);
  loom_serve_model_configuration_deinitialize(&model);
  for (const char* options :
       {",\"session_cache_capacity\":-1",
        ",\"session_cache_capacity\":2147483647",
        ",\"checkpoint_capacity\":2147483647,\"session_cache_capacity\":1",
        ",\"prefix_cache_capacity\":-1",
        ",\"prefix_cache_capacity\":2147483648",
        ",\"prefix_cache_capacity\":2147483647,\"checkpoint_capacity\":1",
        ",\"prefix_cache_capacity\":2147483647,\"session_cache_capacity\":1",
        ",\"checkpoint_capacity\":18446744073709551615"}) {
    const auto invalid = Model("cached", options);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_model_configuration_initialize(
                              iree_make_cstring_view(invalid.c_str()), &model,
                              iree_allocator_system()));
    loom_serve_model_configuration_deinitialize(&model);
  }
}

TEST(ConfigurationTest, MixedModalitiesKeepDisjointOptionsAndOwnedStrings) {
  std::string json =
      "{\"models\":[" + Image("default-image") + "," + Model("text") + "," +
      Image(
          "styled",
          R"(,"adapter":"style\u002esafetensors","width":768,"height":512,"text_tokens":1024,"command_quantum":64,"pending_requests":3,"options":{"checkpoint":"compact.safetensors"})") +
      "]}";
  loom_serve_configuration_t configuration;
  IREE_ASSERT_OK(loom_serve_configuration_initialize(
      iree_make_cstring_view(json.c_str()), &configuration,
      iree_allocator_system()));
  json.assign(json.size(), 'x');
  ASSERT_EQ(configuration.model_count, 3u);
  const auto& a = configuration.models[0];
  const auto& b = configuration.models[2];
  EXPECT_EQ(a.kind, LOOM_SERVE_MODEL_KIND_IMAGE);
  EXPECT_EQ(b.kind, LOOM_SERVE_MODEL_KIND_IMAGE);
  EXPECT_EQ(configuration.models[1].kind, LOOM_SERVE_MODEL_KIND_TEXT);
  EXPECT_EQ(std::string(a.name.data, a.name.size), "default-image");
  EXPECT_EQ(std::string(a.source.data, a.source.size), "models/image");
  EXPECT_EQ(std::string(a.image.checkpoint.data, a.image.checkpoint.size),
            "assets/image");
  EXPECT_EQ(a.image.adapter.size, 0u);
  EXPECT_EQ(a.image.width, 384u);
  EXPECT_EQ(a.image.height, 384u);
  EXPECT_EQ(a.image.text_tokens, 512u);
  EXPECT_EQ(a.image.command_quantum, 0u);
  EXPECT_EQ(a.pending_requests, 32u);
  EXPECT_EQ(std::string(a.options.data, a.options.size), "{}");
  EXPECT_EQ(std::string(b.image.adapter.data, b.image.adapter.size),
            "style.safetensors");
  EXPECT_EQ(b.image.width, 768u);
  EXPECT_EQ(b.image.height, 512u);
  EXPECT_EQ(b.image.text_tokens, 1024u);
  EXPECT_EQ(b.image.command_quantum, 64u);
  EXPECT_EQ(b.pending_requests, 3u);
  EXPECT_EQ(std::string(b.options.data, b.options.size),
            R"({"checkpoint":"compact.safetensors"})");
  loom_serve_configuration_deinitialize(&configuration);
}

TEST(ConfigurationTest, TokenCapacityIsBoundedByContextNotOneModel) {
  for (size_t capacity : {513u, 1024u, 4097u, 262144u}) {
    const std::string json = "{\"models\":[" +
                             Model("wide",
                                   ",\"context_capacity\":262144,"
                                   "\"prefill_capacity\":" +
                                       std::to_string(capacity)) +
                             "]}";
    loom_serve_configuration_t configuration;
    IREE_ASSERT_OK(loom_serve_configuration_initialize(
        iree_make_cstring_view(json.c_str()), &configuration,
        iree_allocator_system()));
    EXPECT_EQ(configuration.models[0].text.prefill_capacity, capacity);
    loom_serve_configuration_deinitialize(&configuration);
  }
}

TEST(ConfigurationTest, EmbeddingPolicyStaysSourceOwned) {
  std::string json = R"({"models":[
    {"kind":"embedding","name":"query","source":"encoder","assets":"weights",
     "options":{"task":"query","batch_units":128}},
    {"kind":"embedding","name":"document","source":"encoder","assets":"weights"}
  ]})";
  loom_serve_configuration_t configuration;
  IREE_ASSERT_OK(loom_serve_configuration_initialize(
      iree_make_cstring_view(json.c_str()), &configuration,
      iree_allocator_system()));
  json.assign(json.size(), 'x');
  ASSERT_EQ(configuration.model_count, 2u);
  const auto& model = configuration.models[0];
  EXPECT_EQ(model.kind, LOOM_SERVE_MODEL_KIND_EMBEDDING);
  EXPECT_EQ(std::string(model.finite.assets.data, model.finite.assets.size),
            "weights");
  EXPECT_EQ(std::string(model.options.data, model.options.size),
            R"({"task":"query","batch_units":128})");
  const auto& defaults = configuration.models[1].options;
  EXPECT_EQ(std::string(defaults.data, defaults.size), "{}");
  loom_serve_configuration_deinitialize(&configuration);
  for (const char* extra : {R"(,"width":32)", R"(,"rows":8)",
                            R"(,"options":[])", R"(,"weights":"wrong")"}) {
    const std::string bad =
        std::string(
            R"({"models":[{"kind":"embedding","name":"m","source":"s","assets":"a")") +
        extra + "}]}";
    iree::Status status(loom_serve_configuration_initialize(
        iree_make_cstring_view(bad.c_str()), &configuration,
        iree_allocator_system()));
    EXPECT_FALSE(status.ok());
  }
}

TEST(ConfigurationTest,
     RejectsAmbiguousOrUnsupportedCatalogsAndCleansPartialState) {
  for (
      const auto& json :
      {std::string("{}"),
       std::string(R"({"models":[null]})"),
       std::string(R"({"models":[{"kind":"image"}]})"),
       std::string(R"({"models":[{"kind":"audio","name":"x"}]})"),
       std::string(R"({"models":[{"kind":"text","name":"x"}]})"),
       "{\"models\":[" + Model("same") + "," + Model("same") + "]}",
       "{\"models\":[" + Model(R"(bad\u0000name)") + "]}",
       "{\"models\":[" + Model("x", R"(,"rows":0)") + "]}",
       "{\"models\":[" + Model("x", R"(,"rows":17)") + "]}",
       "{\"models\":[" + Model("x", R"(,"rows":2,"rows":4)") + "]}",
       "{\"models\":[" + Model("x", R"(,"draft_depth":8)") + "]}",
       "{\"models\":[" + Model("x", R"(,"continuation_epochs":2)") + "]}",
       "{\"models\":[" +
           Model("x", R"(,"prefill_capacity":2,"draft_depth":3)") + "]}",
       "{\"models\":[" + Model("x", R"(,"prefill_capacity":262145)") + "]}",
       "{\"models\":[" +
           Model("x", R"(,"prefill_capacity":1024,"context_capacity":513)") +
           "]}",
       "{\"models\":[" + Model("x", R"(,"pool_capacity":0)") + "]}",
       "{\"models\":[" + Model("x", R"(,"pending_requests":-1)") + "]}",
       "{\"models\":[" + Model("x", R"(,"max_tokens":1.5)") + "]}",
       "{\"models\":[" + Model("x", R"(,"typo":true)") + "]}",
       "{\"models\":[" + Model("x", R"(,"width":384)") + "]}",
       "{\"models\":[" + Model("x", R"(,"command_quantum":1)") + "]}",
       "{\"models\":[" + Model("x", R"(,"options":[])") + "]}",
       "{\"models\":[" + Image("x", R"(,"options":[])") + "]}",
       "{\"models\":[" + Image("x", R"(,"weights":"file.gguf")") + "]}",
       "{\"models\":[" + Image("x", R"(,"rows":1)") + "]}",
       "{\"models\":[" + Image("x", R"(,"width":0)") + "]}",
       "{\"models\":[" + Image("x", R"(,"width":384,"width":384)") + "]}",
       "{\"models\":[" + Image("x", R"(,"height":-1)") + "]}",
       "{\"models\":[" + Image("x", R"(,"height":4294967296)") + "]}",
       "{\"models\":[" + Image("x", R"(,"text_tokens":0)") + "]}",
       "{\"models\":[" + Image("x", R"(,"text_tokens":1.5)") + "]}",
       "{\"models\":[" + Image("x", R"(,"command_quantum":-1)") + "]}",
       "{\"models\":[" + Image("x", R"(,"command_quantum":1.5)") + "]}",
       "{\"models\":[" + Image("x", R"(,"command_quantum":4294967296)") + "]}",
       "{\"models\":[" + Image("x", R"(,"adapter":"")") + "]}",
       "{\"models\":[" + Image("x", R"(,"adapter":"style\u0000file")") + "]}",
       "{\"models\":[" + Image("same") + "," + Model("same") + "]}",
       "{\"models\":[" + Image("valid-image") + "," +
           R"({"kind":"text","name":"bad","source":"models/text","weights":"weights.gguf"})" +
           "]}",
       "{\"models\":[" + Model("valid") + "," + Image("valid-image") + "," +
           Image("bad", R"(,"adapter":"bad\u0000path")") + "]}",
       "{\"models\":[" + Model("x") + "],\"unknown\":0}",
       "{\"models\":[" + Model("x") + "]} trailing"}) {
    SCOPED_TRACE(json);
    loom_serve_configuration_t configuration;
    iree::Status status(loom_serve_configuration_initialize(
        iree_make_cstring_view(json.c_str()), &configuration,
        iree_allocator_system()));
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(configuration.models, nullptr);
    EXPECT_EQ(configuration.model_count, 0u);
    loom_serve_configuration_deinitialize(&configuration);
  }
}

TEST(ConfigurationTest, DecisionAndEmbeddingShareOnlyFiniteConfiguration) {
  loom_serve_configuration_t configuration;
  IREE_ASSERT_OK(loom_serve_configuration_initialize(IREE_SV(R"({"models":[
        {"kind":"decision","name":"judge","source":"joint","assets":"weights","options":{"batch_units":64}},
        {"kind":"embedding","name":"vectors","source":"encoder","assets":"weights"}
      ]})"),
                                                     &configuration,
                                                     iree_allocator_system()));
  ASSERT_EQ(configuration.model_count, 2u);
  EXPECT_EQ(configuration.models[0].kind, LOOM_SERVE_MODEL_KIND_DECISION);
  EXPECT_EQ(configuration.models[1].kind, LOOM_SERVE_MODEL_KIND_EMBEDDING);
  const auto policy = configuration.models[0].options;
  EXPECT_EQ(std::string(policy.data, policy.size), R"({"batch_units":64})");
  loom_serve_configuration_deinitialize(&configuration);
  for (const char* field : {R"("rows":8)", R"("width":32)", R"("weights":"x")",
                            R"("options":[])"}) {
    const std::string json =
        std::string(
            R"({"models":[{"kind":"decision","name":"m","source":"s","assets":"a",)") +
        field + "}]}";
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_configuration_initialize(
                              iree_make_cstring_view(json.c_str()),
                              &configuration, iree_allocator_system()));
    EXPECT_EQ(configuration.models, nullptr);
  }
}

}  // namespace
