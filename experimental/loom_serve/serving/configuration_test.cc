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

std::string Document(const std::string& kind, const std::string& fields = "",
                     const std::string& envelope = "") {
  return R"({"name":"sc\u006fut","definition":{"kind":")" + kind +
         R"(","prepare":"boot.loombc","control":"run.loom","sources":["config.loom","kernel.loombc"])" +
         fields + "}" + envelope + "}";
}

std::string Text(const std::string& fields = "") {
  return Document(
      "text",
      R"(,"weights":"model.gguf","tokenizer":"tokenizer.json")" + fields);
}

std::string Image(const std::string& fields = "") {
  return Document("image", R"(,"checkpoint":"assets")" + fields);
}

class ConfigurationTest : public ::testing::Test {
 protected:
  void TearDown() override {
    loom_serve_model_configuration_deinitialize(&model_);
  }

  iree_status_t Parse(const std::string& json,
                      iree_string_view_t directory = IREE_SV("models/local")) {
    loom_serve_model_configuration_deinitialize(&model_);
    return loom_serve_model_configuration_initialize(
        iree_make_string_view(json.data(), json.size()), directory, &model_,
        iree_allocator_system());
  }

  static std::string String(iree_string_view_t value) {
    return std::string(value.data, value.size);
  }

  // Owned declaration independent of parser input lifetime.
  loom_serve_model_configuration_t model_ = {};
};

TEST_F(ConfigurationTest, OwnsAuthoredPathsAndBaseSeparately) {
  auto json =
      Document("embedding", R"(,"assets":".","options":{"batch_units":64})");
  IREE_ASSERT_OK(Parse(json));
  json.assign(json.size(), 'x');
  EXPECT_EQ(String(model_.name), "scout");
  EXPECT_FALSE(model_.start_drained);
  EXPECT_EQ(model_.pending_requests, 32u);
  EXPECT_EQ(String(model_.sources.base_directory), "models/local");
  EXPECT_EQ(String(model_.sources.prepare), "boot.loombc");
  EXPECT_EQ(String(model_.sources.control), "run.loom");
  ASSERT_EQ(model_.sources.count, 2u);
  EXPECT_EQ(String(model_.sources.values[1]), "kernel.loombc");
  EXPECT_EQ(String(model_.finite.assets), ".");
  EXPECT_EQ(String(model_.options), R"({"batch_units":64})");
  loom_serve_model_configuration_deinitialize(&model_);
  EXPECT_EQ(model_.name.data, nullptr);
  EXPECT_EQ(model_.sources.values, nullptr);
}

TEST_F(ConfigurationTest, AdmissionIsAnIndependentRegistrationSetting) {
  for (const char* field :
       {R"(,"admission":"open")", R"(,"admission":"drained")"}) {
    IREE_ASSERT_OK(
        Parse(Document("decision", R"(,"assets":"weights")", field)));
    EXPECT_EQ(model_.start_drained,
              std::string(field).find("drained") != std::string::npos);
  }
  for (const char* field : {R"(,"admission":"paused")", R"(,"admission":true)",
                            R"(,"admission":"open","admission":"drained")"}) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        Parse(Document("embedding", R"(,"assets":".")", field)));
    EXPECT_EQ(model_.name.data, nullptr);
  }
}

TEST_F(ConfigurationTest, HttpRelativePathsRequireAnExplicitBase) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse(Document("embedding", R"(,"assets":".")"), {}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse(Document("embedding", R"(,"assets":".")",
                                       R"(,"base_directory":"relative")")));
}

TEST_F(ConfigurationTest,
       ExplicitBaseOverridesDocumentAndHttpHasNoImplicitBase) {
  const std::string json =
      Document("embedding", R"(,"assets":".")",
               R"(,"base_directory":"//files/models/vectors")");
  for (auto directory : {IREE_SV("ignored"), iree_string_view_empty()}) {
    IREE_ASSERT_OK(Parse(json, directory));
    EXPECT_EQ(String(model_.sources.base_directory), "//files/models/vectors");
    EXPECT_EQ(String(model_.sources.prepare), "boot.loombc");
    EXPECT_EQ(String(model_.finite.assets), ".");
  }
}

TEST_F(ConfigurationTest, AbsoluteHttpReferencesNeedNoBase) {
  IREE_ASSERT_OK(Parse(
      R"({"name":"vectors","definition":{"kind":"embedding",
          "prepare":"//files/models/boot.loombc",
          "control":"//files/models/run.loom",
          "sources":["//files/models/embed.loombc"],
          "assets":"//files/weights"}})",
      {}));
  EXPECT_EQ(model_.sources.base_directory.size, 0u);
  EXPECT_EQ(String(model_.sources.values[0]), "//files/models/embed.loombc");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(R"({"name":"vectors","definition":{"kind":"embedding",
               "prepare":"//files/models/boot.loombc",
               "control":"//files/models/run.loom",
               "sources":["relative.loombc"],"assets":"//files/weights"}})",
            {}));
}

TEST_F(ConfigurationTest, RequiredProgramsAndPublicNamesRejectBeforeIo) {
  for (
      const char* definition :
      {R"({"kind":"embedding","assets":".","control":"c","sources":["s"]})",
       R"({"kind":"embedding","assets":".","prepare":"p","sources":["s"]})",
       R"({"kind":"embedding","assets":".","prepare":"p","control":"c"})",
       R"({"kind":"embedding","assets":".","prepare":"","control":"c","sources":["s"]})"}) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        Parse(std::string(R"({"name":"x","definition":)") + definition + "}"));
    EXPECT_EQ(model_.sources.values, nullptr);
  }
  for (const char* name :
       {"", "has space", R"(a\u0000b)", R"(a\nb)", R"(\u03bb)"}) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        Parse(std::string(R"({"name":")") + name +
              R"(","definition":{"kind":"embedding","assets":".",
                   "prepare":"p","control":"c","sources":["s"]}})"));
  }
}

TEST_F(ConfigurationTest,
       TextDefaultsAndSpecializationsRemainModelIndependent) {
  IREE_ASSERT_OK(Parse(Text()));
  EXPECT_EQ(model_.kind, LOOM_SERVE_MODEL_KIND_TEXT);
  EXPECT_EQ(model_.text.rows, 4u);
  EXPECT_EQ(model_.text.admission_overtakes, 4u);
  EXPECT_EQ(model_.text.context_capacity, 16384u);
  EXPECT_EQ(model_.text.pool_capacity, 65536u);
  EXPECT_EQ(model_.text.prefill_capacity, 512u);
  EXPECT_EQ(model_.text.max_tokens, 512u);
  EXPECT_EQ(model_.text.draft_depth, 0u);
  EXPECT_EQ(model_.text.continuation_epochs, 1u);
  EXPECT_EQ(model_.text.session_cache_capacity, 0u);
  EXPECT_EQ(model_.text.prefix_cache_capacity, 0u);
  EXPECT_EQ(String(model_.options), "{}");
  IREE_ASSERT_OK(Parse(Text(
      R"(,"rows":8,"context_capacity":262144,"pool_capacity":4096,"prefill_capacity":128,"draft_depth":3,"continuation_epochs":2,"checkpoint_capacity":2,"session_cache_capacity":24,"prefix_cache_capacity":4,"pending_requests":17,"max_tokens":128,"admission_overtakes":0,"options":{"draft_weights":"draft.gguf"})")));
  EXPECT_EQ(model_.text.rows, 8u);
  EXPECT_EQ(model_.text.admission_overtakes, 0u);
  EXPECT_EQ(model_.text.context_capacity, 262144u);
  EXPECT_EQ(model_.text.pool_capacity, 4096u);
  EXPECT_EQ(model_.text.prefill_capacity, 128u);
  EXPECT_EQ(model_.text.checkpoint_capacity, 2u);
  EXPECT_EQ(model_.text.session_cache_capacity, 24u);
  EXPECT_EQ(model_.text.prefix_cache_capacity, 4u);
  EXPECT_EQ(model_.pending_requests, 17u);
  EXPECT_EQ(model_.text.max_tokens, 128u);
  EXPECT_EQ(String(model_.options), R"({"draft_weights":"draft.gguf"})");
}

TEST_F(ConfigurationTest, ImageDefaultsAndSpecializations) {
  IREE_ASSERT_OK(Parse(Image()));
  EXPECT_EQ(model_.kind, LOOM_SERVE_MODEL_KIND_IMAGE);
  EXPECT_EQ(model_.image.width, 384u);
  EXPECT_EQ(model_.image.height, 384u);
  EXPECT_EQ(model_.image.text_tokens, 512u);
  EXPECT_EQ(model_.image.command_quantum, 0u);
  EXPECT_EQ(model_.image.adapter.size, 0u);
  IREE_ASSERT_OK(Parse(Image(
      R"(,"adapter":"style\u002esafetensors","width":768,"height":512,"text_tokens":1024,"command_quantum":64,"pending_requests":3)")));
  EXPECT_EQ(String(model_.image.adapter), "style.safetensors");
  EXPECT_EQ(model_.image.width, 768u);
  EXPECT_EQ(model_.image.height, 512u);
  EXPECT_EQ(model_.image.text_tokens, 1024u);
  EXPECT_EQ(model_.image.command_quantum, 64u);
  EXPECT_EQ(model_.pending_requests, 3u);
}

TEST_F(ConfigurationTest, NativeCapacitiesAreNotOneModelShape) {
  for (unsigned depth = 0; depth <= 7; ++depth) {
    IREE_ASSERT_OK(
        Parse(Text(std::string(R"(,"draft_depth":)") + std::to_string(depth) +
                   R"(,"prefill_capacity":)" + std::to_string(depth + 1) +
                   R"(,"continuation_epochs":)" + (depth ? "2" : "1"))));
    EXPECT_EQ(model_.text.draft_depth, depth);
  }
  for (unsigned capacity : {513u, 1024u, 4097u, 262144u}) {
    IREE_ASSERT_OK(Parse(
        Text(std::string(R"(,"context_capacity":262144,"prefill_capacity":)") +
             std::to_string(capacity))));
    EXPECT_EQ(model_.text.prefill_capacity, capacity);
  }
}

TEST_F(ConfigurationTest, RejectsInvalidAndWrongModalityFields) {
  for (const char* field :
       {R"(,"rows":0)",
        R"(,"rows":17)",
        R"(,"rows":2,"rows":4)",
        R"(,"draft_depth":8)",
        R"(,"continuation_epochs":2)",
        R"(,"prefill_capacity":2,"draft_depth":3)",
        R"(,"prefill_capacity":262145)",
        R"(,"prefill_capacity":1024,"context_capacity":513)",
        R"(,"pool_capacity":0)",
        R"(,"pending_requests":-1)",
        R"(,"max_tokens":1.5)",
        R"(,"typo":true)",
        R"(,"width":384)",
        R"(,"command_quantum":1)",
        R"(,"options":[])",
        R"(,"admission_overtakes":-1)",
        R"(,"admission_overtakes":1.5)",
        R"(,"admission_overtakes":2147483648)",
        R"(,"session_cache_capacity":-1)",
        R"(,"session_cache_capacity":2147483647)",
        R"(,"checkpoint_capacity":2147483647,"session_cache_capacity":1)",
        R"(,"prefix_cache_capacity":-1)",
        R"(,"prefix_cache_capacity":2147483648)",
        R"(,"prefix_cache_capacity":2147483647,"checkpoint_capacity":1)",
        R"(,"prefix_cache_capacity":2147483647,"session_cache_capacity":1)",
        R"(,"checkpoint_capacity":18446744073709551615)"}) {
    SCOPED_TRACE(field);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(Text(field)));
    EXPECT_EQ(model_.sources.values, nullptr);
  }
  for (const char* field :
       {R"(,"options":[])", R"(,"weights":"file.gguf")", R"(,"rows":1)",
        R"(,"width":0)", R"(,"width":384,"width":384)", R"(,"height":-1)",
        R"(,"height":4294967296)", R"(,"text_tokens":0)",
        R"(,"text_tokens":1.5)", R"(,"command_quantum":-1)",
        R"(,"command_quantum":1.5)", R"(,"command_quantum":4294967296)",
        R"(,"adapter":"")", R"(,"adapter":"style\u0000file")"}) {
    SCOPED_TRACE(field);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(Image(field)));
  }
  for (const char* kind : {"embedding", "decision"}) {
    IREE_ASSERT_OK(Parse(Document(kind, R"(,"assets":"weights")")));
    for (const char* field : {R"(,"rows":8)", R"(,"width":32)",
                              R"(,"weights":"x")", R"(,"options":[])"}) {
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          Parse(Document(kind, std::string(R"(,"assets":"weights")") + field)));
    }
  }
}

TEST_F(ConfigurationTest,
       RejectsMalformedDocumentsAndReleasesPartialOwnership) {
  for (const auto& json :
       {std::string("{}"), std::string(R"({"models":[]})"),
        std::string(R"({"name":"x","definition":null})"),
        std::string(R"({"name":"x","definition":{}})"), Document("audio"),
        Document("text"), Document("image"), Document("embedding"),
        Text() + " trailing", Text(R"(,"prepare":"extra.loom")"),
        Text(R"(,"admission":"open")"),
        Document("embedding", R"(,"assets":".")", R"(,"name":"x")")}) {
    SCOPED_TRACE(json);
    iree::Status status(Parse(json));
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(model_.name.data, nullptr);
    EXPECT_EQ(model_.sources.values, nullptr);
  }
  for (const char* value : {"[]", "[null]", "[1]", R"(["a", "bad\u0000path"])",
                            R"(["a", ""])", R"(["a", {}])"}) {
    const std::string json =
        R"({"name":"x","definition":{"kind":"embedding","assets":".","prepare":"p","control":"c","sources":)" +
        std::string(value) + "}}";
    iree::Status status(Parse(json));
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(model_.sources.values, nullptr);
  }
}

}  // namespace
