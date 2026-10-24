// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/model/card.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr char kCard[] = R"json({
  "$schema": "model-card.v1.schema.json",
  "schema_version": 1,
  "model": {"id": "fixture/text", "name": "Text fixture"},
  "programs": {
    "mixed": {
      "directory": "program",
      "inputs": ["program/prepare.loom", "program/control.loombc"],
      "libraries": [{"path": "program/library.data", "format": "loombc",
                     "data": ["program/weights.loom"]}],
      "config": {"width": 4, "quantized": false}
    },
    "source": {"inputs": ["program/all.loom"], "libraries": []}
  },
  "services": {
    "chat": {"kind": "text", "version": 1},
    "future": {"kind": "image", "version": 9000}
  },
  "variants": {
    "small": {"services": {"chat": {
      "programs": ["mixed", "source"],
      "config": {"width": 8, "dtype": "f32"},
      "parameters": {"model": {"files": ["weight/a.safetensors",
        {"path": "weight/b.data", "format": "safetensors"}]}},
      "tokenizers": {"text": {"path": "asset/vocabulary.data",
                               "format": "huggingface-json"}}
    }}},
    "large": {"services": {"chat": {"programs": ["source"]}}}
  },
  "presets": {
    "check": {"service": "chat", "variant": "small", "programs": "mixed",
              "execution": {"rows": 2}, "request": {"max_tokens": 4}}
  }
})json";

std::string String(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class CardTest : public ::testing::Test {
 protected:
  void TearDown() override { loom_model_card_release(card_); }

  iree_status_t Parse(const std::string& json,
                      const loom_model_selector_t* selector = nullptr) {
    loom_model_card_release(card_);
    card_ = nullptr;
    return loom_model_card_parse(
        iree_make_string_view(json.data(), json.size()), selector,
        iree_allocator_system(), &card_);
  }

  std::string Replace(const char* from, const char* to) {
    std::string json = kCard;
    auto position = json.find(from);
    EXPECT_NE(position, std::string::npos);
    if (position != std::string::npos) {
      json.replace(position, strlen(from), to);
    }
    return json;
  }

  loom_model_card_t* card_ = nullptr;
};

constexpr char kPartitionCard[] = R"json({
  "schema_version": 1,
  "model": {"id": "fixture/split", "name": "Partitioned fixture"},
  "programs": {"source": {"inputs": ["prepare.loom"], "libraries": []}},
  "services": {"chat": {"kind": "text", "version": 1}},
  "variants": {"small": {"partitions": {
    "node-0": {"services": {"chat": {"programs": ["source"],
      "parameters": {"model": {"files": ["first.irpa"]}}, "config": {"part": 0}}}},
    "other side": {"services": {"chat": {"programs": ["source"],
      "parameters": {"model": {"files": ["second.irpa"]}}, "config": {"part": 1}}}}
  }}},
  "presets": {"check": {"variant": "small", "service": "chat",
    "programs": "source", "partition": "node-0"}}
})json";

TEST_F(CardTest, NamedPartitionOwnsItsCompleteClosure) {
  IREE_ASSERT_OK(Parse(kPartitionCard));
  EXPECT_EQ(String(loom_model_card_selection(card_)->selector.partition),
            "node-0");
  const loom_model_selector_t selector = {
      .preset = IREE_SV("check"),
      .partition = IREE_SV("other side"),
  };
  std::string json = kPartitionCard;
  IREE_ASSERT_OK(Parse(json, &selector));
  json.assign(json.size(), 'x');
  const auto* selection = loom_model_card_selection(card_);
  EXPECT_EQ(String(selection->selector.partition), "other side");
  EXPECT_EQ(String(selection->config), R"({"part":1})");
  ASSERT_EQ(selection->parameter_count, 1u);
  ASSERT_EQ(selection->parameters[0].files.count, 1u);
  EXPECT_EQ(String(selection->parameters[0].files.values[0].path),
            "second.irpa");
}

TEST_F(CardTest, PartitionChoiceIsExplicitAndNeverFiltersAnUnpartitionedModel) {
  loom_model_selector_t selector = {.variant = IREE_SV("small")};
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse(kPartitionCard, &selector));
  selector.partition = IREE_SV("absent");
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        Parse(kPartitionCard, &selector));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(kCard, &selector));
  std::string mixed = kPartitionCard;
  mixed.insert(mixed.find("\"partitions\""), "\"services\":{},");
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(mixed));
}

TEST_F(CardTest, SelectionOwnsMixedProvidersAndScopedInputs) {
  std::string json = kCard;
  IREE_ASSERT_OK(Parse(json));
  json.assign(json.size(), 'x');
  const auto* selection = loom_model_card_selection(card_);
  EXPECT_EQ(String(selection->selector.preset), "check");
  EXPECT_EQ(String(selection->kind), "text");
  EXPECT_EQ(selection->version, 1u);
  EXPECT_EQ(String(selection->directory), "program");
  ASSERT_EQ(selection->inputs.count, 2u);
  EXPECT_EQ(String(selection->inputs.values[1].file.path),
            "program/control.loombc");
  ASSERT_EQ(selection->libraries.count, 1u);
  EXPECT_EQ(String(selection->libraries.values[0].file.format), "loombc");
  ASSERT_EQ(selection->libraries.values[0].data.count, 1u);
  EXPECT_EQ(String(selection->config),
            R"({"quantized":false,"width":8,"dtype":"f32"})");
  const auto* parameters =
      loom_model_selection_parameter(selection, IREE_SV("model"));
  ASSERT_NE(parameters, nullptr);
  ASSERT_EQ(parameters->files.count, 2u);
  EXPECT_EQ(String(parameters->files.values[1].format), "safetensors");
  EXPECT_EQ(loom_model_selection_parameter(selection, IREE_SV("missing")),
            nullptr);
  const auto* tokenizer =
      loom_model_selection_tokenizer(selection, IREE_SV("text"));
  ASSERT_NE(tokenizer, nullptr);
  EXPECT_EQ(String(tokenizer->file.path), "asset/vocabulary.data");
  EXPECT_EQ(String(tokenizer->file.format), "huggingface-json");
  EXPECT_EQ(String(selection->execution), R"({"rows": 2})");
}

TEST_F(CardTest, OriginalVoiceRecordingsOwnTheirNamesAndMediaTypes) {
  std::string json = Replace(
      "\"config\": {\"width\": 8, \"dtype\": \"f32\"},",
      R"("voices":{"one":{"file":"voice/original.data"},)"
      R"("two":{"file":{"path":"voice/raw.data","origin":"@recordings/source"},)"
      R"("content_type":"audio/pcm;rate=44100;channels=2;format=f32le"}},)");
  IREE_ASSERT_OK(Parse(json));
  json.assign(json.size(), 'x');
  const auto* selection = loom_model_card_selection(card_);
  ASSERT_EQ(selection->voice_count, 2u);
  EXPECT_EQ(String(selection->voices[0].name), "one");
  EXPECT_EQ(String(selection->voices[0].file.path), "voice/original.data");
  EXPECT_TRUE(iree_string_view_is_empty(selection->voices[0].content_type));
  EXPECT_EQ(String(selection->voices[1].name), "two");
  EXPECT_EQ(String(selection->voices[1].file.path), "voice/raw.data");
  EXPECT_EQ(String(selection->voices[1].content_type),
            "audio/pcm;rate=44100;channels=2;format=f32le");
}

TEST_F(CardTest, OriginalVoiceRecordingsRejectAmbiguousAndInvalidDeclarations) {
  for (const char* voice :
       {R"("original.wav")", R"({})", R"({"file":"../original.wav"})",
        R"({"file":{"path":"original.wav","format":"wav"}})",
        R"({"file":"original.data","content_type":""})",
        R"({"file":"original.data","prepared":true})"}) {
    const std::string declaration =
        "\"voices\":{\"narrator\":" + std::string(voice) + "},";
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        Parse(Replace("\"config\": {\"width\": 8, \"dtype\": \"f32\"},",
                      declaration.c_str())));
  }
}

TEST_F(CardTest, ComponentsOwnIndependentVersionsAndCanonicalSettings) {
  std::string json = Replace(
      "\"kind\": \"text\", \"version\": 1",
      R"("kind":"text","version":1,"components":{"image":{"version":3,"options":["patches"]},"audio":{}})");
  json.insert(
      json.find("\"execution\""),
      R"("components":{"image":{"options":{"patches":4},"execution":{"rows":2}}},)");
  IREE_ASSERT_OK(Parse(json));
  json.assign(json.size(), 'x');
  const auto* selection = loom_model_card_selection(card_);
  EXPECT_EQ(selection->version, 1u);
  ASSERT_EQ(selection->component_count, 2u);
  const auto* image =
      loom_model_selection_component(selection, IREE_SV("image"));
  ASSERT_NE(image, nullptr);
  EXPECT_EQ(String(image->export_prefix), "text_image");
  EXPECT_EQ(image->version, 3u);
  EXPECT_EQ(String(image->options), R"({"patches":4})");
  EXPECT_EQ(String(image->execution), R"({"rows":2})");
  const auto* audio =
      loom_model_selection_component(selection, IREE_SV("audio"));
  ASSERT_NE(audio, nullptr);
  EXPECT_EQ(audio->version, 0u);
  EXPECT_EQ(String(audio->options), "{}");
  EXPECT_EQ(loom_model_selection_component(selection, IREE_SV("missing")),
            nullptr);
}

TEST_F(CardTest, ComponentBoundaryRejectsInvalidDeclarationsAndSettings) {
  for (const char* component :
       {R"({"bad-name":{}})", R"({"image":3})", R"({"image":{"version":0}})",
        R"({"image":{"programs":["source"]}})"}) {
    std::string json = kCard;
    json.insert(json.find("\"kind\": \"text\""),
                std::string("\"components\":") + component + ",");
    EXPECT_FALSE(iree::Status(Parse(json)).ok());
    EXPECT_EQ(card_, nullptr);
  }
  std::string json = kCard;
  json.insert(json.find("\"execution\""), R"("components":{"image":{}},)");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(json));
  json.insert(json.find("\"kind\": \"text\""), R"("components":{"image":{}},)");
  IREE_ASSERT_OK(Parse(json));
  json.replace(json.find(R"("image":{})", json.find("\"presets\"")), 10,
               R"("image":{"options":{"patches":4}})");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(json));
}

TEST_F(CardTest, CompatibleOverrideRetainsVariantFacts) {
  const loom_model_selector_t selector = {
      .preset = IREE_SV("check"),
      .programs = IREE_SV("source"),
  };
  IREE_ASSERT_OK(Parse(kCard, &selector));
  const auto* selection = loom_model_card_selection(card_);
  EXPECT_EQ(String(selection->selector.programs), "source");
  EXPECT_EQ(selection->parameter_count, 1u);
  EXPECT_EQ(String(selection->config), R"({"width":8,"dtype":"f32"})");
}

TEST_F(CardTest, InformationalMemoryHasNoSchedulingAuthority) {
  IREE_ASSERT_OK(Parse(Replace(
      "\"execution\": {\"rows\": 2}",
      "\"memory\": {\"weights_bytes\": 180000000, "
      "\"workload\": \"One average request; runtime overhead excluded.\"}, "
      "\"execution\": {\"rows\": 2}")));
  const auto* selection = loom_model_card_selection(card_);
  EXPECT_EQ(String(selection->execution), R"({"rows": 2})");
  EXPECT_EQ(String(selection->config),
            R"({"quantized":false,"width":8,"dtype":"f32"})");
  ASSERT_EQ(selection->parameter_count, 1u);
}

TEST_F(CardTest, IncompatibleCanonicalProgramRequiresExplicitOverride) {
  const loom_model_selector_t selector = {
      .preset = IREE_SV("check"),
      .variant = IREE_SV("large"),
  };
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(kCard, &selector));
  EXPECT_EQ(card_, nullptr);
}

TEST_F(CardTest, ExplicitSelectionDoesNotSilentlyPickAPreset) {
  const loom_model_selector_t selector = {.variant = IREE_SV("small")};
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(kCard, &selector));
  EXPECT_EQ(card_, nullptr);
}

TEST_F(CardTest, AmbiguousPresetsRequireSelection) {
  const std::string json =
      Replace("\"presets\": {",
              "\"presets\": {\"other\": {\"service\": \"chat\", \"variant\": "
              "\"small\", "
              "\"programs\": \"mixed\", \"execution\": {\"rows\": 8}},");
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(json));
  const loom_model_selector_t selector = {.preset = IREE_SV("other")};
  IREE_ASSERT_OK(Parse(json, &selector));
  EXPECT_EQ(String(loom_model_card_selection(card_)->selector.preset), "other");
}

TEST_F(CardTest, UnknownRevisionFailsBeforeInterpretingNewFields) {
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      Parse(Replace("\"schema_version\": 1",
                    "\"schema_version\": 2, \"future_field\": {}")));
  EXPECT_EQ(card_, nullptr);
}

TEST_F(CardTest, CanonicalOptionsBelongToTheSelectedService) {
  std::string json =
      Replace("\"execution\": {\"rows\": 2}",
              "\"options\": {\"batch_size\": 4}, \"execution\": {\"rows\": 2}");
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(json));
  auto position = json.find("\"kind\": \"text\", \"version\": 1");
  json.insert(position, "\"options\": [\"batch_size\"], ");
  IREE_ASSERT_OK(Parse(json));
  EXPECT_EQ(String(loom_model_card_selection(card_)->options),
            R"({"batch_size": 4})");
}

TEST_F(CardTest, MissingVersionIsExplicitDevelopmentContract) {
  IREE_ASSERT_OK(Parse(
      Replace("\"kind\": \"text\", \"version\": 1", "\"kind\": \"text\"")));
  EXPECT_EQ(loom_model_card_selection(card_)->version, 0u);
}

TEST_F(CardTest, EscapedDuplicateKeysFail) {
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(Replace("\"width\": 8", "\"width\": 8, \"w\\u0069dth\": 9")));
}

TEST_F(CardTest, PathsRemainInsideMaterializationRoot) {
  for (const char* path :
       {"../prepare.loom", "/prepare.loom", "a/../b.loom", "a//b.loom", "a/",
        "a/./b.loom", "C:/a.loom", "a\\\\b.loom", "a\\u0000b.loom"}) {
    SCOPED_TRACE(path);
    IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Parse(Replace("program/prepare.loom", path)));
    EXPECT_EQ(card_, nullptr);
  }
}

TEST_F(CardTest, PlainJsonAndScalarConfigurationAreRequired) {
  for (const char* config : {"null", "[]", "{}", "1e999"}) {
    SCOPED_TRACE(config);
    IREE_ASSERT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        Parse(Replace("\"width\": 8",
                      (std::string("\"width\": ") + config).c_str())));
  }
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse(Replace("\"schema_version\": 1,",
                                      "\"schema_version\": 1, // comment\n")));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse(Replace("\"rows\": 2}", "\"rows\": 2,}")));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse(Replace("\"config\": {\"width\": 4",
                                      "\"config_typo\": {\"width\": 4")));
}

}  // namespace
