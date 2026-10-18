// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/recognition/transcription.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

loom_serve_http_part_t Field(iree_string_view_t name,
                             iree_string_view_t contents) {
  return {.name = name, .contents = contents};
}

class TranscriptionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_serve_recognition_document_initialize(4096, iree_allocator_system(),
                                               &document_);
    iree_string_builder_initialize(iree_allocator_system(), &output_);
  }
  void TearDown() override {
    iree_string_builder_deinitialize(&output_);
    loom_serve_recognition_document_deinitialize(&document_);
  }
  iree_status_t Parse(const std::vector<loom_serve_http_part_t>& fields) {
    return loom_serve_transcription_parse({fields.size(), fields.data()},
                                          IREE_SV("recognition"), {16000, 1},
                                          &request_);
  }
  iree_status_t Apply(uint64_t segment, uint64_t begin,
                      iree_string_view_t text) {
    const loom_serve_recognition_transcript_t transcript = {
        .segment = segment,
        .revision = 1,
        .begin = begin,
        .end = begin + 160,
        .flags = LOOM_SERVE_RECOGNITION_TRANSCRIPT_FINAL,
        .text = text,
    };
    const loom_serve_recognition_transcript_t* retained = nullptr;
    return loom_serve_recognition_document_apply(&document_, &transcript,
                                                 &retained);
  }
  std::string Output() {
    const auto view = iree_string_builder_view(&output_);
    return std::string(view.data, view.size);
  }

  // Parsed views borrow each test's input parts.
  loom_serve_transcription_t request_ = {};
  // Latest source-checked hypotheses to format.
  loom_serve_recognition_document_t document_ = {};
  // Response body accumulator.
  iree_string_builder_t output_ = {};
};

TEST_F(TranscriptionTest, BorrowsAudioAndSourceOptionsIndependentOfPartOrder) {
  const char audio[] = {'\0', '\xff', '\x7f', '\0'};
  auto file = Field(IREE_SV("file"), {audio, sizeof(audio)});
  file.filename = IREE_SV("../../untrusted.wav");
  file.content_type = IREE_SV("audio/pcm;rate=16000;channels=1");
  IREE_ASSERT_OK(Parse({file, Field(IREE_SV("prompt"), IREE_SV("Hello, 世界")),
                        Field(IREE_SV("temperature"), IREE_SV("0.75")),
                        Field(IREE_SV("language"), IREE_SV("en")),
                        Field(IREE_SV("model"), IREE_SV("recognition")),
                        Field(IREE_SV("response_format"), IREE_SV("text")),
                        Field(IREE_SV("stream"), IREE_SV("false"))}));
  EXPECT_EQ(request_.audio.data, reinterpret_cast<const uint8_t*>(audio));
  EXPECT_EQ(request_.audio.data_length, sizeof(audio));
  EXPECT_EQ(request_.pcm_sample_rate, 16000u);
  EXPECT_EQ(request_.normalized_frame_count, 2u);
  EXPECT_TRUE(request_.text_response);
  EXPECT_EQ(request_.request.outputs, LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT);
  EXPECT_FLOAT_EQ(request_.request.temperature, 0.75f);
  EXPECT_TRUE(iree_string_view_equal(request_.request.language, IREE_SV("en")));
  EXPECT_TRUE(
      iree_string_view_equal(request_.request.prompt, IREE_SV("Hello, 世界")));

  IREE_ASSERT_OK(
      Parse({Field(IREE_SV("model"), IREE_SV("recognition")), file}));
  EXPECT_FALSE(request_.text_response);
  EXPECT_EQ(request_.request.outputs, LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT);
  EXPECT_FLOAT_EQ(request_.request.temperature, 0.0f);
  EXPECT_EQ(request_.request.language.size, 0u);
  EXPECT_EQ(request_.request.prompt.size, 0u);
}

TEST_F(TranscriptionTest,
       RawPcmRequiresExplicitRateMatchingChannelsAndWholeFrames) {
  const char audio[] = {'\0', '\x20', '\0', '\xe0'};
  auto file = Field(IREE_SV("file"), {audio, sizeof(audio)});
  const auto model = Field(IREE_SV("model"), IREE_SV("recognition"));
  for (const auto content_type : {
           IREE_SV("audio/pcm;rate=16000;channels=1"),
           IREE_SV("Audio/PCM; CHANNELS=\"1\"; RATE=\"16000\""),
           IREE_SV("audio/pcm;rate=\"\\16000\";channels=1"),
       }) {
    file.content_type = content_type;
    IREE_ASSERT_OK(Parse({file, model}));
    EXPECT_EQ(request_.audio.data,
              reinterpret_cast<const uint8_t*>(file.contents.data));
    EXPECT_EQ(request_.pcm_sample_rate, 16000u);
    EXPECT_EQ(request_.normalized_frame_count, 2u);
  }
  for (const auto content_type : {
           IREE_SV("audio/pcm"),
           IREE_SV("audio/pcm;rate=16000"),
           IREE_SV("audio/pcm;channels=1"),
           IREE_SV("audio/pcm;rate=16000;channels=2"),
           IREE_SV("audio/pcm;rate=0;channels=1"),
           IREE_SV("audio/pcm;rate=16000;channels=-1"),
           IREE_SV("audio/pcm;rate=4294983296;channels=1"),
           IREE_SV("audio/pcm;rate=16e3;channels=1"),
           IREE_SV("audio/pcm;rate=16000;channels=1;RATE=16000"),
           IREE_SV("audio/pcm;rate=16000;channels=1;format=f32le"),
       }) {
    file.content_type = content_type;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse({file, model}));
    EXPECT_EQ(request_.audio.data, nullptr);
  }
  file.content_type = IREE_SV("audio/pcm;rate=16000;channels=1");
  file.contents.size = 3;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse({file, model}));

  // Whole samples are insufficient when the model requires a stereo frame.
  file.content_type = IREE_SV("audio/pcm;rate=16000;channels=2");
  file.contents.size = sizeof(audio);
  loom_serve_http_part_t parts[] = {file, model};
  IREE_ASSERT_OK(loom_serve_transcription_parse(
      {2, parts}, IREE_SV("recognition"), {16000, 2}, &request_));
  parts[0].contents.size = 2;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_transcription_parse({2, parts}, IREE_SV("recognition"),
                                     {16000, 2}, &request_));
}

TEST_F(TranscriptionTest, RawPcmRetainsExactNormalizedDuration) {
  const auto model = Field(IREE_SV("model"), IREE_SV("recognition"));
  for (uint32_t rate : {8000u, 16000u, 16001u, 22050u, 24000u, 44100u, 48000u,
                        96000u, 192000u, UINT32_MAX}) {
    const std::string content_type =
        "audio/pcm;rate=" + std::to_string(rate) + ";channels=1";
    for (size_t frames : {1, 2, 3, 27, 997}) {
      SCOPED_TRACE(::testing::Message()
                   << rate << " Hz, " << frames << " frames");
      const std::string audio(frames * sizeof(int16_t), '\0');
      auto file = Field(IREE_SV("file"), {audio.data(), audio.size()});
      file.content_type = {content_type.data(), content_type.size()};
      IREE_ASSERT_OK(Parse({file, model}));
      EXPECT_EQ(request_.pcm_sample_rate, rate);
      EXPECT_EQ(request_.normalized_frame_count,
                (uint64_t{frames} * 16000 + rate - 1) / rate);
    }
  }
}

TEST_F(TranscriptionTest, EncodedAudioDefersFormatSelectionToContainerBytes) {
  auto file = Field(IREE_SV("file"), IREE_SV("opaque container bytes"));
  file.filename = IREE_SV("../../not-a-format.pcm");
  const auto model = Field(IREE_SV("model"), IREE_SV("recognition"));
  for (const auto content_type : {
           IREE_SV(""),
           IREE_SV("audio/wav"),
           IREE_SV("audio/mpeg"),
           IREE_SV("audio/flac"),
           IREE_SV("application/octet-stream"),
       }) {
    file.content_type = content_type;
    IREE_ASSERT_OK(Parse({file, model}));
    EXPECT_EQ(request_.pcm_sample_rate, 0u);
    EXPECT_EQ(request_.normalized_frame_count, 0u);
    EXPECT_EQ(request_.audio.data,
              reinterpret_cast<const uint8_t*>(file.contents.data));
    EXPECT_EQ(request_.audio.data_length, file.contents.size);
  }
}

TEST_F(TranscriptionTest, RejectsAmbiguousOrUnsupportedOptionsBeforeModelWork) {
  auto file = Field(IREE_SV("file"), IREE_SV("pcm16le!"));
  file.content_type = IREE_SV("audio/pcm;rate=16000;channels=1");
  const std::vector<loom_serve_http_part_t> required = {
      Field(IREE_SV("model"), IREE_SV("recognition")), file};
  IREE_ASSERT_OK(Parse(required));
  for (const auto& option : {
           Field(IREE_SV("model"), IREE_SV("recognition")),
           Field(IREE_SV("file"), IREE_SV("duplicate")),
           Field(IREE_SV("unknown"), IREE_SV("")),
           Field(IREE_SV("language"), IREE_SV("\xff")),
           Field(IREE_SV("temperature"), IREE_SV("nan")),
           Field(IREE_SV("temperature"), IREE_SV("-0.1")),
           Field(IREE_SV("temperature"), IREE_SV("1.1")),
           Field(IREE_SV("temperature"), {}),
           Field(IREE_SV("stream"), IREE_SV("true")),
           Field(IREE_SV("stream"), {}),
           Field(IREE_SV("response_format"), IREE_SV("verbose_json")),
           Field(IREE_SV("response_format"), {}),
       }) {
    auto parts = required;
    parts.push_back(option);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(parts));
    EXPECT_EQ(request_.audio.data, nullptr);
  }
  auto parts = required;
  parts.push_back(Field(IREE_SV("prompt"), {}));
  parts.push_back(Field(IREE_SV("prompt"), IREE_SV("second")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(parts));
  parts = required;
  parts[0].filename = IREE_SV("");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(parts));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse({required[0]}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse({required[1]}));
  parts = required;
  parts[1].contents = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(parts));
}

TEST_F(TranscriptionTest, ChronologyAndEscapingPreserveCommittedText) {
  IREE_ASSERT_OK(Apply(3, 400, IREE_SV("last\nline")));
  IREE_ASSERT_OK(Apply(7, 200, IREE_SV("same time")));
  IREE_ASSERT_OK(Apply(5, 200, IREE_SV("\"quoted\"\\text")));
  IREE_ASSERT_OK(Apply(8, 0, IREE_SV("Hello, 世界")));
  IREE_ASSERT_OK(Apply(9, 100, IREE_SV("")));
  IREE_ASSERT_OK(loom_serve_recognition_document_complete(&document_));
  IREE_ASSERT_OK(loom_serve_transcription_format(&document_, false, &output_));
  EXPECT_EQ(Output(),
            "{\"text\":\"Hello, 世界\\n\\\"quoted\\\"\\\\text\\nsame "
            "time\\nlast\\nline\"}");
  iree_string_builder_reset(&output_);
  IREE_ASSERT_OK(loom_serve_transcription_format(&document_, true, &output_));
  EXPECT_EQ(Output(), "Hello, 世界\n\"quoted\"\\text\nsame time\nlast\nline");
}

TEST_F(TranscriptionTest, EmptyCompletedDocumentHasAnEmptyTranscript) {
  IREE_ASSERT_OK(loom_serve_recognition_document_complete(&document_));
  IREE_ASSERT_OK(loom_serve_transcription_format(&document_, false, &output_));
  EXPECT_EQ(Output(), R"({"text":""})");
  iree_string_builder_reset(&output_);
  IREE_ASSERT_OK(loom_serve_transcription_format(&document_, true, &output_));
  EXPECT_EQ(Output(), "");
}

}  // namespace
