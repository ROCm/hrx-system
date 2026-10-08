// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/decision/protocol.h"

#include <limits>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class DecisionProtocolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_allocator_malloc(allocator_, protocol_->request_size,
                                         reinterpret_cast<void**>(&request_)));
    iree_string_builder_initialize(allocator_, &output_);
  }
  void TearDown() override {
    protocol_->deinitialize(request_);
    iree_allocator_free(allocator_, request_);
    iree_string_builder_deinitialize(&output_);
  }
  iree_status_t Parse(const std::string& json) {
    protocol_->deinitialize(request_);
    return protocol_->initialize(
        iree_make_string_view(json.data(), json.size()), request_, allocator_);
  }
  iree_status_t Encode(const std::vector<std::vector<float>>& probabilities) {
    std::vector<std::vector<uint8_t>> storage(probabilities.size());
    std::vector<iree_const_byte_span_t> rows;
    for (size_t i = 0; i < probabilities.size(); ++i) {
      storage[i].resize(probabilities[i].size() * 4);
      for (size_t j = 0; j < probabilities[i].size(); ++j) {
        iree_unaligned_store_le_f32(storage[i].data() + j * 4,
                                    probabilities[i][j]);
      }
      rows.push_back(
          iree_make_const_byte_span(storage[i].data(), storage[i].size()));
    }
    iree_string_builder_reset(&output_);
    return protocol_->response(request_, IREE_SV("fixture"), rows.data(), 17,
                               &output_);
  }
  std::string Output() const {
    const auto view = iree_string_builder_view(&output_);
    return {view.data, view.size};
  }
  // Same native descriptor used by the production finite service.
  const loom_serve_finite_protocol_t* protocol_ =
      loom_serve_decision_protocol();
  // Host allocator owning concrete protocol storage.
  const iree_allocator_t allocator_ = iree_allocator_system();
  // Complete request allocation; concrete protocol size comes from the
  // descriptor.
  loom_serve_finite_request_t* request_ = nullptr;
  // Encoded public response.
  iree_string_builder_t output_;
};

TEST_F(DecisionProtocolTest, PreservesOrderedQuestionsAndStructuredInput) {
  std::string json =
      R"({"model":"fixture","state":{"turn":["a",2]},"questions":{
    "z":{"type":"choice","instructions":["route"],"criteria":{"z":null,"a":{"meaning":"second"}}},
    "s":{"type":"score","instructions":{"q":"score"},"criteria":["low",["mid"],{"level":"high"}]},
    "\u03bb":{"type":"noul","instructions":"yes?","criteria":{"true":["yes"]}}
  }})";
  IREE_ASSERT_OK(Parse(json));
  json.assign(json.size(), 'x');
  ASSERT_EQ(request_->count, 3u);
  iree_const_byte_span_t bytes;
  IREE_ASSERT_OK(iree_vm_buffer_map_read(
      request_->input, 0, iree_vm_buffer_length(request_->input), &bytes));
  EXPECT_EQ(iree_vm_buffer_access(request_->input),
            IREE_VM_BUFFER_ACCESS_FLAG_READ);
  EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data + 16), 2u);  // State object.
  EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data + 24), 3u);
  const uint64_t state = iree_unaligned_load_le_u64(bytes.data);
  const uint64_t length = iree_unaligned_load_le_u64(bytes.data + 8);
  EXPECT_EQ(
      std::string(reinterpret_cast<const char*>(bytes.data + state), length),
      R"({"turn":["a",2]})");
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data + 32 + i * 64 + 16), i);
  }
  IREE_ASSERT_OK(Encode({{0.25f, 0.75f}, {0, 0.5f, 0.5f}, {0.75f, 0.25f}}));
  EXPECT_EQ(
      Output(),
      R"({"model":"fixture","answers":{"z":{"type":"choice","choice":"a","confidence":0.5,"probabilities":{"z":0.25,"a":0.75}},"s":{"type":"score","score":1.5,"legend":{"0":"low","1":["mid"],"2":{"level":"high"}},"confidence":0.25,"probabilities":{"0":0,"1":0.5,"2":0.5}},"λ":{"type":"noul","noul":0.25}},"usage":{"input_tokens":17,"output_tokens":0}})");
}

TEST_F(DecisionProtocolTest, DefinesUniformTieAndSingletonAnswers) {
  IREE_ASSERT_OK(Parse(R"({"model":"m","state":"a","questions":{
    "q":{"type":"ch\u006fice","instructions":"?","criteria":{"z":null,"a":null}},
    "one":{"type":"choice","instructions":"?","criteria":{"only":null}},
    "s":{"type":"score","instructions":"?","criteria":["a","b"]}
  }})"));
  IREE_ASSERT_OK(Encode({{0.5f, 0.5f}, {1.0f}, {0.5f, 0.5f}}));
  EXPECT_EQ(
      Output(),
      R"({"model":"fixture","answers":{"q":{"type":"choice","choice":"z","confidence":0,"probabilities":{"z":0.5,"a":0.5}},"one":{"type":"choice","choice":"only","confidence":1,"probabilities":{"only":1}},"s":{"type":"score","score":0.5,"legend":{"0":"a","1":"b"},"confidence":0,"probabilities":{"0":0.5,"1":0.5}}},"usage":{"input_tokens":17,"output_tokens":0}})");
  IREE_ASSERT_OK(Encode({{0, 1}, {1}, {0, 1}}));
  EXPECT_NE(
      Output().find(R"("score":1,"legend":{"0":"a","1":"b"},"confidence":1)"),
      std::string::npos);
}

TEST_F(DecisionProtocolTest, RejectsMalformedSchemaAndDecodedDuplicateIds) {
  for (
      const char* questions :
      {R"({})", R"([])", R"({"q":null})", R"({"q":{"instructions":"x"}})",
       R"({"q":{"type":"noul"}})",
       R"({"q":{"type":"decode","instructions":"x"}})",
       R"({"q":{"type":"noul","instructions":false}})",
       R"({"q":{"type":"noul","instructions":"x","extra":0}})",
       R"({"q":{"type":"noul","type":"noul","instructions":"x"}})",
       R"({"q":{"type":"noul","instructions":"x","criteria":{"yes":"x"}}})",
       R"({"q":{"type":"choice","instructions":"x","criteria":{}}})",
       R"({"q":{"type":"choice","instructions":"x","criteria":{"a":3}}})",
       R"({"q":{"type":"choice","instructions":"x","criteria":{"a":null,"\u0061":null}}})",
       R"({"q":{"type":"score","instructions":"x","criteria":["one"]}})",
       R"({"q":{"type":"score","instructions":"x","criteria":[null,"two"]}})",
       R"({"q":{"type":"noul","instructions":"x"},"\u0071":{"type":"noul","instructions":"y"}})",
       R"({"":{"type":"noul","instructions":"x"}})",
       R"({"\ud800":{"type":"noul","instructions":"x"}})"}) {
    SCOPED_TRACE(questions);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        Parse(std::string("{\"model\":\"m\",\"state\":\"a\",\"questions\":") +
              questions + "}"));
  }
  for (
      const char* json :
      {R"({"model":"m","state":1,"questions":{"q":{"type":"noul","instructions":"x"}}})",
       R"({"model":"m","state":"a","questions":{"q":{"type":"noul","instructions":"x"}},"images":[]})",
       R"({"model":"m","state":"a","questions":{"q":{"type":"noul","instructions":"x"}}} {})"}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(json));
  }
}

TEST_F(DecisionProtocolTest, CoversQuestionAndOptionCountBoundaries) {
  const std::string prefix =
      R"({"model":"m","state":[],"questions":{"q":{"type":"score","instructions":[],"criteria":[)";
  std::string levels;
  for (size_t i = 0; i < 11; ++i) {
    levels += i ? ",\"level\"" : "\"level\"";
    if (i == 9) {
      IREE_ASSERT_OK(Parse(prefix + levels + "]}}}"));
    }
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse(prefix + levels + "]}}}"));
  std::string criteria;
  for (size_t i = 0; i < 256; ++i) {
    if (i) {
      criteria += ",";
    }
    criteria += "\"o" + std::to_string(i) + "\":null";
    const std::string json =
        R"({"model":"m","state":{},"questions":{"q":{"type":"choice","instructions":"x","criteria":{)" +
        criteria + "}}}}";
    if (i == 254) {
      IREE_ASSERT_OK(Parse(json));
    }
    if (i == 255) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(json));
    }
  }
}

TEST_F(DecisionProtocolTest, RejectsBrokenHeadWithoutRenormalizing) {
  IREE_ASSERT_OK(Parse(
      R"({"model":"m","state":"a","questions":{"q":{"type":"noul","instructions":"?"}}})"));
  for (const std::vector<float>& row :
       {std::vector<float>{1},
        {0.3f, 0.3f},
        {-0.1f, 1.1f},
        {0.5f, std::numeric_limits<float>::quiet_NaN()},
        {0.5f, std::numeric_limits<float>::infinity()}}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Encode({row}));
  }
}

TEST_F(DecisionProtocolTest, KeepsStructuredLegendsStrictJson) {
  for (const char* criterion :
       {R"({"x":1,})", R"({/*comment*/"x":1})", R"(["x",])", R"("\ud800")"}) {
    const std::string json =
        std::string(
            R"({"model":"m","state":"a","questions":{"q":{"type":"score","instructions":"?","criteria":["low",)") +
        criterion + "]}}}";
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(json));
  }
  IREE_ASSERT_OK(Parse(
      R"({"model":"m","state":"/* ordinary text */","questions":{"q":{"type":"score","instructions":"?","criteria":["comma,]",{"url":"https://example.test"}]}}})"));
  IREE_ASSERT_OK(Encode({{0.5f, 0.5f}}));
  EXPECT_NE(
      Output().find(
          R"("legend":{"0":"comma,]","1":{"url":"https://example.test"}})"),
      std::string::npos);
}

TEST_F(DecisionProtocolTest, BoundsSchemaBeforeSourceAdmission) {
  std::string questions;
  for (size_t i = 0; i < 1025; ++i) {
    if (i) {
      questions += ",";
    }
    questions +=
        "\"q" + std::to_string(i) + R"(":{"type":"noul","instructions":"?"})";
    if (i < 1023) {
      continue;
    }
    const std::string json =
        R"({"model":"m","state":"a","questions":{)" + questions + "}}";
    if (i == 1023) {
      IREE_ASSERT_OK(Parse(json));
      EXPECT_EQ(request_->count, 1024u);
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(json));
    }
  }
}

}  // namespace
