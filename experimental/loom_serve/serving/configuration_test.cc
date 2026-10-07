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

std::string Model(const std::string& name, const std::string& options = "") {
  return R"({"kind":"text","name":")" + name +
         R"(","source":"models/text","weights":"weights.gguf","tokenizer":"tokenizer.json")" +
         options + "}";
}

TEST(ConfigurationTest, IndependentModelsOwnDecodedStringsAndOptions) {
  std::string json =
      "{\"models\":[" + Model("scout") + "," +
      Model(
          R"(research\u0065r)",
          R"(,"rows":8,"context_capacity":262144,"pool_capacity":4096,"checkpoint_capacity":2,"prefill_capacity":128,"mtp_depth":3,"continuation_epochs":2,"pending_requests":17,"max_tokens":128)") +
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
  EXPECT_EQ(a.rows, 4u);
  EXPECT_EQ(a.context_capacity, 16384u);
  EXPECT_EQ(a.mtp_depth, 0u);
  EXPECT_EQ(b.rows, 8u);
  EXPECT_EQ(b.context_capacity, 262144u);
  EXPECT_EQ(b.pool_capacity, 4096u);
  EXPECT_EQ(b.checkpoint_capacity, 2u);
  EXPECT_EQ(b.prefill_capacity, 128u);
  EXPECT_EQ(b.mtp_depth, 3u);
  EXPECT_EQ(b.continuation_epochs, 2u);
  EXPECT_EQ(b.pending_requests, 17u);
  EXPECT_EQ(b.max_tokens, 128u);
  loom_serve_configuration_deinitialize(&configuration);
}

TEST(ConfigurationTest,
     RejectsAmbiguousOrUnsupportedCatalogsAndCleansPartialState) {
  for (const auto& json :
       {std::string("{}"),
        std::string(R"({"models":[]})"),
        std::string(R"({"models":[null]})"),
        std::string(R"({"models":[{"kind":"image"}]})"),
        std::string(R"({"models":[{"kind":"text","name":"x"}]})"),
        "{\"models\":[" + Model("same") + "," + Model("same") + "]}",
        "{\"models\":[" + Model(R"(bad\u0000name)") + "]}",
        "{\"models\":[" + Model("x", R"(,"rows":0)") + "]}",
        "{\"models\":[" + Model("x", R"(,"rows":17)") + "]}",
        "{\"models\":[" + Model("x", R"(,"rows":2,"rows":4)") + "]}",
        "{\"models\":[" + Model("x", R"(,"mtp_depth":2)") + "]}",
        "{\"models\":[" + Model("x", R"(,"continuation_epochs":2)") + "]}",
        "{\"models\":[" + Model("x", R"(,"prefill_capacity":2,"mtp_depth":3)") +
            "]}",
        "{\"models\":[" + Model("x", R"(,"prefill_capacity":1024)") + "]}",
        "{\"models\":[" + Model("x", R"(,"pool_capacity":0)") + "]}",
        "{\"models\":[" + Model("x", R"(,"pending_requests":-1)") + "]}",
        "{\"models\":[" + Model("x", R"(,"max_tokens":1.5)") + "]}",
        "{\"models\":[" + Model("x", R"(,"typo":true)") + "]}",
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

}  // namespace
