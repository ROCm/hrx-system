// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>
#include <filesystem>

#include "loom/serve/service/text/image_service_test_fixture.h"

namespace {

using loom_serve::testing::text::ImageTextServiceTest;
using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

TEST_F(ImageTextServiceTest, InputFailureAndDisconnectAreRequestLocal) {
  const std::string invalid_image =
      ImageMessage("a", "image/png", "bm90LWltYWdl", "low", "b");
  const std::string response = CompleteRequest(Request(invalid_image, 2));
  EXPECT_EQ(response.find("HTTP/1.1 400"), 0u) << response;
  EXPECT_EQ(interface_.telemetry.counters->rejected, 1u);
  EXPECT_EQ(interface_.telemetry.counters->failed, 0u);

  const std::string image = ImageMessage("a", "image/png", kPng, "low", "b");
  const int client = Send(Request(image, 2));
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(client, interface_, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    return loom_serve_finite_component_phase(image_component_) !=
           LOOM_SERVE_FINITE_COMPONENT_DECLARED;
  }));
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(client, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
            0);
  ASSERT_EQ(close(client), 0);
  iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* self) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(self));
      },
      connection, iree_infinite_timeout());
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    return !interface_.is_pending(interface_.self) &&
           !interface_.is_active(interface_.self) &&
           !interface_.snapshot(interface_.self).queued_requests;
  }));
  EXPECT_EQ(interface_.telemetry.counters->rejected, 1u);
  EXPECT_EQ(interface_.telemetry.counters->cancelled, 1u);
  EXPECT_EQ(interface_.telemetry.counters->failed, 0u);
  EXPECT_FALSE(loom_serve_finite_component_is_pending(image_component_));
  EXPECT_EQ(Content(CompleteRequest(Request(Message("user", "abxa"), 2))),
            Oracle::Complete("abxa", 2));
}

class ManagedImageTextServiceTest : public SourceTextServiceTest {};

TEST_F(ManagedImageTextServiceTest, LazilyProjectsAnImageThroughTheRouter) {
  StartManaged();
  const char* source_path = std::getenv("LOOM_SERVE_TEST_PROJECTOR_SOURCE");
  ASSERT_NE(source_path, nullptr);
  const std::filesystem::path projector_directory =
      std::filesystem::absolute(source_path).parent_path();
  const std::string image_fields =
      ",\"image\":{\"prepare\":\"" +
      (projector_directory / "prepare.loom").string() + "\",\"control\":\"" +
      (projector_directory / "control.loom").string() + "\",\"assets\":\"" +
      projector_directory.string() + "\"}";
  std::string declaration =
      Declaration("fixture-text", directory_, "open", image_fields);
  const std::string source_list = "\"sources\":[\"advance.loom\"]";
  const size_t source_position = declaration.find(source_list);
  ASSERT_NE(source_position, std::string::npos);
  declaration.replace(source_position, source_list.size(),
                      "\"sources\":[\"advance.loom\",\"" +
                          (projector_directory / "project.loom").string() +
                          "\"]");
  const std::string accepted =
      Read(Send(declaration, "", "/v1/models/register"));
  ASSERT_EQ(accepted.find("HTTP/1.1 202"), 0u) << accepted;
  const std::string ready =
      WaitModel("fixture-text", R"("registration":"ready")");
  EXPECT_NE(ready.find(R"("input_modalities":["text","image"])"),
            std::string::npos)
      << ready;

  Oracle oracle;
  oracle.Consume(Oracle::Id('a'));
  oracle.Consume(255);
  oracle.Consume(Oracle::Id('b'));
  const std::string message = ImageMessage("a", "image/png", kPng, "low", "b");
  EXPECT_EQ(Content(Read(Send(Request(message, 3)))), oracle.Generate(3));

  const std::string removed =
      Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/unregister"));
  ASSERT_EQ(removed.find("HTTP/1.1 202"), 0u) << removed;
  WaitModel("fixture-text", {});
}

}  // namespace
