// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/component.h"
#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

TEST_F(SourceTextServiceTest, TextRegistrationDoesNotPrepareItsImageComponent) {
  StartManaged();
  const auto declaration = Declaration(
      "fixture-text", directory_, "open",
      R"(,"image":{"prepare":"missing-vision/prepare.loombc","control":"missing-vision/control.loom","assets":"missing-vision"})");
  ASSERT_EQ(
      Read(Send(declaration, "", "/v1/models/register")).find("HTTP/1.1 202"),
      0u);
  const auto prepared = WaitModel("fixture-text", R"("registration":"ready")");
  EXPECT_NE(prepared.find(R"("parameters":{"active":false)"), std::string::npos)
      << prepared;
  ASSERT_EQ(Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/drain"))
                .find("HTTP/1.1 200"),
            0u);
  const auto activation =
      Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/activate"));
  ASSERT_EQ(activation.find("HTTP/1.1 202"), 0u) << activation;
  const auto active =
      WaitModel("fixture-text", R"("parameter_operation":"idle")");
  EXPECT_NE(active.find(R"("parameters":{"active":true)"), std::string::npos)
      << active;
  ASSERT_EQ(Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/undrain"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Content(Read(Send(Request(Message("user", "abxa"), 2)))),
            Oracle::Complete("abxa", 2));
  ASSERT_EQ(
      Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/unregister"))
          .find("HTTP/1.1 202"),
      0u);
  WaitModel("fixture-text", {});
  // No image root/checkpoint was available at any point, including removal.
  EXPECT_FALSE(http_.failed());
}

TEST_F(SourceTextServiceTest, FailedTextPreparationRetiresItsDormantImage) {
  StartManaged();
  const auto declaration = Declaration(
      "fixture-text", directory_, "open",
      R"(,"image":{"prepare":"missing-vision/prepare.loom","control":"missing-vision/control.loom","assets":"missing-vision"})");
  auto broken = declaration;
  broken.replace(broken.find(weights_), weights_.size(), weights_ + ".missing");
  ASSERT_EQ(Read(Send(broken, "", "/v1/models/register")).find("HTTP/1.1 202"),
            0u);
  const auto failed = WaitModel("fixture-text", R"("registration":"failed")");
  EXPECT_NE(failed.find(".missing"), std::string::npos) << failed;
  ASSERT_EQ(
      Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/unregister"))
          .find("HTTP/1.1 202"),
      0u);
  WaitModel("fixture-text", {});
  ASSERT_EQ(
      Read(Send(declaration, "", "/v1/models/register")).find("HTTP/1.1 202"),
      0u);
  EXPECT_EQ(Content(Read(Send(Request(Message("user", "abxa"), 2)))),
            Oracle::Complete("abxa", 2));
}

// The media consumer starts demand; the existing text owner must advance and
// join it without treating image preparation as text activation or failure.
class TextComponentServiceTest : public SourceTextServiceTest {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
    auto sources = document_.value().sources;
    sources.prepare = IREE_SV("missing-vision/prepare.loom");
    sources.control = IREE_SV("missing-vision/control.loom");
    const loom_serve_finite_model_options_t options = {
        .sources = sources,
        .asset_directory = IREE_SV("missing-vision"),
        .options = IREE_SV("{}"),
    };
    const loom_serve_wake_callback_t observer = {
        [](void* user_data) {
          iree_notification_post(static_cast<iree_notification_t*>(user_data),
                                 IREE_ALL_WAITERS);
        },
        loom_serve_http_server_notification(server_)};
    IREE_ASSERT_OK(loom_serve_finite_component_allocate(
        device_, &options, observer, host_allocator_, &image_));
    auto service_options = ServiceOptions();
    service_options.image = image_;
    IREE_ASSERT_OK(loom_serve_text_service_allocate(
        model_, &service_options, nullptr, host_allocator_, &service_));
    interface_ = loom_serve_text_service_interface(service_);
    Observe(interface_);
  }

  void TearDown() override {
    if (service_) {
      AdvanceUntil(
          interface_, [&] { return !interface_.is_pending(interface_.self); },
          LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING);
      interface_.set_observer(interface_.self, {});
      loom_serve_text_service_free(service_, IREE_STATUS_OK);
    }
    if (image_) {
      bool complete = false;
      IREE_EXPECT_OK(loom_serve_finite_component_retire(image_, &complete));
      EXPECT_TRUE(complete);
      loom_serve_finite_component_free(image_);
    }
    SourceTextServiceTest::TearDown();
  }

  void BeginImagePreparation() {
    loom_serve_finite_model_t* model = nullptr;
    IREE_ASSERT_OK(loom_serve_finite_component_prepare(image_, &model));
    EXPECT_EQ(model, nullptr);
    EXPECT_TRUE(loom_serve_finite_component_is_pending(image_));
    EXPECT_TRUE(interface_.is_pending(interface_.self));
    EXPECT_FALSE(interface_.is_active(interface_.self));
  }

  void CompleteTextRequest() {
    const int client = Send(Request(Message("user", "abxa"), 2));
    loom_serve_http_connection_t* connection = nullptr;
    ASSERT_NO_FATAL_FAILURE(Accept(client, interface_, &connection));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
      return !interface_.is_pending(interface_.self) &&
             !interface_.is_active(interface_.self) &&
             !interface_.snapshot(interface_.self).queued_requests;
    }));
    EXPECT_EQ(Content(Read(client)), Oracle::Complete("abxa", 2));
  }

  // Owned optional computation, outliving the service that borrows it.
  loom_serve_finite_component_t* image_ = nullptr;
  // Owned text service, released only after both preparation domains join.
  loom_serve_text_service_t* service_ = nullptr;
  // Borrowed callbacks invoked by this serialized test owner.
  loom_serve_http_service_t interface_ = {};
};

TEST_F(TextComponentServiceTest, OrdinaryTextLeavesImageDormant) {
  ASSERT_NO_FATAL_FAILURE(CompleteTextRequest());
  EXPECT_EQ(loom_serve_finite_component_phase(image_),
            LOOM_SERVE_FINITE_COMPONENT_DECLARED);
  EXPECT_EQ(loom_serve_finite_component_report(image_)->commands, 0u);
  EXPECT_TRUE(
      loom_serve_residency_snapshot(loom_serve_text_model_residency(model_))
          .active);
}

TEST_F(TextComponentServiceTest, ImageFailureDoesNotFailConcurrentText) {
  ASSERT_NO_FATAL_FAILURE(BeginImagePreparation());
  ASSERT_NO_FATAL_FAILURE(CompleteTextRequest());
  EXPECT_EQ(loom_serve_finite_component_phase(image_),
            LOOM_SERVE_FINITE_COMPONENT_FAILED);
  loom_serve_finite_model_t* model = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_serve_finite_component_prepare(image_, &model));
  EXPECT_EQ(interface_.telemetry.counters->failed, 0u);
  ASSERT_NO_FATAL_FAILURE(CompleteTextRequest());
}

TEST_F(TextComponentServiceTest, StoppingJoinsImagePreparation) {
  ASSERT_NO_FATAL_FAILURE(BeginImagePreparation());
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      interface_, [&] { return !interface_.is_pending(interface_.self); },
      LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING));
  EXPECT_FALSE(loom_serve_finite_component_is_pending(image_));
  EXPECT_EQ(loom_serve_finite_component_report(image_)->commands, 0u);
}

}  // namespace
