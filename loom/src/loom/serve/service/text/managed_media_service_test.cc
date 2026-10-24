// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>

#include "iree/base/internal/json.h"
#include "loom/serve/media/image_codec.h"
#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::OwnerTurn;
using loom_serve::testing::text::RegisteredSourceTextServiceTest;

std::string Field(const std::string& object, const char* name) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_lookup_object_value(
      iree_make_string_view(object.data(), object.size()),
      iree_make_cstring_view(name), &value));
  return std::string(value.data ? value.data : "", value.size);
}

bool HasField(const std::string& object, const char* name) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_try_lookup_object_value(
      iree_make_string_view(object.data(), object.size()),
      iree_make_cstring_view(name), &value));
  return !iree_string_view_is_empty(value);
}

class ManagedMediaServiceTest : public RegisteredSourceTextServiceTest,
                                public ::testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(RegisteredSourceTextServiceTest::SetUp());
    const char* source = std::getenv("LOOM_SERVE_TEST_PROJECTOR_SOURCE");
    ASSERT_NE(source, nullptr);
    projector_directory_ =
        std::filesystem::absolute(source).parent_path().string();
    assets_ = iree::testing::MakeTempFilePath("managed_projector");
    ASSERT_TRUE(std::filesystem::create_directory(assets_));
    ASSERT_TRUE(
        std::filesystem::copy_file(weights_, assets_ + "/weights.safetensors"));
    ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/weights.loom",
                                           assets_ + "/weights.loom"));
    if (GetParam()) {
      const char* declaration = std::getenv("LOOM_SERVE_TEST_TEXT_IMAGE_CARD");
      ASSERT_NE(declaration, nullptr);
      std::filesystem::copy(
          std::filesystem::absolute(declaration).parent_path(),
          assets_ + "/card", std::filesystem::copy_options::recursive);
    }
  }

  void TearDown() override {
    RegisteredSourceTextServiceTest::TearDown();
    if (!assets_.empty()) {
      std::filesystem::remove_all(assets_);
    }
  }

  std::string ImageDeclaration(const std::string& assets = "") {
    if (GetParam()) {
      auto path = assets_ + "/card/model.json";
      if (!assets.empty()) {
        std::ifstream input(path);
        std::string json((std::istreambuf_iterator<char>(input)), {});
        iree_string_view_t value =
            iree_make_string_view(json.data(), json.size());
        for (const char* name : {"variants", "fixture", "services", "chat",
                                 "parameters", "vision"}) {
          iree_string_view_t next;
          IREE_EXPECT_OK(iree_json_lookup_object_value(
              value, iree_make_cstring_view(name), &next));
          value = next;
        }
        json.replace(
            value.data - json.data(), value.size,
            R"({"files":[{"path":"missing.safetensors","format":"safetensors"}]})");
        path = assets_ + "/card/missing.json";
        std::ofstream output(path);
        output << json;
        output.close();
        EXPECT_TRUE(output.good());
      }
      return "{\"name\":\"fixture-text\",\"admission\":\"drained\",\"model\":{"
             "\"path\":\"" +
             path +
             "\",\"preset\":\"check\"},\"definition\":{\"kind\":\"text\"}}";
    }
    const std::string image =
        ",\"image\":{\"prepare\":\"" + projector_directory_ +
        "/weighted_prepare.loom\",\"control\":\"" + projector_directory_ +
        "/control.loom\",\"assets\":\"" + (assets.empty() ? assets_ : assets) +
        "\"}";
    return Declaration("fixture-text", directory_, "drained", image,
                       {projector_directory_ + "/project.loom",
                        projector_directory_ + "/weighted_project.loom"});
  }

  void Register(const std::string& assets = "") {
    ASSERT_NO_FATAL_FAILURE(StartManaged());
    const auto declaration = ImageDeclaration(assets);
    const auto response = Read(Send(declaration, "", "/v1/models/register"));
    ASSERT_EQ(response.find("HTTP/1.1 202"), 0u) << response;
    if (GetParam()) {
      ASSERT_FALSE(WaitModel("fixture-text", "\"image\":").empty());
    }
  }

  void Control(const char* operation, const char* component = "",
               int expected_status = 200) {
    std::string selector = R"({"model":"fixture-text")";
    if (*component) {
      selector += std::string(",\"component\":\"") + component + "\"";
    }
    selector += "}";
    const auto response =
        Read(Send(selector, "", std::string("/v1/models/") + operation));
    ASSERT_EQ(response.find("HTTP/1.1 " + std::to_string(expected_status)), 0u)
        << response;
  }

  std::string State() {
    const auto response = Read(Send("", "", "/v1/status", "GET"));
    EXPECT_EQ(response.find("HTTP/1.1 200"), 0u) << response;
    const auto body = response.find("\r\n\r\n");
    if (body == std::string::npos) {
      ADD_FAILURE() << response;
      return "{}";
    }
    const auto models = Field(response.substr(body + 4), "data");
    iree_string_view_t model = iree_string_view_empty();
    IREE_EXPECT_OK(iree_json_array_get(
        iree_make_string_view(models.data(), models.size()), 0, &model));
    const std::string object(model.data ? model.data : "", model.size);
    EXPECT_EQ(Field(object, "id"), "fixture-text");
    return Field(object, "loom");
  }

  std::string Parameters(const char* component) {
    return Field(Field(State(), "parameters"), component);
  }

  void FinishOperation() {
    const auto state =
        WaitModel("fixture-text", R"("parameter_operation":"idle")");
    ASSERT_FALSE(state.empty());
    const auto observed = State();
    ASSERT_FALSE(HasField(observed, "error")) << observed;
  }

  void Activate(const char* component = "") {
    ASSERT_NO_FATAL_FAILURE(Control("activate", component, 202));
    ASSERT_NO_FATAL_FAILURE(FinishOperation());
  }

  void DeactivateImage() {
    ASSERT_NO_FATAL_FAILURE(Control("deactivate", "image", 202));
    ASSERT_NO_FATAL_FAILURE(FinishOperation());
    const auto image = Parameters("image");
    if (HasField(image, "active")) {
      EXPECT_EQ(Field(image, "active"), "false") << image;
      EXPECT_EQ(Field(image, "committed_bytes"), "0") << image;
    } else {
      EXPECT_EQ(Field(image, "phase"), "declared") << image;
    }
  }

  static std::string CompleteImage(Oracle& oracle, size_t count) {
    std::string result;
    for (size_t i = 0; i < count; ++i) {
      const int32_t token = oracle.Predict();
      if (token == 3) {
        break;
      }
      result.push_back("abx"[token]);
      oracle.Consume(token);
    }
    return result;
  }

  // Exercise the same native registry adapter at an exact serialized boundary.
  // The sole shared worker cannot run newly queued preparation/readiness work
  // until this owner turn returns; removal never relies on elapsed time.
  void RemoveDuring(loom_serve_parameter_operation_t operation) {
    IREE_ASSERT_OK(loom_serve_text_model_free(model_));
    model_ = nullptr;
    const auto declaration = ImageDeclaration();
    loom_serve_model_configuration_t configuration = {};
    IREE_ASSERT_OK(loom_serve_model_configuration_initialize(
        iree_make_string_view(declaration.data(), declaration.size()),
        iree_string_view_empty(), nullptr, host_allocator_, &configuration));
    const loom_serve_model_registration_options_t options = {
        .device_group = group_,
        .heartbeat_interval = 0,
    };
    loom_serve_registry_t* registry = nullptr;
    IREE_ASSERT_OK(loom_serve_registry_allocate(
        0, nullptr, 1, loom_serve_model_registration_factory(&options),
        host_allocator_, &registry));
    iree_host_size_t index = 0;
    IREE_ASSERT_OK(
        loom_serve_registry_register(registry, &configuration, &index));
    auto* model = loom_serve_registry_at(registry, index);
    IREE_ASSERT_OK(loom_serve_registry_activate(
        model, LOOM_SERVE_MODEL_COMPONENT_PRIMARY));
    auto* notification = loom_serve_http_server_notification(server_);
    const loom_serve_wake_callback_t observer = {
        [](void* data) {
          iree_notification_post(static_cast<iree_notification_t*>(data),
                                 IREE_ALL_WAITERS);
        },
        notification};
    bool image_started = false;
    bool removed = false;
    while (model->incarnation && !HasFatalFailure()) {
      const auto ticket = iree_notification_prepare_wait(notification);
      bool progress = false;
      OwnerTurn::Run(loom_serve_host_executor(host_), [&] {
        if (model->phase == LOOM_SERVE_REGISTRATION_READY) {
          IREE_ASSERT_OK(model->service.advance(
              model->service.self, loom_serve_registry_service_flags(model),
              &progress));
        }
        bool advanced = false;
        IREE_ASSERT_OK(
            loom_serve_registry_advance(registry, index, observer, &advanced));
        progress |= advanced;
        IREE_ASSERT_OK(iree_status_clone(model->status));
        if (model->phase != LOOM_SERVE_REGISTRATION_READY || removed) {
          return;
        }
        if (!image_started &&
            model->parameter_operation == LOOM_SERVE_PARAMETER_OPERATION_NONE) {
          IREE_ASSERT_OK(loom_serve_registry_activate(
              model, LOOM_SERVE_MODEL_COMPONENT_IMAGE));
          IREE_ASSERT_OK(loom_serve_registry_advance(registry, index, observer,
                                                     &advanced));
          progress = true;
          image_started = true;
        }
        if (image_started && model->parameter_operation == operation) {
          const auto image = model->registration.parameters(
              model->registration.self, LOOM_SERVE_MODEL_COMPONENT_IMAGE);
          EXPECT_STREQ(image.phase,
                       operation == LOOM_SERVE_PARAMETER_OPERATION_WAITING
                           ? "preparing"
                           : "ready");
          IREE_ASSERT_OK(
              loom_serve_registry_remove(model, LOOM_SERVE_REMOVAL_ABORT));
          removed = true;
          progress = true;
        }
      });
      if (progress || !model->incarnation || HasFatalFailure()) {
        iree_notification_cancel_wait(notification);
      } else {
        iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
    EXPECT_TRUE(removed);
    EXPECT_EQ(model->incarnation, 0u);
    IREE_EXPECT_OK(loom_serve_registry_free(registry));
  }

  // Runfile sources shared by native prepare and command compilation.
  std::string projector_directory_;
  // Owned checkpoint package, independent of primary parameter residency.
  std::string assets_;
};

TEST_P(ManagedMediaServiceTest, ExplicitWarmupAndPinPolicyAreComponentScoped) {
  ASSERT_NO_FATAL_FAILURE(Register());
  ASSERT_NO_FATAL_FAILURE(Control("pin", "image"));
  EXPECT_EQ(Field(Parameters("image"), "pinned"), "true");
  EXPECT_EQ(Field(Parameters("image"), "phase"), "declared");
  EXPECT_FALSE(HasField(Parameters("image"), "active"));

  // Unqualified management still selects only the language model.
  ASSERT_NO_FATAL_FAILURE(Activate());
  const auto primary = Parameters("primary");
  EXPECT_EQ(Field(primary, "active"), "true");
  EXPECT_EQ(Field(primary, "pinned"), "false");
  EXPECT_EQ(Field(Parameters("image"), "phase"), "declared");
  EXPECT_FALSE(HasField(Parameters("image"), "active"));

  const auto primary_metrics = Read(Send("", "", "/metrics", "GET"));
  EXPECT_EQ(primary_metrics.find("HTTP/1.1 200"), 0u) << primary_metrics;
  EXPECT_NE(primary_metrics.find("component=\"primary\""), std::string::npos);
  EXPECT_EQ(primary_metrics.find("component=\"image\""), std::string::npos);

  // JSON escapes select the same declared role, not a different key.
  ASSERT_NO_FATAL_FAILURE(Activate("\\u0069mage"));
  const auto image = Parameters("image");
  EXPECT_EQ(Field(image, "phase"), "ready") << image;
  EXPECT_EQ(Field(image, "active"), "true") << image;
  EXPECT_EQ(Field(image, "pinned"), "true") << image;
  EXPECT_EQ(Parameters("primary"), primary);
  const auto image_metrics = Read(Send("", "", "/metrics", "GET"));
  EXPECT_EQ(image_metrics.find("HTTP/1.1 200"), 0u) << image_metrics;
  EXPECT_NE(image_metrics.find(
                "loom_serve_model_parameter_pins{model=\"fixture-text\","
                "kind=\"text\",incarnation=\"1\",component=\"image\"} 0\n"),
            std::string::npos)
      << image_metrics;
  EXPECT_EQ(Field(State(), "parameter_component"), "image");
  EXPECT_EQ(Field(Field(State(), "status"), "admission"), "drained");
  ASSERT_NO_FATAL_FAILURE(Control("deactivate", "image", 409));
  ASSERT_NO_FATAL_FAILURE(Control("unpin", "image"));
  EXPECT_EQ(Field(Parameters("image"), "active"), "true");
  ASSERT_NO_FATAL_FAILURE(DeactivateImage());
  EXPECT_EQ(Parameters("primary"), primary);
  ASSERT_NO_FATAL_FAILURE(Activate("image"));
  EXPECT_EQ(Field(Parameters("image"), "active"), "true");
  EXPECT_EQ(Field(Parameters("image"), "pinned"), "false");
}

TEST_P(ManagedMediaServiceTest,
       ImageTrimPreservesTextCheckpointAndCachedMedia) {
  ASSERT_NO_FATAL_FAILURE(Register());
  ASSERT_NO_FATAL_FAILURE(Activate());
  ASSERT_NO_FATAL_FAILURE(Control("undrain"));
  const std::string root = Message("user", "abxa");
  const auto seed =
      Content(Read(Send(Request(root, 2), "X-Loom-Session: text\r\n")));
  EXPECT_EQ(seed, Oracle::Complete("abxa", 2));
  ASSERT_EQ(Read(Send("", "X-Loom-Session: text\r\n", "/v1/checkpoints/text"))
                .find("HTTP/1.1 201"),
            0u);

  const auto message =
      ImageMessage("a", "image/x-portable-arbitrarymap", kPam, "low", "b");
  const auto first =
      Read(Send(Request(message, 4), "X-Loom-Session: image\r\n"));
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM)) {
    EXPECT_EQ(first.find("HTTP/1.1 501"), 0u) << first;
    return;
  }
  // Red 255 + opaque alpha 255 + cohort width 1 + checkpoint bias 2.
  const float feature = 513.0f;
  int32_t bits;
  std::memcpy(&bits, &feature, sizeof(bits));
  Oracle image_oracle;
  for (int32_t token : {0, bits, 1}) {
    image_oracle.Consume(token);
  }
  const auto generated = Content(first);
  EXPECT_EQ(generated, CompleteImage(image_oracle, 4));

  ASSERT_NO_FATAL_FAILURE(Control("drain"));
  const auto primary = Parameters("primary");
  ASSERT_NO_FATAL_FAILURE(DeactivateImage());
  const auto cold = Parameters("image");
  EXPECT_EQ(Parameters("primary"), primary);
  ASSERT_NO_FATAL_FAILURE(Control("undrain"));
  const auto history =
      root + "," + Message("assistant", seed) + "," + Message("user", "xx");
  const auto resumed =
      Read(Send(Request(history, 4),
                "X-Loom-Session: fork\r\nX-Loom-Checkpoint: text\r\n"));
  EXPECT_EQ(Content(resumed), Oracle::Complete("abxa" + seed + "#xx", 4));
  EXPECT_EQ(resumed.find("\"cached_tokens\":0"), std::string::npos) << resumed;

  // A retained multimodal prefix needs its language state, not the projector.
  const auto followup =
      Read(Send(Request(message + "," + Message("assistant", generated) + "," +
                            Message("user", "x"),
                        4),
                "X-Loom-Session: image\r\n"));
  image_oracle.Consume("#x");
  EXPECT_EQ(Content(followup), CompleteImage(image_oracle, 4));
  EXPECT_EQ(followup.find("\"cached_tokens\":0"), std::string::npos)
      << followup;
  EXPECT_EQ(Parameters("image"), cold);

  // Fresh media demand restores the retired model or its elastic weights.
  const auto fresh = Content(Read(Send(Request(message, 4))));
  EXPECT_EQ(fresh, generated);
  EXPECT_EQ(Field(Parameters("image"), "phase"), "ready");
  EXPECT_EQ(Field(Parameters("image"), "active"), "true");
  ASSERT_EQ(
      Read(Send("", "", "/v1/checkpoints/text", "DELETE")).find("HTTP/1.1 200"),
      0u);
}

TEST_P(ManagedMediaServiceTest, FailedImageWarmupLeavesTextAvailable) {
  ASSERT_NO_FATAL_FAILURE(Register(assets_ + "/missing"));
  ASSERT_NO_FATAL_FAILURE(Activate("image"));
  EXPECT_EQ(Field(State(), "registration"), "ready");
  EXPECT_FALSE(HasField(State(), "error"));
  const auto image = Parameters("image");
  EXPECT_EQ(Field(image, "phase"), "failed") << image;
  EXPECT_TRUE(HasField(image, "error")) << image;
  EXPECT_FALSE(HasField(image, "active")) << image;
  const auto metrics = Read(Send("", "", "/metrics", "GET"));
  EXPECT_EQ(metrics.find("component=\"image\""), std::string::npos) << metrics;
  // Repeated demand reports the retained failure without rerunning prepare.
  ASSERT_NO_FATAL_FAILURE(Activate("image"));
  EXPECT_EQ(Parameters("image"), image);
  ASSERT_NO_FATAL_FAILURE(Control("undrain"));
  EXPECT_EQ(Content(Read(Send(Request(Message("user", "abxa"), 4)))),
            Oracle::Complete("abxa", 4));
  ASSERT_NO_FATAL_FAILURE(Control("unregister", "", 202));
  WaitModel("fixture-text", {});
  EXPECT_FALSE(http_.failed());
}

TEST_P(ManagedMediaServiceTest, ComponentSelectorsValidateTheDeclaredBoundary) {
  ASSERT_NO_FATAL_FAILURE(Register());
  for (const char* fields : {R"("component":"missing")", R"("component":0)",
                             R"("component":"image","component":"primary")"}) {
    const auto body = std::string(R"({"model":"fixture-text",)") + fields + "}";
    const auto response = Read(Send(body, "", "/v1/models/activate"));
    EXPECT_EQ(response.find("HTTP/1.1 400"), 0u) << response;
  }
  ASSERT_NO_FATAL_FAILURE(Control("drain", "image", 400));
  ASSERT_NO_FATAL_FAILURE(Control("unregister", "image", 400));
  ASSERT_NO_FATAL_FAILURE(Control("unregister", "", 202));
  WaitModel("fixture-text", {});
  const auto declaration = Declaration("fixture-text", directory_, "drained");
  ASSERT_EQ(
      Read(Send(declaration, "", "/v1/models/register")).find("HTTP/1.1 202"),
      0u);
  ASSERT_NO_FATAL_FAILURE(Control("activate", "image", 404));
  ASSERT_NO_FATAL_FAILURE(Control("pin", "image", 404));
  EXPECT_EQ(Field(State(), "registration"), "registered");
}

TEST_P(ManagedMediaServiceTest, RemovalJoinsAcceptedImagePreparation) {
  RemoveDuring(LOOM_SERVE_PARAMETER_OPERATION_WAITING);
}

TEST_P(ManagedMediaServiceTest, RemovalJoinsAcceptedImageActivation) {
  RemoveDuring(LOOM_SERVE_PARAMETER_OPERATION_ACTIVATING);
}

INSTANTIATE_TEST_SUITE_P(Declarations, ManagedMediaServiceTest,
                         ::testing::Bool());

}  // namespace
