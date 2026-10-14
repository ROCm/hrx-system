// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/image_service_test_fixture.h"

#include <cstdlib>
#include <filesystem>

namespace loom_serve::testing::text {

void ImageComponentTextServiceTest::SetUp() {
  ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
  const char* source_path = std::getenv("LOOM_SERVE_TEST_PROJECTOR_SOURCE");
  ASSERT_NE(source_path, nullptr);
  image_directory_ = std::filesystem::path(source_path).parent_path().string();
  const loom_serve_finite_model_options_t options = {
      .sources =
          {
              .base_directory =
                  iree_make_cstring_view(image_directory_.c_str()),
              .prepare = IREE_SV("prepare.loom"),
              .control = IREE_SV("control.loom"),
              .count = 1,
              .values = &image_provider_,
          },
      .asset_directory = IREE_SV("."),
      .options = IREE_SV("{}"),
  };
  const loom_serve_wake_callback_t observer = {
      [](void* user_data) {
        iree_notification_post(static_cast<iree_notification_t*>(user_data),
                               IREE_ALL_WAITERS);
      },
      loom_serve_http_server_notification(server_)};
  IREE_ASSERT_OK(loom_serve_finite_component_allocate(
      device_, &options, observer, host_allocator_, &image_component_));
}

void ImageComponentTextServiceTest::TearDown() {
  // A router-owned service must relinquish the borrowed component before the
  // deployment owner can retire it and before the device disappears.
  if (http_.joinable()) {
    loom_serve_http_server_request_stop(server_);
    IREE_EXPECT_OK(http_.Join());
  }
  if (image_component_) {
    RetireImageComponent();
    loom_serve_finite_component_free(image_component_);
    image_component_ = nullptr;
  }
  SourceTextServiceTest::TearDown();
}

loom_serve_text_service_options_t
ImageComponentTextServiceTest::ServiceOptions() const {
  auto options = SourceTextServiceTest::ServiceOptions();
  options.image = image_component_;
  return options;
}

void ImageComponentTextServiceTest::TrimImageComponent() {
  auto* notification = loom_serve_http_server_notification(server_);
  bool complete = false;
  while (!complete) {
    const uint32_t ticket = iree_notification_prepare_wait(notification);
    IREE_ASSERT_OK(
        loom_serve_finite_component_trim(image_component_, &complete));
    if (complete) {
      iree_notification_cancel_wait(notification);
    } else {
      iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
}

void ImageComponentTextServiceTest::RetireImageComponent() {
  auto* notification = loom_serve_http_server_notification(server_);
  bool complete = false;
  while (!complete) {
    const uint32_t ticket = iree_notification_prepare_wait(notification);
    IREE_ASSERT_OK(
        loom_serve_finite_component_retire(image_component_, &complete));
    if (complete) {
      iree_notification_cancel_wait(notification);
    } else {
      iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
}

void ImageTextServiceTest::SetUp() {
  ASSERT_NO_FATAL_FAILURE(ImageComponentTextServiceTest::SetUp());
  auto options = ServiceOptions();
  options.prefix_cache_capacity = 2;
  options.chunk_size = 4;
  IREE_ASSERT_OK(loom_serve_text_service_allocate(model_, &options, nullptr,
                                                  host_allocator_, &service_));
  interface_ = loom_serve_text_service_interface(service_);
  Observe(interface_);
}

void ImageTextServiceTest::TearDown() {
  if (service_) {
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
        interface_,
        [&] {
          return !interface_.is_pending(interface_.self) &&
                 !interface_.is_active(interface_.self) &&
                 !interface_.snapshot(interface_.self).queued_requests;
        },
        LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING));
    interface_.set_observer(interface_.self, {});
    loom_serve_text_service_free(service_, IREE_STATUS_OK);
    service_ = nullptr;
  }
  ImageComponentTextServiceTest::TearDown();
}

std::string ImageTextServiceTest::CompleteRequest(const std::string& body,
                                                  const std::string& headers) {
  const int client = Send(body, headers);
  loom_serve_http_connection_t* connection = nullptr;
  Accept(client, interface_, &connection);
  if (HasFatalFailure()) {
    return {};
  }
  AdvanceUntil(interface_, [&] {
    return !interface_.is_pending(interface_.self) &&
           !interface_.is_active(interface_.self) &&
           !interface_.snapshot(interface_.self).queued_requests;
  });
  if (HasFatalFailure()) {
    return {};
  }
  return Read(client);
}

}  // namespace loom_serve::testing::text
