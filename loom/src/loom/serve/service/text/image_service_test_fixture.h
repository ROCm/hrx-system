// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_IMAGE_SERVICE_TEST_FIXTURE_H_
#define LOOM_SERVE_SERVICE_TEXT_IMAGE_SERVICE_TEST_FIXTURE_H_

#include "loom/serve/service/text/service_test_fixture.h"

namespace loom_serve::testing::text {

// Real optional image component shared by projector and integrated text tests.
// The component remains deployment-owned and outlives every borrowing service.
class ImageComponentTextServiceTest : public SourceTextServiceTest {
 protected:
  void SetUp() override;

  void TearDown() override;

  loom_serve_text_service_options_t ServiceOptions() const override;

  void TrimImageComponent();

  void RetireImageComponent();

  // Source package path borrowed through component retirement.
  std::string image_directory_;
  // Direct numerical provider borrowed through component retirement.
  iree_string_view_t image_provider_ = IREE_SV("project.loom");
  // Owned deployment component borrowed by text services and projectors.
  loom_serve_finite_component_t* image_component_ = nullptr;
};

class ImageTextServiceTest : public ImageComponentTextServiceTest {
 protected:
  void SetUp() override;

  void TearDown() override;

  std::string CompleteRequest(const std::string& body,
                              const std::string& headers = "");

  // Owned integrated text service borrowing image_component_.
  loom_serve_text_service_t* service_ = nullptr;
  // Borrowed callbacks invoked by this serialized test owner.
  loom_serve_http_service_t interface_ = {};
};

}  // namespace loom_serve::testing::text

#endif  // LOOM_SERVE_SERVICE_TEXT_IMAGE_SERVICE_TEST_FIXTURE_H_
