// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/endpoint.h"

#include "iree/testing/gtest.h"

namespace {

TEST(ServiceEndpointTest, PrimaryRoutesPreserveSemanticRoles) {
  const struct {
    // Service role shared by native and HTTP callers.
    loom_service_kind_t kind;
    // Primary inference endpoint; activity is native-only.
    const char* endpoint;
  } cases[] = {
      {LOOM_SERVICE_KIND_TEXT, "/v1/chat/completions"},
      {LOOM_SERVICE_KIND_IMAGE, "/v1/images/generations"},
      {LOOM_SERVICE_KIND_EMBEDDING, "/v1/embeddings"},
      {LOOM_SERVICE_KIND_DECISION, "/v1/systemone"},
      {LOOM_SERVICE_KIND_SPEECH, "/v1/audio/speech"},
      {LOOM_SERVICE_KIND_RECOGNITION, "/v1/audio/transcriptions"},
      {LOOM_SERVICE_KIND_ACTIVITY, nullptr},
      {LOOM_SERVICE_KIND_SEQUENCE, "/v1/sequences"},
  };
  for (const auto& row : cases) {
    SCOPED_TRACE(static_cast<int>(row.kind));
    EXPECT_STREQ(loom_serve_service_endpoint(row.kind), row.endpoint);
  }
}

}  // namespace
