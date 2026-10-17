// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/kind.h"

#include "iree/testing/gtest.h"

namespace {

TEST(ModelKindTest, DeclarationAndDiscoveryShareCanonicalRoles) {
  const struct {
    // Typed role passed through registration and native admission.
    loom_serve_model_kind_t kind;
    // Public spelling in model documents and discovery.
    const char* name;
    // Public primary inference endpoint; activity is native-only.
    const char* endpoint;
  } cases[] = {
      {LOOM_SERVE_MODEL_KIND_TEXT, "text", "/v1/chat/completions"},
      {LOOM_SERVE_MODEL_KIND_IMAGE, "image", "/v1/images/generations"},
      {LOOM_SERVE_MODEL_KIND_EMBEDDING, "embedding", "/v1/embeddings"},
      {LOOM_SERVE_MODEL_KIND_DECISION, "decision", "/v1/systemone"},
      {LOOM_SERVE_MODEL_KIND_SPEECH, "speech", "/v1/audio/speech"},
      {LOOM_SERVE_MODEL_KIND_RECOGNITION, "recognition",
       "/v1/audio/transcriptions"},
      {LOOM_SERVE_MODEL_KIND_ACTIVITY, "activity", nullptr},
  };
  for (const auto& row : cases) {
    SCOPED_TRACE(row.name);
    loom_serve_model_kind_t kind = LOOM_SERVE_MODEL_KIND_TEXT;
    ASSERT_TRUE(
        loom_serve_model_kind_parse(iree_make_cstring_view(row.name), &kind));
    EXPECT_EQ(kind, row.kind);
    EXPECT_STREQ(loom_serve_model_kind_name(kind), row.name);
    EXPECT_STREQ(loom_serve_model_kind_endpoint(kind), row.endpoint);
  }
}

TEST(ModelKindTest, InvalidSpellingDoesNotChangeSelection) {
  for (const char* name : {"", "unknown", "Activity", "activity ", "textual"}) {
    SCOPED_TRACE(name);
    loom_serve_model_kind_t kind = LOOM_SERVE_MODEL_KIND_SPEECH;
    EXPECT_FALSE(
        loom_serve_model_kind_parse(iree_make_cstring_view(name), &kind));
    EXPECT_EQ(kind, LOOM_SERVE_MODEL_KIND_SPEECH);
  }
}

}  // namespace
