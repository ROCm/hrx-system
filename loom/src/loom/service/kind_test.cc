// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/kind.h"

#include "iree/testing/gtest.h"

namespace {

TEST(ServiceKindTest, DeclarationAndDiscoveryShareCanonicalRoles) {
  const struct {
    // Typed role passed through registration and native admission.
    loom_service_kind_t kind;
    // Public spelling in model documents and discovery.
    const char* name;
  } cases[] = {
      {LOOM_SERVICE_KIND_TEXT, "text"},
      {LOOM_SERVICE_KIND_IMAGE, "image"},
      {LOOM_SERVICE_KIND_EMBEDDING, "embedding"},
      {LOOM_SERVICE_KIND_DECISION, "decision"},
      {LOOM_SERVICE_KIND_SPEECH, "speech"},
      {LOOM_SERVICE_KIND_RECOGNITION, "recognition"},
      {LOOM_SERVICE_KIND_ACTIVITY, "activity"},
      {LOOM_SERVICE_KIND_SEQUENCE, "sequence"},
  };
  for (const auto& row : cases) {
    SCOPED_TRACE(row.name);
    loom_service_kind_t kind = LOOM_SERVICE_KIND_TEXT;
    ASSERT_TRUE(
        loom_service_kind_parse(iree_make_cstring_view(row.name), &kind));
    EXPECT_EQ(kind, row.kind);
    EXPECT_STREQ(loom_service_kind_name(kind), row.name);
  }
}

TEST(ServiceKindTest, InvalidSpellingDoesNotChangeSelection) {
  for (const char* name : {"", "unknown", "Activity", "activity ", "textual"}) {
    SCOPED_TRACE(name);
    loom_service_kind_t kind = LOOM_SERVICE_KIND_SPEECH;
    EXPECT_FALSE(loom_service_kind_parse(iree_make_cstring_view(name), &kind));
    EXPECT_EQ(kind, LOOM_SERVICE_KIND_SPEECH);
  }
}

}  // namespace
