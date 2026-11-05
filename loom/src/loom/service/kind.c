// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/kind.h"

static const char* const loom_service_kinds[] = {
    [LOOM_SERVICE_KIND_TEXT] = "text",
    [LOOM_SERVICE_KIND_IMAGE] = "image",
    [LOOM_SERVICE_KIND_EMBEDDING] = "embedding",
    [LOOM_SERVICE_KIND_DECISION] = "decision",
    [LOOM_SERVICE_KIND_SPEECH] = "speech",
    [LOOM_SERVICE_KIND_RECOGNITION] = "recognition",
    [LOOM_SERVICE_KIND_ACTIVITY] = "activity",
    [LOOM_SERVICE_KIND_SEQUENCE] = "sequence",
};

bool loom_service_kind_parse(iree_string_view_t value,
                             loom_service_kind_t* out_kind) {
  for (iree_host_size_t i = 1; i < IREE_ARRAYSIZE(loom_service_kinds); ++i) {
    if (iree_string_view_equal(value,
                               iree_make_cstring_view(loom_service_kinds[i]))) {
      *out_kind = (loom_service_kind_t)i;
      return true;
    }
  }
  return false;
}

const char* loom_service_kind_name(loom_service_kind_t kind) {
  return loom_service_kinds[kind];
}
