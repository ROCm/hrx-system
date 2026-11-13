// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/endpoint.h"

const char* loom_serve_service_endpoint(loom_service_kind_t kind) {
  static const char* const endpoints[] = {
      [LOOM_SERVICE_KIND_TEXT] = "/v1/chat/completions",
      [LOOM_SERVICE_KIND_IMAGE] = "/v1/images/generations",
      [LOOM_SERVICE_KIND_EMBEDDING] = "/v1/embeddings",
      [LOOM_SERVICE_KIND_DECISION] = "/v1/systemone",
      [LOOM_SERVICE_KIND_SPEECH] = "/v1/audio/speech",
      [LOOM_SERVICE_KIND_RECOGNITION] = "/v1/audio/transcriptions",
      [LOOM_SERVICE_KIND_ACTIVITY] = NULL,
      [LOOM_SERVICE_KIND_SEQUENCE] = "/v1/sequences",
  };
  return endpoints[kind];
}
