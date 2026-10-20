// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/qos.h"

iree_status_t loom_serve_qos_parse(const loom_serve_http_request_t* request,
                                   loom_serve_execution_qos_t* out_qos) {
  *out_qos = LOOM_SERVE_EXECUTION_QOS_BACKGROUND;
  iree_string_view_t value;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-QoS"),
                                             &value)) {
    return iree_ok_status();
  }
  static const iree_string_view_t names[] = {IREE_SVL("background"),
                                             IREE_SVL("realtime")};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(names); ++i) {
    if (iree_string_view_equal(value, names[i])) {
      *out_qos = (loom_serve_execution_qos_t)i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "X-Loom-QoS must be background or realtime");
}
