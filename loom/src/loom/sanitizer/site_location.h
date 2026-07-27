// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Utilities for attaching compact sanitizer site payloads to source locations.

#ifndef LOOM_SANITIZER_SITE_LOCATION_H_
#define LOOM_SANITIZER_SITE_LOCATION_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/sanitizer/site_payload.h"

#ifdef __cplusplus
extern "C" {
#endif

// Adds a tagged sanitizer site location containing |payload| and wrapping
// |source_location|. The source location may be LOOM_LOCATION_UNKNOWN; final
// site IDs are still assigned from surviving sanitizer ops.
iree_status_t loom_sanitizer_make_site_location(
    loom_module_t* module, loom_location_id_t source_location,
    const loom_sanitizer_site_payload_t* payload,
    loom_location_id_t* out_location);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SANITIZER_SITE_LOCATION_H_
