// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_RECONFIGURATION_H_
#define LOOM_SERVE_SERVICE_TEXT_RECONFIGURATION_H_

#include "loom/serve/service/text/model.h"
#include "loom/service/execution_reconfiguration.h"

#ifdef __cplusplus
extern "C" {
#endif

// Native replacement capability over this model's retained parameters/state.
loom_service_execution_update_t loom_serve_text_model_execution_update(
    loom_serve_text_model_t* model);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_RECONFIGURATION_H_
