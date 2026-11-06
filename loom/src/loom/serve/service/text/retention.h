// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_RETENTION_H_
#define LOOM_SERVE_SERVICE_TEXT_RETENTION_H_

#include "loom/serve/service/text/model.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_serve_text_state_reservation_t;

// Marks references that policy could release on this unborrowed model turn.
// Every other native owner remains protected. Complete the query and clear the
// marks before yielding, submitting work, or changing ownership.
void loom_serve_text_retention_add_row(loom_serve_text_row_t* row);
void loom_serve_text_retention_add_checkpoint(
    loom_serve_text_checkpoint_t* checkpoint);
void loom_serve_text_retention_clear(loom_serve_text_model_t* model);

// Necessary state capacity after eligible histories retire, optionally with an
// incoming prepared request. Replan allows an immediate caller to choose new
// IDs after denial; otherwise the request's exact ranges remain required. This
// never edits logical IDs, mappings, readiness, or actual ownership. Only cold
// host scratch allocation can fail. Clear eligibility after every outcome.
iree_status_t loom_serve_text_retention_forecast(
    loom_serve_text_model_t* model,
    const struct loom_serve_text_state_reservation_t* request, bool replan,
    loom_runtime_residency_projection_t* out_projection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_RETENTION_H_
