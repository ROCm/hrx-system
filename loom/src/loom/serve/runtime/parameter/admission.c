// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/admission.h"

iree_status_t loom_serve_admission_try_acquire(
    loom_serve_residency_t* residency, loom_serve_execution_permit_t* permit,
    loom_serve_execution_qos_t qos,
    loom_serve_residency_admission_t* out_admission) {
  *out_admission = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
  if (permit && loom_serve_execution_is_preempted(permit->scheduler,
                                                  permit->resources, qos)) {
    loom_serve_admission_cancel(residency, permit);
    return iree_ok_status();
  }
  if (permit && permit->phase == LOOM_SERVE_EXECUTION_PERMIT_IDLE) {
    loom_serve_execution_permit_request(permit, qos);
  }
  if (!loom_serve_execution_permit_try_acquire(permit)) {
    loom_serve_residency_cancel_admission(residency);
    return iree_ok_status();
  }
  iree_status_t status =
      loom_serve_residency_try_admit(residency, out_admission);
  if (iree_status_is_ok(status) &&
      *out_admission != LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
    loom_serve_execution_permit_complete(permit);
  }
  return status;
}

void loom_serve_admission_cancel(loom_serve_residency_t* residency,
                                 loom_serve_execution_permit_t* permit) {
  loom_serve_execution_permit_cancel(permit);
  loom_serve_residency_cancel_admission(residency);
}
