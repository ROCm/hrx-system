// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PARAMETER_ADMISSION_H_
#define LOOM_SERVE_RUNTIME_PARAMETER_ADMISSION_H_

#include "loom/serve/runtime/execution/scheduler.h"
#include "loom/serve/runtime/parameter/residency.h"

#ifdef __cplusplus
extern "C" {
#endif

// Composes execution eligibility with physical parameter admission on the
// serialized owner. Policy denial withdraws memory FIFO interest. Physical
// denial releases execution permission, allowing peers to free needed capacity.
// ACQUIRED retains the model-owned permit through activation and any dependent
// uploads. A source chaining activation into numerical execution retains the
// same claim through that compound unit instead of acquiring a second slot.
// Successful readiness releases the claim; a submission or readiness failure
// leaves it with actual model retirement. NULL permit opts out of coordination.
iree_status_t loom_serve_admission_try_acquire(
    loom_serve_residency_t* residency, loom_serve_execution_permit_t* permit,
    loom_serve_execution_qos_t qos,
    loom_serve_residency_admission_t* out_admission);

// Withdraws both unaccepted interests without revoking accepted native work.
// The source separately discards its unadmitted request or prepared cohort.
void loom_serve_admission_cancel(loom_serve_residency_t* residency,
                                 loom_serve_execution_permit_t* permit);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PARAMETER_ADMISSION_H_
