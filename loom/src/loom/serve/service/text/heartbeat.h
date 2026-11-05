// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_HEARTBEAT_H_
#define LOOM_SERVE_SERVICE_TEXT_HEARTBEAT_H_

#include "iree/base/threading/mutex.h"
#include "loom/runtime/host/timer.h"
#include "loom/runtime/telemetry/sample.h"
#include "loom/runtime/telemetry/stream.h"
#include "loom/service/telemetry/record.h"
#include "loom/service/text/statistics.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_text_heartbeat_t {
  // Protects copied statistics, never model work.
  iree_slim_mutex_t mutex;
  // Periodic readiness and callback lifetime, independent of model counters.
  loom_runtime_timer_t timer;
  // Borrowed owner-only stream, alive through deinitialization.
  loom_runtime_telemetry_stream_t* stream;
  // Borrowed owner counters independent of reporting interval and readers.
  const loom_service_telemetry_counters_t* counters;
  // Application-owned counters copied under mutex before reporting.
  loom_service_text_statistics_t snapshot;
} loom_serve_text_heartbeat_t;

// Initializes copied statistics and synchronization without accepting work.
// The application owner updates snapshot under mutex; no callback accesses the
// model. An initialized but unconfigured reporter can be deinitialized.
void loom_serve_text_heartbeat_initialize(
    loom_serve_text_heartbeat_t* heartbeat);

// Copies the last published counters without accessing model/session state.
// Safe while a finite CPU task owns mutable rows or device work is pending.
loom_service_text_statistics_t loom_serve_text_heartbeat_read(
    loom_serve_text_heartbeat_t* heartbeat);

// Samples the published text ownership/work facts independently of heartbeat
// interval and event retention. No model rows or device storage are accessed.
iree_status_t loom_serve_text_heartbeat_sample(
    loom_serve_text_heartbeat_t* heartbeat,
    loom_runtime_telemetry_sample_sink_t sink);

// Configures borrowed I/O and observation storage before the first advance.
// Zero interval disables reporting. The owner installs the timer observer
// before advancing, then clears it only after all accepted work has been
// consumed.
void loom_serve_text_heartbeat_configure(
    loom_serve_text_heartbeat_t* heartbeat, iree_async_proactor_t* proactor,
    loom_runtime_telemetry_stream_t* stream,
    const loom_service_telemetry_counters_t* counters,
    iree_duration_t interval);

// Whether a callback or unconsumed outcome still borrows this reporter.
// Reporting alone is not model activity that can free admission capacity.
bool loom_serve_text_heartbeat_is_pending(
    loom_serve_text_heartbeat_t* heartbeat);

// Consumes readiness, reports copied statistics and arms the next deadline.
// Stopping is sticky and cancels the timer through the poll owner. Continue
// advancing through pending work before clearing the observer or destroying
// storage; no task worker or callback waits for another operation.
iree_status_t loom_serve_text_heartbeat_advance(
    loom_serve_text_heartbeat_t* heartbeat, bool stopping, bool* out_progress);

// Publishes final copied statistics and releases synchronization. All accepted
// work and callback outcomes have already been consumed; this never waits.
void loom_serve_text_heartbeat_deinitialize(
    loom_serve_text_heartbeat_t* heartbeat);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_HEARTBEAT_H_
