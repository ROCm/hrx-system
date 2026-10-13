// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_HOST_WAKE_H_
#define LOOM_SERVE_RUNTIME_HOST_WAKE_H_

// Borrowed readiness notification shared by task, timer and transport owners.
// Producers may invoke the callback while holding their publication lock. The
// callback only makes its consumer runnable; it never waits or calls back into
// the producer. Each registration API specifies when callbacks may begin and
// how detachment joins the last borrowed access.
typedef struct loom_serve_wake_callback_t {
  // Nonblocking wake invoked by readiness producers.
  void (*wake)(void* user_data);
  // Borrowed callback state, live through registration and final retirement.
  void* user_data;
} loom_serve_wake_callback_t;

#endif  // LOOM_SERVE_RUNTIME_HOST_WAKE_H_
