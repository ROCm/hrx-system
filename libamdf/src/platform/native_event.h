// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_NATIVE_EVENT_H_
#define AMDF_SRC_PLATFORM_NATIVE_EVENT_H_

#include "amdf/native_event.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Signals a validated event of this platform's supported native type. The
// public boundary has checked the tag, payload shape and queue capability.
// Explicit wake path: one nonblocking native signal, no allocation, retained
// event, lazy initialization or retry. An already-readable full eventfd is a
// coalesced hint, not a failed wake.
amdf_status_t amdf_platform_native_event_signal(
    const amdf_native_event_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PLATFORM_NATIVE_EVENT_H_
