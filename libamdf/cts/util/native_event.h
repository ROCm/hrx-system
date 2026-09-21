// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_UTIL_NATIVE_EVENT_H_
#define AMDF_CTS_UTIL_NATIVE_EVENT_H_

#include "amdf/native_event.h"
#include "gtest/gtest.h"

namespace amdf::cts {

// Native transport used by this platform's event-loop client.
amdf_native_event_type_t NativeEventType();

// Creates a caller-owned event, initially without a wake credit.
::testing::AssertionResult CreateNativeEvent(amdf_native_event_t* out_event);

// Closes an event after the caller has reconciled its native notifications.
::testing::AssertionResult DestroyNativeEvent(amdf_native_event_t* event);

// Waits indefinitely for a hint and consumes readiness. The outer CTS harness
// bounds hangs; normal device work has no local wall-clock deadline.
::testing::AssertionResult WaitNativeEvent(const amdf_native_event_t& event);

// Consumes existing readiness without waiting, for one-shot/coalescing checks.
::testing::AssertionResult TryConsumeNativeEvent(
    const amdf_native_event_t& event, bool* out_ready);

}  // namespace amdf::cts

#endif  // AMDF_CTS_UTIL_NATIVE_EVENT_H_
