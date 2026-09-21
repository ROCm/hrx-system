// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "util/native_event.h"

#include <errno.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <unistd.h>

namespace amdf::cts {

amdf_native_event_type_t NativeEventType() {
  return AMDF_NATIVE_EVENT_TYPE_EVENTFD;
}

::testing::AssertionResult CreateNativeEvent(amdf_native_event_t* out_event) {
  const int descriptor = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (descriptor < 0) {
    return ::testing::AssertionFailure() << "eventfd: errno " << errno;
  }
  *out_event = {};
  out_event->type = NativeEventType();
  out_event->payload.file_descriptor = descriptor;
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult DestroyNativeEvent(amdf_native_event_t* event) {
  const int result = close(static_cast<int>(event->payload.file_descriptor));
  *event = {};
  return result == 0
             ? ::testing::AssertionSuccess()
             : (::testing::AssertionFailure() << "close: errno " << errno);
}

::testing::AssertionResult TryConsumeNativeEvent(
    const amdf_native_event_t& event, bool* out_ready) {
  eventfd_t value = 0;
  if (eventfd_read(static_cast<int>(event.payload.file_descriptor), &value) ==
      0) {
    *out_ready = true;
    return ::testing::AssertionSuccess();
  }
  if (errno == EAGAIN) {
    *out_ready = false;
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << "eventfd_read: errno " << errno;
}

::testing::AssertionResult WaitNativeEvent(const amdf_native_event_t& event) {
  struct pollfd poll_event = {};
  poll_event.fd = static_cast<int>(event.payload.file_descriptor);
  poll_event.events = POLLIN;
  int result;
  do {
    result = poll(&poll_event, 1, -1);
  } while (result < 0 && errno == EINTR);
  if (result != 1 || poll_event.revents != POLLIN) {
    return ::testing::AssertionFailure()
           << "poll: result " << result << ", events " << poll_event.revents
           << ", errno " << errno;
  }
  bool ready = false;
  const auto status = TryConsumeNativeEvent(event, &ready);
  if (!status) {
    return status;
  }
  return ready
             ? ::testing::AssertionSuccess()
             : (::testing::AssertionFailure() << "readable event lost credit");
}

}  // namespace amdf::cts
