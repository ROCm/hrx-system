// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cross-platform futex primitives for low-level synchronization.
//
// This provides a 32-bit futex API that works across Linux, Windows, and
// Wasm. These are the building blocks for higher-level synchronization
// primitives like iree_slim_mutex_t and iree_notification_t.
//
// The API is 32-bit only because that's the lowest common denominator:
//   - Linux futex syscall: 32-bit
//   - Windows WaitOnAddress: variable size, but we use 32-bit
//   - Wasm: 32-bit (memory.atomic.wait32 / memory.atomic.notify)
//
// For 64-bit futex support (Linux kernel 6.7+ futex2), see
// iree/async/operations/futex.h which provides async io_uring operations
// with size flags.
//
// ThreadSanitizer does not instrument futex syscalls. The raw primitives remain
// available, including Linux shared-address waits that cannot use process-local
// pthread substitutes. Callers establish data ordering with release/acquire
// atomics; a wake alone does not publish memory to the sanitizer. Higher-level
// private locks select pthread implementations through IREE_RUNTIME_USE_FUTEX,
// which remains disabled under ThreadSanitizer.

#ifndef IREE_BASE_THREADING_FUTEX_H_
#define IREE_BASE_THREADING_FUTEX_H_

#include <stdint.h>

#include "iree/base/attributes.h"
#include "iree/base/config.h"
#include "iree/base/status.h"
#include "iree/base/target_platform.h"
#include "iree/base/time.h"

//===----------------------------------------------------------------------===//
// Platform detection
//===----------------------------------------------------------------------===//

// Allow users to fully disable all synchronization for systems that are known
// to never need it. This removes our dependency on pthreads.
#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX) || \
    defined(IREE_PLATFORM_WASM) || defined(IREE_PLATFORM_WINDOWS)
#define IREE_PLATFORM_HAS_FUTEX 1
#endif  // IREE_PLATFORM_*

#if defined(IREE_PLATFORM_HAS_FUTEX) && !defined(IREE_SANITIZER_THREAD)
// TODO: If we have TSan instrumentation for futexes we can enabled them when
// compiling with TSan.
#define IREE_RUNTIME_USE_FUTEX 1
#endif  // IREE_PLATFORM_HAS_FUTEX

#endif  // !IREE_SYNCHRONIZATION_DISABLE_UNSAFE

//===----------------------------------------------------------------------===//
// Platform headers
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_HAS_FUTEX)

#if defined(IREE_PLATFORM_WASM)
// Wasm atomic wait/notify are compiler builtins — no headers needed.
#elif defined(IREE_PLATFORM_WINDOWS)
// Windows headers included via iree/base/target_platform.h
#elif defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)
#include <errno.h>
#include <linux/futex.h>
#include <linux/types.h>
#include <sys/syscall.h>
#include <unistd.h>

// The raw syscall takes kernel words, independent of libc's time_t width.
// Some 32-bit ABIs (including RV32) expose only the time64 syscall.
#if !defined(__NR_futex) && defined(__NR_futex_time64)
#define IREE_LINUX_FUTEX_SYSCALL __NR_futex_time64
typedef int64_t iree_futex_time_word_t;
#else
#define IREE_LINUX_FUTEX_SYSCALL __NR_futex
typedef __kernel_long_t iree_futex_time_word_t;
#endif

#ifndef FUTEX_PRIVATE_FLAG
#define FUTEX_PRIVATE_FLAG 128
#endif  // !FUTEX_PRIVATE_FLAG

#endif  // IREE_PLATFORM_*

#endif  // IREE_PLATFORM_HAS_FUTEX

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Sentinel value to wake all waiters.
#define IREE_ALL_WAITERS INT32_MAX

//===----------------------------------------------------------------------===//
// Futex API
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_HAS_FUTEX)

// Waits in the OS while the value at |address| matches |expected_value|.
// A value mismatch succeeds without blocking. Wakes can be spurious, so the
// caller must recheck its condition with the appropriate memory ordering after
// every successful return, including interruption by an OS signal.
//
// |deadline_ns| can be either IREE_TIME_INFINITE_FUTURE to wait forever or an
// absolute time to wait until prior to returning early with
// IREE_STATUS_DEADLINE_EXCEEDED.
//
// Returns:
//   IREE_STATUS_OK: Wake or value mismatch; recheck the condition.
//   IREE_STATUS_DEADLINE_EXCEEDED: Timeout reached before wake.
//   IREE_STATUS_UNAVAILABLE: Native wait failed for another reason.
static inline iree_status_code_t iree_futex_wait(void* address,
                                                 uint32_t expected_value,
                                                 iree_time_t deadline_ns);

// Wakes at most |count| threads waiting for the |address| to change.
// Use IREE_ALL_WAITERS to wake all waiters. Which waiters are woken is
// undefined and it is not guaranteed that higher priority waiters will be woken
// over lower priority waiters.
static inline void iree_futex_wake(void* address, int32_t count);

//===----------------------------------------------------------------------===//
// Shared (cross-process) futex API
//===----------------------------------------------------------------------===//

// Cross-process variants that operate on physical page addresses instead of
// virtual addresses. On Linux, this omits FUTEX_PRIVATE_FLAG, causing the
// kernel to hash by physical page rather than {mm, virtual address}. This
// allows futex operations to work across processes sharing the same physical
// page (e.g., via mmap MAP_SHARED or shm_open).
//
// On Windows these aliases retain private, same-process semantics:
// WaitOnAddress/WakeByAddress cannot wake another process. Cross-process
// notification requires a separate native wake primitive on that platform.
//
// On Wasm, cross-process shared memory is not meaningful, so these are
// identical to the private variants.
//
// Linux shared waits require a kernel page lookup. Use the private variants
// (iree_futex_wait/wake) when cross-process semantics are not needed.
static inline iree_status_code_t iree_futex_wait_shared(
    void* address, uint32_t expected_value, iree_time_t deadline_ns);
static inline void iree_futex_wake_shared(void* address, int32_t count);

//===----------------------------------------------------------------------===//
// Platform implementations
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_WASM)

// Wasm provides memory.atomic.wait32 and memory.atomic.notify instructions
// via compiler builtins. These map directly to Atomics.wait/Atomics.notify
// in the JS host. memory.atomic.wait32 is only allowed on non-main threads
// (Web Workers); on the main thread it traps. This is fine because IREE's
// futex-based synchronization primitives are only used from worker threads
// in cooperative mode (the main thread uses the proactor poll loop instead).
//
// The timeout is specified in nanoseconds (-1 for infinite).

static inline iree_status_code_t iree_futex_wait(void* address,
                                                 uint32_t expected_value,
                                                 iree_time_t deadline_ns) {
  int64_t timeout_ns = -1;
  if (deadline_ns != IREE_TIME_INFINITE_FUTURE) {
    iree_time_t now_ns = iree_time_now();
    timeout_ns = deadline_ns > now_ns ? (int64_t)(deadline_ns - now_ns) : 0;
  }
  // Returns: 0 = "ok" (woken), 1 = "not-equal", 2 = "timed-out".
  int rc = __builtin_wasm_memory_atomic_wait32(
      (int32_t*)address, (int32_t)expected_value, timeout_ns);
  switch (rc) {
    case 0:
    case 1:
      return IREE_STATUS_OK;
    case 2:
      return IREE_STATUS_DEADLINE_EXCEEDED;
    default:
      return IREE_STATUS_OK;
  }
}

static inline void iree_futex_wake(void* address, int32_t count) {
  __builtin_wasm_memory_atomic_notify((int32_t*)address, (uint32_t)count);
}

// Wasm has no cross-process shared memory — alias to private variants.
static inline iree_status_code_t iree_futex_wait_shared(
    void* address, uint32_t expected_value, iree_time_t deadline_ns) {
  return iree_futex_wait(address, expected_value, deadline_ns);
}
static inline void iree_futex_wake_shared(void* address, int32_t count) {
  iree_futex_wake(address, count);
}

#elif defined(IREE_PLATFORM_WINDOWS)

#pragma comment(lib, "synchronization.lib")

static inline iree_status_code_t iree_futex_wait(void* address,
                                                 uint32_t expected_value,
                                                 iree_time_t deadline_ns) {
  uint32_t timeout_ms = iree_absolute_deadline_to_timeout_ms(deadline_ns);
  if (IREE_LIKELY(WaitOnAddress(address, &expected_value,
                                sizeof(expected_value), timeout_ms) == TRUE)) {
    return IREE_STATUS_OK;
  }
  if (GetLastError() == ERROR_TIMEOUT) {
    // Windows relative waits can expire before the requested interval because
    // their timeout clock is coarser than our monotonic clock. Treat an early
    // expiry as a spurious wake so callers recheck their condition and
    // deadline.
    return iree_time_now() >= deadline_ns ? IREE_STATUS_DEADLINE_EXCEEDED
                                          : IREE_STATUS_OK;
  }
  return IREE_STATUS_UNAVAILABLE;
}

static inline void iree_futex_wake(void* address, int32_t count) {
  if (count == INT32_MAX) {
    WakeByAddressAll(address);
    return;
  }
  for (; count > 0; --count) {
    WakeByAddressSingle(address);
  }
}

// Windows address waits only synchronize threads within the same process.
static inline iree_status_code_t iree_futex_wait_shared(
    void* address, uint32_t expected_value, iree_time_t deadline_ns) {
  return iree_futex_wait(address, expected_value, deadline_ns);
}
static inline void iree_futex_wake_shared(void* address, int32_t count) {
  iree_futex_wake(address, count);
}

#elif defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)

typedef struct iree_futex_timespec_t {
  // Whole seconds in the kernel's relative timeout.
  iree_futex_time_word_t tv_sec;
  // Remaining nanoseconds within the second.
  iree_futex_time_word_t tv_nsec;
} iree_futex_timespec_t;

// Encodes a finite, nonnegative duration. Returns false if a time32 syscall
// can cover only part of the interval. Its expiry is then a spurious wake.
static inline bool iree_futex_timespec_from_duration(
    iree_duration_t duration_ns, iree_futex_timespec_t* out_timeout) {
  const int64_t seconds = duration_ns / 1000000000;
  if (sizeof(out_timeout->tv_sec) == 4 && seconds > INT32_MAX) {
    out_timeout->tv_sec = INT32_MAX;
    out_timeout->tv_nsec = 999999999;
    return false;
  }
  out_timeout->tv_sec = (iree_futex_time_word_t)seconds;
  out_timeout->tv_nsec = (iree_futex_time_word_t)(duration_ns % 1000000000);
  return true;
}

static inline iree_status_code_t iree_futex_wait_linux(void* address,
                                                       uint32_t expected_value,
                                                       iree_time_t deadline_ns,
                                                       int operation) {
  iree_futex_timespec_t timeout;
  const iree_futex_timespec_t* timeout_ptr = NULL;
  bool covers_deadline = true;
  if (deadline_ns != IREE_TIME_INFINITE_FUTURE) {
    covers_deadline = iree_futex_timespec_from_duration(
        iree_absolute_deadline_to_timeout_ns(deadline_ns), &timeout);
    timeout_ptr = &timeout;
  }
  int rc = syscall(IREE_LINUX_FUTEX_SYSCALL, address, operation, expected_value,
                   timeout_ptr, NULL, 0);
  if (IREE_LIKELY(rc == 0) || errno == EAGAIN || errno == EINTR) {
    return IREE_STATUS_OK;
  } else if (errno == ETIMEDOUT) {
    return covers_deadline ? IREE_STATUS_DEADLINE_EXCEEDED : IREE_STATUS_OK;
  }
  return IREE_STATUS_UNAVAILABLE;
}

static inline iree_status_code_t iree_futex_wait(void* address,
                                                 uint32_t expected_value,
                                                 iree_time_t deadline_ns) {
  return iree_futex_wait_linux(address, expected_value, deadline_ns,
                               FUTEX_WAIT | FUTEX_PRIVATE_FLAG);
}

static inline void iree_futex_wake(void* address, int32_t count) {
  syscall(IREE_LINUX_FUTEX_SYSCALL, address, FUTEX_WAKE | FUTEX_PRIVATE_FLAG,
          count, NULL, NULL, 0);
}

// Cross-process variants: omit FUTEX_PRIVATE_FLAG so the kernel hashes by
// physical page instead of {mm, virtual address}. This allows futex operations
// across processes sharing the same physical page via mmap MAP_SHARED.
static inline iree_status_code_t iree_futex_wait_shared(
    void* address, uint32_t expected_value, iree_time_t deadline_ns) {
  return iree_futex_wait_linux(address, expected_value, deadline_ns,
                               FUTEX_WAIT);
}

static inline void iree_futex_wake_shared(void* address, int32_t count) {
  syscall(IREE_LINUX_FUTEX_SYSCALL, address, FUTEX_WAKE, count, NULL, NULL, 0);
}

#endif  // IREE_PLATFORM_*

#endif  // IREE_PLATFORM_HAS_FUTEX

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_THREADING_FUTEX_H_
