// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory_budget.h"

#include "iree/async/notification.h"

struct iree_hal_memory_budget_t {
  // Public references and outstanding credits retain the account.
  iree_atomic_ref_count_t ref_count;
  // Allocator retained by the account owner through all outstanding credits.
  iree_allocator_t host_allocator;
  // Optional retained capacity publication source.
  iree_async_notification_t* notification;
  // Maximum admitted physical bytes; a lower limit does not revoke owners.
  iree_atomic_int64_t limit;
  // Reserved or actually held bytes, including permanent native losses.
  iree_atomic_int64_t charged;
  // Monotonic subset of charged bytes whose native owners were lost.
  iree_atomic_int64_t abandoned;
  // First owned loss diagnostic; retained until account destruction.
  iree_atomic_intptr_t loss_status;
};

IREE_API_EXPORT iree_status_t iree_hal_memory_budget_create(
    uint64_t limit, iree_async_notification_t* notification,
    iree_allocator_t host_allocator, iree_hal_memory_budget_t** out_budget) {
  *out_budget = NULL;
  iree_hal_memory_budget_t* budget = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*budget), (void**)&budget));
  iree_atomic_ref_count_init(&budget->ref_count);
  budget->host_allocator = host_allocator;
  budget->notification = notification;
  iree_async_notification_retain(notification);
  iree_atomic_store(&budget->limit, (int64_t)limit, iree_memory_order_relaxed);
  *out_budget = budget;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_hal_memory_budget_retain(
    iree_hal_memory_budget_t* budget) {
  if (budget) {
    iree_atomic_ref_count_inc(&budget->ref_count);
  }
}

IREE_API_EXPORT void iree_hal_memory_budget_release(
    iree_hal_memory_budget_t* budget) {
  if (budget && iree_atomic_ref_count_dec(&budget->ref_count) == 1) {
    iree_status_ignore((iree_status_t)iree_atomic_load(
        &budget->loss_status, iree_memory_order_relaxed));
    iree_async_notification_release(budget->notification);
    iree_allocator_free(budget->host_allocator, budget);
  }
}

static void iree_hal_memory_budget_publish(iree_hal_memory_budget_t* budget) {
  if (budget->notification) {
    iree_async_notification_signal_if_observed(budget->notification,
                                               IREE_ALL_WAITERS);
  }
}

IREE_API_EXPORT void iree_hal_memory_budget_set_limit(
    iree_hal_memory_budget_t* budget, uint64_t limit) {
  const uint64_t previous = (uint64_t)iree_atomic_exchange(
      &budget->limit, (int64_t)limit, iree_memory_order_acq_rel);
  if (limit > previous) {
    iree_hal_memory_budget_publish(budget);
  }
}

IREE_API_EXPORT uint64_t
iree_hal_memory_budget_limit(const iree_hal_memory_budget_t* budget) {
  return (uint64_t)iree_atomic_load(&budget->limit, iree_memory_order_acquire);
}

IREE_API_EXPORT uint64_t
iree_hal_memory_budget_charged(const iree_hal_memory_budget_t* budget) {
  return (uint64_t)iree_atomic_load(&budget->charged,
                                    iree_memory_order_acquire);
}

IREE_API_EXPORT uint64_t
iree_hal_memory_budget_abandoned(const iree_hal_memory_budget_t* budget) {
  return (uint64_t)iree_atomic_load(&budget->abandoned,
                                    iree_memory_order_acquire);
}

IREE_API_EXPORT iree_status_t
iree_hal_memory_budget_query_loss(const iree_hal_memory_budget_t* budget) {
  return iree_status_clone((iree_status_t)iree_atomic_load(
      &budget->loss_status, iree_memory_order_acquire));
}

IREE_API_EXPORT bool iree_hal_memory_budget_try_reserve(
    iree_hal_memory_budget_t* budget, uint64_t byte_length,
    iree_hal_memory_credit_t* out_credit) {
  *out_credit = (iree_hal_memory_credit_t){0};
  if (!byte_length) {
    return true;
  }
  int64_t observed =
      iree_atomic_load(&budget->charged, iree_memory_order_relaxed);
  for (;;) {
    const uint64_t limit = iree_hal_memory_budget_limit(budget);
    if ((uint64_t)observed > limit ||
        byte_length > limit - (uint64_t)observed) {
      return false;
    }
    if (iree_atomic_compare_exchange_weak(
            &budget->charged, &observed,
            (int64_t)((uint64_t)observed + byte_length),
            iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
      iree_hal_memory_budget_retain(budget);
      *out_credit = (iree_hal_memory_credit_t){budget, byte_length};
      return true;
    }
  }
}

IREE_API_EXPORT iree_hal_memory_credit_t iree_hal_memory_credit_split(
    iree_hal_memory_credit_t* credit, uint64_t byte_length) {
  IREE_ASSERT_LE(byte_length, credit->byte_length);
  if (!byte_length) {
    return (iree_hal_memory_credit_t){0};
  }
  iree_hal_memory_credit_t result = {credit->budget, byte_length};
  credit->byte_length -= byte_length;
  if (credit->byte_length) {
    iree_hal_memory_budget_retain(result.budget);
  } else {
    credit->budget = NULL;
  }
  return result;
}

IREE_API_EXPORT void iree_hal_memory_credit_return(
    iree_hal_memory_credit_t* credit) {
  iree_hal_memory_credit_settle(credit, 0, iree_ok_status());
}

IREE_API_EXPORT void iree_hal_memory_credit_settle(
    iree_hal_memory_credit_t* credit, uint64_t abandoned_byte_length,
    iree_status_t status) {
  IREE_ASSERT_LE(abandoned_byte_length, credit->byte_length);
  iree_hal_memory_budget_t* budget = credit->budget;
  const uint64_t returned_byte_length =
      credit->byte_length - abandoned_byte_length;
  *credit = (iree_hal_memory_credit_t){0};
  if (!budget) {
    iree_status_ignore(status);
    return;
  }
  if (!iree_status_is_ok(status)) {
    intptr_t expected = (intptr_t)iree_ok_status();
    if (iree_atomic_compare_exchange_strong(
            &budget->loss_status, &expected, (intptr_t)status,
            iree_memory_order_release, iree_memory_order_relaxed)) {
      status = iree_ok_status();
    }
  }
  iree_status_ignore(status);
  iree_atomic_fetch_add(&budget->abandoned, (int64_t)abandoned_byte_length,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_sub(&budget->charged, (int64_t)returned_byte_length,
                        iree_memory_order_release);
  iree_hal_memory_budget_publish(budget);
  iree_hal_memory_budget_release(budget);
}
