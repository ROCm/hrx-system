// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_BUDGET_H_
#define IREE_HAL_MEMORY_BUDGET_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_notification_t iree_async_notification_t;
typedef struct iree_hal_memory_budget_t iree_hal_memory_budget_t;

// Move-owned physical capacity. A credit covers reserved, pending, live, and
// retiring native bytes, independently of execution queues or buffer views.
// It must be returned or transferred exactly once. Zero initialization is
// empty.
typedef struct iree_hal_memory_credit_t {
  // Retained account, or NULL for an empty credit.
  iree_hal_memory_budget_t* budget;
  // Native payload bytes exclusively owned by this credit.
  uint64_t byte_length;
} iree_hal_memory_credit_t;

// Creates an explicit physical capacity account shared by any number of native
// sources. A zero limit admits no bytes. The optional notification is retained
// and signalled whenever capacity returns or the limit increases; its progress
// owner must outlive the budget and accepted observations.
IREE_API_EXPORT iree_status_t iree_hal_memory_budget_create(
    uint64_t limit, iree_async_notification_t* notification,
    iree_allocator_t host_allocator, iree_hal_memory_budget_t** out_budget);
IREE_API_EXPORT void iree_hal_memory_budget_retain(
    iree_hal_memory_budget_t* budget);
IREE_API_EXPORT void iree_hal_memory_budget_release(
    iree_hal_memory_budget_t* budget);

// Changes future admission. Reducing the limit does not revoke accepted owners.
IREE_API_EXPORT void iree_hal_memory_budget_set_limit(
    iree_hal_memory_budget_t* budget, uint64_t limit);
IREE_API_EXPORT uint64_t
iree_hal_memory_budget_limit(const iree_hal_memory_budget_t* budget);
IREE_API_EXPORT uint64_t
iree_hal_memory_budget_charged(const iree_hal_memory_budget_t* budget);
IREE_API_EXPORT uint64_t
iree_hal_memory_budget_abandoned(const iree_hal_memory_budget_t* budget);

// Clones the first complete native loss diagnostic. Loss reduces available
// capacity but does not poison unrelated acquisitions with adequate capacity.
IREE_API_EXPORT iree_status_t
iree_hal_memory_budget_query_loss(const iree_hal_memory_budget_t* budget);

// Atomically admits the complete amount or leaves the account unchanged.
// Multi-account callers reserve all credits before submitting any native work
// and return the earlier credits when a later account denies admission.
IREE_API_EXPORT bool iree_hal_memory_budget_try_reserve(
    iree_hal_memory_budget_t* budget, uint64_t byte_length,
    iree_hal_memory_credit_t* out_credit);

// Transfers part of a credit without changing the physical charge. The caller
// owns |credit| exclusively and supplies a length no larger than its balance.
IREE_API_EXPORT iree_hal_memory_credit_t iree_hal_memory_credit_split(
    iree_hal_memory_credit_t* credit, uint64_t byte_length);

// Returns unused admission, or capacity whose native return is established.
// Logical buffer release and public operation cancellation do not establish
// native return. Clears the consumed credit.
IREE_API_EXPORT void iree_hal_memory_credit_return(
    iree_hal_memory_credit_t* credit);

// Settles a terminal native outcome: returned bytes become available, while
// abandoned bytes remain charged for the account lifetime. Consumes |status|
// with its complete diagnostic payload and clears the credit. The native owner
// supplies an abandoned length no larger than the held credit.
IREE_API_EXPORT void iree_hal_memory_credit_settle(
    iree_hal_memory_credit_t* credit, uint64_t abandoned_byte_length,
    iree_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_BUDGET_H_
