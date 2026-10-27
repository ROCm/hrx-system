// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_TOKENIZER_CACHE_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_TOKENIZER_CACHE_H_

#include "iree/tokenizer/tokenizer.h"
#include "loomc/source.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_tokenizer_cache_t loom_serve_tokenizer_cache_t;
typedef struct loom_serve_tokenizer_t loom_serve_tokenizer_t;

// Disposition of an exact tokenizer acquisition.
typedef enum loom_serve_tokenizer_acquisition_e {
  // The caller owns construction and must complete the identity exactly once.
  LOOM_SERVE_TOKENIZER_ACQUISITION_PRODUCER = 0,
  // Another producer owns construction and will complete this subscription.
  LOOM_SERVE_TOKENIZER_ACQUISITION_PENDING = 1,
  // A retained completed tokenizer was returned directly to the caller.
  LOOM_SERVE_TOKENIZER_ACQUISITION_COMPLETE = 2,
} loom_serve_tokenizer_acquisition_t;

// Delivers one pending acquisition. Status and tokenizer ownership transfer to
// the callback independently of the producer and other subscribers.
typedef void (*loom_serve_tokenizer_completion_fn_t)(
    void* user_data, iree_status_t status, loom_serve_tokenizer_t* tokenizer);

// Caller-owned node retained only for a pending acquisition. The cache removes
// the node before invoking completion.
typedef struct loom_serve_tokenizer_subscription_t {
  // Next subscriber for the same exact in-flight tokenizer.
  struct loom_serve_tokenizer_subscription_t* next;
  // Completion transferring this subscriber's independent result.
  loom_serve_tokenizer_completion_fn_t completion;
  // Opaque subscriber state retained through completion.
  void* user_data;
} loom_serve_tokenizer_subscription_t;

// Allocates an empty process-local cache. The cache owns one reference to each
// completed tokenizer until trim or destruction.
iree_status_t loom_serve_tokenizer_cache_allocate(
    iree_allocator_t host_allocator, loom_serve_tokenizer_cache_t** out_cache);

// Releases cache ownership. All producers and model references must already
// have retired. NULL is accepted.
void loom_serve_tokenizer_cache_free(loom_serve_tokenizer_cache_t* cache);

// Releases completed entries with no active model or load borrower. In-flight
// producers and live borrowers preserve their entries.
void loom_serve_tokenizer_cache_trim(loom_serve_tokenizer_cache_t* cache);

// Acquires one tokenizer identified by its exact source bytes. The cache
// retains the first source so later acquisitions can compare immutable bytes
// directly. A new identity selects the caller as producer; an in-flight
// identity retains subscription; and a completed identity returns an
// independent retained reference. Pending completion may run after this
// function returns. source is borrowed only through the call; subscription
// must remain valid through its callback.
iree_status_t loom_serve_tokenizer_cache_acquire(
    loom_serve_tokenizer_cache_t* cache, loomc_source_t* source,
    loom_serve_tokenizer_subscription_t* subscription,
    loom_serve_tokenizer_acquisition_t* out_acquisition,
    loom_serve_tokenizer_t** out_tokenizer);

// Completes a producer acquisition exactly once. Success consumes tokenizer,
// retains the completed value in the cache, returns an independent reference
// to the producer, and delivers one reference to every subscriber. Failure
// consumes any partial tokenizer, broadcasts owned status copies, and removes
// the identity so a later caller can retry. status remains owned by the caller;
// a returned publication failure must be joined into its outcome.
iree_status_t loom_serve_tokenizer_cache_complete(
    loom_serve_tokenizer_cache_t* cache, const loomc_source_t* source,
    iree_status_t status, iree_tokenizer_t* tokenizer,
    loom_serve_tokenizer_t** out_tokenizer);

// Retains/releases one immutable shared tokenizer. NULL is accepted.
void loom_serve_tokenizer_retain(loom_serve_tokenizer_t* tokenizer);
void loom_serve_tokenizer_release(loom_serve_tokenizer_t* tokenizer);

// Borrows the immutable tokenizer value through the shared reference lifetime.
const iree_tokenizer_t* loom_serve_tokenizer_value(
    const loom_serve_tokenizer_t* tokenizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_TOKENIZER_CACHE_H_
