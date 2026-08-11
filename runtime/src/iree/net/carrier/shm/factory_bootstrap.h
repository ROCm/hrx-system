// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_NET_CARRIER_SHM_FACTORY_BOOTSTRAP_H_
#define IREE_NET_CARRIER_SHM_FACTORY_BOOTSTRAP_H_

#include "iree/async/primitive.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/net/connection.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_shm_bootstrap_t iree_net_shm_bootstrap_t;
typedef struct iree_net_shm_factory_t iree_net_shm_factory_t;

// Selects which side of the cross-process handshake the worker performs.
typedef enum iree_net_shm_bootstrap_role_e {
  IREE_NET_SHM_BOOTSTRAP_ROLE_SERVER = 0,
  IREE_NET_SHM_BOOTSTRAP_ROLE_CLIENT = 1,
} iree_net_shm_bootstrap_role_t;

// Describes a terminal bootstrap outcome independent of its failure status.
enum iree_net_shm_bootstrap_completion_flag_bits_e {
  IREE_NET_SHM_BOOTSTRAP_COMPLETION_FLAG_NONE = 0u,

  // Cooperative cancellation completed before a connection was assembled.
  // The completion status is OK and |connection| is NULL.
  IREE_NET_SHM_BOOTSTRAP_COMPLETION_FLAG_CANCELLED = 1u << 0,
};
typedef uint32_t iree_net_shm_bootstrap_completion_flags_t;

// Receives the terminal bootstrap result on the proactor thread.
//
// On success, |status| is OK and |connection| is transferred to the callback.
// On failure, |connection| is NULL. The bootstrap object is valid only for the
// duration of the callback and is destroyed after it returns.
typedef void (*iree_net_shm_bootstrap_completion_fn_t)(
    void* user_data, iree_status_t status,
    iree_net_shm_bootstrap_completion_flags_t flags,
    iree_net_connection_t* connection);

typedef struct iree_net_shm_bootstrap_callback_t {
  // Function invoked with the terminal result.
  iree_net_shm_bootstrap_completion_fn_t fn;
  // Opaque value passed to |fn|.
  void* user_data;
} iree_net_shm_bootstrap_callback_t;

// Prepares a cold-path cross-process handshake worker.
//
// The completion notification is armed on |proactor| before this returns, but
// the worker remains suspended until iree_net_shm_bootstrap_launch() is called.
// This lets listeners publish the bootstrap in their cancellation set before
// peer I/O can begin.
//
// The call must be serialized with |proactor| polling through the subsequent
// iree_net_shm_bootstrap_launch(). Production factory paths perform both from a
// proactor callback. This keeps the submitted completion wait from dispatching
// before the caller publishes the suspended bootstrap.
//
// On success, |channel| is consumed and set to NONE. On failure, the caller
// retains |channel| and no callback will fire.
iree_status_t iree_net_shm_bootstrap_prepare(
    iree_net_shm_factory_t* factory, iree_net_shm_bootstrap_role_t role,
    iree_async_primitive_t* channel, iree_async_proactor_t* proactor,
    iree_net_shm_bootstrap_callback_t callback, iree_allocator_t host_allocator,
    iree_net_shm_bootstrap_t** out_bootstrap);

// Starts a prepared bootstrap worker. Must be called exactly once.
void iree_net_shm_bootstrap_launch(iree_net_shm_bootstrap_t* bootstrap);

// Requests cancellation and interrupts any blocking peer I/O.
//
// Cancellation is terminal and idempotent. The completion callback still fires
// exactly once after the worker has exited and all handshake resources have
// been reclaimed.
void iree_net_shm_bootstrap_cancel(iree_net_shm_bootstrap_t* bootstrap);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_SHM_FACTORY_BOOTSTRAP_H_
