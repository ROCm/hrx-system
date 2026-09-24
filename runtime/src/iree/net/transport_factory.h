// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transport factory: creates connections and listeners for a transport type.
//
// A factory is a stateful object for one transport implementation. It owns any
// transport resources shared by connections and listeners while keeping those
// connection objects independently reference counted.
//
// Factories are registered with a transport registry at startup using
// HAL-driver style registration. See iree/net/transport_registry.h.
//
// Ownership:
//   - Factories are reference counted (create/retain/release). Any holder can
//     retain a factory to extend its lifetime independently.
//   - The transport registry retains factories on registration and releases
//     them when the registry is freed. Other holders may independently retain
//     factories obtained via registry lookup.
//   - Factories own shared resources (library handles, device lists).
//   - Factories do NOT own connections (callers own connections).

#ifndef IREE_NET_TRANSPORT_FACTORY_H_
#define IREE_NET_TRANSPORT_FACTORY_H_

#include "iree/async/api.h"
#include "iree/async/buffer_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Transport capabilities
//===----------------------------------------------------------------------===//

// Capabilities advertised by a transport. Query these before connecting to
// determine what features are available and optimize the connection strategy.
typedef enum iree_net_transport_capability_bits_e {
  IREE_NET_TRANSPORT_CAPABILITY_NONE = 0u,

  // Every accepted send is delivered or completed with a terminal failure;
  // data is never silently dropped. Without this, callers must implement
  // their own reliability layer.
  IREE_NET_TRANSPORT_CAPABILITY_RELIABLE = 1u << 0,

  // Successfully delivered data arrives in send order.
  IREE_NET_TRANSPORT_CAPABILITY_ORDERED = 1u << 1,

} iree_net_transport_capability_bits_t;
typedef uint32_t iree_net_transport_capabilities_t;

//===----------------------------------------------------------------------===//
// Callbacks
//===----------------------------------------------------------------------===//

typedef struct iree_net_connection_t iree_net_connection_t;

// Callback invoked when an async connect operation completes.
//
// On success, |status| is OK and |connection| is a new connection with
// ref_count=1. The caller takes ownership and must eventually release it.
//
// On failure, |status| contains the error and |connection| is NULL.
// The callback owns |status| in both cases and must propagate or release it.
typedef void(IREE_API_PTR* iree_net_transport_connect_fn_t)(
    void* user_data, iree_status_t status, iree_net_connection_t* connection);

// Connect completion callback and opaque caller data.
typedef struct iree_net_transport_connect_callback_t {
  // Function invoked when connection creation reaches a terminal result.
  iree_net_transport_connect_fn_t fn;
  // Opaque value passed to |fn|.
  void* user_data;
} iree_net_transport_connect_callback_t;

// Caller-owned storage for one connection attempt. Initialize before passing
// to factory_connect and keep at a stable address through its terminal callback
// and the initiating call's return. Cancellation callers must also join before
// storage is reused or deinitialized. No connection or native operation is
// exposed through this object.
typedef struct iree_net_transport_connect_operation_t {
  // Serializes factory binding, cancellation intent, and result publication.
  iree_slim_mutex_t mutex;
  // Factory-private access detached before the terminal callback.
  struct {
    // Records cancellation with mutex held; never invokes a user callback.
    void (*cancel_fn)(void* user_data);
    // Borrowed factory state kept alive until the binding is detached.
    void* user_data;
  } binding;
} iree_net_transport_connect_operation_t;

// Initializes idle connection-attempt storage. No allocation is performed.
IREE_API_EXPORT void iree_net_transport_connect_operation_initialize(
    iree_net_transport_connect_operation_t* operation);

// Deinitializes storage after the attempt and all caller accesses have joined.
IREE_API_EXPORT void iree_net_transport_connect_operation_deinitialize(
    iree_net_transport_connect_operation_t* operation);

// Requests cancellation after successful admission. Thread-safe and idempotent
// for that attempt; records intent without waiting for native or peer progress.
// The existing terminal connect callback joins all admitted work. Cancellation
// can lose to result publication; after publication this is a no-op. This is
// not a pre-cancel token for a future attempt. No user callback runs inline.
IREE_API_EXPORT void iree_net_transport_connect_operation_cancel(
    iree_net_transport_connect_operation_t* operation);

// Callback invoked when a listener accepts a new incoming connection.
//
// On success, |status| is OK and |connection| is a new connection with
// ref_count=1. The caller takes ownership. This callback may fire multiple
// times as new connections arrive.
//
// On failure, |status| contains the error and |connection| is NULL. The
// failed accept does not transfer a connection. It does not release or stop
// the listener; the owner may continue receiving accept callbacks or begin
// listener shutdown in response.
//
// The callback owns |status| in both cases and must propagate or release it.
typedef void(IREE_API_PTR* iree_net_listener_accept_fn_t)(
    void* user_data, iree_status_t status, iree_net_connection_t* connection);

// Listener accept callback and opaque caller data.
typedef struct iree_net_listener_accept_callback_t {
  // Function invoked for each accepted connection or listener error.
  iree_net_listener_accept_fn_t fn;
  // Opaque value passed to |fn|.
  void* user_data;
} iree_net_listener_accept_callback_t;

//===----------------------------------------------------------------------===//
// iree_net_listener_t
//===----------------------------------------------------------------------===//

typedef struct iree_net_listener_t iree_net_listener_t;
typedef struct iree_net_listener_vtable_t iree_net_listener_vtable_t;

// Callback invoked when a listener has fully stopped accepting connections.
//
// After this callback fires, no more accept callbacks will be delivered and
// iree_net_listener_free() is safe to call. This mirrors the carrier
// deactivate pattern where the callback signals safe-to-destroy.
typedef void(IREE_API_PTR* iree_net_listener_stopped_fn_t)(void* user_data);
typedef struct iree_net_listener_stopped_callback_t {
  // Function invoked after all listener operations have drained.
  iree_net_listener_stopped_fn_t fn;
  // Opaque value passed to |fn|.
  void* user_data;
} iree_net_listener_stopped_callback_t;

// A network listener that accepts incoming connections.
//
// Listeners are created via iree_net_transport_factory_create_listener() and
// begin accepting connections immediately. Each accepted connection is
// delivered via the accept callback provided at creation time.
//
// Lifecycle:
//   LISTENING -> stop() -> STOPPING -> stopped callback -> STOPPED -> free()
//
// Calling free() before the stopped callback fires is a programming error.
// Implementations may have pending I/O operations referencing the listener's
// internal state; the stopped callback guarantees these have completed.
struct iree_net_listener_t {
  // Vtable implementing listener operations.
  const iree_net_listener_vtable_t* vtable;
};

struct iree_net_listener_vtable_t {
  // Frees a fully stopped listener.
  void (*free)(iree_net_listener_t* listener);
  // Begins asynchronous listener shutdown and drain.
  iree_status_t (*stop)(iree_net_listener_t* listener,
                        iree_net_listener_stopped_callback_t callback);
  // Formats the actual bound transport address into caller storage.
  iree_status_t (*query_bound_address)(iree_net_listener_t* listener,
                                       iree_host_size_t buffer_capacity,
                                       char* buffer,
                                       iree_string_view_t* out_address);
};

// Frees a listener and releases all associated resources.
// The listener must be stopped (stopped callback must have fired) before
// freeing. Freeing before the stopped callback fires is a programming error.
static inline void iree_net_listener_free(iree_net_listener_t* listener) {
  if (listener) {
    listener->vtable->free(listener);
  }
}

// Initiates graceful shutdown of the listener.
//
// A stop issued on the listener's poll owner closes further successful accept
// publication before returning. In particular, an accept callback can stop
// after its desired connection count without taking ownership of more peers.
// Native accepts already in flight are retired by the listener. A concurrent
// stop from another thread may overlap an already-dispatched accept callback;
// the stopped callback remains the lifetime join. Native failures may still
// be reported through the accept callback during shutdown.
//
// This cancels any pending accept operations. The |callback| fires exactly
// once via the proactor when all pending accepts have drained and no more
// accept callbacks will be delivered. After the callback fires,
// iree_net_listener_free() is safe to call.
//
// The callback is always delivered asynchronously via the proactor, never
// synchronously from this call.
static inline iree_status_t iree_net_listener_stop(
    iree_net_listener_t* listener,
    iree_net_listener_stopped_callback_t callback) {
  if (!callback.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "listener stopped callback is required");
  }
  return listener->vtable->stop(listener, callback);
}

// Queries the address the listener is bound to, in the same string format
// accepted by connect and create_listener.
//
// This is the round-trip companion to create_listener: after binding to a
// dynamic address (e.g., TCP port 0), this returns the actual assigned address
// that a client would pass to connect to reach this listener.
//
// The result is written into the caller-provided |buffer| of |buffer_capacity|
// bytes, and |out_address| is set to a view into that buffer.
//
// Transport-specific formats include TCP `127.0.0.1:54321` addresses and
// loopback listener names.
static inline iree_status_t iree_net_listener_query_bound_address(
    iree_net_listener_t* listener, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_address) {
  return listener->vtable->query_bound_address(listener, buffer_capacity,
                                               buffer, out_address);
}

//===----------------------------------------------------------------------===//
// iree_net_transport_factory_t
//===----------------------------------------------------------------------===//

typedef struct iree_net_transport_factory_t iree_net_transport_factory_t;
typedef struct iree_net_transport_factory_vtable_t
    iree_net_transport_factory_vtable_t;

// A factory that creates connections and listeners for a specific transport.
//
// Each transport type provides its own factory implementation. Factories may
// retain shared platform or registration state that would be expensive to
// create for every connection.
//
// Factories are typically registered with a transport registry and looked up
// by scheme (for example, "tcp" or "loopback"). See
// iree/net/transport_registry.h.
//
// Concrete implementations embed this as their first member.
struct iree_net_transport_factory_t {
  // Reference count controlling factory lifetime.
  iree_atomic_ref_count_t ref_count;
  // Vtable implementing transport-specific factory operations.
  const iree_net_transport_factory_vtable_t* vtable;
};

struct iree_net_transport_factory_vtable_t {
  // Destroys the factory after its final reference is released.
  void (*destroy)(iree_net_transport_factory_t* factory);
  // Queries connection properties available from this factory.
  iree_net_transport_capabilities_t (*query_capabilities)(
      iree_net_transport_factory_t* factory);
  // Begins asynchronously connecting with operation->mutex held. Attach the
  // private cancellation binding before admitting owner work. On failure the
  // public wrapper clears the binding and no callback is owed. On success the
  // implementation detaches under that mutex before its terminal callback.
  iree_status_t (*connect)(iree_net_transport_factory_t* factory,
                           iree_string_view_t address,
                           iree_async_proactor_t* proactor,
                           iree_async_buffer_pool_t* receive_pool,
                           iree_net_transport_connect_callback_t callback,
                           iree_net_transport_connect_operation_t* operation);
  // Creates a listener and begins accepting peer connections.
  iree_status_t (*create_listener)(
      iree_net_transport_factory_t* factory, iree_string_view_t bind_address,
      iree_async_proactor_t* proactor, iree_async_buffer_pool_t* receive_pool,
      iree_net_listener_accept_callback_t accept_callback,
      iree_allocator_t host_allocator, iree_net_listener_t** out_listener);
};

// Initializes base factory fields. Called by transport implementations.
static inline void iree_net_transport_factory_initialize(
    const iree_net_transport_factory_vtable_t* vtable,
    iree_net_transport_factory_t* out_factory) {
  IREE_ASSERT_ARGUMENT(vtable);
  IREE_ASSERT_ARGUMENT(out_factory);
  iree_atomic_ref_count_init(&out_factory->ref_count);
  out_factory->vtable = vtable;
}

// Retains a reference to the factory.
static inline void iree_net_transport_factory_retain(
    iree_net_transport_factory_t* factory) {
  if (!factory) {
    return;
  }
  iree_atomic_ref_count_inc(&factory->ref_count);
}

// Releases a reference to the factory.
// When the last reference is released, the factory is destroyed and all
// associated resources are freed.
static inline void iree_net_transport_factory_release(
    iree_net_transport_factory_t* factory) {
  if (!factory) {
    return;
  }
  if (iree_atomic_ref_count_dec(&factory->ref_count) == 1) {
    factory->vtable->destroy(factory);
  }
}

// Returns the capabilities supported by this transport before connecting.
static inline iree_net_transport_capabilities_t
iree_net_transport_factory_query_capabilities(
    iree_net_transport_factory_t* factory) {
  return factory->vtable->query_capabilities(factory);
}

// Initiates an asynchronous connection to the given address.
//
// The |address| format is transport-specific:
// TCP accepts `host:port`; loopback accepts the registered listener name.
// Its storage is needed only for the duration of this call.
//
// The |proactor| handles I/O completions for this connection. All callbacks
// for this connection will fire on the proactor's thread.
//
// The |receive_pool| provides buffers for incoming data during connection setup
// when required by the transport. Transports with connection-specific receive
// registration may create their own receive pools instead. The pending connect
// and resulting connection retain all asynchronous resources they require, so
// the caller may release its proactor and pool references after this call.
//
// The |callback| fires exactly once via the proactor when the connection
// succeeds or fails. The callback is always delivered asynchronously, never
// synchronously from this call. On success, the callback receives a new
// connection that the caller owns.
//
// The caller supplies initialized, stable |operation| storage for cancellation
// and lifetime. The callback may run on the proactor before this call returns;
// callers join both before releasing storage. No factory references the
// operation after its callback begins. Reuse also requires joining concurrent
// cancellation callers; a late cancel must not reach a new attempt.
//
// Returns synchronous errors immediately (e.g., invalid address format,
// allocation failure). Asynchronous errors (connection refused, timeout) are
// delivered via the callback.
static inline iree_status_t iree_net_transport_factory_connect(
    iree_net_transport_factory_t* factory, iree_string_view_t address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* receive_pool,
    iree_net_transport_connect_callback_t callback,
    iree_net_transport_connect_operation_t* operation) {
  if (!callback.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "transport connect callback is required");
  }
  if (!operation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "transport connect operation is required");
  }
  iree_slim_mutex_lock(&operation->mutex);
  iree_status_t status = iree_ok_status();
  if (operation->binding.cancel_fn) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "transport connect operation is already pending");
  } else {
    status = factory->vtable->connect(factory, address, proactor, receive_pool,
                                      callback, operation);
    if (!iree_status_is_ok(status)) {
      operation->binding.cancel_fn = NULL;
      operation->binding.user_data = NULL;
    }
  }
  iree_slim_mutex_unlock(&operation->mutex);
  return status;
}

// Creates a listener that accepts incoming connections on the given address.
//
// The |bind_address| format is transport-specific:
// TCP accepts `host:port`; loopback accepts a listener name.
// Its storage is needed only for the duration of this call.
//
// The |proactor| handles I/O completions. All accept callbacks fire on the
// proactor's thread.
//
// The |receive_pool| provides buffers for accepted connections' initial
// receives when required by the transport. Transports with connection-specific
// receive registration may create their own receive pools instead. A
// successful listener retains all asynchronous resources required by itself
// and accepted connections, so the caller may release its proactor and pool
// references.
//
// The |accept_callback| fires each time a new connection is accepted. On
// success, the callback receives a new connection that the caller owns. The
// callback may fire multiple times, once per accepted connection.
//
// On success, |*out_listener| receives the new listener. The caller owns it
// and must eventually stop and free it.
static inline iree_status_t iree_net_transport_factory_create_listener(
    iree_net_transport_factory_t* factory, iree_string_view_t bind_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* receive_pool,
    iree_net_listener_accept_callback_t accept_callback,
    iree_allocator_t host_allocator, iree_net_listener_t** out_listener) {
  if (!out_listener) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "listener output is required");
  }
  *out_listener = NULL;
  if (!accept_callback.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "listener accept callback is required");
  }
  return factory->vtable->create_listener(factory, bind_address, proactor,
                                          receive_pool, accept_callback,
                                          host_allocator, out_listener);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_TRANSPORT_FACTORY_H_
