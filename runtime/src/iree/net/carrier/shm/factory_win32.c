// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cross-process SHM factory operations for Windows.
//
// Named pipes provide the bootstrap channel. ConnectNamedPipe admission is
// monitored by the proactor, while each handle exchange runs on a dedicated
// cancellable bootstrap worker. No blocking peer I/O runs on the proactor
// thread.

#include "iree/net/carrier/shm/factory_state.h"

#if defined(IREE_PLATFORM_WINDOWS)

#include <string.h>
#include <windows.h>

#include "iree/async/event.h"
#include "iree/async/operations/scheduling.h"
#include "iree/net/carrier/shm/factory_bootstrap.h"

// Prefix for Windows named pipe paths.
static const WCHAR iree_net_shm_pipe_prefix[] = L"\\\\.\\pipe\\";
#define IREE_NET_SHM_PIPE_PREFIX_LENGTH 9

// Maximum total pipe path length in wide characters, including the terminator.
#define IREE_NET_SHM_MAX_PIPE_PATH_LENGTH (MAX_PATH + 1)

// Bounds server-side bootstrap threads per listener. Additional clients remain
// in the named-pipe connection queue until a worker completes.
#define IREE_NET_SHM_WIN32_MAX_PENDING_BOOTSTRAPS 16

static iree_string_view_t iree_net_shm_win32_strip_pipe_prefix(
    iree_string_view_t address) {
  return iree_make_string_view(address.data + 5, address.size - 5);
}

static iree_status_t iree_net_shm_win32_build_pipe_path(iree_string_view_t name,
                                                        WCHAR* out_path,
                                                        int* out_path_length) {
  if (name.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "pipe name is empty");
  }

  int wide_name_length = MultiByteToWideChar(
      CP_UTF8, MB_ERR_INVALID_CHARS, name.data, (int)name.size, NULL, 0);
  if (wide_name_length <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipe name contains invalid UTF-8");
  }

  int total_length = IREE_NET_SHM_PIPE_PREFIX_LENGTH + wide_name_length;
  if (total_length >= IREE_NET_SHM_MAX_PIPE_PATH_LENGTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipe path too long (%d wide chars, max %d)",
                            total_length,
                            IREE_NET_SHM_MAX_PIPE_PATH_LENGTH - 1);
  }

  memcpy(out_path, iree_net_shm_pipe_prefix,
         IREE_NET_SHM_PIPE_PREFIX_LENGTH * sizeof(WCHAR));
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, name.data, (int)name.size,
                      out_path + IREE_NET_SHM_PIPE_PREFIX_LENGTH,
                      wide_name_length);
  out_path[total_length] = L'\0';
  *out_path_length = total_length;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Cross-process listener
//===----------------------------------------------------------------------===//

typedef enum iree_net_shm_win32_listener_state_e {
  IREE_NET_SHM_WIN32_LISTENER_STATE_LISTENING = 0,
  IREE_NET_SHM_WIN32_LISTENER_STATE_STOPPING,
  IREE_NET_SHM_WIN32_LISTENER_STATE_STOPPED,
} iree_net_shm_win32_listener_state_t;

typedef struct iree_net_shm_win32_listener_t iree_net_shm_win32_listener_t;

// One accepted pipe being bootstrapped off the proactor thread.
typedef struct iree_net_shm_win32_pending_bootstrap_t {
  // Listener receiving the terminal bootstrap callback.
  iree_net_shm_win32_listener_t* listener;
  // Prepared worker operation, valid until its callback begins.
  iree_net_shm_bootstrap_t* bootstrap;
  // Intrusive linkage in the listener's cancellation set.
  struct iree_net_shm_win32_pending_bootstrap_t* next;
} iree_net_shm_win32_pending_bootstrap_t;

struct iree_net_shm_win32_listener_t {
  iree_net_listener_t base;
  // Factory retained for listener lifetime.
  iree_net_shm_factory_t* factory;
  // Proactor delivering accept and bootstrap completions.
  iree_async_proactor_t* proactor;
  // Current named-pipe instance, or INVALID_HANDLE_VALUE when consumed.
  HANDLE pipe_handle;
  // Event signaled by the current ConnectNamedPipe operation.
  iree_async_event_t* event;
  // Kernel state for the current ConnectNamedPipe operation.
  OVERLAPPED overlapped;
  // Proactor operation awaiting |event|.
  iree_async_event_wait_operation_t event_wait_operation;
  // Consumer accept callback.
  struct {
    // Function receiving accepted connections and terminal accept errors.
    iree_net_listener_accept_callback_t fn;
    // Opaque value passed to |fn|.
    void* user_data;
  } accept;
  // Protects lifecycle and pending bootstrap fields from stop callers.
  iree_slim_mutex_t mutex;
  // Current listener lifecycle state.
  iree_net_shm_win32_listener_state_t state;
  // True while |event_wait_operation| is owned by the proactor.
  bool accept_pending;
  // True while the kernel may access |overlapped|.
  bool connect_operation_pending;
  // True while |stop_operation| is owned by the proactor.
  bool stop_operation_pending;
  // True after the stopped callback has been claimed for delivery.
  bool stopped_delivered;
  // Accepted channels currently running bootstrap workers.
  iree_net_shm_win32_pending_bootstrap_t* pending_bootstraps;
  // Number of entries in |pending_bootstraps|.
  iree_host_size_t pending_bootstrap_count;
  // Callback delivered once accept and bootstrap work has drained.
  iree_net_listener_stopped_callback_t stopped_callback;
  // NOP used when stop begins with no asynchronous work to drain.
  iree_async_nop_operation_t stop_operation;
  // Allocator used for listener and pending bootstrap state.
  iree_allocator_t host_allocator;
  // Full address string retained for query_bound_address.
  iree_host_size_t address_length;
  // Null-terminated address storage.
  char address[];
};

static const iree_net_listener_vtable_t iree_net_shm_win32_listener_vtable;

static void iree_net_shm_win32_listener_accept_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags);

static iree_status_t iree_net_shm_win32_listener_rearm(
    iree_net_shm_win32_listener_t* listener);

static void iree_net_shm_win32_listener_maybe_finish_stop(
    iree_net_shm_win32_listener_t* listener) {
  iree_net_listener_stopped_callback_t callback = {0};
  iree_slim_mutex_lock(&listener->mutex);
  if (listener->state == IREE_NET_SHM_WIN32_LISTENER_STATE_STOPPING &&
      !listener->accept_pending && !listener->connect_operation_pending &&
      !listener->stop_operation_pending && !listener->pending_bootstraps &&
      !listener->stopped_delivered) {
    listener->state = IREE_NET_SHM_WIN32_LISTENER_STATE_STOPPED;
    listener->stopped_delivered = true;
    callback = listener->stopped_callback;
  }
  iree_slim_mutex_unlock(&listener->mutex);
  if (callback.fn) callback.fn(callback.user_data);
}

// Cancels and exactly retires a pending ConnectNamedPipe operation. This is
// used only when the proactor event wait itself failed; normal listener stop
// lets the event wait observe the kernel cancellation completion directly.
static iree_status_t iree_net_shm_win32_listener_cancel_and_await_connect(
    iree_net_shm_win32_listener_t* listener) {
  if (!listener->connect_operation_pending) return iree_ok_status();

  iree_status_t status = iree_ok_status();
  if (!CancelIoEx(listener->pipe_handle, &listener->overlapped)) {
    DWORD error = GetLastError();
    if (error != ERROR_NOT_FOUND) {
      status =
          iree_make_status(iree_status_code_from_win32_error(error),
                           "CancelIoEx failed while retiring ConnectNamedPipe");
    }
  }

  DWORD bytes_transferred = 0;
  if (!GetOverlappedResult(listener->pipe_handle, &listener->overlapped,
                           &bytes_transferred, /*bWait=*/TRUE)) {
    DWORD error = GetLastError();
    if (error != ERROR_OPERATION_ABORTED) {
      status = iree_status_join(
          status, iree_make_status(iree_status_code_from_win32_error(error),
                                   "failed to retire ConnectNamedPipe"));
    }
  }
  listener->connect_operation_pending = false;
  return status;
}

static iree_status_t iree_net_shm_win32_listener_create_pipe(
    iree_net_shm_win32_listener_t* listener) {
  IREE_ASSERT(listener->pipe_handle == INVALID_HANDLE_VALUE);
  iree_string_view_t name = iree_net_shm_win32_strip_pipe_prefix(
      iree_make_string_view(listener->address, listener->address_length));

  WCHAR pipe_path[IREE_NET_SHM_MAX_PIPE_PATH_LENGTH];
  int pipe_path_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_net_shm_win32_build_pipe_path(name, pipe_path, &pipe_path_length));

  listener->pipe_handle =
      CreateNamedPipeW(pipe_path, PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
                       PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT |
                           PIPE_REJECT_REMOTE_CLIENTS,
                       PIPE_UNLIMITED_INSTANCES, 4096, 4096, 0, NULL);
  if (listener->pipe_handle == INVALID_HANDLE_VALUE) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "CreateNamedPipeW failed for pipe '%.*s'",
                            (int)name.size, name.data);
  }
  return iree_ok_status();
}

// Starts ConnectNamedPipe and arms its event wait. The listener mutex is held
// across this function so stop cannot observe partially submitted state.
static iree_status_t iree_net_shm_win32_listener_start_accept_locked(
    iree_net_shm_win32_listener_t* listener) {
  IREE_ASSERT(!listener->accept_pending);
  IREE_ASSERT(!listener->connect_operation_pending);
  IREE_ASSERT(listener->pipe_handle != INVALID_HANDLE_VALUE);

  HANDLE event_handle = (HANDLE)listener->event->primitive.value.win32_handle;
  if (!ResetEvent(event_handle)) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "ResetEvent failed for ConnectNamedPipe");
  }

  memset(&listener->overlapped, 0, sizeof(listener->overlapped));
  listener->overlapped.hEvent = event_handle;

  iree_status_t status = iree_ok_status();
  BOOL connected =
      ConnectNamedPipe(listener->pipe_handle, &listener->overlapped);
  if (connected) {
    status = iree_async_event_set(listener->event);
  } else {
    DWORD error = GetLastError();
    if (error == ERROR_IO_PENDING) {
      listener->connect_operation_pending = true;
    } else if (error == ERROR_PIPE_CONNECTED) {
      status = iree_async_event_set(listener->event);
    } else {
      status = iree_make_status(iree_status_code_from_win32_error(error),
                                "ConnectNamedPipe failed");
    }
  }

  if (iree_status_is_ok(status)) {
    memset(&listener->event_wait_operation, 0,
           sizeof(listener->event_wait_operation));
    iree_async_operation_initialize(
        &listener->event_wait_operation.base,
        IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT, IREE_ASYNC_OPERATION_FLAG_NONE,
        iree_net_shm_win32_listener_accept_complete, listener);
    listener->event_wait_operation.event = listener->event;
    status = iree_async_proactor_submit_one(
        listener->proactor, &listener->event_wait_operation.base);
    listener->accept_pending = iree_status_is_ok(status);
  }

  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, iree_net_shm_win32_listener_cancel_and_await_connect(listener));
    CloseHandle(listener->pipe_handle);
    listener->pipe_handle = INVALID_HANDLE_VALUE;
  }
  return status;
}

static iree_status_t iree_net_shm_win32_listener_rearm(
    iree_net_shm_win32_listener_t* listener) {
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&listener->mutex);
  bool can_accept =
      listener->state == IREE_NET_SHM_WIN32_LISTENER_STATE_LISTENING &&
      !listener->accept_pending &&
      listener->pending_bootstrap_count <
          IREE_NET_SHM_WIN32_MAX_PENDING_BOOTSTRAPS;
  if (can_accept) {
    status = iree_net_shm_win32_listener_create_pipe(listener);
    if (iree_status_is_ok(status)) {
      status = iree_net_shm_win32_listener_start_accept_locked(listener);
    }
  }
  iree_slim_mutex_unlock(&listener->mutex);
  return status;
}

static void iree_net_shm_win32_pending_bootstrap_complete(
    void* user_data, iree_status_t status,
    iree_net_shm_bootstrap_completion_flags_t flags,
    iree_net_connection_t* connection) {
  iree_net_shm_win32_pending_bootstrap_t* pending =
      (iree_net_shm_win32_pending_bootstrap_t*)user_data;
  iree_net_shm_win32_listener_t* listener = pending->listener;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&listener->mutex);
  iree_net_shm_win32_pending_bootstrap_t** previous =
      &listener->pending_bootstraps;
  while (*previous && *previous != pending) {
    previous = &(*previous)->next;
  }
  IREE_ASSERT(*previous == pending,
              "completed SHM bootstrap missing from listener");
  *previous = pending->next;
  --listener->pending_bootstrap_count;
  iree_slim_mutex_unlock(&listener->mutex);
  iree_allocator_free(listener->host_allocator, pending);

  if (iree_any_bit_set(flags,
                       IREE_NET_SHM_BOOTSTRAP_COMPLETION_FLAG_CANCELLED)) {
    IREE_ASSERT(!connection);
    if (!iree_status_is_ok(status)) {
      listener->accept.fn(listener->accept.user_data, status, NULL);
    }
  } else {
    listener->accept.fn(listener->accept.user_data, status, connection);
  }

  iree_status_t rearm_status = iree_net_shm_win32_listener_rearm(listener);
  if (!iree_status_is_ok(rearm_status)) {
    listener->accept.fn(listener->accept.user_data, rearm_status, NULL);
  }
  iree_net_shm_win32_listener_maybe_finish_stop(listener);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_net_shm_win32_listener_start_bootstrap(
    iree_net_shm_win32_listener_t* listener) {
  IREE_ASSERT(listener->pipe_handle != INVALID_HANDLE_VALUE);
  iree_async_primitive_t channel =
      iree_async_primitive_from_win32_handle((uintptr_t)listener->pipe_handle);
  listener->pipe_handle = INVALID_HANDLE_VALUE;

  iree_net_shm_win32_pending_bootstrap_t* pending = NULL;
  iree_status_t status = iree_allocator_malloc(
      listener->host_allocator, sizeof(*pending), (void**)&pending);
  if (iree_status_is_ok(status)) {
    memset(pending, 0, sizeof(*pending));
    pending->listener = listener;
    status = iree_net_shm_bootstrap_prepare(
        listener->factory, IREE_NET_SHM_BOOTSTRAP_ROLE_SERVER, &channel,
        listener->proactor,
        (iree_net_shm_bootstrap_callback_t){
            .fn = iree_net_shm_win32_pending_bootstrap_complete,
            .user_data = pending,
        },
        listener->host_allocator, &pending->bootstrap);
  }

  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&listener->mutex);
    pending->next = listener->pending_bootstraps;
    listener->pending_bootstraps = pending;
    ++listener->pending_bootstrap_count;
    bool cancel_immediately =
        listener->state != IREE_NET_SHM_WIN32_LISTENER_STATE_LISTENING;
    iree_slim_mutex_unlock(&listener->mutex);
    if (cancel_immediately) {
      iree_net_shm_bootstrap_cancel(pending->bootstrap);
    }
    iree_net_shm_bootstrap_launch(pending->bootstrap);
  } else {
    iree_allocator_free(listener->host_allocator, pending);
    iree_async_primitive_close(&channel);
  }
  return status;
}

static void iree_net_shm_win32_listener_accept_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_shm_win32_listener_t* listener =
      (iree_net_shm_win32_listener_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT(!iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE));

  iree_slim_mutex_lock(&listener->mutex);
  IREE_ASSERT(listener->accept_pending);
  listener->accept_pending = false;
  bool is_stopping =
      listener->state != IREE_NET_SHM_WIN32_LISTENER_STATE_LISTENING;
  iree_slim_mutex_unlock(&listener->mutex);

  bool connect_cancelled = false;
  if (iree_status_is_ok(status) && listener->connect_operation_pending) {
    DWORD bytes_transferred = 0;
    if (!GetOverlappedResult(listener->pipe_handle, &listener->overlapped,
                             &bytes_transferred, /*bWait=*/FALSE)) {
      DWORD error = GetLastError();
      if (is_stopping && error == ERROR_OPERATION_ABORTED) {
        connect_cancelled = true;
      } else {
        status = iree_make_status(iree_status_code_from_win32_error(error),
                                  "ConnectNamedPipe failed");
      }
    }
    listener->connect_operation_pending = false;
  } else if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, iree_net_shm_win32_listener_cancel_and_await_connect(listener));
  }

  if (iree_status_is_ok(status) && !connect_cancelled) {
    status = iree_net_shm_win32_listener_start_bootstrap(listener);
  } else if (listener->pipe_handle != INVALID_HANDLE_VALUE) {
    CloseHandle(listener->pipe_handle);
    listener->pipe_handle = INVALID_HANDLE_VALUE;
  }

  if (!iree_status_is_ok(status)) {
    listener->accept.fn(listener->accept.user_data, status, NULL);
  }

  iree_status_t rearm_status = iree_net_shm_win32_listener_rearm(listener);
  if (!iree_status_is_ok(rearm_status)) {
    listener->accept.fn(listener->accept.user_data, rearm_status, NULL);
  }
  iree_net_shm_win32_listener_maybe_finish_stop(listener);
  IREE_TRACE_ZONE_END(z0);
}

static void iree_net_shm_win32_listener_stop_deferred_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_shm_win32_listener_t* listener =
      (iree_net_shm_win32_listener_t*)user_data;
  iree_slim_mutex_lock(&listener->mutex);
  IREE_ASSERT(listener->stop_operation_pending);
  listener->stop_operation_pending = false;
  iree_slim_mutex_unlock(&listener->mutex);
  if (!iree_status_is_ok(status)) {
    listener->accept.fn(listener->accept.user_data, status, NULL);
  }
  iree_net_shm_win32_listener_maybe_finish_stop(listener);
}

static void iree_net_shm_win32_listener_free(
    iree_net_listener_t* base_listener) {
  iree_net_shm_win32_listener_t* listener =
      (iree_net_shm_win32_listener_t*)base_listener;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT(listener->stopped_delivered,
              "SHM listener freed before stopped callback");
  IREE_ASSERT(!listener->accept_pending);
  IREE_ASSERT(!listener->connect_operation_pending);
  IREE_ASSERT(!listener->stop_operation_pending);
  IREE_ASSERT(!listener->pending_bootstraps);
  if (listener->pipe_handle != INVALID_HANDLE_VALUE) {
    CloseHandle(listener->pipe_handle);
  }
  iree_async_event_release(listener->event);
  iree_async_proactor_release(listener->proactor);
  iree_net_transport_factory_release(&listener->factory->base);
  iree_slim_mutex_deinitialize(&listener->mutex);
  iree_allocator_t host_allocator = listener->host_allocator;
  iree_allocator_free(host_allocator, listener);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_net_shm_win32_listener_stop(
    iree_net_listener_t* base_listener,
    iree_net_listener_stopped_callback_t callback) {
  iree_net_shm_win32_listener_t* listener =
      (iree_net_shm_win32_listener_t*)base_listener;
  IREE_TRACE_ZONE_BEGIN(z0);

  bool rearm_after_failure = false;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&listener->mutex);
  if (listener->state != IREE_NET_SHM_WIN32_LISTENER_STATE_LISTENING) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "SHM listener is already stopping");
  } else {
    listener->state = IREE_NET_SHM_WIN32_LISTENER_STATE_STOPPING;
    listener->stopped_callback = callback;
    for (iree_net_shm_win32_pending_bootstrap_t* pending =
             listener->pending_bootstraps;
         pending; pending = pending->next) {
      iree_net_shm_bootstrap_cancel(pending->bootstrap);
    }

    if (listener->accept_pending && listener->connect_operation_pending &&
        !CancelIoEx(listener->pipe_handle, &listener->overlapped)) {
      DWORD error = GetLastError();
      if (error != ERROR_NOT_FOUND) {
        status = iree_make_status(iree_status_code_from_win32_error(error),
                                  "CancelIoEx failed for ConnectNamedPipe");
      }
    }

    if (!listener->accept_pending && !listener->pending_bootstraps) {
      iree_async_operation_initialize(
          &listener->stop_operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
          IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS,
          iree_net_shm_win32_listener_stop_deferred_complete, listener);
      listener->stop_operation_pending = true;
      iree_status_t submit_status = iree_async_proactor_submit_one(
          listener->proactor, &listener->stop_operation.base);
      if (!iree_status_is_ok(submit_status)) {
        listener->stop_operation_pending = false;
        listener->state = IREE_NET_SHM_WIN32_LISTENER_STATE_LISTENING;
        listener->stopped_callback = (iree_net_listener_stopped_callback_t){0};
        rearm_after_failure = true;
      }
      status = iree_status_join(status, submit_status);
    }
  }
  iree_slim_mutex_unlock(&listener->mutex);

  if (rearm_after_failure) {
    status =
        iree_status_join(status, iree_net_shm_win32_listener_rearm(listener));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_net_shm_win32_listener_query_bound_address(
    iree_net_listener_t* base_listener, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_address) {
  iree_net_shm_win32_listener_t* listener =
      (iree_net_shm_win32_listener_t*)base_listener;
  if (buffer_capacity < listener->address_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer too small for bound address");
  }
  memcpy(buffer, listener->address, listener->address_length);
  *out_address = iree_make_string_view(buffer, listener->address_length);
  return iree_ok_status();
}

static const iree_net_listener_vtable_t iree_net_shm_win32_listener_vtable = {
    .free = iree_net_shm_win32_listener_free,
    .stop = iree_net_shm_win32_listener_stop,
    .query_bound_address = iree_net_shm_win32_listener_query_bound_address,
};

iree_status_t iree_net_shm_factory_create_listener_win32(
    iree_net_shm_factory_t* factory, iree_string_view_t bind_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_listener_accept_callback_t accept_callback, void* user_data,
    iree_allocator_t host_allocator, iree_net_listener_t** out_listener) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)recv_pool;
  *out_listener = NULL;

  iree_string_view_t name = iree_net_shm_win32_strip_pipe_prefix(bind_address);
  WCHAR pipe_path[IREE_NET_SHM_MAX_PIPE_PATH_LENGTH];
  int pipe_path_length = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_net_shm_win32_build_pipe_path(name, pipe_path, &pipe_path_length));

  iree_async_event_t* event = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_async_event_create(proactor, &event));

  iree_host_size_t total_size = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      iree_sizeof_struct(iree_net_shm_win32_listener_t), &total_size,
      IREE_STRUCT_FIELD_FAM(bind_address.size + 1, char));
  iree_net_shm_win32_listener_t* listener = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&listener);
  }
  if (iree_status_is_ok(status)) {
    memset(listener, 0, total_size);
    listener->base.vtable = &iree_net_shm_win32_listener_vtable;
    listener->factory = factory;
    iree_net_transport_factory_retain(&factory->base);
    listener->proactor = proactor;
    iree_async_proactor_retain(proactor);
    listener->pipe_handle = INVALID_HANDLE_VALUE;
    listener->event = event;
    listener->accept.fn = accept_callback;
    listener->accept.user_data = user_data;
    iree_slim_mutex_initialize(&listener->mutex);
    listener->host_allocator = host_allocator;
    listener->address_length = bind_address.size;
    memcpy(listener->address, bind_address.data, bind_address.size);
    listener->address[bind_address.size] = '\0';
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_shm_win32_listener_rearm(listener);
  }

  if (iree_status_is_ok(status)) {
    *out_listener = &listener->base;
  } else if (listener) {
    iree_slim_mutex_deinitialize(&listener->mutex);
    iree_async_proactor_release(listener->proactor);
    iree_net_transport_factory_release(&listener->factory->base);
    iree_async_event_release(listener->event);
    iree_allocator_free(host_allocator, listener);
  } else {
    iree_async_event_release(event);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Cross-process connect
//===----------------------------------------------------------------------===//

typedef struct iree_net_shm_win32_connect_state_t {
  // Consumer connect callback.
  struct {
    // Function receiving the terminal connection result.
    iree_net_transport_connect_callback_t fn;
    // Opaque value passed to |fn|.
    void* user_data;
  } callback;
  // Bootstrap operation valid until its callback begins.
  iree_net_shm_bootstrap_t* bootstrap;
  // Allocator used for this state.
  iree_allocator_t host_allocator;
} iree_net_shm_win32_connect_state_t;

static void iree_net_shm_win32_connect_bootstrap_complete(
    void* user_data, iree_status_t status,
    iree_net_shm_bootstrap_completion_flags_t flags,
    iree_net_connection_t* connection) {
  iree_net_shm_win32_connect_state_t* state =
      (iree_net_shm_win32_connect_state_t*)user_data;
  state->bootstrap = NULL;
  if (iree_any_bit_set(flags,
                       IREE_NET_SHM_BOOTSTRAP_COMPLETION_FLAG_CANCELLED) &&
      iree_status_is_ok(status)) {
    status = iree_make_status(IREE_STATUS_CANCELLED,
                              "SHM connection bootstrap cancelled");
  }
  state->callback.fn(state->callback.user_data, status, connection);
  iree_allocator_free(state->host_allocator, state);
}

iree_status_t iree_net_shm_factory_connect_win32(
    iree_net_shm_factory_t* factory, iree_string_view_t address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_transport_connect_callback_t callback, void* user_data) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)recv_pool;

  iree_string_view_t name = iree_net_shm_win32_strip_pipe_prefix(address);
  WCHAR pipe_path[IREE_NET_SHM_MAX_PIPE_PATH_LENGTH];
  int pipe_path_length = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_net_shm_win32_build_pipe_path(name, pipe_path, &pipe_path_length));

  HANDLE pipe = CreateFileW(pipe_path, GENERIC_READ | GENERIC_WRITE, 0, NULL,
                            OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
  if (pipe == INVALID_HANDLE_VALUE) {
    DWORD error = GetLastError();
    IREE_TRACE_ZONE_END(z0);
    if (error == ERROR_PIPE_BUSY) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "all pipe instances busy for '%.*s'; retry later",
                              (int)name.size, name.data);
    }
    if (error == ERROR_FILE_NOT_FOUND) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no server listening on pipe '%.*s'",
                              (int)name.size, name.data);
    }
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "CreateFileW failed for pipe '%.*s'",
                            (int)name.size, name.data);
  }

  iree_net_shm_win32_connect_state_t* state = NULL;
  iree_status_t status = iree_allocator_malloc(factory->host_allocator,
                                               sizeof(*state), (void**)&state);
  if (iree_status_is_ok(status)) {
    memset(state, 0, sizeof(*state));
    state->callback.fn = callback;
    state->callback.user_data = user_data;
    state->host_allocator = factory->host_allocator;

    iree_async_primitive_t channel =
        iree_async_primitive_from_win32_handle((uintptr_t)pipe);
    status = iree_net_shm_bootstrap_prepare(
        factory, IREE_NET_SHM_BOOTSTRAP_ROLE_CLIENT, &channel, proactor,
        (iree_net_shm_bootstrap_callback_t){
            .fn = iree_net_shm_win32_connect_bootstrap_complete,
            .user_data = state,
        },
        factory->host_allocator, &state->bootstrap);
    if (iree_status_is_ok(status)) {
      pipe = INVALID_HANDLE_VALUE;
      iree_net_shm_bootstrap_launch(state->bootstrap);
    }
  }

  if (!iree_status_is_ok(status)) {
    if (pipe != INVALID_HANDLE_VALUE) CloseHandle(pipe);
    iree_allocator_free(factory->host_allocator, state);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

#endif  // IREE_PLATFORM_WINDOWS
