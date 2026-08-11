// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cross-process SHM carrier tests.
//
// Uses the coordinated test harness to spawn actual separate processes that
// establish SHM carrier pairs via handshake over a local channel. Validates
// the full cross-process path: handle exchange, SHM mapping in separate
// address spaces, shared wake notifications, MPSC ring operation, and direct
// read/write across processes.
//
// The in-process tests (carrier_test.cc, handshake_test.cc, CTS) exercise
// the carrier API thoroughly but use socketpair() or create_pair() — both
// within a single process. These tests verify that the handshake protocol
// actually works when the fd/handle passing, SHM mapping, and notification
// primitives cross the process boundary.
//
// Platform channels:
//   POSIX:   Unix domain socket in the temp directory.
//   Windows: Named pipe (\\.\pipe\<name>) derived from the temp directory.

#include "iree/base/api.h"  // Must precede platform checks for IREE_PLATFORM_*.

#if defined(IREE_PLATFORM_WINDOWS)
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#include <atomic>
#include <cerrno>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/net/carrier/shm/carrier.h"
#include "iree/net/carrier/shm/factory.h"
#include "iree/net/carrier/shm/handshake.h"
#include "iree/net/carrier/shm/shared_wake.h"
#include "iree/net/connection.h"
#include "iree/net/message_endpoint.h"
#include "iree/net/transport_factory.h"
#include "iree/testing/coordinated_test.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_status_t PollProactorOnce(
    iree_async_proactor_t* proactor,
    iree_host_size_t* out_completed_count = nullptr) {
  iree_status_t status = iree_async_proactor_poll(
      proactor, iree_infinite_timeout(), out_completed_count);
  if (iree_status_is_deadline_exceeded(status)) {
    iree_status_free(status);
    return iree_ok_status();
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Error reporting for child processes
//===----------------------------------------------------------------------===//

// Role functions run in child processes without gtest. This macro prints the
// failing expression and status, then returns 1 (failure exit code).
#define XPROC_CHECK_OK(expr)                                   \
  do {                                                         \
    iree_status_t xproc_status__ = (expr);                     \
    if (!iree_status_is_ok(xproc_status__)) {                  \
      fprintf(stderr, "XPROC_CHECK_OK failed: %s\n  ", #expr); \
      iree_status_fprint(stderr, xproc_status__);              \
      fprintf(stderr, "\n");                                   \
      iree_status_free(xproc_status__);                        \
      return 1;                                                \
    }                                                          \
  } while (0)

#define XPROC_CHECK(cond, ...)                           \
  do {                                                   \
    if (!(cond)) {                                       \
      fprintf(stderr, "XPROC_CHECK failed: %s ", #cond); \
      fprintf(stderr, __VA_ARGS__);                      \
      fprintf(stderr, "\n");                             \
      return 1;                                          \
    }                                                    \
  } while (0)

//===----------------------------------------------------------------------===//
// RAII context for cross-process carrier setup
//===----------------------------------------------------------------------===//

// Manages proactor, shared_wake, carrier, and server handle lifetime.
// Used by role functions for automatic cleanup on all exit paths.
struct XProcContext {
  iree_async_proactor_t* proactor = nullptr;
  iree_net_shm_shared_wake_t* shared_wake = nullptr;
  iree_net_carrier_t* carrier = nullptr;

  // Platform-specific server handle. The server binds/creates the channel;
  // the accepted/connected handle transfers to the xproc context on successful
  // handshake, so this may be invalidated before the destructor runs.
#if defined(IREE_PLATFORM_WINDOWS)
  HANDLE pipe_handle = INVALID_HANDLE_VALUE;
#else
  int listen_fd = -1;
#endif

  // Recv capture: accumulates received data for verification.
  std::vector<uint8_t> recv_buffer;
  std::atomic<iree_host_size_t> recv_total_bytes{0};

  // Direct-write signal capture.
  std::atomic<int> signal_count{0};

  // Last direct-write signal immediate value.
  std::atomic<uint32_t> signal_immediate{0};

  // Completion tracking: counts send completions.
  std::atomic<int> completion_count{0};
  std::atomic<iree_host_size_t> completion_bytes{0};

  ~XProcContext() {
    if (carrier) {
      // Set a null recv handler to avoid callbacks during teardown.
      iree_net_carrier_set_recv_handler(carrier, NullRecvHandler());
      iree_net_carrier_set_signal_handler(carrier, {nullptr, nullptr});
      DeactivateAndDrain();
      iree_net_carrier_release(carrier);
    }
    iree_net_shm_shared_wake_release(shared_wake);
    iree_async_proactor_release(proactor);
#if defined(IREE_PLATFORM_WINDOWS)
    if (pipe_handle != INVALID_HANDLE_VALUE) CloseHandle(pipe_handle);
#else
    if (listen_fd >= 0) close(listen_fd);
#endif
  }

  // Static recv handler that captures data into this context.
  static iree_status_t RecvHandler(void* user_data, iree_async_span_t data,
                                   iree_async_buffer_lease_t* lease) {
    auto* context = static_cast<XProcContext*>(user_data);
    uint8_t* ptr = iree_async_span_ptr(data);
    context->recv_buffer.insert(context->recv_buffer.end(), ptr,
                                ptr + data.length);
    context->recv_total_bytes.fetch_add(data.length, std::memory_order_relaxed);
    iree_async_buffer_lease_release(lease);
    return iree_ok_status();
  }

  iree_net_carrier_recv_handler_t AsRecvHandler() {
    return {RecvHandler, this};
  }

  static iree_status_t SignalHandler(void* user_data, uint32_t immediate) {
    auto* context = static_cast<XProcContext*>(user_data);
    context->signal_immediate.store(immediate, std::memory_order_relaxed);
    context->signal_count.fetch_add(1, std::memory_order_relaxed);
    return iree_ok_status();
  }

  iree_net_carrier_signal_handler_t AsSignalHandler() {
    return {SignalHandler, this};
  }

  // Static completion callback that tracks send completions.
  static void CompletionCallback(void* callback_user_data,
                                 iree_net_carrier_completion_kind_t kind,
                                 uint64_t operation_user_data,
                                 iree_status_t status,
                                 iree_host_size_t bytes_transferred,
                                 iree_async_buffer_lease_t* recv_lease) {
    (void)kind;
    (void)operation_user_data;
    (void)recv_lease;
    auto* context = static_cast<XProcContext*>(callback_user_data);
    context->completion_count.fetch_add(1, std::memory_order_relaxed);
    context->completion_bytes.fetch_add(bytes_transferred,
                                        std::memory_order_relaxed);
    iree_status_free(status);
  }

  iree_net_carrier_callback_t AsCallback() {
    return {CompletionCallback, this};
  }

  // Null recv handler for teardown.
  static iree_status_t NullRecvFn(void* user_data, iree_async_span_t data,
                                  iree_async_buffer_lease_t* lease) {
    iree_async_buffer_lease_release(lease);
    return iree_ok_status();
  }

  static iree_net_carrier_recv_handler_t NullRecvHandler() {
    return {NullRecvFn, nullptr};
  }

  // Polls the proactor until |condition| returns true.
  bool PollUntil(std::function<bool()> condition) {
    while (!condition()) {
      iree_host_size_t completed = 0;
      iree_status_t status = PollProactorOnce(proactor, &completed);
      if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
      }
    }
    return true;
  }

  // Deactivates the carrier and drains remaining operations.
  void DeactivateAndDrain() {
    if (!carrier) return;
    iree_net_carrier_state_t state = iree_net_carrier_state(carrier);
    if (state == IREE_NET_CARRIER_STATE_DEACTIVATED) return;
    if (state != IREE_NET_CARRIER_STATE_CREATED &&
        state != IREE_NET_CARRIER_STATE_ACTIVE) {
      iree_status_abort(iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "an existing carrier deactivation must be joined through its "
          "original callback"));
    }
    std::atomic<bool> deactivated{false};
    iree_status_t status = iree_net_carrier_deactivate(
        carrier,
        [](void* user_data) {
          static_cast<std::atomic<bool>*>(user_data)->store(
              true, std::memory_order_release);
        },
        &deactivated);
    if (!iree_status_is_ok(status)) iree_status_abort(status);
    while (!deactivated.load(std::memory_order_acquire)) {
      iree_host_size_t completed = 0;
      status = PollProactorOnce(proactor, &completed);
      if (!iree_status_is_ok(status)) iree_status_abort(status);
    }
  }
};

struct EndpointReadyState {
  bool fired = false;
  iree_status_code_t status_code = IREE_STATUS_OK;
  iree_net_message_endpoint_t endpoint = {nullptr, nullptr};

  static void Callback(void* user_data, iree_status_t status,
                       iree_net_message_endpoint_t endpoint) {
    auto* state = static_cast<EndpointReadyState*>(user_data);
    state->fired = true;
    state->status_code = iree_status_code(status);
    state->endpoint = endpoint;
    iree_status_free(status);
  }
};

struct FactoryXProcContext {
  iree_async_proactor_t* proactor = nullptr;
  iree_async_slab_t* slab = nullptr;
  iree_async_region_t* region = nullptr;
  iree_async_buffer_pool_t* recv_pool = nullptr;
  iree_net_transport_factory_t* factory = nullptr;
  iree_net_listener_t* listener = nullptr;
  iree_net_connection_t* connection = nullptr;
  bool connection_ready = false;
  iree_status_code_t connection_status_code = IREE_STATUS_OK;

  ~FactoryXProcContext() {
    if (listener) {
      bool stopped = false;
      iree_status_t status = iree_net_listener_stop(
          listener,
          {[](void* user_data) { *static_cast<bool*>(user_data) = true; },
           &stopped});
      if (iree_status_is_ok(status)) {
        PollUntil([&] { return stopped; });
      } else {
        iree_status_abort(status);
      }
      iree_net_listener_free(listener);
    }
    if (connection) {
      bool deactivated = false;
      iree_net_connection_deactivate(
          connection,
          {[](void* user_data) { *static_cast<bool*>(user_data) = true; },
           &deactivated});
      PollUntil([&] { return deactivated; });
    }
    iree_net_connection_release(connection);
    iree_net_transport_factory_release(factory);
    iree_async_buffer_pool_release(recv_pool);
    iree_async_region_release(region);
    iree_async_slab_release(slab);
    iree_async_proactor_release(proactor);
  }

  iree_status_t Initialize(uint16_t endpoint_count) {
    IREE_RETURN_IF_ERROR(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor));

    iree_async_slab_options_t slab_options = {0};
    slab_options.buffer_size = 4096;
    slab_options.buffer_count = 16;
    IREE_RETURN_IF_ERROR(
        iree_async_slab_create(slab_options, iree_allocator_system(), &slab));
    IREE_RETURN_IF_ERROR(iree_async_proactor_register_slab(
        proactor, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region));
    IREE_RETURN_IF_ERROR(iree_async_buffer_pool_create(
        region, iree_allocator_system(), &recv_pool));

    iree_net_shm_carrier_options_t options =
        iree_net_shm_carrier_options_default();
    options.max_endpoint_count = endpoint_count;
    return iree_net_shm_factory_create(options, iree_allocator_system(),
                                       &factory);
  }

  bool PollUntil(std::function<bool()> condition) {
    while (!condition()) {
      iree_host_size_t completed = 0;
      iree_status_t status = PollProactorOnce(proactor, &completed);
      if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
      }
    }
    return true;
  }

  static void AcceptCallback(void* user_data, iree_status_t status,
                             iree_net_connection_t* connection) {
    auto* context = static_cast<FactoryXProcContext*>(user_data);
    context->connection_ready = true;
    context->connection_status_code = iree_status_code(status);
    context->connection = connection;
    iree_status_free(status);
  }

  static void ConnectCallback(void* user_data, iree_status_t status,
                              iree_net_connection_t* connection) {
    AcceptCallback(user_data, status, connection);
  }

  static iree_status_t EndpointMessage(void* user_data,
                                       iree_const_byte_span_t message,
                                       iree_async_buffer_lease_t* lease) {
    (void)user_data;
    (void)message;
    (void)lease;
    return iree_ok_status();
  }

  static void EndpointError(void* user_data, iree_status_t status) {
    auto* status_code = static_cast<iree_status_code_t*>(user_data);
    *status_code = iree_status_code(status);
    iree_status_free(status);
  }

  iree_status_t OpenActivateAndDeactivateEndpoints(uint16_t endpoint_count) {
    std::vector<EndpointReadyState> ready(endpoint_count);
    for (uint16_t i = 0; i < endpoint_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_net_connection_open_endpoint(
          connection, {EndpointReadyState::Callback, &ready[i]}));
    }
    if (!PollUntil([&] {
          for (const auto& state : ready) {
            if (!state.fired) return false;
          }
          return true;
        })) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "proactor failed before endpoint ready callbacks fired");
    }

    std::vector<iree_status_code_t> endpoint_errors(endpoint_count,
                                                    IREE_STATUS_OK);
    for (uint16_t i = 0; i < endpoint_count; ++i) {
      if (ready[i].status_code != IREE_STATUS_OK) {
        return iree_make_status(ready[i].status_code,
                                "endpoint %" PRIu16 " open failed", i);
      }
      if (!ready[i].endpoint.self) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "endpoint %" PRIu16 " opened as NULL", i);
      }
      iree_net_message_endpoint_set_callbacks(
          ready[i].endpoint,
          {EndpointMessage, EndpointError, &endpoint_errors[i]});
      IREE_RETURN_IF_ERROR(
          iree_net_message_endpoint_activate(ready[i].endpoint));
    }

    std::vector<uint8_t> deactivated(endpoint_count, 0);
    for (uint16_t i = 0; i < endpoint_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_net_message_endpoint_deactivate(
          ready[i].endpoint,
          [](void* user_data) { *static_cast<uint8_t*>(user_data) = 1; },
          &deactivated[i]));
    }
    if (!PollUntil([&] {
          for (uint8_t value : deactivated) {
            if (!value) return false;
          }
          return true;
        })) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "proactor failed before endpoint deactivate callbacks fired");
    }
    for (uint16_t i = 0; i < endpoint_count; ++i) {
      if (endpoint_errors[i] != IREE_STATUS_OK) {
        return iree_make_status(endpoint_errors[i],
                                "endpoint %" PRIu16 " transport error", i);
      }
    }
    return iree_ok_status();
  }
};

//===----------------------------------------------------------------------===//
// Channel and handshake helpers
//===----------------------------------------------------------------------===//

// Creates the proactor and shared_wake (shared mode for cross-process).
static iree_status_t SetupProactor(XProcContext* context) {
  IREE_RETURN_IF_ERROR(iree_async_proactor_create_platform(
      iree_async_proactor_options_default(), iree_allocator_system(),
      &context->proactor));
  return iree_net_shm_shared_wake_create_shared(
      context->proactor, iree_allocator_system(), &context->shared_wake);
}

// Builds a platform-appropriate channel address from the temp directory.
//
// POSIX:   Unix domain socket path in the temp directory.
// Windows: Named pipe path (\\.\pipe\<basename>) derived from the temp
//          directory name, which is unique per test run.
static void MakeAddress(const char* temp_directory, char* out_address,
                        size_t capacity) {
#if defined(IREE_PLATFORM_WINDOWS)
  const char* basename = temp_directory;
  for (const char* p = temp_directory; *p; ++p) {
    if (*p == '/' || *p == '\\') basename = p + 1;
  }
  snprintf(out_address, capacity, "\\\\.\\pipe\\%s", basename);
#else
  snprintf(out_address, capacity, "%s/c.sock", temp_directory);
#endif
}

#if defined(IREE_PLATFORM_WINDOWS)

// Converts a narrow (UTF-8) string to a wide string for Windows APIs.
static iree_status_t NarrowToWide(const char* narrow, WCHAR* wide,
                                  int wide_capacity) {
  int length = MultiByteToWideChar(CP_UTF8, 0, narrow, -1, wide, wide_capacity);
  if (length <= 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "MultiByteToWideChar failed: %lu",
                            (unsigned long)GetLastError());
  }
  return iree_ok_status();
}

// Server: creates a named pipe and stores the handle in the context.
// The pipe is created with FILE_FLAG_OVERLAPPED because the handshake uses
// overlapped ReadFile/WriteFile with manual-reset events.
static iree_status_t ServerBind(const char* address, XProcContext* context) {
  WCHAR wide_path[MAX_PATH + 1];
  IREE_RETURN_IF_ERROR(NarrowToWide(address, wide_path, MAX_PATH + 1));

  HANDLE pipe =
      CreateNamedPipeW(wide_path, PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
                       PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                       1,     // Single instance (one client per test scenario).
                       4096,  // Output buffer size.
                       4096,  // Input buffer size.
                       0,  // Default timeout (unused by this byte-mode pipe).
                       NULL);  // Default security.
  if (pipe == INVALID_HANDLE_VALUE) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "CreateNamedPipeW failed: %lu",
                            (unsigned long)GetLastError());
  }
  context->pipe_handle = pipe;
  return iree_ok_status();
}

// Server: waits for a client to connect to the pipe, then runs the server-side
// handshake. Creates a carrier from the handshake result.
//
// Uses overlapped ConnectNamedPipe with a blocking wait via event. The pipe
// handle transfers to the xproc context on a successful handshake.
static iree_status_t ServerAcceptAndHandshake(
    XProcContext* context, iree_net_carrier_callback_t callback) {
  // Wait for client connection via overlapped ConnectNamedPipe.
  HANDLE event = CreateEventW(NULL, /*bManualReset=*/TRUE,
                              /*bInitialState=*/FALSE, NULL);
  if (!event) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "CreateEvent for ConnectNamedPipe failed: %lu",
                            (unsigned long)GetLastError());
  }

  OVERLAPPED overlapped;
  memset(&overlapped, 0, sizeof(overlapped));
  overlapped.hEvent = event;

  BOOL connected = ConnectNamedPipe(context->pipe_handle, &overlapped);
  if (!connected) {
    DWORD error = GetLastError();
    if (error == ERROR_IO_PENDING) {
      // Normal case: waiting for client. Block until it arrives.
      DWORD wait_result = WaitForSingleObject(event, INFINITE);
      if (wait_result != WAIT_OBJECT_0) {
        CloseHandle(event);
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "WaitForSingleObject failed: %lu",
                                (unsigned long)GetLastError());
      }
      DWORD bytes_transferred = 0;
      if (!GetOverlappedResult(context->pipe_handle, &overlapped,
                               &bytes_transferred, FALSE)) {
        CloseHandle(event);
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "ConnectNamedPipe overlapped result failed: %lu",
            (unsigned long)GetLastError());
      }
    } else if (error == ERROR_PIPE_CONNECTED) {
      // Client connected between CreateNamedPipeW and ConnectNamedPipe.
    } else {
      CloseHandle(event);
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "ConnectNamedPipe failed: %lu",
                              (unsigned long)error);
    }
  }
  CloseHandle(event);

  // Pass the connected pipe to the handshake, which takes ownership and
  // transfers it to the xproc context on success and closes it on failure.
  iree_async_primitive_t channel =
      iree_async_primitive_from_win32_handle((uintptr_t)context->pipe_handle);
  context->pipe_handle = INVALID_HANDLE_VALUE;

  iree_net_shm_handshake_result_t handshake_result;
  memset(&handshake_result, 0, sizeof(handshake_result));
  IREE_RETURN_IF_ERROR(iree_net_shm_handshake_server(
      channel, context->shared_wake, iree_net_shm_carrier_options_default(),
      context->proactor, iree_allocator_system(), &handshake_result));

  iree_status_t status =
      iree_net_shm_carrier_create(&handshake_result.carrier_params, callback,
                                  iree_allocator_system(), &context->carrier);
  if (!iree_status_is_ok(status)) {
    iree_net_shm_xproc_context_release(handshake_result.context);
  }
  return status;
}

// Client: opens the named pipe and runs the client-side handshake.
// Creates a carrier from the handshake result.
static iree_status_t ClientConnectAndHandshake(
    const char* address, XProcContext* context,
    iree_net_carrier_callback_t callback) {
  WCHAR wide_path[MAX_PATH + 1];
  IREE_RETURN_IF_ERROR(NarrowToWide(address, wide_path, MAX_PATH + 1));

  // CreateFileW on a named pipe connects synchronously.
  HANDLE pipe = CreateFileW(wide_path, GENERIC_READ | GENERIC_WRITE,
                            0,     // No sharing.
                            NULL,  // Default security.
                            OPEN_EXISTING,
                            FILE_FLAG_OVERLAPPED,  // Required by handshake.
                            NULL);                 // No template.
  if (pipe == INVALID_HANDLE_VALUE) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "CreateFileW (pipe connect) failed: %lu",
                            (unsigned long)GetLastError());
  }

  // The handshake takes ownership of the channel.
  iree_async_primitive_t channel =
      iree_async_primitive_from_win32_handle((uintptr_t)pipe);

  iree_net_shm_handshake_result_t handshake_result;
  memset(&handshake_result, 0, sizeof(handshake_result));
  IREE_RETURN_IF_ERROR(iree_net_shm_handshake_client(
      channel, context->shared_wake, context->proactor, iree_allocator_system(),
      &handshake_result));

  iree_status_t status =
      iree_net_shm_carrier_create(&handshake_result.carrier_params, callback,
                                  iree_allocator_system(), &context->carrier);
  if (!iree_status_is_ok(status)) {
    iree_net_shm_xproc_context_release(handshake_result.context);
  }
  return status;
}

#else  // POSIX

// Server: creates a Unix domain socket, binds, and listens. The socket path
// is stored in the temp directory and serves as the rendezvous address for
// the client.
static iree_status_t ServerBind(const char* address, XProcContext* context) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    return iree_make_status(IREE_STATUS_INTERNAL, "socket(AF_UNIX) failed: %s",
                            strerror(errno));
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  size_t path_length = strlen(address);
  if (path_length >= sizeof(addr.sun_path)) {
    close(fd);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "socket path too long (%zu >= %zu)", path_length,
                            sizeof(addr.sun_path));
  }
  memcpy(addr.sun_path, address, path_length + 1);

  if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
    close(fd);
    return iree_make_status(IREE_STATUS_INTERNAL, "bind(%s) failed: %s",
                            address, strerror(errno));
  }
  if (listen(fd, 1) != 0) {
    close(fd);
    return iree_make_status(IREE_STATUS_INTERNAL, "listen(%s) failed: %s",
                            address, strerror(errno));
  }

  context->listen_fd = fd;
  return iree_ok_status();
}

// Server: accepts one connection and runs the server-side handshake.
// Creates a carrier from the handshake result.
static iree_status_t ServerAcceptAndHandshake(
    XProcContext* context, iree_net_carrier_callback_t callback) {
  int client_fd = accept(context->listen_fd, nullptr, nullptr);
  if (client_fd < 0) {
    return iree_make_status(IREE_STATUS_INTERNAL, "accept() failed: %s",
                            strerror(errno));
  }

  // The handshake takes ownership of the fd.
  iree_async_primitive_t channel = iree_async_primitive_from_fd(client_fd);

  iree_net_shm_handshake_result_t handshake_result;
  memset(&handshake_result, 0, sizeof(handshake_result));
  IREE_RETURN_IF_ERROR(iree_net_shm_handshake_server(
      channel, context->shared_wake, iree_net_shm_carrier_options_default(),
      context->proactor, iree_allocator_system(), &handshake_result));

  iree_status_t status =
      iree_net_shm_carrier_create(&handshake_result.carrier_params, callback,
                                  iree_allocator_system(), &context->carrier);
  if (!iree_status_is_ok(status)) {
    iree_net_shm_xproc_context_release(handshake_result.context);
  }
  return status;
}

// Client: connects to the server's Unix domain socket and runs the
// client-side handshake. Creates a carrier from the handshake result.
static iree_status_t ClientConnectAndHandshake(
    const char* address, XProcContext* context,
    iree_net_carrier_callback_t callback) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    return iree_make_status(IREE_STATUS_INTERNAL, "socket(AF_UNIX) failed: %s",
                            strerror(errno));
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  size_t path_length = strlen(address);
  if (path_length >= sizeof(addr.sun_path)) {
    close(fd);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "socket path too long");
  }
  memcpy(addr.sun_path, address, path_length + 1);

  if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
    close(fd);
    return iree_make_status(IREE_STATUS_INTERNAL, "connect(%s) failed: %s",
                            address, strerror(errno));
  }

  // The handshake takes ownership of the fd.
  iree_async_primitive_t channel = iree_async_primitive_from_fd(fd);

  iree_net_shm_handshake_result_t handshake_result;
  memset(&handshake_result, 0, sizeof(handshake_result));
  IREE_RETURN_IF_ERROR(iree_net_shm_handshake_client(
      channel, context->shared_wake, context->proactor, iree_allocator_system(),
      &handshake_result));

  iree_status_t status =
      iree_net_shm_carrier_create(&handshake_result.carrier_params, callback,
                                  iree_allocator_system(), &context->carrier);
  if (!iree_status_is_ok(status)) {
    iree_net_shm_xproc_context_release(handshake_result.context);
  }
  return status;
}

#endif  // IREE_PLATFORM_WINDOWS

// Sends a message via the carrier. Convenience wrapper around the scatter-
// gather send API.
static iree_status_t SendMessage(iree_net_carrier_t* carrier, const void* data,
                                 iree_host_size_t length) {
  iree_async_span_t span =
      iree_async_span_from_ptr(const_cast<void*>(data), length);
  iree_net_send_params_t params = {};
  params.data.values = &span;
  params.data.count = 1;
  params.flags = IREE_NET_SEND_FLAG_NONE;
  params.user_data = 0;
  return iree_net_carrier_send(carrier, &params);
}

//===----------------------------------------------------------------------===//
// Test 1: Handshake and carrier creation
//===----------------------------------------------------------------------===//

static int handshake_server_role(int argc, char** argv,
                                 const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(ServerBind(address, &context));

  iree_coordinated_test_signal_ready(temp_directory);

  iree_net_carrier_callback_t no_callback = {nullptr, nullptr};
  XPROC_CHECK_OK(ServerAcceptAndHandshake(&context, no_callback));
  XPROC_CHECK(context.carrier != nullptr, "carrier is null");

  // Activate and immediately deactivate to verify the carrier is functional.
  iree_net_carrier_set_recv_handler(context.carrier,
                                    XProcContext::NullRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  return 0;
}

static int handshake_client_role(int argc, char** argv,
                                 const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));

  iree_net_carrier_callback_t no_callback = {nullptr, nullptr};
  XPROC_CHECK_OK(ClientConnectAndHandshake(address, &context, no_callback));
  XPROC_CHECK(context.carrier != nullptr, "carrier is null");

  iree_net_carrier_set_recv_handler(context.carrier,
                                    XProcContext::NullRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  return 0;
}

//===----------------------------------------------------------------------===//
// Test 2: Send/recv round trip
//===----------------------------------------------------------------------===//

static const char kClientMessage[] = "hello from client";
static const char kServerMessage[] = "hello from server";

static int sendrecv_server_role(int argc, char** argv,
                                const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(ServerBind(address, &context));

  iree_coordinated_test_signal_ready(temp_directory);

  XPROC_CHECK_OK(ServerAcceptAndHandshake(&context, context.AsCallback()));

  // Activate and wait for the client's message.
  iree_net_carrier_set_recv_handler(context.carrier, context.AsRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  XPROC_CHECK(context.PollUntil([&] {
    return context.recv_total_bytes.load() >= strlen(kClientMessage);
  }),
              "client message did not arrive");

  XPROC_CHECK(context.recv_buffer.size() == strlen(kClientMessage),
              "expected %zu bytes, got %zu", strlen(kClientMessage),
              context.recv_buffer.size());
  XPROC_CHECK(memcmp(context.recv_buffer.data(), kClientMessage,
                     strlen(kClientMessage)) == 0,
              "client message mismatch");

  // Send response.
  XPROC_CHECK_OK(
      SendMessage(context.carrier, kServerMessage, strlen(kServerMessage)));

  return 0;
}

static int sendrecv_client_role(int argc, char** argv,
                                const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(
      ClientConnectAndHandshake(address, &context, context.AsCallback()));

  // Activate and send our message.
  iree_net_carrier_set_recv_handler(context.carrier, context.AsRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  XPROC_CHECK_OK(
      SendMessage(context.carrier, kClientMessage, strlen(kClientMessage)));

  // Wait for the server's response.
  XPROC_CHECK(context.PollUntil([&] {
    return context.recv_total_bytes.load() >= strlen(kServerMessage);
  }),
              "server response did not arrive");

  XPROC_CHECK(context.recv_buffer.size() == strlen(kServerMessage),
              "expected %zu bytes, got %zu", strlen(kServerMessage),
              context.recv_buffer.size());
  XPROC_CHECK(memcmp(context.recv_buffer.data(), kServerMessage,
                     strlen(kServerMessage)) == 0,
              "server response mismatch");

  return 0;
}

//===----------------------------------------------------------------------===//
// Test 3: Direct write with signaling
//===----------------------------------------------------------------------===//

// Known pattern written by the client at a fixed SHM offset.
static const iree_host_size_t kDirectWriteOffset = 0x1000;
static const iree_host_size_t kDirectWriteLength = 64;

static void FillPattern(uint8_t* buffer, iree_host_size_t length,
                        uint8_t seed) {
  for (iree_host_size_t i = 0; i < length; ++i) {
    buffer[i] = (uint8_t)(seed + i);
  }
}

static int dwrite_server_role(int argc, char** argv,
                              const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(ServerBind(address, &context));

  iree_coordinated_test_signal_ready(temp_directory);

  XPROC_CHECK_OK(ServerAcceptAndHandshake(&context, context.AsCallback()));

  // Activate and wait for the signaling direct_write immediate. The direct
  // write bytes land in SHM region memory before the signal is delivered.
  iree_net_carrier_set_recv_handler(context.carrier,
                                    XProcContext::NullRecvHandler());
  iree_net_carrier_set_signal_handler(context.carrier,
                                      context.AsSignalHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  XPROC_CHECK(
      context.PollUntil([&] { return context.signal_count.load() >= 1; }),
      "direct_write signal did not arrive");
  XPROC_CHECK(context.signal_immediate.load() == 0xABCD1234u,
              "direct_write signal immediate mismatch");

  // Verify the received data matches the expected pattern.
  uint8_t expected[kDirectWriteLength];
  FillPattern(expected, kDirectWriteLength, 0xAB);

  iree_net_shm_region_info_t region_info = {};
  XPROC_CHECK_OK(
      iree_net_shm_carrier_query_region(context.carrier, 0, &region_info));
  XPROC_CHECK(memcmp((uint8_t*)region_info.base_ptr + kDirectWriteOffset,
                     expected, kDirectWriteLength) == 0,
              "direct_write data mismatch");

  return 0;
}

static int dwrite_client_role(int argc, char** argv,
                              const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(
      ClientConnectAndHandshake(address, &context, context.AsCallback()));

  iree_net_carrier_set_recv_handler(context.carrier,
                                    XProcContext::NullRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  // Prepare the source data and write it at kDirectWriteOffset in region 0
  // with SIGNAL_RECEIVER flag so the server's signal handler fires.
  uint8_t source_data[kDirectWriteLength];
  FillPattern(source_data, kDirectWriteLength, 0xAB);

  iree_net_direct_write_params_t params = {};
  params.local = iree_async_span_from_ptr(source_data, sizeof(source_data));
  params.remote = iree_net_remote_handle_t{{0, kDirectWriteOffset}};
  params.flags = IREE_NET_DIRECT_WRITE_FLAG_SIGNAL_RECEIVER;
  params.immediate = 0xABCD1234u;
  params.user_data = 0;
  XPROC_CHECK_OK(iree_net_carrier_direct_write(context.carrier, &params));

  // Wait for the send completion.
  XPROC_CHECK(
      context.PollUntil([&] { return context.completion_count.load() >= 1; }),
      "direct_write did not complete");

  return 0;
}

//===----------------------------------------------------------------------===//
// Test 4: File transfer sideband
//===----------------------------------------------------------------------===//

#if IREE_FILE_IO_ENABLE

static const char kFileTransferContents[] = "shm sideband file contents";
static const char kFileTransferAck[] = "file-transfer-ok";

static iree_net_carrier_capabilities_t FileTransferCapability() {
#if defined(IREE_PLATFORM_WINDOWS)
  return IREE_NET_CARRIER_CAPABILITY_WIN32_HANDLE_TRANSFER;
#else
  return IREE_NET_CARRIER_CAPABILITY_POSIX_FD_TRANSFER;
#endif  // IREE_PLATFORM_WINDOWS
}

static iree_net_file_handle_transfer_type_t FileTransferType() {
#if defined(IREE_PLATFORM_WINDOWS)
  return IREE_NET_FILE_HANDLE_TRANSFER_TYPE_WIN32_HANDLE;
#else
  return IREE_NET_FILE_HANDLE_TRANSFER_TYPE_POSIX_FD;
#endif  // IREE_PLATFORM_WINDOWS
}

static const char* FileTransferTypeName() {
#if defined(IREE_PLATFORM_WINDOWS)
  return "Win32 HANDLE";
#else
  return "POSIX fd";
#endif  // IREE_PLATFORM_WINDOWS
}

static iree_status_t WriteTestFile(const char* path, const void* contents,
                                   iree_host_size_t length) {
  return iree_io_file_contents_write(
      iree_make_cstring_view(path), iree_make_const_byte_span(contents, length),
      iree_allocator_system());
}

#if defined(IREE_PLATFORM_WINDOWS)

static iree_status_t ReadTestFd(int fd, void* contents,
                                iree_host_size_t length) {
  HANDLE file_handle = (HANDLE)_get_osfhandle(fd);
  if (file_handle == INVALID_HANDLE_VALUE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "file descriptor is not backed by a valid Win32 HANDLE");
  }

  HANDLE event = CreateEventW(NULL, /*bManualReset=*/TRUE,
                              /*bInitialState=*/FALSE, NULL);
  if (!event) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "CreateEventW failed for file read");
  }

  uint8_t* cursor = (uint8_t*)contents;
  iree_host_size_t remaining = length;
  uint64_t offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && remaining > 0) {
    OVERLAPPED overlapped;
    memset(&overlapped, 0, sizeof(overlapped));
    overlapped.hEvent = event;
    overlapped.Offset = (DWORD)(offset & 0xFFFFFFFFu);
    overlapped.OffsetHigh = (DWORD)(offset >> 32);
    ResetEvent(event);

    DWORD chunk_length =
        (DWORD)iree_min(remaining, (iree_host_size_t)UINT32_MAX);
    BOOL read_ok =
        ReadFile(file_handle, cursor, chunk_length, NULL, &overlapped);
    if (!read_ok) {
      DWORD error = GetLastError();
      if (error != ERROR_IO_PENDING) {
        status =
            iree_make_status(iree_status_code_from_win32_error(error),
                             "ReadFile failed for transferred file handle");
      }
    }

    DWORD bytes_read = 0;
    if (iree_status_is_ok(status) &&
        !GetOverlappedResult(file_handle, &overlapped, &bytes_read, TRUE)) {
      status =
          iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                           "GetOverlappedResult failed for transferred "
                           "file read");
    }
    if (iree_status_is_ok(status) && bytes_read == 0) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "unexpected EOF while reading transferred file");
    }
    if (iree_status_is_ok(status)) {
      cursor += bytes_read;
      offset += bytes_read;
      remaining -= (iree_host_size_t)bytes_read;
    }
  }

  CloseHandle(event);
  return status;
}

#else

static iree_status_t ReadTestFd(int fd, void* contents,
                                iree_host_size_t length) {
  uint8_t* cursor = (uint8_t*)contents;
  iree_host_size_t remaining = length;
  off_t offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && remaining > 0) {
    ssize_t read_count = pread(fd, cursor, remaining, offset);
    if (read_count < 0) {
      if (errno != EINTR) {
        status = iree_make_status(iree_status_code_from_errno(errno),
                                  "pread failed");
      }
    } else if (read_count == 0) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "unexpected EOF while reading transferred fd");
    } else {
      cursor += read_count;
      offset += read_count;
      remaining -= (iree_host_size_t)read_count;
    }
  }
  return status;
}

#endif  // IREE_PLATFORM_WINDOWS

static int file_transfer_server_role(int argc, char** argv,
                                     const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(ServerBind(address, &context));

  iree_coordinated_test_signal_ready(temp_directory);

  XPROC_CHECK_OK(ServerAcceptAndHandshake(&context, context.AsCallback()));
  XPROC_CHECK(iree_all_bits_set(iree_net_carrier_capabilities(context.carrier),
                                FileTransferCapability()),
              "carrier does not advertise %s transfer", FileTransferTypeName());

  iree_net_carrier_set_recv_handler(context.carrier, context.AsRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  char file_path[256];
  snprintf(file_path, sizeof(file_path), "%s/file-transfer.dat",
           temp_directory);
  XPROC_CHECK_OK(WriteTestFile(file_path, kFileTransferContents,
                               strlen(kFileTransferContents)));

  iree_io_file_handle_t* file_handle = nullptr;
  XPROC_CHECK_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS |
          IREE_IO_FILE_MODE_SHARE_READ | IREE_IO_FILE_MODE_ASYNC,
      iree_make_cstring_view(file_path), iree_allocator_system(),
      &file_handle));

  iree_net_file_handle_transfer_type_t transfer_type =
      IREE_NET_FILE_HANDLE_TRANSFER_TYPE_NONE;
  iree_host_size_t payload_length = 0;
  XPROC_CHECK_OK(iree_net_carrier_query_file_handle_transfer(
      context.carrier, file_handle, &transfer_type, &payload_length));
  XPROC_CHECK(transfer_type == FileTransferType(),
              "unexpected transfer type %u", (uint32_t)transfer_type);

  std::vector<uint8_t> transfer_payload(payload_length);
  XPROC_CHECK_OK(iree_net_carrier_export_file_handle(
      context.carrier, file_handle, transfer_type,
      iree_make_byte_span(transfer_payload.data(), transfer_payload.size())));
  iree_io_file_handle_release(file_handle);

  XPROC_CHECK_OK(SendMessage(context.carrier, transfer_payload.data(),
                             transfer_payload.size()));

  XPROC_CHECK(context.PollUntil([&] {
    return context.recv_total_bytes.load() >= strlen(kFileTransferAck);
  }),
              "file transfer acknowledgement did not arrive");

  return 0;
}

static int file_transfer_client_role(int argc, char** argv,
                                     const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(
      ClientConnectAndHandshake(address, &context, context.AsCallback()));

  iree_net_carrier_set_recv_handler(context.carrier, context.AsRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  XPROC_CHECK(
      context.PollUntil([&] { return context.recv_total_bytes.load() > 0; }),
      "file transfer payload did not arrive");

  iree_io_file_handle_t* file_handle = nullptr;
  XPROC_CHECK_OK(iree_net_carrier_import_file_handle(
      context.carrier, FileTransferType(),
      iree_make_const_byte_span(context.recv_buffer.data(),
                                context.recv_buffer.size()),
      iree_allocator_system(), &file_handle));

  const int fd = iree_io_file_handle_value(file_handle).fd;
  char contents[sizeof(kFileTransferContents)] = {0};
  XPROC_CHECK_OK(ReadTestFd(fd, contents, strlen(kFileTransferContents)));
  iree_io_file_handle_release(file_handle);

  XPROC_CHECK(strcmp(contents, kFileTransferContents) == 0,
              "transferred fd contents mismatch");

  XPROC_CHECK_OK(
      SendMessage(context.carrier, kFileTransferAck, strlen(kFileTransferAck)));

  return 0;
}

#endif  // IREE_FILE_IO_ENABLE

//===----------------------------------------------------------------------===//
// Test 5: Factory multi-endpoint connection
//===----------------------------------------------------------------------===//

static constexpr uint16_t kFactoryEndpointCount = 3;

static std::string MakeFactoryAddressString(const char* temp_directory) {
#if defined(IREE_PLATFORM_WINDOWS)
  const char* basename = temp_directory;
  for (const char* p = temp_directory; *p; ++p) {
    if (*p == '/' || *p == '\\') basename = p + 1;
  }
  return std::string("pipe:") + basename;
#else
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));
  return std::string("unix:") + address;
#endif  // IREE_PLATFORM_WINDOWS
}

static int factory_multi_endpoint_server_role(int argc, char** argv,
                                              const char* temp_directory) {
  FactoryXProcContext context;
  XPROC_CHECK_OK(context.Initialize(kFactoryEndpointCount));

  std::string address = MakeFactoryAddressString(temp_directory);
  XPROC_CHECK_OK(iree_net_transport_factory_create_listener(
      context.factory, iree_make_cstring_view(address.c_str()),
      context.proactor, context.recv_pool, FactoryXProcContext::AcceptCallback,
      &context, iree_allocator_system(), &context.listener));

  iree_coordinated_test_signal_ready(temp_directory);

  XPROC_CHECK(context.PollUntil([&] { return context.connection_ready; }),
              "factory accept did not complete");
  XPROC_CHECK(context.connection_status_code == IREE_STATUS_OK,
              "factory accept failed with status %d",
              (int)context.connection_status_code);
  XPROC_CHECK(context.connection != nullptr,
              "factory accept produced a NULL connection");
  XPROC_CHECK(iree_net_connection_max_endpoint_count(context.connection) ==
                  kFactoryEndpointCount,
              "server endpoint count mismatch");
  XPROC_CHECK_OK(
      context.OpenActivateAndDeactivateEndpoints(kFactoryEndpointCount));
  return 0;
}

static int factory_multi_endpoint_client_role(int argc, char** argv,
                                              const char* temp_directory) {
  FactoryXProcContext context;
  XPROC_CHECK_OK(context.Initialize(kFactoryEndpointCount));

  std::string address = MakeFactoryAddressString(temp_directory);
  XPROC_CHECK_OK(iree_net_transport_factory_connect(
      context.factory, iree_make_cstring_view(address.c_str()),
      context.proactor, context.recv_pool, FactoryXProcContext::ConnectCallback,
      &context));

  XPROC_CHECK(context.PollUntil([&] { return context.connection_ready; }),
              "factory connect did not complete");
  XPROC_CHECK(context.connection_status_code == IREE_STATUS_OK,
              "factory connect failed with status %d",
              (int)context.connection_status_code);
  XPROC_CHECK(context.connection != nullptr,
              "factory connect produced a NULL connection");
  XPROC_CHECK(iree_net_connection_max_endpoint_count(context.connection) ==
                  kFactoryEndpointCount,
              "client endpoint count mismatch");
  XPROC_CHECK_OK(
      context.OpenActivateAndDeactivateEndpoints(kFactoryEndpointCount));
  return 0;
}

//===----------------------------------------------------------------------===//
// Test 6: Listener stop cancels a stalled bootstrap
//===----------------------------------------------------------------------===//

// Opens the factory's platform channel without running the handshake. The
// caller owns the returned primitive.
static iree_status_t OpenRawFactoryChannel(
    const char* temp_directory, iree_async_primitive_t* out_channel) {
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));
#if defined(IREE_PLATFORM_WINDOWS)
  WCHAR wide_path[MAX_PATH + 1];
  IREE_RETURN_IF_ERROR(NarrowToWide(address, wide_path, MAX_PATH + 1));
  HANDLE pipe = CreateFileW(wide_path, GENERIC_READ | GENERIC_WRITE,
                            /*dwShareMode=*/0, /*lpSecurityAttributes=*/NULL,
                            OPEN_EXISTING, FILE_FLAG_OVERLAPPED,
                            /*hTemplateFile=*/NULL);
  if (pipe == INVALID_HANDLE_VALUE) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "CreateFileW failed for raw factory channel");
  }
  *out_channel = iree_async_primitive_from_win32_handle((uintptr_t)pipe);
#else
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "socket(AF_UNIX) failed for raw factory channel");
  }
  struct sockaddr_un socket_address;
  memset(&socket_address, 0, sizeof(socket_address));
  socket_address.sun_family = AF_UNIX;
  iree_host_size_t address_length = strlen(address);
  if (address_length >= sizeof(socket_address.sun_path)) {
    close(fd);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "raw factory socket path is too long");
  }
  memcpy(socket_address.sun_path, address, address_length + 1);
  if (connect(fd, (struct sockaddr*)&socket_address, sizeof(socket_address)) !=
      0) {
    iree_status_t status =
        iree_make_status(iree_status_code_from_errno(errno),
                         "connect failed for raw factory channel");
    close(fd);
    return status;
  }
  *out_channel = iree_async_primitive_from_fd(fd);
#endif  // IREE_PLATFORM_WINDOWS
  return iree_ok_status();
}

static int factory_stalled_bootstrap_server_role(int argc, char** argv,
                                                 const char* temp_directory) {
  FactoryXProcContext context;
  XPROC_CHECK_OK(context.Initialize(/*endpoint_count=*/1));

  std::string address = MakeFactoryAddressString(temp_directory);
  XPROC_CHECK_OK(iree_net_transport_factory_create_listener(
      context.factory, iree_make_cstring_view(address.c_str()),
      context.proactor, context.recv_pool, FactoryXProcContext::AcceptCallback,
      &context, iree_allocator_system(), &context.listener));
  iree_coordinated_test_signal_ready(temp_directory);

  // The raw peer connects but never sends ACCEPT. Once a user operation
  // completes, the listener callback has inserted and launched the pending
  // bootstrap worker. Empty proactor wakes do not establish that state.
  iree_host_size_t completed_count = 0;
  while (completed_count == 0) {
    XPROC_CHECK_OK(PollProactorOnce(context.proactor, &completed_count));
  }
  XPROC_CHECK(!context.connection_ready,
              "stalled bootstrap unexpectedly delivered a connection");

  bool stopped = false;
  XPROC_CHECK_OK(iree_net_listener_stop(
      context.listener,
      {[](void* user_data) { *static_cast<bool*>(user_data) = true; },
       &stopped}));
  XPROC_CHECK(context.PollUntil([&] { return stopped; }),
              "listener stop did not complete");
  iree_net_listener_free(context.listener);
  context.listener = nullptr;

  XPROC_CHECK(!context.connection_ready,
              "cancelled bootstrap invoked the accept callback");
  XPROC_CHECK(context.connection == nullptr,
              "cancelled bootstrap produced a connection");
  return 0;
}

static int factory_stalled_bootstrap_client_role(int argc, char** argv,
                                                 const char* temp_directory) {
  iree_async_primitive_t channel = iree_async_primitive_none();
  XPROC_CHECK_OK(OpenRawFactoryChannel(temp_directory, &channel));

  iree_net_shm_handshake_header_t header;
  iree_net_shm_handshake_handles_t handles =
      iree_net_shm_handshake_handles_empty();
  iree_status_t status = iree_net_shm_handshake_recv(
      channel, /*cancellation=*/NULL, &header, &handles);
  if (iree_status_is_ok(status)) {
    bool valid_offer = header.magic == IREE_NET_SHM_HANDSHAKE_MAGIC &&
                       header.version == IREE_NET_SHM_HANDSHAKE_VERSION &&
                       header.type == IREE_NET_SHM_HANDSHAKE_MESSAGE_OFFER;
    iree_net_shm_handshake_handles_close(&handles);
    XPROC_CHECK(valid_offer, "raw factory peer received a malformed OFFER");

    handles = iree_net_shm_handshake_handles_empty();
    status = iree_net_shm_handshake_recv(channel, /*cancellation=*/NULL,
                                         &header, &handles);
  }
  iree_net_shm_handshake_handles_close(&handles);
  iree_async_primitive_close(&channel);

  iree_status_code_t status_code = iree_status_code(status);
  iree_status_free(status);
  XPROC_CHECK(status_code == IREE_STATUS_UNAVAILABLE,
              "stalled peer expected channel closure, got status %d",
              (int)status_code);
  return 0;
}

//===----------------------------------------------------------------------===//
// Test 7: Direct read across processes
//===----------------------------------------------------------------------===//

// The server writes data at this offset; the client reads it.
static const iree_host_size_t kDirectReadOffset = 0x2000;
static const iree_host_size_t kDirectReadLength = 48;
static const char kDataReadyMarker[] = "ready";
static const char kDataReadyAck[] = "ack";

static int dread_server_role(int argc, char** argv,
                             const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(ServerBind(address, &context));

  iree_coordinated_test_signal_ready(temp_directory);

  XPROC_CHECK_OK(ServerAcceptAndHandshake(&context, context.AsCallback()));

  // Write known data directly into the SHM region at the agreed offset.
  // The server is the creator of the SHM region, so region 0 is its mapping.
  iree_net_shm_region_info_t region_info = {};
  XPROC_CHECK_OK(
      iree_net_shm_carrier_query_region(context.carrier, 0, &region_info));

  XPROC_CHECK(kDirectReadOffset + kDirectReadLength <= region_info.size,
              "offset+length exceeds region size");

  uint8_t write_data[kDirectReadLength];
  FillPattern(write_data, kDirectReadLength, 0xCD);
  memcpy((uint8_t*)region_info.base_ptr + kDirectReadOffset, write_data,
         kDirectReadLength);

  // Activate and tell the client the data is ready.
  iree_net_carrier_set_recv_handler(context.carrier, context.AsRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  XPROC_CHECK_OK(
      SendMessage(context.carrier, kDataReadyMarker, strlen(kDataReadyMarker)));

  // Wait for the client's ack.
  XPROC_CHECK(context.PollUntil([&] {
    return context.recv_total_bytes.load() >= strlen(kDataReadyAck);
  }),
              "client acknowledgement did not arrive");

  return 0;
}

static int dread_client_role(int argc, char** argv,
                             const char* temp_directory) {
  XProcContext context;
  char address[256];
  MakeAddress(temp_directory, address, sizeof(address));

  XPROC_CHECK_OK(SetupProactor(&context));
  XPROC_CHECK_OK(
      ClientConnectAndHandshake(address, &context, context.AsCallback()));

  // Activate and wait for the server's "data ready" marker.
  iree_net_carrier_set_recv_handler(context.carrier, context.AsRecvHandler());
  XPROC_CHECK_OK(iree_net_carrier_activate(context.carrier));

  XPROC_CHECK(context.PollUntil([&] {
    return context.recv_total_bytes.load() >= strlen(kDataReadyMarker);
  }),
              "data-ready marker did not arrive");

  // Direct read from the agreed offset. The client's region 0 is its mapping
  // of the same SHM object the server created.
  uint8_t read_buffer[kDirectReadLength];
  memset(read_buffer, 0, sizeof(read_buffer));

  iree_net_direct_read_params_t params = {};
  params.local = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  params.remote = iree_net_remote_handle_t{{0, kDirectReadOffset}};
  params.user_data = 0;
  XPROC_CHECK_OK(iree_net_carrier_direct_read(context.carrier, &params));

  // Verify the data matches what the server wrote.
  uint8_t expected[kDirectReadLength];
  FillPattern(expected, kDirectReadLength, 0xCD);
  XPROC_CHECK(memcmp(read_buffer, expected, kDirectReadLength) == 0,
              "direct_read data mismatch");

  // Send ack so the server can exit cleanly.
  XPROC_CHECK_OK(
      SendMessage(context.carrier, kDataReadyAck, strlen(kDataReadyAck)));

  return 0;
}

//===----------------------------------------------------------------------===//
// Test configs
//===----------------------------------------------------------------------===//

static const iree_test_role_t kHandshakeRoles[] = {
    {"handshake_server", handshake_server_role, /*signals_ready=*/true},
    {"handshake_client", handshake_client_role, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kHandshakeConfig = {
    /*.roles=*/kHandshakeRoles,
    /*.role_count=*/2,
};

static const iree_test_role_t kSendRecvRoles[] = {
    {"sendrecv_server", sendrecv_server_role, /*signals_ready=*/true},
    {"sendrecv_client", sendrecv_client_role, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kSendRecvConfig = {
    /*.roles=*/kSendRecvRoles,
    /*.role_count=*/2,
};

static const iree_test_role_t kDirectWriteRoles[] = {
    {"dwrite_server", dwrite_server_role, /*signals_ready=*/true},
    {"dwrite_client", dwrite_client_role, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kDirectWriteConfig = {
    /*.roles=*/kDirectWriteRoles,
    /*.role_count=*/2,
};

#if IREE_FILE_IO_ENABLE
static const iree_test_role_t kFileTransferRoles[] = {
    {"file_transfer_server", file_transfer_server_role,
     /*signals_ready=*/true},
    {"file_transfer_client", file_transfer_client_role,
     /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kFileTransferConfig = {
    /*.roles=*/kFileTransferRoles,
    /*.role_count=*/2,
};
#endif  // IREE_FILE_IO_ENABLE

static const iree_test_role_t kFactoryMultiEndpointRoles[] = {
    {"factory_multi_endpoint_server", factory_multi_endpoint_server_role,
     /*signals_ready=*/true},
    {"factory_multi_endpoint_client", factory_multi_endpoint_client_role,
     /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kFactoryMultiEndpointConfig = {
    /*.roles=*/kFactoryMultiEndpointRoles,
    /*.role_count=*/2,
};

static const iree_test_role_t kFactoryStalledBootstrapRoles[] = {
    {"factory_stalled_bootstrap_server", factory_stalled_bootstrap_server_role,
     /*signals_ready=*/true},
    {"factory_stalled_bootstrap_client", factory_stalled_bootstrap_client_role,
     /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kFactoryStalledBootstrapConfig = {
    /*.roles=*/kFactoryStalledBootstrapRoles,
    /*.role_count=*/2,
};

static const iree_test_role_t kDirectReadRoles[] = {
    {"dread_server", dread_server_role, /*signals_ready=*/true},
    {"dread_client", dread_client_role, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kDirectReadConfig = {
    /*.roles=*/kDirectReadRoles,
    /*.role_count=*/2,
};

// Combined config with all roles for child dispatch. The coordinated_test_main
// uses this to find the right entry function when --iree_test_role is set.
static const iree_test_role_t kAllRoles[] = {
    {"handshake_server", handshake_server_role, /*signals_ready=*/true},
    {"handshake_client", handshake_client_role, /*signals_ready=*/false},
    {"sendrecv_server", sendrecv_server_role, /*signals_ready=*/true},
    {"sendrecv_client", sendrecv_client_role, /*signals_ready=*/false},
    {"dwrite_server", dwrite_server_role, /*signals_ready=*/true},
    {"dwrite_client", dwrite_client_role, /*signals_ready=*/false},
#if IREE_FILE_IO_ENABLE
    {"file_transfer_server", file_transfer_server_role,
     /*signals_ready=*/true},
    {"file_transfer_client", file_transfer_client_role,
     /*signals_ready=*/false},
#endif  // IREE_FILE_IO_ENABLE
    {"factory_multi_endpoint_server", factory_multi_endpoint_server_role,
     /*signals_ready=*/true},
    {"factory_multi_endpoint_client", factory_multi_endpoint_client_role,
     /*signals_ready=*/false},
    {"factory_stalled_bootstrap_server", factory_stalled_bootstrap_server_role,
     /*signals_ready=*/true},
    {"factory_stalled_bootstrap_client", factory_stalled_bootstrap_client_role,
     /*signals_ready=*/false},
    {"dread_server", dread_server_role, /*signals_ready=*/true},
    {"dread_client", dread_client_role, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kAllRolesConfig = {
    /*.roles=*/kAllRoles,
    /*.role_count=*/sizeof(kAllRoles) / sizeof(kAllRoles[0]),
};
IREE_COORDINATED_TEST_REGISTER(kAllRolesConfig);

//===----------------------------------------------------------------------===//
// Proactor availability check
//===----------------------------------------------------------------------===//

// Checks that a platform proactor can be created. If unavailable (e.g.,
// missing kernel support), the test is skipped. Called in the launcher
// process before spawning children.
static bool ProactorAvailable() {
  iree_async_proactor_t* proactor = nullptr;
  iree_status_t status =
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor);
  if (iree_status_is_unavailable(status)) {
    iree_status_free(status);
    return false;
  }
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return false;
  }
  iree_async_proactor_release(proactor);
  return true;
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

TEST(XProcCarrier, HandshakeAndCreate) {
  if (!ProactorAvailable()) GTEST_SKIP() << "Platform proactor unavailable";
  ASSERT_EQ(0, iree_coordinated_test_run(iree_coordinated_test_argc(),
                                         iree_coordinated_test_argv(),
                                         &kHandshakeConfig));
}

TEST(XProcCarrier, SendRecvRoundTrip) {
  if (!ProactorAvailable()) GTEST_SKIP() << "Platform proactor unavailable";
  ASSERT_EQ(0, iree_coordinated_test_run(iree_coordinated_test_argc(),
                                         iree_coordinated_test_argv(),
                                         &kSendRecvConfig));
}

TEST(XProcCarrier, DirectWriteSignaling) {
  if (!ProactorAvailable()) GTEST_SKIP() << "Platform proactor unavailable";
  ASSERT_EQ(0, iree_coordinated_test_run(iree_coordinated_test_argc(),
                                         iree_coordinated_test_argv(),
                                         &kDirectWriteConfig));
}

#if IREE_FILE_IO_ENABLE
TEST(XProcCarrier, FileTransferSideband) {
  if (!ProactorAvailable()) GTEST_SKIP() << "Platform proactor unavailable";
  ASSERT_EQ(0, iree_coordinated_test_run(iree_coordinated_test_argc(),
                                         iree_coordinated_test_argv(),
                                         &kFileTransferConfig));
}
#endif  // IREE_FILE_IO_ENABLE

TEST(XProcCarrier, FactoryMultiEndpointConnection) {
  if (!ProactorAvailable()) GTEST_SKIP() << "Platform proactor unavailable";
  ASSERT_EQ(0, iree_coordinated_test_run(iree_coordinated_test_argc(),
                                         iree_coordinated_test_argv(),
                                         &kFactoryMultiEndpointConfig));
}

TEST(XProcCarrier, FactoryStopCancelsStalledBootstrap) {
  if (!ProactorAvailable()) GTEST_SKIP() << "Platform proactor unavailable";
  ASSERT_EQ(0, iree_coordinated_test_run(iree_coordinated_test_argc(),
                                         iree_coordinated_test_argv(),
                                         &kFactoryStalledBootstrapConfig));
}

TEST(XProcCarrier, DirectReadAcrossProcesses) {
  if (!ProactorAvailable()) GTEST_SKIP() << "Platform proactor unavailable";
  ASSERT_EQ(0, iree_coordinated_test_run(iree_coordinated_test_argc(),
                                         iree_coordinated_test_argv(),
                                         &kDirectReadConfig));
}

}  // namespace
