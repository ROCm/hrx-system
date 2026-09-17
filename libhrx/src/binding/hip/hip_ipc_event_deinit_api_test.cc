// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"

namespace {

const char* CandidateLibPath() {
  if (const char* env = std::getenv("HRX_TEST_LIBAMDHIP64");
      env && *env != '\0') {
    return env;
  }
#ifdef HRX_TEST_LIBAMDHIP64_PATH
  return HRX_TEST_LIBAMDHIP64_PATH;
#else
  return nullptr;
#endif
}

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipHALDeinitFn = hipError_t (*)(void);
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipCtxCreateFn = hipError_t (*)(hipCtx_t* context, unsigned int flags,
                                      hipDevice_t device);
using HipStreamCreateWithFlagsFn = hipError_t (*)(hipStream_t* stream,
                                                  unsigned int flags);
using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t stream);
using HipStreamWaitEventFn = hipError_t (*)(hipStream_t stream,
                                            hipEvent_t event,
                                            unsigned int flags);
using HipEventCreateWithFlagsFn = hipError_t (*)(hipEvent_t* event,
                                                 unsigned int flags);
using HipEventDestroyFn = hipError_t (*)(hipEvent_t event);
using HipEventRecordFn = hipError_t (*)(hipEvent_t event, hipStream_t stream);
using HipEventQueryFn = hipError_t (*)(hipEvent_t event);
using HipEventSynchronizeFn = hipError_t (*)(hipEvent_t event);
using HipIpcGetEventHandleFn = hipError_t (*)(hipIpcEventHandle_t* handle,
                                              hipEvent_t event);
using HipIpcOpenEventHandleFn = hipError_t (*)(hipEvent_t* event,
                                               hipIpcEventHandle_t handle);

template <typename T>
T ResolveHipSymbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

struct HipRuntimeApi {
  bool Load() {
    const char* library_path = CandidateLibPath();
    if (!library_path) {
      std::fprintf(stderr, "libamdhip64 path was not provided\n");
      return false;
    }
    library = dlopen(library_path, RTLD_LAZY | RTLD_LOCAL);
    if (!library) {
      std::fprintf(stderr, "cannot dlopen %s: %s\n", library_path, dlerror());
      return false;
    }

#define HRX_RESOLVE_HIP(field, symbol)                          \
  do {                                                          \
    field = ResolveHipSymbol<decltype(field)>(library, symbol); \
    if (!field) {                                               \
      std::fprintf(stderr, "cannot resolve %s\n", symbol);      \
      return false;                                             \
    }                                                           \
  } while (false)
    HRX_RESOLVE_HIP(init, "hipInit");
    HRX_RESOLVE_HIP(hal_deinit, "hipHALDeinit");
    HRX_RESOLVE_HIP(get_device, "hipGetDevice");
    HRX_RESOLVE_HIP(context_create, "hipCtxCreate");
    HRX_RESOLVE_HIP(stream_create_with_flags, "hipStreamCreateWithFlags");
    HRX_RESOLVE_HIP(stream_synchronize, "hipStreamSynchronize");
    HRX_RESOLVE_HIP(stream_wait_event, "hipStreamWaitEvent");
    HRX_RESOLVE_HIP(event_create_with_flags, "hipEventCreateWithFlags");
    HRX_RESOLVE_HIP(event_destroy, "hipEventDestroy");
    HRX_RESOLVE_HIP(event_record, "hipEventRecord");
    HRX_RESOLVE_HIP(event_query, "hipEventQuery");
    HRX_RESOLVE_HIP(event_synchronize, "hipEventSynchronize");
    HRX_RESOLVE_HIP(ipc_get_event_handle, "hipIpcGetEventHandle");
    HRX_RESOLVE_HIP(ipc_open_event_handle, "hipIpcOpenEventHandle");
#undef HRX_RESOLVE_HIP
    return true;
  }

  hipError_t Initialize() {
    const hipError_t result = init(/*flags=*/0);
    initialized = result == hipSuccess;
    return result;
  }

  hipError_t Shutdown() {
    if (!initialized) return hipSuccess;
    initialized = false;
    return hal_deinit();
  }

  ~HipRuntimeApi() {
    if (initialized) {
      const hipError_t result = Shutdown();
      if (result != hipSuccess) {
        std::fprintf(stderr, "hipHALDeinit failed: %d\n",
                     static_cast<int>(result));
      }
    }
  }

  // Loaded after fork and intentionally retained until process exit.
  void* library = nullptr;
  bool initialized = false;
  HipInitFn init = nullptr;
  HipHALDeinitFn hal_deinit = nullptr;
  HipGetDeviceFn get_device = nullptr;
  HipCtxCreateFn context_create = nullptr;
  HipStreamCreateWithFlagsFn stream_create_with_flags = nullptr;
  HipStreamSynchronizeFn stream_synchronize = nullptr;
  HipStreamWaitEventFn stream_wait_event = nullptr;
  HipEventCreateWithFlagsFn event_create_with_flags = nullptr;
  HipEventDestroyFn event_destroy = nullptr;
  HipEventRecordFn event_record = nullptr;
  HipEventQueryFn event_query = nullptr;
  HipEventSynchronizeFn event_synchronize = nullptr;
  HipIpcGetEventHandleFn ipc_get_event_handle = nullptr;
  HipIpcOpenEventHandleFn ipc_open_event_handle = nullptr;
};

bool HipSucceeded(hipError_t result, const char* operation) {
  if (result == hipSuccess) return true;
  std::fprintf(stderr, "%s failed: %d\n", operation, static_cast<int>(result));
  return false;
}

struct ProcessHandles {
  // Event used for a completed cross-process stream wait.
  hipIpcEventHandle_t completed_wait;
  // Event left pending while the importing runtime deinitializes.
  hipIpcEventHandle_t pending_deinit_wait;
};

bool SendAll(int socket_fd, const void* data, size_t data_length) {
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  size_t offset = 0;
  while (offset < data_length) {
    ssize_t send_count = -1;
    do {
      send_count =
          send(socket_fd, bytes + offset, data_length - offset, MSG_NOSIGNAL);
    } while (send_count < 0 && errno == EINTR);
    if (send_count <= 0) return false;
    offset += static_cast<size_t>(send_count);
  }
  return true;
}

bool ReceiveAll(int socket_fd, void* data, size_t data_length) {
  uint8_t* bytes = static_cast<uint8_t*>(data);
  size_t offset = 0;
  while (offset < data_length) {
    ssize_t receive_count = -1;
    do {
      receive_count = recv(socket_fd, bytes + offset, data_length - offset, 0);
    } while (receive_count < 0 && errno == EINTR);
    if (receive_count <= 0) return false;
    offset += static_cast<size_t>(receive_count);
  }
  return true;
}

bool SendMarker(int socket_fd, char marker) {
  return SendAll(socket_fd, &marker, sizeof(marker));
}

bool ReceiveMarker(int socket_fd, char expected_marker) {
  char marker = 0;
  if (!ReceiveAll(socket_fd, &marker, sizeof(marker))) {
    std::fprintf(stderr, "did not receive process marker '%c'\n",
                 expected_marker);
    return false;
  }
  if (marker != expected_marker) {
    std::fprintf(stderr, "received process marker '%c', expected '%c'\n",
                 marker, expected_marker);
    return false;
  }
  return true;
}

bool RunCompletedEventCycle(HipRuntimeApi& api) {
  hipEvent_t event = nullptr;
  if (api.event_create_with_flags(
          &event, hipEventInterprocess | hipEventDisableTiming) != hipSuccess) {
    return false;
  }
  const bool completed =
      api.event_record(event, /*stream=*/nullptr) == hipSuccess &&
      api.event_synchronize(event) == hipSuccess;
  return api.event_destroy(event) == hipSuccess && completed;
}

int ImporterProcessMain(int socket_fd) {
  HipRuntimeApi api;
  if (!api.Load() || !HipSucceeded(api.Initialize(), "importer hipInit")) {
    return 1;
  }

  ProcessHandles handles = {};
  if (!ReceiveAll(socket_fd, &handles, sizeof(handles))) {
    std::fprintf(stderr, "importer did not receive event handles\n");
    return 1;
  }

  hipEvent_t completed_event = nullptr;
  hipEvent_t duplicate_event = nullptr;
  hipEvent_t pending_event = nullptr;
  if (!HipSucceeded(
          api.ipc_open_event_handle(&completed_event, handles.completed_wait),
          "first hipIpcOpenEventHandle") ||
      !HipSucceeded(
          api.ipc_open_event_handle(&duplicate_event, handles.completed_wait),
          "duplicate hipIpcOpenEventHandle") ||
      !HipSucceeded(api.ipc_open_event_handle(&pending_event,
                                              handles.pending_deinit_wait),
                    "pending hipIpcOpenEventHandle") ||
      completed_event == duplicate_event ||
      api.event_query(completed_event) != hipErrorNotReady ||
      api.event_query(duplicate_event) != hipErrorNotReady ||
      api.event_query(pending_event) != hipErrorNotReady) {
    std::fprintf(stderr, "importer failed pending event validation\n");
    return 1;
  }

  int device = -1;
  hipCtx_t wait_context = nullptr;
  hipStream_t wait_stream = nullptr;
  if (!HipSucceeded(api.get_device(&device), "hipGetDevice") ||
      !HipSucceeded(api.context_create(&wait_context, /*flags=*/0, device),
                    "hipCtxCreate") ||
      !HipSucceeded(
          api.stream_create_with_flags(&wait_stream, hipStreamNonBlocking),
          "hipStreamCreateWithFlags") ||
      !HipSucceeded(
          api.stream_wait_event(wait_stream, completed_event, /*flags=*/0),
          "completed hipStreamWaitEvent") ||
      !SendMarker(socket_fd, 'W') ||
      !HipSucceeded(api.stream_synchronize(wait_stream),
                    "completed hipStreamSynchronize") ||
      api.event_query(completed_event) != hipSuccess ||
      api.event_query(duplicate_event) != hipSuccess ||
      !HipSucceeded(api.event_destroy(completed_event),
                    "first imported hipEventDestroy") ||
      !HipSucceeded(api.event_destroy(duplicate_event),
                    "duplicate imported hipEventDestroy")) {
    std::fprintf(stderr, "importer failed completed event wait\n");
    return 1;
  }

  // hipStreamWaitEvent returns only after the monitor-owned callback and proxy
  // semaphore are published. Destroying the imported handle then leaves those
  // resources owned solely by the submitted wait. The stream and its context
  // deliberately remain live and unsynchronized while the exporter keeps the
  // source event pending through the importer's deinit/reinit cycle.
  if (!HipSucceeded(
          api.stream_wait_event(wait_stream, pending_event, /*flags=*/0),
          "pending hipStreamWaitEvent") ||
      !HipSucceeded(api.event_destroy(pending_event),
                    "pending imported hipEventDestroy") ||
      api.event_query(pending_event) != hipErrorInvalidResourceHandle ||
      !SendMarker(socket_fd, 'P') ||
      !HipSucceeded(api.Shutdown(), "pending-wait hipHALDeinit") ||
      !HipSucceeded(api.Initialize(), "second hipInit") ||
      !RunCompletedEventCycle(api) ||
      !HipSucceeded(api.Shutdown(), "second hipHALDeinit") ||
      !SendMarker(socket_fd, 'D')) {
    std::fprintf(stderr, "importer failed deinit/reinit cycle\n");
    return 1;
  }
  return 0;
}

class ChildProcess {
 public:
  explicit ChildProcess(pid_t process_id) : process_id_(process_id) {}

  ~ChildProcess() { Terminate(); }

  void Terminate() {
    if (process_id_ <= 0) return;
    (void)kill(process_id_, SIGKILL);
    int status = 0;
    while (waitpid(process_id_, &status, 0) < 0 && errno == EINTR) {
    }
    process_id_ = -1;
  }

  bool WaitForSuccess() {
    int status = 0;
    pid_t wait_result = -1;
    do {
      wait_result = waitpid(process_id_, &status, 0);
    } while (wait_result < 0 && errno == EINTR);
    process_id_ = -1;
    return wait_result > 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
  }

 private:
  pid_t process_id_;
};

bool ExporterProcessMain(int socket_fd, ChildProcess& child) {
  HipRuntimeApi api;
  if (!api.Load() || !HipSucceeded(api.Initialize(), "exporter hipInit")) {
    return false;
  }

  hipEvent_t completed_event = nullptr;
  hipEvent_t pending_event = nullptr;
  ProcessHandles handles = {};
  bool succeeded =
      HipSucceeded(
          api.event_create_with_flags(
              &completed_event, hipEventInterprocess | hipEventDisableTiming),
          "completed hipEventCreateWithFlags") &&
      HipSucceeded(
          api.event_create_with_flags(
              &pending_event, hipEventInterprocess | hipEventDisableTiming),
          "pending hipEventCreateWithFlags") &&
      HipSucceeded(
          api.ipc_get_event_handle(&handles.completed_wait, completed_event),
          "completed hipIpcGetEventHandle") &&
      HipSucceeded(
          api.ipc_get_event_handle(&handles.pending_deinit_wait, pending_event),
          "pending hipIpcGetEventHandle") &&
      api.event_query(completed_event) == hipErrorNotReady &&
      api.event_query(pending_event) == hipErrorNotReady &&
      SendAll(socket_fd, &handles, sizeof(handles)) &&
      ReceiveMarker(socket_fd, 'W');

  if (succeeded) {
    succeeded =
        HipSucceeded(api.event_record(completed_event, /*stream=*/nullptr),
                     "completed hipEventRecord") &&
        HipSucceeded(api.event_synchronize(completed_event),
                     "completed hipEventSynchronize") &&
        ReceiveMarker(socket_fd, 'P') &&
        api.event_query(pending_event) == hipErrorNotReady &&
        ReceiveMarker(socket_fd, 'D');
  }

  if (!succeeded) child.Terminate();

  // Keep the never-recorded source carrier live until the importer's second
  // deinit completed. On a failed handshake, closing the socket and process
  // guard ensure a wedged importer is killed and reaped.
  if (completed_event) {
    succeeded = HipSucceeded(api.event_destroy(completed_event),
                             "completed hipEventDestroy") &&
                succeeded;
  }
  if (pending_event) {
    succeeded = HipSucceeded(api.event_destroy(pending_event),
                             "pending hipEventDestroy") &&
                succeeded;
  }
  succeeded =
      HipSucceeded(api.Shutdown(), "exporter hipHALDeinit") && succeeded;
  (void)shutdown(socket_fd, SHUT_RDWR);
  (void)close(socket_fd);
  if (!succeeded) return false;
  return child.WaitForSuccess();
}

TEST(HipIpcEventDeinitApiTest,
     PendingImportedWaitDeinitializesAndReinitializes) {
  int sockets[2] = {-1, -1};
  ASSERT_EQ(0, socketpair(AF_UNIX, SOCK_STREAM, /*protocol=*/0, sockets));

  // Neither process has loaded libamdhip64 or initialized ROCr at this point.
  const pid_t process_id = fork();
  ASSERT_GE(process_id, 0);
  if (process_id == 0) {
    (void)close(sockets[0]);
    const int result = ImporterProcessMain(sockets[1]);
    (void)close(sockets[1]);
    _exit(result);
  }

  (void)close(sockets[1]);
  ChildProcess child(process_id);
  EXPECT_TRUE(ExporterProcessMain(sockets[0], child));
}

}  // namespace
