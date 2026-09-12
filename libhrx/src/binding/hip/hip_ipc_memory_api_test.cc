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
#include <cstring>
#include <limits>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"

namespace {

constexpr char kImporterArgument[] = "--hrx-ipc-memory-importer-fd";
constexpr char kExporterLibraryPathEnv[] = "HRX_TEST_EXPORT_LIBAMDHIP64";
constexpr char kImporterLibraryPathEnv[] = "HRX_TEST_IMPORT_LIBAMDHIP64";
constexpr size_t kElementCount = 256;
constexpr size_t kExtentSizes[] = {1, 257};
constexpr size_t kIpcAllocationSizeOffset = 32;
constexpr uint32_t kInitialPatternSeed = UINT32_C(0x13579BDF);
constexpr uint32_t kUpdatedPatternSeed = UINT32_C(0x2468ACE0);

const char* DefaultHrxLibPath() {
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

const char* RuntimeLibPath(const char* process_path_env) {
  if (const char* env = std::getenv(process_path_env); env && *env != '\0') {
    return env;
  }
  return DefaultHrxLibPath();
}

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipHALDeinitFn = hipError_t (*)(void);
using HipMallocFn = hipError_t (*)(void** device_ptr, size_t byte_length);
using HipFreeFn = hipError_t (*)(hipDeviceptr_t device_ptr);
using HipMemcpyFn = hipError_t (*)(void* target, const void* source,
                                   size_t byte_length, hipMemcpyKind kind);
using HipMemGetAddressRangeFn = hipError_t (*)(hipDeviceptr_t* base,
                                               size_t* size,
                                               hipDeviceptr_t device_ptr);
using HipIpcGetMemHandleFn = hipError_t (*)(hipIpcMemHandle_t* handle,
                                            void* device_ptr);
using HipIpcOpenMemHandleFn = hipError_t (*)(void** device_ptr,
                                             hipIpcMemHandle_t handle,
                                             unsigned int flags);
using HipIpcCloseMemHandleFn = hipError_t (*)(void* device_ptr);

template <typename T>
T ResolveHipSymbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

struct HipRuntimeApi {
  bool Load(const char* process_path_env) {
    const char* library_path = RuntimeLibPath(process_path_env);
    if (!library_path) {
      std::fprintf(stderr, "libamdhip64 path was not provided\n");
      return false;
    }
    const char* default_hrx_library_path = DefaultHrxLibPath();
    is_hrx_runtime = default_hrx_library_path &&
                     std::strcmp(library_path, default_hrx_library_path) == 0;
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
    hal_deinit =
        ResolveHipSymbol<decltype(hal_deinit)>(library, "hipHALDeinit");
    if (!hal_deinit && is_hrx_runtime) {
      std::fprintf(stderr, "cannot resolve hipHALDeinit\n");
      return false;
    }
    HRX_RESOLVE_HIP(malloc, "hipMalloc");
    HRX_RESOLVE_HIP(free, "hipFree");
    HRX_RESOLVE_HIP(memcpy, "hipMemcpy");
    HRX_RESOLVE_HIP(mem_get_address_range, "hipMemGetAddressRange");
    HRX_RESOLVE_HIP(ipc_get_mem_handle, "hipIpcGetMemHandle");
    HRX_RESOLVE_HIP(ipc_open_mem_handle, "hipIpcOpenMemHandle");
    HRX_RESOLVE_HIP(ipc_close_mem_handle, "hipIpcCloseMemHandle");
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
    return hal_deinit ? hal_deinit() : hipSuccess;
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

  // Loaded independently after exec and retained until process exit.
  void* library = nullptr;
  // Whether hipInit succeeded and process-local shutdown remains pending.
  bool initialized = false;
  // Whether this process loaded HRX under test rather than an interop peer.
  bool is_hrx_runtime = false;
  // Initializes the loaded HIP runtime.
  HipInitFn init = nullptr;
  // Deinitializes the loaded runtime when it exports the HRX extension.
  HipHALDeinitFn hal_deinit = nullptr;
  // Allocates device memory owned by the exporting process.
  HipMallocFn malloc = nullptr;
  // Releases exporting allocations and tests imported-pointer rejection.
  HipFreeFn free = nullptr;
  // Copies validation patterns between host and device memory.
  HipMemcpyFn memcpy = nullptr;
  // Queries the visible bounds of an exported or imported allocation.
  HipMemGetAddressRangeFn mem_get_address_range = nullptr;
  // Exports an allocation into a process-independent memory handle.
  HipIpcGetMemHandleFn ipc_get_mem_handle = nullptr;
  // Imports an exported allocation into the receiving process.
  HipIpcOpenMemHandleFn ipc_open_mem_handle = nullptr;
  // Releases one reference to an imported allocation.
  HipIpcCloseMemHandleFn ipc_close_mem_handle = nullptr;
};

bool HipSucceeded(hipError_t result, const char* operation) {
  if (result == hipSuccess) return true;
  std::fprintf(stderr, "%s failed: %d\n", operation, static_cast<int>(result));
  return false;
}

void FillPattern(uint32_t seed, uint32_t* data) {
  for (size_t i = 0; i < kElementCount; ++i) {
    data[i] = seed ^ (static_cast<uint32_t>(i + 1) * UINT32_C(0x9E3779B9));
  }
}

bool CopyAndVerifyPattern(HipRuntimeApi& api, void* device_ptr, uint32_t seed,
                          const char* operation) {
  uint32_t actual[kElementCount] = {};
  uint32_t expected[kElementCount] = {};
  FillPattern(seed, expected);
  if (!HipSucceeded(
          api.memcpy(actual, device_ptr, sizeof(actual), hipMemcpyDeviceToHost),
          operation)) {
    return false;
  }
  if (std::memcmp(actual, expected, sizeof(actual)) != 0) {
    std::fprintf(stderr, "%s returned mismatched allocation contents\n",
                 operation);
    return false;
  }
  return true;
}

bool IsZeroed(const void* data, size_t data_length) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < data_length; ++i) {
    if (bytes[i] != 0) return false;
  }
  return true;
}

uint64_t IpcHandleAllocationSize(const hipIpcMemHandle_t& handle) {
  static_assert(kIpcAllocationSizeOffset + sizeof(uint64_t) <=
                sizeof(handle.reserved));
  uint64_t allocation_size = 0;
  std::memcpy(&allocation_size, handle.reserved + kIpcAllocationSizeOffset,
              sizeof(allocation_size));
  return allocation_size;
}

bool ValidateExactAddressRange(HipRuntimeApi& api, void* device_ptr,
                               size_t expected_size, const char* operation) {
  hipDeviceptr_t base = reinterpret_cast<hipDeviceptr_t>(uintptr_t{1});
  size_t size = std::numeric_limits<size_t>::max();
  if (!HipSucceeded(api.mem_get_address_range(&base, &size, device_ptr),
                    operation) ||
      base != device_ptr || size != expected_size) {
    std::fprintf(stderr,
                 "%s returned base %p and size %zu, expected %p and %zu\n",
                 operation, base, size, device_ptr, expected_size);
    return false;
  }

  const hipDeviceptr_t end = reinterpret_cast<hipDeviceptr_t>(
      reinterpret_cast<uintptr_t>(device_ptr) + expected_size);
  const hipDeviceptr_t sentinel_base =
      reinterpret_cast<hipDeviceptr_t>(uintptr_t{1});
  base = sentinel_base;
  size = std::numeric_limits<size_t>::max();
  const hipError_t end_result = api.mem_get_address_range(&base, &size, end);
  if (api.is_hrx_runtime &&
      (end_result == hipSuccess || base != sentinel_base ||
       size != std::numeric_limits<size_t>::max())) {
    std::fprintf(stderr,
                 "%s accepted the first byte beyond the allocation or "
                 "modified failure outputs (result=%d, base=%p, size=%zu)\n",
                 operation, static_cast<int>(end_result), base, size);
    return false;
  }

  uint8_t source[258] = {};
  if (!HipSucceeded(
          api.memcpy(device_ptr, source, expected_size, hipMemcpyHostToDevice),
          "exact-boundary hipMemcpy")) {
    std::fprintf(stderr,
                 "%s rejected a transfer ending at the allocation boundary\n",
                 operation);
    return false;
  }
  const hipError_t overrun_result =
      api.memcpy(device_ptr, source, expected_size + 1, hipMemcpyHostToDevice);
  if (overrun_result == hipSuccess) {
    std::fprintf(stderr,
                 "%s accepted a transfer extending beyond the allocation\n",
                 operation);
    return false;
  }
  return true;
}

bool SendAll(int socket_fd, const void* data, size_t data_length) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  size_t offset = 0;
  while (offset < data_length) {
    ssize_t send_count = -1;
    do {
      send_count =
          send(socket_fd, bytes + offset, data_length - offset, MSG_NOSIGNAL);
    } while (send_count < 0 && errno == EINTR);
    if (send_count < 0) {
      std::fprintf(stderr, "send failed after %zu of %zu bytes: %s\n", offset,
                   data_length, std::strerror(errno));
      return false;
    }
    if (send_count == 0) {
      std::fprintf(stderr, "send returned zero after %zu of %zu bytes\n",
                   offset, data_length);
      return false;
    }
    offset += static_cast<size_t>(send_count);
  }
  return true;
}

bool ReceiveAll(int socket_fd, void* data, size_t data_length) {
  auto* bytes = static_cast<uint8_t*>(data);
  size_t offset = 0;
  while (offset < data_length) {
    ssize_t receive_count = -1;
    do {
      receive_count = recv(socket_fd, bytes + offset, data_length - offset, 0);
    } while (receive_count < 0 && errno == EINTR);
    if (receive_count < 0) {
      std::fprintf(stderr, "recv failed after %zu of %zu bytes: %s\n", offset,
                   data_length, std::strerror(errno));
      return false;
    }
    if (receive_count == 0) {
      std::fprintf(stderr, "recv reached EOF after %zu of %zu bytes\n", offset,
                   data_length);
      return false;
    }
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

bool ImportExactExtents(HipRuntimeApi& api, int socket_fd) {
  for (size_t expected_size : kExtentSizes) {
    uint64_t requested_size = 0;
    hipIpcMemHandle_t handle = {};
    if (!ReceiveAll(socket_fd, &requested_size, sizeof(requested_size)) ||
        !ReceiveAll(socket_fd, &handle, sizeof(handle)) ||
        requested_size != expected_size) {
      std::fprintf(stderr,
                   "importer did not receive the expected %zu-byte handle\n",
                   expected_size);
      return false;
    }

    void* allocation = nullptr;
    if (!HipSucceeded(api.ipc_open_mem_handle(&allocation, handle,
                                              hipIpcMemLazyEnablePeerAccess),
                      "extent hipIpcOpenMemHandle") ||
        !allocation) {
      return false;
    }
    bool succeeded = ValidateExactAddressRange(
        api, allocation, expected_size, "imported hipMemGetAddressRange");
    succeeded = HipSucceeded(api.ipc_close_mem_handle(allocation),
                             "extent hipIpcCloseMemHandle") &&
                succeeded;
    if (!succeeded || !SendMarker(socket_fd, 'E')) return false;
  }
  return true;
}

int ImporterProcessMain(int socket_fd) {
  HipRuntimeApi api;
  if (!api.Load(kImporterLibraryPathEnv) ||
      !HipSucceeded(api.Initialize(), "importer hipInit")) {
    return 1;
  }

  hipIpcMemHandle_t handle = {};
  if (!ReceiveAll(socket_fd, &handle, sizeof(handle))) {
    std::fprintf(stderr, "importer did not receive a memory handle\n");
    return 1;
  }

  if (api.is_hrx_runtime) {
    void* output = reinterpret_cast<void*>(uintptr_t{1});
    if (api.ipc_open_mem_handle(&output, handle, /*flags=*/0) !=
            hipErrorInvalidValue ||
        output != nullptr) {
      std::fprintf(stderr, "invalid open flags did not clear the output\n");
      return 1;
    }
    output = reinterpret_cast<void*>(uintptr_t{1});
    if (api.ipc_open_mem_handle(
            &output, handle,
            hipIpcMemLazyEnablePeerAccess | static_cast<unsigned int>(2)) !=
            hipErrorInvalidValue ||
        output != nullptr ||
        api.ipc_open_mem_handle(nullptr, handle,
                                hipIpcMemLazyEnablePeerAccess) !=
            hipErrorInvalidValue ||
        api.ipc_close_mem_handle(nullptr) != hipErrorInvalidValue) {
      std::fprintf(stderr, "importer failed public argument validation\n");
      return 1;
    }
  }

  void* first_device_ptr = nullptr;
  void* second_device_ptr = nullptr;
  if (!HipSucceeded(api.ipc_open_mem_handle(&first_device_ptr, handle,
                                            hipIpcMemLazyEnablePeerAccess),
                    "first hipIpcOpenMemHandle") ||
      !first_device_ptr ||
      !CopyAndVerifyPattern(api, first_device_ptr, kInitialPatternSeed,
                            "initial imported hipMemcpy")) {
    std::fprintf(stderr, "importer failed initial open validation\n");
    return 1;
  }
  if (api.is_hrx_runtime &&
      (!HipSucceeded(api.ipc_open_mem_handle(&second_device_ptr, handle,
                                             hipIpcMemLazyEnablePeerAccess),
                     "duplicate hipIpcOpenMemHandle") ||
       first_device_ptr != second_device_ptr)) {
    std::fprintf(stderr, "importer failed duplicate-open validation\n");
    return 1;
  }

  if (!SendMarker(socket_fd, 'R') || !ReceiveMarker(socket_fd, 'U') ||
      !CopyAndVerifyPattern(api, first_device_ptr, kUpdatedPatternSeed,
                            "updated imported hipMemcpy")) {
    std::fprintf(stderr, "importer failed shared-data update validation\n");
    return 1;
  }

  if (api.is_hrx_runtime) {
    int unrelated_storage = 0;
    if (api.free(first_device_ptr) != hipErrorInvalidValue ||
        api.ipc_close_mem_handle(&unrelated_storage) != hipErrorInvalidValue ||
        !HipSucceeded(api.ipc_close_mem_handle(first_device_ptr),
                      "first hipIpcCloseMemHandle") ||
        !CopyAndVerifyPattern(api, second_device_ptr, kUpdatedPatternSeed,
                              "post-close imported hipMemcpy") ||
        api.free(second_device_ptr) != hipErrorInvalidValue ||
        !HipSucceeded(api.ipc_close_mem_handle(second_device_ptr),
                      "final hipIpcCloseMemHandle") ||
        api.ipc_close_mem_handle(first_device_ptr) != hipErrorInvalidValue ||
        api.free(first_device_ptr) != hipErrorInvalidValue) {
      std::fprintf(stderr, "importer failed close/free lifetime validation\n");
      return 1;
    }
  } else if (!HipSucceeded(api.ipc_close_mem_handle(first_device_ptr),
                           "interop hipIpcCloseMemHandle")) {
    return 1;
  }

  if (!SendMarker(socket_fd, 'C') || !ImportExactExtents(api, socket_fd) ||
      !ReceiveMarker(socket_fd, 'F') ||
      !HipSucceeded(api.Shutdown(), "importer hipHALDeinit") ||
      !SendMarker(socket_fd, 'D')) {
    std::fprintf(stderr, "importer failed final lifetime handshake\n");
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

class ScopedFileDescriptor {
 public:
  explicit ScopedFileDescriptor(int file_descriptor)
      : file_descriptor_(file_descriptor) {}

  ~ScopedFileDescriptor() {
    if (file_descriptor_ >= 0) (void)close(file_descriptor_);
  }

  ScopedFileDescriptor(const ScopedFileDescriptor&) = delete;
  ScopedFileDescriptor& operator=(const ScopedFileDescriptor&) = delete;

  int get() const { return file_descriptor_; }

  bool Close() {
    if (file_descriptor_ < 0) return true;
    const int file_descriptor = file_descriptor_;
    file_descriptor_ = -1;
    return close(file_descriptor) == 0;
  }

  int Release() {
    const int file_descriptor = file_descriptor_;
    file_descriptor_ = -1;
    return file_descriptor;
  }

 private:
  int file_descriptor_;
};

bool ExportExactExtents(HipRuntimeApi& api, int socket_fd) {
  for (size_t requested_size : kExtentSizes) {
    void* allocation = nullptr;
    bool succeeded =
        HipSucceeded(api.malloc(&allocation, requested_size),
                     "extent hipMalloc") &&
        allocation &&
        ValidateExactAddressRange(api, allocation, requested_size,
                                  "exported hipMemGetAddressRange");

    hipIpcMemHandle_t handle = {};
    if (succeeded) {
      succeeded = HipSucceeded(api.ipc_get_mem_handle(&handle, allocation),
                               "extent hipIpcGetMemHandle") &&
                  IpcHandleAllocationSize(handle) == requested_size;
      if (!succeeded) {
        std::fprintf(stderr,
                     "exported IPC handle size did not match %zu bytes\n",
                     requested_size);
      }
    }
    if (succeeded) {
      const uint64_t wire_size = requested_size;
      succeeded = SendAll(socket_fd, &wire_size, sizeof(wire_size)) &&
                  SendAll(socket_fd, &handle, sizeof(handle)) &&
                  ReceiveMarker(socket_fd, 'E');
    }

    if (allocation) {
      succeeded =
          HipSucceeded(api.free(allocation), "extent hipFree") && succeeded;
    }
    if (!succeeded) return false;
  }
  return true;
}

bool ExporterProcessMain(int socket_fd, ChildProcess& child) {
  HipRuntimeApi api;
  void* allocation = nullptr;
  bool succeeded = api.Load(kExporterLibraryPathEnv) &&
                   HipSucceeded(api.Initialize(), "exporter hipInit");

  uint32_t initial_data[kElementCount] = {};
  FillPattern(kInitialPatternSeed, initial_data);
  hipIpcMemHandle_t invalid_handle;
  std::memset(&invalid_handle, 0xA5, sizeof(invalid_handle));
  if (succeeded) {
    succeeded =
        HipSucceeded(api.malloc(&allocation, sizeof(initial_data)),
                     "hipMalloc") &&
        HipSucceeded(api.memcpy(allocation, initial_data, sizeof(initial_data),
                                hipMemcpyHostToDevice),
                     "initial exporter hipMemcpy");
  }
  if (succeeded && api.is_hrx_runtime) {
    succeeded =
        api.ipc_get_mem_handle(nullptr, allocation) == hipErrorInvalidValue &&
        api.ipc_get_mem_handle(&invalid_handle, nullptr) ==
            hipErrorInvalidValue &&
        IsZeroed(&invalid_handle, sizeof(invalid_handle)) &&
        api.ipc_close_mem_handle(allocation) == hipErrorInvalidValue;
  }

  hipIpcMemHandle_t handle = {};
  if (succeeded) {
    succeeded = HipSucceeded(api.ipc_get_mem_handle(&handle, allocation),
                             "hipIpcGetMemHandle") &&
                !IsZeroed(&handle, sizeof(handle)) &&
                SendAll(socket_fd, &handle, sizeof(handle)) &&
                ReceiveMarker(socket_fd, 'R');
  }

  uint32_t updated_data[kElementCount] = {};
  FillPattern(kUpdatedPatternSeed, updated_data);
  if (succeeded) {
    succeeded =
        HipSucceeded(api.memcpy(allocation, updated_data, sizeof(updated_data),
                                hipMemcpyHostToDevice),
                     "updated exporter hipMemcpy") &&
        SendMarker(socket_fd, 'U') && ReceiveMarker(socket_fd, 'C');
  }

  if (succeeded) {
    succeeded = HipSucceeded(api.free(allocation), "exporter hipFree");
    if (succeeded) allocation = nullptr;
  }
  if (succeeded) succeeded = ExportExactExtents(api, socket_fd);
  if (succeeded) {
    succeeded = SendMarker(socket_fd, 'F') && ReceiveMarker(socket_fd, 'D');
  }

  if (!succeeded) child.Terminate();
  if (allocation) {
    succeeded =
        HipSucceeded(api.free(allocation), "cleanup hipFree") && succeeded;
  }
  succeeded =
      HipSucceeded(api.Shutdown(), "exporter hipHALDeinit") && succeeded;
  (void)shutdown(socket_fd, SHUT_RDWR);
  (void)close(socket_fd);
  if (!succeeded) return false;
  return child.WaitForSuccess();
}

TEST(HipIpcMemoryApiTest, SharesAllocationAcrossIndependentProcesses) {
  int sockets[2] = {-1, -1};
  ASSERT_EQ(0, socketpair(AF_UNIX, SOCK_STREAM, /*protocol=*/0, sockets));
  ScopedFileDescriptor exporter_socket(sockets[0]);
  ScopedFileDescriptor importer_socket(sockets[1]);

  char child_socket_argument[32] = {};
  const int argument_length =
      std::snprintf(child_socket_argument, sizeof(child_socket_argument), "%d",
                    importer_socket.get());
  ASSERT_GT(argument_length, 0);
  ASSERT_LT(static_cast<size_t>(argument_length),
            sizeof(child_socket_argument));

  // Neither process has loaded libamdhip64 or initialized ROCr. The child
  // immediately replaces its image and initializes its selected runtime
  // independently.
  const pid_t process_id = fork();
  ASSERT_GE(process_id, 0);
  if (process_id == 0) {
    if (!exporter_socket.Close()) _exit(126);
    execl("/proc/self/exe", "hip_ipc_memory_api_test", kImporterArgument,
          child_socket_argument, static_cast<char*>(nullptr));
    _exit(127);
  }

  ChildProcess child(process_id);
  ASSERT_TRUE(importer_socket.Close());
  EXPECT_TRUE(ExporterProcessMain(exporter_socket.Release(), child));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::strcmp(argv[1], kImporterArgument) == 0) {
    errno = 0;
    char* end = nullptr;
    const long socket_fd = std::strtol(argv[2], &end, 10);
    if (errno != 0 || end == argv[2] || *end != '\0' || socket_fd < 0 ||
        socket_fd > std::numeric_limits<int>::max()) {
      std::fprintf(stderr, "invalid importer socket descriptor: %s\n", argv[2]);
      return 1;
    }
    const int result = ImporterProcessMain(static_cast<int>(socket_fd));
    (void)shutdown(static_cast<int>(socket_fd), SHUT_RDWR);
    (void)close(static_cast<int>(socket_fd));
    return result;
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
