// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/aql/queue/publication.h"
#include "libamdf/cts/gpu/kernels/byte_copy_unaligned.h"
#include "libamdf/cts/gpu/kernels/byte_copy_unaligned_kernels.h"
#include "libamdf/cts/gpu/kernels/cross_process_progress.h"
#include "libamdf/cts/gpu/kernels/cross_process_progress_kernels.h"
#include "libamdf/cts/gpu/pm4/dispatch_fixture.h"
#include "util/device_cache.h"
#include "util/provider.h"

extern char** environ;

namespace {

constexpr int kChildSocketDescriptor = 198;
constexpr uint32_t kProtocolVersion = 1;

enum class MessageKind : uint32_t {
  kConfiguration = 1,
  kPrepared = 2,
  kActivate = 3,
  kEnteringActivation = 4,
  kActive = 5,
  kRelease = 6,
};

struct Message {
  // Version of the fixed parent/child protocol.
  uint32_t version;
  // Framed protocol state.
  MessageKind kind;
  // State-specific endpoint identity or status payload.
  uint64_t values[2];
};
static_assert(sizeof(Message) == 24);
static_assert(std::is_trivially_copyable_v<Message>);

bool IsOuterTestRunnerVariable(const char* entry) {
  constexpr std::array<std::string_view, 7> kNames = {
      "TEST_PREMATURE_EXIT_FILE", "TEST_TOTAL_SHARDS",  "TEST_SHARD_INDEX",
      "TEST_SHARD_STATUS_FILE",   "GTEST_TOTAL_SHARDS", "GTEST_SHARD_INDEX",
      "GTEST_SHARD_STATUS_FILE",
  };
  const std::string_view variable(entry);
  for (const std::string_view name : kNames) {
    if (variable.starts_with(name) && variable.size() > name.size() &&
        variable[name.size()] == '=') {
      return true;
    }
  }
  return false;
}

::testing::AssertionResult SendMessage(int descriptor, MessageKind kind,
                                       uint64_t first = 0,
                                       uint64_t second = 0) {
  const Message message = {
      .version = kProtocolVersion,
      .kind = kind,
      .values = {first, second},
  };
  ssize_t length = 0;
  do {
    length = send(descriptor, &message, sizeof(message), MSG_NOSIGNAL);
  } while (length < 0 && errno == EINTR);
  if (length < 0) {
    const int error = errno;
    return ::testing::AssertionFailure()
           << "protocol send failed: " << std::strerror(error);
  }
  if (length != sizeof(message)) {
    return ::testing::AssertionFailure()
           << "protocol send wrote " << length << " of " << sizeof(message)
           << " bytes";
  }
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult ReceiveMessage(int descriptor,
                                          MessageKind expected_kind,
                                          Message* out_message) {
  Message message = {};
  ssize_t length = 0;
  do {
    length = recv(descriptor, &message, sizeof(message), 0);
  } while (length < 0 && errno == EINTR);
  if (length < 0) {
    const int error = errno;
    return ::testing::AssertionFailure()
           << "protocol receive failed: " << std::strerror(error);
  }
  if (length != sizeof(message)) {
    return ::testing::AssertionFailure()
           << "protocol receive read " << length << " of " << sizeof(message)
           << " bytes";
  }
  if (message.version != kProtocolVersion) {
    return ::testing::AssertionFailure()
           << "protocol version " << message.version << ", expected "
           << kProtocolVersion;
  }
  if (message.kind != expected_kind) {
    return ::testing::AssertionFailure()
           << "protocol state " << static_cast<uint32_t>(message.kind)
           << ", expected " << static_cast<uint32_t>(expected_kind);
  }
  *out_message = message;
  return ::testing::AssertionSuccess();
}

class ChildProcess {
 public:
  ChildProcess() = default;
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;

  ~ChildProcess() {
    CloseSocket();
    Reap();
  }

  ::testing::AssertionResult Start(std::string_view library_path) {
    int sockets[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sockets) != 0) {
      const int error = errno;
      return ::testing::AssertionFailure()
             << "socketpair failed: " << std::strerror(error);
    }

    posix_spawn_file_actions_t file_actions;
    int result = posix_spawn_file_actions_init(&file_actions);
    if (result != 0) {
      close(sockets[0]);
      close(sockets[1]);
      return ::testing::AssertionFailure()
             << "spawn file-actions initialization failed: "
             << std::strerror(result);
    }
    // adddup2 also clears close-on-exec when both descriptors are equal.
    // Recording it unconditionally keeps the inherited endpoint valid even
    // if the socketpair happens to allocate the fixed child descriptor.
    result = posix_spawn_file_actions_adddup2(&file_actions, sockets[1],
                                              kChildSocketDescriptor);
    if (result == 0 && sockets[0] != kChildSocketDescriptor) {
      result = posix_spawn_file_actions_addclose(&file_actions, sockets[0]);
    }
    if (result == 0 && sockets[1] != kChildSocketDescriptor) {
      result = posix_spawn_file_actions_addclose(&file_actions, sockets[1]);
    }
    if (result != 0) {
      posix_spawn_file_actions_destroy(&file_actions);
      close(sockets[0]);
      close(sockets[1]);
      return ::testing::AssertionFailure()
             << "spawn file action failed: " << std::strerror(result);
    }

    std::array<std::string, 5> arguments = {
        "/proc/self/exe",
        "--amdf_native_lifetime=process",
        "--amdf_library=" + std::string(library_path),
        "--gtest_filter=CrossProcessProgressChildTest."
        "DISABLED_ActivateSelectedGpu",
        "--gtest_also_run_disabled_tests",
    };
    std::array<char*, 6> argument_values = {};
    for (size_t i = 0; i < arguments.size(); ++i) {
      argument_values[i] = arguments[i].data();
    }
    // The enclosing Bazel/GoogleTest invocation owns its exit marker and shard
    // identity. The re-exec is one controlled child role, not another shard;
    // sharing those variables lets it remove the parent's marker or skip the
    // only child case. All deployment and provider variables remain inherited.
    std::vector<char*> child_environment;
    for (char** entry = environ; *entry != nullptr; ++entry) {
      if (!IsOuterTestRunnerVariable(*entry)) {
        child_environment.push_back(*entry);
      }
    }
    child_environment.push_back(nullptr);
    result =
        posix_spawn(&process_id_, arguments[0].c_str(), &file_actions, nullptr,
                    argument_values.data(), child_environment.data());
    posix_spawn_file_actions_destroy(&file_actions);
    if (result != 0) {
      process_id_ = -1;
      close(sockets[0]);
      close(sockets[1]);
      return ::testing::AssertionFailure()
             << "child spawn failed: " << std::strerror(result);
    }
    close(sockets[1]);
    socket_descriptor_ = sockets[0];
    return ::testing::AssertionSuccess();
  }

  ::testing::AssertionResult Send(MessageKind kind, uint64_t first = 0,
                                  uint64_t second = 0) const {
    return SendMessage(socket_descriptor_, kind, first, second);
  }

  ::testing::AssertionResult Receive(MessageKind expected_kind,
                                     Message* out_message) const {
    return ReceiveMessage(socket_descriptor_, expected_kind, out_message);
  }

  ::testing::AssertionResult Wait() {
    CloseSocket();
    int status = 0;
    pid_t result = 0;
    do {
      result = waitpid(process_id_, &status, 0);
    } while (result < 0 && errno == EINTR);
    if (result < 0) {
      const int error = errno;
      return ::testing::AssertionFailure()
             << "child wait failed: " << std::strerror(error);
    }
    process_id_ = -1;
    if (!WIFEXITED(status)) {
      return ::testing::AssertionFailure()
             << "child terminated by signal " << WTERMSIG(status);
    }
    if (WEXITSTATUS(status) != 0) {
      return ::testing::AssertionFailure()
             << "child exited with status " << WEXITSTATUS(status);
    }
    return ::testing::AssertionSuccess();
  }

 private:
  void CloseSocket() {
    if (socket_descriptor_ >= 0) {
      close(socket_descriptor_);
      socket_descriptor_ = -1;
    }
  }

  void Reap() {
    if (process_id_ < 0) {
      return;
    }
    int status = 0;
    pid_t result = 0;
    do {
      result = waitpid(process_id_, &status, 0);
    } while (result < 0 && errno == EINTR);
    process_id_ = -1;
  }

  // Parent endpoint of the framed process-control channel.
  int socket_descriptor_ = -1;
  // Live child process, reaped exactly once.
  pid_t process_id_ = -1;
};

::testing::AssertionResult FindLoadedProviderPath(std::string* out_path) {
  const amdf_query_api_fn_t query_api = amdf_cts_provider_query_api();
  static_assert(sizeof(query_api) == sizeof(void*));
  void* address = nullptr;
  std::memcpy(&address, &query_api, sizeof(address));
  Dl_info info = {};
  if (address == nullptr || dladdr(address, &info) == 0 ||
      info.dli_fname == nullptr || info.dli_fname[0] == 0) {
    return ::testing::AssertionFailure()
           << "cannot locate the runtime-loaded libamdf provider";
  }
  *out_path = info.dli_fname;
  return ::testing::AssertionSuccess();
}

uint32_t AdvanceRecurrence(uint32_t state, uint32_t iteration_count) {
  uint32_t accumulated_multiplier = 1;
  uint32_t accumulated_increment = 0;
  uint32_t multiplier = kernels::cross_process_progress::kMultiplier;
  uint32_t increment = kernels::cross_process_progress::kIncrement;
  while (iteration_count != 0) {
    if ((iteration_count & 1u) != 0) {
      accumulated_multiplier *= multiplier;
      accumulated_increment = accumulated_increment * multiplier + increment;
    }
    increment *= multiplier + 1u;
    multiplier *= multiplier;
    iteration_count >>= 1;
  }
  return accumulated_multiplier * state + accumulated_increment;
}

template <typename T>
void StoreBytes(std::span<uint8_t> bytes, size_t offset, const T& value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void ExpectBytesEqual(std::span<const uint8_t> observed,
                      std::span<const uint8_t> expected,
                      std::string_view description) {
  ASSERT_EQ(observed.size(), expected.size());
  size_t mismatch_count = 0;
  size_t first_mismatch = 0;
  for (size_t i = 0; i < observed.size(); ++i) {
    if (observed[i] == expected[i]) {
      continue;
    }
    if (mismatch_count == 0) {
      first_mismatch = i;
    }
    ++mismatch_count;
  }
  if (mismatch_count != 0) {
    ADD_FAILURE() << description << " differs in " << mismatch_count
                  << " bytes, first at byte " << first_mismatch << ": observed "
                  << static_cast<uint32_t>(observed[first_mismatch])
                  << ", expected "
                  << static_cast<uint32_t>(expected[first_mismatch]);
  }
}

TEST(CrossProcessProgressChildTest, DISABLED_ActivateSelectedGpu) {
  Message configuration = {};
  ASSERT_TRUE(ReceiveMessage(kChildSocketDescriptor,
                             MessageKind::kConfiguration, &configuration));
  amdf_endpoint_id_t endpoint_id = {};
  endpoint_id.words[0] = configuration.values[0];
  endpoint_id.words[1] = configuration.values[1];

  CtsDeviceCache& cache = GetCtsDeviceCache();
  amdf_instance_t* instance = nullptr;
  amdf_endpoint_t* endpoint = nullptr;
  amdf_status_t status = cache.GetInstance(&instance);
  if (amdf_status_is_ok(status)) {
    status = cache.OpenEndpoint(endpoint_id, &endpoint);
  }
  ASSERT_TRUE(
      SendMessage(kChildSocketDescriptor, MessageKind::kPrepared, status));
  ASSERT_TRUE(amdf_status_is_ok(status))
      << "child preparation failed: domain=" << amdf_status_domain(status)
      << " code=" << amdf_status_code(status);
  ASSERT_NE(instance, nullptr);
  ASSERT_NE(endpoint, nullptr);

  Message activation = {};
  ASSERT_TRUE(ReceiveMessage(kChildSocketDescriptor, MessageKind::kActivate,
                             &activation));
  ASSERT_TRUE(
      SendMessage(kChildSocketDescriptor, MessageKind::kEnteringActivation));
  amdf_device_t* device = nullptr;
  status = cache.GetGpuDevice(endpoint, &device);
  ASSERT_TRUE(
      SendMessage(kChildSocketDescriptor, MessageKind::kActive, status));
  ASSERT_TRUE(amdf_status_is_ok(status))
      << "child GPU activation failed: domain=" << amdf_status_domain(status)
      << " code=" << amdf_status_code(status);
  ASSERT_NE(device, nullptr);

  Message release = {};
  ASSERT_TRUE(
      ReceiveMessage(kChildSocketDescriptor, MessageKind::kRelease, &release));
}

template <typename DispatchTest>
class CrossProcessProgressTest : public DispatchTest {
 protected:
  void RunWorkload();
};

template <typename DispatchTest>
void CrossProcessProgressTest<DispatchTest>::RunWorkload() {
  namespace progress = kernels::cross_process_progress;
  constexpr bool kAqlCarrier = std::is_same_v<DispatchTest, AqlDispatchTest>;
  static_assert(kAqlCarrier || std::is_same_v<DispatchTest, Pm4DispatchTest>);

  constexpr uint32_t kPageByteLength = 4096;
  constexpr uint32_t kPayloadByteLength = 1024 * 1024;
  constexpr uint32_t kAllocationByteLength =
      kPayloadByteLength + 2 * kPageByteLength;
  constexpr uint32_t kIterationLimit = 300'000'000;
  constexpr uint32_t kSeed = 0x31415927u;
  constexpr uint32_t kCopyGridSize = 4096;
  constexpr uint32_t kArgumentSlotByteLength = 64;
  constexpr uint32_t kComputeCompletionByteOffset = 64;
  constexpr uint32_t kCopyCompletionByteOffset = 128;
  constexpr uint8_t kSourceGuard = 0x63;
  constexpr uint8_t kReadbackGuard = 0xa9;
  constexpr uint8_t kControlGuard = 0xd7;
  constexpr uint8_t kArgumentGuard = 0x4d;
  constexpr aql::FenceScopes kSystemScopes = {
      aql::FenceScope::kSystem,
      aql::FenceScope::kSystem,
  };

  const auto* progress_kernel =
      progress::kKernels.Find(this->gpu_endpoint_info_);
  ASSERT_NE(progress_kernel, nullptr)
      << "missing compiled cross-process progress kernel for endpoint";
  ASSERT_EQ(progress_kernel->workgroup_size(), 1u);
  ASSERT_LE(progress_kernel->arguments.byte_length, kArgumentSlotByteLength);
  const kernels::Kernel* copy_kernel = nullptr;
  if constexpr (kAqlCarrier) {
    copy_kernel =
        kernels::byte_copy_unaligned::kKernels.Find(this->gpu_endpoint_info_);
    ASSERT_NE(copy_kernel, nullptr)
        << "missing compiled byte-copy kernel for endpoint";
    ASSERT_EQ(copy_kernel->workgroup_size(), 64u);
    ASSERT_LE(copy_kernel->arguments.byte_length, kArgumentSlotByteLength);
  }

  GpuMemory* control = nullptr;
  GpuMemory* source = nullptr;
  GpuMemory* readback = nullptr;
  GpuMemory* compute_arguments = nullptr;
  GpuMemory* copy_arguments = nullptr;
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  ASSERT_NO_FATAL_FAILURE(
      this->CreateMemory(kReadWrite, kPageByteLength, &control));
  ASSERT_NO_FATAL_FAILURE(
      this->CreateMemory(kReadWrite, kAllocationByteLength, &source));
  ASSERT_NO_FATAL_FAILURE(
      this->CreateMemory(kReadWrite, kAllocationByteLength, &readback));
  ASSERT_NO_FATAL_FAILURE(this->CreateMemory(
      AMDF_MEMORY_ACCESS_READ, kPageByteLength, &compute_arguments));
  if constexpr (kAqlCarrier) {
    ASSERT_NO_FATAL_FAILURE(this->CreateMemory(
        AMDF_MEMORY_ACCESS_READ, kPageByteLength, &copy_arguments));
  }

  ASSERT_EQ(control->device_address % alignof(progress::Control), 0u);
  ASSERT_EQ(source->device_address % 16, 0u);
  ASSERT_EQ(readback->device_address % 16, 0u);
  ASSERT_EQ(
      compute_arguments->device_address % progress_kernel->arguments.alignment,
      0u);
  if constexpr (kAqlCarrier) {
    ASSERT_EQ(copy_arguments->device_address % copy_kernel->arguments.alignment,
              0u);
  }

  std::vector<uint8_t> expected_source(kAllocationByteLength, kSourceGuard);
  std::vector<uint8_t> expected_readback(kAllocationByteLength, kReadbackGuard);
  for (uint32_t byte = 0; byte < kPayloadByteLength; ++byte) {
    expected_source[kPageByteLength + byte] =
        static_cast<uint8_t>(uint64_t{37} * byte + 17 * (byte / 256) + 0x2b);
  }
  std::memcpy(source->host.pointer, expected_source.data(),
              expected_source.size());
  std::memcpy(readback->host.pointer, expected_readback.data(),
              expected_readback.size());

  std::vector<uint8_t> expected_control(kPageByteLength, kControlGuard);
  progress::Control initial_protocol;
  std::memset(&initial_protocol, kControlGuard, sizeof(initial_protocol));
  initial_protocol.started = 0;
  initial_protocol.release = 0;
  initial_protocol.iteration_count = 0;
  initial_protocol.result = 0;
  StoreBytes(std::span<uint8_t>(expected_control), 0, initial_protocol);
  std::memcpy(control->host.pointer, expected_control.data(),
              expected_control.size());

  aql::Signal* compute_signal = nullptr;
  aql::Signal* copy_signal = nullptr;
  uint32_t* compute_pm4_completion = nullptr;
  uint32_t* copy_pm4_completion = nullptr;
  auto* control_bytes = static_cast<uint8_t*>(control->host.pointer);
  if constexpr (kAqlCarrier) {
    const aql::Signal initial_signal = aql::UserSignal(1);
    const aql::Signal expected_signal = aql::UserSignal(0);
    compute_signal = reinterpret_cast<aql::Signal*>(
        control_bytes + kComputeCompletionByteOffset);
    copy_signal = reinterpret_cast<aql::Signal*>(control_bytes +
                                                 kCopyCompletionByteOffset);
    *compute_signal = initial_signal;
    *copy_signal = initial_signal;
    StoreBytes(std::span<uint8_t>(expected_control),
               kComputeCompletionByteOffset, expected_signal);
    StoreBytes(std::span<uint8_t>(expected_control), kCopyCompletionByteOffset,
               expected_signal);
  } else {
    compute_pm4_completion = reinterpret_cast<uint32_t*>(
        control_bytes + kComputeCompletionByteOffset);
    copy_pm4_completion =
        reinterpret_cast<uint32_t*>(control_bytes + kCopyCompletionByteOffset);
    *compute_pm4_completion = 0;
    *copy_pm4_completion = 0;
    const uint32_t completed = 1;
    StoreBytes(std::span<uint8_t>(expected_control),
               kComputeCompletionByteOffset, completed);
    StoreBytes(std::span<uint8_t>(expected_control), kCopyCompletionByteOffset,
               completed);
  }

  const progress::Arguments compute_payload = {
      .control = control->device_address,
      .payload = source->device_address + kPageByteLength,
      .iteration_limit = kIterationLimit,
      .seed = kSeed,
  };
  std::vector<uint8_t> expected_compute_arguments(kPageByteLength,
                                                  kArgumentGuard);
  std::memset(expected_compute_arguments.data(), 0, kArgumentSlotByteLength);
  std::memcpy(expected_compute_arguments.data(), &compute_payload,
              sizeof(compute_payload));
  std::memcpy(compute_arguments->host.pointer,
              expected_compute_arguments.data(),
              expected_compute_arguments.size());

  std::vector<uint8_t> expected_copy_arguments;
  if constexpr (kAqlCarrier) {
    const kernels::byte_copy_unaligned::Arguments copy_payload = {
        .source = source->device_address + kPageByteLength,
        .target = readback->device_address + kPageByteLength,
        .byte_length = kPayloadByteLength,
        .grid_size_x = kCopyGridSize,
        .grid_size_y = 1,
        .workgroup_size_x = copy_kernel->workgroup_size(),
    };
    expected_copy_arguments.assign(kPageByteLength, kArgumentGuard);
    std::memset(expected_copy_arguments.data(), 0, kArgumentSlotByteLength);
    constexpr uint32_t kCopyArgumentSemanticByteLength =
        offsetof(kernels::byte_copy_unaligned::Arguments, workgroup_size_x) +
        sizeof(uint32_t);
    std::memcpy(expected_copy_arguments.data(), &copy_payload,
                kCopyArgumentSemanticByteLength);
    std::memcpy(copy_arguments->host.pointer, expected_copy_arguments.data(),
                expected_copy_arguments.size());
  }

  GpuUserQueue* compute_aql_queue = nullptr;
  GpuUserQueue* copy_aql_queue = nullptr;
  GpuCommandQueue* compute_pm4_queue = nullptr;
  GpuCommandQueue* copy_pm4_queue = nullptr;
  uint64_t compute_packet_index = 0;
  uint64_t copy_packet_index = 0;
  uint64_t compute_descriptor = 0;
  uint64_t copy_descriptor = 0;
  Pm4ComputeProgram compute_program = {};
  if constexpr (kAqlCarrier) {
    ASSERT_NO_FATAL_FAILURE(this->CreateQueue(&compute_aql_queue));
    ASSERT_NO_FATAL_FAILURE(this->CreateQueue(&copy_aql_queue));
    ASSERT_NO_FATAL_FAILURE(this->PublishKernel(
        *compute_aql_queue, *progress_kernel, "cross_process_progress",
        &compute_packet_index, &compute_descriptor));
    ASSERT_NO_FATAL_FAILURE(this->PublishKernel(
        *copy_aql_queue, *copy_kernel, "cross_process_byte_copy",
        &copy_packet_index, &copy_descriptor));
  } else {
    ASSERT_NO_FATAL_FAILURE(this->CreateQueue(&compute_pm4_queue));
    ASSERT_NO_FATAL_FAILURE(this->CreateQueue(&copy_pm4_queue));
    compute_program = {
        .entry_address = 0,
        .resource1 = progress_kernel->program.resource1,
        .resource2 = progress_kernel->program.resource2,
        .resource3 = progress_kernel->program.resource3,
        .group_segment_byte_length = progress_kernel->group_segment_byte_length,
        .wavefront_size = progress_kernel->wavefront_size,
        .workgroup_size = {1, 1, 1},
    };
    ASSERT_NO_FATAL_FAILURE(this->PrepareProgram(
        progress_kernel->executable, progress_kernel->entry_byte_offset,
        &compute_program, "cross_process_progress"));
  }

  amdf_endpoint_info_t endpoint_info = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(endpoint_info),
  };
  ASSERT_EQ(this->api_->endpoint_query_info(this->endpoint_, &endpoint_info),
            AMDF_STATUS_OK);
  std::string provider_path;
  ASSERT_TRUE(FindLoadedProviderPath(&provider_path));
  ChildProcess child;
  ASSERT_TRUE(child.Start(provider_path));
  ASSERT_TRUE(child.Send(MessageKind::kConfiguration, endpoint_info.id.words[0],
                         endpoint_info.id.words[1]));
  Message child_message = {};
  ASSERT_TRUE(child.Receive(MessageKind::kPrepared, &child_message));
  const amdf_status_t preparation_status = child_message.values[0];
  ASSERT_TRUE(amdf_status_is_ok(preparation_status))
      << "child preparation failed: domain="
      << amdf_status_domain(preparation_status)
      << " code=" << amdf_status_code(preparation_status);

  if constexpr (kAqlCarrier) {
    const auto dispatch = aql::Dispatch(
        aql::HeaderBarrier::kDisabled, {1, {1, 1, 1}, {1, 1, 1}},
        progress_kernel->private_segment_byte_length,
        progress_kernel->group_segment_byte_length, compute_descriptor,
        compute_arguments->device_address,
        control->device_address + kComputeCompletionByteOffset, kSystemScopes);
    GpuStoreRelease(compute_aql_queue->host.write_index_address,
                    compute_packet_index + 1);
    ASSERT_TRUE(
        aql::Publish(*compute_aql_queue, compute_packet_index++, dispatch));
  } else {
    ASSERT_GE(compute_pm4_queue->words().size(), 128u);
    Pm4CommandWriter commands(compute_pm4_queue->words().data(),
                              *this->pm4_profile_);
    commands.SystemBarrier();
    commands.BindCompute(compute_program, compute_arguments->device_address,
                         this->family_.format_features);
    commands.Dispatch(compute_program, 1, 1, 1);
    // This is the production-shaped post-dispatch compute join implicated by
    // #3043, followed by an independent SYSTEM completion publication.
    commands.SystemBarrier();
    commands.ReleaseSystem32(
        control->device_address + kComputeCompletionByteOffset, 1);
    commands.PadToEightWords();
    ASSERT_NO_FATAL_FAILURE(compute_pm4_queue->Publish(
        this->api_, this->gpu_api_, commands.word_count()));
    this->RecordProperty("cross_process_compute_pm4_command_bytes",
                         commands.word_count() * sizeof(uint32_t));
  }

  auto* protocol = static_cast<progress::Control*>(control->host.pointer);
  ASSERT_TRUE(GpuWaitEqual<uint32_t>(
      reinterpret_cast<uintptr_t>(&protocol->started), 1));
  ASSERT_TRUE(child.Send(MessageKind::kActivate));
  ASSERT_TRUE(child.Receive(MessageKind::kEnteringActivation, &child_message));
  ASSERT_TRUE(child.Receive(MessageKind::kActive, &child_message));
  const amdf_status_t activation_status = child_message.values[0];
  ASSERT_TRUE(amdf_status_is_ok(activation_status))
      << "child activation failed: domain="
      << amdf_status_domain(activation_status)
      << " code=" << amdf_status_code(activation_status);
  GpuStoreRelease(reinterpret_cast<uintptr_t>(&protocol->release), 1u);

  if constexpr (kAqlCarrier) {
    ASSERT_TRUE(GpuWaitEqual<int64_t>(
        reinterpret_cast<uintptr_t>(&compute_signal->value), 0));
  } else {
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(compute_pm4_completion), 1));
  }
  progress::Control observed_protocol;
  std::memcpy(&observed_protocol, protocol, sizeof(observed_protocol));
  ASSERT_GT(observed_protocol.iteration_count, 0u);
  ASSERT_LE(observed_protocol.iteration_count, kIterationLimit);
  const uint32_t expected_result =
      AdvanceRecurrence(kSeed, observed_protocol.iteration_count);
  EXPECT_EQ(observed_protocol.result, expected_result);
  if constexpr (kAqlCarrier) {
    ASSERT_NO_FATAL_FAILURE(
        compute_aql_queue->WaitConsumed(this->api_, compute_packet_index));
  } else {
    ASSERT_NO_FATAL_FAILURE(compute_pm4_queue->WaitRetired(this->api_));
  }

  if constexpr (kAqlCarrier) {
    const auto dispatch = aql::Dispatch(
        aql::HeaderBarrier::kDisabled,
        {1,
         {static_cast<uint16_t>(copy_kernel->workgroup_size()), 1, 1},
         {kCopyGridSize, 1, 1}},
        copy_kernel->private_segment_byte_length,
        copy_kernel->group_segment_byte_length, copy_descriptor,
        copy_arguments->device_address,
        control->device_address + kCopyCompletionByteOffset, kSystemScopes);
    GpuStoreRelease(copy_aql_queue->host.write_index_address,
                    copy_packet_index + 1);
    ASSERT_TRUE(aql::Publish(*copy_aql_queue, copy_packet_index++, dispatch));
    ASSERT_TRUE(GpuWaitEqual<int64_t>(
        reinterpret_cast<uintptr_t>(&copy_signal->value), 0));
  } else {
    ASSERT_GE(copy_pm4_queue->words().size(), 64u);
    Pm4CommandWriter commands(copy_pm4_queue->words().data(),
                              *this->pm4_profile_);
    commands.SystemBarrier();
    commands.DmaCopy(source->device_address + kPageByteLength,
                     readback->device_address + kPageByteLength,
                     kPayloadByteLength);
    commands.WaitDma();
    commands.ReleaseSystem32(
        control->device_address + kCopyCompletionByteOffset, 1);
    commands.PadToEightWords();
    ASSERT_NO_FATAL_FAILURE(copy_pm4_queue->Publish(this->api_, this->gpu_api_,
                                                    commands.word_count()));
    this->RecordProperty("cross_process_copy_pm4_command_bytes",
                         commands.word_count() * sizeof(uint32_t));
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(copy_pm4_completion), 1));
  }

  std::vector<uint8_t> observed_source(kAllocationByteLength);
  std::vector<uint8_t> observed_readback(kAllocationByteLength);
  std::vector<uint8_t> observed_control(kPageByteLength);
  std::vector<uint8_t> observed_compute_arguments(kPageByteLength);
  std::vector<uint8_t> observed_copy_arguments;
  std::memcpy(observed_source.data(), source->host.pointer,
              observed_source.size());
  std::memcpy(observed_readback.data(), readback->host.pointer,
              observed_readback.size());
  std::memcpy(observed_control.data(), control->host.pointer,
              observed_control.size());
  std::memcpy(observed_compute_arguments.data(),
              compute_arguments->host.pointer,
              observed_compute_arguments.size());
  if constexpr (kAqlCarrier) {
    observed_copy_arguments.resize(kPageByteLength);
    std::memcpy(observed_copy_arguments.data(), copy_arguments->host.pointer,
                observed_copy_arguments.size());
    ASSERT_NO_FATAL_FAILURE(
        copy_aql_queue->WaitConsumed(this->api_, copy_packet_index));
  } else {
    ASSERT_NO_FATAL_FAILURE(copy_pm4_queue->WaitRetired(this->api_));
  }

  ASSERT_TRUE(child.Send(MessageKind::kRelease));
  ASSERT_TRUE(child.Wait());

  progress::Control expected_protocol = initial_protocol;
  expected_protocol.started = 1;
  expected_protocol.release = 1;
  expected_protocol.iteration_count = observed_protocol.iteration_count;
  expected_protocol.result = expected_result;
  StoreBytes(std::span<uint8_t>(expected_control), 0, expected_protocol);
  StoreBytes(std::span<uint8_t>(expected_source), kPageByteLength,
             expected_result);
  std::memcpy(expected_readback.data() + kPageByteLength,
              expected_source.data() + kPageByteLength, kPayloadByteLength);

  ExpectBytesEqual(observed_source, expected_source, "source allocation");
  ExpectBytesEqual(observed_readback, expected_readback, "readback allocation");
  if constexpr (kAqlCarrier) {
    ExpectBytesEqual(std::span<const uint8_t>(observed_control)
                         .first(sizeof(progress::Control)),
                     std::span<const uint8_t>(expected_control)
                         .first(sizeof(progress::Control)),
                     "control protocol");
    aql::Signal observed_compute_signal;
    aql::Signal observed_copy_signal;
    std::memcpy(&observed_compute_signal,
                observed_control.data() + kComputeCompletionByteOffset,
                sizeof(observed_compute_signal));
    std::memcpy(&observed_copy_signal,
                observed_control.data() + kCopyCompletionByteOffset,
                sizeof(observed_copy_signal));
    EXPECT_EQ(observed_compute_signal.kind, 1);
    EXPECT_EQ(observed_compute_signal.value, 0);
    EXPECT_EQ(observed_copy_signal.kind, 1);
    EXPECT_EQ(observed_copy_signal.value, 0);
    constexpr size_t kControlGuardByteOffset =
        kCopyCompletionByteOffset + sizeof(aql::Signal);
    ExpectBytesEqual(std::span<const uint8_t>(observed_control)
                         .subspan(kControlGuardByteOffset),
                     std::span<const uint8_t>(expected_control)
                         .subspan(kControlGuardByteOffset),
                     "control guard");
  } else {
    ExpectBytesEqual(observed_control, expected_control, "control allocation");
  }
  ExpectBytesEqual(observed_compute_arguments, expected_compute_arguments,
                   "compute arguments");
  if constexpr (kAqlCarrier) {
    ExpectBytesEqual(observed_copy_arguments, expected_copy_arguments,
                     "copy arguments");
  }

  this->RecordProperty("cross_process_carrier", kAqlCarrier ? "AQL" : "PM4");
  this->RecordProperty("cross_process_iteration_limit", kIterationLimit);
  this->RecordProperty("cross_process_iteration_count",
                       observed_protocol.iteration_count);
  this->RecordProperty("cross_process_reached_iteration_limit",
                       observed_protocol.iteration_count == kIterationLimit);
  this->RecordProperty("cross_process_result",
                       std::to_string(observed_protocol.result));
  this->RecordProperty("cross_process_payload_byte_length", kPayloadByteLength);
  this->RecordProperty("cross_process_native_lifetime",
                       GetCtsDeviceCache().native_lifetime());
  this->RecordProperty("cross_process_queue_family", this->family_.ordinal);
  this->RecordProperty("cross_process_queue_command_type",
                       this->family_.command_type);
  this->RecordProperty("cross_process_queue_format_features",
                       std::to_string(this->family_.format_features));
  this->RecordProperty("cross_process_control_memory_profile",
                       control->info.memory_profile_ordinal);
  this->RecordProperty("cross_process_control_memory_class",
                       control->info.memory_class);
  this->RecordProperty("cross_process_source_memory_profile",
                       source->info.memory_profile_ordinal);
  this->RecordProperty("cross_process_source_memory_class",
                       source->info.memory_class);
  this->RecordProperty("cross_process_readback_memory_profile",
                       readback->info.memory_profile_ordinal);
  this->RecordProperty("cross_process_readback_memory_class",
                       readback->info.memory_class);
  this->RecordProperty("cross_process_argument_memory_profile",
                       compute_arguments->info.memory_profile_ordinal);
  this->RecordProperty("cross_process_argument_memory_class",
                       compute_arguments->info.memory_class);
  this->RecordProperty("cross_process_child_cleanup", 1);
  this->RecordProperty("cross_process_native_retirement", 1);
}

using CrossProcessProgressAqlTest = CrossProcessProgressTest<AqlDispatchTest>;
using CrossProcessProgressPm4Test = CrossProcessProgressTest<Pm4DispatchTest>;

TEST_F(CrossProcessProgressAqlTest,
       ActivationDuringComputePreservesBulkReadback) {
  ASSERT_NO_FATAL_FAILURE(RunWorkload());
}

TEST_F(CrossProcessProgressPm4Test,
       ActivationDuringComputePreservesBulkReadback) {
  ASSERT_NO_FATAL_FAILURE(RunWorkload());
}

}  // namespace
