// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#include "experimental/xdna/cts/execution_fixture.h"
#include "iree/async/operations/net.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/util/local_stream.h"
#include "iree/base/internal/shm.h"
#include "iree/testing/coordinated_test.h"
#include "iree/testing/gtest.h"
#include "libamdf/cts/util/device_cache.h"
#include "libamdf/cts/util/provider.h"

namespace iree::experimental::xdna::testing {
namespace {

// Roles execute outside RUN_ALL_TESTS. Their exit codes, including failures in
// the shared execution fixture, are checked by the coordinating parent.
void Check(bool condition, const char* expression, int line) {
  if (!condition) {
    std::fprintf(stderr, "shared mapping role failed at line %d: %s\n", line,
                 expression);
    std::abort();
  }
}
#define ROLE_CHECK(expression) Check((expression), #expression, __LINE__)

void CheckStatus(iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    std::abort();
  }
}

struct Transfer {
  // Terminal callback witness; the outer test harness bounds hangs.
  bool done = false;

  iree_async_local_stream_callback_t callback() {
    return {+[](void* user_data, iree_status_t status) {
              CheckStatus(status);
              auto* transfer = static_cast<Transfer*>(user_data);
              ROLE_CHECK(!transfer->done);
              transfer->done = true;
            },
            this};
  }

  static void Complete(void* user_data, iree_async_operation_t*,
                       iree_status_t status,
                       iree_async_completion_flags_t flags) {
    ROLE_CHECK(!(flags & IREE_ASYNC_COMPLETION_FLAG_MORE));
    auto callback = static_cast<Transfer*>(user_data)->callback();
    callback.fn(callback.user_data, status);
  }

  void Wait(iree_async_proactor_t* proactor) {
    while (!done) {
      CheckStatus(
          iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
    }
  }
};

enum class Role { kProducer, kImporter };

// Cold setup/control path only. Socket and pipe setup use the async APIs;
// local_stream transfers backing handles, never process or device addresses.
class Connection {
 public:
  Connection(Role role, const char* directory) {
    CheckStatus(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
#if defined(IREE_PLATFORM_WINDOWS)
    std::string path = directory;
    std::string name = path.substr(path.find_last_of("\\/") + 1) + "-xdna";
    auto name_view = iree_make_string_view(name.data(), name.size());
    if (role == Role::kProducer) {
      CheckStatus(iree_async_local_stream_pipe_create(
          name_view, 1, IREE_ASYNC_LOCAL_STREAM_PIPE_FLAG_FIRST_INSTANCE,
          &channel_));
      iree_coordinated_test_signal_ready(directory);
    } else {
      CheckStatus(iree_async_local_stream_pipe_open(name_view, &channel_));
    }
    CheckStatus(iree_async_local_stream_create(
        proactor_, channel_, 1, iree_allocator_system(), &stream_));
    if (role == Role::kProducer) {
      Transfer accepted;
      CheckStatus(
          iree_async_local_stream_pipe_accept(stream_, accepted.callback()));
      accepted.Wait(proactor_);
    }
#else
    std::string path = std::string(directory) + "/xdna.sock";
    iree_async_address_t address = {};
    CheckStatus(iree_async_address_from_unix(
        iree_make_string_view(path.data(), path.size()), &address));
    CheckStatus(
        iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM,
                                 IREE_ASYNC_SOCKET_OPTION_NONE, &socket_));
    Transfer connected;
    if (role == Role::kProducer) {
      CheckStatus(iree_async_socket_bind(socket_, &address));
      CheckStatus(iree_async_socket_listen(socket_, 1));
      iree_coordinated_test_signal_ready(directory);
      iree_async_socket_accept_operation_t accept = {};
      iree_async_operation_initialize(
          &accept.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT,
          IREE_ASYNC_OPERATION_FLAG_NONE, Transfer::Complete, &connected);
      accept.listen_socket = socket_;
      CheckStatus(iree_async_proactor_submit_one(proactor_, &accept.base));
      connected.Wait(proactor_);
      iree_async_socket_release(socket_);
      socket_ = accept.accepted_socket;
    } else {
      iree_async_socket_connect_operation_t connect = {};
      iree_async_operation_initialize(
          &connect.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT,
          IREE_ASYNC_OPERATION_FLAG_NONE, Transfer::Complete, &connected);
      connect.socket = socket_;
      connect.address = address;
      CheckStatus(iree_async_proactor_submit_one(proactor_, &connect.base));
      connected.Wait(proactor_);
    }
    CheckStatus(iree_async_local_stream_create(
        proactor_, socket_->primitive, 1, iree_allocator_system(), &stream_));
#endif  // IREE_PLATFORM_WINDOWS
  }

  void SendAndWait(iree_const_byte_span_t data,
                   const iree_async_primitive_t* resource = nullptr) {
    Transfer sent;
    CheckStatus(iree_async_local_stream_send(stream_, data, resource ? 1 : 0,
                                             resource, sent.callback()));
    sent.Wait(proactor_);
  }

  void ReceiveAndWait(iree_byte_span_t data,
                      iree_async_primitive_t* out_resource = nullptr) {
    Transfer received;
    CheckStatus(
        iree_async_local_stream_receive(stream_, data, out_resource ? 1 : 0,
                                        out_resource, received.callback()));
    received.Wait(proactor_);
  }

  ~Connection() {
    Transfer deactivated;
    CheckStatus(iree_async_local_stream_deactivate(
        stream_, {+[](void* user_data) {
                    static_cast<Transfer*>(user_data)->done = true;
                  },
                  &deactivated}));
    deactivated.Wait(proactor_);
    iree_async_local_stream_destroy(stream_);
#if defined(IREE_PLATFORM_WINDOWS)
    iree_async_primitive_close(&channel_);
#else
    iree_async_socket_release(socket_);
#endif  // IREE_PLATFORM_WINDOWS
    iree_async_proactor_release(proactor_);
  }

 private:
  // Role-owned executor, released after all native stream borrows retire.
  iree_async_proactor_t* proactor_ = nullptr;
  // Exact-transfer helper borrowing the connection through deactivation.
  iree_async_local_stream_t* stream_ = nullptr;
#if defined(IREE_PLATFORM_WINDOWS)
  // Owned overlapped pipe, kept separate from managed IOCP sockets.
  iree_async_primitive_t channel_ = {};
#else
  // Owner of the connected descriptor borrowed by the transfer helper.
  iree_async_socket_t* socket_ = nullptr;
#endif  // IREE_PLATFORM_WINDOWS
};

// The resource helper transfers handles, not the sender's integer values.
iree_async_primitive_t BorrowSharedHandle(iree_shm_handle_t handle) {
#if defined(IREE_PLATFORM_WINDOWS)
  return iree_async_primitive_from_win32_handle(handle.value);
#else
  return iree_async_primitive_from_fd(static_cast<int>(handle.value));
#endif  // IREE_PLATFORM_WINDOWS
}

iree_shm_handle_t BorrowSharedHandle(iree_async_primitive_t primitive) {
#if defined(IREE_PLATFORM_WINDOWS)
  ROLE_CHECK(primitive.type == IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE);
  return {primitive.value.win32_handle};
#else
  ROLE_CHECK(primitive.type == IREE_ASYNC_PRIMITIVE_TYPE_FD);
  return {static_cast<uintptr_t>(primitive.value.fd)};
#endif  // IREE_PLATFORM_WINDOWS
}

std::array<BindingValues, 3> ExpectedValues() {
  std::array<BindingValues, 3> expected;
  for (size_t i = 0; i < kElementCount; ++i) {
    expected[0][i] = kValues[i];
    expected[1][i] = kValues[(i * 3 + 5) % kElementCount];
    expected[2][i] = expected[0][i] * expected[1][i];
  }
  return expected;
}

int Producer(int, char**, const char* directory) {
  Connection connection(Role::kProducer, directory);
  uint64_t byte_length = 0;
  connection.ReceiveAndWait(
      iree_make_byte_span(&byte_length, sizeof(byte_length)));
  iree_shm_mapping_t mapping = {};
  CheckStatus(iree_shm_create(nullptr, byte_length, &mapping));
  std::memset(mapping.base, kGuardValue, mapping.size);
  auto values = ExpectedValues();
  for (auto& value : values[2]) {
    value = ~value;
  }
  for (size_t ordinal = 0; ordinal < values.size(); ++ordinal) {
    auto* data = static_cast<uint8_t*>(mapping.base) +
                 ordinal * kBindingStorageByteLength + kBindingByteOffset;
    for (size_t i = 0; i < kElementCount; ++i) {
      iree_unaligned_store_le_u32(data + i * sizeof(uint32_t),
                                  values[ordinal][i]);
    }
  }
  auto resource = BorrowSharedHandle(mapping.handle);
  uint64_t mapping_size = mapping.size;
  connection.SendAndWait(
      iree_make_const_byte_span(&mapping_size, sizeof(mapping_size)),
      &resource);
  // Send completion alone cannot release the Windows source handle. The
  // importer acknowledges its independent mapping before the source closes.
  uint8_t imported = 0;
  connection.ReceiveAndWait(iree_make_byte_span(&imported, sizeof(imported)));
  ROLE_CHECK(imported == 1);
  iree_shm_close(&mapping);
  const uint8_t released = 1;
  connection.SendAndWait(
      iree_make_const_byte_span(&released, sizeof(released)));
  return 0;
}

class ImportedExecution : public XdnaSharedMappingFixture {
 public:
  void Initialize(amdf_memory_profile_t* out_profile,
                  uint64_t* out_byte_length) {
    ASSERT_NO_FATAL_FAILURE(SetUp());
    ASSERT_NO_FATAL_FAILURE(
        QuerySharedRegistration(out_profile, out_byte_length));
  }

  void Execute(iree_shm_mapping_t mapping, const amdf_memory_profile_t& profile,
               uint64_t byte_length) {
    shared_ = mapping;
    ASSERT_NO_FATAL_FAILURE(RegisterSharedBindings(profile, byte_length));
    ASSERT_NO_FATAL_FAILURE(PrepareExecution(resolved_bindings_, &first_));
    // The producer initialized every input and poisoned output. Publish those
    // shared CPU writes without rewriting them in the consuming process.
    for (const auto& binding : bindings_) {
      ASSERT_EQ(api_->host_mapping_cache_control(binding.storage.mapping,
                                                 binding.host_cache.publish, 0,
                                                 kBindingStorageByteLength),
                AMDF_STATUS_OK);
    }
    ASSERT_NO_FATAL_FAILURE(RunExecution(first_));
    ASSERT_NO_FATAL_FAILURE(VerifyBindings(ExpectedValues()));
  }

  void Finish() { ASSERT_NO_FATAL_FAILURE(TearDown()); }

 private:
  void TestBody() override {}
};

int Importer(int argc, char** argv, const char* directory) {
  const char prefix[] = "--amdf_native_lifetime=";
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], prefix, sizeof(prefix) - 1) != 0) {
      continue;
    }
    const char* value = argv[i] + sizeof(prefix) - 1;
    ROLE_CHECK(std::strcmp(value, "process") == 0 ||
               std::strcmp(value, "instance") == 0);
    GetCtsDeviceCache().SetNativeLifetime(std::strcmp(value, "instance") == 0
                                              ? AMDF_NATIVE_LIFETIME_INSTANCE
                                              : AMDF_NATIVE_LIFETIME_PROCESS);
  }
  ROLE_CHECK(amdf_cts_provider_initialize(&argc, &argv));
  ImportedExecution execution;
  amdf_memory_profile_t profile = {};
  uint64_t byte_length = 0;
  execution.Initialize(&profile, &byte_length);
  ROLE_CHECK(!::testing::Test::HasFailure());
  iree_shm_mapping_t mapping = {};
  {
    Connection connection(Role::kImporter, directory);
    connection.SendAndWait(
        iree_make_const_byte_span(&byte_length, sizeof(byte_length)));
    uint64_t mapping_size = 0;
    iree_async_primitive_t resource = {};
    connection.ReceiveAndWait(
        iree_make_byte_span(&mapping_size, sizeof(mapping_size)), &resource);
    ROLE_CHECK(mapping_size >= byte_length);
    // open_handle creates independent ownership, leaving the received handle
    // ours to close before acknowledging import to the producer.
    CheckStatus(iree_shm_open_handle(BorrowSharedHandle(resource), mapping_size,
                                     &mapping));
    iree_async_primitive_close(&resource);
    const uint8_t imported = 1;
    connection.SendAndWait(
        iree_make_const_byte_span(&imported, sizeof(imported)));
    uint8_t released = 0;
    connection.ReceiveAndWait(iree_make_byte_span(&released, sizeof(released)));
    ROLE_CHECK(released == 1);
  }
  // The source mapping and handle are gone. Only the importer owns backing;
  // IPC lifetime is no longer involved in registration or device execution.
  execution.Execute(std::exchange(mapping, {}), profile, byte_length);
  ROLE_CHECK(!::testing::Test::HasFailure());
  execution.Finish();
  ROLE_CHECK(!::testing::Test::HasFailure());
  ROLE_CHECK(GetCtsDeviceCache().Deinitialize() == AMDF_STATUS_OK);
  ROLE_CHECK(amdf_cts_provider_deinitialize());
  return 0;
}

const iree_test_role_t kRoles[] = {
    {"producer", Producer, true},
    {"importer", Importer, false},
};
const iree_coordinated_test_config_t kConfig = {kRoles, IREE_ARRAYSIZE(kRoles)};
IREE_COORDINATED_TEST_REGISTER(kConfig);

TEST(XdnaSharedMappingProcessTest, ExecutesAfterProducerRelease) {
  ASSERT_EQ(iree_coordinated_test_run(iree_coordinated_test_argc(),
                                      iree_coordinated_test_argv(), &kConfig),
            0);
}

}  // namespace
}  // namespace iree::experimental::xdna::testing
