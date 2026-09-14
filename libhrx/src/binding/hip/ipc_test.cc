// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/ipc.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hip {
namespace {

static_assert(HIP_IPC_HANDLE_SIZE == 64);
static_assert(sizeof(hipIpcMemHandle_t) == HIP_IPC_HANDLE_SIZE);
static_assert(sizeof(hipIpcEventHandle_t) == HIP_IPC_HANDLE_SIZE);
static_assert(hipIpcMemLazyEnablePeerAccess == 1);

using IpcGetMemHandleFn = hipError_t (*)(hipIpcMemHandle_t*, void*);
using IpcOpenMemHandleFn = hipError_t (*)(void**, hipIpcMemHandle_t,
                                          unsigned int);
using IpcCloseMemHandleFn = hipError_t (*)(void*);
using IpcGetEventHandleFn = hipError_t (*)(hipIpcEventHandle_t*, hipEvent_t);
using IpcOpenEventHandleFn = hipError_t (*)(hipEvent_t*, hipIpcEventHandle_t);

static_assert(std::is_same_v<decltype(&hipIpcGetMemHandle), IpcGetMemHandleFn>);
static_assert(
    std::is_same_v<decltype(&hipIpcOpenMemHandle), IpcOpenMemHandleFn>);
static_assert(
    std::is_same_v<decltype(&hipIpcCloseMemHandle), IpcCloseMemHandleFn>);
static_assert(
    std::is_same_v<decltype(&hipIpcGetEventHandle), IpcGetEventHandleFn>);
static_assert(
    std::is_same_v<decltype(&hipIpcOpenEventHandle), IpcOpenEventHandleFn>);

template <typename T, typename Handle>
T ReadWireValue(const Handle& handle, size_t byte_offset) {
  T value = {};
  std::memcpy(&value, handle.reserved + byte_offset, sizeof(value));
  return value;
}

bool IsZeroed(const void* data, size_t data_length) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < data_length; ++i) {
    if (bytes[i] != 0) return false;
  }
  return true;
}

iree_hal_streaming_ipc_memory_descriptor_t MakeDescriptor() {
  iree_hal_streaming_ipc_memory_descriptor_t descriptor = {};
  for (size_t i = 0; i < sizeof(descriptor.token); ++i) {
    descriptor.token[i] = static_cast<uint8_t>(i * 7 + 3);
  }
  descriptor.allocation_size = UINT64_C(0x0102030405060708);
  descriptor.byte_offset = UINT64_C(0x0000000000001234);
  return descriptor;
}

TEST(IpcMemoryWireTest, MatchesRocmLayoutAndRoundTrips) {
  const iree_hal_streaming_ipc_memory_descriptor_t source = MakeDescriptor();
  constexpr int32_t kCreatorProcessId = 12345;
  constexpr iree_host_size_t kDeviceOrdinal = 3;

  hipIpcMemHandle_t handle;
  std::memset(&handle, 0xA5, sizeof(handle));
  IREE_ASSERT_OK(iree_hip_ipc_memory_handle_encode(&source, kDeviceOrdinal,
                                                   kCreatorProcessId, &handle));

  EXPECT_EQ(0,
            std::memcmp(handle.reserved, source.token, sizeof(source.token)));
  EXPECT_EQ(source.allocation_size, ReadWireValue<uint64_t>(handle, 32));
  EXPECT_EQ(source.byte_offset, ReadWireValue<uint64_t>(handle, 40));
  EXPECT_EQ(kCreatorProcessId, ReadWireValue<int32_t>(handle, 48));
  EXPECT_EQ(kDeviceOrdinal, ReadWireValue<int32_t>(handle, 52));
  EXPECT_TRUE(IsZeroed(handle.reserved + 56, 8));

  // Stock reserves the final eight bytes. Import ignores them.
  std::memset(handle.reserved + 56, 0x7B, 8);
  iree_hal_streaming_ipc_memory_descriptor_t decoded = {};
  iree_device_size_t view_size = 0;
  IREE_ASSERT_OK(iree_hip_ipc_memory_handle_decode(
      handle, /*current_process_id=*/54321,
      /*visible_device_count=*/4, &decoded, &view_size));
  EXPECT_EQ(0, std::memcmp(decoded.token, source.token, sizeof(source.token)));
  EXPECT_EQ(source.allocation_size, decoded.allocation_size);
  EXPECT_EQ(source.byte_offset, decoded.byte_offset);
  EXPECT_EQ(kCreatorProcessId, decoded.exporter_process_id);
  EXPECT_EQ(kDeviceOrdinal, decoded.exporter_device_ordinal);
  EXPECT_EQ(source.allocation_size - source.byte_offset, view_size);
}

TEST(IpcMemoryWireTest, EncodeFailureClearsOutput) {
  hipIpcMemHandle_t handle;
  std::memset(&handle, 0xA5, sizeof(handle));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hip_ipc_memory_handle_encode(
                            /*descriptor=*/nullptr, /*device_ordinal=*/0,
                            /*creator_process_id=*/123, &handle));
  EXPECT_TRUE(IsZeroed(&handle, sizeof(handle)));

  iree_hal_streaming_ipc_memory_descriptor_t descriptor = MakeDescriptor();
  descriptor.byte_offset = descriptor.allocation_size;
  std::memset(&handle, 0xA5, sizeof(handle));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hip_ipc_memory_handle_encode(&descriptor, /*device_ordinal=*/0,
                                        /*creator_process_id=*/123, &handle));
  EXPECT_TRUE(IsZeroed(&handle, sizeof(handle)));

  descriptor = MakeDescriptor();
  std::memset(&handle, 0xA5, sizeof(handle));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hip_ipc_memory_handle_encode(
          &descriptor,
          static_cast<iree_host_size_t>(std::numeric_limits<int32_t>::max()) +
              1,
          /*creator_process_id=*/123, &handle));
  EXPECT_TRUE(IsZeroed(&handle, sizeof(handle)));
}

TEST(IpcMemoryWireTest, MissingDecodeOutputStillClearsOtherOutput) {
  const iree_hal_streaming_ipc_memory_descriptor_t source = MakeDescriptor();
  hipIpcMemHandle_t handle = {};
  IREE_ASSERT_OK(iree_hip_ipc_memory_handle_encode(
      &source, /*device_ordinal=*/0, /*creator_process_id=*/123, &handle));

  iree_device_size_t view_size = 123;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hip_ipc_memory_handle_decode(
          handle, /*current_process_id=*/456, /*visible_device_count=*/1,
          /*out_descriptor=*/nullptr, &view_size));
  EXPECT_EQ(0, view_size);

  iree_hal_streaming_ipc_memory_descriptor_t descriptor;
  std::memset(&descriptor, 0xA5, sizeof(descriptor));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hip_ipc_memory_handle_decode(handle, /*current_process_id=*/456,
                                        /*visible_device_count=*/1, &descriptor,
                                        /*out_view_size=*/nullptr));
  EXPECT_TRUE(IsZeroed(&descriptor, sizeof(descriptor)));
}

void ExpectDecodeFailure(hipIpcMemHandle_t handle, int32_t process_id,
                         iree_host_size_t device_count,
                         iree_status_code_t expected_code) {
  iree_hal_streaming_ipc_memory_descriptor_t descriptor;
  std::memset(&descriptor, 0xA5, sizeof(descriptor));
  iree_device_size_t view_size = 123;
  iree_status_t status = iree_hip_ipc_memory_handle_decode(
      handle, process_id, device_count, &descriptor, &view_size);
  EXPECT_EQ(expected_code, iree_status_code(status));
  iree_status_free(status);
  EXPECT_TRUE(IsZeroed(&descriptor, sizeof(descriptor)));
  EXPECT_EQ(0, view_size);
}

TEST(IpcMemoryWireTest, RejectsMalformedHandlesAndClearsOutputs) {
  const iree_hal_streaming_ipc_memory_descriptor_t source = MakeDescriptor();
  hipIpcMemHandle_t handle = {};
  IREE_ASSERT_OK(iree_hip_ipc_memory_handle_encode(
      &source, /*device_ordinal=*/1, /*creator_process_id=*/123, &handle));

  hipIpcMemHandle_t malformed = handle;
  const uint64_t zero = 0;
  std::memcpy(malformed.reserved + 32, &zero, sizeof(zero));
  ExpectDecodeFailure(malformed, /*process_id=*/456, /*device_count=*/2,
                      IREE_STATUS_INVALID_ARGUMENT);

  malformed = handle;
  std::memcpy(malformed.reserved + 40, &source.allocation_size,
              sizeof(source.allocation_size));
  ExpectDecodeFailure(malformed, /*process_id=*/456, /*device_count=*/2,
                      IREE_STATUS_INVALID_ARGUMENT);

  malformed = handle;
  constexpr int32_t kInvalidDeviceOrdinal = -1;
  std::memcpy(malformed.reserved + 52, &kInvalidDeviceOrdinal,
              sizeof(kInvalidDeviceOrdinal));
  ExpectDecodeFailure(malformed, /*process_id=*/456, /*device_count=*/2,
                      IREE_STATUS_INVALID_ARGUMENT);

  malformed = handle;
  constexpr int32_t kUnavailableDeviceOrdinal = 2;
  std::memcpy(malformed.reserved + 52, &kUnavailableDeviceOrdinal,
              sizeof(kUnavailableDeviceOrdinal));
  ExpectDecodeFailure(malformed, /*process_id=*/456, /*device_count=*/2,
                      IREE_STATUS_INVALID_ARGUMENT);
}

TEST(IpcMemoryWireTest, SameProcessCheckPrecedesPlacementValidation) {
  const iree_hal_streaming_ipc_memory_descriptor_t source = MakeDescriptor();
  hipIpcMemHandle_t handle = {};
  constexpr int32_t kProcessId = 123;
  IREE_ASSERT_OK(iree_hip_ipc_memory_handle_encode(
      &source, /*device_ordinal=*/1, kProcessId, &handle));

  std::memcpy(handle.reserved + 40, &source.allocation_size,
              sizeof(source.allocation_size));
  ExpectDecodeFailure(handle, kProcessId, /*device_count=*/0,
                      IREE_STATUS_PERMISSION_DENIED);
}

TEST(IpcMemoryResultTest, MapsExportFailures) {
  EXPECT_EQ(hipSuccess,
            iree_hip_ipc_memory_export_status_to_result(iree_ok_status()));
  EXPECT_EQ(hipErrorNotSupported,
            iree_hip_ipc_memory_export_status_to_result(
                iree_status_from_code(IREE_STATUS_UNIMPLEMENTED)));
  EXPECT_EQ(hipErrorOutOfMemory,
            iree_hip_ipc_memory_export_status_to_result(
                iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED)));
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_ipc_memory_export_status_to_result(
                iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT)));
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_ipc_memory_export_status_to_result(
                iree_status_from_code(IREE_STATUS_ABORTED)));
}

void MakeEventToken(uint8_t token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE]) {
  for (size_t i = 0; i < IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE; ++i) {
    token[i] = static_cast<uint8_t>(i * 11 + 5);
  }
}

TEST(IpcEventWireTest, MatchesRocmTypeOneLayoutAndRoundTrips) {
  uint8_t source_token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE];
  MakeEventToken(source_token);
  constexpr int32_t kCreatorProcessId = 12345;

  hipIpcEventHandle_t handle;
  std::memset(&handle, 0xA5, sizeof(handle));
  IREE_ASSERT_OK(iree_hip_ipc_event_handle_encode(source_token,
                                                  kCreatorProcessId, &handle));

  EXPECT_EQ(1u, ReadWireValue<uint32_t>(handle, 0));
  EXPECT_EQ(kCreatorProcessId, ReadWireValue<int32_t>(handle, 4));
  EXPECT_EQ(
      0, std::memcmp(handle.reserved + 8, source_token, sizeof(source_token)));
  EXPECT_TRUE(IsZeroed(handle.reserved + 40, 24));

  // Stock reserves the final 24 bytes. Import ignores them.
  std::memset(handle.reserved + 40, 0x7B, 24);
  uint8_t decoded_token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE];
  std::memset(decoded_token, 0xA5, sizeof(decoded_token));
  IREE_ASSERT_OK(iree_hip_ipc_event_handle_decode(
      handle, /*current_process_id=*/54321, decoded_token));
  EXPECT_EQ(0, std::memcmp(decoded_token, source_token, sizeof(source_token)));
}

TEST(IpcEventWireTest, TreatsRocrSignalTokenAsOpaque) {
  uint8_t zero_token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE] = {};
  hipIpcEventHandle_t handle;
  std::memset(&handle, 0xA5, sizeof(handle));
  IREE_ASSERT_OK(iree_hip_ipc_event_handle_encode(
      zero_token, /*creator_process_id=*/123, &handle));
  EXPECT_TRUE(IsZeroed(handle.reserved + 8, sizeof(zero_token)));
  EXPECT_TRUE(IsZeroed(handle.reserved + 40, 24));

  uint8_t decoded_token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE];
  std::memset(decoded_token, 0xA5, sizeof(decoded_token));
  IREE_ASSERT_OK(iree_hip_ipc_event_handle_decode(
      handle, /*current_process_id=*/456, decoded_token));
  EXPECT_TRUE(IsZeroed(decoded_token, sizeof(decoded_token)));
}

TEST(IpcEventWireTest, EncodeFailureClearsOutput) {
  uint8_t token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE];
  MakeEventToken(token);
  hipIpcEventHandle_t handle;
  std::memset(&handle, 0xA5, sizeof(handle));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hip_ipc_event_handle_encode(
                            token, /*creator_process_id=*/0, &handle));
  EXPECT_TRUE(IsZeroed(&handle, sizeof(handle)));
}

void ExpectEventDecodeFailure(hipIpcEventHandle_t handle, int32_t process_id,
                              iree_status_code_t expected_code) {
  uint8_t token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE];
  std::memset(token, 0xA5, sizeof(token));
  iree_status_t status =
      iree_hip_ipc_event_handle_decode(handle, process_id, token);
  EXPECT_EQ(expected_code, iree_status_code(status));
  iree_status_free(status);
  EXPECT_TRUE(IsZeroed(token, sizeof(token)));
}

TEST(IpcEventWireTest, RejectsMalformedAndSameProcessHandles) {
  uint8_t source_token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE];
  MakeEventToken(source_token);
  constexpr int32_t kCreatorProcessId = 123;
  hipIpcEventHandle_t handle = {};
  IREE_ASSERT_OK(iree_hip_ipc_event_handle_encode(source_token,
                                                  kCreatorProcessId, &handle));

  hipIpcEventHandle_t malformed = handle;
  constexpr uint32_t kEmulatedType = 0;
  std::memcpy(malformed.reserved, &kEmulatedType, sizeof(kEmulatedType));
  ExpectEventDecodeFailure(malformed, /*process_id=*/456,
                           IREE_STATUS_INVALID_ARGUMENT);

  malformed = handle;
  constexpr uint32_t kUnknownType = 2;
  std::memcpy(malformed.reserved, &kUnknownType, sizeof(kUnknownType));
  ExpectEventDecodeFailure(malformed, /*process_id=*/456,
                           IREE_STATUS_INVALID_ARGUMENT);

  malformed = handle;
  constexpr int32_t kInvalidProcessId = 0;
  std::memcpy(malformed.reserved + 4, &kInvalidProcessId,
              sizeof(kInvalidProcessId));
  ExpectEventDecodeFailure(malformed, /*process_id=*/456,
                           IREE_STATUS_INVALID_ARGUMENT);

  ExpectEventDecodeFailure(handle, kCreatorProcessId,
                           IREE_STATUS_PERMISSION_DENIED);
}

TEST(IpcEventResultTest, MapsOperationFailures) {
  EXPECT_EQ(hipSuccess,
            iree_hip_ipc_event_operation_status_to_result(iree_ok_status()));
  EXPECT_EQ(hipErrorContextIsDestroyed,
            iree_hip_ipc_event_operation_status_to_result(
                iree_status_from_code(IREE_STATUS_ABORTED)));
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_ipc_event_operation_status_to_result(
                iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT)));
  EXPECT_EQ(hipErrorOutOfMemory,
            iree_hip_ipc_event_operation_status_to_result(
                iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED)));
  EXPECT_EQ(hipErrorNotFound,
            iree_hip_ipc_event_operation_status_to_result(
                iree_status_from_code(IREE_STATUS_NOT_FOUND)));
  EXPECT_EQ(hipErrorInvalidContext,
            iree_hip_ipc_event_operation_status_to_result(
                iree_status_from_code(IREE_STATUS_PERMISSION_DENIED)));
  EXPECT_EQ(hipErrorNotSupported,
            iree_hip_ipc_event_operation_status_to_result(
                iree_status_from_code(IREE_STATUS_UNIMPLEMENTED)));
  EXPECT_EQ(hipErrorNotReady,
            iree_hip_ipc_event_operation_status_to_result(
                iree_status_from_code(IREE_STATUS_UNAVAILABLE)));
  EXPECT_EQ(hipErrorNotInitialized,
            iree_hip_ipc_event_operation_status_to_result(
                iree_status_from_code(IREE_STATUS_FAILED_PRECONDITION)));
  EXPECT_EQ(hipErrorUnknown, iree_hip_ipc_event_operation_status_to_result(
                                 iree_status_from_code(IREE_STATUS_INTERNAL)));
}

TEST(IpcEventResultTest, MapsExportFailures) {
  EXPECT_EQ(hipSuccess,
            iree_hip_ipc_event_export_status_to_result(iree_ok_status()));
  EXPECT_EQ(hipErrorOutOfMemory,
            iree_hip_ipc_event_export_status_to_result(
                iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED)));
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_ipc_event_export_status_to_result(
                iree_status_from_code(IREE_STATUS_UNIMPLEMENTED)));
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_ipc_event_export_status_to_result(
                iree_status_from_code(IREE_STATUS_ABORTED)));
}

TEST(IpcEventResultTest, MapsOpenAndAdmissionFailures) {
  EXPECT_EQ(hipSuccess,
            iree_hip_ipc_event_open_status_to_result(iree_ok_status()));
  EXPECT_EQ(hipErrorContextIsDestroyed,
            iree_hip_ipc_event_open_status_to_result(
                iree_status_from_code(IREE_STATUS_ABORTED)));
  EXPECT_EQ(hipErrorInvalidContext,
            iree_hip_ipc_event_open_status_to_result(
                iree_status_from_code(IREE_STATUS_PERMISSION_DENIED)));
  EXPECT_EQ(hipErrorOutOfMemory,
            iree_hip_ipc_event_open_status_to_result(
                iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED)));
  EXPECT_EQ(hipErrorNotSupported,
            iree_hip_ipc_event_open_status_to_result(
                iree_status_from_code(IREE_STATUS_UNIMPLEMENTED)));
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_ipc_event_open_status_to_result(
                iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT)));
}

TEST(IpcEventCapabilityTest, FeatureOffBuildReportsUnsupported) {
  EXPECT_FALSE(iree_hip_ipc_event_supported());
}

}  // namespace
}  // namespace iree::hip
