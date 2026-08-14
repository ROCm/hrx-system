// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL remote protocol: control channel messages.
//
// Request/response pairs, fire-and-forget messages, and server-initiated
// notifications on the control channel. Each message is wrapped in a control
// envelope (iree_hal_remote_control_envelope_t). Responses additionally carry
// a response prefix (iree_hal_remote_control_response_prefix_t) between the
// envelope and the message-specific payload.
//
// Naming convention:
//   _request_t   Request payload of a request/response pair.
//   _response_t  Response payload (omitted when only status is returned).
//   _t           Fire-and-forget messages and notifications.
//
// ## Dependency policy
//
// Includes common.h for shared wire format types (resource IDs, buffer params,
// memory heaps). Does not include HAL headers.

#ifndef IREE_HAL_REMOTE_PROTOCOL_CONTROL_H_
#define IREE_HAL_REMOTE_PROTOCOL_CONTROL_H_

#include "iree/hal/remote/protocol/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Upload flags
//===----------------------------------------------------------------------===//

// Upload delivery flags shared by EXECUTABLE_UPLOAD and COMMAND_BUFFER_UPLOAD.
// Exactly one of INLINE_DATA or BULK_REFERENCE must be set.
#define IREE_HAL_REMOTE_UPLOAD_FLAG_INLINE_DATA (1u << 0)
#define IREE_HAL_REMOTE_UPLOAD_FLAG_BULK_REFERENCE (1u << 1)

//===----------------------------------------------------------------------===//
// Control message envelope
//===----------------------------------------------------------------------===//

// Control message envelope. Prepended to every control channel DATA frame
// payload. Requests and responses share the same message_type value; the
// IS_RESPONSE flag in message_flags distinguishes direction.
typedef struct iree_hal_remote_control_envelope_t {
  uint16_t message_type;   // iree_hal_remote_control_type_t
  uint16_t message_flags;  // Combination of IREE_HAL_REMOTE_CONTROL_FLAG_*.
  uint32_t request_id;     // Client-assigned, echoed in response. 0 = notif.
  uint32_t control_epoch;  // Monotonic, incremented by CREATE-type messages.
  uint32_t reserved;       // Must be 0.
} iree_hal_remote_control_envelope_t;
static_assert(sizeof(iree_hal_remote_control_envelope_t) == 16, "");
static_assert(offsetof(iree_hal_remote_control_envelope_t, message_type) == 0,
              "");
static_assert(offsetof(iree_hal_remote_control_envelope_t, message_flags) == 2,
              "");
static_assert(offsetof(iree_hal_remote_control_envelope_t, request_id) == 4,
              "");
static_assert(offsetof(iree_hal_remote_control_envelope_t, control_epoch) == 8,
              "");

// Control message flag bits.
#define IREE_HAL_REMOTE_CONTROL_FLAG_IS_RESPONSE (1u << 0)
#define IREE_HAL_REMOTE_CONTROL_FLAG_FIRE_AND_FORGET (1u << 1)

// Response prefix. Immediately follows the envelope when IS_RESPONSE is set.
// The response-specific payload (if any) follows this prefix.
typedef struct iree_hal_remote_control_response_prefix_t {
  uint32_t status_code;  // 0 = OK, else iree_status_code_t.
  uint32_t reserved;     // Must be 0.
} iree_hal_remote_control_response_prefix_t;
static_assert(sizeof(iree_hal_remote_control_response_prefix_t) == 8, "");

//===----------------------------------------------------------------------===//
// Control message types
//===----------------------------------------------------------------------===//

// Control message type identifiers. Types marked [epoch] increment the
// control_epoch counter. Queue ops depending on resources created by these
// messages include wait={control:epoch} in their frontier.
typedef enum iree_hal_remote_control_type_e {
  // ── Device ──────────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_DEVICE_TRIM = 0x0003,

  // ── Semaphore ───────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_SEMAPHORE_CREATE = 0x0010,  // [epoch]
  IREE_HAL_REMOTE_CONTROL_SEMAPHORE_QUERY = 0x0011,
  IREE_HAL_REMOTE_CONTROL_SEMAPHORE_SIGNAL = 0x0012,  // fire-and-forget
  IREE_HAL_REMOTE_CONTROL_SEMAPHORE_WAIT = 0x0013,

  // ── Executable ──────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_EXECUTABLE_UPLOAD = 0x0020,  // [epoch]
  IREE_HAL_REMOTE_CONTROL_EXECUTABLE_QUERY_FUNCTION = 0x0021,
  IREE_HAL_REMOTE_CONTROL_EXECUTABLE_QUERY_PARAMETERS = 0x0022,
  IREE_HAL_REMOTE_CONTROL_EXECUTABLE_LOOKUP_GLOBAL = 0x0025,
  IREE_HAL_REMOTE_CONTROL_EXECUTABLE_GLOBAL_BUFFER = 0x0026,

  // ── Command Buffer ──────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_COMMAND_BUFFER_UPLOAD = 0x0030,  // [epoch]

  // ── File ────────────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_FILE_OPEN = 0x0040,        // [epoch]
  IREE_HAL_REMOTE_CONTROL_FILE_CLOSE = 0x0041,       // fire-and-forget
  IREE_HAL_REMOTE_CONTROL_FILE_REGISTER = 0x0042,    // [epoch]
  IREE_HAL_REMOTE_CONTROL_FILE_UNREGISTER = 0x0043,  // fire-and-forget
  IREE_HAL_REMOTE_CONTROL_FILE_LIST = 0x0044,

  // ── Buffer ──────────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_BUFFER_ALLOC = 0x0050,   // [epoch]
  IREE_HAL_REMOTE_CONTROL_BUFFER_IMPORT = 0x0051,  // [epoch]
  IREE_HAL_REMOTE_CONTROL_BUFFER_MAP = 0x0052,
  IREE_HAL_REMOTE_CONTROL_BUFFER_UNMAP = 0x0053,
  IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_QUERY_CAPABILITIES = 0x0055,
  IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_QUERY_GRANULARITY = 0x0056,
  IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_RESERVE = 0x0057,  // [epoch]
  IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_RELEASE = 0x0058,  // [epoch]
  IREE_HAL_REMOTE_CONTROL_BUFFER_PHYSICAL_ALLOC = 0x0059,   // [epoch]
  IREE_HAL_REMOTE_CONTROL_BUFFER_PHYSICAL_FREE = 0x005A,    // [epoch]
  IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_MAP = 0x005B,
  IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_UNMAP = 0x005C,
  IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_PROTECT = 0x005D,
  IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_ADVISE = 0x005E,

  // ── Host Call ───────────────────────────────────────────────────────────
  // Reserved for future explicit server-side named handlers. HAL
  // queue_host_call callbacks are client-local function pointers and are
  // handled by the remote client without crossing the control channel.
  IREE_HAL_REMOTE_CONTROL_HOST_CALL_REGISTER = 0x0060,    // [epoch]
  IREE_HAL_REMOTE_CONTROL_HOST_CALL_UNREGISTER = 0x0061,  // fire-and-forget

  // ── Lifecycle ───────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_RESOURCE_RELEASE_BATCH = 0x0070,  // fire-and-forget

  // ── Profiling ───────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_PROFILING_BEGIN = 0x0080,
  IREE_HAL_REMOTE_CONTROL_PROFILING_FLUSH = 0x0081,
  IREE_HAL_REMOTE_CONTROL_PROFILING_END = 0x0082,

  // ── Extensions ──────────────────────────────────────────────────────────
  IREE_HAL_REMOTE_CONTROL_DEVICE_EXTENSION = 0x00F0,

  // ── Notifications (server → client, request_id=0) ──────────────────────
  IREE_HAL_REMOTE_CONTROL_NOTIFY_RESOURCE_ERROR = 0x00E0,
  IREE_HAL_REMOTE_CONTROL_NOTIFY_DEVICE_LOST = 0x00E1,
  IREE_HAL_REMOTE_CONTROL_NOTIFY_MEMORY_PRESSURE = 0x00E2,
} iree_hal_remote_control_type_t;

//===----------------------------------------------------------------------===//
// Device messages
//===----------------------------------------------------------------------===//

// DEVICE_TRIM request. Releases unused device resources.
typedef struct iree_hal_remote_device_trim_request_t {
  uint32_t flags;     // Reserved, must be 0.
  uint32_t reserved;  // Must be 0.
} iree_hal_remote_device_trim_request_t;
static_assert(sizeof(iree_hal_remote_device_trim_request_t) == 8, "");
// Response: status only (no _response_t struct).

//===----------------------------------------------------------------------===//
// Semaphore messages
//===----------------------------------------------------------------------===//

// SEMAPHORE_CREATE request. Creates a timeline semaphore. [epoch]
typedef struct iree_hal_remote_semaphore_create_request_t {
  iree_hal_remote_resource_id_t provisional_id;  // PROVISIONAL=1
  uint64_t initial_value;
} iree_hal_remote_semaphore_create_request_t;
static_assert(sizeof(iree_hal_remote_semaphore_create_request_t) == 16, "");

// SEMAPHORE_CREATE response. Returns the server-assigned canonical ID.
typedef struct iree_hal_remote_semaphore_create_response_t {
  iree_hal_remote_resource_id_t resolved_id;  // PROVISIONAL=0
} iree_hal_remote_semaphore_create_response_t;
static_assert(sizeof(iree_hal_remote_semaphore_create_response_t) == 8, "");

// SEMAPHORE_QUERY request. Reads the current semaphore value.
typedef struct iree_hal_remote_semaphore_query_request_t {
  iree_hal_remote_resource_id_t semaphore_id;
} iree_hal_remote_semaphore_query_request_t;
static_assert(sizeof(iree_hal_remote_semaphore_query_request_t) == 8, "");

// SEMAPHORE_QUERY response. Returns the current value.
typedef struct iree_hal_remote_semaphore_query_response_t {
  uint64_t value;
} iree_hal_remote_semaphore_query_response_t;
static_assert(sizeof(iree_hal_remote_semaphore_query_response_t) == 8, "");

// SEMAPHORE_SIGNAL. Fire-and-forget: advances the semaphore to new_value.
typedef struct iree_hal_remote_semaphore_signal_t {
  iree_hal_remote_resource_id_t semaphore_id;
  uint64_t new_value;
} iree_hal_remote_semaphore_signal_t;
static_assert(sizeof(iree_hal_remote_semaphore_signal_t) == 16, "");

// SEMAPHORE_WAIT request. Waits until the semaphore reaches minimum_value
// or the timeout expires. The server MUST process this asynchronously
// (register a callback and continue processing other control messages) to
// avoid head-of-line blocking on the control channel. The response is sent
// when the wait completes; the envelope's request_id matches it to the
// original request.
typedef struct iree_hal_remote_semaphore_wait_request_t {
  iree_hal_remote_resource_id_t semaphore_id;
  uint64_t minimum_value;
  uint64_t timeout_ns;  // IREE_DURATION_INFINITE for unbounded wait.
} iree_hal_remote_semaphore_wait_request_t;
static_assert(sizeof(iree_hal_remote_semaphore_wait_request_t) == 24, "");
// Response: status only (OK or DEADLINE_EXCEEDED).

//===----------------------------------------------------------------------===//
// Executable messages
//===----------------------------------------------------------------------===//

// EXECUTABLE_UPLOAD request. Loads a native executable artifact. [epoch]
//
// |target_ordinal| indexes the immutable executable target table advertised
// for device 0 during bootstrap. The server resolves the ordinal against its
// local copy of the same device spec so the target passed to
// iree_hal_device_load_executable is borrowed from that device.
//
// Data delivery is controlled by |upload_flags|: either inline (data follows
// this struct) or bulk (referenced by |bulk_transfer_id|, with server
// processing deferred until the bulk transfer completes).
typedef struct iree_hal_remote_executable_upload_request_t {
  iree_hal_remote_resource_id_t provisional_id;  // PROVISIONAL=1
  uint64_t target_ordinal;    // Ordinal in the bootstrapped target table.
  uint64_t queue_affinity;    // iree_hal_queue_affinity_t.
  uint64_t data_length;       // Byte count of executable data.
  uint64_t bulk_transfer_id;  // Valid when BULK_REFERENCE is set.
  uint32_t load_flags;        // iree_hal_executable_load_flags_t.
  uint16_t constant_count;    // Specialization constant count.
  uint16_t upload_flags;      // IREE_HAL_REMOTE_UPLOAD_FLAG_*.
  // Followed by:
  //   uint32_t constants[constant_count]  (padded to 8-byte alignment)
  //   [if INLINE_DATA]: uint8_t data[data_length]
} iree_hal_remote_executable_upload_request_t;
static_assert(sizeof(iree_hal_remote_executable_upload_request_t) == 48, "");
static_assert(offsetof(iree_hal_remote_executable_upload_request_t,
                       load_flags) == 40,
              "");

// EXECUTABLE_UPLOAD response. Returns the resolved ID and function count.
typedef struct iree_hal_remote_executable_upload_response_t {
  iree_hal_remote_resource_id_t resolved_id;  // PROVISIONAL=0
  uint32_t function_count;  // Number of entry points in the executable.
  uint32_t reserved;        // Must be 0.
} iree_hal_remote_executable_upload_response_t;
static_assert(sizeof(iree_hal_remote_executable_upload_response_t) == 16, "");

// EXECUTABLE_QUERY_FUNCTION request. Queries metadata for a specific entry
// point.
typedef struct iree_hal_remote_executable_query_function_request_t {
  iree_hal_remote_resource_id_t executable_id;
  uint64_t function_value;
} iree_hal_remote_executable_query_function_request_t;
static_assert(sizeof(iree_hal_remote_executable_query_function_request_t) == 16,
              "");

// EXECUTABLE_QUERY_FUNCTION response. Returns fixed function metadata followed
// by the function name bytes. The name is UTF-8, not null-terminated, and is
// only valid for the lifetime of the response payload; clients that expose it
// through HAL APIs must copy it into executable-owned storage.
typedef struct iree_hal_remote_executable_query_function_response_t {
  // iree_hal_executable_function_flags_t behavior bits.
  uint64_t flags;
  // Static or minimum workgroup size.
  uint32_t workgroup_size[3];
  // Reserved occupancy information; must be 0.
  int32_t occupancy_reserved;
  // Total byte length of constants expected.
  uint32_t constant_byte_length;
  // Total number of bindings expected.
  uint16_t binding_count;
  // Total number of logical parameters.
  uint16_t parameter_count;
  // Byte length of the following function name.
  uint16_t name_length;
  // Must be 0.
  uint16_t reserved0;
  // Maximum invocations accepted in one workgroup, or 0 if unspecified.
  uint32_t maximum_workgroup_invocations;
  // iree_hal_executable_function_resource_flags_t available fields.
  uint32_t resource_usage_provided_flags;
  // Fixed workgroup-local memory required per workgroup, in bytes.
  uint32_t fixed_workgroup_local_memory_size;
  // Fixed private memory required per invocation, in bytes.
  uint32_t fixed_private_memory_size;
  // Number of 32-bit register units used per invocation.
  uint32_t invocation_register_count;
  // Must be 0.
  uint64_t reserved1;
  // Followed by:
  //   char name[name_length]
} iree_hal_remote_executable_query_function_response_t;
static_assert(sizeof(iree_hal_remote_executable_query_function_response_t) ==
                  64,
              "");
static_assert(offsetof(iree_hal_remote_executable_query_function_response_t,
                       maximum_workgroup_invocations) == 36,
              "");

// EXECUTABLE_QUERY_PARAMETERS request. Queries reflected parameters for a
// specific function. The server returns at most capacity entries.
typedef struct iree_hal_remote_executable_query_parameters_request_t {
  iree_hal_remote_resource_id_t executable_id;
  uint64_t function_value;
  uint16_t capacity;
  uint16_t reserved[3];  // Must be 0.
} iree_hal_remote_executable_query_parameters_request_t;
static_assert(sizeof(iree_hal_remote_executable_query_parameters_request_t) ==
                  24,
              "");

// Wire representation of iree_hal_executable_function_parameter_t. Parameter
// names are serialized out-of-line in the containing response so the struct is
// pointer-free and stable across processes.
typedef struct iree_hal_remote_executable_function_parameter_t {
  // HAL dispatch byte offset or binding ordinal.
  uint16_t offset;
  // Target ABI byte offset when the corresponding flag is set.
  uint16_t native_abi_offset;
  // iree_hal_executable_function_parameter_flags_t bitfield.
  uint16_t flags;
  // Parameter byte size.
  uint16_t size;
  // Byte length of the out-of-line parameter name.
  uint16_t name_length;
  // iree_hal_executable_function_parameter_type_t value.
  uint8_t type;
  // Reserved for future protocol use and must be zero.
  uint8_t reserved[5];
} iree_hal_remote_executable_function_parameter_t;
static_assert(sizeof(iree_hal_remote_executable_function_parameter_t) == 16,
              "");

// EXECUTABLE_QUERY_PARAMETERS response. Fixed parameter records are followed
// by concatenated UTF-8 parameter name bytes in record order. Names are not
// null-terminated and are only valid for the lifetime of the response payload;
// clients that expose them through HAL APIs must copy them into
// executable-owned storage.
typedef struct iree_hal_remote_executable_query_parameters_response_t {
  uint16_t parameter_count;
  uint16_t reserved;  // Must be 0.
  uint32_t name_data_length;
  // Followed by:
  //   iree_hal_remote_executable_function_parameter_t
  //   parameters[parameter_count] char names[name_data_length]
} iree_hal_remote_executable_query_parameters_response_t;
static_assert(sizeof(iree_hal_remote_executable_query_parameters_response_t) ==
                  8,
              "");

// EXECUTABLE_LOOKUP_GLOBAL request. Looks up an executable global by name.
typedef struct iree_hal_remote_executable_lookup_global_request_t {
  iree_hal_remote_resource_id_t executable_id;
  uint16_t name_length;  // Byte length of the following global name.
  uint16_t reserved0;    // Must be 0.
  uint32_t reserved1;    // Must be 0.
  // Followed by:
  //   char name[name_length]
} iree_hal_remote_executable_lookup_global_request_t;
static_assert(sizeof(iree_hal_remote_executable_lookup_global_request_t) == 16,
              "");

// EXECUTABLE_LOOKUP_GLOBAL response.
typedef struct iree_hal_remote_executable_lookup_global_response_t {
  uint64_t byte_length;
  uint32_t found;     // 0 if no global matched; otherwise 1.
  uint32_t reserved;  // Must be 0.
} iree_hal_remote_executable_lookup_global_response_t;
static_assert(sizeof(iree_hal_remote_executable_lookup_global_response_t) == 16,
              "");

// EXECUTABLE_GLOBAL_BUFFER request. Resolves an executable global to the
// buffer instance selected by |queue_affinity|.
typedef struct iree_hal_remote_executable_global_buffer_request_t {
  iree_hal_remote_resource_id_t executable_id;
  uint64_t queue_affinity;  // iree_hal_queue_affinity_t.
  uint16_t name_length;     // Byte length of the following global name.
  uint16_t reserved0;       // Must be 0.
  uint32_t reserved1;       // Must be 0.
  // Followed by:
  //   char name[name_length]
} iree_hal_remote_executable_global_buffer_request_t;
static_assert(sizeof(iree_hal_remote_executable_global_buffer_request_t) == 24,
              "");

// EXECUTABLE_GLOBAL_BUFFER response. The resource ID refers to the exact
// buffer alias returned by the server executable. Client proxies must treat it
// as a zero-offset buffer of |byte_length| bytes instead of reconstructing any
// server-local root allocation.
typedef struct iree_hal_remote_executable_global_buffer_response_t {
  iree_hal_remote_resource_id_t resolved_id;  // PROVISIONAL=0
  iree_hal_remote_buffer_params_t params;
  uint64_t byte_length;
  uint32_t placement_flags;  // iree_hal_buffer_placement_flags_t
  uint32_t reserved;         // Must be 0.
} iree_hal_remote_executable_global_buffer_response_t;
static_assert(sizeof(iree_hal_remote_executable_global_buffer_response_t) == 56,
              "");

//===----------------------------------------------------------------------===//
// Command buffer messages
//===----------------------------------------------------------------------===//

// COMMAND_BUFFER_UPLOAD request. Uploads a reusable command buffer. [epoch]
// Same upload delivery model as EXECUTABLE_UPLOAD (inline or bulk). The server
// owns local validation and resource retention; |mode| only carries reusable
// metadata-retention intent across the protocol.
typedef struct iree_hal_remote_command_buffer_upload_request_t {
  // Provisional command buffer resource ID.
  iree_hal_remote_resource_id_t provisional_id;
  // Client command buffer mode hints.
  uint32_t mode;
  // Command categories used by the recording.
  uint32_t categories;
  // Maximum binding table slot count.
  uint16_t binding_capacity;
  // Exactly one IREE_HAL_REMOTE_UPLOAD_FLAG_* delivery mode.
  uint16_t upload_flags;
  // Reserved for future use and must be zero.
  uint32_t reserved;
  // Byte count of serialized command stream data.
  uint64_t data_length;
  // Transfer ID when BULK_REFERENCE is selected; otherwise zero.
  uint64_t bulk_transfer_id;
  // [if INLINE_DATA]: uint8_t data[data_length]  (padded to 8-byte alignment)
} iree_hal_remote_command_buffer_upload_request_t;
static_assert(sizeof(iree_hal_remote_command_buffer_upload_request_t) == 40,
              "");
static_assert(offsetof(iree_hal_remote_command_buffer_upload_request_t,
                       data_length) == 24,
              "");

// COMMAND_BUFFER_UPLOAD response. Returns the resolved ID.
typedef struct iree_hal_remote_command_buffer_upload_response_t {
  iree_hal_remote_resource_id_t resolved_id;  // PROVISIONAL=0
} iree_hal_remote_command_buffer_upload_response_t;
static_assert(sizeof(iree_hal_remote_command_buffer_upload_response_t) == 8,
              "");

//===----------------------------------------------------------------------===//
// File messages
//===----------------------------------------------------------------------===//

typedef uint8_t iree_hal_remote_file_external_type_t;

// External file handle namespaces used by FILE_REGISTER.
//
// These values describe handles transferred by a transport-supported external
// handle mechanism. Process-local integers such as POSIX fds are only valid
// when the transport actually transfers the underlying handle rights; they must
// never be interpreted as ordinary scalar payload across machines.
enum iree_hal_remote_file_external_type_e {
  IREE_HAL_REMOTE_FILE_EXTERNAL_TYPE_NONE = 0u,
  IREE_HAL_REMOTE_FILE_EXTERNAL_TYPE_POSIX_FD = 1u,
  IREE_HAL_REMOTE_FILE_EXTERNAL_TYPE_WIN32_HANDLE = 2u,
};

typedef uint32_t iree_hal_remote_file_registration_capabilities_t;

// FILE_REGISTER external handle capabilities advertised by a transport/server.
enum iree_hal_remote_file_registration_capability_bits_e {
  IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_NONE = 0u,
  IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_POSIX_FD = 1u << 0,
  IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_WIN32_HANDLE = 1u << 1,
};

static inline iree_hal_remote_file_registration_capabilities_t
iree_hal_remote_file_registration_capability_for_external_type(
    iree_hal_remote_file_external_type_t external_type) {
  switch (external_type) {
    case IREE_HAL_REMOTE_FILE_EXTERNAL_TYPE_POSIX_FD:
      return IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_POSIX_FD;
    case IREE_HAL_REMOTE_FILE_EXTERNAL_TYPE_WIN32_HANDLE:
      return IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_WIN32_HANDLE;
    default:
      return IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_NONE;
  }
}

// FILE_OPEN request. Opens a server-side file by logical name. [epoch]
// The server resolves logical names through its explicit allow-list and the
// client never sees real filesystem paths. The logical namespace may contain
// operator-configured symlinks; the portable client-controlled suffix grammar
// rejects absolute paths, parent traversal, empty path segments, and alternate
// platform separators. The client provides a provisional_id so queue ops
// (FILE_READ/FILE_WRITE) can reference the file immediately without waiting
// for the response round-trip. Servers park those queue ops until this control
// request resolves the provisional ID. Fire-and-forget opens do not produce a
// response; failures are reported through any parked/subsequent queue ops using
// the provisional file ID.
typedef struct iree_hal_remote_file_open_request_t {
  iree_hal_remote_resource_id_t provisional_id;  // PROVISIONAL=1
  uint16_t path_length;  // UTF-8 byte count (not null-terminated).
  uint16_t mode;         // Access mode (read, write, read-write).
  uint32_t flags;        // Reserved, must be 0.
  // Followed by:
  //   uint8_t path[path_length]  (padded to 8-byte alignment)
} iree_hal_remote_file_open_request_t;
static_assert(sizeof(iree_hal_remote_file_open_request_t) == 16, "");

// FILE_OPEN response. Returns the resolved file ID and metadata.
typedef struct iree_hal_remote_file_open_response_t {
  iree_hal_remote_resource_id_t resolved_id;  // PROVISIONAL=0
  uint64_t file_size;
  uint32_t granted_access;  // Actual access granted (may differ from request).
  uint32_t reserved;        // Must be 0.
} iree_hal_remote_file_open_response_t;
static_assert(sizeof(iree_hal_remote_file_open_response_t) == 24, "");
static_assert(offsetof(iree_hal_remote_file_open_response_t, file_size) == 8,
              "");

// FILE_CLOSE. Fire-and-forget: closes a previously opened file.
typedef struct iree_hal_remote_file_close_t {
  iree_hal_remote_resource_id_t file_id;
} iree_hal_remote_file_close_t;
static_assert(sizeof(iree_hal_remote_file_close_t) == 8, "");

// FILE_REGISTER request. Registers an external file handle for use with
// queue-ordered I/O operations (FILE_READ/FILE_WRITE). [epoch] The
// external_type identifies the handle namespace; handle_payload carries
// type-specific metadata for the transport-supported handle transfer.
typedef struct iree_hal_remote_file_register_request_t {
  iree_hal_remote_resource_id_t provisional_id;  // PROVISIONAL=1
  uint32_t external_type;          // iree_hal_remote_file_external_type_t.
  uint32_t access_flags;           // Access mode for the registered file.
  uint32_t handle_payload_length;  // Byte count of type-specific handle data.
  uint32_t reserved;               // Must be 0.
  // Followed by:
  //   uint8_t handle_payload[handle_payload_length]  (padded to 8-byte align)
} iree_hal_remote_file_register_request_t;
static_assert(sizeof(iree_hal_remote_file_register_request_t) == 24, "");

// FILE_REGISTER response. Returns the resolved file ID and discovered size.
typedef struct iree_hal_remote_file_register_response_t {
  iree_hal_remote_resource_id_t resolved_id;  // PROVISIONAL=0
  uint64_t file_size;
} iree_hal_remote_file_register_response_t;
static_assert(sizeof(iree_hal_remote_file_register_response_t) == 16, "");

// FILE_UNREGISTER. Fire-and-forget: unregisters a previously registered file.
typedef struct iree_hal_remote_file_unregister_t {
  iree_hal_remote_resource_id_t file_id;
} iree_hal_remote_file_unregister_t;
static_assert(sizeof(iree_hal_remote_file_unregister_t) == 8, "");

// FILE_LIST request. Lists files available on the server, optionally filtered
// by a glob pattern. Variable-length tail carries the pattern.
typedef struct iree_hal_remote_file_list_request_t {
  uint16_t pattern_length;  // 0 = list all available files.
  uint16_t reserved0;       // Must be 0.
  uint32_t reserved1;       // Must be 0.
  // Followed by:
  //   uint8_t pattern[pattern_length]  (padded to 8-byte alignment)
} iree_hal_remote_file_list_request_t;
static_assert(sizeof(iree_hal_remote_file_list_request_t) == 8, "");

// Entry in a FILE_LIST response. Variable-length: fixed header followed by
// the file name string.
typedef struct iree_hal_remote_file_list_entry_t {
  uint64_t file_size;
  uint16_t name_length;  // UTF-8 byte count (not null-terminated).
  uint16_t reserved0;    // Must be 0.
  uint32_t reserved1;    // Must be 0.
  // Followed by:
  //   uint8_t name[name_length]  (padded to 8-byte alignment)
} iree_hal_remote_file_list_entry_t;
static_assert(sizeof(iree_hal_remote_file_list_entry_t) == 16, "");
static_assert(offsetof(iree_hal_remote_file_list_entry_t, name_length) == 8,
              "");

// FILE_LIST response. Variable-length: entry_count followed by that many
// iree_hal_remote_file_list_entry_t records (each itself variable-length).
typedef struct iree_hal_remote_file_list_response_t {
  uint32_t entry_count;
  uint32_t reserved;  // Must be 0.
  // Followed by:
  //   iree_hal_remote_file_list_entry_t entries[entry_count]
  //   (each entry is variable-length due to its name string)
} iree_hal_remote_file_list_response_t;
static_assert(sizeof(iree_hal_remote_file_list_response_t) == 8, "");

//===----------------------------------------------------------------------===//
// Buffer messages
//===----------------------------------------------------------------------===//

// BUFFER_ALLOC request. Allocates a persistent buffer (model weights, I/O
// staging). [epoch] For transient queue-ordered allocation, use the
// BUFFER_ALLOCA queue op instead.
typedef struct iree_hal_remote_buffer_alloc_request_t {
  iree_hal_remote_resource_id_t provisional_id;  // PROVISIONAL=1
  iree_hal_remote_buffer_params_t params;        // 32 bytes.
  uint64_t allocation_size;
} iree_hal_remote_buffer_alloc_request_t;
static_assert(sizeof(iree_hal_remote_buffer_alloc_request_t) == 48, "");
static_assert(offsetof(iree_hal_remote_buffer_alloc_request_t, params) == 8,
              "");
static_assert(offsetof(iree_hal_remote_buffer_alloc_request_t,
                       allocation_size) == 40,
              "");

// BUFFER_ALLOC response. Returns the resolved buffer ID and concrete server
// allocation properties.
typedef struct iree_hal_remote_buffer_alloc_response_t {
  iree_hal_remote_resource_id_t resolved_id;  // PROVISIONAL=0
  iree_hal_remote_buffer_params_t params;
  uint64_t allocation_size;
  uint64_t byte_length;
  uint32_t placement_flags;  // iree_hal_buffer_placement_flags_t
  uint32_t reserved;         // Must be 0.
} iree_hal_remote_buffer_alloc_response_t;
static_assert(sizeof(iree_hal_remote_buffer_alloc_response_t) == 64, "");

// BUFFER_IMPORT request. Imports an externally-owned buffer (shared memory,
// DMA-BUF, platform handle). [epoch] The external_type identifies the import
// mechanism; the variable-length handle_payload carries type-specific metadata
// (e.g., DMA-BUF fd + plane offsets/strides, AHardwareBuffer serialization,
// SHM handle + size). This extensible layout avoids baking platform-specific
// handle structures into the fixed protocol.
typedef struct iree_hal_remote_buffer_import_request_t {
  iree_hal_remote_resource_id_t provisional_id;  // PROVISIONAL=1
  iree_hal_remote_buffer_params_t params;        // 32 bytes.
  uint64_t allocation_size;
  uint32_t external_type;          // External buffer type identifier.
  uint32_t handle_payload_length;  // Byte count of type-specific handle data.
  // Followed by:
  //   uint8_t handle_payload[handle_payload_length]  (padded to 8-byte align)
} iree_hal_remote_buffer_import_request_t;
static_assert(sizeof(iree_hal_remote_buffer_import_request_t) == 56, "");
static_assert(offsetof(iree_hal_remote_buffer_import_request_t, params) == 8,
              "");
static_assert(offsetof(iree_hal_remote_buffer_import_request_t,
                       external_type) == 48,
              "");

// BUFFER_IMPORT response. Returns the resolved buffer ID.
typedef struct iree_hal_remote_buffer_import_response_t {
  iree_hal_remote_resource_id_t resolved_id;  // PROVISIONAL=0
} iree_hal_remote_buffer_import_response_t;
static_assert(sizeof(iree_hal_remote_buffer_import_response_t) == 8, "");

// BUFFER_MAP flags.
#define IREE_HAL_REMOTE_BUFFER_MAP_FLAG_BULK_TRANSFER (1u << 0)

// BUFFER_MAP request. Reads buffer contents from the server. Inline maps return
// the data in the control response. Bulk maps stream the data under
// |transfer_id| on the bulk channel and return only metadata in the control
// response. No persistent server-side mapping state is created.
typedef struct iree_hal_remote_buffer_map_request_t {
  iree_hal_remote_resource_id_t buffer_id;
  uint32_t memory_access;  // iree_hal_memory_access_t bits (READ, WRITE, etc.)
  uint32_t flags;          // IREE_HAL_REMOTE_BUFFER_MAP_FLAG_*.
  uint64_t offset;
  uint64_t length;
  uint64_t transfer_id;  // Required when BULK_TRANSFER is set; otherwise 0.
} iree_hal_remote_buffer_map_request_t;
static_assert(sizeof(iree_hal_remote_buffer_map_request_t) == 40, "");

// BUFFER_MAP response. Inline READ responses carry mapped_length bytes after
// this header and set transfer_id=0. Bulk READ responses set transfer_id to the
// bulk transfer carrying mapped_length bytes and carry no inline data. When
// only WRITE|DISCARD was requested, mapped_length and transfer_id are 0.
typedef struct iree_hal_remote_buffer_map_response_t {
  uint64_t mapped_offset;  // Actual start offset of the mapped region.
  uint64_t mapped_length;  // Actual byte count of the mapped region.
  uint64_t transfer_id;    // Bulk transfer ID, or 0 for inline/no data.
  // Followed by: uint8_t data[mapped_length] for inline READ responses.
} iree_hal_remote_buffer_map_response_t;
static_assert(sizeof(iree_hal_remote_buffer_map_response_t) == 24, "");

// BUFFER_UNMAP flags.
#define IREE_HAL_REMOTE_BUFFER_UNMAP_FLAG_BULK_TRANSFER (1u << 0)

// BUFFER_UNMAP request. Writes buffer contents to the server. Inline requests
// carry length bytes after this header. Bulk requests stream the bytes under
// |transfer_id| on the bulk channel and carry no inline data. The server
// responds only after the write completes so the client can safely proceed with
// queue operations that depend on the data being present.
typedef struct iree_hal_remote_buffer_unmap_request_t {
  iree_hal_remote_resource_id_t buffer_id;
  uint64_t offset;
  uint64_t length;
  uint32_t flags;        // IREE_HAL_REMOTE_BUFFER_UNMAP_FLAG_*.
  uint32_t reserved;     // Must be 0.
  uint64_t transfer_id;  // Required when BULK_TRANSFER is set; otherwise 0.
  // Followed by: uint8_t data[length] for inline requests.
} iree_hal_remote_buffer_unmap_request_t;
static_assert(sizeof(iree_hal_remote_buffer_unmap_request_t) == 40, "");
// Response: status only (no body).

// BUFFER_VIRTUAL_QUERY_CAPABILITIES request. Queries allocator-level virtual
// memory capabilities that are not parameter-specific.
typedef struct iree_hal_remote_buffer_virtual_query_capabilities_request_t {
  uint32_t flags;     // Reserved, must be 0.
  uint32_t reserved;  // Must be 0.
} iree_hal_remote_buffer_virtual_query_capabilities_request_t;
static_assert(
    sizeof(iree_hal_remote_buffer_virtual_query_capabilities_request_t) == 8,
    "");

// BUFFER_VIRTUAL_QUERY_CAPABILITIES response. Nonzero when the server allocator
// exposes the HAL virtual memory surface.
typedef struct iree_hal_remote_buffer_virtual_query_capabilities_response_t {
  uint32_t supports_virtual_memory;
  uint32_t reserved;  // Must be 0.
} iree_hal_remote_buffer_virtual_query_capabilities_response_t;
static_assert(
    sizeof(iree_hal_remote_buffer_virtual_query_capabilities_response_t) == 8,
    "");

// BUFFER_VIRTUAL_QUERY_GRANULARITY request.
typedef struct iree_hal_remote_buffer_virtual_query_granularity_request_t {
  iree_hal_remote_buffer_params_t params;
} iree_hal_remote_buffer_virtual_query_granularity_request_t;
static_assert(
    sizeof(iree_hal_remote_buffer_virtual_query_granularity_request_t) == 32,
    "");

// BUFFER_VIRTUAL_QUERY_GRANULARITY response.
typedef struct iree_hal_remote_buffer_virtual_query_granularity_response_t {
  uint64_t minimum_page_size;
  uint64_t recommended_page_size;
} iree_hal_remote_buffer_virtual_query_granularity_response_t;
static_assert(
    sizeof(iree_hal_remote_buffer_virtual_query_granularity_response_t) == 16,
    "");

// BUFFER_VIRTUAL_RESERVE request. Reserves a virtual address range on the
// server.
typedef struct iree_hal_remote_buffer_virtual_reserve_request_t {
  uint64_t queue_affinity;  // iree_hal_queue_affinity_t
  uint64_t size;
} iree_hal_remote_buffer_virtual_reserve_request_t;
static_assert(sizeof(iree_hal_remote_buffer_virtual_reserve_request_t) == 16,
              "");

// BUFFER_VIRTUAL_RESERVE response. Returns the resolved virtual buffer
// resource.
typedef struct iree_hal_remote_buffer_virtual_reserve_response_t {
  iree_hal_remote_resource_id_t resolved_id;
  iree_hal_remote_buffer_params_t params;
  uint64_t allocation_size;
  uint32_t placement_flags;  // iree_hal_buffer_placement_flags_t
  uint32_t reserved;         // Must be 0.
} iree_hal_remote_buffer_virtual_reserve_response_t;
static_assert(sizeof(iree_hal_remote_buffer_virtual_reserve_response_t) == 56,
              "");

// BUFFER_VIRTUAL_RELEASE request. Releases a virtual address reservation.
typedef struct iree_hal_remote_buffer_virtual_release_request_t {
  iree_hal_remote_resource_id_t buffer_id;
} iree_hal_remote_buffer_virtual_release_request_t;
static_assert(sizeof(iree_hal_remote_buffer_virtual_release_request_t) == 8,
              "");
// Response: status only (no body).

// BUFFER_PHYSICAL_ALLOC request. Allocates physical memory for later mapping.
typedef struct iree_hal_remote_buffer_physical_alloc_request_t {
  iree_hal_remote_buffer_params_t params;
  uint64_t size;
} iree_hal_remote_buffer_physical_alloc_request_t;
static_assert(sizeof(iree_hal_remote_buffer_physical_alloc_request_t) == 40,
              "");

// BUFFER_PHYSICAL_ALLOC response.
typedef struct iree_hal_remote_buffer_physical_alloc_response_t {
  iree_hal_remote_resource_id_t resolved_id;
} iree_hal_remote_buffer_physical_alloc_response_t;
static_assert(sizeof(iree_hal_remote_buffer_physical_alloc_response_t) == 8,
              "");

// BUFFER_PHYSICAL_FREE request. Frees physical memory.
typedef struct iree_hal_remote_buffer_physical_free_request_t {
  iree_hal_remote_resource_id_t physical_memory_id;
} iree_hal_remote_buffer_physical_free_request_t;
static_assert(sizeof(iree_hal_remote_buffer_physical_free_request_t) == 8, "");
// Response: status only (no body).

// BUFFER_VIRTUAL_MAP request. Maps physical memory into a virtual reservation.
typedef struct iree_hal_remote_buffer_virtual_map_request_t {
  iree_hal_remote_resource_id_t buffer_id;
  iree_hal_remote_resource_id_t physical_memory_id;
  uint64_t virtual_offset;
  uint64_t physical_offset;
  uint64_t size;
} iree_hal_remote_buffer_virtual_map_request_t;
static_assert(sizeof(iree_hal_remote_buffer_virtual_map_request_t) == 40, "");
// Response: status only (no body).

// BUFFER_VIRTUAL_UNMAP request. Unmaps part of a virtual reservation.
typedef struct iree_hal_remote_buffer_virtual_unmap_request_t {
  iree_hal_remote_resource_id_t buffer_id;
  uint64_t virtual_offset;
  uint64_t size;
} iree_hal_remote_buffer_virtual_unmap_request_t;
static_assert(sizeof(iree_hal_remote_buffer_virtual_unmap_request_t) == 24, "");
// Response: status only (no body).

// BUFFER_VIRTUAL_PROTECT request. Changes protection on a virtual memory range.
typedef struct iree_hal_remote_buffer_virtual_protect_request_t {
  iree_hal_remote_resource_id_t buffer_id;
  uint64_t virtual_offset;
  uint64_t size;
  uint64_t queue_affinity;  // iree_hal_queue_affinity_t
  uint64_t protection;      // iree_hal_memory_protection_t
} iree_hal_remote_buffer_virtual_protect_request_t;
static_assert(sizeof(iree_hal_remote_buffer_virtual_protect_request_t) == 40,
              "");
// Response: status only (no body).

// BUFFER_VIRTUAL_ADVISE request. Applies usage advice to a virtual memory
// range.
typedef struct iree_hal_remote_buffer_virtual_advise_request_t {
  iree_hal_remote_resource_id_t buffer_id;
  uint64_t virtual_offset;
  uint64_t size;
  uint64_t queue_affinity;  // iree_hal_queue_affinity_t
  uint64_t advice;          // iree_hal_memory_advice_t
} iree_hal_remote_buffer_virtual_advise_request_t;
static_assert(sizeof(iree_hal_remote_buffer_virtual_advise_request_t) == 40,
              "");
// Response: status only (no body).

//===----------------------------------------------------------------------===//
// Profiling messages
//===----------------------------------------------------------------------===//

// PROFILING_BEGIN request. Starts a HAL-native structured profiling session on
// the wrapped server device. The client-owned sink is not serialized here; the
// server creates a relay sink that forwards callback payloads over the bulk
// channel using iree_hal_remote_profile_transfer_header_t records.
typedef struct iree_hal_remote_profiling_begin_request_t {
  // iree_hal_device_profiling_data_families_t requested by the client.
  uint64_t data_families;
  // Capture-filter command buffer identifier, or 0 when inactive.
  uint64_t command_buffer_id;
  // iree_hal_device_profiling_flags_t behavior bits.
  uint32_t flags;
  // iree_hal_profile_capture_filter_flags_t active filter fields.
  uint32_t capture_filter_flags;
  // Capture-filter command index, valid when its flag is set.
  uint32_t command_index;
  // Capture-filter physical device ordinal, valid when its flag is set.
  uint32_t physical_device_ordinal;
  // Capture-filter queue ordinal, valid when its flag is set.
  uint32_t queue_ordinal;
  // Byte length of the executable function glob pattern.
  uint32_t executable_function_pattern_length;
  // Number of counter set selection records in the variable-length tail.
  uint32_t counter_set_count;
  // Total number of counter names across all counter set selections.
  uint32_t counter_name_count;
  // Must be 0.
  uint32_t reserved[2];
  // Followed by:
  //   char executable_function_pattern[executable_function_pattern_length]
  //       (padded to 8-byte alignment)
  //   iree_hal_remote_profile_counter_set_selection_t
  //       counter_sets[counter_set_count], each followed by:
  //     char name[name_length]  (padded to 8-byte alignment)
  //     iree_hal_remote_profile_counter_name_t
  //         counter_names[counter_name_count], each followed by:
  //       char name[name_length]  (padded to 8-byte alignment)
} iree_hal_remote_profiling_begin_request_t;
static_assert(sizeof(iree_hal_remote_profiling_begin_request_t) == 56, "");
static_assert(offsetof(iree_hal_remote_profiling_begin_request_t,
                       data_families) == 0,
              "");
static_assert(offsetof(iree_hal_remote_profiling_begin_request_t,
                       counter_set_count) == 40,
              "");

// Variable-length counter set selection embedded in PROFILING_BEGIN.
typedef struct iree_hal_remote_profile_counter_set_selection_t {
  // iree_hal_profile_counter_set_selection_flags_t behavior bits.
  uint32_t flags;
  // Number of counter names following this counter set name.
  uint32_t counter_name_count;
  // Byte length of the counter set name following this header.
  uint32_t name_length;
  // Must be 0.
  uint32_t reserved;
  // Followed by:
  //   char name[name_length]  (padded to 8-byte alignment)
  //   iree_hal_remote_profile_counter_name_t counter_names[counter_name_count],
  //       each followed by:
  //     char name[name_length]  (padded to 8-byte alignment)
} iree_hal_remote_profile_counter_set_selection_t;
static_assert(sizeof(iree_hal_remote_profile_counter_set_selection_t) == 16,
              "");

// Variable-length counter name embedded in a counter set selection.
typedef struct iree_hal_remote_profile_counter_name_t {
  // Byte length of the counter name following this header.
  uint32_t name_length;
  // Must be 0.
  uint32_t reserved;
  // Followed by:
  //   char name[name_length]  (padded to 8-byte alignment)
} iree_hal_remote_profile_counter_name_t;
static_assert(sizeof(iree_hal_remote_profile_counter_name_t) == 8, "");

// PROFILING_FLUSH request. Flushes profile data for the active session.
typedef struct iree_hal_remote_profiling_flush_request_t {
  uint32_t flags;     // Reserved, must be 0.
  uint32_t reserved;  // Must be 0.
} iree_hal_remote_profiling_flush_request_t;
static_assert(sizeof(iree_hal_remote_profiling_flush_request_t) == 8, "");
// Response: status only (no _response_t struct).

// PROFILING_END request. Ends the active profile session.
typedef struct iree_hal_remote_profiling_end_request_t {
  uint32_t flags;     // Reserved, must be 0.
  uint32_t reserved;  // Must be 0.
} iree_hal_remote_profiling_end_request_t;
static_assert(sizeof(iree_hal_remote_profiling_end_request_t) == 8, "");
// Response: status only (no _response_t struct).

//===----------------------------------------------------------------------===//
// Host call messages
//===----------------------------------------------------------------------===//

// HOST_CALL_REGISTER request. Reserved for future explicit server-side named
// handlers. HAL queue_host_call callbacks are client-local function pointers
// and are not represented by this protocol message.
typedef struct iree_hal_remote_host_call_register_request_t {
  uint64_t call_id;      // Client-chosen ID, unique within session.
  uint16_t name_length;  // UTF-8 handler name (not null-terminated).
  uint16_t flags;        // Reserved, must be 0.
  uint32_t reserved;     // Must be 0.
  // Followed by:
  //   uint8_t name[name_length]  (padded to 8-byte alignment)
} iree_hal_remote_host_call_register_request_t;
static_assert(sizeof(iree_hal_remote_host_call_register_request_t) == 16, "");
// Response: status only (validates call_id uniqueness).

// HOST_CALL_UNREGISTER. Fire-and-forget: unregisters a call handler.
typedef struct iree_hal_remote_host_call_unregister_t {
  uint64_t call_id;
} iree_hal_remote_host_call_unregister_t;
static_assert(sizeof(iree_hal_remote_host_call_unregister_t) == 8, "");

//===----------------------------------------------------------------------===//
// Lifecycle messages
//===----------------------------------------------------------------------===//

// RESOURCE_RELEASE_BATCH. Legacy control-channel release of resources.
//
// Current clients send iree_hal_remote_resource_release_op_t on the queue
// channel so releases are ordered with COMMAND frames that may reference the
// resources. Control-channel releases are still accepted for non-queue
// resources and session teardown compatibility.
typedef struct iree_hal_remote_resource_release_batch_t {
  uint32_t resource_count;
  uint32_t reserved;  // Must be 0.
  // Followed by:
  //   iree_hal_remote_resource_id_t resource_ids[resource_count]
} iree_hal_remote_resource_release_batch_t;
static_assert(sizeof(iree_hal_remote_resource_release_batch_t) == 8, "");

//===----------------------------------------------------------------------===//
// Extension messages
//===----------------------------------------------------------------------===//

// DEVICE_EXTENSION request. ioctl-style escape hatch for device-specific
// control operations. The server dispatches by device_type + operation.
typedef struct iree_hal_remote_device_extension_request_t {
  uint32_t device_type;     // Namespace (CUDA=1, HIP=2, Vulkan=3, ...).
  uint32_t operation;       // Extension-defined operation code.
  uint32_t payload_length;  // Byte count of opaque payload.
  uint32_t reserved;        // Must be 0.
  // Followed by:
  //   uint8_t payload[payload_length]  (padded to 8-byte alignment)
} iree_hal_remote_device_extension_request_t;
static_assert(sizeof(iree_hal_remote_device_extension_request_t) == 16, "");

// DEVICE_EXTENSION response. Extension-defined response payload.
typedef struct iree_hal_remote_device_extension_response_t {
  uint32_t payload_length;  // Byte count of opaque response payload.
  uint32_t reserved;        // Must be 0.
  // Followed by:
  //   uint8_t payload[payload_length]  (padded to 8-byte alignment)
} iree_hal_remote_device_extension_response_t;
static_assert(sizeof(iree_hal_remote_device_extension_response_t) == 8, "");

//===----------------------------------------------------------------------===//
// Notification messages
//===----------------------------------------------------------------------===//

// NOTIFY_RESOURCE_ERROR. Server → client notification that a resource has
// entered an error state (e.g., semaphore failure, executable load error).
typedef struct iree_hal_remote_notify_resource_error_t {
  iree_hal_remote_resource_id_t resource_id;
  uint32_t error_code;      // iree_status_code_t
  uint16_t message_length;  // UTF-8 diagnostic message.
  uint16_t reserved;        // Must be 0.
  // Followed by:
  //   uint8_t message[message_length]  (padded to 8-byte alignment)
} iree_hal_remote_notify_resource_error_t;
static_assert(sizeof(iree_hal_remote_notify_resource_error_t) == 16, "");

// NOTIFY_DEVICE_LOST. Server → client notification that the device is no
// longer functional. All pending operations fail. Session must be re-created.
typedef struct iree_hal_remote_notify_device_lost_t {
  uint32_t error_code;      // iree_status_code_t
  uint16_t message_length;  // UTF-8 diagnostic message.
  uint16_t reserved;        // Must be 0.
  // Followed by:
  //   uint8_t message[message_length]  (padded to 8-byte alignment)
} iree_hal_remote_notify_device_lost_t;
static_assert(sizeof(iree_hal_remote_notify_device_lost_t) == 8, "");

// NOTIFY_MEMORY_PRESSURE. Server → client notification of memory pressure.
// Client should release unused buffers or reduce allocation rate.
typedef struct iree_hal_remote_notify_memory_pressure_t {
  uint32_t pressure_flags;  // Severity/type of pressure.
  uint32_t reserved;        // Must be 0.
} iree_hal_remote_notify_memory_pressure_t;
static_assert(sizeof(iree_hal_remote_notify_memory_pressure_t) == 8, "");

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_PROTOCOL_CONTROL_H_
