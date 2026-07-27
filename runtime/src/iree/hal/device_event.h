// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DEVICE_EVENT_H_
#define IREE_HAL_DEVICE_EVENT_H_

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_t iree_hal_device_t;

//===----------------------------------------------------------------------===//
// iree_hal_device_event_t
//===----------------------------------------------------------------------===//

// ABI version for |iree_hal_device_event_t|.
#define IREE_HAL_DEVICE_EVENT_ABI_VERSION_0 0u

// ABI version for |iree_hal_device_event_site_t|.
#define IREE_HAL_DEVICE_EVENT_SITE_ABI_VERSION_0 0u

// ABI version for |iree_hal_device_asan_report_t|.
#define IREE_HAL_DEVICE_ASAN_REPORT_ABI_VERSION_0 0u

// ABI version for |iree_hal_device_ubsan_report_t|.
#define IREE_HAL_DEVICE_UBSAN_REPORT_ABI_VERSION_0 0u

// ABI version for |iree_hal_device_tsan_report_t|.
#define IREE_HAL_DEVICE_TSAN_REPORT_ABI_VERSION_0 0u

// ABI version for |iree_hal_device_printf_event_t|.
#define IREE_HAL_DEVICE_PRINTF_EVENT_ABI_VERSION_0 0u

// ABI version for |iree_hal_device_driver_failure_event_t|.
#define IREE_HAL_DEVICE_DRIVER_FAILURE_EVENT_ABI_VERSION_0 0u

// Event family emitted by a HAL device or driver.
typedef uint32_t iree_hal_device_event_type_t;
enum iree_hal_device_event_type_bits_t {
  IREE_HAL_DEVICE_EVENT_TYPE_NONE = 0u,
  // Low-level backend, device, queue, or driver failure report.
  IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE = 1u,
  // Address sanitizer report.
  IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT = 2u,
  // Undefined behavior sanitizer report.
  IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT = 3u,
  // Device-originated printf output.
  IREE_HAL_DEVICE_EVENT_TYPE_PRINTF = 4u,
  // Device-originated host-call diagnostic report.
  IREE_HAL_DEVICE_EVENT_TYPE_HOST_CALL = 5u,
  // Thread sanitizer report.
  IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT = 6u,
  // First event type reserved for implementation-specific payloads.
  IREE_HAL_DEVICE_EVENT_TYPE_USER = 0x8000u,
};

// Severity assigned by the event producer.
typedef uint32_t iree_hal_device_event_severity_t;
enum iree_hal_device_event_severity_bits_t {
  IREE_HAL_DEVICE_EVENT_SEVERITY_TRACE = 0u,
  IREE_HAL_DEVICE_EVENT_SEVERITY_INFO = 1u,
  IREE_HAL_DEVICE_EVENT_SEVERITY_WARNING = 2u,
  IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR = 3u,
  IREE_HAL_DEVICE_EVENT_SEVERITY_FATAL = 4u,
};

// Bitfield specifying event-family properties.
typedef uint64_t iree_hal_device_event_flags_t;
enum iree_hal_device_event_flag_bits_t {
  IREE_HAL_DEVICE_EVENT_FLAG_NONE = 0u,
};

// Source attribution for a device event.
typedef struct iree_hal_device_event_source_t {
  // Borrowed device identity pointer, or NULL when not applicable.
  iree_hal_device_t* device;
  // Stable device identifier string when available.
  iree_string_view_t device_id;
  // Backend name such as "amdgpu", "hip", "vulkan", "metal", or "webgpu".
  iree_string_view_t driver_id;
  // Physical device ordinal, or UINT32_MAX when not applicable.
  uint32_t physical_device_ordinal;
  // Queue ordinal, or UINT32_MAX when not applicable.
  uint32_t queue_ordinal;
  // Backend-assigned executable identifier, or 0 when not applicable.
  uint64_t executable_id;
  // Executable export ordinal, or UINT32_MAX when not applicable.
  uint32_t export_ordinal;
} iree_hal_device_event_source_t;

// Returns default event source attribution.
static inline iree_hal_device_event_source_t
iree_hal_device_event_source_default(void) {
  iree_hal_device_event_source_t source;
  memset(&source, 0, sizeof(source));
  source.physical_device_ordinal = UINT32_MAX;
  source.queue_ordinal = UINT32_MAX;
  source.export_ordinal = UINT32_MAX;
  return source;
}

// Bitfield specifying resolved event-site properties.
typedef uint64_t iree_hal_device_event_site_flags_t;
enum iree_hal_device_event_site_flag_bits_t {
  IREE_HAL_DEVICE_EVENT_SITE_FLAG_NONE = 0u,
};

// Optional resolved source-site metadata for a device event.
//
// Site records provide a target-neutral carrier for compiler, runtime, or
// executable metadata associated with an event payload. Events may still carry
// family-specific site ids in their payload; this record is the resolved view
// suitable for diagnostics.
typedef struct iree_hal_device_event_site_t {
  // Size of this record in bytes.
  uint32_t record_length;
  // ABI version of this site record.
  uint32_t abi_version;
  // Site-record flags.
  iree_hal_device_event_site_flags_t flags;
  // Producer-assigned site identifier.
  uint64_t site_id;
  // Borrowed source file path or name when available.
  iree_string_view_t source_file;
  // One-based start line when available, or 0.
  uint32_t start_line;
  // One-based start column when available, or 0.
  uint32_t start_column;
  // One-based end line when available, or 0.
  uint32_t end_line;
  // One-based end column when available, or 0.
  uint32_t end_column;
  // Borrowed function name when available.
  iree_string_view_t function_name;
  // Borrowed compiler operation name when available.
  iree_string_view_t operation_name;
  // Borrowed producer-specific site payload bytes when available.
  iree_const_byte_span_t producer_payload;
} iree_hal_device_event_site_t;

// Returns default event site values.
static inline iree_hal_device_event_site_t iree_hal_device_event_site_default(
    void) {
  iree_hal_device_event_site_t site;
  memset(&site, 0, sizeof(site));
  site.record_length = sizeof(site);
  site.abi_version = IREE_HAL_DEVICE_EVENT_SITE_ABI_VERSION_0;
  site.source_file = iree_string_view_empty();
  site.function_name = iree_string_view_empty();
  site.operation_name = iree_string_view_empty();
  site.producer_payload = iree_const_byte_span_empty();
  return site;
}

// HAL device event envelope passed to iree_hal_device_event_sink_t.
//
// Events and all referenced payload storage are borrowed and valid only for the
// duration of the sink callback. Sinks that retain event data beyond the
// callback must copy it.
typedef struct iree_hal_device_event_t {
  // Size of this record in bytes.
  uint32_t record_length;
  // ABI version of this event envelope.
  uint32_t abi_version;
  // Concrete event family.
  iree_hal_device_event_type_t type;
  // Producer-assigned severity.
  iree_hal_device_event_severity_t severity;
  // Event-family flags.
  iree_hal_device_event_flags_t flags;
  // Monotonic producer sequence when available, or 0.
  uint64_t sequence;
  // Host timestamp in nanoseconds when available, or 0.
  uint64_t host_time_ns;
  // Source attribution for the event producer.
  iree_hal_device_event_source_t source;
  // Optional resolved event-site metadata.
  const iree_hal_device_event_site_t* site;
  // Event-family payload selected by |type|.
  iree_const_byte_span_t payload;
  // Optional backend-native payload for advanced tools.
  iree_const_byte_span_t implementation_payload;
} iree_hal_device_event_t;

// Returns default event envelope values.
static inline iree_hal_device_event_t iree_hal_device_event_default(void) {
  iree_hal_device_event_t event;
  memset(&event, 0, sizeof(event));
  event.record_length = sizeof(event);
  event.abi_version = IREE_HAL_DEVICE_EVENT_ABI_VERSION_0;
  event.source = iree_hal_device_event_source_default();
  event.site = NULL;
  event.payload = iree_const_byte_span_empty();
  event.implementation_payload = iree_const_byte_span_empty();
  return event;
}

//===----------------------------------------------------------------------===//
// Device event payloads
//===----------------------------------------------------------------------===//

// ASAN access kind that triggered a report.
typedef uint32_t iree_hal_device_asan_access_kind_t;
enum iree_hal_device_asan_access_kind_bits_t {
  IREE_HAL_DEVICE_ASAN_ACCESS_KIND_UNKNOWN = 0u,
  IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ = 1u,
  IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE = 2u,
  IREE_HAL_DEVICE_ASAN_ACCESS_KIND_ATOMIC = 3u,
};

// Bitfield specifying ASAN report properties.
typedef uint32_t iree_hal_device_asan_report_flags_t;
enum iree_hal_device_asan_report_flag_bits_t {
  IREE_HAL_DEVICE_ASAN_REPORT_FLAG_NONE = 0u,
};

// HAL-level ASAN report payload.
typedef struct iree_hal_device_asan_report_t {
  // Size of this record in bytes.
  uint32_t record_length;
  // ABI version of this report payload.
  uint32_t abi_version;
  // Instrumented access kind that triggered the report.
  iree_hal_device_asan_access_kind_t access_kind;
  // ASAN report flags.
  iree_hal_device_asan_report_flags_t flags;
  // Application address that failed the ASAN check.
  uint64_t fault_address;
  // Access size in bytes.
  uint64_t access_length;
  // Compiler-assigned instrumentation site identifier.
  uint64_t site_id;
  // Shadow address consulted by the check, or 0 when unavailable.
  uint64_t shadow_address;
  // Shadow value observed by the check, or 0 when unavailable.
  uint64_t shadow_value;
  // Workgroup id that produced the report.
  uint32_t workgroup_id[3];
  // Workitem id that produced the report.
  uint32_t workitem_id[3];
  // Device-visible dispatch packet pointer, or 0 when unavailable.
  uint64_t source_dispatch_ptr;
} iree_hal_device_asan_report_t;

// UBSAN check kind that triggered a report.
typedef uint32_t iree_hal_device_ubsan_check_kind_t;
enum iree_hal_device_ubsan_check_kind_bits_t {
  IREE_HAL_DEVICE_UBSAN_CHECK_KIND_UNKNOWN = 0u,
  IREE_HAL_DEVICE_UBSAN_CHECK_KIND_INTEGER_OVERFLOW = 1u,
  IREE_HAL_DEVICE_UBSAN_CHECK_KIND_DIVIDE_BY_ZERO = 2u,
  IREE_HAL_DEVICE_UBSAN_CHECK_KIND_ALIGNMENT = 3u,
  IREE_HAL_DEVICE_UBSAN_CHECK_KIND_FLOAT_NAN_CONTRACT = 4u,
  IREE_HAL_DEVICE_UBSAN_CHECK_KIND_UNREACHABLE = 5u,
};

// Bitfield specifying UBSAN report properties.
typedef uint32_t iree_hal_device_ubsan_report_flags_t;
enum iree_hal_device_ubsan_report_flag_bits_t {
  IREE_HAL_DEVICE_UBSAN_REPORT_FLAG_NONE = 0u,
};

// HAL-level UBSAN report payload.
typedef struct iree_hal_device_ubsan_report_t {
  // Size of this record in bytes.
  uint32_t record_length;
  // ABI version of this report payload.
  uint32_t abi_version;
  // Check kind that triggered the report.
  iree_hal_device_ubsan_check_kind_t check_kind;
  // UBSAN report flags.
  iree_hal_device_ubsan_report_flags_t flags;
  // Compiler-assigned instrumentation site identifier.
  uint64_t site_id;
  // Check-specific first operand or value.
  uint64_t operand0;
  // Check-specific second operand or value.
  uint64_t operand1;
  // Workgroup id that produced the report.
  uint32_t workgroup_id[3];
  // Workitem id that produced the report.
  uint32_t workitem_id[3];
  // Device-visible dispatch packet pointer, or 0 when unavailable.
  uint64_t source_dispatch_ptr;
} iree_hal_device_ubsan_report_t;

// TSAN check kind that triggered a report.
typedef uint32_t iree_hal_device_tsan_check_kind_t;
enum iree_hal_device_tsan_check_kind_bits_t {
  IREE_HAL_DEVICE_TSAN_CHECK_KIND_UNKNOWN = 0u,
  IREE_HAL_DEVICE_TSAN_CHECK_KIND_DATA_RACE = 1u,
};

// TSAN memory space containing the observed address.
typedef uint32_t iree_hal_device_tsan_memory_space_t;
enum iree_hal_device_tsan_memory_space_bits_t {
  IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_UNKNOWN = 0u,
  IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_GLOBAL = 1u,
  IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_WORKGROUP = 2u,
  IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_PRIVATE = 3u,
};

// TSAN access kind participating in a race report.
typedef uint32_t iree_hal_device_tsan_access_kind_t;
enum iree_hal_device_tsan_access_kind_bits_t {
  IREE_HAL_DEVICE_TSAN_ACCESS_KIND_UNKNOWN = 0u,
  IREE_HAL_DEVICE_TSAN_ACCESS_KIND_READ = 1u,
  IREE_HAL_DEVICE_TSAN_ACCESS_KIND_WRITE = 2u,
  IREE_HAL_DEVICE_TSAN_ACCESS_KIND_READ_WRITE = 3u,
  IREE_HAL_DEVICE_TSAN_ACCESS_KIND_ATOMIC = 4u,
};

// Bitfield specifying TSAN report properties.
typedef uint32_t iree_hal_device_tsan_report_flags_t;
enum iree_hal_device_tsan_report_flag_bits_t {
  IREE_HAL_DEVICE_TSAN_REPORT_FLAG_NONE = 0u,
  // Current access was an atomic memory operation.
  IREE_HAL_DEVICE_TSAN_REPORT_FLAG_CURRENT_ATOMIC = 1u << 0,
  // Prior access was an atomic memory operation.
  IREE_HAL_DEVICE_TSAN_REPORT_FLAG_PRIOR_ATOMIC = 1u << 1,
  // Prior workitem id is stored as a linear local id in prior_workitem_id[0].
  IREE_HAL_DEVICE_TSAN_REPORT_FLAG_PRIOR_WORKITEM_LINEAR = 1u << 2,
  // Current workitem id is stored as a linear local id in
  // current_workitem_id[0].
  IREE_HAL_DEVICE_TSAN_REPORT_FLAG_CURRENT_WORKITEM_LINEAR = 1u << 3,
};

// HAL-level TSAN report payload.
typedef struct iree_hal_device_tsan_report_t {
  // Size of this record in bytes.
  uint32_t record_length;
  // ABI version of this report payload.
  uint32_t abi_version;
  // Check kind that triggered the report.
  iree_hal_device_tsan_check_kind_t check_kind;
  // TSAN report flags.
  iree_hal_device_tsan_report_flags_t flags;
  // Memory space containing |memory_address|.
  iree_hal_device_tsan_memory_space_t memory_space;
  // Access kind performed by the reporting workitem.
  iree_hal_device_tsan_access_kind_t current_access_kind;
  // Access kind previously observed for the same memory location.
  iree_hal_device_tsan_access_kind_t prior_access_kind;
  // Access length in bytes.
  uint32_t access_length;
  // Compiler-assigned instrumentation site identifier for the current access.
  uint64_t current_site_id;
  // Compiler-assigned instrumentation site identifier for the prior access.
  uint64_t prior_site_id;
  // Address or memory-space-relative byte offset that raced.
  uint64_t memory_address;
  // Shadow address consulted by the check, or 0 when unavailable.
  uint64_t shadow_address;
  // Shadow value observed by the check, or 0 when unavailable.
  uint64_t shadow_value;
  // Workgroup id that produced the current access report.
  uint32_t current_workgroup_id[3];
  // Workitem id that produced the current access report. When
  // IREE_HAL_DEVICE_TSAN_REPORT_FLAG_CURRENT_WORKITEM_LINEAR is set, element 0
  // is the linear local id and elements 1/2 are zero.
  uint32_t current_workitem_id[3];
  // Workgroup id that produced the prior access.
  uint32_t prior_workgroup_id[3];
  // Workitem id that produced the prior access. When
  // IREE_HAL_DEVICE_TSAN_REPORT_FLAG_PRIOR_WORKITEM_LINEAR is set, element 0 is
  // the linear local id and elements 1/2 are zero.
  uint32_t prior_workitem_id[3];
  // Device-visible dispatch packet pointer, or 0 when unavailable.
  uint64_t source_dispatch_ptr;
} iree_hal_device_tsan_report_t;
static_assert(sizeof(iree_hal_device_tsan_report_t) == 128,
              "TSAN report payload size is part of the HAL event ABI");

// Device printf stream classification.
typedef uint32_t iree_hal_device_printf_stream_t;
enum iree_hal_device_printf_stream_bits_t {
  IREE_HAL_DEVICE_PRINTF_STREAM_DEFAULT = 0u,
  IREE_HAL_DEVICE_PRINTF_STREAM_STDOUT = 1u,
  IREE_HAL_DEVICE_PRINTF_STREAM_STDERR = 2u,
};

// Bitfield specifying printf event properties.
typedef uint32_t iree_hal_device_printf_flags_t;
enum iree_hal_device_printf_flag_bits_t {
  IREE_HAL_DEVICE_PRINTF_FLAG_NONE = 0u,
};

// Device printf payload.
typedef struct iree_hal_device_printf_event_t {
  // Size of this record in bytes.
  uint32_t record_length;
  // ABI version of this printf payload.
  uint32_t abi_version;
  // Stream hint for the output.
  iree_hal_device_printf_stream_t stream;
  // Printf event flags.
  iree_hal_device_printf_flags_t flags;
  // Format string identifier, or 0 when |text| is already formatted.
  uint64_t format_id;
  // Borrowed already-formatted text when available.
  iree_string_view_t text;
  // Borrowed encoded arguments when available.
  iree_const_byte_span_t arguments;
} iree_hal_device_printf_event_t;

// Bitfield specifying driver failure event properties.
typedef uint32_t iree_hal_device_driver_failure_flags_t;
enum iree_hal_device_driver_failure_flag_bits_t {
  IREE_HAL_DEVICE_DRIVER_FAILURE_FLAG_NONE = 0u,
};

// Low-level backend/device/driver failure payload.
typedef struct iree_hal_device_driver_failure_event_t {
  // Size of this record in bytes.
  uint32_t record_length;
  // ABI version of this failure payload.
  uint32_t abi_version;
  // IREE status code corresponding to the failure.
  iree_status_code_t status_code;
  // Driver failure event flags.
  iree_hal_device_driver_failure_flags_t flags;
  // Backend-native result code, or 0 when unavailable.
  uint64_t backend_result_code;
  // Borrowed human-readable message when available.
  iree_string_view_t message;
} iree_hal_device_driver_failure_event_t;

//===----------------------------------------------------------------------===//
// iree_hal_device_event_sink_t
//===----------------------------------------------------------------------===//

// Receives one complete device event.
//
// Sinks may be invoked from driver-owned service, callback, or completion
// threads and may be invoked concurrently for different devices or physical
// devices. Implementations must not call back into the originating HAL device
// or submit/wait/destroy work from the callback. Slow processing should copy
// the event into application-owned storage and return.
typedef void(IREE_API_PTR* iree_hal_device_event_sink_fn_t)(
    void* user_data, const iree_hal_device_event_t* event);

// Value-type event sink copied into HAL devices at creation time.
typedef struct iree_hal_device_event_sink_t {
  // Callback receiving one complete event.
  iree_hal_device_event_sink_fn_t fn;
  // Opaque application-owned callback data.
  void* user_data;
} iree_hal_device_event_sink_t;

// Returns true when |sink| has a callback suitable for device creation.
static inline bool iree_hal_device_event_sink_is_valid(
    iree_hal_device_event_sink_t sink) {
  return sink.fn != NULL;
}

// Publishes |event| to |sink|.
static inline void iree_hal_device_event_sink_publish(
    iree_hal_device_event_sink_t sink, const iree_hal_device_event_t* event) {
  IREE_ASSERT_ARGUMENT(sink.fn);
  IREE_ASSERT_ARGUMENT(event);
  sink.fn(sink.user_data, event);
}

// Discard sink callback used by iree_hal_device_event_sink_discard().
static inline void iree_hal_device_event_sink_discard_callback(
    void* user_data, const iree_hal_device_event_t* event) {
  (void)user_data;
  (void)event;
}

// Returns a sink that discards every event.
static inline iree_hal_device_event_sink_t iree_hal_device_event_sink_discard(
    void) {
  iree_hal_device_event_sink_t sink;
  sink.fn = iree_hal_device_event_sink_discard_callback;
  sink.user_data = NULL;
  return sink;
}

// Returns a sink that formats known events to stderr.
IREE_API_EXPORT iree_hal_device_event_sink_t
iree_hal_device_event_sink_stderr(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DEVICE_EVENT_H_
