// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_TELEMETRY_H_
#define AMDF_TELEMETRY_H_

#include "amdf/observation.h"
#include "amdf/observation_values.h"

/// First caller-scheduled device telemetry contract.
#define AMDF_TELEMETRY_EXTENSION_VERSION_1 1u

/// Most recent telemetry extension described by this header.
#define AMDF_TELEMETRY_EXTENSION_VERSION_LATEST \
  AMDF_TELEMETRY_EXTENSION_VERSION_1

/// Opaque query identity, interpreted only with its live device and API
/// version. A changed native schema or interpretation has a different source
/// identity. IDs are not native handles, persistent identities, or
/// physical-domain IDs.
typedef uint64_t amdf_telemetry_source_id_t;

/// Opaque metric identity within a source. Never an enumeration array index.
typedef uint64_t amdf_telemetry_metric_id_t;

/// Stable meaning used for selection; independent of live metric/read IDs.
/// Domain, instance, and aggregation complete the meaning. For example, GPU
/// and CPU compute clocks share a kind, while edge and hotspot temperatures
/// never do. Unsupported meanings are absent, never replaced by similar units.
typedef uint32_t amdf_telemetry_metric_kind_t;
enum amdf_telemetry_metric_kind_e {
  /// Measured execution-engine clock, not a setpoint or timestamp timebase.
  AMDF_TELEMETRY_METRIC_CLOCK_COMPUTE = 1,
  /// Measured memory clock; not an effective DDR/GDDR transfer rate.
  AMDF_TELEMETRY_METRIC_CLOCK_MEMORY = 2,
  /// Measured memory-controller clock, distinct from the memory clock.
  AMDF_TELEMETRY_METRIC_CLOCK_MEMORY_CONTROLLER = 3,
  /// Measured SoC clock.
  AMDF_TELEMETRY_METRIC_CLOCK_SOC = 4,
  /// Measured fabric/interconnect clock.
  AMDF_TELEMETRY_METRIC_CLOCK_FABRIC = 5,
  /// Temperature at the native edge sensor.
  AMDF_TELEMETRY_METRIC_TEMPERATURE_EDGE = 6,
  /// Native hottest-point/junction temperature.
  AMDF_TELEMETRY_METRIC_TEMPERATURE_HOTSPOT = 7,
  /// Memory temperature, with the measured region given by domain/instance.
  AMDF_TELEMETRY_METRIC_TEMPERATURE_MEMORY = 8,
  /// Temperature without an established edge/hotspot/memory interpretation.
  AMDF_TELEMETRY_METRIC_TEMPERATURE = 9,
  /// Observed native power of the specified domain, never a configured cap.
  /// Firmware can estimate consumption from activity instead of a power meter.
  AMDF_TELEMETRY_METRIC_POWER = 10,
  /// Measured energy of the specified domain; format gives time aggregation.
  AMDF_TELEMETRY_METRIC_ENERGY = 11,
  /// Compute activity ratio for the specified native measurement interval.
  AMDF_TELEMETRY_METRIC_ACTIVITY_COMPUTE = 12,
  /// Memory activity ratio; not inferred bandwidth or occupancy.
  AMDF_TELEMETRY_METRIC_ACTIVITY_MEMORY = 13,
  /// Measured traffic entering the specified domain.
  AMDF_TELEMETRY_METRIC_TRAFFIC_READ = 14,
  /// Measured traffic leaving the specified domain.
  AMDF_TELEMETRY_METRIC_TRAFFIC_WRITE = 15,
  /// Residency in a native active power state, such as CPU C0. This does not
  /// establish useful instruction throughput or execution-unit utilization.
  AMDF_TELEMETRY_METRIC_ACTIVITY_ACTIVE = 16,
};

/// Diagnostic provenance, separate from the portable metric meaning.
/// Strings are immutable until library unload; no native resource is retained.
/// A capture copies these strings, the metric's typed metadata, and its device
/// identity. Live query IDs alone do not make a persistent measurement record.
typedef struct amdf_telemetry_origin_t {
  /// Native interface key, such as "linux.amdgpu.gpu_metrics".
  const char* interface_name;
  /// Exact native layout/interpretation key, such as "3.0" or "hwmon-abi".
  const char* schema;
  /// Native member/sensor key; instance indices remain separate typed fields.
  const char* field;
  /// Native array-member index, or UINT32_MAX for a scalar. This is not a
  /// physical-domain ordinal and need not equal domain_instance.
  uint32_t field_index;
  /// Reserved; zero.
  uint32_t reserved;
} amdf_telemetry_origin_t;

/// Region measured by a sensor, not the execution engine used to access it.
/// Domains can overlap: board, package, and engine powers are not additive.
/// These categories do not establish identity across separate devices.
typedef uint32_t amdf_telemetry_domain_kind_t;
enum amdf_telemetry_domain_kind_e {
  AMDF_TELEMETRY_DOMAIN_UNKNOWN = 0,
  AMDF_TELEMETRY_DOMAIN_BOARD = 1,
  AMDF_TELEMETRY_DOMAIN_PACKAGE = 2,
  AMDF_TELEMETRY_DOMAIN_GPU = 3,
  AMDF_TELEMETRY_DOMAIN_CPU = 4,
  AMDF_TELEMETRY_DOMAIN_CPU_CORE = 5,
  AMDF_TELEMETRY_DOMAIN_NPU = 6,
  AMDF_TELEMETRY_DOMAIN_NPU_COLUMN = 7,
  AMDF_TELEMETRY_DOMAIN_SOC = 8,
  AMDF_TELEMETRY_DOMAIN_MEMORY_CONTROLLER = 9,
  AMDF_TELEMETRY_DOMAIN_MEMORY = 10,
  /// A firmware power-management group, potentially spanning APU and dGPU.
  /// The descriptor describes the established group; not a GPU-only reading.
  AMDF_TELEMETRY_DOMAIN_POWER_BUDGET = 11,
  AMDF_TELEMETRY_DOMAIN_SYSTEM = 12,
};

/// Aggregate across a domain's instances instead of one native instance.
#define AMDF_TELEMETRY_ALL_INSTANCES UINT32_MAX

/// Lifetime of a native domain-instance identity within one query source.
/// Neither scope establishes identity across sources or separate devices.
typedef uint32_t amdf_telemetry_instance_scope_t;
enum amdf_telemetry_instance_scope_e {
  /// The ordinal names the same native instance across reads on this device.
  AMDF_TELEMETRY_INSTANCE_SCOPE_DEVICE = 0,
  /// The ordinal pairs measurements only within one device_read result.
  /// Native compaction can assign it to a different instance on the next read.
  /// Matching source timestamps do not establish identity across calls.
  AMDF_TELEMETRY_INSTANCE_SCOPE_SNAPSHOT = 1,
};

/// Cost and consistency facts for one source read.
typedef uint32_t amdf_telemetry_source_flags_t;
enum amdf_telemetry_source_flag_bits_e {
  /// The explicit native query may block; no deadline or readiness guarantee.
  AMDF_TELEMETRY_SOURCE_FLAG_NATIVE_MAY_BLOCK = 1u << 0,
  /// The native read can wake a runtime-suspended device.
  AMDF_TELEMETRY_SOURCE_FLAG_MAY_WAKE_DEVICE = 1u << 1,
  /// Values come from one native table/query result, not sequential queries.
  /// This does not establish simultaneous sensor acquisition or equal age.
  AMDF_TELEMETRY_SOURCE_FLAG_NATIVE_SNAPSHOT = 1u << 2,
};

/// Complete cold-path description of one metric and its query source.
/// Names are NUL-terminated immutable strings valid until library unload.
/// They require no release. Persisting a descriptor requires copying strings.
/// Source metadata is identical for all descriptors with the same source ID.
/// Fixed stride within extension v1; subsequent incompatible records use a
/// new negotiated extension version rather than changing an array's stride.
typedef struct amdf_telemetry_metric_info_t {
  /// Source accepted by device_read; nonzero and device-scoped.
  amdf_telemetry_source_id_t source_id;
  /// Stable diagnostic source name, never a platform dispatch selector.
  const char* source_name;
  /// Explicit read cost and native snapshot properties.
  amdf_telemetry_source_flags_t source_flags;
  /// Region measured, which may include CPU/NPU hardware on a GPU source.
  amdf_telemetry_domain_kind_t domain;
  /// Metric accepted by device_read for this source; nonzero.
  amdf_telemetry_metric_id_t metric_id;
  /// Native instance ordinal, or ALL_INSTANCES for an established aggregate.
  /// Scalar domains use zero. CPU indices are not OS logical-CPU numbers.
  uint32_t domain_instance;
  /// Identity lifetime of domain_instance within this source and domain.
  amdf_telemetry_instance_scope_t instance_scope;
  /// Portable meaning, distinct from the device-local read ID.
  amdf_telemetry_metric_kind_t kind;
  /// Native representation, physical units, scaling, and aggregation.
  amdf_observation_value_format_t format;
  /// Stable diagnostic metric name. Selection uses typed metadata or IDs.
  const char* name;
  /// Native provenance for saved captures and external documentation lookup.
  amdf_telemetry_origin_t origin;
} amdf_telemetry_metric_info_t;

/// Interpretation of a duration supplied with this particular reading.
typedef uint32_t amdf_telemetry_duration_kind_t;
enum amdf_telemetry_duration_kind_e {
  AMDF_TELEMETRY_DURATION_UNKNOWN = 0,
  /// Actual width of a rectangular averaging window, not its endpoint.
  AMDF_TELEMETRY_DURATION_AVERAGING_WINDOW = 1,
  /// Native filter time constant; not the span of a rectangular average.
  AMDF_TELEMETRY_DURATION_FILTER_TIME_CONSTANT = 2,
};

/// One value in the order of the caller's requested metric IDs.
typedef struct amdf_telemetry_value_t {
  /// OK for a valid native reading, including zero; NOT_FOUND for a native
  /// unavailable indication in an otherwise successful query. Query errors fail
  /// the entire call instead. Validity does not establish freshness/accuracy.
  amdf_status_t status;
  /// Native value, interpreted by the descriptor; zeroed when unavailable.
  amdf_observation_value_t value;
  /// Meaning of duration_ns for this reading.
  amdf_telemetry_duration_kind_t duration_kind;
  /// Reserved; zero.
  uint32_t reserved;
  /// Current native window/filter duration in ns; zero when kind is UNKNOWN.
  uint64_t duration_ns;
} amdf_telemetry_value_t;

/// Native timestamp namespace; equal frequencies do not imply equal epochs.
typedef uint32_t amdf_telemetry_clock_domain_t;
enum amdf_telemetry_clock_domain_e {
  AMDF_TELEMETRY_CLOCK_NONE = 0,
  /// The observation extension's system-wide profiling host clock.
  AMDF_TELEMETRY_CLOCK_HOST = 1,
  /// The auxiliary system clock in observation_clock_sample_t.system.
  /// Linux uses CLOCK_BOOTTIME. No correlation follows from read proximity.
  AMDF_TELEMETRY_CLOCK_SYSTEM = 2,
  /// A native clock local to this source and timestamp member (driver_time
  /// versus source_time). Reset/epoch continuity may be unreported; no
  /// synthetic generation or assumed relation between the two members.
  AMDF_TELEMETRY_CLOCK_SOURCE = 3,
};

/// Event identified by a native timestamp, when the source establishes it.
typedef uint32_t amdf_telemetry_timestamp_point_t;
enum amdf_telemetry_timestamp_point_e {
  AMDF_TELEMETRY_TIMESTAMP_POINT_UNSPECIFIED = 0,
  AMDF_TELEMETRY_TIMESTAMP_POINT_DRIVER_READ = 1,
  AMDF_TELEMETRY_TIMESTAMP_POINT_SENSOR_SAMPLE = 2,
  AMDF_TELEMETRY_TIMESTAMP_POINT_WINDOW_END = 3,
};

/// Optional native clock observation. NONE makes every other field zero.
typedef struct amdf_telemetry_timestamp_t {
  /// Timestamp namespace; SOURCE also includes the containing member's
  /// identity.
  amdf_telemetry_clock_domain_t domain;
  /// Native event timed; UNSPECIFIED preserves a timestamp of unknown meaning.
  amdf_telemetry_timestamp_point_t point;
  /// Counter units/width; zero frequency preserves unknown native tick units.
  amdf_observation_clock_info_t clock;
  /// Unconverted native timestamp; zero is valid when domain is not NONE.
  uint64_t value;
} amdf_telemetry_timestamp_t;

/// Time metadata for one explicit read, independent of sensor freshness.
typedef struct amdf_telemetry_sample_t {
  /// Observation host clock immediately before the native data read.
  uint64_t host_before;
  /// Observation host clock immediately after the native data read.
  uint64_t host_after;
  /// Timestamp attached by the driver, when provided by the source.
  amdf_telemetry_timestamp_t driver_time;
  /// Timestamp supplied by hardware/firmware, when provided by the source.
  amdf_telemetry_timestamp_t source_time;
} amdf_telemetry_sample_t;

/// Caller-scheduled device telemetry. Obtaining this table is inert.
///
/// Operations borrow an already-live device. They create no execution object,
/// GPU allocation, sampling session, worker, timer, or retained sensor handle.
/// They acquire no profiling authority and change no clock/power policy.
/// Enumeration is a cold native query; reads are explicit service calls. The
/// library performs no heap allocation or unbounded retry in either operation.
/// Native driver blocking and runtime-power costs remain visible below.
///
/// Calls use caller-owned output, retain no supplied pointers, and are safe
/// concurrently with other reads using disjoint outputs. The caller keeps the
/// device alive and excludes device teardown. Input and output ranges must not
/// overlap. There is no capture cleanup.
typedef struct amdf_telemetry_api_t {
  /// Size in bytes of this negotiated table.
  uint32_t structure_size;
  /// Telemetry extension version implemented by this table.
  uint32_t extension_version;

  /// Enumerates complete supported metric descriptors, grouped by source ID.
  ///
  /// Zero capacity queries the total, allows metrics=NULL, and returns OK.
  /// Insufficient nonzero capacity writes the available prefix and total
  /// out_count and returns BUFFER_TOO_SMALL, matching endpoint_enumerate. Other
  /// failures leave both outputs unchanged. With no supported route the count
  /// is zero and status is OK; permission, removal, and native errors are
  /// failures, not absence.
  ///
  /// A listed metric is a measurement, never a configured target, minimum,
  /// maximum, or enforced limit masquerading as an actual clock/power value.
  /// It has an established interpretation, not a promise of a
  /// currently available value. Optional unsupported routes contribute no
  /// descriptors. Values with unestablished scaling are not advertised as
  /// physical measurements. Enumeration may query/wake native firmware and
  /// block; it is not endpoint discovery or an inert capability accessor.
  ///
  /// IDs retain their meanings across calls on the same live device. A
  /// SNAPSHOT-scoped metric names a native slot, not a persistent instance.
  /// Fields sharing its domain and ordinal pair only within one read. New
  /// meanings/schema require new IDs; stale IDs fail instead of selecting a
  /// different route. No catalog cache, history, or registry is retained.
  amdf_status_t(AMDF_CALL* device_enumerate_metrics)(
      amdf_device_t* device, uint32_t capacity,
      amdf_telemetry_metric_info_t* metrics, uint32_t* out_count);

  /// Reads selected metrics from exactly one source, preserving request order.
  ///
  /// metric_count must be nonzero. metric_ids and values reference that many
  /// elements; out_sample is required. Duplicate IDs are allowed and return
  /// the same observation. The source is queried once, even when multiple
  /// fields are selected. Separate sysfs files are separate sources; multiple
  /// native queries never become one fabricated coherent snapshot.
  ///
  /// Input validation precedes native access. Unknown IDs return NOT_FOUND;
  /// a changed native schema returns VERSION_MISMATCH. No automatic fallback
  /// changes source, semantics, units, or read cost. All call-level failures
  /// leave values and out_sample unchanged. On success every value has a
  /// status; unavailable native fields are NOT_FOUND, never a successful zero.
  /// A native route that can substitute configured/default data for failed
  /// measurements without reporting that distinction cannot supply a metric
  /// through this API. An uncertainty flag would not make it a measurement.
  ///
  /// host_before/after bracket the explicit query, not sensor acquisition.
  /// No fresh reading, queue completion, GPU visibility, or timestamp
  /// correlation is implied. Native stamps/filter durations are preserved
  /// where established; absent facts remain absent. Reads can return the same
  /// cached observation repeatedly. No sleep, wait-for-change, background
  /// polling, hidden retry, or allocation occurs in library code.
  amdf_status_t(AMDF_CALL* device_read)(
      amdf_device_t* device, amdf_telemetry_source_id_t source_id,
      uint32_t metric_count, const amdf_telemetry_metric_id_t* metric_ids,
      amdf_telemetry_value_t* values, amdf_telemetry_sample_t* out_sample);
} amdf_telemetry_api_t;

#endif  // AMDF_TELEMETRY_H_
