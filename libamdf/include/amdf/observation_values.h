// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_OBSERVATION_VALUES_H_
#define AMDF_OBSERVATION_VALUES_H_

#include "amdf/base.h"

/// Numeric representation shared by sensor and programmable-counter results.
typedef uint32_t amdf_observation_value_kind_t;
enum amdf_observation_value_kind_e {
  AMDF_OBSERVATION_VALUE_KIND_U64 = 1,
  AMDF_OBSERVATION_VALUE_KIND_I64 = 2,
  AMDF_OBSERVATION_VALUE_KIND_F64 = 3,
};

/// A number whose active member is selected by its descriptor's value kind.
/// Integer counters remain integers; converting them to double is a caller
/// presentation choice. In particular, subtract cumulative integers before
/// scaling instead of first rounding two large absolute counts to double.
typedef union amdf_observation_value_t {
  /// Exact unsigned native integer.
  uint64_t u64;
  /// Exact signed native integer.
  int64_t i64;
  /// Native IEEE-754 binary64 value.
  double f64;
} amdf_observation_value_t;

/// Unit of the scaled value, independent of its native numeric representation.
typedef uint32_t amdf_observation_unit_t;
enum amdf_observation_unit_e {
  /// Raw source quantity without an established physical interpretation.
  /// Telemetry never advertises this; native counter captures may preserve it.
  AMDF_OBSERVATION_UNIT_NATIVE = 0,
  AMDF_OBSERVATION_UNIT_COUNT = 1,
  AMDF_OBSERVATION_UNIT_HERTZ = 2,
  AMDF_OBSERVATION_UNIT_WATTS = 3,
  AMDF_OBSERVATION_UNIT_JOULES = 4,
  AMDF_OBSERVATION_UNIT_CELSIUS = 5,
  AMDF_OBSERVATION_UNIT_VOLTS = 6,
  AMDF_OBSERVATION_UNIT_AMPERES = 7,
  AMDF_OBSERVATION_UNIT_BYTES = 8,
  AMDF_OBSERVATION_UNIT_BYTES_PER_SECOND = 9,
  /// Dimensionless ratio: one represents 100 percent.
  AMDF_OBSERVATION_UNIT_RATIO = 10,
  AMDF_OBSERVATION_UNIT_SECONDS = 11,
  AMDF_OBSERVATION_UNIT_CYCLES = 12,
};

/// Meaning of a measurement, independent of when its host query happened.
typedef uint32_t amdf_observation_value_semantic_t;
enum amdf_observation_value_semantic_e {
  /// The native source does not establish its temporal aggregation.
  AMDF_OBSERVATION_VALUE_SEMANTIC_UNSPECIFIED = 0,
  /// A point measurement; its age can still be unknown.
  AMDF_OBSERVATION_VALUE_SEMANTIC_POINT = 1,
  /// An interval average; its window can still be unknown.
  AMDF_OBSERVATION_VALUE_SEMANTIC_AVERAGE = 2,
  /// A filtered measurement, distinct from a rectangular interval average.
  AMDF_OBSERVATION_VALUE_SEMANTIC_FILTERED = 3,
  /// A cumulative native count; reset and multiple wraps may be unobservable.
  AMDF_OBSERVATION_VALUE_SEMANTIC_CUMULATIVE = 4,
  /// A difference over the explicitly described collection interval.
  AMDF_OBSERVATION_VALUE_SEMANTIC_DELTA = 5,
};

/// Numeric interpretation reusable by telemetry and GPU counter layouts.
/// The physical value is native_value * scale_numerator / scale_denominator
/// in unit. The positive rational is exact; unknown scaling is not represented
/// by a successful physical-value descriptor. NATIVE uses 1/1 and promises
/// only the raw quantity, not a physical unit. Temperature is Celsius at the
/// native zero point, so this format needs no affine offset.
typedef struct amdf_observation_value_format_t {
  /// Active member of amdf_observation_value_t.
  amdf_observation_value_kind_t kind;
  /// Unit after applying the rational scale.
  amdf_observation_unit_t unit;
  /// Positive numerator of the exact conversion from native units.
  uint64_t scale_numerator;
  /// Positive denominator of the exact conversion from native units.
  uint64_t scale_denominator;
  /// Native aggregation semantics, independent of the host read bracket.
  amdf_observation_value_semantic_t semantic;
  /// Effective modulo width of an integer counter, not its storage width.
  /// Applies to cumulative/delta counters; zero for non-counters or unknown.
  uint32_t counter_bit_count;
} amdf_observation_value_format_t;

#endif  // AMDF_OBSERVATION_VALUES_H_
