// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-owned diagnostics over packetized target-low functions.
//
// This layer runs after an emission frame has joined schedule and allocation
// tables into emitter-order packet views. It gives optional target packages a
// cold diagnostic hook for authored low.func/low.op packets without teaching
// generic tools about target-specific descriptor IDs or performance hazards.

#ifndef LOOM_TARGET_LOW_PACKET_DIAGNOSTICS_H_
#define LOOM_TARGET_LOW_PACKET_DIAGNOSTICS_H_

#include "iree/base/api.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/packet.h"
#include "loom/error/emitter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_low_packet_diagnostic_context_t
    loom_target_low_packet_diagnostic_context_t;
typedef struct loom_target_low_packet_diagnostic_provider_t
    loom_target_low_packet_diagnostic_provider_t;

typedef enum loom_target_low_packet_diagnostic_flag_bits_e {
  // Emits target-owned packet selection and lowering remarks.
  LOOM_TARGET_LOW_PACKET_DIAGNOSTIC_TARGET_PACKETS = 1u << 0,
  // All target-low packet diagnostic flags known to this header.
  LOOM_TARGET_LOW_PACKET_DIAGNOSTIC_ALL =
      LOOM_TARGET_LOW_PACKET_DIAGNOSTIC_TARGET_PACKETS,
} loom_target_low_packet_diagnostic_flag_bits_t;
typedef uint32_t loom_target_low_packet_diagnostic_flags_t;

typedef iree_status_t (*loom_target_low_packet_diagnostic_try_packet_fn_t)(
    const loom_target_low_packet_diagnostic_provider_t* provider,
    loom_target_low_packet_diagnostic_context_t* context,
    const loom_low_packet_view_t* packet, bool* out_handled);

struct loom_target_low_packet_diagnostic_provider_t {
  // Stable provider name available to callbacks when emitting diagnostics.
  iree_string_view_t name;
  // Attempts to diagnose one packet. Sets |out_handled| false when the packet
  // does not belong to this provider.
  loom_target_low_packet_diagnostic_try_packet_fn_t try_diagnose_packet;
};

typedef struct loom_target_low_packet_diagnostic_provider_list_t {
  // Total number of values in the list.
  iree_host_size_t count;
  // Value list or NULL if no values.
  const loom_target_low_packet_diagnostic_provider_t* const* values;
} loom_target_low_packet_diagnostic_provider_list_t;

// Creates a packet diagnostic provider list from borrowed storage.
static inline loom_target_low_packet_diagnostic_provider_list_t
loom_target_low_packet_diagnostic_provider_list_make(
    const loom_target_low_packet_diagnostic_provider_t* const* values,
    iree_host_size_t count) {
  loom_target_low_packet_diagnostic_provider_list_t list = {
      /*.count=*/count,
      /*.values=*/values,
  };
  return list;
}

// Returns an empty packet diagnostic provider list.
static inline loom_target_low_packet_diagnostic_provider_list_t
loom_target_low_packet_diagnostic_provider_list_empty(void) {
  loom_target_low_packet_diagnostic_provider_list_t list = {0};
  return list;
}

// Returns true if |list| has no packet diagnostic providers.
static inline bool loom_target_low_packet_diagnostic_provider_list_is_empty(
    loom_target_low_packet_diagnostic_provider_list_t list) {
  return list.count == 0;
}

typedef struct loom_target_low_packet_diagnostics_options_t {
  // Optional target-specific packet diagnostic providers.
  loom_target_low_packet_diagnostic_provider_list_t provider_list;
  // Optional target-specific feedback diagnostics to emit.
  loom_target_low_packet_diagnostic_flags_t diagnostic_flags;
  // Structured diagnostic emitter for user-facing packet remarks.
  iree_diagnostic_emitter_t emitter;
} loom_target_low_packet_diagnostics_options_t;

typedef struct loom_target_low_packet_diagnostics_result_t {
  // Number of error diagnostics emitted.
  uint32_t error_count;
  // Number of warning diagnostics emitted.
  uint32_t warning_count;
  // Number of remark diagnostics emitted.
  uint32_t remark_count;
} loom_target_low_packet_diagnostics_result_t;

// Emits target-owned diagnostics for one packetized low.func.def.
iree_status_t loom_target_low_packet_diagnostics_emit_function(
    const loom_low_emission_frame_t* frame,
    const loom_target_low_packet_diagnostics_options_t* options,
    loom_target_low_packet_diagnostics_result_t* out_result);

// Returns the source module being diagnosed.
const loom_module_t* loom_target_low_packet_diagnostics_module(
    const loom_target_low_packet_diagnostic_context_t* context);

// Returns the scheduled table being diagnosed.
const loom_low_schedule_table_t* loom_target_low_packet_diagnostics_schedule(
    const loom_target_low_packet_diagnostic_context_t* context);

// Returns the allocation table being diagnosed.
const loom_low_allocation_table_t*
loom_target_low_packet_diagnostics_allocation(
    const loom_target_low_packet_diagnostic_context_t* context);

// Returns the optional feedback diagnostics requested by the caller.
loom_target_low_packet_diagnostic_flags_t
loom_target_low_packet_diagnostics_diagnostic_flags(
    const loom_target_low_packet_diagnostic_context_t* context);

// Emits ERR_BACKEND_018 for a target-owned packet decision.
iree_status_t loom_target_low_packet_diagnostics_record_packet(
    loom_target_low_packet_diagnostic_context_t* context,
    const loom_low_packet_view_t* packet, iree_string_view_t packet_category,
    iree_string_view_t decision);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_LOW_PACKET_DIAGNOSTICS_H_
