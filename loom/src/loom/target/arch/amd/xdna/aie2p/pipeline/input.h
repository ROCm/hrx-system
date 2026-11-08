// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_INPUT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_INPUT_H_

#include "loom/analysis/pipeline/continuations.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable source geometry relocated against an ordinary entry buffer.
typedef struct loom_aie2p_native_input_descriptor_t {
  // Next descriptor requiring invocation-time relocation.
  struct loom_aie2p_native_input_descriptor_t* next;
  // Shim owning this descriptor.
  const loom_aie2p_native_tile_t* tile;
  // Index in that shim's descriptor bank.
  uint16_t descriptor;
  // Published entry buffer ordinal.
  uint32_t binding;
  // First accessed byte within the selected buffer.
  uint64_t byte_offset;
  // Accessible envelope beginning at byte_offset.
  uint64_t byte_length;
  // Immutable descriptor words, excluding the relocated address pair.
  uint32_t words[8];
} loom_aie2p_native_input_descriptor_t;

typedef struct loom_aie2p_native_input_phase_t {
  // Retained source phase; loops never expand into descriptor lists.
  const loom_pipeline_transport_input_phase_t* source;
  // Local destination with independently owned record credits.
  const loom_aie2p_native_channel_t* destination;
  // Local DMA receiver and its startup window.
  loom_aie2p_native_distribution_endpoint_t receiver;
  // Immutable shim descriptor for this source address period.
  uint16_t descriptor;
  // Shim executions for one complete source address period.
  uint32_t repetitions;
  // Physical source geometry before framing and relocation.
  uint32_t words[8];
} loom_aie2p_native_input_phase_t;

// Queues and packet routes shared by nonoverlapping input occurrences on one
// admitted controller core. Descriptor geometry remains occurrence-owned.
typedef struct loom_aie2p_native_input_path_t {
  // Next physical path owned by the same instruction engine.
  struct loom_aie2p_native_input_path_t* next;
  // First occurrence supplying the fixed receiver topology and configuration.
  const struct loom_aie2p_native_input_t* first;
  // Last assigned occurrence in invocation-major selection order. Independent
  // inputs in that invocation cannot borrow this path's retirement boundary.
  const struct loom_aie2p_native_input_t* last;
  // Physical shim serving the stable caller buffers.
  loom_aie2p_native_tile_t* shim;
  // Memory-to-stream engine shared by all source phases.
  uint8_t input_engine;
  // Selected core-to-shim control packet class.
  uint8_t control_packet;
  // Tile-relative input task publication register.
  uint32_t input_queue;
} loom_aie2p_native_input_path_t;

typedef struct loom_aie2p_native_input_t {
  // Communication-only source worker.
  uint32_t worker;
  // Resident invocation owning receiver startup and terminal queue retirement.
  const loom_pipeline_invocation_t* invocation;
  // Complete phase counts from common transport analysis.
  const loom_pipeline_transport_input_t* source;
  // Existing instruction worker controlling this input.
  uint32_t controller;
  // Exact demand observations, or NULL when the complete immutable source is
  // admitted once at its controller's invocation entry.
  const loom_pipeline_continuation_t* continuation;
  // Native receiver and source descriptor selection per phase.
  loom_aie2p_native_input_phase_t* phases;
  // All phase receivers share one ordered physical queue and payload route.
  bool shared_receiver;
  // Physical queues and routes, independent of this occurrence's descriptors.
  loom_aie2p_native_input_path_t* path;
  // Private SSA bulk-position counter assigned during worker materialization.
  iree_host_size_t state_index;
  // Generated source-admission helper in the observing worker's context.
  loom_symbol_ref_t begin;
  // Child-entry receiver publication; root receivers start in configuration.
  loom_symbol_ref_t start;
  // Child-exit retirement of source and receiver queues.
  loom_symbol_ref_t retire;
  // Next input controlled by the same existing arithmetic worker.
  struct loom_aie2p_native_input_t* next;
} loom_aie2p_native_input_t;

// Admits stable caller-buffer phases before assigning instruction engines.
// No instruction worker or payload staging allocation is introduced.
iree_status_t loom_aie2p_native_select_input(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, iree_host_size_t worker,
    bool* out_eligible);

// Allocates descriptors and routes from the retained source geometry and its
// invocation controller.
iree_status_t loom_aie2p_native_route_input(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    loom_aie2p_native_input_t* input, uint32_t* next_source,
    uint8_t* next_controller, bool* out_valid);

// Emits the ordinary helper and read-site actions for one selected input.
iree_status_t loom_aie2p_native_prepare_input(
    loom_aie2p_native_context_t* context, loom_aie2p_native_input_t* input);
// Starts child receivers and either admits the complete source or initializes
// its one demand-position value. No state is required without a continuation.
iree_status_t loom_aie2p_native_initialize_input(
    loom_builder_t* builder, const loom_aie2p_native_input_t* input,
    loom_value_id_t* state);
iree_status_t loom_aie2p_native_emit_input(
    loom_builder_t* builder, const loom_aie2p_native_input_t* input,
    iree_host_size_t action, bool after, loom_value_id_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_INPUT_H_
