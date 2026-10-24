// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/matrix_result_wait.h"

#include <string.h>

#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/cfg_dataflow.h"
#include "loom/util/cfg_graph.h"

static_assert(sizeof(loom_amdgpu_matrix_result_identity_t) == 8,
              "forwarding identity equality requires a padding-free value");
static_assert(sizeof(loom_amdgpu_matrix_result_source_t) == 36,
              "matrix register windows must fit the retained source row");

// A consumer can exclude one full-range forwarding identity. The strongest
// two distinct identities therefore answer every partial-overlap query. The
// exact-range floors are independent: forwarding can require *more* progress
// than partial overlap, notably for two-pass SGEMM consumed by XDL.
enum {
  LOOM_AMDGPU_MATRIX_RESULT_ORDINARY_COUNT = 3,
  LOOM_AMDGPU_MATRIX_RESULT_FORWARDING_START = 3,
  LOOM_AMDGPU_MATRIX_RESULT_FLOOR_START = 9,
  LOOM_AMDGPU_MATRIX_RESULT_ACCUMULATOR_READ = 11,
  LOOM_AMDGPU_MATRIX_RESULT_RELEASE_COUNT = 12,
  LOOM_AMDGPU_MATRIX_RESULT_FRONTIER_CHUNK_CAPACITY = 32,
};

typedef struct loom_amdgpu_matrix_result_release_t {
  // Absolute instruction position releasing this obligation; zero is empty.
  uint64_t position;
  // Index of the retained source establishing this release.
  uint32_t source;
} loom_amdgpu_matrix_result_release_t;

typedef struct loom_amdgpu_matrix_result_active_t {
  // Maxima for ordinary uses, three forwarding pairs, two exact floors and
  // the accumulator input's read window.
  loom_amdgpu_matrix_result_release_t
      releases[LOOM_AMDGPU_MATRIX_RESULT_RELEASE_COUNT];
  // Traversal epoch owning this physical register's state.
  uint32_t epoch;
} loom_amdgpu_matrix_result_active_t;

typedef struct loom_amdgpu_matrix_result_residuals_t {
  // Source ordinal for each release channel.
  uint32_t sources[LOOM_AMDGPU_MATRIX_RESULT_RELEASE_COUNT];
  // Instruction slots still required at block exit; zero is empty. Separate
  // arrays avoid three padding bytes per channel in every sparse row.
  uint8_t remaining[LOOM_AMDGPU_MATRIX_RESULT_RELEASE_COUNT];
} loom_amdgpu_matrix_result_residuals_t;

typedef struct loom_amdgpu_matrix_result_frontier_t {
  // Next physical register in this block's sparse exit.
  struct loom_amdgpu_matrix_result_frontier_t* next;
  // Physical register index in the disjoint VGPR/AGPR space.
  uint32_t physical_index;
  // Residual windows projected to the block's exit position.
  loom_amdgpu_matrix_result_residuals_t releases;
} loom_amdgpu_matrix_result_frontier_t;

typedef struct loom_amdgpu_matrix_result_frontier_chunk_t {
  // Number of initialized frontier records.
  uint16_t count;
  // Stable sparse rows shared by static solving and final refinement.
  loom_amdgpu_matrix_result_frontier_t
      rows[LOOM_AMDGPU_MATRIX_RESULT_FRONTIER_CHUNK_CAPACITY];
} loom_amdgpu_matrix_result_frontier_chunk_t;

typedef struct loom_amdgpu_matrix_result_block_t {
  // Conservative exit, refined in place after final planning visits the block.
  loom_amdgpu_matrix_result_frontier_t* outgoing;
  // Guaranteed native issue progress, saturated at UINT8_MAX.
  uint8_t minimum_instruction_count;
} loom_amdgpu_matrix_result_block_t;

struct loom_amdgpu_matrix_result_wait_t {
  // Immutable schedule supplying final packet order and the common CFG.
  const loom_low_schedule_table_t* schedule;
  // Immutable physical allocation supplying producer and consumer storage.
  const loom_low_allocation_table_t* allocation;
  // Target's qualified result-separation policy.
  loom_amdgpu_matrix_wait_profile_t profile;
  // Arena spanning static collection and final wait placement.
  iree_arena_allocator_t* arena;
  // Exact-sized immutable source stream in scheduled packet order.
  loom_amdgpu_matrix_result_source_t* sources;
  // Number of sources established by classification.
  iree_host_size_t source_count;
  // Next source to collect or consume in the current traversal.
  iree_host_size_t source_cursor;
  // Sparse block summaries, absent for a single-block function.
  loom_amdgpu_matrix_result_block_t* blocks;
  // Dense active scratch indexed by physical register.
  loom_amdgpu_matrix_result_active_t* active;
  // Physical indices initialized in the active traversal epoch.
  uint32_t* active_indices;
  // Number of initialized entries in active_indices.
  uint32_t active_count;
  // Active traversal epoch, independent of issue progress.
  uint32_t epoch;
  // Absolute instruction position within the active block.
  uint64_t position;
  // Physical register count across both banks.
  uint32_t physical_count;
  // Start of the AGPR bank, immediately after the allocated VGPR extent.
  uint32_t agpr_base;
  // Current region block index.
  uint16_t block_index;
  // Descriptor-set-local accumulator register class.
  uint8_t agpr_reg_class_id;
  // Dense index for the currently edited sparse block exit.
  loom_amdgpu_matrix_result_frontier_t** frontier_index;
  // Stable block-owned head represented by frontier_index.
  loom_amdgpu_matrix_result_frontier_t** indexed_frontier;
  // Current chunk of stable sparse frontier rows.
  loom_amdgpu_matrix_result_frontier_chunk_t* frontier_chunk;
};

static void loom_amdgpu_matrix_result_reset(
    loom_amdgpu_matrix_result_wait_t* state) {
  if (++state->epoch == 0) {
    memset(state->active, 0, state->physical_count * sizeof(*state->active));
    state->epoch = 1;
  }
  state->active_count = 0;
  state->position = 0;
}

static loom_amdgpu_matrix_result_active_t* loom_amdgpu_matrix_result_active(
    loom_amdgpu_matrix_result_wait_t* state, uint32_t physical_index) {
  loom_amdgpu_matrix_result_active_t* active = &state->active[physical_index];
  if (active->epoch != state->epoch) {
    *active = (loom_amdgpu_matrix_result_active_t){.epoch = state->epoch};
    state->active_indices[state->active_count++] = physical_index;
  }
  return active;
}

static bool loom_amdgpu_matrix_result_same_identity(
    const loom_amdgpu_matrix_result_wait_t* state, uint32_t lhs, uint32_t rhs) {
  return memcmp(&state->sources[lhs].identity, &state->sources[rhs].identity,
                sizeof(loom_amdgpu_matrix_result_identity_t)) == 0;
}

static void loom_amdgpu_matrix_result_merge_release(
    loom_amdgpu_matrix_result_release_t* target,
    loom_amdgpu_matrix_result_release_t source) {
  if (source.position > target->position) {
    *target = source;
  }
}

static void loom_amdgpu_matrix_result_merge_pair(
    const loom_amdgpu_matrix_result_wait_t* state,
    loom_amdgpu_matrix_result_release_t pair[2],
    loom_amdgpu_matrix_result_release_t source) {
  if (source.position == 0) {
    return;
  }
  const bool same_first =
      pair[0].position != 0 && loom_amdgpu_matrix_result_same_identity(
                                   state, pair[0].source, source.source);
  if (source.position > pair[0].position) {
    if (!same_first) {
      pair[1] = pair[0];
    }
    pair[0] = source;
  } else if (!same_first && source.position > pair[1].position) {
    pair[1] = source;
  }
}

static void loom_amdgpu_matrix_result_publish(
    loom_amdgpu_matrix_result_wait_t* state,
    const loom_amdgpu_matrix_result_source_t* source) {
  const uint32_t source_index = (uint32_t)(source - state->sources);
  for (uint32_t i = 0; i < source->identity.count; ++i) {
    loom_amdgpu_matrix_result_release_t* releases =
        loom_amdgpu_matrix_result_active(state, source->identity.base + i)
            ->releases;
    for (unsigned use = 0; use < 3; ++use) {
      loom_amdgpu_matrix_result_merge_release(
          &releases[use],
          (loom_amdgpu_matrix_result_release_t){
              .position = state->position + source->ordinary[use],
              .source = source_index,
          });
      if (source->accumulator[use] != 0) {
        loom_amdgpu_matrix_result_merge_pair(
            state,
            &releases[LOOM_AMDGPU_MATRIX_RESULT_FORWARDING_START + use * 2],
            (loom_amdgpu_matrix_result_release_t){
                .position = state->position + source->accumulator[use],
                .source = source_index,
            });
      }
    }
  }
  if (source->floor_ordinal != 0) {
    loom_amdgpu_matrix_result_release_t* releases =
        loom_amdgpu_matrix_result_active(state, source->identity.base)
            ->releases;
    loom_amdgpu_matrix_result_merge_release(
        &releases[LOOM_AMDGPU_MATRIX_RESULT_FLOOR_START +
                  source->floor_ordinal - 1],
        (loom_amdgpu_matrix_result_release_t){
            .position = state->position + source->floor_ordinal * 2,
            .source = source_index,
        });
  }
  if (source->accumulator_read != 0) {
    for (uint32_t i = 0; i < source->identity.count; ++i) {
      loom_amdgpu_matrix_result_merge_release(
          &loom_amdgpu_matrix_result_active(state, source->accumulator_base + i)
               ->releases[LOOM_AMDGPU_MATRIX_RESULT_ACCUMULATOR_READ],
          (loom_amdgpu_matrix_result_release_t){
              .position = state->position + source->accumulator_read,
              .source = source_index,
          });
    }
  }
}

static void loom_amdgpu_matrix_result_index_frontier(
    loom_amdgpu_matrix_result_wait_t* state,
    loom_amdgpu_matrix_result_frontier_t** frontier) {
  if (frontier == state->indexed_frontier) {
    return;
  }
  if (state->indexed_frontier != NULL) {
    for (loom_amdgpu_matrix_result_frontier_t* row = *state->indexed_frontier;
         row != NULL; row = row->next) {
      state->frontier_index[row->physical_index] = NULL;
    }
  }
  for (loom_amdgpu_matrix_result_frontier_t* row = *frontier; row != NULL;
       row = row->next) {
    state->frontier_index[row->physical_index] = row;
  }
  state->indexed_frontier = frontier;
}

static void loom_amdgpu_matrix_result_merge_incoming(
    loom_amdgpu_matrix_result_wait_t* state,
    const loom_amdgpu_matrix_result_frontier_t* frontier) {
  for (; frontier != NULL; frontier = frontier->next) {
    loom_amdgpu_matrix_result_active_t* active =
        &state->active[frontier->physical_index];
    if (active->epoch != state->epoch) {
      active->epoch = state->epoch;
      state->active_indices[state->active_count++] = frontier->physical_index;
      for (unsigned i = 0; i < LOOM_AMDGPU_MATRIX_RESULT_RELEASE_COUNT; ++i) {
        active->releases[i] = (loom_amdgpu_matrix_result_release_t){
            .position = frontier->releases.remaining[i],
            .source = frontier->releases.sources[i],
        };
      }
      continue;
    }
    loom_amdgpu_matrix_result_release_t* releases = active->releases;
    for (unsigned i = 0; i < LOOM_AMDGPU_MATRIX_RESULT_RELEASE_COUNT; ++i) {
      const loom_amdgpu_matrix_result_release_t release = {
          .position = frontier->releases.remaining[i],
          .source = frontier->releases.sources[i],
      };
      if (i >= LOOM_AMDGPU_MATRIX_RESULT_FORWARDING_START &&
          i < LOOM_AMDGPU_MATRIX_RESULT_FLOOR_START) {
        const unsigned pair =
            LOOM_AMDGPU_MATRIX_RESULT_FORWARDING_START +
            ((i - LOOM_AMDGPU_MATRIX_RESULT_FORWARDING_START) & ~1u);
        loom_amdgpu_matrix_result_merge_pair(state, &releases[pair], release);
      } else {
        loom_amdgpu_matrix_result_merge_release(&releases[i], release);
      }
    }
  }
}

static bool loom_amdgpu_matrix_result_project(
    const loom_amdgpu_matrix_result_wait_t* state, uint32_t physical_index,
    loom_amdgpu_matrix_result_residuals_t* residuals) {
  bool live = false;
  const loom_amdgpu_matrix_result_active_t* active =
      &state->active[physical_index];
  for (unsigned i = 0; i < LOOM_AMDGPU_MATRIX_RESULT_RELEASE_COUNT; ++i) {
    const loom_amdgpu_matrix_result_release_t release = active->releases[i];
    const uint8_t remaining =
        release.position > state->position
            ? (uint8_t)(release.position - state->position)
            : 0;
    residuals->sources[i] = release.source;
    residuals->remaining[i] = remaining;
    live |= remaining != 0;
  }
  return live;
}

static bool loom_amdgpu_matrix_result_merge_residuals(
    const loom_amdgpu_matrix_result_wait_t* state,
    const loom_amdgpu_matrix_result_residuals_t* source,
    loom_amdgpu_matrix_result_residuals_t* target) {
  bool changed = false;
  for (unsigned i = 0; i < LOOM_AMDGPU_MATRIX_RESULT_RELEASE_COUNT; ++i) {
    if (i >= LOOM_AMDGPU_MATRIX_RESULT_FORWARDING_START &&
        i < LOOM_AMDGPU_MATRIX_RESULT_FLOOR_START) {
      loom_amdgpu_matrix_result_release_t pair[2] = {
          {.position = target->remaining[i], .source = target->sources[i]},
          {.position = target->remaining[i + 1],
           .source = target->sources[i + 1]},
      };
      for (unsigned j = 0; j < 2; ++j) {
        loom_amdgpu_matrix_result_merge_pair(
            state, pair,
            (loom_amdgpu_matrix_result_release_t){
                .position = source->remaining[i + j],
                .source = source->sources[i + j],
            });
      }
      for (unsigned j = 0; j < 2; ++j) {
        changed |=
            target->remaining[i + j] != pair[j].position ||
            (pair[j].position != 0 && target->sources[i + j] != pair[j].source);
        target->sources[i + j] = pair[j].source;
        target->remaining[i + j] = (uint8_t)pair[j].position;
      }
      ++i;
    } else if (source->remaining[i] > target->remaining[i]) {
      target->sources[i] = source->sources[i];
      target->remaining[i] = source->remaining[i];
      changed = true;
    }
  }
  return changed;
}

static iree_status_t loom_amdgpu_matrix_result_merge_row(
    loom_amdgpu_matrix_result_wait_t* state,
    loom_amdgpu_matrix_result_frontier_t** frontier, uint32_t physical_index,
    const loom_amdgpu_matrix_result_residuals_t* residuals, bool* out_changed) {
  loom_amdgpu_matrix_result_frontier_t* row =
      state->frontier_index[physical_index];
  if (row != NULL) {
    *out_changed |= loom_amdgpu_matrix_result_merge_residuals(state, residuals,
                                                              &row->releases);
    return iree_ok_status();
  }
  if (state->frontier_chunk == NULL ||
      state->frontier_chunk->count ==
          LOOM_AMDGPU_MATRIX_RESULT_FRONTIER_CHUNK_CAPACITY) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(state->arena,
                                             sizeof(*state->frontier_chunk),
                                             (void**)&state->frontier_chunk));
    state->frontier_chunk->count = 0;
  }
  row = &state->frontier_chunk->rows[state->frontier_chunk->count++];
  row->next = *frontier;
  row->physical_index = physical_index;
  row->releases = *residuals;
  *frontier = row;
  state->frontier_index[physical_index] = row;
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_matrix_result_merge_outgoing(
    loom_amdgpu_matrix_result_wait_t* state,
    loom_amdgpu_matrix_result_frontier_t** frontier, bool* out_changed) {
  *out_changed = false;
  loom_amdgpu_matrix_result_index_frontier(state, frontier);
  for (uint32_t i = 0; i < state->active_count; ++i) {
    const uint32_t physical_index = state->active_indices[i];
    loom_amdgpu_matrix_result_residuals_t residuals;
    if (!loom_amdgpu_matrix_result_project(state, physical_index, &residuals)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_result_merge_row(
        state, frontier, physical_index, &residuals, out_changed));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_matrix_result_wait_allocate(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    loom_amdgpu_matrix_wait_profile_t profile, iree_host_size_t source_count,
    iree_arena_allocator_t* arena,
    loom_amdgpu_matrix_result_wait_t** out_state) {
  *out_state = NULL;
  if (profile == LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN || source_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(**out_state), (void**)out_state));
  **out_state = (loom_amdgpu_matrix_result_wait_t){
      .schedule = schedule,
      .allocation = allocation,
      .profile = profile,
      .arena = arena,
      .source_count = source_count,
  };
  loom_amdgpu_matrix_result_wait_t* state = *out_state;
  state->agpr_reg_class_id =
      loom_amdgpu_agpr_reg_class_id(state->schedule->target.descriptor_set);
  state->agpr_base = state->allocation->physical_extents
                         .ends_by_reg_class[LOOM_AMDGPU_REG_CLASS_ID_VGPR];
  state->physical_count =
      state->agpr_base + state->allocation->physical_extents
                             .ends_by_reg_class[state->agpr_reg_class_id];
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->arena, source_count,
                                                 sizeof(*state->sources),
                                                 (void**)&state->sources));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->physical_count, sizeof(*state->active),
      (void**)&state->active));
  memset(state->active, 0, state->physical_count * sizeof(*state->active));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->physical_count, sizeof(*state->active_indices),
      (void**)&state->active_indices));
  if (state->schedule->block_count > 1) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, state->schedule->block_count, sizeof(*state->blocks),
        (void**)&state->blocks));
    memset(state->blocks, 0,
           state->schedule->block_count * sizeof(*state->blocks));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, state->physical_count, sizeof(*state->frontier_index),
        (void**)&state->frontier_index));
    memset(state->frontier_index, 0,
           state->physical_count * sizeof(*state->frontier_index));
  }
  return iree_ok_status();
}

void loom_amdgpu_matrix_result_wait_begin_static_block(
    loom_amdgpu_matrix_result_wait_t* state, uint16_t block_index) {
  loom_amdgpu_matrix_result_reset(state);
  state->block_index = block_index;
}

void loom_amdgpu_matrix_result_wait_collect(
    loom_amdgpu_matrix_result_wait_t* state,
    const loom_low_packet_view_t* packet, bool matrix_source,
    uint8_t minimum_instruction_count) {
  state->position += minimum_instruction_count;
  if (!matrix_source) {
    return;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      state->schedule->target.descriptor_set;
  const loom_amdgpu_descriptor_traits_t traits =
      loom_amdgpu_descriptor_traits(descriptor_set, packet->descriptor);
  const loom_amdgpu_descriptor_ref_t reference =
      loom_amdgpu_descriptor_ref_for_descriptor(descriptor_set,
                                                packet->descriptor);
  const loom_amdgpu_matrix_contract_descriptor_t* contract =
      loom_amdgpu_matrix_contract_wait_state_descriptor_for_low_descriptor_ref(
          reference);
  const loom_low_allocation_assignment_t* result =
      loom_low_packet_result_assignment(state->allocation, packet, 0);
  loom_amdgpu_matrix_result_source_t* source =
      &state->sources[state->source_cursor++];
  *source = (loom_amdgpu_matrix_result_source_t){
      .packet_index = (uint32_t)packet->packet_index,
      .node_index = packet->node_index,
      .instruction =
          {
              .descriptor_ref = reference,
              .pass_count =
                  (uint8_t)loom_amdgpu_descriptor_matrix_pass_count(traits),
              .pipeline = iree_any_bit_set(
                              traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_DGEMM)
                              ? LOOM_AMDGPU_MATRIX_PIPELINE_DGEMM
                          : iree_any_bit_set(
                                traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_XDL)
                              ? LOOM_AMDGPU_MATRIX_PIPELINE_XDL
                              : LOOM_AMDGPU_MATRIX_PIPELINE_SGEMM,
          },
      .identity =
          {
              .base =
                  result->location_base +
                  (result->descriptor_reg_class_id == state->agpr_reg_class_id
                       ? state->agpr_base
                       : 0),
              .count = (uint16_t)result->location_count,
              .descriptor_ref =
                  state->profile == LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX90A
                      ? 0
                      : reference,
          },
      .accumulator_operand =
          contract->family == LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC ? 0 : 2,
  };
  const loom_low_allocation_assignment_t* accumulator =
      loom_low_packet_operand_assignment(state->allocation, packet,
                                         source->accumulator_operand);
  if (accumulator != NULL &&
      accumulator->location_kind ==
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    source->accumulator_base =
        accumulator->location_base +
        (accumulator->descriptor_reg_class_id == state->agpr_reg_class_id
             ? state->agpr_base
             : 0);
    // An aliased result's longer overwrite window already protects SrcC.
    if (source->accumulator_base != source->identity.base) {
      source->accumulator_read =
          (uint8_t)loom_amdgpu_matrix_wait_result_cycle_count(
              state->profile, source->instruction,
              LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_ACCUMULATOR_WRITE,
              (loom_amdgpu_matrix_instruction_t){0});
    }
  }
  for (unsigned i = 0; i < 3; ++i) {
    source->ordinary[i] = (uint8_t)loom_amdgpu_matrix_wait_result_cycle_count(
        state->profile, source->instruction,
        (loom_amdgpu_matrix_wait_result_use_t)(i + 1),
        (loom_amdgpu_matrix_instruction_t){0});
    source->accumulator[i] =
        (uint8_t)loom_amdgpu_matrix_wait_result_cycle_count(
            state->profile, source->instruction,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP,
            (loom_amdgpu_matrix_instruction_t){.pipeline = (uint8_t)i});
  }
  if (state->profile != LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX90A &&
      source->instruction.pass_count == 2) {
    source->floor_ordinal = 1;
  } else if (source->instruction.pipeline ==
                 LOOM_AMDGPU_MATRIX_PIPELINE_DGEMM &&
             source->instruction.pass_count == 4) {
    source->floor_ordinal = 2;
  }
  if (state->blocks != NULL) {
    loom_amdgpu_matrix_result_publish(state, source);
  }
}

iree_status_t loom_amdgpu_matrix_result_wait_end_static_block(
    loom_amdgpu_matrix_result_wait_t* state) {
  if (state->blocks == NULL) {
    return iree_ok_status();
  }
  loom_amdgpu_matrix_result_block_t* block = &state->blocks[state->block_index];
  block->minimum_instruction_count =
      (uint8_t)iree_min(state->position, UINT8_MAX);
  bool changed = false;
  return loom_amdgpu_matrix_result_merge_outgoing(state, &block->outgoing,
                                                  &changed);
}

static iree_status_t loom_amdgpu_matrix_result_transfer(void* user_data,
                                                        uint16_t source_block,
                                                        uint16_t target_block,
                                                        bool* out_changed) {
  loom_amdgpu_matrix_result_wait_t* state =
      (loom_amdgpu_matrix_result_wait_t*)user_data;
  loom_amdgpu_matrix_result_block_t* target = &state->blocks[target_block];
  loom_amdgpu_matrix_result_index_frontier(state, &target->outgoing);
  *out_changed = false;
  for (const loom_amdgpu_matrix_result_frontier_t* row =
           state->blocks[source_block].outgoing;
       row != NULL; row = row->next) {
    loom_amdgpu_matrix_result_residuals_t residuals;
    bool live = false;
    for (unsigned i = 0; i < LOOM_AMDGPU_MATRIX_RESULT_RELEASE_COUNT; ++i) {
      const uint8_t remaining = row->releases.remaining[i];
      residuals.sources[i] = row->releases.sources[i];
      residuals.remaining[i] =
          remaining > target->minimum_instruction_count
              ? remaining - target->minimum_instruction_count
              : 0;
      live |= residuals.remaining[i] != 0;
    }
    if (live) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_result_merge_row(
          state, &target->outgoing, row->physical_index, &residuals,
          out_changed));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_matrix_result_wait_finalize_static(
    loom_amdgpu_matrix_result_wait_t* state) {
  IREE_ASSERT_EQ(state->source_cursor, state->source_count);
  state->source_cursor = 0;
  if (state->blocks == NULL) {
    return iree_ok_status();
  }
  return loom_cfg_dataflow_solve(&state->schedule->cfg_graph,
                                 loom_amdgpu_matrix_result_transfer, state,
                                 state->arena);
}

void loom_amdgpu_matrix_result_wait_begin_block(
    loom_amdgpu_matrix_result_wait_t* state, uint16_t block_index) {
  loom_amdgpu_matrix_result_reset(state);
  state->block_index = block_index;
  if (state->blocks != NULL) {
    const loom_cfg_graph_t* graph = &state->schedule->cfg_graph;
    const loom_cfg_block_index_span_t predecessors =
        loom_cfg_graph_predecessors(graph, block_index);
    for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
      if (loom_cfg_graph_block_is_reachable(graph, predecessors.values[i])) {
        loom_amdgpu_matrix_result_merge_incoming(
            state, state->blocks[predecessors.values[i]].outgoing);
      }
    }
  }
}

const loom_amdgpu_matrix_result_source_t*
loom_amdgpu_matrix_result_wait_source_at(
    const loom_amdgpu_matrix_result_wait_t* state, uint32_t packet_index) {
  return state->source_cursor < state->source_count &&
                 state->sources[state->source_cursor].packet_index ==
                     packet_index
             ? &state->sources[state->source_cursor]
             : NULL;
}

static void loom_amdgpu_matrix_result_match_release(
    const loom_amdgpu_matrix_result_wait_t* state,
    loom_amdgpu_matrix_result_release_t release, uint64_t position,
    uint32_t base, uint32_t count, loom_amdgpu_matrix_wait_result_use_t use,
    loom_amdgpu_matrix_instruction_t consumer,
    loom_amdgpu_matrix_result_match_t* match) {
  if (release.position <= position ||
      release.position - position <= match->remaining) {
    return;
  }
  const loom_amdgpu_matrix_result_source_t* source =
      &state->sources[release.source];
  if (use == LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT &&
      (source->identity.base != base || source->identity.count != count)) {
    use = LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP;
  }
  *match = (loom_amdgpu_matrix_result_match_t){
      .source = source,
      .use = use,
      .required = loom_amdgpu_matrix_wait_result_cycle_count(
          state->profile, source->instruction, use, consumer),
      .remaining = (uint16_t)(release.position - position),
  };
}

static void loom_amdgpu_matrix_result_wait_match(
    const loom_amdgpu_matrix_result_wait_t* state, uint16_t register_class,
    uint32_t base, uint32_t count, uint64_t instruction_offset,
    loom_amdgpu_matrix_wait_result_use_t use,
    loom_amdgpu_matrix_instruction_t consumer,
    loom_amdgpu_matrix_result_match_t* match) {
  if (register_class != LOOM_AMDGPU_REG_CLASS_ID_VGPR &&
      register_class != state->agpr_reg_class_id) {
    return;
  }
  if (register_class == state->agpr_reg_class_id) {
    base += state->agpr_base;
  }
  const uint64_t position = state->position + instruction_offset;
  const bool accumulator =
      use == LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT;
  const loom_amdgpu_matrix_result_identity_t identity = {
      .base = base,
      .count = (uint16_t)count,
      .descriptor_ref =
          state->profile == LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX90A
              ? 0
              : consumer.descriptor_ref,
  };
  for (uint32_t i = 0; i < count; ++i) {
    const loom_amdgpu_matrix_result_active_t* active = &state->active[base + i];
    if (active->epoch != state->epoch) {
      continue;
    }
    const loom_amdgpu_matrix_result_release_t* release = NULL;
    if (!accumulator) {
      const unsigned channel =
          use == LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB
              ? 0
              : (unsigned)use - 1;
      release = &active->releases[channel];
    } else {
      release = &active->releases[LOOM_AMDGPU_MATRIX_RESULT_FORWARDING_START +
                                  consumer.pipeline * 2];
      if (release->position > position) {
        const loom_amdgpu_matrix_result_source_t* source =
            &state->sources[release->source];
        if (memcmp(&source->identity, &identity, sizeof(identity)) == 0) {
          ++release;
        }
      }
    }
    loom_amdgpu_matrix_result_match_release(state, *release, position, base,
                                            count, use, consumer, match);
    if (use == LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_WRITE) {
      loom_amdgpu_matrix_result_match_release(
          state, active->releases[LOOM_AMDGPU_MATRIX_RESULT_ACCUMULATOR_READ],
          position, base, count,
          LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_ACCUMULATOR_WRITE, consumer,
          match);
    }
    if (accumulator && i == 0) {
      // Generator validation establishes the only two exact-floor shapes.
      if (count == 4) {
        loom_amdgpu_matrix_result_match_release(
            state, active->releases[LOOM_AMDGPU_MATRIX_RESULT_FLOOR_START],
            position, base, count, use, consumer, match);
      }
      if (count == 2 &&
          consumer.pipeline == LOOM_AMDGPU_MATRIX_PIPELINE_DGEMM &&
          consumer.pass_count == 4) {
        loom_amdgpu_matrix_result_match_release(
            state, active->releases[LOOM_AMDGPU_MATRIX_RESULT_FLOOR_START + 1],
            position, base, count, use, consumer, match);
      }
    }
  }
}

static void loom_amdgpu_matrix_result_match_assignment(
    const loom_amdgpu_matrix_result_wait_t* state,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_matrix_wait_result_use_t use,
    loom_amdgpu_matrix_instruction_t consumer,
    loom_amdgpu_matrix_result_match_t* match) {
  if (assignment != NULL &&
      assignment->location_kind ==
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    loom_amdgpu_matrix_result_wait_match(
        state, assignment->descriptor_reg_class_id, assignment->location_base,
        assignment->location_count, 0, use, consumer, match);
  }
}

loom_amdgpu_matrix_result_match_t loom_amdgpu_matrix_result_wait_inspect(
    const loom_amdgpu_matrix_result_wait_t* state,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_matrix_result_source_t* source,
    const loom_amdgpu_structural_packet_info_t* structural,
    bool memory_consumer) {
  loom_amdgpu_matrix_result_match_t match = {0};
  if (structural->moves.count != 0) {
    for (iree_host_size_t i = 0; i < structural->moves.count; ++i) {
      const loom_low_move_t* move =
          &state->allocation->moves[structural->moves.start + i];
      for (uint32_t j = 0; j < loom_low_move_access_count(move); ++j) {
        const loom_low_move_location_t* read = loom_low_move_read(move, j);
        const loom_low_move_location_t* write = loom_low_move_write(move, j);
        if (read->location_kind ==
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
          loom_amdgpu_matrix_result_wait_match(
              state, read->descriptor_reg_class_id, read->location, 1, i,
              LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_VALU_READ,
              (loom_amdgpu_matrix_instruction_t){0}, &match);
        }
        if (write->location_kind ==
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
          loom_amdgpu_matrix_result_wait_match(
              state, write->descriptor_reg_class_id, write->location, 1, i,
              LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_WRITE,
              (loom_amdgpu_matrix_instruction_t){0}, &match);
        }
      }
    }
    return match;
  }
  if (packet->descriptor == NULL &&
      (loom_low_br_isa(packet->node->op) ||
       !iree_any_bit_set(structural->flags,
                         LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES))) {
    return match;
  }
  for (uint16_t i = 0; i < packet->node->operand_count; ++i) {
    const loom_amdgpu_matrix_wait_result_use_t use =
        source != NULL
            ? (i == source->accumulator_operand
                   ? LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT
                   : LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB)
            : (memory_consumer ? LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MEMORY_READ
                               : LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_VALU_READ);
    loom_amdgpu_matrix_result_match_assignment(
        state, loom_low_packet_operand_assignment(state->allocation, packet, i),
        use,
        source != NULL ? source->instruction
                       : (loom_amdgpu_matrix_instruction_t){0},
        &match);
  }
  if (source == NULL) {
    for (uint16_t i = 0; i < packet->node->result_count; ++i) {
      loom_amdgpu_matrix_result_match_assignment(
          state,
          loom_low_packet_result_assignment(state->allocation, packet, i),
          LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_WRITE,
          (loom_amdgpu_matrix_instruction_t){0}, &match);
    }
  }
  return match;
}

void loom_amdgpu_matrix_result_wait_advance(
    loom_amdgpu_matrix_result_wait_t* state, uint64_t instruction_count) {
  state->position += instruction_count;
}

void loom_amdgpu_matrix_result_wait_commit(
    loom_amdgpu_matrix_result_wait_t* state,
    const loom_amdgpu_matrix_result_source_t* source) {
  if (source != NULL) {
    loom_amdgpu_matrix_result_publish(state, source);
    ++state->source_cursor;
  }
}

void loom_amdgpu_matrix_result_wait_end_block(
    loom_amdgpu_matrix_result_wait_t* state) {
  if (state->blocks == NULL) {
    return;
  }
  // Actual waits only shorten conservative lifetimes. The static solve already
  // allocated every physical row final planning can retain.
  for (loom_amdgpu_matrix_result_frontier_t* row =
           state->blocks[state->block_index].outgoing;
       row != NULL; row = row->next) {
    if (state->active[row->physical_index].epoch == state->epoch) {
      loom_amdgpu_matrix_result_project(state, row->physical_index,
                                        &row->releases);
    } else {
      memset(&row->releases, 0, sizeof(row->releases));
    }
  }
}
