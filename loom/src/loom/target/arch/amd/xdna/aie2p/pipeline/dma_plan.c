// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_plan.h"

#include "loom/target/arch/amd/xdna/aie2p/array/binding.h"

const loom_aie2p_native_memory_layout_t loom_aie2p_compute_memory = {
    .descriptor_base =
        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD0_BUFFER_LENGTH,
    .descriptor_word_count = 6,
    .address_word = 0,
    .address =
        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD0_BASE_ADDRESS,
    .iteration_word = 4,
    .iteration =
        {
            .step =
                LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD4_ITERATION_STEP_SIZE,
            .wrap =
                LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD4_ITERATION_WRAP,
            .current =
                LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD4_ITERATION_CURRENT,
        },
    .lock_word = 5,
    .chain =
        {
            .word = 5,
            .next =
                LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD5_NEXT_BD,
            .enable =
                LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD5_USE_NEXT_BD,
        },
    .locks =
        {
            .acquire_id =
                LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD5_LOCK_ACQUIRE_ID,
            .acquire_value =
                LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD5_LOCK_ACQUIRE_VALUE,
            .acquire_enable =
                LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD5_LOCK_ACQUIRE_ENABLE,
            .release_id =
                LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD5_LOCK_RELEASE_ID,
            .release_value =
                LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD5_LOCK_RELEASE_VALUE,
            .valid =
                LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD5_VALID_BD,
        },
    .lock_value = LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_LOCK_VALUE_VALUE,
    .directions =
        {
            {
                .reset =
                    LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_CONTROL_RESET,
                .start =
                    LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_START_QUEUE_START_BD_ID,
                .repeat =
                    LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_START_QUEUE_REPEAT_COUNT,
                .status =
                    {
                        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_STATUS_TASK_QUEUE_SIZE,
                        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_STATUS_CHANNEL_RUNNING,
                        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_STATUS_STALLED_LOCK_ACQUIRE,
                        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_STATUS_STALLED_LOCK_RELEASE,
                        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_STATUS_STALLED_STREAM_BACKPRESSURE,
                        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_MM2S_STATUS_STALLED_COMPLETION_TOKEN,
                    },
            },
            {
                .reset =
                    LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_CONTROL_RESET,
                .start =
                    LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_START_QUEUE_START_BD_ID,
                .repeat =
                    LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_START_QUEUE_REPEAT_COUNT,
                .status =
                    {
                        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_STATUS_TASK_QUEUE_SIZE,
                        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_STATUS_CHANNEL_RUNNING,
                        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_STATUS_STALLED_LOCK_ACQUIRE,
                        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_STATUS_STALLED_LOCK_RELEASE,
                        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_STATUS_STALLED_STREAM_STARVATION,
                        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_CHANNEL_S2MM_STATUS_STALLED_TOKEN_OR_COUNT_FIFO,
                    },
            },
        },
};

const loom_aie2p_native_memory_layout_t loom_aie2p_sram_memory = {
    .descriptor_base =
        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD0_BUFFER_LENGTH,
    .descriptor_word_count = 8,
    .address_word = 1,
    .address = LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD1_BASE_ADDRESS,
    .iteration_word = 6,
    .iteration =
        {
            .step =
                LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD6_ITERATION_STEP_SIZE,
            .wrap =
                LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD6_ITERATION_WRAP,
            .current =
                LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD6_ITERATION_CURRENT,
        },
    .lock_word = 7,
    .chain =
        {
            .word = 1,
            .next = LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD1_NEXT_BD,
            .enable =
                LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD1_USE_NEXT_BD,
        },
    .locks =
        {
            .acquire_id =
                LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD7_LOCK_ACQUIRE_ID,
            .acquire_value =
                LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD7_LOCK_ACQUIRE_VALUE,
            .acquire_enable =
                LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD7_LOCK_ACQUIRE_ENABLE,
            .release_id =
                LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD7_LOCK_RELEASE_ID,
            .release_value =
                LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD7_LOCK_RELEASE_VALUE,
            .valid = LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD7_VALID_BD,
        },
    .lock_value = LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_LOCK_VALUE_VALUE,
    .local_address_base = 0x80000,
    .local_lock_base = 64,
    .directions =
        {
            {
                .reset =
                    LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_CONTROL_RESET,
                .start =
                    LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_START_QUEUE_START_BD_ID,
                .repeat =
                    LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_START_QUEUE_REPEAT_COUNT,
                .status =
                    {
                        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_STATUS_TASK_QUEUE_SIZE,
                        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_STATUS_CHANNEL_RUNNING,
                        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_STATUS_STALLED_LOCK_ACQUIRE,
                        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_STATUS_STALLED_LOCK_RELEASE,
                        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_STATUS_STALLED_STREAM_BACKPRESSURE,
                        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_STATUS_STALLED_COMPLETION_TOKEN,
                    },
            },
            {
                .reset =
                    LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_S2MM_CONTROL_RESET,
                .start =
                    LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_S2MM_START_QUEUE_START_BD_ID,
                .repeat =
                    LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_S2MM_START_QUEUE_REPEAT_COUNT,
                .status =
                    {
                        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_S2MM_STATUS_TASK_QUEUE_SIZE,
                        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_S2MM_STATUS_CHANNEL_RUNNING,
                        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_S2MM_STATUS_STALLED_LOCK_ACQUIRE,
                        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_S2MM_STATUS_STALLED_LOCK_RELEASE,
                        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_S2MM_STATUS_STALLED_STREAM_STARVATION,
                        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_S2MM_STATUS_STALLED_TOKEN_OR_COUNT_FIFO,
                    },
            },
        },
};

bool loom_aie2p_native_encode_descriptor(
    const loom_xdna_array_family_t* family,
    const loom_storage_geometry_t* geometry, const loom_xdna_dma_facts_t* dma,
    uint32_t record_bytes, uint32_t record_count, loom_xdna_tile_kind_t kind,
    uint32_t* words, iree_string_view_t* reason) {
  loom_aie2p_array_binding_plan_t layout = {0};
  if (!loom_aie2p_array_resolve_binding_transfer(
          geometry, geometry->rank, family, 0, false, 0, record_bytes,
          record_count, dma, &layout, reason)) {
    return false;
  }
  words[0] = layout.transfer_byte_length / dma->transfer_length_granularity -
             dma->transfer_length_offset;
  if (kind == LOOM_XDNA_TILE_KIND_SHIM_NOC) {
    words[4] = loom_xdna_register_field_encode_admitted(
        LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD4_BURST_LENGTH, 3);
    words[5] = loom_xdna_register_field_encode_admitted(
        LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD5_AXI_CACHE, 2);
    words[7] = loom_xdna_register_field_encode_admitted(
        LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_VALID_BD, 1);
    const loom_xdna_register_field_id_t step[] = {
        LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD3_D0_STEP_SIZE,
        LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD4_D1_STEP_SIZE,
        LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD5_D2_STEP_SIZE};
    const loom_xdna_register_field_id_t wrap[] = {
        LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD3_D0_WRAP,
        LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD4_D1_WRAP};
    for (uint8_t i = 0; i < layout.dma_dimension_count; ++i) {
      words[3 + i] |= loom_xdna_register_field_encode_admitted(
          step[i], layout.dma_dimensions[i].step_size - 1);
      if (i + 1 < layout.dma_dimension_count) {
        words[3 + i] |= loom_xdna_register_field_encode_admitted(
            wrap[i], layout.dma_dimensions[i].wrap);
      }
    }
  } else if (kind == LOOM_XDNA_TILE_KIND_MEMORY) {
    words[7] = loom_xdna_register_field_encode_admitted(
        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD7_VALID_BD, 1);
    const loom_xdna_register_field_id_t step[] = {
        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD2_D0_STEP_SIZE,
        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD3_D1_STEP_SIZE,
        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD4_D2_STEP_SIZE,
        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD5_D3_STEP_SIZE};
    const loom_xdna_register_field_id_t wrap[] = {
        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD2_D0_WRAP,
        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD3_D1_WRAP,
        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD4_D2_WRAP};
    for (uint8_t i = 0; i < layout.dma_dimension_count; ++i) {
      words[2 + i] |= loom_xdna_register_field_encode_admitted(
          step[i], layout.dma_dimensions[i].step_size - 1);
      if (i + 1 < layout.dma_dimension_count) {
        words[2 + i] |= loom_xdna_register_field_encode_admitted(
            wrap[i], layout.dma_dimensions[i].wrap);
      }
    }
  } else {
    words[5] = loom_xdna_register_field_encode_admitted(
        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD5_VALID_BD, 1);
    const loom_xdna_register_field_id_t step[] = {
        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD2_D0_STEP_SIZE,
        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD2_D1_STEP_SIZE,
        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD3_D2_STEP_SIZE};
    const loom_xdna_register_field_id_t wrap[] = {
        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD3_D0_WRAP,
        LOOM_XDNA_REGISTER_FIELD_COMPUTE_MEMORY_DMA_BD_WORD3_D1_WRAP};
    for (uint8_t i = 0; i < layout.dma_dimension_count; ++i) {
      words[i < 2 ? 2 : 3] |= loom_xdna_register_field_encode_admitted(
          step[i], layout.dma_dimensions[i].step_size - 1);
      if (i + 1 < layout.dma_dimension_count) {
        words[3] |= loom_xdna_register_field_encode_admitted(
            wrap[i], layout.dma_dimensions[i].wrap);
      }
    }
  }
  return true;
}

bool loom_aie2p_native_encode_ingress(const loom_xdna_array_family_t* family,
                                      const loom_storage_geometry_t* record,
                                      const loom_xdna_dma_facts_t* dma,
                                      uint32_t record_bytes,
                                      uint32_t byte_stride,
                                      uint32_t repetitions, uint32_t words[8],
                                      uint32_t* out_repetitions) {
  iree_string_view_t reason;
  if (byte_stride) {
    // Local descriptor lengths and FIFO credits delimit records independently
    // of the shim's transfer boundaries. Preserve the complete sequence shape
    // in one external descriptor when its native address fields can express it.
    loom_storage_geometry_t sequence = *record;
    bool representable = true;
    if (record->layout_kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE ||
        byte_stride != record_bytes) {
      const uint64_t stride_bits = (uint64_t)byte_stride * 8;
      representable = record->rank < LOOM_TYPE_MAX_RANK &&
                      stride_bits % record->element_bit_count == 0;
      if (representable) {
        memmove(sequence.axes + 1, sequence.axes,
                record->rank * sizeof(*sequence.axes));
        sequence.axes[0] = (loom_storage_geometry_axis_t){
            .extent = repetitions,
            .element_stride = stride_bits / record->element_bit_count};
        sequence.rank++;
        sequence.layout_kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED;
      }
    }
    if (representable && loom_aie2p_native_encode_descriptor(
                             family, &sequence, dma, record_bytes, repetitions,
                             LOOM_XDNA_TILE_KIND_SHIM_NOC, words, &reason)) {
      *out_repetitions = 1;
      return true;
    }
    // A short stream can retain all native dimensions for each record and use
    // the independent descriptor iterator for its sequence axis.
    const uint64_t step = byte_stride / dma->transfer_length_granularity;
    if (repetitions > (UINT64_C(1) << dma->iteration_bits) ||
        step > (UINT64_C(1) << dma->step_size_bits)) {
      return false;
    }
    words[6] =
        loom_xdna_register_field_encode_admitted(
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD6_ITERATION_WRAP,
            repetitions - 1) |
        loom_xdna_register_field_encode_admitted(
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD6_ITERATION_STEP_SIZE,
            step - 1);
  }
  if (!loom_aie2p_native_encode_descriptor(family, record, dma, record_bytes, 1,
                                           LOOM_XDNA_TILE_KIND_SHIM_NOC, words,
                                           &reason)) {
    return false;
  }
  *out_repetitions = repetitions;
  return true;
}

// A finite DMA sequence fits all its tasks in the hardware queue before any
// completion wait. Fixed-slot chains cover whole ring rotations. A final task
// reuses an aligned suffix or uses a separate prefix for a partial rotation.
iree_status_t loom_aie2p_native_build_sequence(
    const loom_xdna_dma_facts_t* dma,
    const loom_aie2p_native_memory_layout_t* layout, uint16_t first_descriptor,
    uint16_t available_descriptors, loom_xdna_dma_direction_t direction,
    uint32_t repetitions, uint32_t capacity, uint32_t byte_offset,
    uint32_t byte_stride, const uint32_t template_words[8],
    loom_aie2p_native_dma_sequence_t* out_sequence,
    iree_string_view_t* out_reason, bool* out_valid,
    iree_arena_allocator_t* arena) {
  *out_valid = false;
  if (!available_descriptors) {
    *out_reason = IREE_SV("an available descriptor in the DMA channel's bank");
    return iree_ok_status();
  }
  // Refilling from the sequential configuration engine can deadlock another
  // stream whose records are needed to release this stream's FIFO credits.
  const uint32_t repeat_limit = dma->maximum_task_repeat_count;
  const uint32_t queue_depth = dma->task_queue_depth;
  uint32_t chain_length = 1;
  uint32_t tail_length = 0;
  uint32_t tail_descriptor = 0;
  uint32_t cycles = repetitions;
  uint32_t required_descriptors = 1;
  uint32_t task_count =
      (uint32_t)iree_host_size_ceil_div(repetitions, repeat_limit);
  if (task_count > queue_depth) {
    required_descriptors = 0;
    for (uint32_t span = capacity; span <= available_descriptors;
         span += capacity) {
      const uint32_t candidate_cycles = repetitions / span;
      const uint32_t candidate_tail = repetitions % span;
      const bool separate_tail = candidate_tail % capacity != 0;
      const uint32_t candidate_descriptors =
          span + (separate_tail ? candidate_tail : 0);
      const uint32_t candidate_tasks =
          (uint32_t)iree_host_size_ceil_div(candidate_cycles, repeat_limit) +
          (candidate_tail != 0);
      if (candidate_descriptors > available_descriptors ||
          candidate_tasks > queue_depth ||
          (required_descriptors &&
           candidate_descriptors >= required_descriptors)) {
        continue;
      }
      chain_length = span;
      tail_length = candidate_tail;
      tail_descriptor = separate_tail ? span : span - candidate_tail;
      cycles = candidate_cycles;
      required_descriptors = candidate_descriptors;
      task_count = candidate_tasks;
    }
    if (!required_descriptors) {
      *out_reason = IREE_SV(
          "a finite record stream within DMA descriptor and task-queue "
          "capacity");
      return iree_ok_status();
    }
  }
  *out_sequence = (loom_aie2p_native_dma_sequence_t){
      .descriptor = first_descriptor,
      .descriptor_count = (uint16_t)required_descriptors,
      .task_count = (uint8_t)task_count,
      .rewind_iteration = layout && chain_length == 1 && repetitions % capacity,
  };
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena,
                          required_descriptors * sizeof(*out_sequence->words) +
                              task_count * sizeof(*out_sequence->tasks),
                          (void**)&out_sequence->words));
  out_sequence->tasks = (uint32_t*)(out_sequence->words + required_descriptors);
  const uint8_t chain_word = layout ? layout->chain.word : 7;
  const loom_xdna_register_field_id_t chain_next =
      layout ? layout->chain.next
             : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_NEXT_BD;
  const loom_xdna_register_field_id_t chain_enable =
      layout ? layout->chain.enable
             : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_USE_NEXT_BD;
  for (uint32_t i = 0; i < required_descriptors; ++i) {
    uint32_t* words = out_sequence->words[i];
    memcpy(words, template_words, sizeof(*out_sequence->words));
    const uint32_t ordinal = i < chain_length ? i : i - chain_length;
    const uint32_t slot = chain_length == 1 ? 0 : ordinal % capacity;
    if (layout) {
      words[layout->address_word] |= loom_xdna_register_field_encode_admitted(
          layout->address,
          (layout->local_address_base + byte_offset + slot * byte_stride) >>
              dma->address_encoding_shift);
      if (chain_length == 1) {
        const uint32_t step_units =
            byte_stride / dma->transfer_length_granularity;
        words[layout->iteration_word] =
            loom_xdna_register_field_encode_admitted(layout->iteration.wrap,
                                                     capacity - 1) |
            loom_xdna_register_field_encode_admitted(
                layout->iteration.step, step_units ? step_units - 1 : 0);
      }
    }
    const uint32_t chain_end =
        i < chain_length ? chain_length : required_descriptors;
    if (i + 1 < chain_end) {
      words[chain_word] |=
          loom_xdna_register_field_encode_admitted(
              chain_next, out_sequence->descriptor + i + 1) |
          loom_xdna_register_field_encode_admitted(chain_enable, 1);
    }
  }
  const bool sender = direction == LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM;
  const loom_xdna_register_field_id_t start =
      layout
          ? layout->directions[direction - 1].start
          : (sender
                 ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_START_BD_ID
                 : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_START_BD_ID);
  const loom_xdna_register_field_id_t repeat =
      layout
          ? layout->directions[direction - 1].repeat
          : (sender
                 ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_REPEAT_COUNT
                 : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_REPEAT_COUNT);
  uint32_t task = 0;
  while (cycles) {
    const uint32_t count = iree_min(cycles, repeat_limit);
    out_sequence->tasks[task++] =
        loom_xdna_register_field_encode_admitted(start,
                                                 out_sequence->descriptor) |
        loom_xdna_register_field_encode_admitted(repeat, count - 1);
    cycles -= count;
  }
  if (tail_length) {
    out_sequence->tasks[task] = loom_xdna_register_field_encode_admitted(
        start, out_sequence->descriptor + tail_descriptor);
  }
  *out_valid = true;
  return iree_ok_status();
}
