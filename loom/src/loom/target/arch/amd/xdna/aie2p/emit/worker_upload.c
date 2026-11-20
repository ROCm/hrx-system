// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/worker_upload.h"

#include <string.h>

#include "loom/target/arch/amd/xdna/aie2p/emit/startup_records.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/worker_upload_plan.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/control_packet.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_plan.h"

// A circuit-mode slave ignores packet filters. On first use both port banks
// are unknown; between drained waves only the changed span needs to be written.
static loom_aie2p_program_record_t* loom_aie2p_worker_switch_records(
    const loom_aie2p_native_switch_t* state,
    const loom_aie2p_native_switch_t* previous,
    loom_aie2p_program_record_t* next) {
  for (unsigned i = LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS;
       i <= LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES; ++i) {
    const loom_aie2p_native_register_bank_t* bank = &state->banks[i];
    uint32_t begin = 0, end = bank->word_count;
    if (previous) {
      const uint32_t* old = previous->banks[i].words;
      while (begin < end && bank->words[begin] == old[begin]) {
        ++begin;
      }
      while (end > begin && bank->words[end - 1] == old[end - 1]) {
        --end;
      }
    }
    if (begin < end) {
      *next++ = loom_aie2p_startup_block(bank->address + 4 * begin,
                                         bank->words + begin, end - begin);
    }
  }
  return next;
}

static loom_aie2p_program_record_t* loom_aie2p_worker_initialize(
    const loom_xdna_array_family_t* family, const loom_xdna_tile_facts_t* shim,
    const loom_aie2p_worker_upload_plan_t* upload,
    const loom_aie2p_xdna_tile_t* tiles, iree_host_size_t tile_count,
    loom_aie2p_program_record_t* next) {
  const loom_xdna_register_field_id_t mux =
      loom_xdna_shim_dma_mux_field(LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
                                   shim->dma.memory_to_stream_port_base);
  for (uint16_t column = 0; column < family->column_count; ++column) {
    if (!upload->senders[column]) {
      continue;
    }
    const loom_xdna_tile_coordinate_t source = {column, shim->first_row};
    *next++ = (loom_aie2p_program_record_t){
        .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32,
        .value.register_mask_write32 =
            {
                loom_aie2p_startup_address(family, source, mux),
                loom_xdna_register_field_encode(mux, -1),
                loom_xdna_register_field_encode(mux, 1),
            },
    };
    *next++ = loom_aie2p_startup_write(
        loom_aie2p_startup_address(
            family, source,
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_CONTROL_CONTROLLER_ID),
        0);
  }
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    if (upload->workers[i]) {
      *next++ = loom_aie2p_startup_write(
          loom_aie2p_startup_address(family, tiles[i].coordinate,
                                     loom_aie2p_compute_memory.lock_value),
          0);
    }
  }
  return next;
}

static loom_aie2p_program_record_t* loom_aie2p_worker_wave_records(
    const loom_xdna_array_family_t* family, const loom_xdna_tile_facts_t* shim,
    const loom_aie2p_worker_wave_t* wave,
    const loom_aie2p_native_switch_t** previous, uint32_t (*descriptors)[8],
    loom_aie2p_program_upload_patch_t* patches,
    const loom_aie2p_program_record_t* records,
    loom_aie2p_program_record_t* next, uint32_t* data_size) {
  for (const loom_aie2p_native_switch_t* state = wave->switches; state;
       state = state->next) {
    const iree_host_size_t slot =
        loom_xdna_array_physical_tile_index(family, state->coordinate);
    next = loom_aie2p_worker_switch_records(state, previous[slot], next);
    previous[slot] = state;
  }
  const loom_aie2p_native_connections_t* connections = &wave->connections;
  for (iree_host_size_t i = 0; i < connections->group_count; ++i) {
    const loom_xdna_tile_coordinate_t source =
        connections->groups[i].source.coordinate;
    const uint32_t offset = (uint32_t)iree_host_align(
        *data_size, LOOM_AIE2P_PROGRAM_UPLOAD_ALIGNMENT);
    // Every stream ends with its own three-word destination credit.
    const uint32_t length = (uint32_t)(wave->streams[i].word_count + 3) * 4;
    loom_aie2p_native_encode_linear_descriptor(&shim->dma, shim->kind, length,
                                               descriptors[i]);
    patches[i] = (loom_aie2p_program_upload_patch_t){(uint32_t)(next - records),
                                                     offset, length};
    *next++ = loom_aie2p_startup_block(
        loom_aie2p_startup_address(
            family, source,
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD0_BUFFER_LENGTH),
        descriptors[i], 8);
    *data_size = offset + length;
  }
  for (iree_host_size_t i = 0; i < connections->group_count; ++i) {
    *next++ = loom_aie2p_startup_write(
        loom_aie2p_startup_address(
            family, connections->groups[i].source.coordinate,
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_START_BD_ID),
        0);
  }
  const loom_xdna_register_field_id_t lock =
      loom_aie2p_compute_memory.lock_value;
  for (iree_host_size_t i = 0; i < connections->destination_count; ++i) {
    *next++ = loom_aie2p_startup_wait(
        loom_aie2p_startup_address(
            family, connections->destinations[i].coordinate, lock),
        loom_xdna_register_field_encode(lock, -1),
        loom_xdna_register_field_encode(lock, wave->credit));
  }
  for (iree_host_size_t i = 0; i < connections->group_count; ++i) {
    const loom_aie2p_native_queue_status_t sender =
        loom_aie2p_native_queue_status(
            family, connections->groups[i].source.coordinate, NULL, 0,
            LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM);
    *next++ = loom_aie2p_startup_wait((uint32_t)sender.address,
                                      sender.pending_mask, 0);
  }
  return next;
}

static void loom_aie2p_worker_pack_contents(
    const loom_xdna_array_family_t* family,
    const loom_aie2p_worker_upload_plan_t* upload,
    const loom_aie2p_program_upload_patch_t* patches,
    iree_const_byte_span_t prefix, uint8_t* data) {
  if (prefix.data_length) {
    memcpy(data, prefix.data, prefix.data_length);
  }
  iree_host_size_t end = prefix.data_length;
  const loom_xdna_register_field_id_t lock =
      loom_aie2p_compute_memory.lock_value;
  const uint32_t header = loom_aie2p_control_packet_parity(0);
  const uint32_t address =
      loom_aie2p_startup_address(
          family, upload->waves[0].connections.destinations[0].coordinate,
          lock) &
      ((UINT32_C(1) << family->row_shift) - 1);
  const uint32_t credit_write = loom_aie2p_control_packet_write(address, 1);
  for (iree_host_size_t i = 0; i < upload->wave_count; ++i) {
    const loom_aie2p_worker_wave_t* wave = &upload->waves[i];
    for (iree_host_size_t j = 0; j < wave->connections.group_count;
         ++j, ++patches) {
      const loom_aie2p_worker_code_stream_t* stream = &wave->streams[j];
      memset(data + end, 0, patches->source_offset - end);
      uint8_t* cursor = data + patches->source_offset;
      for (iree_host_size_t k = 0; k < stream->word_count; ++k, cursor += 4) {
        iree_unaligned_store_le_u32(cursor, stream->words[k]);
      }
      iree_unaligned_store_le_u32(cursor, header);
      iree_unaligned_store_le_u32(cursor + 4, credit_write);
      iree_unaligned_store_le_u32(
          cursor + 8, loom_xdna_register_field_encode(lock, wave->credit));
      end = patches->source_offset + patches->byte_length;
    }
  }
}

iree_status_t loom_aie2p_worker_upload_install(
    const loom_aie2p_execution_plan_t* plan,
    const loom_aie2p_xdna_tile_t* tiles, iree_host_size_t tile_count,
    iree_host_size_t first_record, iree_arena_allocator_t* arena,
    loom_aie2p_array_program_t* program) {
  const iree_arena_checkpoint_t checkpoint = iree_arena_checkpoint_save(arena);
  loom_aie2p_worker_upload_plan_t upload;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_worker_upload_plan(plan, tiles, tile_count, arena, &upload));
  if (!upload.wave_count) {
    iree_arena_checkpoint_restore(&checkpoint);
    return iree_ok_status();
  }
  const loom_aie2p_program_upload_t* prior = program->upload;
  const loom_aie2p_program_record_t* base_records =
      prior ? prior->records : program->array_records;
  const iree_host_size_t base_count =
      prior ? prior->record_count : program->array_record_count;
  const iree_host_size_t prior_count = prior ? prior->patch_count : 0;
  const iree_const_byte_span_t prefix =
      prior ? prior->contents : iree_const_byte_span_empty();
  const loom_xdna_array_family_t* family = plan->routing.family;
  const loom_xdna_tile_facts_t* shim =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_SHIM_NOC);
  const iree_host_size_t places = loom_xdna_array_physical_tile_count(family);
  const loom_aie2p_native_switch_t** previous = NULL;
  loom_aie2p_program_record_t* records = NULL;
  loom_aie2p_program_upload_patch_t* patches = NULL;
  uint32_t (*descriptors)[8] = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, places, sizeof(*previous), (void**)&previous));
  memset(previous, 0, places * sizeof(*previous));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, base_count + upload.record_capacity,
                                sizeof(*records), (void**)&records));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, prior_count + upload.transfer_count,
                                sizeof(*patches), (void**)&patches));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, upload.transfer_count,
                                                 sizeof(*descriptors),
                                                 (void**)&descriptors));
  memset(descriptors, 0, upload.transfer_count * sizeof(*descriptors));
  iree_host_size_t prefix_patches = 0;
  while (prefix_patches < prior_count &&
         prior->patches[prefix_patches].record_index < first_record) {
    patches[prefix_patches] = prior->patches[prefix_patches];
    ++prefix_patches;
  }
  loom_aie2p_program_record_t* first = records + first_record;
  loom_aie2p_program_record_t* next = loom_aie2p_worker_initialize(
      family, shim, &upload, tiles, tile_count, first);
  uint32_t data_size = (uint32_t)prefix.data_length;
  iree_host_size_t transfer = 0;
  for (iree_host_size_t i = 0; i < upload.wave_count; ++i) {
    const loom_aie2p_worker_wave_t* wave = &upload.waves[i];
    next = loom_aie2p_worker_wave_records(
        family, shim, wave, previous, descriptors + transfer,
        patches + prefix_patches + transfer, records, next, &data_size);
    transfer += wave->connections.group_count;
  }
  iree_host_size_t traffic = data_size - prefix.data_length;
  for (const loom_aie2p_program_record_t* record = first; record < next;
       ++record) {
    traffic += loom_aie2p_program_native_record_size(record);
  }
  if (traffic >= upload.inline_size) {
    iree_arena_checkpoint_restore(&checkpoint);
    return iree_ok_status();
  }
  // Retain all initial data-section loads and application commands. Only the
  // linked worker entries identified by this upload become DMA-backed.
  if (first_record) {
    memcpy(records, base_records, first_record * sizeof(*records));
  }
  if (base_count > first_record) {
    memcpy(next, base_records + first_record,
           (base_count - first_record) * sizeof(*records));
  }
  const uint32_t added_records = (uint32_t)(next - first);
  for (iree_host_size_t i = prefix_patches; i < prior_count; ++i) {
    patches[transfer + i] = prior->patches[i];
    patches[transfer + i].record_index += added_records;
  }
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, data_size, (void**)&data));
  loom_aie2p_worker_pack_contents(family, &upload, patches + prefix_patches,
                                  prefix, data);
  loom_aie2p_program_upload_t* result = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*result), (void**)&result));
  *result = (loom_aie2p_program_upload_t){
      .records = records,
      .record_count = base_count + added_records,
      .contents = iree_make_const_byte_span(data, data_size),
      .patches = patches,
      .patch_count = prior_count + transfer,
      .worker_entries = upload.workers,
  };
  program->upload = result;
  return iree_ok_status();
}
