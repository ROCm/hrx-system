// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_catalog.h"

#include <string.h>

#include "loom/target/arch/amd/xdna/aie2p/pipeline/control_packet.h"
#include "loom/target/arch/amd/xdna/array/registers.h"
#include "loom/util/adaptive_sort.h"

typedef struct loom_aie2p_program_catalog_stream_t {
  // Exact linked contents and program-memory address.
  const loom_native_section_t* section;
  // Non-root group ordinal, also the relative descriptor index.
  iree_host_size_t ordinal;
  // Framed offset within the catalog; identical streams share this range.
  uint32_t offset;
  // Framed byte length, including the final ordered completion read.
  uint32_t length;
} loom_aie2p_program_catalog_stream_t;

static int loom_aie2p_program_catalog_compare(
    const loom_aie2p_program_catalog_stream_t* a,
    const loom_aie2p_program_catalog_stream_t* b) {
  if (a->section->address != b->section->address) {
    return a->section->address < b->section->address ? -1 : 1;
  }
  const iree_const_byte_span_t x = a->section->contents;
  const iree_const_byte_span_t y = b->section->contents;
  if (x.data_length != y.data_length) {
    return x.data_length < y.data_length ? -1 : 1;
  }
  return memcmp(x.data, y.data, x.data_length);
}

static bool loom_aie2p_program_catalog_less(
    const loom_aie2p_program_catalog_stream_t* a,
    const loom_aie2p_program_catalog_stream_t* b) {
  return loom_aie2p_program_catalog_compare(a, b) < 0;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_aie2p_program_catalog_sort,
                          loom_aie2p_program_catalog_stream_t,
                          loom_aie2p_program_catalog_less)

static const loom_native_section_t* loom_aie2p_program_catalog_section(
    const loom_aie2p_program_layout_group_t* group,
    const loom_aie2p_linked_tile_t* linked) {
  return &linked->assembly
              .sections[linked->assembly
                            .contribution_layouts
                                [group->native.section_contribution_index]
                            .section_index];
}

static loom_aie2p_program_record_t loom_aie2p_program_catalog_block(
    uint32_t address, const uint32_t* words, iree_host_size_t count) {
  return (loom_aie2p_program_record_t){
      .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32,
      .value.register_block_write32 = {address, words, count}};
}

iree_status_t loom_aie2p_program_catalog_build(
    loom_aie2p_code_transport_t* transport,
    const loom_aie2p_code_transport_binding_t* binding,
    const loom_aie2p_program_layout_t* layout,
    const loom_aie2p_linked_tile_t* linked, iree_arena_allocator_t* arena,
    loom_aie2p_program_catalog_t* out_catalog,
    iree_string_view_t* out_requirement) {
  *out_catalog = (loom_aie2p_program_catalog_t){0};
  *out_requirement = iree_string_view_empty();
  const loom_xdna_array_family_t* family = transport->plan.routing.family;
  const loom_xdna_tile_facts_t* memory =
      loom_xdna_array_tile_facts(family, binding->memory);
  const loom_xdna_tile_facts_t* compute =
      loom_xdna_array_tile_facts(family, binding->compute);
  loom_aie2p_execution_tile_t* occupancy =
      &transport->plan.tiles[binding->memory.column * family->row_count +
                             binding->memory.row];
  const iree_host_size_t stream_count = binding->descriptor_count;
  loom_aie2p_program_catalog_stream_t* streams = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, stream_count, sizeof(*streams), (void**)&streams));
  for (iree_host_size_t i = 0; i < stream_count; ++i) {
    streams[i] = (loom_aie2p_program_catalog_stream_t){
        .section =
            loom_aie2p_program_catalog_section(&layout->groups[i + 1], linked),
        .ordinal = i};
  }
  loom_aie2p_program_catalog_sort(streams, stream_count);
  uint64_t catalog_length = 0;
  for (iree_host_size_t i = 0; i < stream_count; ++i) {
    if (i != 0 &&
        loom_aie2p_program_catalog_compare(&streams[i - 1], &streams[i]) == 0) {
      streams[i].offset = streams[i - 1].offset;
      streams[i].length = streams[i - 1].length;
    } else {
      const uint64_t length =
          iree_host_align(streams[i].section->contents.data_length, 4);
      streams[i].offset = (uint32_t)catalog_length;
      streams[i].length = (uint32_t)(length + 8 * ((length + 15) / 16) + 8);
      catalog_length += streams[i].length;
    }
  }
  const uint64_t offset =
      iree_host_align(occupancy->memory_length, memory->dma.address_alignment);
  if (offset + catalog_length > memory->memory.local_capacity) {
    *out_requirement =
        IREE_SV("SRAM capacity for the complete framed native code catalog");
    return iree_ok_status();
  }
  occupancy->memory_length = offset + catalog_length;
  const iree_host_size_t record_count = binding->descriptor_count + 3;
  loom_aie2p_program_record_t* records = NULL;
  uint32_t* words = NULL;
  uint32_t* descriptors = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, record_count, sizeof(*records), (void**)&records));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, catalog_length / 4, sizeof(*words), (void**)&words));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, binding->descriptor_count, 8 * sizeof(*descriptors),
      (void**)&descriptors));
  memset(descriptors, 0, binding->descriptor_count * 8 * sizeof(*descriptors));
  const uint16_t engine = binding->engine;
  const loom_xdna_register_field_id_t reset_field =
      LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_CHANNEL_MM2S_CONTROL_RESET;
  const uint32_t reset_address = (uint32_t)loom_xdna_register_field_address(
      family, reset_field, binding->memory, &engine);
  records[0] = (loom_aie2p_program_record_t){
      .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32,
      .value.register_write32 = {
          reset_address, loom_xdna_register_field_encode(reset_field, 1)}};
  const uint32_t status_address =
      (uint32_t)loom_xdna_register_field_address(
          family,
          LOOM_XDNA_REGISTER_FIELD_CORE_TILE_CONTROL_STATUS_FIRST_HEADER_PARITY_ERROR,
          binding->compute, NULL) &
      ((UINT32_C(1) << family->row_shift) - 1);
  for (iree_host_size_t i = 0; i < stream_count; ++i) {
    const loom_aie2p_program_catalog_stream_t* stream = &streams[i];
    const loom_native_section_t* section = stream->section;
    if (i == 0 || streams[i - 1].offset != stream->offset) {
      iree_host_size_t position = stream->offset / 4;
      const iree_host_size_t length =
          iree_host_align(section->contents.data_length, 4);
      for (iree_host_size_t byte = 0; byte < length; byte += 16) {
        const iree_host_size_t count = iree_min(length - byte, 16) / 4;
        words[position++] = loom_aie2p_control_packet_parity(0);
        words[position++] = loom_aie2p_control_packet_write(
            compute->memory.program_load_base + (uint32_t)section->address -
                compute->memory.program_base + (uint32_t)byte,
            count);
        for (iree_host_size_t j = 0; j < count; ++j) {
          const iree_host_size_t begin = byte + j * 4;
          uint8_t padded[4] = {0};
          memcpy(
              padded, section->contents.data + begin,
              iree_min(section->contents.data_length - begin, sizeof(padded)));
          words[position++] = iree_unaligned_load_le_u32(padded);
        }
      }
      words[position++] = loom_aie2p_control_packet_parity(0);
      words[position++] = loom_aie2p_control_packet_read(status_address, 1,
                                                         binding->reply_packet);
    }
    uint32_t* descriptor = &descriptors[stream->ordinal * 8];
    descriptor[0] = loom_xdna_register_field_encode(
        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD0_BUFFER_LENGTH,
        stream->length / 4);
    descriptor[1] = loom_xdna_register_field_encode(
        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD1_BASE_ADDRESS,
        (memory->memory.local_load_base + offset + stream->offset) >>
            memory->dma.address_encoding_shift);
    descriptor[7] = loom_xdna_register_field_encode(
        LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD7_VALID_BD, 1);
    const uint16_t ordinal =
        binding->first_descriptor + (uint16_t)stream->ordinal;
    records[stream->ordinal + 1] = loom_aie2p_program_catalog_block(
        (uint32_t)loom_xdna_register_field_address(
            family,
            LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_DMA_BD_WORD0_BUFFER_LENGTH,
            binding->memory, &ordinal),
        descriptor, 8);
  }
  const uint32_t address =
      ((uint32_t)binding->memory.column << family->column_shift) |
      ((uint32_t)binding->memory.row << family->row_shift) |
      (memory->memory.local_base + (uint32_t)offset);
  records[record_count - 2] =
      loom_aie2p_program_catalog_block(address, words, catalog_length / 4);
  records[record_count - 1] = (loom_aie2p_program_record_t){
      .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32,
      .value.register_write32 = {
          reset_address, loom_xdna_register_field_encode(reset_field, 0)}};
  *out_catalog =
      (loom_aie2p_program_catalog_t){.records = records,
                                     .record_count = record_count,
                                     .offset = (uint32_t)offset,
                                     .byte_length = (uint32_t)catalog_length};
  return iree_ok_status();
}

iree_status_t loom_aie2p_program_catalog_install(
    const loom_aie2p_code_transport_t* transport,
    const loom_aie2p_program_catalog_t* catalogs,
    iree_host_size_t catalog_count, iree_arena_allocator_t* arena,
    loom_aie2p_array_program_t* program) {
  iree_host_size_t count = program->array_record_count;
  for (const loom_aie2p_native_switch_t* state = transport->plan.switches;
       state; state = state->next) {
    count += LOOM_AIE2P_NATIVE_SWITCH_BANK_COUNT;
  }
  for (iree_host_size_t i = 0; i < catalog_count; ++i) {
    count += catalogs[i].record_count;
  }
  loom_aie2p_program_record_t* records = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, count, sizeof(*records),
                                                 (void**)&records));
  memcpy(records, program->array_records,
         program->array_record_count * sizeof(*records));
  iree_host_size_t position = program->array_record_count;
  for (const loom_aie2p_native_switch_t* state = transport->plan.switches;
       state; state = state->next) {
    for (unsigned i = 0; i < LOOM_AIE2P_NATIVE_SWITCH_BANK_COUNT; ++i) {
      const loom_aie2p_native_register_bank_t* bank = &state->banks[i];
      records[position++] = loom_aie2p_program_catalog_block(
          bank->address, bank->words, bank->word_count);
    }
  }
  for (iree_host_size_t i = 0; i < catalog_count; ++i) {
    if (!catalogs[i].record_count) {
      continue;
    }
    memcpy(records + position, catalogs[i].records,
           catalogs[i].record_count * sizeof(*records));
    position += catalogs[i].record_count;
  }
  program->array_records = records;
  program->array_record_count = count;
  return iree_ok_status();
}
