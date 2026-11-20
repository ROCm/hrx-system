// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_broadcast.h"

#include <string.h>

#include "loom/target/arch/amd/xdna/aie2p/pipeline/control_packet.h"
#include "loom/util/adaptive_sort.h"

typedef struct loom_aie2p_program_slice_t {
  // Worker that owns this piece of a 16-byte tile-control block.
  iree_host_size_t worker;
  // Actual destination and byte extent, including partial first/last blocks.
  loom_aie2p_worker_code_t code;
} loom_aie2p_program_slice_t;

// Only slices covering exactly the same address interval are sorted together.
static bool loom_aie2p_program_slice_less(const loom_aie2p_program_slice_t* a,
                                          const loom_aie2p_program_slice_t* b) {
  const int order = memcmp(a->code.contents.data, b->code.contents.data,
                           a->code.contents.data_length);
  return order ? order < 0 : a->worker < b->worker;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_aie2p_program_slice_sort,
                          loom_aie2p_program_slice_t,
                          loom_aie2p_program_slice_less)

// The winning contiguous group in the sorted slices. A single copy can be
// sent directly to its worker without requiring a broadcast.
static void loom_aie2p_program_shared_slice(loom_aie2p_program_slice_t* slices,
                                            iree_host_size_t count,
                                            iree_host_size_t* out_first,
                                            iree_host_size_t* out_count) {
  loom_aie2p_program_slice_sort(slices, count);
  for (iree_host_size_t first = 0; first < count;) {
    const iree_const_byte_span_t bytes = slices[first].code.contents;
    iree_host_size_t end = first + 1;
    while (end < count && memcmp(bytes.data, slices[end].code.contents.data,
                                 bytes.data_length) == 0) {
      ++end;
    }
    const iree_host_size_t matches = end - first;
    if (matches >= 2 && (matches > *out_count ||
                         (matches == *out_count &&
                          slices[first].worker < slices[*out_first].worker))) {
      *out_first = first;
      *out_count = matches;
    }
    first = end;
  }
}

static void loom_aie2p_program_stream_append(
    loom_aie2p_worker_code_stream_t* stream, uint32_t header,
    const loom_aie2p_worker_code_t* code) {
  uint32_t* end = loom_aie2p_control_packet_store_writes(
      header, code->address, code->contents,
      stream->words + stream->word_count);
  stream->word_count = (iree_host_size_t)(end - stream->words);
}

iree_status_t loom_aie2p_program_broadcast_build(
    const loom_aie2p_worker_code_t* workers, iree_host_size_t worker_count,
    iree_arena_allocator_t* arena, loom_aie2p_program_broadcast_t* out_result) {
  *out_result = (loom_aie2p_program_broadcast_t){0};
  if (!worker_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, worker_count, sizeof(*out_result->individual),
      (void**)&out_result->individual));
  loom_aie2p_program_slice_t* slices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, worker_count, sizeof(*slices), (void**)&slices));
  uint32_t first_address = UINT32_MAX, end_address = 0;
  iree_host_size_t shared_capacity = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    const loom_aie2p_worker_code_t* code = &workers[i];
    const iree_host_size_t size = code->contents.data_length;
    const iree_host_size_t capacity =
        loom_aie2p_control_packet_write_word_count(code->address, size);
    shared_capacity = iree_min(shared_capacity, capacity);
    first_address = iree_min(first_address, code->address);
    end_address = iree_max(end_address, code->address + (uint32_t)size);
    loom_aie2p_worker_code_stream_t* stream = &out_result->individual[i];
    *stream = (loom_aie2p_worker_code_stream_t){0};
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, capacity, sizeof(*stream->words), (void**)&stream->words));
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, shared_capacity, sizeof(*out_result->shared.words),
      (void**)&out_result->shared.words));
  const uint32_t header = loom_aie2p_control_packet_parity(0);
  for (uint32_t block = first_address & ~15u; block < end_address;
       block += 16) {
    iree_host_size_t count = 0;
    bool same_extent = true;
    for (iree_host_size_t i = 0; i < worker_count; ++i) {
      const loom_aie2p_worker_code_t* code = &workers[i];
      const uint32_t begin = iree_max(block, code->address);
      const uint32_t end = iree_min(
          block + 16, code->address + (uint32_t)code->contents.data_length);
      if (end <= begin) {
        continue;
      }
      const iree_const_byte_span_t bytes = iree_make_const_byte_span(
          code->contents.data + (begin - code->address), end - begin);
      if (count) {
        same_extent &= begin == slices[0].code.address &&
                       bytes.data_length == slices[0].code.contents.data_length;
      }
      slices[count++] = (loom_aie2p_program_slice_t){i, {begin, bytes}};
    }
    iree_host_size_t shared_first = 0, shared_count = 0;
    if (count == worker_count && count >= 2 && same_extent) {
      loom_aie2p_program_shared_slice(slices, count, &shared_first,
                                      &shared_count);
    }
    if (shared_count) {
      loom_aie2p_program_stream_append(&out_result->shared, header,
                                       &slices[shared_first].code);
    }
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (i < shared_first || i >= shared_first + shared_count) {
        loom_aie2p_program_stream_append(
            &out_result->individual[slices[i].worker], header, &slices[i].code);
      }
    }
  }
  return iree_ok_status();
}
