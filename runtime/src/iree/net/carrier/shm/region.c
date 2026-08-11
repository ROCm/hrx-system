// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/region.h"

#include <string.h>

iree_status_t iree_net_shm_region_layout_calculate(
    uint32_t ring_capacity, iree_host_size_t mapping_alignment,
    iree_net_shm_region_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(out_layout);
  memset(out_layout, 0, sizeof(*out_layout));

  if (ring_capacity < IREE_MPSC_QUEUE_MIN_CAPACITY ||
      !iree_host_size_is_power_of_two((iree_host_size_t)ring_capacity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ring capacity must be a power of two >= %" PRIu32,
                            IREE_MPSC_QUEUE_MIN_CAPACITY);
  }
  if (mapping_alignment < IREE_NET_SHM_REGION_MIN_MAPPING_ALIGNMENT ||
      mapping_alignment > IREE_NET_SHM_REGION_MAX_MAPPING_ALIGNMENT ||
      !iree_host_size_is_power_of_two(mapping_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "mapping alignment %" PRIhsz " must be a power of two between %" PRIhsz
        " and %" PRIhsz,
        mapping_alignment, IREE_NET_SHM_REGION_MIN_MAPPING_ALIGNMENT,
        IREE_NET_SHM_REGION_MAX_MAPPING_ALIGNMENT);
  }

  iree_host_size_t ring_size = iree_mpsc_queue_required_size(ring_capacity);
  iree_host_size_t ring_b_offset = 0;
  iree_host_size_t unaligned_region_size = 0;
  iree_host_size_t region_size = 0;
  if (!iree_host_size_checked_add(IREE_NET_SHM_REGION_OFFSET_RINGS, ring_size,
                                  &ring_b_offset) ||
      !iree_host_size_checked_add(ring_b_offset, ring_size,
                                  &unaligned_region_size) ||
      !iree_host_size_checked_align(unaligned_region_size, mapping_alignment,
                                    &region_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SHM region size overflow for ring capacity %" PRIu32, ring_capacity);
  }

  out_layout->ring_capacity = ring_capacity;
  out_layout->mapping_alignment = mapping_alignment;
  out_layout->ring_size = ring_size;
  out_layout->ring_a_offset = IREE_NET_SHM_REGION_OFFSET_RINGS;
  out_layout->ring_b_offset = ring_b_offset;
  out_layout->region_size = region_size;
  return iree_ok_status();
}

static iree_status_t iree_net_shm_region_validate_mapping(
    const iree_net_shm_region_layout_t* layout,
    const iree_shm_mapping_t* mapping) {
  if (!mapping->base) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SHM region mapping base must not be NULL");
  }
  if (mapping->size != layout->region_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SHM region mapping size %" PRIhsz
                            " does not match expected %" PRIhsz,
                            mapping->size, layout->region_size);
  }
  return iree_ok_status();
}

iree_status_t iree_net_shm_region_initialize(
    const iree_net_shm_region_layout_t* layout, iree_shm_mapping_t* mapping,
    iree_mpsc_queue_t* out_ring_a, iree_mpsc_queue_t* out_ring_b) {
  IREE_ASSERT_ARGUMENT(layout);
  IREE_ASSERT_ARGUMENT(mapping);
  IREE_ASSERT_ARGUMENT(out_ring_a);
  IREE_ASSERT_ARGUMENT(out_ring_b);
  memset(out_ring_a, 0, sizeof(*out_ring_a));
  memset(out_ring_b, 0, sizeof(*out_ring_b));

  IREE_RETURN_IF_ERROR(iree_net_shm_region_validate_mapping(layout, mapping));

  memset(mapping->base, 0, mapping->size);
  iree_net_shm_region_header_t* header =
      (iree_net_shm_region_header_t*)((uint8_t*)mapping->base +
                                      IREE_NET_SHM_REGION_OFFSET_HEADER);
  header->magic = IREE_NET_SHM_REGION_MAGIC;
  header->version = IREE_NET_SHM_REGION_VERSION;
  header->ring_capacity = layout->ring_capacity;

  iree_mpsc_queue_t ring_a;
  iree_mpsc_queue_t ring_b;
  iree_status_t status = iree_mpsc_queue_initialize(
      (uint8_t*)mapping->base + layout->ring_a_offset, layout->ring_size,
      layout->ring_capacity, &ring_a);
  if (iree_status_is_ok(status)) {
    status = iree_mpsc_queue_initialize(
        (uint8_t*)mapping->base + layout->ring_b_offset, layout->ring_size,
        layout->ring_capacity, &ring_b);
  }
  if (iree_status_is_ok(status)) {
    *out_ring_a = ring_a;
    *out_ring_b = ring_b;
  }
  return status;
}

iree_status_t iree_net_shm_region_open(
    const iree_net_shm_region_layout_t* layout,
    const iree_shm_mapping_t* mapping, iree_mpsc_queue_t* out_ring_a,
    iree_mpsc_queue_t* out_ring_b) {
  IREE_ASSERT_ARGUMENT(layout);
  IREE_ASSERT_ARGUMENT(mapping);
  IREE_ASSERT_ARGUMENT(out_ring_a);
  IREE_ASSERT_ARGUMENT(out_ring_b);
  memset(out_ring_a, 0, sizeof(*out_ring_a));
  memset(out_ring_b, 0, sizeof(*out_ring_b));

  IREE_RETURN_IF_ERROR(iree_net_shm_region_validate_mapping(layout, mapping));

  iree_net_shm_region_header_t header;
  memcpy(&header,
         (const uint8_t*)mapping->base + IREE_NET_SHM_REGION_OFFSET_HEADER,
         sizeof(header));
  if (header.magic != IREE_NET_SHM_REGION_MAGIC) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SHM region magic mismatch: got 0x%08x, expected 0x%08x", header.magic,
        IREE_NET_SHM_REGION_MAGIC);
  }
  if (header.version != IREE_NET_SHM_REGION_VERSION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SHM region version mismatch: got %" PRIu32
                            ", expected %" PRIu32,
                            header.version, IREE_NET_SHM_REGION_VERSION);
  }
  if (header.ring_capacity != layout->ring_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SHM region ring capacity mismatch: got %" PRIu32
                            ", expected %" PRIu32,
                            header.ring_capacity, layout->ring_capacity);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(header.reserved); ++i) {
    if (header.reserved[i] != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "SHM region reserved byte %" PRIhsz " must be zero", i);
    }
  }

  iree_mpsc_queue_t ring_a;
  iree_mpsc_queue_t ring_b;
  iree_status_t status =
      iree_mpsc_queue_open((uint8_t*)mapping->base + layout->ring_a_offset,
                           layout->ring_size, &ring_a);
  if (iree_status_is_ok(status) && ring_a.capacity != layout->ring_capacity) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SHM Ring A capacity mismatch: got %" PRIu32
                              ", expected %" PRIu32,
                              ring_a.capacity, layout->ring_capacity);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_mpsc_queue_open((uint8_t*)mapping->base + layout->ring_b_offset,
                             layout->ring_size, &ring_b);
  }
  if (iree_status_is_ok(status) && ring_b.capacity != layout->ring_capacity) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SHM Ring B capacity mismatch: got %" PRIu32
                              ", expected %" PRIu32,
                              ring_b.capacity, layout->ring_capacity);
  }
  if (iree_status_is_ok(status)) {
    *out_ring_a = ring_a;
    *out_ring_b = ring_b;
  }
  return status;
}
