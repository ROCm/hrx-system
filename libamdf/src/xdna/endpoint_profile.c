// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/endpoint_profile.h"

#include <stddef.h>

#include "libamdf/src/xdna/target/npu4/bootstrap.h"
#include "libamdf/src/xdna/target/npu5/bootstrap.h"

_Static_assert(sizeof(amdf_xdna_endpoint_profile_t) <= 64,
               "resolved XDNA execution profile must remain cache compact");

// Static profiles contain only properties that are invariant for an exact PCI
// identity. AIE4 geometry is firmware-reported and is intentionally absent.
static const amdf_xdna_endpoint_info_t amdf_xdna_npu1_endpoint_info = {
    .type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO,
    .structure_size = sizeof(amdf_xdna_endpoint_info_t),
    .architecture = AMDF_XDNA_ARCHITECTURE_AIE2,
    .array =
        {
            .column_origin = 1,
            .column_count = 4,
            .row_count = 6,
            .column_stride = UINT64_C(1) << 25,
        },
    .context =
        {
            .scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_SPATIAL |
                                AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED,
            .minimum_column_count = 1,
            .maximum_column_count = 4,
            .column_count_granularity = 1,
            .maximum_live_context_count = 6,
            .maximum_hardware_context_count = 6,
        },
    .target_id = "amd.xdna.phoenix.1502_00",
};

static const amdf_xdna_endpoint_info_t amdf_xdna_npu4_endpoint_info = {
    .type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO,
    .structure_size = sizeof(amdf_xdna_endpoint_info_t),
    .architecture = AMDF_XDNA_ARCHITECTURE_AIE2P,
    .array =
        {
            .column_origin = 0,
            .column_count = 8,
            .row_count = 6,
            .column_stride = UINT64_C(1) << 25,
        },
    .context =
        {
            .scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED,
            .minimum_column_count = 1,
            .maximum_column_count = 8,
            .column_count_granularity = 1,
            .maximum_live_context_count = 32,
            .maximum_hardware_context_count = 16,
        },
    .instruction =
        {
            .maximum_byte_length = UINT32_MAX & ~UINT64_C(3),
            .address_alignment = 32 * 1024,
            .byte_length_granularity = 4,
            .format =
                {
                    .format = AMDF_XDNA_BINARY_FORMAT_TRANSACTION,
                    .version = AMDF_XDNA_TRANSACTION_FORMAT_VERSION_0_1,
                },
        },
    .target_id = "amd.xdna.strix.17f0_10",
};

static const amdf_xdna_endpoint_info_t amdf_xdna_npu5_endpoint_info = {
    .type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO,
    .structure_size = sizeof(amdf_xdna_endpoint_info_t),
    .architecture = AMDF_XDNA_ARCHITECTURE_AIE2P,
    .array =
        {
            .column_origin = 0,
            .column_count = 8,
            .row_count = 6,
            .column_stride = UINT64_C(1) << 25,
        },
    .context =
        {
            .scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED,
            .minimum_column_count = 1,
            .maximum_column_count = 8,
            .column_count_granularity = 1,
            .maximum_live_context_count = 32,
            .maximum_hardware_context_count = 16,
        },
    .instruction =
        {
            .maximum_byte_length = UINT32_MAX & ~UINT64_C(3),
            .address_alignment = 32 * 1024,
            .byte_length_granularity = 4,
            .format =
                {
                    .format = AMDF_XDNA_BINARY_FORMAT_TRANSACTION,
                    .version = AMDF_XDNA_TRANSACTION_FORMAT_VERSION_0_1,
                },
        },
    .target_id = "amd.xdna.strix_halo.17f0_11",
};

static const amdf_xdna_endpoint_info_t amdf_xdna_npu6_endpoint_info = {
    .type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO,
    .structure_size = sizeof(amdf_xdna_endpoint_info_t),
    .architecture = AMDF_XDNA_ARCHITECTURE_AIE2P,
    .array =
        {
            .column_origin = 0,
            .column_count = 8,
            .row_count = 6,
            .column_stride = UINT64_C(1) << 25,
        },
    .context =
        {
            .scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED,
            .minimum_column_count = 1,
            .maximum_column_count = 8,
            .column_count_granularity = 1,
            .maximum_live_context_count = 32,
            .maximum_hardware_context_count = 16,
        },
    .target_id = "amd.xdna.krackan.17f0_20",
};

static const amdf_xdna_endpoint_profile_t amdf_xdna_npu1_profile = {
    .info = &amdf_xdna_npu1_endpoint_info,
};

static const amdf_xdna_endpoint_profile_t amdf_xdna_npu4_profile = {
    .info = &amdf_xdna_npu4_endpoint_info,
    .execution_capabilities =
        AMDF_XDNA_EXECUTION_CAPABILITY_TRANSACTION_INTERPRETER_V1 |
        AMDF_XDNA_EXECUTION_CAPABILITY_ELF_INSTRUCTIONS,
    .bootstrap = &amdf_xdna_npu4_bootstrap,
    .firmware_heap_byte_length = 64u * 1024u * 1024u,
    .dma = {.byte_offset = UINT32_C(0x80000000), .address_bit_count = 48},
    .transaction = {.device_generation = 4},
    .rows =
        {
            .shim_origin = 0,
            .shim_count = 1,
            .memory_origin = 1,
            .memory_count = 1,
            .core_origin = 2,
            .core_count = 4,
        },
};

static const amdf_xdna_endpoint_profile_t amdf_xdna_npu5_profile = {
    .info = &amdf_xdna_npu5_endpoint_info,
    .execution_capabilities =
        AMDF_XDNA_EXECUTION_CAPABILITY_TRANSACTION_INTERPRETER_V1 |
        AMDF_XDNA_EXECUTION_CAPABILITY_ELF_INSTRUCTIONS,
    .bootstrap = &amdf_xdna_npu5_bootstrap,
    .firmware_heap_byte_length = 64u * 1024u * 1024u,
    .dma = {.byte_offset = UINT32_C(0x80000000), .address_bit_count = 48},
    .transaction =
        {
            .device_generation = 4,
        },
    .rows =
        {
            .shim_origin = 0,
            .shim_count = 1,
            .memory_origin = 1,
            .memory_count = 1,
            .core_origin = 2,
            .core_count = 4,
        },
};

static const amdf_xdna_endpoint_profile_t amdf_xdna_npu6_profile = {
    .info = &amdf_xdna_npu6_endpoint_info,
};

typedef struct amdf_xdna_profile_identity_t {
  // PCI device identifier used only for central profile selection.
  uint32_t device_id;
  // PCI revision identifier used only for central profile selection.
  uint32_t revision_id;
  // Process-lifetime execution profile selected for the exact identity.
  const amdf_xdna_endpoint_profile_t* profile;
} amdf_xdna_profile_identity_t;

// Exact PCI identities are confined to this selector. The native object graph
// receives the resolved profile and never observes these keys.
static const amdf_xdna_profile_identity_t amdf_xdna_profile_identities[] = {
    {0x1502u, 0x00u, &amdf_xdna_npu1_profile},
    {0x17F0u, 0x10u, &amdf_xdna_npu4_profile},
    {0x17F0u, 0x11u, &amdf_xdna_npu5_profile},
    {0x17F0u, 0x20u, &amdf_xdna_npu6_profile},
};

const amdf_xdna_endpoint_profile_t* amdf_xdna_endpoint_profile_select(
    const amdf_endpoint_info_t* endpoint_info) {
  if (endpoint_info->engine_kind != AMDF_ENGINE_KIND_XDNA) {
    return NULL;
  }
  if (endpoint_info->pci.vendor_id != 0x1022u) {
    return NULL;
  }
  for (size_t i = 0; i < sizeof(amdf_xdna_profile_identities) /
                             sizeof(amdf_xdna_profile_identities[0]);
       ++i) {
    const amdf_xdna_profile_identity_t* identity =
        &amdf_xdna_profile_identities[i];
    if (endpoint_info->pci.device_id == identity->device_id &&
        endpoint_info->pci.revision_id == identity->revision_id) {
      return identity->profile;
    }
  }
  return NULL;
}

const amdf_xdna_endpoint_info_t* amdf_xdna_endpoint_profile_get_info(
    const amdf_xdna_endpoint_profile_t* profile) {
  return profile->info;
}
