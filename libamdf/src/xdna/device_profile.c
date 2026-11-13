// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/device_profile.h"

#include <string.h>

#include "libamdf/src/xdna/target/npu4/bootstrap.h"
#include "libamdf/src/xdna/target/npu5/bootstrap.h"

typedef struct amdf_xdna_endpoint_identity_t {
  // Exact AMD PCI device identifier, including classic, PF, and VF roles.
  uint16_t device_id;
  // Exact PCI revision; an unlisted revision has no established identity.
  uint8_t revision_id;
  // Instruction and array architecture family, independent of native services.
  amdf_xdna_architecture_t architecture;
  // Process-lifetime canonical target identifier.
  const char* target_id;
} amdf_xdna_endpoint_identity_t;

// Native identity and role assignments:
// https://github.com/amd/xdna-driver/blob/24dfd68696d4aba8708ba32da39c008de73c2618/drivers/accel/amdxdna/amdxdna_pci_drv.c
// Recognition is passive and does not imply an activation or execution profile.
static const amdf_xdna_endpoint_identity_t amdf_xdna_endpoint_identities[] = {
    {0x1502, 0x00, AMDF_XDNA_ARCHITECTURE_AIE2, "amd.xdna.phoenix.1502_00"},
    {0x17f0, 0x10, AMDF_XDNA_ARCHITECTURE_AIE2P, "amd.xdna.strix.17f0_10"},
    {0x17f0, 0x11, AMDF_XDNA_ARCHITECTURE_AIE2P, "amd.xdna.strix_halo.17f0_11"},
    {0x17f0, 0x20, AMDF_XDNA_ARCHITECTURE_AIE2P, "amd.xdna.krackan.17f0_20"},
    {0x17f1, 0x10, AMDF_XDNA_ARCHITECTURE_AIE4, "amd.xdna.aie4.17f1_10"},
    {0x17f1, 0x13, AMDF_XDNA_ARCHITECTURE_AIE4, "amd.xdna.aie4.17f1_13"},
    {0x17f1, 0x14, AMDF_XDNA_ARCHITECTURE_AIE4, "amd.xdna.aie4.17f1_14"},
    {0x17f1, 0x15, AMDF_XDNA_ARCHITECTURE_AIE4, "amd.xdna.aie4.17f1_15"},
    {0x17f2, 0x10, AMDF_XDNA_ARCHITECTURE_AIE4, "amd.xdna.aie4.17f2_10"},
    {0x17f2, 0x13, AMDF_XDNA_ARCHITECTURE_AIE4, "amd.xdna.aie4.17f2_13"},
    {0x17f2, 0x14, AMDF_XDNA_ARCHITECTURE_AIE4, "amd.xdna.aie4.17f2_14"},
    {0x17f2, 0x15, AMDF_XDNA_ARCHITECTURE_AIE4, "amd.xdna.aie4.17f2_15"},
    {0x17f3, 0x10, AMDF_XDNA_ARCHITECTURE_AIE4, "amd.xdna.aie4.17f3_10"},
    {0x17f3, 0x13, AMDF_XDNA_ARCHITECTURE_AIE4, "amd.xdna.aie4.17f3_13"},
    {0x17f3, 0x14, AMDF_XDNA_ARCHITECTURE_AIE4, "amd.xdna.aie4.17f3_14"},
    {0x17f3, 0x15, AMDF_XDNA_ARCHITECTURE_AIE4, "amd.xdna.aie4.17f3_15"},
    {0x1b0a, 0x00, AMDF_XDNA_ARCHITECTURE_AIE4, "amd.xdna.aie4.1b0a_00"},
    {0x1b0b, 0x00, AMDF_XDNA_ARCHITECTURE_AIE4, "amd.xdna.aie4.1b0b_00"},
    {0x1b0c, 0x00, AMDF_XDNA_ARCHITECTURE_AIE4, "amd.xdna.aie4.1b0c_00"},
};

bool amdf_xdna_query_endpoint_info(const amdf_endpoint_info_t* endpoint_info,
                                   amdf_xdna_endpoint_info_t* out_info) {
  if (endpoint_info->engine_kind != AMDF_ENGINE_KIND_XDNA ||
      endpoint_info->pci.vendor_id != 0x1022u) {
    return false;
  }
  for (size_t i = 0; i < sizeof(amdf_xdna_endpoint_identities) /
                             sizeof(amdf_xdna_endpoint_identities[0]);
       ++i) {
    const amdf_xdna_endpoint_identity_t* identity =
        &amdf_xdna_endpoint_identities[i];
    if (endpoint_info->pci.device_id == identity->device_id &&
        endpoint_info->pci.revision_id == identity->revision_id) {
      amdf_xdna_endpoint_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO,
          .structure_size = sizeof(info),
          .architecture = identity->architecture,
      };
      memcpy(info.target_id, identity->target_id,
             strlen(identity->target_id) + 1);
      *out_info = info;
      return true;
    }
  }
  return false;
}

bool amdf_xdna_device_profile_initialize(
    const amdf_endpoint_info_t* endpoint_info,
    amdf_xdna_device_info_t* device_info,
    amdf_xdna_device_profile_t* out_profile) {
  amdf_xdna_endpoint_info_t identity;
  if (!amdf_xdna_query_endpoint_info(endpoint_info, &identity) ||
      identity.architecture != AMDF_XDNA_ARCHITECTURE_AIE2P) {
    return false;
  }

  amdf_xdna_device_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO,
      .structure_size = sizeof(info),
  };
  amdf_xdna_device_profile_t profile = {.info = device_info};
  // AIE2P register and transaction encodings, independent of array geometry.
  // Native metadata supplies all row and column counts during activation.
  info.array.column_stride = UINT64_C(1) << 25;
  info.context.minimum_column_count = 1;
  info.context.column_count_granularity = 1;
  info.instruction.maximum_byte_length = UINT32_MAX & ~UINT64_C(3);
  info.instruction.address_alignment = 32u * 1024u;
  info.instruction.byte_length_granularity = 4;
  info.instruction.format = (amdf_xdna_binary_format_info_t){
      .format = AMDF_XDNA_BINARY_FORMAT_TRANSACTION,
      .version = AMDF_XDNA_TRANSACTION_FORMAT_VERSION_0_1,
  };
  profile.execution_capabilities =
      AMDF_XDNA_EXECUTION_CAPABILITY_TRANSACTION_INTERPRETER_V1 |
      AMDF_XDNA_EXECUTION_CAPABILITY_ELF_INSTRUCTIONS;
  // Both firmware families admit the generated no-effects PDI. Strix Halo
  // uses different native context accounting from Strix and Krackan.
  profile.bootstrap = endpoint_info->pci.revision_id == 0x11u
                          ? &amdf_xdna_npu5_bootstrap
                          : &amdf_xdna_npu4_bootstrap;
  profile.firmware_heap_byte_length = 64u * 1024u * 1024u;
  profile.dma.byte_offset = UINT32_C(0x80000000);
  profile.dma.address_bit_count = 48;
  profile.transaction.device_generation = 4;
  *device_info = info;
  *out_profile = profile;
  return true;
}
