// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/target/memory.h"

// KFD reports ISA target versions, which are not raw GC IP versions. Ordinary
// GPU VA uses the low canonical interval on 48/57-bit targets. These envelopes
// describe architecture, not native VM sizing, reservations or exhaustion.
typedef struct amdf_gpu_kfd_target_memory_t {
  // Decimal ISA identity: major * 10000 + minor * 100 + stepping.
  uint32_t gfx_target_version;
  // PCI device identity, or zero when package class is fixed by this ISA.
  uint16_t pci_device_id;
  // Width of the ordinary low GPU address interval.
  uint8_t address_bit_count;
  // Whether CPU and GPU are integrated in this package.
  bool integrated;
} amdf_gpu_kfd_target_memory_t;

// The ISA mapping follows KFD's kgd2kfd_probe; address ceilings follow AMDGPU
// GMC VM descriptions. gfx942 spans APU and discrete packages, so its entries
// require explicit PCI identities instead of treating an unknown SKU as
// discrete.
static const amdf_gpu_kfd_target_memory_t amdf_gpu_kfd_target_memory[] = {
    {70000, 0, 40, true},       {70001, 0, 40, false},
    {80001, 0, 40, true},       {80002, 0, 40, false},
    {80003, 0, 40, false},      {90000, 0, 47, false},
    {90002, 0, 47, true},       {90004, 0, 47, false},
    {90006, 0, 47, false},      {90008, 0, 47, false},
    {90010, 0, 47, false},      {90012, 0, 47, true},
    {90402, 0x74a0, 47, true},  {90402, 0x74a1, 47, false},
    {90402, 0x74a2, 47, false}, {90402, 0x74a5, 47, false},
    {90402, 0x74a8, 47, false}, {90402, 0x74a9, 47, false},
    {90402, 0x74b5, 47, false}, {90402, 0x74b6, 47, false},
    {90402, 0x74b9, 47, false}, {90402, 0x74bd, 47, false},
    {90500, 0, 47, false},      {100100, 0, 47, false},
    {100101, 0, 47, false},     {100102, 0, 47, false},
    {100103, 0, 47, true},      {100300, 0, 47, false},
    {100301, 0, 47, false},     {100302, 0, 47, false},
    {100303, 0, 47, true},      {100304, 0, 47, false},
    {100305, 0, 47, true},      {100306, 0, 47, true},
    {110000, 0, 47, false},     {110001, 0, 47, false},
    {110002, 0, 47, false},     {110003, 0, 47, true},
    {110500, 0, 47, true},      {110501, 0, 47, true},
    {110502, 0, 47, true},      {110503, 0, 47, true},
    {110504, 0, 47, true},      {110700, 0, 47, true},
    {110701, 0, 47, true},      {120000, 0, 47, false},
    {120001, 0, 47, false},     {120500, 0, 56, false},
};

bool amdf_gpu_kfd_target_memory_initialize(uint32_t pci_device_id,
                                           uint32_t page_size,
                                           amdf_gpu_kfd_topology_t* topology) {
  const uint32_t gfx_target_version =
      topology->properties.gfx_ip.major * 10000 +
      topology->properties.gfx_ip.minor * 100 +
      topology->properties.gfx_ip.stepping;
  for (size_t i = 0; i < sizeof(amdf_gpu_kfd_target_memory) /
                             sizeof(amdf_gpu_kfd_target_memory[0]);
       ++i) {
    const amdf_gpu_kfd_target_memory_t* target = &amdf_gpu_kfd_target_memory[i];
    if (target->gfx_target_version != gfx_target_version ||
        (target->pci_device_id != 0 &&
         target->pci_device_id != pci_device_id)) {
      continue;
    }
    // AMDGPU reserves the bottom 64KiB. Activation obtains the installed range,
    // including any additional native limits, before allocating or mapping.
    topology->virtual_address.begin = UINT64_C(1) << 16;
    topology->virtual_address.end = UINT64_C(1) << target->address_bit_count;
    topology->virtual_address.alignment = page_size;
    topology->memory_features =
        amdf_gpu_kfd_topology_memory_features(topology, target->integrated);
    return true;
  }
  return false;
}
