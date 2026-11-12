// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/call_clobbers.h"

#include <stdlib.h>

#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/registers.h"

static void loom_aie2p_call_clobbers_restore(
    const loom_low_descriptor_set_t* descriptors, uint16_t physical_register,
    loom_aie2p_register_unit_set_t* units) {
  const loom_low_physical_register_t* view =
      &descriptors->physical_registers[physical_register];
  for (uint8_t i = 0; i < view->atomic_unit_count; ++i) {
    const uint16_t unit =
        descriptors
            ->physical_register_atomic_units[view->atomic_unit_start + i];
    units->words[unit / 64] &= ~(UINT64_C(1) << (unit % 64));
  }
}

static bool loom_aie2p_call_clobbers_overlap(
    const loom_low_descriptor_set_t* descriptors, uint16_t physical_register,
    const loom_aie2p_register_unit_set_t* units) {
  const loom_low_physical_register_t* view =
      &descriptors->physical_registers[physical_register];
  for (uint8_t i = 0; i < view->atomic_unit_count; ++i) {
    const uint16_t unit =
        descriptors
            ->physical_register_atomic_units[view->atomic_unit_start + i];
    if (units->words[unit / 64] & (UINT64_C(1) << (unit % 64))) {
      return true;
    }
  }
  return false;
}

static int loom_aie2p_call_clobbers_compare(const void* lhs_ptr,
                                            const void* rhs_ptr) {
  const loom_low_call_clobber_t* lhs = lhs_ptr;
  const loom_low_call_clobber_t* rhs = rhs_ptr;
  if (lhs->location != rhs->location) {
    return lhs->location < rhs->location ? -1 : 1;
  }
  return (lhs->register_class > rhs->register_class) -
         (lhs->register_class < rhs->register_class);
}

iree_status_t loom_aie2p_call_clobbers_build(
    loom_aie2p_register_unit_set_t native_writes,
    loom_aie2p_physical_register_id_t restored_register,
    const loom_low_call_contract_t* carriers,
    loom_symbol_ref_array_t direct_callees,
    loom_low_call_contract_provider_t call_contracts,
    iree_arena_allocator_t* arena, loom_low_call_clobber_list_t* out_clobbers) {
  *out_clobbers = (loom_low_call_clobber_list_t){0};
  const loom_low_descriptor_set_t* descriptors =
      loom_aie2p_core_descriptor_set();
  if (restored_register != LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID) {
    loom_aie2p_call_clobbers_restore(descriptors, restored_register,
                                     &native_writes);
  }
  iree_host_size_t boundary_count =
      (iree_host_size_t)carriers->argument_count + carriers->result_count;
  for (iree_host_size_t i = 0; i < direct_callees.count; ++i) {
    const loom_low_call_contract_t* callee = call_contracts.query(
        call_contracts.user_data, direct_callees.values[i]);
    boundary_count += callee->clobbers.count;
  }
  iree_host_size_t count = 0;
  loom_low_call_clobber_t* clobbers = NULL;
  for (uint16_t pass = 0; pass < 2; ++pass) {
    iree_host_size_t index = 0;
    for (uint16_t class_id = 0; class_id < descriptors->reg_class_count;
         ++class_id) {
      const loom_low_reg_class_t* reg_class =
          &descriptors->reg_classes[class_id];
      if (!loom_low_reg_class_uses_explicit_physical_registers(reg_class)) {
        continue;
      }
      for (uint16_t i = 0; i < reg_class->allocatable_count; ++i) {
        const uint16_t physical_register =
            loom_low_descriptor_set_physical_register_candidate(descriptors,
                                                                class_id, i);
        if (!loom_aie2p_call_clobbers_overlap(descriptors, physical_register,
                                              &native_writes)) {
          continue;
        }
        if (pass != 0) {
          clobbers[index] = (loom_low_call_clobber_t){
              .register_class = class_id,
              .location = physical_register,
              .count = 1,
          };
        }
        ++index;
      }
    }
    if (pass == 0) {
      count = index + boundary_count;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, count, sizeof(*clobbers), (void**)&clobbers));
    }
  }
  iree_host_size_t index = count - boundary_count;
  for (uint16_t side = 0; side < 2; ++side) {
    const loom_low_allocation_abi_location_t* locations =
        side == 0 ? carriers->arguments : carriers->results;
    const uint16_t side_count =
        side == 0 ? carriers->argument_count : carriers->result_count;
    for (uint16_t i = 0; i < side_count; ++i) {
      clobbers[index++] = (loom_low_call_clobber_t){
          .register_class = locations[i].descriptor_reg_class_id,
          .location = locations[i].location_base,
          .count = 1,
      };
    }
  }
  for (iree_host_size_t i = 0; i < direct_callees.count; ++i) {
    const loom_low_call_contract_t* callee = call_contracts.query(
        call_contracts.user_data, direct_callees.values[i]);
    for (iree_host_size_t j = 0; j < callee->clobbers.count; ++j) {
      clobbers[index++] = callee->clobbers.values[j];
    }
  }
  if (count != 0) {
    qsort(clobbers, count, sizeof(*clobbers), loom_aie2p_call_clobbers_compare);
  }
  iree_host_size_t unique_count = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (unique_count == 0 ||
        loom_aie2p_call_clobbers_compare(&clobbers[unique_count - 1],
                                         &clobbers[i]) != 0) {
      clobbers[unique_count++] = clobbers[i];
    }
  }
  *out_clobbers =
      (loom_low_call_clobber_list_t){.values = clobbers, .count = unique_count};
  return iree_ok_status();
}
