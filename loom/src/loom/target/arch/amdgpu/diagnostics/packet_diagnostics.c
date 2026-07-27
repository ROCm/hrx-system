// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/diagnostics/packet_diagnostics.h"

#include "loom/target/arch/amdgpu/refs/target_refs.h"

static bool loom_amdgpu_packet_diagnostics_is_crosslane(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  return descriptor ==
             loom_amdgpu_descriptor_ref_descriptor(
                 descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_DS_SWIZZLE_B32) ||
         descriptor ==
             loom_amdgpu_descriptor_ref_descriptor(
                 descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_DS_PERMUTE_B32) ||
         descriptor ==
             loom_amdgpu_descriptor_ref_descriptor(
                 descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32) ||
         descriptor ==
             loom_amdgpu_descriptor_ref_descriptor(
                 descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_FI_B32);
}

static iree_status_t loom_amdgpu_low_packet_diagnostic_try_packet(
    const loom_target_low_packet_diagnostic_provider_t* provider,
    loom_target_low_packet_diagnostic_context_t* context,
    const loom_low_packet_view_t* packet, bool* out_handled) {
  *out_handled = false;
  if (!iree_any_bit_set(
          loom_target_low_packet_diagnostics_diagnostic_flags(context),
          LOOM_TARGET_LOW_PACKET_DIAGNOSTIC_TARGET_PACKETS)) {
    return iree_ok_status();
  }
  if (packet->descriptor == NULL) {
    return iree_ok_status();
  }
  const loom_low_schedule_table_t* schedule =
      loom_target_low_packet_diagnostics_schedule(context);
  if (!loom_amdgpu_packet_diagnostics_is_crosslane(
          schedule->target.descriptor_set, packet->descriptor)) {
    return iree_ok_status();
  }
  *out_handled = true;
  return loom_target_low_packet_diagnostics_record_packet(
      context, packet, IREE_SV("lds-crosslane"), IREE_SV("selected"));
}

const loom_target_low_packet_diagnostic_provider_t
    loom_amdgpu_low_packet_diagnostic_provider_storage = {
        .name = IREE_SVL("amdgpu.packet_diagnostics"),
        .try_diagnose_packet = loom_amdgpu_low_packet_diagnostic_try_packet,
};
