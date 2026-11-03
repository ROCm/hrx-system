// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/function_plan_build.h"

#include "loom/target/arch/amdgpu/planning/packet_plan.h"

iree_status_t loom_amdgpu_function_plan_build(
    const loom_low_emission_frame_t* frame, iree_diagnostic_emitter_t emitter,
    const loom_amdgpu_packet_plan_options_t* packet_options,
    iree_arena_allocator_t* arena, bool* out_accepted,
    loom_amdgpu_function_plan_t* out_plan) {
  *out_accepted = false;
  *out_plan = (loom_amdgpu_function_plan_t){0};

  const loom_amdgpu_native_preflight_options_t preflight_options = {
      .emitter = emitter,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_native_preflight_analyze(
      &frame->schedule, &frame->allocation, &preflight_options,
      &out_plan->preflight));
  if (out_plan->preflight.error_count != 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_build(
      &frame->schedule.requirements.storage_layout, arena,
      &out_plan->storage_layout));
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, packet_options, arena,
      /*out_vopd_rejections=*/NULL, &out_plan->packet_plan));
  *out_accepted = true;
  return iree_ok_status();
}
