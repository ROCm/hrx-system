// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/webgpu/webgpu_builtins.h"

#include "iree/hal/driver/webgpu/webgpu.h"
#include "iree/hal/driver/webgpu/webgpu_imports.h"

//===----------------------------------------------------------------------===//
// WGSL shader source
//===----------------------------------------------------------------------===//

// Linear and rectangular transfers share each pipeline body. Descriptor-sized
// tiles keep WGSL address arithmetic in u32 while host offsets remain u64.
// Edge dwords are updated atomically: distinct rows may occupy the same word.
static const char iree_hal_webgpu_builtin_fill_wgsl[] =
    "struct Params {\n"
    "  source_offset: u32, target_offset: u32, row_length: u32, flags: u32,\n"
    "  source_row_stride: u32, target_row_stride: u32,\n"
    "  source_slice_stride: u32, target_slice_stride: u32,\n"
    "  row_count: u32, slice_count: u32, pattern: u32, padding: u32,\n"
    "};\n"
    "@group(0) @binding(0) var<storage, read_write> target: "
    "array<atomic<u32>>;\n"
    "@group(0) @binding(1) var<uniform> params: Params;\n"
    "fn store_word(index: u32, value: u32, mask: u32) {\n"
    "  if (mask == 0xffffffffu) { atomicStore(&target[index], value); return; "
    "}\n"
    "  var previous = atomicLoad(&target[index]);\n"
    "  loop {\n"
    "    let result = atomicCompareExchangeWeak(&target[index], previous,\n"
    "        (previous & ~mask) | (value & mask));\n"
    "    if (result.exchanged) { break; }\n"
    "    previous = result.old_value;\n"
    "  }\n"
    "}\n"
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) id: vec3<u32>,\n"
    "        @builtin(num_workgroups) groups: vec3<u32>) {\n"
    "  let target_start = params.target_offset + id.y * "
    "params.target_row_stride +\n"
    "      id.z * params.target_slice_stride;\n"
    "  let source_start = params.source_offset + id.y * "
    "params.source_row_stride +\n"
    "      id.z * params.source_slice_stride;\n"
    "  let end = target_start + params.row_length;\n"
    "  let word_count = (params.row_length + (target_start & 3u) + 3u) / 4u;\n"
    "  var ordinal = id.x;\n"
    "  loop {\n"
    "    if (ordinal >= word_count) { break; }\n"
    "    let word = (target_start >> 2u) + ordinal;\n"
    "    let position = word << 2u;\n"
    "    let lo = max(position, target_start);\n"
    "    let hi = min(position + 4u, end);\n"
    "    let mask = (0xffffffffu << ((lo - position) * 8u)) &\n"
    "        (0xffffffffu >> ((position + 4u - hi) * 8u));\n"
    "    let shift = ((position - target_start) & 3u) * 8u;\n"
    "    let value = (params.pattern >> shift) |\n"
    "        (params.pattern << ((32u - shift) & 31u));\n"
    "    store_word(word, value, mask);\n"
    "    ordinal += groups.x * 256u;\n"
    "  }\n"
    "}\n";

static const char iree_hal_webgpu_builtin_copy_wgsl[] =
    "struct Params {\n"
    "  source_offset: u32, target_offset: u32, row_length: u32, flags: u32,\n"
    "  source_row_stride: u32, target_row_stride: u32,\n"
    "  source_slice_stride: u32, target_slice_stride: u32,\n"
    "  row_count: u32, slice_count: u32, pattern: u32, padding: u32,\n"
    "};\n"
    "@group(0) @binding(0) var<storage> source: array<u32>;\n"
    "@group(0) @binding(1) var<storage, read_write> target: "
    "array<atomic<u32>>;\n"
    "@group(0) @binding(2) var<uniform> params: Params;\n"
    "fn load_word(index: u32) -> u32 {\n"
    "  if ((params.flags & 1u) != 0u) { return atomicLoad(&target[index]); }\n"
    "  return source[index];\n"
    "}\n"
    "fn store_word(index: u32, value: u32, mask: u32) {\n"
    "  if (mask == 0xffffffffu) { atomicStore(&target[index], value); return; "
    "}\n"
    "  var previous = atomicLoad(&target[index]);\n"
    "  loop {\n"
    "    let result = atomicCompareExchangeWeak(&target[index], previous,\n"
    "        (previous & ~mask) | (value & mask));\n"
    "    if (result.exchanged) { break; }\n"
    "    previous = result.old_value;\n"
    "  }\n"
    "}\n"
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) id: vec3<u32>,\n"
    "        @builtin(num_workgroups) groups: vec3<u32>) {\n"
    "  let target_start = params.target_offset + id.y * "
    "params.target_row_stride +\n"
    "      id.z * params.target_slice_stride;\n"
    "  let source_start = params.source_offset + id.y * "
    "params.source_row_stride +\n"
    "      id.z * params.source_slice_stride;\n"
    "  let end = target_start + params.row_length;\n"
    "  let word_count = (params.row_length + (target_start & 3u) + 3u) / 4u;\n"
    "  var ordinal = id.x;\n"
    "  loop {\n"
    "    if (ordinal >= word_count) { break; }\n"
    "    let word = (target_start >> 2u) + ordinal;\n"
    "    let position = word << 2u;\n"
    "    let lo = max(position, target_start);\n"
    "    let hi = min(position + 4u, end);\n"
    "    let mask = (0xffffffffu << ((lo - position) * 8u)) &\n"
    "        (0xffffffffu >> ((position + 4u - hi) * 8u));\n"
    "    var value = 0u;\n"
    "    if (mask == 0xffffffffu) {\n"
    "      let source_position = source_start + (position - target_start);\n"
    "      let source_word = source_position >> 2u;\n"
    "      let shift = (source_position & 3u) * 8u;\n"
    "      value = load_word(source_word);\n"
    "      if (shift != 0u) {\n"
    "        value = (value >> shift) | (load_word(source_word + 1u) << (32u - "
    "shift));\n"
    "      }\n"
    "    } else {\n"
    "      for (var byte = lo; byte < hi; byte++) {\n"
    "        let source_position = source_start + (byte - target_start);\n"
    "        let source_byte = (load_word(source_position >> 2u) >>\n"
    "            ((source_position & 3u) * 8u)) & 0xffu;\n"
    "        value |= source_byte << ((byte & 3u) * 8u);\n"
    "      }\n"
    "    }\n"
    "    store_word(word, value, mask);\n"
    "    ordinal += groups.x * 256u;\n"
    "  }\n"
    "}\n";

//===----------------------------------------------------------------------===//
// Bind group layout entry (matches bridge protocol)
//===----------------------------------------------------------------------===//

// Wire format for a single bind group layout entry as expected by
// device_create_bind_group_layout. 6 uint32_t values = 24 bytes.
typedef struct iree_hal_webgpu_bind_group_layout_entry_t {
  uint32_t binding;
  uint32_t visibility;
  uint32_t buffer_type;
  uint32_t has_dynamic_offset;
  uint32_t min_binding_size_lo;
  uint32_t min_binding_size_hi;
} iree_hal_webgpu_bind_group_layout_entry_t;

//===----------------------------------------------------------------------===//
// Pipeline creation helpers
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_webgpu_builtins_create_pipeline(
    iree_hal_webgpu_handle_t device_handle, const char* wgsl_source,
    iree_host_size_t wgsl_length, const char* entry_point,
    const iree_hal_webgpu_bind_group_layout_entry_t* layout_entries,
    iree_host_size_t layout_entry_count, iree_hal_webgpu_handle_t* out_pipeline,
    iree_hal_webgpu_handle_t* out_bind_group_layout) {
  // Create the bind group layout.
  iree_hal_webgpu_handle_t bind_group_layout =
      iree_hal_webgpu_import_device_create_bind_group_layout(
          device_handle, (uint32_t)(uintptr_t)layout_entries,
          (uint32_t)layout_entry_count);
  if (!bind_group_layout) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "device.createBindGroupLayout() returned null for builtin shader");
  }

  // Create the pipeline layout from the single bind group layout.
  iree_hal_webgpu_handle_t pipeline_layout =
      iree_hal_webgpu_import_device_create_pipeline_layout(
          device_handle, (uint32_t)(uintptr_t)&bind_group_layout,
          /*layout_count=*/1);
  if (!pipeline_layout) {
    iree_hal_webgpu_import_handle_release(bind_group_layout);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "device.createPipelineLayout() returned null for builtin shader");
  }

  // Create the compute pipeline.
  iree_hal_webgpu_handle_t pipeline =
      iree_hal_webgpu_import_device_create_compute_pipeline(
          device_handle, pipeline_layout, (uint32_t)(uintptr_t)wgsl_source,
          (uint32_t)wgsl_length, (uint32_t)(uintptr_t)entry_point,
          (uint32_t)strlen(entry_point));

  // The pipeline layout handle is consumed by the pipeline — release ours.
  iree_hal_webgpu_import_handle_release(pipeline_layout);

  if (!pipeline) {
    iree_hal_webgpu_import_handle_release(bind_group_layout);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "device.createComputePipeline() returned null for builtin shader");
  }

  *out_pipeline = pipeline;
  *out_bind_group_layout = bind_group_layout;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_webgpu_builtins_initialize(
    iree_hal_webgpu_handle_t device_handle,
    iree_hal_webgpu_builtins_t* out_builtins) {
  IREE_ASSERT_ARGUMENT(out_builtins);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_builtins, 0, sizeof(*out_builtins));

  // Fill shader: binding 0 = storage(rw), binding 1 = uniform(48 bytes).
  iree_hal_webgpu_bind_group_layout_entry_t fill_entries[2] = {
      {0, IREE_HAL_WEBGPU_SHADER_STAGE_COMPUTE,
       IREE_HAL_WEBGPU_BUFFER_BINDING_TYPE_STORAGE, 0, 0, 0},
      {1, IREE_HAL_WEBGPU_SHADER_STAGE_COMPUTE,
       IREE_HAL_WEBGPU_BUFFER_BINDING_TYPE_UNIFORM, 0, 48, 0},
  };
  iree_status_t status = iree_hal_webgpu_builtins_create_pipeline(
      device_handle, iree_hal_webgpu_builtin_fill_wgsl,
      strlen(iree_hal_webgpu_builtin_fill_wgsl), "main", fill_entries,
      IREE_ARRAYSIZE(fill_entries), &out_builtins->fill_pipeline,
      &out_builtins->fill_bind_group_layout);

  // Copy shader: binding 0 = storage(ro), binding 1 = storage(rw),
  //              binding 2 = uniform(48 bytes).
  if (iree_status_is_ok(status)) {
    iree_hal_webgpu_bind_group_layout_entry_t copy_entries[3] = {
        {0, IREE_HAL_WEBGPU_SHADER_STAGE_COMPUTE,
         IREE_HAL_WEBGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE, 0, 0, 0},
        {1, IREE_HAL_WEBGPU_SHADER_STAGE_COMPUTE,
         IREE_HAL_WEBGPU_BUFFER_BINDING_TYPE_STORAGE, 0, 0, 0},
        {2, IREE_HAL_WEBGPU_SHADER_STAGE_COMPUTE,
         IREE_HAL_WEBGPU_BUFFER_BINDING_TYPE_UNIFORM, 0, 48, 0},
    };
    status = iree_hal_webgpu_builtins_create_pipeline(
        device_handle, iree_hal_webgpu_builtin_copy_wgsl,
        strlen(iree_hal_webgpu_builtin_copy_wgsl), "main", copy_entries,
        IREE_ARRAYSIZE(copy_entries), &out_builtins->copy_pipeline,
        &out_builtins->copy_bind_group_layout);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_webgpu_builtins_deinitialize(out_builtins);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_webgpu_builtins_deinitialize(
    iree_hal_webgpu_builtins_t* builtins) {
  iree_hal_webgpu_import_handle_release(builtins->fill_pipeline);
  iree_hal_webgpu_import_handle_release(builtins->fill_bind_group_layout);
  iree_hal_webgpu_import_handle_release(builtins->copy_pipeline);
  iree_hal_webgpu_import_handle_release(builtins->copy_bind_group_layout);
  memset(builtins, 0, sizeof(*builtins));
}
