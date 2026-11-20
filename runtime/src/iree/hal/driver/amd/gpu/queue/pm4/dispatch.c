// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/pm4/dispatch.h"

#include "iree/hal/driver/amd/gpu/queue/pm4/commands.h"

// The relocation-free programs in dispatch_info.loom write the public grid
// in the representation requested by the consumer: workitems in a dispatch
// packet and workgroups in native argument fields. Runtime-only builds use
// the stored ISA; dispatch_program_test checks both ABIs on every family.
static const uint32_t iree_hal_amd_gpu_pm4_dispatch_program_gfx11[] = {
    0xf4080100u, 0xf8000000u, 0x7e000280u, 0xbf89fc07u, 0xdc520006u,
    0x01060000u, 0xdc4a0004u, 0x02060000u, 0xf4000002u, 0xf8000000u,
    0xf4000042u, 0xf8000004u, 0xf4000082u, 0xf8000008u, 0xbf8907f7u,
    0x360602ffu, 0x0000ffffu, 0xd6100001u, 0x02412101u, 0xbf890007u,
    0xd72c0002u, 0x02020400u, 0xbf870193u, 0xd72c0003u, 0x02020601u,
    0xd72c0001u, 0x02020202u, 0xdc6a000cu, 0x00060200u, 0xdc6a0010u,
    0x00060300u, 0xdc6a0014u, 0x00060100u, 0xbc7c0000u, 0xbfb00000u,
};

static const uint32_t iree_hal_amd_gpu_pm4_dispatch_program_gfx12[] = {
    0xf4004100u, 0xf8000000u, 0x7e000280u, 0xbfc70000u, 0xee050006u,
    0x00000001u, 0x00000600u, 0xee048006u, 0x00000002u, 0x00000400u,
    0xf4000002u, 0xf8000000u, 0xf4000042u, 0xf8000004u, 0xf4000082u,
    0xf8000008u, 0xbfc00001u, 0x360602ffu, 0x0000ffffu, 0xd6100001u,
    0x02412101u, 0xbfc00000u, 0xbfc70000u, 0xd72c0002u, 0x02020400u,
    0xbf870193u, 0xd72c0003u, 0x02020601u, 0xd72c0001u, 0x02020202u,
    0xee068006u, 0x01000000u, 0x00000c00u, 0xee068006u, 0x01800000u,
    0x00001000u, 0xee068006u, 0x00800000u, 0x00001400u, 0xbfc10000u,
    0xbfb00000u,
};

static const uint32_t iree_hal_amd_gpu_pm4_dispatch_program_gfx125[] = {
    0xee174000u, 0x00040000u, 0x00000000u, 0x7e000000u, 0xb9800641u,
    0x00000001u, 0xf4004100u, 0xf8000000u, 0x7e000280u, 0xbfc70000u,
    0xee050006u, 0x00000001u, 0x00000600u, 0xee048006u, 0x00000002u,
    0x00000400u, 0xf4000082u, 0xf8000000u, 0xf40000c2u, 0xf8000004u,
    0xf4000202u, 0xf8000008u, 0xbfc00001u, 0x360602ffu, 0x0000ffffu,
    0xd6100001u, 0x02412101u, 0xbfc00000u, 0xbfc70000u, 0xd72c0002u,
    0x02020402u, 0xbf870193u, 0xd72c0003u, 0x02020603u, 0xd72c0001u,
    0x02020208u, 0xee068006u, 0x01000000u, 0x00000c00u, 0xee068006u,
    0x01800000u, 0x00001000u, 0xee068006u, 0x00800000u, 0x00001400u,
    0xbfc10000u, 0xbfb00000u,
};

static const uint32_t iree_hal_amd_gpu_pm4_count_program_gfx11[] = {
    0xf4040080u, 0xf8000010u, 0xf4080100u, 0xf8000000u, 0xbe810080u,
    0xbf89fc07u, 0xbe800001u, 0xbf040300u, 0xbf870009u, 0xbfa10029u,
    0x7e000280u, 0x84888200u, 0x80080802u, 0x82090980u, 0xbe8a0006u,
    0x8008080au, 0xbe8a0007u, 0x8209090au, 0xbc7c0000u, 0xdc520000u,
    0x00080000u, 0xbf8903f7u, 0x32020090u, 0x360400ffu, 0x0000ffffu,
    0x32000092u, 0xbf870003u, 0x36020283u, 0xbe880001u, 0xd441000au,
    0x02020008u, 0xbe8c017eu, 0xbf870001u, 0x8b8a0c0au, 0xbfa1000eu,
    0x7e060208u, 0xbf870131u, 0x4a080701u, 0xd6460003u, 0x04090503u,
    0x80088108u, 0x30080882u, 0xbc7c0000u, 0xdc520000u, 0x04040004u,
    0xbf8903f7u, 0xdc6a0000u, 0x00060403u, 0xbfa0ffecu, 0x80008100u,
    0xbfa0ffd4u, 0xbc7c0000u, 0xbfb00000u,
};

static const uint32_t iree_hal_amd_gpu_pm4_count_program_gfx12[] = {
    0xf4002080u, 0xf8000010u, 0xf4004100u, 0xf8000000u, 0xbe810080u,
    0xbfc70000u, 0xbe800001u, 0xbf040300u, 0xbf870009u, 0xbfa1002fu,
    0x7e000280u, 0x84888200u, 0xbf88fffeu, 0x80080802u, 0x82090980u,
    0xbe8a0006u, 0xbf88fffeu, 0x8008080au, 0xbe8a0007u, 0x8209090au,
    0xbfc10000u, 0xee050008u, 0x00000000u, 0x00000000u, 0xbfc00000u,
    0x32020090u, 0x360400ffu, 0x0000ffffu, 0x32000092u, 0xbf870003u,
    0x36020283u, 0xbe880001u, 0xbf88fffeu, 0xd441000au, 0x02020008u,
    0xbe8c017eu, 0xbf870001u, 0x8b8a0c0au, 0xbfa10010u, 0x7e060208u,
    0xbf870131u, 0x4a080701u, 0xd6460003u, 0x04090503u, 0x80088108u,
    0x30080882u, 0xbfc10000u, 0xee050004u, 0x00000004u, 0x00000004u,
    0xbfc00000u, 0xee068006u, 0x02000000u, 0x00000003u, 0xbfa0ffe9u,
    0x80008100u, 0xbfa0ffceu, 0xbfc10000u, 0xbfb00000u,
};

static const uint32_t iree_hal_amd_gpu_pm4_count_program_gfx125[] = {
    0xee174000u, 0x00040000u, 0x00000000u, 0x7e000000u, 0xb9800641u,
    0x00000001u, 0xf4002080u, 0xf8000010u, 0xf4004100u, 0xf8000000u,
    0xbe890080u, 0xbfc70000u, 0xbe880009u, 0xbf040308u, 0xbf870009u,
    0xbfa1002cu, 0x7e000280u, 0x848a8208u, 0x800a0a02u, 0x820b0b80u,
    0xbe8c0006u, 0x800a0a0cu, 0xbe8c0007u, 0x820b0b0cu, 0xbfc10000u,
    0xee05000au, 0x00000001u, 0x00000000u, 0xbfc00000u, 0x32040290u,
    0x360602ffu, 0x0000ffffu, 0x32020292u, 0xbf870003u, 0x36040483u,
    0xbe8c0009u, 0xd441000eu, 0x0202020cu, 0xbe90017eu, 0xbf870001u,
    0x8b8e100eu, 0xbfa10010u, 0x7e08020cu, 0xbf870131u, 0x4a0a0902u,
    0xd6460004u, 0x040d0504u, 0x800c810cu, 0x300a0a82u, 0xbfc10000u,
    0xee050004u, 0x00000006u, 0x00000005u, 0xbfc00000u, 0xee068006u,
    0x03000000u, 0x00000004u, 0xbfa0ffeau, 0x80088108u, 0xbfa0ffd1u,
    0xbfc10000u, 0xbfb00000u,
};

static iree_status_t iree_hal_amd_gpu_pm4_geometry_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip, bool counts,
    iree_hal_amd_gpu_pm4_dispatch_program_t* out_program) {
  iree_hal_amd_gpu_pm4_dispatch_program_t program = {
      .kernel = {.compute_pgm_rsrc2 = 4, .code_properties = 0x408},
  };
  if (gfxip.major == 11 &&
      (gfxip.minor == 0 || gfxip.minor == 5 || gfxip.minor == 7)) {
    program.code =
        counts ? iree_make_const_byte_span(
                     iree_hal_amd_gpu_pm4_count_program_gfx11,
                     sizeof(iree_hal_amd_gpu_pm4_count_program_gfx11))
               : iree_make_const_byte_span(
                     iree_hal_amd_gpu_pm4_dispatch_program_gfx11,
                     sizeof(iree_hal_amd_gpu_pm4_dispatch_program_gfx11));
    program.kernel.compute_pgm_rsrc1 = 0xe0af0000u;
  } else if (gfxip.major == 12 && gfxip.minor == 0) {
    program.code =
        counts ? iree_make_const_byte_span(
                     iree_hal_amd_gpu_pm4_count_program_gfx12,
                     sizeof(iree_hal_amd_gpu_pm4_count_program_gfx12))
               : iree_make_const_byte_span(
                     iree_hal_amd_gpu_pm4_dispatch_program_gfx12,
                     sizeof(iree_hal_amd_gpu_pm4_dispatch_program_gfx12));
    program.kernel.compute_pgm_rsrc1 = 0xe00f0000u;
  } else if (gfxip.major == 12 && gfxip.minor == 5) {
    program.code =
        counts ? iree_make_const_byte_span(
                     iree_hal_amd_gpu_pm4_count_program_gfx125,
                     sizeof(iree_hal_amd_gpu_pm4_count_program_gfx125))
               : iree_make_const_byte_span(
                     iree_hal_amd_gpu_pm4_dispatch_program_gfx125,
                     sizeof(iree_hal_amd_gpu_pm4_dispatch_program_gfx125));
    program.kernel.compute_pgm_rsrc1 = 0xc00f0000u;
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "dispatch packet preparation is unavailable for "
                            "gfx%u.%u.%u",
                            gfxip.major, gfxip.minor, gfxip.stepping);
  }
  *out_program = program;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_dispatch_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip,
    iree_hal_amd_gpu_pm4_dispatch_program_t* out_program) {
  return iree_hal_amd_gpu_pm4_geometry_program_query(gfxip, false, out_program);
}

iree_status_t iree_hal_amd_gpu_pm4_count_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip,
    iree_hal_amd_gpu_pm4_dispatch_program_t* out_program) {
  return iree_hal_amd_gpu_pm4_geometry_program_query(gfxip, true, out_program);
}

iree_hal_amd_gpu_pm4_dispatch_storage_t iree_hal_amd_gpu_pm4_dispatch_storage(
    const iree_hal_amd_gpu_pm4_function_t* function,
    iree_hal_dispatch_flags_t flags) {
  const bool indirect = iree_hal_dispatch_uses_indirect_arguments(flags);
  iree_hal_amd_gpu_pm4_dispatch_storage_t storage = {
      .length = indirect ? 0 : function->base.kernarg->byte_length,
      .alignment = indirect ? 1 : function->base.kernarg->alignment,
  };
  if (function->packet && iree_hal_dispatch_uses_indirect_parameters(flags)) {
    storage.alignment = iree_max(storage.alignment, 8u);
    storage.prepare_offset = (storage.length + 7u) & ~7u;
    storage.length = storage.prepare_offset + 16u;
  }
  if (!indirect && iree_hal_dispatch_uses_indirect_parameters(flags) &&
      function->base.kernarg->indirect_copy_count) {
    storage.alignment = iree_max(storage.alignment, 8u);
    storage.count_offset = (storage.length + 7u) & ~7u;
    storage.length = storage.count_offset +
                     sizeof(iree_hal_amd_gpu_pm4_count_arguments_t) +
                     4u * function->base.kernarg->indirect_copy_count;
  }
  return storage;
}

iree_status_t iree_hal_amd_gpu_pm4_dispatch_prepare(
    const iree_hal_amd_gpu_pm4_function_t* function,
    uint32_t maximum_group_segment_byte_length,
    uint32_t maximum_scratch_wave_byte_length,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_host_size_t binding_count, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_dispatch_t* out_dispatch) {
  const iree_hal_amd_gpu_metadata_kernel_t* metadata = function->base.metadata;
  uint16_t launch_requirements =
      function->launch.requirements &
      ~(IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_DISPATCH_POINTER |
        IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_TSAN);
  uint32_t scratch_wave_byte_length = 0;
  if (maximum_scratch_wave_byte_length &&
      (launch_requirements & IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_SCRATCH)) {
    const uint64_t required =
        iree_align_uint64((uint64_t)metadata->private_segment_fixed_size
                              << function->launch.wavefront_size_log2,
                          256);
    if (!required || required > maximum_scratch_wave_byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "kernel private storage requires %" PRIu64
                              " bytes per wave; native capacity is %u",
                              required, maximum_scratch_wave_byte_length);
    }
    scratch_wave_byte_length = (uint32_t)required;
    launch_requirements &= ~IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_SCRATCH;
  }
  const uint32_t argument_requirements =
      function->base.kernarg->requirements &
      ~((flags & IREE_HAL_DISPATCH_FLAG_COOPERATIVE) &&
                !iree_hal_dispatch_uses_indirect_arguments(flags)
            ? IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_GRID_SYNC
            : 0u);
  if (launch_requirements || argument_requirements) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "kernel '%.*s' requires native launch services 0x%x and argument "
        "services 0x%x that this queue has not provisioned "
        "(fixed private segment: %u bytes per workitem)",
        (int)metadata->reflection_name.size, metadata->reflection_name.data,
        launch_requirements, argument_requirements,
        metadata->private_segment_fixed_size);
  }
  if (metadata->has_workgroup_cluster_size) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 workgroup clusters require native queue "
                            "admission and cluster launch state");
  }
  iree_hal_amd_gpu_pm4_dispatch_t dispatch = {.scratch_wave_byte_length =
                                                  scratch_wave_byte_length};
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_dispatch_prepare(
      &function->base, maximum_group_segment_byte_length,
      function->packet != NULL, config, constants, binding_count, flags,
      &dispatch.geometry));
  *out_dispatch = dispatch;
  return iree_ok_status();
}

void iree_hal_amd_gpu_pm4_dispatch_emplace_arguments(
    const iree_hal_amd_gpu_pm4_function_t* function,
    const iree_hal_amd_gpu_pm4_dispatch_t* dispatch,
    iree_const_byte_span_t constants, const uint64_t* binding_addresses,
    iree_hal_dispatch_flags_t flags,
    const iree_hal_amd_gpu_pm4_dispatch_storage_t* storage, void* target) {
  iree_hal_amd_gpu_dispatch_emplace_arguments(&function->base,
                                              &dispatch->geometry, constants,
                                              binding_addresses, flags, target);
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    const uint32_t kernarg_length =
        iree_hal_dispatch_uses_indirect_arguments(flags)
            ? 0
            : function->base.kernarg->byte_length;
    if (storage->length > kernarg_length) {
      memset((uint8_t*)target + kernarg_length, 0,
             storage->length - kernarg_length);
    }
    if (storage->count_offset) {
      const iree_hal_amd_gpu_kernarg_layout_t* layout = function->base.kernarg;
      iree_hal_amd_gpu_pm4_count_arguments_t* arguments =
          (void*)((uint8_t*)target + storage->count_offset);
      arguments->table_offset = storage->count_offset + sizeof(*arguments);
      arguments->range_count = layout->indirect_copy_count;
      uint32_t* entries = (uint32_t*)(arguments + 1);
      const iree_hal_amd_gpu_kernarg_copy_t* ranges =
          iree_hal_amd_gpu_kernarg_layout_indirect_copies(layout);
      for (uint16_t i = 0; i < layout->indirect_copy_count; ++i) {
        // The source is the three public DWORDs; destinations are native bytes.
        entries[i] = ranges[i].target_offset |
                     ((uint32_t)(ranges[i].source_offset / 4u) << 16) |
                     ((uint32_t)(ranges[i].length / 4u) << 18);
      }
    }
  }
}

void iree_hal_amd_gpu_pm4_dispatch_emplace_packet(
    const iree_hal_amd_gpu_pm4_function_t* function,
    const iree_hal_amd_gpu_pm4_dispatch_t* dispatch,
    iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_dispatch_packet_t* target) {
  *target = (iree_hal_amd_gpu_pm4_dispatch_packet_t){
      .header = 2,
      .setup = 1,
      .private_segment_size =
          function->base.metadata->private_segment_fixed_size,
      .group_segment_size = dispatch->geometry.group_segment_byte_length,
      .kernel_object = function->packet->kernel_object,
  };
  for (int i = 0; i < 3; ++i) {
    const uint32_t size = dispatch->geometry.workgroup_size[i];
    const uint32_t partial = dispatch->geometry.partial_size[i];
    target->workgroup_size[i] = (uint16_t)size;
    target->grid_size[i] = dispatch->geometry.workgroup_count[i] * size -
                           (partial ? size - partial : 0);
    if (size != 1 || dispatch->geometry.workgroup_count[i] != 1) {
      target->setup = (uint16_t)(i + 1);
    }
  }
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    target->setup = 3;
  }
}

static uint32_t* iree_hal_amd_gpu_pm4_dispatch_emit_writer(
    uint32_t* words, const iree_hal_amd_gpu_pm4_launch_t* prepare,
    uint64_t argument_address) {
  const uint32_t one[3] = {1, 1, 1};
  const uint32_t zero[3] = {0};
  const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
      .kernarg_pointer = argument_address,
  };
  words =
      iree_hal_amd_gpu_pm4_launch_emit(words, prepare, one, zero, 0, &inputs);
  return iree_hal_amd_gpu_pm4_emit_dispatch_direct(words, one,
                                                   prepare->dispatch_initiator);
}

uint32_t* iree_hal_amd_gpu_pm4_dispatch_emit_prepare(
    uint32_t* words, const iree_hal_amd_gpu_pm4_launch_t* packet,
    uint64_t packet_arguments, const iree_hal_amd_gpu_pm4_launch_t* counts,
    uint64_t count_arguments, uint32_t cache_control) {
  if (packet) {
    words = iree_hal_amd_gpu_pm4_dispatch_emit_writer(words, packet,
                                                      packet_arguments);
  }
  if (counts) {
    words = iree_hal_amd_gpu_pm4_dispatch_emit_writer(words, counts,
                                                      count_arguments);
  }
  words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
  return iree_hal_amd_gpu_pm4_emit_acquire(words, cache_control);
}
