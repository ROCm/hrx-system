// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/aql/indirect.h"

// Relocation-free instructions from indirect.loom. The source comparison covers
// every instruction and the complete native descriptor for each target family.
static const uint32_t iree_hal_amd_gpu_aql_indirect_gfx90a[] = {
    0xc00a0100u, 0x00000000u, 0xc00a0200u, 0x00000018u, 0xc0020080u,
    0x00000030u, 0xc0060300u, 0x00000028u, 0xc0060380u, 0x00000010u,
    0x7e000280u, 0xbe8000c1u, 0xbf8cc07fu, 0xc0020043u, 0x00000000u,
    0xc00200c3u, 0x00000004u, 0xc0020403u, 0x00000008u, 0xbf01000cu,
    0xbf8cc07fu, 0x92060901u, 0x92070a03u, 0x92090b10u, 0x7e020206u,
    0x7e040207u, 0x7e060209u, 0xdc70800cu, 0x00040100u, 0xdc708010u,
    0x00040200u, 0xdc708014u, 0x00040300u, 0xbf840005u, 0x80060c0eu,
    0x7e020201u, 0x8207800fu, 0xdc708000u, 0x00060100u, 0xbf01000du,
    0xbf840005u, 0x80060d0eu, 0x7e020203u, 0x8207800fu, 0xdc708000u,
    0x00060100u, 0xbf010002u, 0xbf840005u, 0x8000020eu, 0x8201800fu,
    0x7e020210u, 0xdc708000u, 0x00000100u, 0xbf8cc07fu, 0xbf8c0f70u,
    0xe0a00000u, 0x00000000u, 0x7e020208u, 0xbf8c0f70u, 0xdc708000u,
    0x00040100u, 0xbf8c0f70u, 0xbf810000u,
};

static const uint32_t iree_hal_amd_gpu_aql_indirect_gfx94[] = {
    0xc00a0100u, 0x00000000u, 0xc00a0200u, 0x00000018u, 0xc0020080u,
    0x00000030u, 0xc0060300u, 0x00000028u, 0xc0060380u, 0x00000010u,
    0x7e000280u, 0xbe8000c1u, 0xbf8cc07fu, 0xc0020043u, 0x00000000u,
    0xc00200c3u, 0x00000004u, 0xc0020403u, 0x00000008u, 0xbf01000cu,
    0xbf8cc07fu, 0x92060901u, 0x92070a03u, 0x92090b10u, 0x7e020206u,
    0x7e040207u, 0x7e060209u, 0xdc70800cu, 0x00040100u, 0xdc708010u,
    0x00040200u, 0xdc708014u, 0x00040300u, 0xbf840005u, 0x80060c0eu,
    0x7e020201u, 0x8207800fu, 0xdc708000u, 0x00060100u, 0xbf01000du,
    0xbf840005u, 0x80060d0eu, 0x7e020203u, 0x8207800fu, 0xdc708000u,
    0x00060100u, 0xbf010002u, 0xbf840005u, 0x8000020eu, 0x8201800fu,
    0x7e020210u, 0xdc708000u, 0x00000100u, 0xbf8cc07fu, 0xbf8c0f70u,
    0xe0a0c000u, 0x00000000u, 0x7e020208u, 0xbf8c0f70u, 0xde718000u,
    0x00040100u, 0xbf8c0f70u, 0xbf810000u,
};

static const uint32_t iree_hal_amd_gpu_aql_indirect_gfx11[] = {
    0xf4080100u, 0xf8000000u, 0xf4080200u, 0xf8000018u, 0xf4000080u,
    0xf8000030u, 0xf4040300u, 0xf8000028u, 0xf4040380u, 0xf8000010u,
    0x7e000280u, 0xbe8000c1u, 0xbf89fc07u, 0xf4000043u, 0xf8000000u,
    0xf40000c3u, 0xf8000004u, 0xf4000403u, 0xf8000008u, 0xbf01000cu,
    0xbf89fc07u, 0x96060901u, 0x96070a03u, 0x96090b10u, 0x7e020206u,
    0x7e040207u, 0x7e060209u, 0xdc6a000cu, 0x00040100u, 0xdc6a0010u,
    0x00040200u, 0xdc6a0014u, 0x00040300u, 0xbfa10005u, 0x7e020201u,
    0x80060c0eu, 0x8207800fu, 0xdc6a0000u, 0x00060100u, 0xbf01000du,
    0xbf870009u, 0xbfa10005u, 0x7e020203u, 0x80060d0eu, 0x8207800fu,
    0xdc6a0000u, 0x00060100u, 0xbf010002u, 0xbf870009u, 0xbfa10005u,
    0x7e020210u, 0x8000020eu, 0x8201800fu, 0xdc6a0000u, 0x00000100u,
    0x7e020208u, 0xbf89fc07u, 0xbf8903f7u, 0xbc7c0000u, 0xdc6a0000u,
    0x00040100u, 0xbc7c0000u, 0xbfb00000u,
};

static const uint32_t iree_hal_amd_gpu_aql_indirect_gfx12[] = {
    0xf4004100u, 0xf8000000u, 0xf4004200u, 0xf8000018u, 0xf4000080u,
    0xf8000030u, 0xf4002300u, 0xf8000028u, 0xf4002380u, 0xf8000010u,
    0x7e000280u, 0xbe8000c1u, 0xbfc70000u, 0xf4000043u, 0xf8000000u,
    0xf40000c3u, 0xf8000004u, 0xf4000403u, 0xf8000008u, 0xbf01000cu,
    0xbfc70000u, 0x96060901u, 0x96070a03u, 0x96090b10u, 0x7e020206u,
    0x7e040207u, 0x7e060209u, 0xee068004u, 0x00800000u, 0x00000c00u,
    0xee068004u, 0x01000000u, 0x00001000u, 0xee068004u, 0x01800000u,
    0x00001400u, 0xbfa10006u, 0x7e020201u, 0x80060c0eu, 0x8207800fu,
    0xee068006u, 0x00800000u, 0x00000000u, 0xbf01000du, 0xbf870009u,
    0xbfa10006u, 0x7e020203u, 0x80060d0eu, 0x8207800fu, 0xee068006u,
    0x00800000u, 0x00000000u, 0xbf010002u, 0xbf870009u, 0xbfa10006u,
    0x7e020210u, 0x8000020eu, 0x8201800fu, 0xee068000u, 0x00800000u,
    0x00000000u, 0xbfc60000u, 0xbfc10000u, 0xee0b007cu, 0x000c0000u,
    0x00000000u, 0x7e020208u, 0xbfc10000u, 0xee068004u, 0x008c0000u,
    0x00000000u, 0xbfc10000u, 0xbfb00000u,
};

static const uint32_t iree_hal_amd_gpu_aql_indirect_gfx125[] = {
    0xee174000u, 0x00040000u, 0x00000000u, 0x7e000000u, 0xb9800641u,
    0x00000001u, 0xf4004100u, 0xf8000000u, 0xf4004200u, 0xf8000018u,
    0xf4000080u, 0xf8000030u, 0xf4002300u, 0xf8000028u, 0xf4002380u,
    0xf8000010u, 0x7e000280u, 0xbe8300c1u, 0xbfc70000u, 0xf4000403u,
    0xf8000000u, 0xf4000443u, 0xf8000004u, 0xf4000483u, 0xf8000008u,
    0xbf01030cu, 0xbfc70000u, 0x96090910u, 0x960a0a11u, 0x960b0b12u,
    0x7e020209u, 0x7e04020au, 0x7e06020bu, 0xee068004u, 0x00800000u,
    0x00000c00u, 0xee068004u, 0x01000000u, 0x00001000u, 0xee068004u,
    0x01800000u, 0x00001400u, 0xbfa10006u, 0x7e020210u, 0x80000c0eu,
    0x8201800fu, 0xee068000u, 0x00800000u, 0x00000000u, 0xbf01030du,
    0xbf870009u, 0xbfa10006u, 0x7e040211u, 0x80060d0eu, 0x8207800fu,
    0xee068006u, 0x01000000u, 0x00000000u, 0xbf010302u, 0xbf870009u,
    0xbfa10006u, 0x7e060212u, 0x8002020eu, 0x8203800fu, 0xee068002u,
    0x01800000u, 0x00000000u, 0xbfc60000u, 0xbfc10000u, 0xee0b007cu,
    0x000c0000u, 0x00000000u, 0x7e080208u, 0xbfc10000u, 0xee068004u,
    0x020c0000u, 0x00000000u, 0xbfc10000u, 0xbfb00000u,
};

iree_status_t iree_hal_amd_gpu_aql_indirect_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip, iree_const_byte_span_t* out_code,
    iree_hal_amd_gpu_image_kernel_t* out_kernel) {
  *out_code = iree_const_byte_span_empty();
  *out_kernel = (iree_hal_amd_gpu_image_kernel_t){
      .entry_offset = 256, .compute_pgm_rsrc2 = 4, .code_properties = 0x408};
  if (gfxip.major == 9 && gfxip.minor == 0 && gfxip.stepping == 10) {
    *out_code =
        iree_make_const_byte_span(iree_hal_amd_gpu_aql_indirect_gfx90a,
                                  sizeof(iree_hal_amd_gpu_aql_indirect_gfx90a));
    out_kernel->compute_pgm_rsrc1 = 0xaf0080;
    out_kernel->compute_pgm_rsrc2 = 8;
    out_kernel->code_properties = 8;
  } else if (gfxip.major == 9 && (gfxip.minor == 4 || gfxip.minor == 5)) {
    *out_code =
        iree_make_const_byte_span(iree_hal_amd_gpu_aql_indirect_gfx94,
                                  sizeof(iree_hal_amd_gpu_aql_indirect_gfx94));
    out_kernel->compute_pgm_rsrc1 = 0xaf0080;
    out_kernel->code_properties = 8;
  } else if (gfxip.major == 11 &&
             (gfxip.minor == 0 || gfxip.minor == 5 || gfxip.minor == 7)) {
    *out_code =
        iree_make_const_byte_span(iree_hal_amd_gpu_aql_indirect_gfx11,
                                  sizeof(iree_hal_amd_gpu_aql_indirect_gfx11));
    out_kernel->compute_pgm_rsrc1 = 0xe0af0000;
  } else if (gfxip.major == 12 && gfxip.minor == 0) {
    *out_code =
        iree_make_const_byte_span(iree_hal_amd_gpu_aql_indirect_gfx12,
                                  sizeof(iree_hal_amd_gpu_aql_indirect_gfx12));
    out_kernel->compute_pgm_rsrc1 = 0xe00f0000;
  } else if (gfxip.major == 12 && gfxip.minor == 5) {
    *out_code =
        iree_make_const_byte_span(iree_hal_amd_gpu_aql_indirect_gfx125,
                                  sizeof(iree_hal_amd_gpu_aql_indirect_gfx125));
    out_kernel->compute_pgm_rsrc1 = 0xc00f0000;
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AQL indirect publication is unavailable for "
                            "gfx%u.%u.%u",
                            gfxip.major, gfxip.minor, gfxip.stepping);
  }
  return iree_ok_status();
}

void iree_hal_amd_gpu_aql_indirect_prepare(
    const iree_hal_amd_gpu_kernarg_layout_t* layout,
    const iree_hal_amd_gpu_dispatch_geometry_t* geometry,
    iree_hal_dispatch_flags_t flags, uint64_t packet, uint64_t counts,
    uint64_t arguments, uint32_t header,
    iree_hal_amd_gpu_aql_indirect_arguments_t* out_arguments) {
  *out_arguments = (iree_hal_amd_gpu_aql_indirect_arguments_t){
      .packet = packet,
      .counts = counts,
      .arguments = arguments,
      .header = header,
      .workgroup_size = {geometry->workgroup_size[0],
                         geometry->workgroup_size[1],
                         geometry->workgroup_size[2]},
      .count_offsets = {UINT32_MAX, UINT32_MAX, UINT32_MAX},
  };
  if (!iree_hal_dispatch_uses_indirect_arguments(flags)) {
    const iree_hal_amd_gpu_kernarg_copy_t* copies =
        iree_hal_amd_gpu_kernarg_layout_indirect_copies(layout);
    for (uint16_t i = 0; i < layout->indirect_copy_count; ++i) {
      for (uint16_t offset = 0; offset < copies[i].length; offset += 4) {
        out_arguments->count_offsets[(copies[i].source_offset + offset) / 4] =
            copies[i].target_offset + offset;
      }
    }
  }
}

iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_indirect_packet(
    uint64_t kernel_object, uint64_t arguments) {
  const iree_hal_amd_gpu_aql_dispatch_t dispatch = {
      .header = IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH |
                IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM | (1u << 16),
      .workgroup_size = {1, 1, 1},
      .grid_size = {1, 1, 1},
      .kernel_object = kernel_object,
      .kernarg_address = arguments,
  };
  return iree_hal_amd_gpu_aql_packet_dispatch(&dispatch);
}
