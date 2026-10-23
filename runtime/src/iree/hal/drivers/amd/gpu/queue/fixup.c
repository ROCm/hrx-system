// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/fixup.h"

void iree_hal_amd_gpu_fixup_apply(const iree_hal_amd_gpu_fixup_t* entries,
                                  uint32_t entry_count, const uint64_t* values,
                                  void* target) {
  for (uint32_t i = 0; i < entry_count; ++i) {
    const iree_hal_amd_gpu_fixup_t entry = entries[i];
    iree_unaligned_store_le_u32(
        (uint8_t*)target + entry.target_offset,
        (uint32_t)((values[entry.source & 0xFFFFFFu] + entry.addend) >>
                   ((entry.source >> 24) & 63)));
  }
}

// Relocation-free instruction words from fixup.loom. The source comparison
// covers the complete code and descriptor ABI for every supported family.
static const uint32_t iree_hal_amd_gpu_fixup_gfx90a[] = {
    0xc0020080u, 0x00000018u, 0xc0060180u, 0x00000010u, 0xc00a0200u,
    0x00000000u, 0x9200c004u, 0x68020000u, 0xbf8cc07fu, 0xd0c90000u,
    0x00000501u, 0xbe802000u, 0xbf84002eu, 0xbe850080u, 0x8e828a04u,
    0x80020208u, 0x82030309u, 0x8e8c8a04u, 0x800c0c08u, 0x820d0d09u,
    0x24000084u, 0xdc508004u, 0x010c0000u, 0x8e8c8a04u, 0x800c0c08u,
    0x820d0d09u, 0x8e848a04u, 0x80040408u, 0x82050509u, 0xdc50800cu,
    0x02040000u, 0xdc508008u, 0x040c0000u, 0xbf800000u, 0xdc508000u,
    0x00020000u, 0x7e060280u, 0xbf8c0f73u, 0x260a02ffu, 0x00ffffffu,
    0x240a0a83u, 0xdc548000u, 0x060a0005u, 0x20020298u, 0xbf8c0f73u,
    0xd28f0002u, 0x000204a0u, 0xbf8c0f72u, 0x28040504u, 0x28060680u,
    0xbf8c0f70u, 0xd1190202u, 0x00020506u, 0xd11c0203u, 0x000a0707u,
    0xd2900002u, 0x00020501u, 0xdc708000u, 0x00060200u, 0xbefe0100u,
    0xbf8c0f70u, 0xbf810000u,
};

static const uint32_t iree_hal_amd_gpu_fixup_gfx94[] = {
    0xc00200c0u, 0x00000018u, 0xc0060100u, 0x00000010u, 0xc00a0200u,
    0x00000000u, 0x9200c002u, 0x68020000u, 0xbf8cc07fu, 0xd0c90000u,
    0x00000701u, 0xbe802000u, 0xbf84002fu, 0xbe830080u, 0x8e868a02u,
    0x80060608u, 0x82070709u, 0x8e8c8a02u, 0x800c0c08u, 0x820d0d09u,
    0x24000084u, 0xdc508004u, 0x010c0000u, 0x8e8c8a02u, 0x800c0c08u,
    0x820d0d09u, 0x8e828a02u, 0x80020208u, 0x82030309u, 0xdc50800cu,
    0x02020000u, 0xdc508008u, 0x040c0000u, 0xbf800000u, 0xdc508000u,
    0x00060000u, 0x7e060280u, 0xbf8c0f73u, 0x260a02ffu, 0x00ffffffu,
    0x240a0a83u, 0xdc548000u, 0x060a0005u, 0x20020298u, 0xbf8c0f73u,
    0xd28f0002u, 0x000204a0u, 0xbf8c0f72u, 0x28040504u, 0x28060680u,
    0xbf8c0f70u, 0xd1190202u, 0x00020506u, 0xbf800001u, 0xd11c0203u,
    0x000a0707u, 0xd2900002u, 0x00020501u, 0xdc708000u, 0x00040200u,
    0xbefe0100u, 0xbf8c0f70u, 0xbf810000u,
};

static const uint32_t iree_hal_amd_gpu_fixup_gfx11[] = {
    0xf40000c0u, 0xf8000018u, 0x9604c002u, 0x4a020004u, 0xf4040100u,
    0xf8000010u, 0xf4080200u, 0xf8000000u, 0xbf89fc07u, 0xbf870091u,
    0xd4490000u, 0x02000701u, 0xbe802100u, 0xbfa10031u, 0xbe830080u,
    0x84868a02u, 0x30000084u, 0x80060608u, 0x82070709u, 0x848c8a02u,
    0x800c0c08u, 0x820d0d09u, 0xdc520004u, 0x010c0000u, 0x848c8a02u,
    0x800c0c08u, 0x820d0d09u, 0x84828a02u, 0x80020208u, 0x82030309u,
    0xdc52000cu, 0x02020000u, 0xdc520008u, 0x040c0000u, 0xdc520000u,
    0x00060000u, 0x7e060280u, 0xbf890ff7u, 0x360a02ffu, 0x00ffffffu,
    0x32020298u, 0xbf870222u, 0x300a0a83u, 0xbf890bf7u, 0xd73c0002u,
    0x020204a0u, 0xdc560000u, 0x060a0005u, 0xbf890bf7u, 0x38040504u,
    0x38060680u, 0xbf8903f7u, 0xbf870002u, 0xd7000202u, 0x02020506u,
    0xbf870101u, 0xd5200203u, 0x000a0707u, 0xbf870001u, 0xd73d0002u,
    0x02020501u, 0xdc6a0000u, 0x00040200u, 0xbefe0100u, 0xbc7c0000u,
    0xbfb00000u,
};

static const uint32_t iree_hal_amd_gpu_fixup_gfx12[] = {
    0xf4000080u, 0xf8000018u, 0x9603c075u, 0x4a020003u, 0xf4002100u,
    0xf8000010u, 0xf4004200u, 0xf8000000u, 0xbfc70000u, 0xbf870091u,
    0xd4490000u, 0x02000501u, 0xbe802100u, 0xbfa10041u, 0xbe830080u,
    0xbe820075u, 0xbf88fffeu, 0x84868a02u, 0x30000084u, 0x80020608u,
    0x82060709u, 0xbe8c0075u, 0xbe8d0003u, 0x848c8a0cu, 0x800c0c08u,
    0x820d0d09u, 0xee05000cu, 0x00000001u, 0x00000400u, 0xbe8c0075u,
    0xbe8d0003u, 0x848c8a0cu, 0x800c0c08u, 0x820d0d09u, 0xbe8e0075u,
    0xbe8f0003u, 0x848e8a0eu, 0x80080e08u, 0x82090f09u, 0xee050008u,
    0x00000002u, 0x00000c00u, 0xee05000cu, 0x00000004u, 0x00000800u,
    0xbe830006u, 0xee050002u, 0x00000000u, 0x00000000u, 0x7e060280u,
    0xbfc00003u, 0x360a02ffu, 0x00ffffffu, 0x32020298u, 0xbf870222u,
    0x300a0a83u, 0xbfc00002u, 0xd51f0002u, 0x020204a0u, 0xee05400au,
    0x00000006u, 0x00000005u, 0xbfc00002u, 0x38040504u, 0x38060680u,
    0xbfc00000u, 0xbf870002u, 0xd7000202u, 0x02020506u, 0xbf870101u,
    0xbf88f1feu, 0xd5200203u, 0x000a0707u, 0xbf870001u, 0xd73d0002u,
    0x02020501u, 0xee068004u, 0x01000000u, 0x00000000u, 0xbefe0100u,
    0xbfc10000u, 0xbfb00000u,
};

static const uint32_t iree_hal_amd_gpu_fixup_gfx125[] = {
    0xee174000u, 0x00040000u, 0x00000000u, 0x7e000000u, 0xb9800641u,
    0x00000001u, 0xf4000080u, 0xf8000018u, 0x9603c075u, 0x4a020003u,
    0xf4002100u, 0xf8000010u, 0xf4004200u, 0xf8000000u, 0xbfc70000u,
    0xbf870091u, 0xd4490002u, 0x02000501u, 0xbe822102u, 0xbfa1003fu,
    0xbe810080u, 0xbe800075u, 0x84868a00u, 0x30000084u, 0x80000608u,
    0x82060709u, 0xbe8c0075u, 0xbe8d0001u, 0x848c8a0cu, 0x800c0c08u,
    0x820d0d09u, 0xee05000cu, 0x00000001u, 0x00000400u, 0xbe8e0075u,
    0xbe8f0001u, 0x848e8a0eu, 0x800e0e08u, 0x820f0f09u, 0xbe900075u,
    0xbe910001u, 0x84908a10u, 0x80081008u, 0x82091109u, 0xee050008u,
    0x00000002u, 0x00000c00u, 0xee05000eu, 0x00000004u, 0x00000800u,
    0xbe810006u, 0xee050000u, 0x00000005u, 0x00000000u, 0x7e060280u,
    0xbfc00003u, 0x360c02ffu, 0x00ffffffu, 0x32020298u, 0xbf870222u,
    0x300c0c83u, 0xbfc00002u, 0xd51f0002u, 0x020204a0u, 0xee05400au,
    0x00000008u, 0x00000006u, 0xbfc00002u, 0x38040504u, 0x38060680u,
    0xbfc00000u, 0xbf870002u, 0xd7000602u, 0x02020508u, 0xbf870101u,
    0xd5200603u, 0x001a0709u, 0xbf870001u, 0xd73d0002u, 0x02020501u,
    0xee068004u, 0x01000000u, 0x00000005u, 0xbfc50000u, 0xbefe0102u,
    0xbfc10000u, 0xbfb00000u,
};

iree_status_t iree_hal_amd_gpu_fixup_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip, iree_const_byte_span_t* out_code,
    iree_hal_amd_gpu_image_kernel_t* out_kernel) {
  *out_code = iree_const_byte_span_empty();
  *out_kernel = (iree_hal_amd_gpu_image_kernel_t){0};
  if (gfxip.major == 9 && gfxip.minor == 0 && gfxip.stepping == 10) {
    *out_code = iree_make_const_byte_span(
        iree_hal_amd_gpu_fixup_gfx90a, sizeof(iree_hal_amd_gpu_fixup_gfx90a));
    *out_kernel = (iree_hal_amd_gpu_image_kernel_t){
        .compute_pgm_rsrc3 = 0x1u,
        .compute_pgm_rsrc1 = 0xaf0080u,
        .compute_pgm_rsrc2 = 0x88u,
        .code_properties = 0x8u,
    };
  } else if (gfxip.major == 9 && (gfxip.minor == 4 || gfxip.minor == 5)) {
    *out_code = iree_make_const_byte_span(iree_hal_amd_gpu_fixup_gfx94,
                                          sizeof(iree_hal_amd_gpu_fixup_gfx94));
    *out_kernel = (iree_hal_amd_gpu_image_kernel_t){
        .compute_pgm_rsrc3 = 0x1u,
        .compute_pgm_rsrc1 = 0xaf0080u,
        .compute_pgm_rsrc2 = 0x84u,
        .code_properties = 0x8u,
    };
  } else if (gfxip.major == 11 &&
             (gfxip.minor == 0 || gfxip.minor == 5 || gfxip.minor == 7)) {
    *out_code = iree_make_const_byte_span(iree_hal_amd_gpu_fixup_gfx11,
                                          sizeof(iree_hal_amd_gpu_fixup_gfx11));
    *out_kernel = (iree_hal_amd_gpu_image_kernel_t){
        .compute_pgm_rsrc1 = 0xe0af0000u,
        .compute_pgm_rsrc2 = 0x84u,
        .code_properties = 0x408u,
    };
  } else if (gfxip.major == 12 && gfxip.minor == 0) {
    *out_code = iree_make_const_byte_span(iree_hal_amd_gpu_fixup_gfx12,
                                          sizeof(iree_hal_amd_gpu_fixup_gfx12));
    *out_kernel = (iree_hal_amd_gpu_image_kernel_t){
        .compute_pgm_rsrc1 = 0xe00f0000u,
        .compute_pgm_rsrc2 = 0x84u,
        .code_properties = 0x408u,
    };
  } else if (gfxip.major == 12 && gfxip.minor == 5) {
    *out_code = iree_make_const_byte_span(
        iree_hal_amd_gpu_fixup_gfx125, sizeof(iree_hal_amd_gpu_fixup_gfx125));
    *out_kernel = (iree_hal_amd_gpu_image_kernel_t){
        .compute_pgm_rsrc1 = 0xc00f0000u,
        .compute_pgm_rsrc2 = 0x84u,
        .code_properties = 0x408u,
    };
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "fixup builtin is unavailable for gfx%u.%u.%u",
                            gfxip.major, gfxip.minor, gfxip.stepping);
  }
  return iree_ok_status();
}
