// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/fixup.h"

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"

// Relocation-free instruction words from fixup.loom. The source comparison
// covers the complete code and descriptor ABI for every supported family.
static const uint32_t iree_hal_amd_gpu_pm4_fixup_gfx11[] = {
    0xf40000c0u, 0xf8000018u, 0x9604c002u, 0x4a020004u, 0xf4040100u,
    0xf8000010u, 0xf4080200u, 0xf8000000u, 0xbf89fc07u, 0xbf870091u,
    0xd4490000u, 0x02000701u, 0xbe802100u, 0xbfa10032u, 0xbe830080u,
    0x84868a02u, 0x30000084u, 0x80060608u, 0x82070709u, 0x848c8a02u,
    0x800c0c08u, 0xdc520000u, 0x01060000u, 0x820d0d09u, 0xdc520004u,
    0x020c0000u, 0xbe860002u, 0xbe870003u, 0x84828a02u, 0x80020208u,
    0x82030309u, 0xdc520008u, 0x03020000u, 0x84828a06u, 0x80020208u,
    0x82030309u, 0xdc52000cu, 0x04020000u, 0x7e0a0280u, 0xbf890bf7u,
    0x360004ffu, 0x00ffffffu, 0x32040498u, 0xbf870222u, 0x30000083u,
    0xbf8903f7u, 0xd73c0004u, 0x020208a0u, 0xdc560000u, 0x060a0000u,
    0x38000903u, 0x38060a80u, 0xbf8903f7u, 0xbf870002u, 0xd7000204u,
    0x02020106u, 0xbf870101u, 0xd5200205u, 0x000a0707u, 0xbf870001u,
    0xd73d0002u, 0x02020902u, 0xdc6a0000u, 0x00040201u, 0xbefe0100u,
    0xbc7c0000u, 0xbfb00000u,
};

static const uint32_t iree_hal_amd_gpu_pm4_fixup_gfx12[] = {
    0xf4000080u, 0xf8000018u, 0x9603c075u, 0x4a020003u, 0xf4002100u,
    0xf8000010u, 0xf4004200u, 0xf8000000u, 0xbfc70000u, 0xbf870091u,
    0xd4490000u, 0x02000501u, 0xbe802100u, 0xbfa1003eu, 0xbe820080u,
    0xbe860075u, 0xbe870002u, 0x84868a06u, 0x30000084u, 0x80060608u,
    0x82070709u, 0xbe8c0075u, 0xbe8d0002u, 0x848c8a0cu, 0x800c0c08u,
    0xee050006u, 0x00000001u, 0x00000000u, 0x820d0d09u, 0xee05000cu,
    0x00000002u, 0x00000400u, 0xbe860075u, 0xbe870002u, 0xbe830002u,
    0xbe820075u, 0x84868a06u, 0x80060608u, 0x82070709u, 0xee050006u,
    0x00000003u, 0x00000800u, 0x84828a02u, 0x80020208u, 0x82030309u,
    0xee050002u, 0x00000004u, 0x00000c00u, 0x7e0a0280u, 0xbfc00002u,
    0x360004ffu, 0x00ffffffu, 0x32040498u, 0xbf870222u, 0x30000083u,
    0xbfc00000u, 0xd51f0004u, 0x020208a0u, 0xee05400au, 0x00000006u,
    0x00000000u, 0x38000903u, 0x38060a80u, 0xbfc00000u, 0xbf870002u,
    0xd7000204u, 0x02020106u, 0xbf870101u, 0xd5200205u, 0x000a0707u,
    0xbf870001u, 0xd73d0002u, 0x02020902u, 0xee068004u, 0x01000000u,
    0x00000001u, 0xbefe0100u, 0xbfc10000u, 0xbfb00000u,
};

static const uint32_t iree_hal_amd_gpu_pm4_fixup_gfx125[] = {
    0xee174000u, 0x00040000u, 0x00000000u, 0x7e000000u, 0xb9800641u,
    0x00000001u, 0xf4000080u, 0xf8000018u, 0x9603c075u, 0x4a020003u,
    0xf4002100u, 0xf8000010u, 0xf4004200u, 0xf8000000u, 0xbfc70000u,
    0xbf870091u, 0xd4490002u, 0x02000501u, 0xbe822102u, 0xbfa1003eu,
    0xbe800080u, 0xbe860075u, 0xbe870000u, 0x84868a06u, 0x30000084u,
    0x80060608u, 0x82070709u, 0xbe8c0075u, 0xbe8d0000u, 0x848c8a0cu,
    0x800c0c08u, 0xee050006u, 0x00000001u, 0x00000000u, 0x820d0d09u,
    0xee05000cu, 0x00000002u, 0x00000400u, 0xbe8e0075u, 0xbe8f0000u,
    0xbe810000u, 0xbe800075u, 0x848e8a0eu, 0x800e0e08u, 0x820f0f09u,
    0xee05000eu, 0x00000003u, 0x00000800u, 0x84808a00u, 0x80000008u,
    0x82010109u, 0xee050000u, 0x00000004u, 0x00000c00u, 0x7e0a0280u,
    0xbfc00002u, 0x360c04ffu, 0x00ffffffu, 0x32040498u, 0xbf870222u,
    0x300c0c83u, 0xbfc00000u, 0xd51f0004u, 0x020208a0u, 0xee05400au,
    0x00000008u, 0x00000006u, 0x38060903u, 0x38080a80u, 0xbfc00000u,
    0xbf870002u, 0xd7000808u, 0x02020708u, 0xbf870101u, 0xd5200809u,
    0x00220909u, 0xbf870001u, 0xd73d0002u, 0x02021102u, 0xee068004u,
    0x01000000u, 0x00000001u, 0xbfc50000u, 0xbefe0102u, 0xbfc10000u,
    0xbfb00000u,
};

iree_status_t iree_hal_amd_gpu_pm4_fixup_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip, iree_const_byte_span_t* out_code,
    iree_hal_amd_gpu_image_kernel_t* out_kernel) {
  *out_code = iree_const_byte_span_empty();
  *out_kernel = (iree_hal_amd_gpu_image_kernel_t){.compute_pgm_rsrc2 = 0x84,
                                                  .code_properties = 0x408};
  if (gfxip.major == 11 &&
      (gfxip.minor == 0 || gfxip.minor == 5 || gfxip.minor == 7)) {
    *out_code =
        iree_make_const_byte_span(iree_hal_amd_gpu_pm4_fixup_gfx11,
                                  sizeof(iree_hal_amd_gpu_pm4_fixup_gfx11));
    out_kernel->compute_pgm_rsrc1 = 0xe0af0000u;
  } else if (gfxip.major == 12 && gfxip.minor == 0) {
    *out_code =
        iree_make_const_byte_span(iree_hal_amd_gpu_pm4_fixup_gfx12,
                                  sizeof(iree_hal_amd_gpu_pm4_fixup_gfx12));
    out_kernel->compute_pgm_rsrc1 = 0xe00f0000u;
  } else if (gfxip.major == 12 && gfxip.minor == 5) {
    *out_code =
        iree_make_const_byte_span(iree_hal_amd_gpu_pm4_fixup_gfx125,
                                  sizeof(iree_hal_amd_gpu_pm4_fixup_gfx125));
    out_kernel->compute_pgm_rsrc1 = 0xc00f0000u;
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "resident program fixups are unavailable for "
                            "gfx%u.%u.%u",
                            gfxip.major, gfxip.minor, gfxip.stepping);
  }
  return iree_ok_status();
}

uint32_t* iree_hal_amd_gpu_pm4_fixup_emit(
    uint32_t* words, const iree_hal_amd_gpu_pm4_launch_t* launch,
    uint64_t arguments_address, uint32_t entry_count) {
  const uint32_t size[3] = {64, 1, 1};
  const uint32_t partial[3] = {0, 0, 0};
  const uint32_t groups[3] = {(entry_count - 1) / 64 + 1, 1, 1};
  const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {.kernarg_pointer =
                                                           arguments_address};
  words = iree_hal_amd_gpu_pm4_launch_emit(words, launch, size, partial, 0,
                                           &inputs);
  return iree_hal_amd_gpu_pm4_emit_dispatch_direct(words, groups,
                                                   launch->dispatch_initiator);
}
