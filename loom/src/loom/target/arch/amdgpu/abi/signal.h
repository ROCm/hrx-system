// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU signal ABI constants used by Loom code generation.
//
// This header intentionally mirrors only the device-visible layout facts needed
// by the compiler. It must not include runtime/src/iree/hal/drivers/amdgpu/abi
// headers because their host side includes HSA headers, while the Loom compiler
// should stay independent from HSA except in execution/tooling code.

#ifndef LOOM_TARGET_ARCH_AMDGPU_ABI_SIGNAL_H_
#define LOOM_TARGET_ARCH_AMDGPU_ABI_SIGNAL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// AMD signal kind stored as a 64-bit ABI field in iree_amd_signal_t.
typedef int64_t loom_amdgpu_signal_kind_t;

enum loom_amdgpu_signal_kind_e {
  // Unassigned signal storage.
  LOOM_AMDGPU_SIGNAL_KIND_INVALID = 0,
  // User-defined signal supporting ordinary signal operations.
  LOOM_AMDGPU_SIGNAL_KIND_USER = 1,
  // Agent-defined doorbell signal owned by a hardware queue.
  LOOM_AMDGPU_SIGNAL_KIND_DOORBELL = -1,
};

enum loom_amdgpu_signal_layout_e {
  LOOM_AMDGPU_SIGNAL_BYTE_LENGTH = 64u,
  LOOM_AMDGPU_SIGNAL_KIND_OFFSET = 0u,
  LOOM_AMDGPU_SIGNAL_VALUE_OFFSET = 8u,
  LOOM_AMDGPU_SIGNAL_HARDWARE_DOORBELL_PTR_OFFSET = 8u,
  LOOM_AMDGPU_SIGNAL_EVENT_MAILBOX_PTR_OFFSET = 16u,
  LOOM_AMDGPU_SIGNAL_EVENT_ID_OFFSET = 24u,
  LOOM_AMDGPU_SIGNAL_RESERVED1_OFFSET = 28u,
  LOOM_AMDGPU_SIGNAL_START_TS_OFFSET = 32u,
  LOOM_AMDGPU_SIGNAL_END_TS_OFFSET = 40u,
  LOOM_AMDGPU_SIGNAL_QUEUE_PTR_OFFSET = 48u,
  LOOM_AMDGPU_SIGNAL_RESERVED3_ARRAY_0_OFFSET = 56u,
  LOOM_AMDGPU_SIGNAL_RESERVED3_ARRAY_1_OFFSET = 60u,
};

enum loom_amdgpu_signal_mailbox_e {
  // Mask used by ROCm device-libs for GFX10 mailbox message ids.
  LOOM_AMDGPU_SIGNAL_MAILBOX_MESSAGE_ID_GFX10_MASK = 0x7FFFFFu,
  // Mask used by ROCm device-libs for GFX9, GFX11, and GFX12 message ids.
  LOOM_AMDGPU_SIGNAL_MAILBOX_MESSAGE_ID_GFX9_11_12_MASK = 0xFFFFFFu,
  // Legacy pre-GFX9 mask retained for source-level compatibility.
  LOOM_AMDGPU_SIGNAL_MAILBOX_MESSAGE_ID_LEGACY_MASK = 0xFFu,
  // AMDGPU send-message immediate for host interrupt notification.
  LOOM_AMDGPU_SIGNAL_INTERRUPT_SENDMSG = 1u,
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_ABI_SIGNAL_H_
