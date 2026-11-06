// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/transfer.h"

#include "iree/hal/driver/amd/gpu/queue/sdma/commands.h"

iree_hal_amd_gpu_dma_transfer_encoding_t
iree_hal_amd_gpu_sdma_transfer_encoding(amdf_queue_format_features_t features) {
  return (iree_hal_amd_gpu_dma_transfer_encoding_t){
      .features = features,
      .max_packet_length = IREE_HAL_AMD_GPU_SDMA_MAX_PACKET_LENGTH,
      .max_packet_words = 7,
      .copy = iree_hal_amd_gpu_sdma_emit_copy,
      .fill = iree_hal_amd_gpu_sdma_emit_fill,
  };
}
