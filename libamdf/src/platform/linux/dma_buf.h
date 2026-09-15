// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_LINUX_DMA_BUF_H_
#define AMDF_SRC_PLATFORM_LINUX_DMA_BUF_H_

#include "amdf/amdf.h"

// Immutable kernel file facts for one live DMA-BUF.
typedef struct amdf_linux_dma_buf_info_t {
  // Identity of the DMA-BUF pseudo-file while any reference remains live.
  amdf_physical_memory_id_t physical_backing_id;
  // Complete physical backing extent reported by the DMA-BUF inode.
  uint64_t byte_length;
} amdf_linux_dma_buf_info_t;

#ifdef __cplusplus
extern "C" {
#endif

// Queries one borrowed DMA-BUF descriptor. Failure leaves `out_info`
// byte-for-byte unchanged.
amdf_status_t amdf_linux_dma_buf_query(int descriptor,
                                       amdf_linux_dma_buf_info_t* out_info);

// Closes one move-owned DMA-BUF descriptor as an infallible external-memory
// release callback.
void AMDF_CALL
amdf_linux_dma_buf_release(void* user_data, amdf_external_memory_type_t type,
                           amdf_external_memory_payload_t payload);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_PLATFORM_LINUX_DMA_BUF_H_
