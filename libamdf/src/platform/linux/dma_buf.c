// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/linux/dma_buf.h"

#include <limits.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

#include "libamdf/src/platform/linux/file.h"

amdf_status_t amdf_linux_dma_buf_query(int descriptor,
                                       amdf_linux_dma_buf_info_t* out_info) {
  if (descriptor < 0 || out_info == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  struct stat file_info = {0};
  if (fstat(descriptor, &file_info) != 0) return amdf_linux_error(errno);
  if (file_info.st_size <= 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_linux_dma_buf_info_t info = {
      .physical_backing_id = {.words = {(uint64_t)file_info.st_dev,
                                        (uint64_t)file_info.st_ino}},
      .byte_length = (uint64_t)file_info.st_size,
  };
  if (!amdf_physical_memory_id_is_valid(&info.physical_backing_id)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  *out_info = info;
  return AMDF_STATUS_OK;
}

void AMDF_CALL
amdf_linux_dma_buf_release(void* user_data, amdf_external_memory_type_t type,
                           amdf_external_memory_payload_t payload) {
  (void)user_data;
  amdf_assert(type == AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD);
  amdf_assert(payload.file_descriptor >= 0 &&
              payload.file_descriptor <= INT_MAX);
  const int result = close((int)payload.file_descriptor);
  amdf_assert(result == 0 &&
              "closing an owned DMA-BUF descriptor must succeed");
  (void)result;
}
