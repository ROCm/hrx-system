// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_LINUX_FILE_POOL_IO_H_
#define AMDF_CTS_GPU_LINUX_FILE_POOL_IO_H_

#include <linux/io_uring.h>
#include <linux/stddef.h>

#include <cstddef>
#include <cstdint>
#include <vector>

// Native file ownership for a reusable, caller-owned pool. It neither creates
// GPU resources nor accesses payload bytes between file submission and its
// terminal completion. Release follows GPU retirement and precedes pool
// release.
class FilePoolIo {
 public:
  void CreateFile(const std::vector<uint32_t>& seed, bool direct,
                  size_t buffer_alignment);
  void Register(void* pointer, size_t byte_length);
  void Begin(bool write, void* pointer, uint32_t byte_length,
             uint64_t file_offset, uint64_t identity);
  void Complete(uint64_t identity, uint32_t byte_length);
  void CancelAndRetire(void* pointer, uint32_t byte_length,
                       uint64_t file_offset);
  void VerifyFile(const std::vector<uint32_t>& expected);
  void Release();

 private:
  void Submit(const io_uring_sqe& request);
  void Next(io_uring_cqe* out_completion);

  // Unlinked private regular file, retained through fixed-file unregistration.
  int file_ = -1;
  // Native ring; no SQPOLL thread or device-driven submission is requested.
  int ring_ = -1;
  // Native ring geometry returned at creation.
  io_uring_params parameters_ = {};
  // Host-owned SQ control and native submission indices.
  uint8_t* submission_ = nullptr;
  // Native CQ control, possibly sharing the SQ mapping.
  uint8_t* completion_ = nullptr;
  // Native request array; entries are reused only after SQ consumption.
  io_uring_sqe* entries_ = nullptr;
  // Full mapped SQ extent in bytes.
  size_t submission_length_ = 0;
  // Full mapped CQ extent in bytes.
  size_t completion_length_ = 0;
  // Accepted requests awaiting terminal CQE consumption.
  uint32_t pending_ = 0;
  // Whether the ring owns a pin on the original pool view.
  bool registered_buffer_ = false;
  // Whether the ring retains the private file.
  bool registered_file_ = false;
};

#endif  // AMDF_CTS_GPU_LINUX_FILE_POOL_IO_H_
