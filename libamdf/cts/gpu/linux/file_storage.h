// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_LINUX_FILE_STORAGE_H_
#define AMDF_CTS_GPU_LINUX_FILE_STORAGE_H_

#include <cstddef>

enum class FileMode { kBuffered, kDirect };

// Prepares a caller-owned, empty private file before any extents are allocated.
// Direct mode requires disk backing and native geometry compatible with the
// caller's buffer alignment. Btrfs files are made uncompressed and NOCOW before
// seeding, which also disables data checksums on the empty file. Unsupported
// storage skips the case; callers propagate IsSkipped() before further setup.
void PrepareFileStorage(int file, FileMode mode, size_t buffer_alignment);

// Flushes and evicts the buffered seed, verifies an empty page cache, and
// enables O_DIRECT. The caller retains the descriptor through all accepted I/O.
void EnableDirectFileIo(int file, size_t byte_length);

// After all file/GPU users retire, verifies that direct I/O left the page cache
// empty, then disables O_DIRECT for independent buffered payload verification.
// This observation does not establish durability or exclude native bounce
// pages.
void FinishDirectFileIo(int file, size_t byte_length);

#endif  // AMDF_CTS_GPU_LINUX_FILE_STORAGE_H_
