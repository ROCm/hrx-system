// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_UTIL_FILE_STORAGE_H_
#define AMDF_CTS_UTIL_FILE_STORAGE_H_

#include <cstddef>

enum class FileMode { kBuffered, kDirect };

// Initializes a caller-owned, empty private Linux file with |seed| bytes.
// Direct mode requires disk backing and native geometry compatible with
// |buffer_alignment|. Btrfs files are made uncompressed and NOCOW before any
// extents are allocated. The seed is copied to aligned storage and written with
// O_DIRECT, then cache residency is checked before workload I/O can begin.
// Buffered mode writes the seed directly. Unsupported storage skips the case;
// callers propagate IsSkipped() before further setup. The caller retains the
// descriptor through all accepted I/O.
void InitializeFileStorage(int file, FileMode mode, size_t buffer_alignment,
                           const void* seed, size_t byte_length);

// After all file/GPU users retire, verifies that direct I/O left the page cache
// empty, then disables O_DIRECT for independent buffered payload verification.
// This observation does not establish durability or exclude native bounce
// pages.
void FinishDirectFileIo(int file, size_t byte_length);

#endif  // AMDF_CTS_UTIL_FILE_STORAGE_H_
