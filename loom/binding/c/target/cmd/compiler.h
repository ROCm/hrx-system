// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_CMD_COMPILER_STORAGE_H_
#define LOOMC_TARGET_CMD_COMPILER_STORAGE_H_

#include "loom/binding/c/target/cmd/native_source_cache.h"
#include "loomc/target/cmd/compiler.h"

// Returns the immutable index retained by the compiler.
loomc_link_index_t* loomc_cmd_compiler_index(loomc_cmd_compiler_t* compiler);

// Returns the source cache whose lifetime is bounded by this compiler.
loomc_cmd_native_source_cache_t* loomc_cmd_compiler_sources(
    loomc_cmd_compiler_t* compiler);

#endif  // LOOMC_TARGET_CMD_COMPILER_STORAGE_H_
