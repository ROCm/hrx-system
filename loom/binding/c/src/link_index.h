// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_LINK_INDEX_STORAGE_H_
#define LOOMC_LINK_INDEX_STORAGE_H_

#include "loom/link/module_index.h"
#include "loomc/context.h"
#include "loomc/link_index.h"
#include "loomc/source.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the context retained by a frozen public link index.
LOOMC_API_PRIVATE loomc_context_t* loomc_link_index_context(
    const loomc_link_index_t* link_index);

// Returns the internal module index owned by a frozen public link index.
LOOMC_API_PRIVATE const loom_link_module_index_t* loomc_link_index_module_index(
    const loomc_link_index_t* link_index);

// Returns the retained source for provider_ordinal, or NULL if out of range.
LOOMC_API_PRIVATE const loomc_source_t* loomc_link_index_source_for_provider(
    const loomc_link_index_t* link_index, loomc_host_size_t provider_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_LINK_INDEX_STORAGE_H_
