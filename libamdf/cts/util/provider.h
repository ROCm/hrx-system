// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_UTIL_PROVIDER_H_
#define AMDF_CTS_UTIL_PROVIDER_H_

#include "amdf/amdf.h"

// Initializes the test provider and consumes any provider-specific arguments.
// Returns zero after printing a diagnostic when initialization fails.
int amdf_cts_provider_initialize(int* argument_count, char*** argument_values);

// Releases provider state, including any runtime-loaded library. Returns zero
// after printing a diagnostic when the provider cannot be released cleanly.
int amdf_cts_provider_deinitialize(void);

// Returns the query entry point supplied by the initialized provider.
amdf_query_api_fn_t amdf_cts_provider_query_api(void);

#endif  // AMDF_CTS_UTIL_PROVIDER_H_
