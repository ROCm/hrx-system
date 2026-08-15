// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_COMPILE_REPORT_STORAGE_H_
#define LOOMC_COMPILE_REPORT_STORAGE_H_

#include "loomc/compile_report.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates one public compile-report option descriptor.
LOOMC_API_PRIVATE loomc_status_t loomc_compile_report_options_validate(
    const loomc_compile_report_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_COMPILE_REPORT_STORAGE_H_
