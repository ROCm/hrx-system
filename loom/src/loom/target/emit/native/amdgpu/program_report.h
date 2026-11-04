// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU compile-report composition for retained-call entry closures.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_REPORT_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_REPORT_H_

#include "iree/base/api.h"
#include "loom/target/emit/native/amdgpu/program_artifact.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Composes every directly reachable helper into one entry report.
//
// Function evidence is resident once in this logical entry closure. Dynamic
// evidence is scaled by the exact direct-call multiplicity retained by Low
// reporting. Final artifact-wide physical accounting remains with the code
// object owner because several entries may share the same helper body.
iree_status_t loom_amdgpu_program_report_compose_entry(
    const loom_amdgpu_program_artifact_root_t* root,
    loom_target_compile_report_t* report);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_REPORT_H_
