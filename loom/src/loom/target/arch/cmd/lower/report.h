// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_REPORT_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_REPORT_H_

#include "loom/target/arch/cmd/lower/program_plan.h"
#include "loom/target/arch/cmd/lower/program_writer.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Copies requested observations before resolved encoding tables expire. Report
// strings borrow |plan|, which must remain alive until report formatting ends.
iree_status_t loom_cmd_program_record_report(
    const loom_cmd_program_plan_t* plan, iree_host_size_t root_index,
    const loom_cmd_program_tables_t* tables, iree_arena_allocator_t* arena,
    loom_target_compile_report_t* report);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_REPORT_H_
