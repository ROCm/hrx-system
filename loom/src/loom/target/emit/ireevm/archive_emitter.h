// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE VM archive emission for prepared target-low Loom modules.
//
// This packages prepared low.func.def entries into the same runtime artifact
// consumed by iree_vm_bytecode_module_create. Source-to-low lowering and
// target-low preparation are owned by the caller's compile pipeline.

#ifndef LOOM_TARGET_EMIT_IREEVM_ARCHIVE_EMITTER_H_
#define LOOM_TARGET_EMIT_IREEVM_ARCHIVE_EMITTER_H_

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/ir.h"
#include "loom/target/compile_report.h"
#include "loom/target/emit/ireevm/module_archive.h"
#include "loom/verify/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_ireevm_archive_emit_options_t {
  // VM module name stored in the emitted archive. Empty uses "loom".
  iree_string_view_t module_name;
  // Diagnostic sink used for verification, lowering, scheduling, and
  // allocation diagnostics. A NULL callback still counts diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;
  // Source resolver used to render carets for op-backed diagnostics.
  loom_source_resolver_t source_resolver;
  // Maximum diagnostics to emit before the active subsystem stops walking.
  // Zero uses a conservative default.
  uint32_t max_errors;
  // Optional caller-owned structured compile report to populate.
  loom_target_compile_report_t* report;
} loom_ireevm_archive_emit_options_t;

// Emits |module| into an allocator-owned IREE VM bytecode module archive.
//
// |module| must already contain prepared target-low VM function definitions and
// imports. Target records are resolved from the IR and VM-compatible symbols
// are emitted as one module. |out_emitted| is false when verification or target
// diagnostics rejected the module; status remains reserved for infrastructure
// failures and API contract violations. The caller owns |out_archive| when
// |out_emitted| is true and must release it with
// loom_ireevm_module_archive_deinitialize.
iree_status_t loom_ireevm_emit_module_archive_from_ir(
    loom_module_t* module, const loom_ireevm_archive_emit_options_t* options,
    iree_allocator_t allocator, bool* out_emitted,
    loom_ireevm_module_archive_t* out_archive);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_IREEVM_ARCHIVE_EMITTER_H_
