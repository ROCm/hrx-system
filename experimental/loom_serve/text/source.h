// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_TEXT_SOURCE_H_
#define EXPERIMENTAL_LOOM_SERVE_TEXT_SOURCE_H_

#include "experimental/loom_serve/text/model.h"
#include "experimental/loom_serve/text/storage.h"
#include "iree/module/hal/types.h"
#include "iree/vm/buffer.h"
#include "iree/vm/environment.h"
#include "iree/vm/module.h"
#include "loom/serve/runtime/program/compilation.h"
#include "loom/serve/runtime/program/preparation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source-owned preparation, independent of device bindings and invocation.
// Private construction may run on a shared worker; the model retains this
// representation and its descriptor tables through all warm program consumers.
typedef struct loom_serve_text_source_t {
  // Allocation policy for source-owned metadata and bytecode.
  iree_allocator_t allocator;
  // Copied diagnostics, extended by subsequent native stage compilation.
  loom_serve_compilation_report_t compilation;
  // Construction registry for canonical VM/HAL reference descriptors.
  iree_vm_environment_t* environment;
  // Resolved byte-buffer reference descriptors.
  iree_vm_ref_types_t vm_types;
  // Resolved HAL reference descriptors, without device capabilities.
  iree_hal_module_types_t types;
  // Owned source declarations and parameter policies.
  loom_serve_preparation_t* preparation;
  // Compiled control bytecode awaiting native capability linking.
  iree_vm_module_t* module;
  // Opaque source state retained independently of the bootstrap invocation.
  iree_vm_buffer_t* control_state;
  // Owned parsed vocabulary reused across requests and tokenizer borrowers.
  iree_tokenizer_t* tokenizer;
  // Source terminal marker resolved once against the tokenizer.
  int32_t eos_token;
  // Effective prefill capacity selected by source preparation.
  iree_host_size_t prefill_capacity;
  // Leading stages sharing one immutable parameter placement.
  iree_host_size_t shared_stage_count;
} loom_serve_text_source_t;

// Consumes validated model options and produces source metadata plus the
// caller-owned storage description. Both outputs retain partial ownership on
// failure and must be deinitialized by their owners. No device memory, queue
// submission or shared pool mutation occurs. Option views are borrowed only
// through this call; source-returned state may retain its own copied inputs.
iree_status_t loom_serve_text_source_initialize(
    const loom_serve_text_options_t* options,
    loom_serve_text_source_t* out_source,
    loom_serve_text_storage_t* out_storage, iree_allocator_t allocator);

// Releases partial or complete construction after all native model/program
// consumers have retired. Storage returned by initialize has its own owner.
void loom_serve_text_source_deinitialize(loom_serve_text_source_t* source);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // EXPERIMENTAL_LOOM_SERVE_TEXT_SOURCE_H_
