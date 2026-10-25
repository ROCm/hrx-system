// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_PROGRAM_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_PROGRAM_H_

#include "iree/vm/environment.h"
#include "iree/vm/invocation.h"
#include "iree/vm/process.h"
#include "iree/vm/program.h"
#include "iree/vm/reflection.h"
#include "loom/serve/runtime/program/compilation.h"
#include "loom/serve/runtime/program/sources.h"
#include "loomc/target.h"

#ifdef __cplusplus
extern "C" {
#endif

// One source-JIT host program and serialized invocation, shared across
// requests. Native modules own any asynchronous resources submitted by the
// program.
typedef struct loom_serve_program_t loom_serve_program_t;

// One synchronous host call into a source-defined model program. Scalar fields
// have exact VM widths; reference fields carry canonical environment
// identities. Function-valued arguments and results are not part of the serving
// host ABI.
typedef struct loom_serve_program_binding_t {
  // Exact export name in the model module.
  iree_string_view_t name;
  // Source-ordered argument types supplied by the host caller.
  iree_vm_signature_type_span_t arguments;
  // Source-ordered result types consumed by the host caller.
  iree_vm_signature_type_span_t results;
  // Borrowed destination for the process-bound function on successful binding.
  iree_vm_function_t* function;
} loom_serve_program_binding_t;

// Compiles source into an owned module named "model" without linking native
// libraries or creating a process. The environment supplies reference types;
// its provider scopes outlive the module and escaped references. This finite
// CPU operation may run on a private preparation task. sources supplies the
// namespace for source_path. No compiler storage or source mapping survives;
// report accumulates copied diagnostics on failure. target_environment supplies
// VM support and the complete source catalog grammar and is borrowed for the
// call.
iree_status_t loom_serve_program_compile(
    loomc_target_environment_t* target_environment,
    iree_vm_environment_t* environment, const loom_serve_sources_t* sources,
    iree_string_view_t source_path, iree_host_size_t root_count,
    const iree_string_view_t* roots, loom_serve_compilation_report_t* report,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module);

// Links prepared code to its runtime capabilities and creates one process.
// Retains module and libraries on success; caller ownership never transfers.
// Performs no compilation or source I/O. Native library creation and linking
// run on the owner of the shared resources those capabilities expose.
iree_status_t loom_serve_program_link(iree_vm_module_t* module,
                                      iree_vm_module_span_t libraries,
                                      iree_allocator_t host_allocator,
                                      loom_serve_program_t** out_program);

// Compiles the nonempty roots list from source_path into one module "model",
// links it with libraries, and creates one process. The environment's provider
// scopes must outlive the program and any returned references. Libraries are
// retained by the program. sources supplies the namespace for source_path.
// target_environment supplies VM support and the complete source grammar.
// No compiler storage or source-file mapping survives this call.
// report accumulates copied source diagnostics, including on failure.
iree_status_t loom_serve_program_allocate(
    loomc_target_environment_t* target_environment,
    iree_vm_environment_t* environment, const loom_serve_sources_t* sources,
    iree_string_view_t source_path, iree_host_size_t root_count,
    const iree_string_view_t* roots, iree_vm_module_span_t libraries,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    loom_serve_program_t** out_program);

// Service-role counterparts of compile/link/allocate. Card-backed programs
// resolve roles as kind_role_vN, or kind_component_role_vN for components
// (without the version suffix for development); direct program
// namespaces retain their exact authored names. The linked program retains
// the selected contract for bind_service and lookup_service.
// Parameter policies and other non-service programs use the exact-name APIs.
// additional_roots names exact source-selected exports compiled beside the
// service roles. Those names receive no service prefix or version suffix.
iree_status_t loom_serve_program_compile_service(
    loomc_target_environment_t* target_environment,
    iree_vm_environment_t* environment, const loom_serve_sources_t* sources,
    iree_string_view_t source_path, iree_host_size_t root_count,
    const iree_string_view_t* roles, iree_string_view_list_t additional_roots,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);
iree_status_t loom_serve_program_link_service(
    iree_vm_module_t* module, loom_model_card_t* card,
    const loom_model_component_t* component, iree_vm_module_span_t libraries,
    iree_allocator_t host_allocator, loom_serve_program_t** out_program);
iree_status_t loom_serve_program_allocate_service(
    loomc_target_environment_t* target_environment,
    iree_vm_environment_t* environment, const loom_serve_sources_t* sources,
    iree_string_view_t source_path, iree_host_size_t root_count,
    const iree_string_view_t* roles, iree_vm_module_span_t libraries,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    loom_serve_program_t** out_program);

// Resolves an exact authored export without invoking it.
iree_status_t loom_serve_program_lookup(loom_serve_program_t* program,
                                        iree_string_view_t name,
                                        iree_vm_function_t* out_function);

// Resolves a service role through the retained card contract. Direct program
// callers without a card retain their authored role name.
iree_status_t loom_serve_program_lookup_service(
    loom_serve_program_t* program, iree_string_view_t role,
    iree_vm_function_t* out_function);

// Releases VM ownership only. The caller joins native asynchronous work before
// destroying the program or any host payloads that work borrows.
void loom_serve_program_free(loom_serve_program_t* program);

// Validates and binds exact host exports without invoking source code. Missing
// exports, incompatible signatures, and yielding functions fail with the export
// name and expected contract. Earlier destinations may be set if a later
// binding fails; none may be used until the complete call succeeds. Bound
// functions borrow program. This cold operation runs before publishing a
// service.
iree_status_t loom_serve_program_bind(
    loom_serve_program_t* program, iree_host_size_t binding_count,
    const loom_serve_program_binding_t* bindings);

// Binds service roles using the same card contract as compile_service and
// lookup_service. Binding names are roles; argument/result contracts and
// ownership are otherwise identical to bind.
iree_status_t loom_serve_program_bind_service(
    loom_serve_program_t* program, iree_host_size_t binding_count,
    const loom_serve_program_binding_t* bindings);

// Borrowed process and invocation for ordinary VM lookup/invoke APIs. Calls
// using this invocation are serialized by its owner; sessions are plain data.
iree_vm_process_t* loom_serve_program_process(loom_serve_program_t* program);
iree_vm_invocation_t* loom_serve_program_invocation(
    loom_serve_program_t* program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_PROGRAM_H_
