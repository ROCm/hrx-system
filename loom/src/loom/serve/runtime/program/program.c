// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/program.h"

#include <stdio.h>
#include <string.h>

#include "iree/vm/bytecode/module.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/program/jit.h"

struct loom_serve_program_t {
  // Host allocation policy for the owner and all VM storage.
  iree_allocator_t host_allocator;
  // Owned linked program retaining bytecode and native libraries.
  iree_vm_program_t* program;
  // Borrowed executable module retained by the linked program.
  iree_vm_module_t* module;
  // Optional retained service contract; generic programs use exact symbols.
  loom_model_card_t* card;
  // Borrowed component contract owned by card; NULL selects its primary
  // service.
  const loom_model_component_t* component;
  // Owned process borrowing no session state.
  iree_vm_process_t* process;
  // Fixed invocation backing, independent of request and token counts.
  iree_alignas(iree_max_align_t) uint8_t storage[16384];
  // Invocation borrowing storage; calls are serialized by the model owner.
  iree_vm_invocation_t* invocation;
};

void loom_serve_program_free(loom_serve_program_t* program) {
  if (!program) {
    return;
  }
  iree_vm_process_release(program->process);
  if (program->invocation) {
    iree_vm_invocation_deinitialize(program->invocation);
  }
  iree_vm_program_release(program->program);
  loom_model_card_release(program->card);
  iree_allocator_free(program->host_allocator, program);
}

iree_status_t loom_serve_program_compile(
    loomc_target_environment_t* target_environment,
    iree_vm_environment_t* environment, const loom_serve_sources_t* sources,
    iree_string_view_t source_path, iree_host_size_t root_count,
    const iree_string_view_t* roots, loom_serve_compilation_report_t* report,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, source_path.data, source_path.size);
  *out_module = NULL;
  iree_const_byte_span_t image = iree_const_byte_span_empty();
  iree_status_t status = loom_serve_jit_compile_vm(
      target_environment, sources, source_path, root_count, roots, report,
      host_allocator, &image);
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_module_create_trusted(
        environment, IREE_SV("model"),
        (iree_vm_bytecode_module_storage_t){image, host_allocator},
        host_allocator, out_module);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(host_allocator, (void*)image.data);
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t loom_serve_program_service_names(
    const loom_model_selection_t* model,
    const loom_model_component_t* component, iree_host_size_t count,
    const iree_string_view_t* roles, iree_allocator_t host_allocator,
    iree_string_view_t** out_names) {
  *out_names = NULL;
  const iree_string_view_t prefix =
      component ? component->export_prefix : model->kind;
  const uint32_t version = component ? component->version : model->version;
  char suffix[16] = {0};
  const iree_host_size_t suffix_length =
      version
          ? (iree_host_size_t)snprintf(suffix, sizeof(suffix), "_v%u", version)
          : 0;
  iree_host_size_t size = 0;
  bool representable =
      iree_host_size_checked_mul(count, sizeof(**out_names), &size);
  for (iree_host_size_t i = 0; i < count && representable; ++i) {
    representable = iree_host_size_checked_add(size, prefix.size, &size) &&
                    iree_host_size_checked_add(size, roles[i].size, &size) &&
                    iree_host_size_checked_add(size, 1 + suffix_length, &size);
  }
  if (!representable) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "service export names exceed host storage");
  }
  iree_string_view_t* names = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&names));
  char* cursor = (char*)(names + count);
  for (iree_host_size_t i = 0; i < count; ++i) {
    char* begin = cursor;
    memcpy(cursor, prefix.data, prefix.size);
    cursor += prefix.size;
    *cursor++ = '_';
    memcpy(cursor, roles[i].data, roles[i].size);
    cursor += roles[i].size;
    memcpy(cursor, suffix, suffix_length);
    cursor += suffix_length;
    names[i] = iree_make_string_view(begin, cursor - begin);
  }
  *out_names = names;
  return iree_ok_status();
}

iree_status_t loom_serve_program_compile_service(
    loomc_target_environment_t* target_environment,
    iree_vm_environment_t* environment, const loom_serve_sources_t* sources,
    iree_string_view_t source_path, iree_host_size_t root_count,
    const iree_string_view_t* roles, iree_string_view_list_t additional_roots,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  *out_module = NULL;
  iree_string_view_t* names = NULL;
  if (sources->card) {
    IREE_RETURN_IF_ERROR(loom_serve_program_service_names(
        loom_model_card_selection(sources->card), sources->component,
        root_count, roles, host_allocator, &names));
  }
  iree_status_t status = iree_ok_status();
  iree_string_view_t* combined_roots = NULL;
  if (additional_roots.count) {
    status = iree_allocator_malloc_array(
        host_allocator, root_count + additional_roots.count,
        sizeof(*combined_roots), (void**)&combined_roots);
    if (iree_status_is_ok(status)) {
      memcpy(combined_roots, names ? names : roles,
             root_count * sizeof(*combined_roots));
      memcpy(combined_roots + root_count, additional_roots.values,
             additional_roots.count * sizeof(*combined_roots));
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_compile(
        target_environment, environment, sources, source_path,
        root_count + additional_roots.count,
        combined_roots ? combined_roots : (names ? names : roles), report,
        host_allocator, out_module);
  }
  iree_allocator_free(host_allocator, combined_roots);
  iree_allocator_free(host_allocator, names);
  return status;
}

iree_status_t loom_serve_program_link_service(
    iree_vm_module_t* module, loom_model_card_t* card,
    const loom_model_component_t* component, iree_vm_module_span_t libraries,
    iree_allocator_t host_allocator, loom_serve_program_t** out_program) {
  IREE_RETURN_IF_ERROR(
      loom_serve_program_link(module, libraries, host_allocator, out_program));
  (*out_program)->card = card;
  (*out_program)->component = component;
  loom_model_card_retain(card);
  return iree_ok_status();
}

iree_status_t loom_serve_program_allocate_service(
    loomc_target_environment_t* target_environment,
    iree_vm_environment_t* environment, const loom_serve_sources_t* sources,
    iree_string_view_t source_path, iree_host_size_t root_count,
    const iree_string_view_t* roles, iree_vm_module_span_t libraries,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    loom_serve_program_t** out_program) {
  *out_program = NULL;
  iree_vm_module_t* module = NULL;
  iree_status_t status = loom_serve_program_compile_service(
      target_environment, environment, sources, source_path, root_count, roles,
      (iree_string_view_list_t){0}, report, host_allocator, &module);
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_link_service(module, sources->card,
                                             sources->component, libraries,
                                             host_allocator, out_program);
  }
  iree_vm_module_release(module);
  return status;
}

iree_status_t loom_serve_program_lookup(loom_serve_program_t* program,
                                        iree_string_view_t name,
                                        iree_vm_function_t* out_function) {
  return iree_vm_process_lookup_function(program->process, IREE_SV("model"),
                                         name, out_function);
}

iree_status_t loom_serve_program_lookup_service(
    loom_serve_program_t* program, iree_string_view_t name,
    iree_vm_function_t* out_function) {
  iree_string_view_t* names = NULL;
  if (program->card) {
    IREE_RETURN_IF_ERROR(loom_serve_program_service_names(
        loom_model_card_selection(program->card), program->component, 1, &name,
        program->host_allocator, &names));
  }
  iree_status_t status =
      loom_serve_program_lookup(program, names ? names[0] : name, out_function);
  iree_allocator_free(program->host_allocator, names);
  return status;
}

iree_status_t loom_serve_program_link(iree_vm_module_t* module,
                                      iree_vm_module_span_t libraries,
                                      iree_allocator_t host_allocator,
                                      loom_serve_program_t** out_program) {
  *out_program = NULL;
  loom_serve_program_t* program = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*program),
                                             (void**)&program));
  program->host_allocator = host_allocator;
  iree_status_t status =
      iree_vm_program_create((iree_vm_program_modules_t){module, libraries},
                             host_allocator, &program->program);
  if (iree_status_is_ok(status)) {
    program->module = module;
    status = iree_vm_invocation_initialize(
        iree_make_byte_span(program->storage, sizeof(program->storage)),
        &program->invocation);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_process_create(program->program, program->invocation,
                                    iree_vm_variant_span_empty(),
                                    host_allocator, &program->process);
  }
  if (iree_status_is_ok(status)) {
    *out_program = program;
  } else {
    loom_serve_program_free(program);
  }
  return status;
}

iree_status_t loom_serve_program_allocate(
    loomc_target_environment_t* target_environment,
    iree_vm_environment_t* environment, const loom_serve_sources_t* sources,
    iree_string_view_t source_path, iree_host_size_t root_count,
    const iree_string_view_t* roots, iree_vm_module_span_t libraries,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    loom_serve_program_t** out_program) {
  *out_program = NULL;
  iree_vm_module_t* module = NULL;
  iree_status_t status = loom_serve_program_compile(
      target_environment, environment, sources, source_path, root_count, roots,
      report, host_allocator, &module);
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_program_link(module, libraries, host_allocator, out_program);
  }
  iree_vm_module_release(module);
  return status;
}

static iree_string_view_t loom_serve_program_scalar_name(
    iree_vm_scalar_type_t type) {
  static const iree_string_view_t names[] = {
      IREE_SVL("none"),   IREE_SVL("i8"),  IREE_SVL("i16"),
      IREE_SVL("i32"),    IREE_SVL("i64"), IREE_SVL("f8e4m3fn"),
      IREE_SVL("f8e5m2"), IREE_SVL("f16"), IREE_SVL("bf16"),
      IREE_SVL("f32"),    IREE_SVL("f64"),
  };
  return names[type];
}

static iree_status_t loom_serve_program_check_fields(
    const iree_vm_module_t* module, iree_string_view_t export_name,
    const char* side, iree_vm_signature_type_span_t expected,
    iree_vm_module_signature_side_t actual) {
  if (expected.count != actual.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model export '%.*s' %s count mismatch: host "
                            "expects %zu, source declares %u",
                            (int)export_name.size, export_name.data, side,
                            expected.count, actual.count);
  }
  for (iree_host_size_t i = 0; i < expected.count; ++i) {
    const iree_vm_signature_type_t required = expected.data[i];
    const iree_vm_module_signature_type_t field = actual.data[i];
    iree_vm_ref_type_t ref = NULL;
    if (field.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF) {
      IREE_RETURN_IF_ERROR(
          iree_vm_module_ref_type_by_ordinal(module, field.type_ordinal, &ref));
    }
    if ((required.kind == IREE_VM_SIGNATURE_TYPE_KIND_SCALAR &&
         field.kind == required.value.scalar) ||
        (required.kind == IREE_VM_SIGNATURE_TYPE_KIND_REF &&
         ref == required.value.ref)) {
      continue;
    }
    const iree_string_view_t expected_name =
        required.kind == IREE_VM_SIGNATURE_TYPE_KIND_REF
            ? required.value.ref->type_name
            : loom_serve_program_scalar_name(required.value.scalar);
    const iree_string_view_t actual_name =
        ref ? ref->type_name
        : field.kind == IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION
            ? IREE_SV("function")
            : loom_serve_program_scalar_name((iree_vm_scalar_type_t)field.kind);
    const iree_string_view_t expected_namespace =
        required.kind == IREE_VM_SIGNATURE_TYPE_KIND_REF
            ? required.value.ref->table->namespace_name
            : IREE_SV("");
    const iree_string_view_t actual_namespace =
        ref ? ref->table->namespace_name : IREE_SV("");
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "model export '%.*s' %s %zu type mismatch: host expects %.*s%s%.*s, "
        "source declares %.*s%s%.*s",
        (int)export_name.size, export_name.data, side, i,
        (int)expected_namespace.size, expected_namespace.data,
        expected_namespace.size ? "." : "", (int)expected_name.size,
        expected_name.data, (int)actual_namespace.size, actual_namespace.data,
        actual_namespace.size ? "." : "", (int)actual_name.size,
        actual_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_program_bind_names(
    loom_serve_program_t* program, iree_host_size_t binding_count,
    const loom_serve_program_binding_t* bindings,
    const iree_string_view_t* names) {
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    const loom_serve_program_binding_t* binding = &bindings[i];
    const iree_string_view_t name = names ? names[i] : binding->name;
    iree_vm_export_t export_value = {0};
    IREE_RETURN_IF_ERROR(
        iree_vm_module_lookup_export(program->module, name, &export_value));
    iree_vm_module_export_declaration_t declaration = {0};
    IREE_RETURN_IF_ERROR(iree_vm_module_query_export(
        program->module, export_value.ordinal, &declaration));
    iree_vm_module_callable_type_declaration_t callable = {0};
    IREE_RETURN_IF_ERROR(iree_vm_module_query_callable_type(
        program->module, declaration.callable_type_ordinal, &callable));
    if (iree_any_bit_set(callable.flags,
                         IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "model export '%.*s' must be synchronous; source "
                              "declares a yielding function",
                              (int)name.size, name.data);
    }
    IREE_RETURN_IF_ERROR(loom_serve_program_check_fields(
        program->module, name, "argument", binding->arguments,
        callable.signature.arguments));
    IREE_RETURN_IF_ERROR(loom_serve_program_check_fields(
        program->module, name, "result", binding->results,
        callable.signature.results));
    IREE_RETURN_IF_ERROR(iree_vm_function_from_export(
        program->process, export_value, binding->function));
  }
  return iree_ok_status();
}

iree_status_t loom_serve_program_bind(
    loom_serve_program_t* program, iree_host_size_t binding_count,
    const loom_serve_program_binding_t* bindings) {
  return loom_serve_program_bind_names(program, binding_count, bindings, NULL);
}

iree_status_t loom_serve_program_bind_service(
    loom_serve_program_t* program, iree_host_size_t binding_count,
    const loom_serve_program_binding_t* bindings) {
  if (!program->card) {
    return loom_serve_program_bind(program, binding_count, bindings);
  }
  iree_string_view_t* roles = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      program->host_allocator, binding_count, sizeof(*roles), (void**)&roles));
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    roles[i] = bindings[i].name;
  }
  iree_string_view_t* names = NULL;
  iree_status_t status = loom_serve_program_service_names(
      loom_model_card_selection(program->card), program->component,
      binding_count, roles, program->host_allocator, &names);
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_program_bind_names(program, binding_count, bindings, names);
  }
  iree_allocator_free(program->host_allocator, names);
  iree_allocator_free(program->host_allocator, roles);
  return status;
}

iree_vm_process_t* loom_serve_program_process(loom_serve_program_t* program) {
  return program->process;
}

iree_vm_invocation_t* loom_serve_program_invocation(
    loom_serve_program_t* program) {
  return program->invocation;
}
