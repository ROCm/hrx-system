// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/kernel.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/binding/c/src/compile.h"
#include "loom/binding/c/src/context.h"
#include "loom/binding/c/src/emit.h"
#include "loom/binding/c/src/module.h"
#include "loom/binding/c/src/pass_program.h"
#include "loom/binding/c/src/product.h"
#include "loom/binding/c/src/result.h"
#include "loom/binding/c/src/workspace.h"
#include "loomc/iree.h"

typedef struct loomc_kernel_product_impl_t {
  // Generic immutable product interface exposed to callers.
  loomc_product_t base;

  // Allocator used for product-owned storage.
  loomc_allocator_t allocator;

  // Retained result owning artifact strings and byte sequences.
  loomc_result_t* result;

  // Product-owned root projections in request order.
  loomc_kernel_product_root_t* roots;
} loomc_kernel_product_impl_t;

static void loomc_kernel_product_destroy(loomc_product_t* base_product) {
  loomc_kernel_product_impl_t* product =
      (loomc_kernel_product_impl_t*)base_product;
  const loomc_allocator_t allocator = product->allocator;
  loomc_result_release(product->result);
  loomc_allocator_free(allocator, product);
}

static const loomc_product_descriptor_t loomc_kernel_product_descriptor_ = {
    .destroy = loomc_kernel_product_destroy,
};

static const loomc_kernel_product_impl_t* loomc_kernel_product_const_cast(
    const loomc_product_t* product) {
  return loomc_product_isa(product, &loomc_kernel_product_descriptor_)
             ? (const loomc_kernel_product_impl_t*)product
             : NULL;
}

static loomc_status_t loomc_kernel_product_validate_request_goals(
    const loomc_request_t* request) {
  const loomc_request_root_t* roots = loomc_request_roots(request);
  const loomc_host_size_t root_count = loomc_request_root_count(request);
  for (loomc_host_size_t i = 0; i < root_count; ++i) {
    switch (roots[i].goal) {
      case LOOMC_KERNEL_ROOT_GOAL_EXECUTABLE_ENTRY:
      case LOOMC_KERNEL_ROOT_GOAL_HOST_LAUNCHABLE:
        break;
      default:
        return loomc_make_status(
            LOOMC_STATUS_INVALID_ARGUMENT,
            "kernel request root has an unsupported product goal");
    }
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_kernel_product_validate_arguments(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* pass_program, const loomc_request_t* request,
    const loomc_compile_options_t* compile_options, loomc_allocator_t allocator,
    const loomc_target_specialization_options_t** out_target_specialization) {
  *out_target_specialization = NULL;
  if (compiler == NULL || workspace == NULL || pass_program == NULL ||
      request == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compiler, workspace, pass_program, and request must not be NULL");
  }
  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator must be valid");
  }
  if (loomc_request_product_descriptor(request) !=
      loomc_kernel_product_descriptor()) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "request does not require an executable kernel product");
  }
  LOOMC_RETURN_IF_ERROR(loomc_kernel_product_validate_request_goals(request));

  loomc_context_t* context = loomc_compiler_context(compiler);
  if (loomc_pass_program_context(pass_program) != context) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "pass program was created with another context");
  }
  if (loomc_context_target_environment(context) == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_FAILED_PRECONDITION,
        "kernel product compilation requires a target environment");
  }
  if (compile_options != NULL &&
      (compile_options->artifact_flags &
       LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG) != 0) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "kernel root goals control launch-config artifact production");
  }
  LOOMC_RETURN_IF_ERROR(loomc_compile_resolve_target_specialization(
      compiler, compile_options, out_target_specialization));
  return loomc_compile_validate_config_module(compiler, NULL, compile_options);
}

static loomc_status_t loomc_kernel_product_prepare_compile_roots(
    const loomc_request_t* request, iree_arena_allocator_t* arena,
    loomc_compile_root_set_t* out_root_set) {
  *out_root_set = (loomc_compile_root_set_t){0};
  const loomc_host_size_t root_count = loomc_request_root_count(request);
  const loomc_request_root_t* request_roots = loomc_request_roots(request);

  iree_host_size_t symbol_offset = 0;
  iree_host_size_t launch_offset = 0;
  iree_host_size_t storage_size = 0;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(IREE_STRUCT_LAYOUT(
      0, &storage_size,
      IREE_STRUCT_FIELD_ALIGNED(root_count, loom_symbol_id_t,
                                iree_alignof(loom_symbol_id_t), &symbol_offset),
      IREE_STRUCT_FIELD_ALIGNED(root_count, loomc_request_root_ordinal_t,
                                iree_alignof(loomc_request_root_ordinal_t),
                                &launch_offset))));
  uint8_t* storage = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(
      iree_arena_allocate(arena, storage_size, (void**)&storage)));
  loom_symbol_id_t* symbol_ids = (loom_symbol_id_t*)(storage + symbol_offset);
  loomc_request_root_ordinal_t* launch_root_ordinals =
      (loomc_request_root_ordinal_t*)(storage + launch_offset);

  iree_host_size_t launch_root_count = 0;
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    symbol_ids[i] = request_roots[i].symbol_ordinal;
    if (request_roots[i].goal == LOOMC_KERNEL_ROOT_GOAL_HOST_LAUNCHABLE) {
      launch_root_ordinals[launch_root_count++] =
          (loomc_request_root_ordinal_t)i;
    }
  }
  *out_root_set = (loomc_compile_root_set_t){
      .symbol_ids = symbol_ids,
      .count = root_count,
      .launch_root_ordinals = launch_root_ordinals,
      .launch_root_count = launch_root_count,
  };
  return loomc_ok_status();
}

static loomc_status_t loomc_kernel_product_allocate(
    loomc_result_t* result, loomc_host_size_t root_count,
    loomc_allocator_t allocator, loomc_kernel_product_impl_t** out_product) {
  *out_product = NULL;
  const loomc_host_size_t artifact_count = loomc_result_artifact_count(result);
  iree_host_size_t artifacts_offset = 0;
  iree_host_size_t roots_offset = 0;
  iree_host_size_t allocation_size = 0;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(IREE_STRUCT_LAYOUT(
      sizeof(loomc_kernel_product_impl_t), &allocation_size,
      IREE_STRUCT_FIELD(artifact_count, loomc_artifact_t, &artifacts_offset),
      IREE_STRUCT_FIELD(root_count, loomc_kernel_product_root_t,
                        &roots_offset))));

  loomc_kernel_product_impl_t* product = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
      allocator, allocation_size, (void**)&product));
  memset(product, 0, sizeof(*product));
  product->allocator = allocator;
  product->result = result;
  loomc_result_retain(result);

  uint8_t* storage = (uint8_t*)product;
  loomc_artifact_t* artifacts = (loomc_artifact_t*)(storage + artifacts_offset);
  for (iree_host_size_t i = 0; i < artifact_count; ++i) {
    artifacts[i] = *loomc_result_artifact_at(result, i);
  }
  product->roots = (loomc_kernel_product_root_t*)(storage + roots_offset);
  loomc_product_initialize(&loomc_kernel_product_descriptor_, artifacts,
                           artifact_count, root_count,
                           /*requirement_count=*/0, &product->base);
  *out_product = product;
  return loomc_ok_status();
}

static loomc_status_t loomc_kernel_product_record_export_projections(
    const loom_target_emit_artifact_t* artifact,
    iree_host_size_t function_version_count,
    uint32_t* function_ordinals_by_version) {
  if (artifact->export_projection_count != 0 &&
      artifact->export_projections == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INTERNAL,
        "target emitter returned an invalid export projection table");
  }
  for (iree_host_size_t i = 0; i < artifact->export_projection_count; ++i) {
    const loom_target_emit_export_projection_t projection =
        artifact->export_projections[i];
    if (projection.function_version_ordinal >= function_version_count) {
      return loomc_make_status(
          LOOMC_STATUS_INTERNAL,
          "target emitter returned an out-of-range function version");
    }
    uint32_t* function_ordinal =
        &function_ordinals_by_version[projection.function_version_ordinal];
    if (*function_ordinal != LOOMC_KERNEL_FUNCTION_ORDINAL_INVALID &&
        *function_ordinal != projection.ordinal) {
      return loomc_make_status(
          LOOMC_STATUS_INTERNAL,
          "target emitter returned ambiguous function projections");
    }
    *function_ordinal = projection.ordinal;
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_kernel_product_populate_roots(
    const loomc_request_t* request,
    const loomc_compile_operation_t* compile_operation,
    const loomc_emit_operation_t* emit_operation, iree_arena_allocator_t* arena,
    loomc_kernel_product_impl_t* product) {
  const iree_host_size_t root_count = loomc_request_root_count(request);
  const loomc_request_root_t* request_roots = loomc_request_roots(request);
  const iree_host_size_t function_version_count =
      compile_operation->function_version_count;
  if (compile_operation->root_count != root_count ||
      compile_operation->root_function_version_ordinals == NULL ||
      function_version_count == 0) {
    return loomc_make_status(
        LOOMC_STATUS_INTERNAL,
        "kernel compilation produced no stable root function versions");
  }
  if (emit_operation->primary_artifact_ordinal == IREE_HOST_SIZE_MAX ||
      emit_operation->primary_artifact_ordinal > UINT32_MAX) {
    return loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "kernel executable artifact ordinal exceeds the product domain");
  }

  iree_host_size_t executable_offset = 0;
  iree_host_size_t launch_offset = 0;
  iree_host_size_t storage_size = 0;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(IREE_STRUCT_LAYOUT(
      0, &storage_size,
      IREE_STRUCT_FIELD(function_version_count, uint32_t, &executable_offset),
      IREE_STRUCT_FIELD(function_version_count, uint32_t, &launch_offset))));
  uint8_t* storage = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(
      iree_arena_allocate(arena, storage_size, (void**)&storage)));
  uint32_t* executable_ordinals = (uint32_t*)(storage + executable_offset);
  uint32_t* launch_ordinals = (uint32_t*)(storage + launch_offset);
  memset(executable_ordinals, 0xFF,
         function_version_count * sizeof(*executable_ordinals));
  memset(launch_ordinals, 0xFF,
         function_version_count * sizeof(*launch_ordinals));

  LOOMC_RETURN_IF_ERROR(loomc_kernel_product_record_export_projections(
      &emit_operation->target_artifact, function_version_count,
      executable_ordinals));
  if (compile_operation->launch_config_artifact.contents != NULL) {
    LOOMC_RETURN_IF_ERROR(loomc_kernel_product_record_export_projections(
        &compile_operation->launch_config_artifact, function_version_count,
        launch_ordinals));
  }

  const uint32_t executable_artifact_ordinal =
      (uint32_t)emit_operation->primary_artifact_ordinal;
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    const loom_function_version_ordinal_t version_ordinal =
        compile_operation->root_function_version_ordinals[i];
    if (version_ordinal >= function_version_count ||
        executable_ordinals[version_ordinal] ==
            LOOMC_KERNEL_FUNCTION_ORDINAL_INVALID) {
      return loomc_make_status(
          LOOMC_STATUS_INTERNAL,
          "kernel root has no target executable function projection");
    }
    loomc_kernel_product_root_t root = {
        .executable_artifact_ordinal = executable_artifact_ordinal,
        .executable_function_ordinal = executable_ordinals[version_ordinal],
        .launch_config_artifact_ordinal = LOOMC_KERNEL_ARTIFACT_ORDINAL_INVALID,
        .launch_config_function_ordinal = LOOMC_KERNEL_FUNCTION_ORDINAL_INVALID,
    };
    if (request_roots[i].goal == LOOMC_KERNEL_ROOT_GOAL_HOST_LAUNCHABLE) {
      if (compile_operation->launch_config_artifact_ordinal ==
              IREE_HOST_SIZE_MAX ||
          compile_operation->launch_config_artifact_ordinal > UINT32_MAX) {
        return loomc_make_status(
            LOOMC_STATUS_RESOURCE_EXHAUSTED,
            "kernel launch artifact ordinal exceeds the product domain");
      }
      if (launch_ordinals[version_ordinal] ==
          LOOMC_KERNEL_FUNCTION_ORDINAL_INVALID) {
        return loomc_make_status(
            LOOMC_STATUS_INTERNAL,
            "host-launchable kernel root has no launch function projection");
      }
      root.launch_config_artifact_ordinal =
          (uint32_t)compile_operation->launch_config_artifact_ordinal;
      root.launch_config_function_ordinal = launch_ordinals[version_ordinal];
    }
    product->roots[i] = root;
  }
  return loomc_ok_status();
}

loomc_status_t loomc_kernel_product_build_request(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* pass_program, const loomc_request_t* request,
    const loomc_compile_options_t* compile_options,
    const loomc_emit_options_t* emit_options, loomc_allocator_t allocator,
    loomc_product_t** out_product, loomc_result_t** out_result) {
  if (out_product == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_product and out_result must not be NULL");
  }
  *out_product = NULL;
  *out_result = NULL;

  const loomc_target_specialization_options_t* target_specialization = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_kernel_product_validate_arguments(
      compiler, workspace, pass_program, request, compile_options, allocator,
      &target_specialization));
  loomc_context_t* context = loomc_compiler_context(compiler);
  loomc_target_environment_t* target_environment =
      loomc_context_target_environment(context);

  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  loomc_kernel_product_impl_t* product = NULL;
  loomc_compile_operation_t compile_operation = {0};
  loomc_emit_operation_t emit_operation = {0};
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &scratch_arena);

  loomc_status_t status = loomc_module_deserialize_from_source(
      context, workspace, loomc_request_source(request),
      /*options=*/NULL, allocator, &module, &result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_compile_validate_request_roots(module, request);
  }
  loomc_compile_root_set_t root_set = {0};
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_kernel_product_prepare_compile_roots(request, &scratch_arena,
                                                        &root_set);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_compile_module_into_result(
        compiler, workspace, pass_program, module, compile_options,
        target_specialization, &root_set, result, &compile_operation);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status =
        loomc_emit_module_into_result(target_environment, workspace, module,
                                      emit_options, result, &emit_operation);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      root_set.launch_root_count != 0) {
    status = loomc_compile_add_launch_config_artifact(result, compile_options,
                                                      &compile_operation);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_kernel_product_allocate(
        result, loomc_request_root_count(request), allocator, &product);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_kernel_product_populate_roots(
        request, &compile_operation, &emit_operation, &scratch_arena, product);
  }
  if (loomc_status_is_ok(status)) {
    if (loomc_result_succeeded(result)) {
      *out_product = &product->base;
      product = NULL;
    }
    *out_result = result;
    result = NULL;
  }

  loomc_product_release(product != NULL ? &product->base : NULL);
  loomc_emit_operation_deinitialize(&emit_operation);
  loomc_compile_operation_deinitialize(&compile_operation);
  iree_arena_deinitialize(&scratch_arena);
  loomc_module_release(module);
  loomc_result_release(result);
  return status;
}

const loomc_product_descriptor_t* loomc_kernel_product_descriptor(void) {
  return &loomc_kernel_product_descriptor_;
}

bool loomc_kernel_product_root_at(const loomc_product_t* base_product,
                                  loomc_host_size_t ordinal,
                                  loomc_kernel_product_root_t* out_root) {
  const loomc_kernel_product_impl_t* product =
      loomc_kernel_product_const_cast(base_product);
  if (product == NULL || out_root == NULL ||
      ordinal >= loomc_product_export_count(base_product)) {
    return false;
  }
  *out_root = product->roots[ordinal];
  return true;
}
