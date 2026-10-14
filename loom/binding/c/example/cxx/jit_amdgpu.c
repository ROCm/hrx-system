// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/init.h"
#include "loom/binding/c/example/cxx/kernels.h"
#include "loomc/compile.h"
#include "loomc/import/cxx.h"
#include "loomc/iree.h"
#include "loomc/launch_config.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/amdgpu/iree_hal.h"

enum { kElementCount = 128, kGuardCount = 4 };

typedef struct jit_state_t {
  // Async services required by the live device.
  iree_async_proactor_pool_t* proactor_pool;
  // Completion tracking shared by the device group.
  iree_async_frontier_tracker_t* frontier_tracker;
  // Queue group owning registration with the completion tracker.
  iree_hal_device_group_t* device_group;
  // Explicit storage shared by the dispatch and transfer families.
  iree_hal_pool_t* pool;
  // Physical execution device and allocator owner.
  iree_hal_device_t* device;
  // Borrowed provisioned dispatch queue.
  iree_hal_queue_t* dispatch_queue;
  // Borrowed provisioned transfer queue.
  iree_hal_queue_t* transfer_queue;
  // Physical devices covered by the dispatch queue's family.
  iree_hal_physical_device_affinity_t physical_affinity;
  // Compiler target package.
  loomc_target_environment_t* target_environment;
  // Immutable compiler context.
  loomc_context_t* context;
  // Mutable import and compilation storage.
  loomc_workspace_t* workspace;
  // Target facts obtained from the selected live device.
  loomc_target_profile_t* profile;
  // Exact HAL executable target paired with the compiler profile.
  const iree_hal_executable_target_t* executable_target;
  // Prepared compiler reusable across source submissions.
  loomc_compiler_t* compiler;
  // Prepared lowering pipeline reusable across source submissions.
  loomc_pass_program_t* pipeline;
  // Imported compilation unit containing both kernels.
  loomc_module_t* module;
  // Exact launch choices, reusable across compilations of the source module.
  loomc_module_t* config;
  // Current operation diagnostics and artifacts.
  loomc_result_t* result;
  // Evaluator for the source-authored launch contracts.
  loomc_launch_config_program_t* launch_program;
  // Runtime executable loaded from the emitted HSACO.
  iree_hal_executable_t* executable;
} jit_state_t;

static void deinitialize(jit_state_t* state) {
  iree_hal_executable_release(state->executable);
  loomc_launch_config_program_release(state->launch_program);
  loomc_result_release(state->result);
  loomc_module_release(state->module);
  loomc_module_release(state->config);
  loomc_pass_program_release(state->pipeline);
  loomc_compiler_release(state->compiler);
  loomc_target_profile_release(state->profile);
  loomc_workspace_release(state->workspace);
  loomc_context_release(state->context);
  loomc_target_environment_release(state->target_environment);
  iree_hal_pool_release(state->pool);
  iree_hal_device_group_release(state->device_group);
  iree_hal_device_release(state->device);
  iree_async_frontier_tracker_release(state->frontier_tracker);
  iree_async_proactor_pool_release(state->proactor_pool);
}

static iree_status_t check_result(jit_state_t* state) {
  if (loomc_result_succeeded(state->result)) {
    return iree_ok_status();
  }
  for (size_t i = 0; i < loomc_result_diagnostic_count(state->result); ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(state->result, i);
    fprintf(stderr, "%.*s: %.*s\n", (int)diagnostic->code.size,
            diagnostic->code.data, (int)diagnostic->message.size,
            diagnostic->message.data);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "source compilation was rejected");
}

static void reset_result(jit_state_t* state) {
  loomc_result_release(state->result);
  state->result = NULL;
}

static iree_status_t initialize(jit_state_t* state, const char* device_uri) {
  iree_allocator_t allocator = iree_allocator_system();
  IREE_RETURN_IF_ERROR(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));
  IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
      iree_numa_node_count(), NULL, iree_async_proactor_pool_options_default(),
      allocator, &state->proactor_pool));
  iree_hal_device_create_params_t params =
      iree_hal_device_create_params_default();
  params.proactor_pool = state->proactor_pool;
  IREE_RETURN_IF_ERROR(iree_hal_create_device(
      iree_hal_driver_registry_default(), iree_make_cstring_view(device_uri),
      &params, allocator, &state->device));
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), allocator,
      &state->frontier_tracker));
  IREE_RETURN_IF_ERROR(iree_hal_device_group_create_from_device(
      state->device, state->frontier_tracker, allocator, &state->device_group));
  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(iree_hal_device_spec(state->device));
  for (size_t i = 0; i < queues->family_count; ++i) {
    const iree_hal_queue_family_spec_t* family = &queues->families[i];
    if (!family->provisioned_queue_count) {
      continue;
    }
    if (!state->dispatch_queue &&
        iree_all_bits_set(family->role_flags,
                          IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH)) {
      state->dispatch_queue = iree_hal_device_queue(state->device, i, 0);
      state->physical_affinity = family->physical_device_affinity;
    }
    if (!state->transfer_queue &&
        iree_all_bits_set(family->role_flags,
                          IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER)) {
      state->transfer_queue = iree_hal_device_queue(state->device, i, 0);
    }
  }
  if (!state->dispatch_queue || !state->transfer_queue) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "device needs dispatch and transfer queues");
  }
  iree_hal_pool_family_access_t family_accesses[2] = {
      {
          .family = iree_hal_queue_family(state->dispatch_queue),
          .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      },
  };
  iree_host_size_t family_access_count = 1;
  if (iree_hal_queue_family(state->transfer_queue) ==
      family_accesses[0].family) {
    family_accesses[0].usage |= IREE_HAL_BUFFER_USAGE_TRANSFER;
  } else {
    family_accesses[family_access_count++] = (iree_hal_pool_family_access_t){
        .family = iree_hal_queue_family(state->transfer_queue),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
    };
  }
  const iree_hal_pool_scope_t pool_scope = {
      .family_count = family_access_count,
      .families = family_accesses,
  };
  iree_hal_slab_pool_options_t pool_options;
  iree_hal_slab_pool_options_initialize(&pool_options);
  pool_options.trace_name = iree_make_cstring_view("loom.jit_amdgpu");
  IREE_RETURN_IF_ERROR(iree_hal_slab_pool_create(
      state->device_group, pool_scope, &pool_options, allocator, &state->pool));
  IREE_RETURN_IF_ERROR(
      iree_status_from_loomc(loomc_target_environment_create_amdgpu(
          loomc_allocator_system(), &state->target_environment)));
  const loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .target_environment = state->target_environment,
  };
  const loomc_context_options_t context_options = {.next = &target_options};
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_context_create(
      &context_options, loomc_allocator_system(), &state->context)));
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_workspace_create(
      NULL, loomc_allocator_system(), &state->workspace)));
  const loomc_amdgpu_iree_hal_target_options_t hal_target_options = {
      .device = state->device,
      .physical_device_affinity = state->physical_affinity,
  };
  loomc_iree_hal_target_selection_t selection = {0};
  IREE_RETURN_IF_ERROR(
      iree_status_from_loomc(loomc_target_select_amdgpu_iree_hal(
          state->target_environment, &hal_target_options,
          loomc_allocator_system(), &selection, &state->result)));
  state->profile = selection.target_profile;
  state->executable_target = selection.executable_target;
  IREE_RETURN_IF_ERROR(check_result(state));
  reset_result(state);
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_compiler_create(
      state->context, NULL, loomc_allocator_system(), &state->compiler)));
  const loomc_target_pipeline_options_t pipeline_options = {
      .kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
  };
  IREE_RETURN_IF_ERROR(
      iree_status_from_loomc(loomc_pass_program_create_from_target_pipeline(
          state->context, &pipeline_options, loomc_allocator_system(),
          &state->pipeline, &state->result)));
  IREE_RETURN_IF_ERROR(check_result(state));
  reset_result(state);
  return iree_ok_status();
}

static const loomc_artifact_t* find_artifact(jit_state_t* state,
                                             loomc_artifact_kind_t kind) {
  for (size_t i = 0; i < loomc_result_artifact_count(state->result); ++i) {
    const loomc_artifact_t* artifact =
        loomc_result_artifact_at(state->result, i);
    if (artifact->kind == kind) {
      return artifact;
    }
  }
  return NULL;
}

static iree_status_t compile_source(jit_state_t* state,
                                    const char* include_root) {
  const iree_file_toc_t* file = loomc_cxx_example_kernels_create();
  const loomc_source_options_t source_options = {
      .identifier = loomc_make_cstring_view(file->name),
      .contents = loomc_make_byte_span(file->data, file->size),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(
      loomc_source_create(&source_options, loomc_allocator_system(), &source)));
  loomc_cxx_import_options_t import_options = {0};
  const loomc_string_view_t include_path =
      loomc_make_cstring_view(include_root);
  if (include_root) {
    import_options.flags = LOOMC_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES;
    import_options.system_include_paths = &include_path;
    import_options.system_include_path_count = 1;
  }
  iree_status_t status = iree_status_from_loomc(loomc_module_import_cxx(
      state->context, state->workspace, source, &import_options,
      loomc_allocator_system(), &state->module, &state->result));
  loomc_source_release(source);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(check_result(state));
  reset_result(state);

  // Config modules supply specialization values through the ordinary compiler
  // API. The imported range predicates validate these choices before lowering.
  const char* config_text =
      "config.def @residual.workgroup_count.x = 2 : index\n"
      "config.def @residual.workgroup_count.y = 1 : index\n"
      "config.def @residual.workgroup_count.z = 1 : index\n";
  const loomc_source_options_t config_source_options = {
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("launch.loom"),
      .contents = loomc_make_byte_span(config_text, strlen(config_text)),
  };
  source = NULL;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_source_create(
      &config_source_options, loomc_allocator_system(), &source)));
  status = iree_status_from_loomc(loomc_module_deserialize_text_from_source(
      state->context, state->workspace, source, NULL, loomc_allocator_system(),
      &state->config, &state->result));
  loomc_source_release(source);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(check_result(state));
  reset_result(state);

  const loomc_target_specialization_t specializations[] = {
      {loomc_make_cstring_view("affine"), state->profile},
      {loomc_make_cstring_view("residual"), state->profile},
  };
  const loomc_target_specialization_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .specializations = specializations,
      .specialization_count = IREE_ARRAYSIZE(specializations),
  };
  const loomc_compile_options_t compile_options = {
      .next = &target_options,
      .module_name = loomc_make_cstring_view("cxx_kernels"),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
      .config_flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      .config_module = state->config,
  };
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_compile_module(
      state->compiler, state->workspace, state->pipeline, state->module,
      &compile_options, loomc_allocator_system(), &state->result)));
  IREE_RETURN_IF_ERROR(check_result(state));
  const loomc_artifact_t* launch =
      find_artifact(state, LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG);
  if (!launch) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "launch artifact is absent");
  }
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_launch_config_program_load(
      launch, loomc_allocator_system(), &state->launch_program)));
  reset_result(state);

  const loomc_emit_options_t emit_options = {
      .artifact_format =
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_emit_module(
      state->target_environment, state->workspace, state->module, &emit_options,
      loomc_allocator_system(), &state->result)));
  IREE_RETURN_IF_ERROR(check_result(state));
  const loomc_artifact_t* hsaco =
      find_artifact(state, LOOMC_ARTIFACT_KIND_EXECUTABLE);
  if (!hsaco) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "HSACO artifact is absent");
  }
  loomc_byte_span_t contents = loomc_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_byte_sequence_clone(
      hsaco->contents, loomc_allocator_system(), &contents)));
  iree_hal_executable_load_params_t load;
  iree_hal_executable_load_params_initialize(&load);
  load.executable_data =
      iree_make_const_byte_span(contents.data, contents.data_length);
  status = iree_hal_executable_load(
      iree_hal_queue_family(state->dispatch_queue), state->executable_target,
      &load, &state->executable);
  if (iree_status_is_ok(status)) {
    printf("C++ -> Loom -> HSACO: %zu bytes; two kernels\n",
           contents.data_length);
  }
  loomc_allocator_free(loomc_allocator_system(), (void*)contents.data);
  reset_result(state);
  return status;
}

static iree_status_t execute_kernel(jit_state_t* state, const char* name,
                                    float factor) {
  loomc_launch_config_function_t launch_function =
      loomc_launch_config_function_invalid();
  IREE_RETURN_IF_ERROR(
      iree_status_from_loomc(loomc_launch_config_program_lookup_function(
          state->launch_program, loomc_make_cstring_view(name),
          &launch_function)));
  loomc_launch_config_t launch = {0};
  IREE_RETURN_IF_ERROR(
      iree_status_from_loomc(loomc_launch_config_program_invoke(
          state->launch_program, launch_function, NULL, 0, &launch)));
  iree_hal_executable_function_t function;
  IREE_RETURN_IF_ERROR(iree_hal_executable_lookup_function_by_name(
      state->executable, iree_make_cstring_view(name), &function));

  float input[kElementCount + 2 * kGuardCount];
  float output[IREE_ARRAYSIZE(input)];
  for (size_t i = 0; i < IREE_ARRAYSIZE(input); ++i) {
    input[i] = ((float)i - 64.0f) * 0.25f;
    output[i] = -12345.0f;
  }
  iree_hal_buffer_t* input_buffer = NULL;
  iree_hal_buffer_t* output_buffer = NULL;
  iree_hal_semaphore_t* semaphore = NULL;
  iree_status_t status =
      iree_hal_pool_allocate_buffer(state->pool,
                                    iree_hal_make_pool_reservation_request(
                                        sizeof(input), iree_alignof(float)),
                                    iree_infinite_timeout(), &input_buffer);
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_pool_allocate_buffer(state->pool,
                                      iree_hal_make_pool_reservation_request(
                                          sizeof(output), iree_alignof(float)),
                                      iree_infinite_timeout(), &output_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        state->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore);
  }
  uint64_t completion = 0;
  uint64_t upload = 1, dispatch = 2, download = 3;
  const iree_hal_semaphore_list_t uploaded = {1, &semaphore, &upload};
  const iree_hal_semaphore_list_t dispatched = {1, &semaphore, &dispatch};
  const iree_hal_semaphore_list_t downloaded = {1, &semaphore, &download};
  if (iree_status_is_ok(status)) {
    const iree_hal_transfer_operation_t operations[] = {
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
         .upload = {.source = input,
                    .target_buffer = input_buffer,
                    .length = sizeof(input)}},
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
         .upload = {.source = output,
                    .target_buffer = output_buffer,
                    .length = sizeof(output)}},
    };
    status = iree_hal_queue_transfer(
        state->transfer_queue, iree_hal_semaphore_list_empty(), uploaded,
        IREE_ARRAYSIZE(operations), operations, /*barriers=*/NULL);
    if (iree_status_is_ok(status)) {
      completion = upload;
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_ref_t references[] = {
        iree_hal_make_buffer_ref(input_buffer, kGuardCount * sizeof(float),
                                 kElementCount * sizeof(float)),
        iree_hal_make_buffer_ref(output_buffer, kGuardCount * sizeof(float),
                                 kElementCount * sizeof(float)),
    };
    const iree_hal_buffer_ref_list_t bindings = {IREE_ARRAYSIZE(references),
                                                 references};
    status = iree_hal_queue_dispatch(
        state->dispatch_queue, uploaded, dispatched, state->executable,
        function,
        iree_hal_make_static_dispatch_config(launch.workgroup_count.x,
                                             launch.workgroup_count.y,
                                             launch.workgroup_count.z),
        iree_const_byte_span_empty(), bindings, /*barriers=*/NULL,
        IREE_HAL_DISPATCH_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      completion = dispatch;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_download(state->transfer_queue, dispatched,
                                     downloaded, output_buffer, 0, output,
                                     sizeof(output), /*barriers=*/NULL);
    if (iree_status_is_ok(status)) {
      completion = download;
    }
  }
  // Host arrays and device allocations remain live through the last successful
  // submission, including when a subsequent queue operation fails.
  if (completion) {
    status = iree_status_join(
        status,
        iree_hal_semaphore_wait(semaphore, completion, iree_infinite_timeout(),
                                IREE_ASYNC_WAIT_FLAG_NONE));
  }
  for (size_t i = 0; i < IREE_ARRAYSIZE(output) && iree_status_is_ok(status);
       ++i) {
    float expected = i >= kGuardCount && i < kGuardCount + kElementCount
                         ? factor * input[i] + 1.0f
                         : -12345.0f;
    if (output[i] != expected) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "%s[%zu]: got %g, expected %g", name, i,
                                output[i], expected);
    }
  }
  if (iree_status_is_ok(status)) {
    printf("%s: %u values correct; guards intact\n", name, kElementCount);
  }
  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
  return status;
}

int main(int argc, char** argv) {
  if (argc > 3) {
    fprintf(stderr, "usage: jit_amdgpu [device-uri] [external-include-root]\n");
    return 1;
  }
  jit_state_t state = {0};
  iree_status_t status = initialize(&state, argc > 1 ? argv[1] : "amdgpu");
  if (iree_status_is_ok(status)) {
    status = compile_source(&state, argc > 2 ? argv[2] : NULL);
  }
  if (iree_status_is_ok(status)) {
    status = execute_kernel(&state, "affine", 3.0f);
  }
  if (iree_status_is_ok(status)) {
    status = execute_kernel(&state, "residual", 4.0f);
  }
  deinitialize(&state);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }
  return 0;
}
