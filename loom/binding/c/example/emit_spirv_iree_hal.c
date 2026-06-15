// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/init.h"
#include "loomc/iree.h"
#include "loomc/loomc.h"
#include "loomc/target/iree_hal.h"
#include "loomc/target/spirv.h"
#include "loomc/target/spirv/iree_hal.h"

static const char kDefaultDeviceUri[] = "vulkan";

static const char kSourceText[] =
    "spirv.target<vulkan1_3> @target {abi = hal_kernel}\n"
    "\n"
    "kernel.def target(@target) @double_i32_at_byte_offset() {\n"
    "  %unit = index.constant 1 : index\n"
    "  kernel.launch.config workgroups(%unit, %unit, %unit) "
    "workgroup_size(%unit, %unit, %unit) : index\n"
    "} launch(%input: buffer, %output: buffer, %byte_offset: offset) {\n"
    "  %byte_offset_aligned = index.assume %byte_offset [mul(%byte_offset, "
    "4)] : offset\n"
    "  %input_aligned = buffer.assume.alignment %input {minimum_alignment = "
    "4} : buffer\n"
    "  %output_aligned = buffer.assume.alignment %output {minimum_alignment = "
    "4} : buffer\n"
    "  %input_view = buffer.view %input_aligned[%byte_offset_aligned] : "
    "buffer -> view<1xi32, #dense>\n"
    "  %loaded = view.load %input_view[0] : view<1xi32, #dense> -> i32\n"
    "  %doubled = scalar.addi %loaded, %loaded : i32\n"
    "  %output_view = buffer.view %output_aligned[%byte_offset_aligned] : "
    "buffer -> view<1xi32, #dense>\n"
    "  view.store %doubled, %output_view[0] : i32, view<1xi32, #dense>\n"
    "  kernel.return\n"
    "}\n";

typedef struct emit_spirv_iree_hal_state_t {
  // IREE HAL device URI supplied by the caller.
  const char* device_uri;

  // Optional output path supplied by the caller.
  const char* output_path;

  // True when live HAL discovery found no usable SPIR-V/Vulkan device.
  bool skipped;

  // Human-readable reason printed to stderr when skipped is true.
  const char* skip_message;

  // Async I/O pool retained by the HAL device during creation.
  iree_async_proactor_pool_t* proactor_pool;

  // Live IREE HAL device used as the source of target facts.
  iree_hal_device_t* device;

  // Live IREE HAL executable cache used to validate SPIR-V loadability.
  iree_hal_executable_cache_t* executable_cache;

  // SPIR-V target package linked into this embedding binary.
  loomc_target_environment_t* target_environment;

  // Shared API context with the SPIR-V target dialect registered.
  loomc_context_t* context;

  // Per-worker scratch storage used by deserialize, compile, and emit.
  loomc_workspace_t* workspace;

  // Immutable source containing the Loom kernel module.
  loomc_source_t* source;

  // Mutable module compiled and emitted by this invocation.
  loomc_module_t* module;

  // Live IREE HAL-derived SPIR-V target profile.
  loomc_target_profile_t* target_profile;

  // Invocation-ready target selection derived from the profile.
  loomc_target_selection_t* target_selection;

  // Immutable prepared compiler handle.
  loomc_compiler_t* compiler;

  // Prepared target pipeline shared across invocations.
  loomc_pass_program_t* pass_program;

  // Last operation result, reset between phases.
  loomc_result_t* result;
} emit_spirv_iree_hal_state_t;

static void print_status(loomc_status_t status) {
  const char* code = loomc_status_code_string(loomc_status_code(status));
  loomc_string_view_t message = loomc_status_message(status);
  if (loomc_string_view_is_empty(message)) {
    fprintf(stderr, "%s\n", code);
    return;
  }
  fprintf(stderr, "%s: %.*s\n", code, (int)message.size, message.data);
}

static void print_iree_status_detail(const char* label, iree_status_t status) {
  const char* code = iree_status_code_string(iree_status_code(status));
  iree_string_view_t message = iree_status_message(status);
  if (iree_string_view_is_empty(message)) {
    fprintf(stderr, "%s: %s\n", label, code);
    return;
  }
  fprintf(stderr, "%s: %s: %.*s\n", label, code, (int)message.size,
          message.data);
}

static void print_result_diagnostics(const loomc_result_t* result) {
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    if (diagnostic == NULL) {
      continue;
    }
    fprintf(stderr, "%.*s: %.*s\n", (int)diagnostic->code.size,
            diagnostic->code.data, (int)diagnostic->message.size,
            diagnostic->message.data);
  }
}

static void emit_spirv_iree_hal_state_initialize(
    emit_spirv_iree_hal_state_t* state, const char* device_uri,
    const char* output_path) {
  memset(state, 0, sizeof(*state));
  state->device_uri = device_uri;
  state->output_path = output_path;
}

static void emit_spirv_iree_hal_state_deinitialize(
    emit_spirv_iree_hal_state_t* state) {
  loomc_result_release(state->result);
  loomc_pass_program_release(state->pass_program);
  loomc_compiler_release(state->compiler);
  loomc_target_selection_release(state->target_selection);
  loomc_target_profile_release(state->target_profile);
  loomc_module_release(state->module);
  loomc_source_release(state->source);
  loomc_workspace_release(state->workspace);
  loomc_context_release(state->context);
  loomc_target_environment_release(state->target_environment);
  iree_hal_executable_cache_release(state->executable_cache);
  iree_hal_device_release(state->device);
  iree_async_proactor_pool_release(state->proactor_pool);
}

static void emit_spirv_iree_hal_state_reset_result(
    emit_spirv_iree_hal_state_t* state) {
  loomc_result_release(state->result);
  state->result = NULL;
}

static void emit_spirv_iree_hal_state_skip(emit_spirv_iree_hal_state_t* state,
                                           const char* message) {
  state->skipped = true;
  state->skip_message = message;
}

static bool iree_status_code_indicates_live_skip(iree_status_code_t code) {
  switch (code) {
    case IREE_STATUS_NOT_FOUND:
    case IREE_STATUS_FAILED_PRECONDITION:
    case IREE_STATUS_UNAVAILABLE:
    case IREE_STATUS_INCOMPATIBLE:
      return true;
    default:
      return false;
  }
}

static loomc_status_t require_successful_result(const loomc_result_t* result,
                                                const char* failure_message) {
  if (loomc_result_succeeded(result)) {
    return loomc_ok_status();
  }
  print_result_diagnostics(result);
  return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION, failure_message);
}

static const loomc_artifact_t* find_result_artifact(
    const loomc_result_t* result, loomc_artifact_kind_t kind,
    loomc_string_view_t format) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact == NULL) {
      continue;
    }
    if (artifact->kind == kind &&
        loomc_string_view_equal(artifact->format, format)) {
      return artifact;
    }
  }
  return NULL;
}

static loomc_status_t create_hal_device_and_cache(
    emit_spirv_iree_hal_state_t* state) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t iree_status = iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default());
  if (!iree_status_is_ok(iree_status)) {
    return loomc_status_from_iree(iree_status);
  }

  iree_status = iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/NULL,
      iree_async_proactor_pool_options_default(), host_allocator,
      &state->proactor_pool);
  if (!iree_status_is_ok(iree_status)) {
    return loomc_status_from_iree(iree_status);
  }

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = state->proactor_pool;
  iree_status =
      iree_hal_create_device(iree_hal_driver_registry_default(),
                             iree_make_cstring_view(state->device_uri),
                             &create_params, host_allocator, &state->device);
  if (!iree_status_is_ok(iree_status)) {
    if (iree_status_code_indicates_live_skip(iree_status_code(iree_status))) {
      print_iree_status_detail("IREE HAL device creation failed", iree_status);
      iree_status_free(iree_status);
      emit_spirv_iree_hal_state_skip(
          state, "IREE HAL device could not be created on this machine");
      return loomc_ok_status();
    }
    return loomc_status_from_iree(iree_status);
  }

  iree_status = iree_hal_executable_cache_create(
      state->device, iree_make_cstring_view("loomc-example"),
      &state->executable_cache);
  if (!iree_status_is_ok(iree_status)) {
    return loomc_status_from_iree(iree_status);
  }
  return loomc_ok_status();
}

static loomc_status_t create_target_environment_and_context(
    emit_spirv_iree_hal_state_t* state) {
  loomc_status_t status = loomc_target_environment_create_spirv(
      loomc_allocator_system(), &state->target_environment);
  loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_environment = state->target_environment,
  };
  loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(context_options),
      .next = &target_options,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_context_create(&context_options, loomc_allocator_system(),
                                  &state->context);
  }
  return status;
}

static loomc_status_t create_workspace_and_source(
    emit_spirv_iree_hal_state_t* state) {
  loomc_status_t status =
      loomc_workspace_create(NULL, loomc_allocator_system(), &state->workspace);
  loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("double_i32_at_byte_offset.loom"),
      .contents = loomc_make_byte_span(kSourceText, sizeof(kSourceText) - 1),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_source_create(&source_options, loomc_allocator_system(),
                                 &state->source);
  }
  return status;
}

static loomc_status_t create_target_profile_and_selection(
    emit_spirv_iree_hal_state_t* state) {
  const loomc_iree_hal_profile_provider_t* providers[] = {
      loomc_spirv_iree_hal_profile_provider(),
  };
  loomc_iree_hal_profile_options_t profile_options = {
      .type = LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      .structure_size = sizeof(profile_options),
      .identifier = loomc_make_cstring_view("iree-hal-device"),
      .device = state->device,
      .executable_cache = state->executable_cache,
      .providers = providers,
      .provider_count = 1,
  };
  loomc_status_t status = loomc_target_profile_create_iree_hal(
      state->target_environment, &profile_options, loomc_allocator_system(),
      &state->target_profile, &state->result);
  if (loomc_status_is_ok(status) && !loomc_result_succeeded(state->result)) {
    print_result_diagnostics(state->result);
    emit_spirv_iree_hal_state_skip(
        state,
        "IREE HAL device did not produce a usable SPIR-V target profile");
  }
  if (loomc_status_is_ok(status) && !state->skipped) {
    emit_spirv_iree_hal_state_reset_result(state);
    status = loomc_target_selection_create_from_profile(
        state->target_profile, loomc_allocator_system(),
        &state->target_selection);
  }
  return status;
}

static loomc_status_t require_live_profile_feature(
    emit_spirv_iree_hal_state_t* state, loomc_spirv_feature_t feature,
    const char* skip_message) {
  loomc_target_fact_state_t feature_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  loomc_status_t status = loomc_spirv_target_profile_query_feature(
      state->target_profile, feature, &feature_state);
  if (loomc_status_is_ok(status) &&
      feature_state != LOOMC_TARGET_FACT_STATE_TRUE) {
    emit_spirv_iree_hal_state_skip(state, skip_message);
  }
  return status;
}

static loomc_status_t check_live_profile_support(
    emit_spirv_iree_hal_state_t* state) {
  loomc_status_t status = require_live_profile_feature(
      state, LOOMC_SPIRV_FEATURE_VULKAN_SHADER,
      "IREE HAL device cannot run Vulkan SPIR-V shaders");
  if (loomc_status_is_ok(status) && !state->skipped) {
    status = require_live_profile_feature(
        state, LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
        "IREE HAL device does not expose buffer device address");
  }
  return status;
}

static uint32_t spirv_version_major(uint64_t version) {
  return (uint32_t)((version >> 16) & UINT64_C(0xFF));
}

static uint32_t spirv_version_minor(uint64_t version) {
  return (uint32_t)((version >> 8) & UINT64_C(0xFF));
}

static loomc_status_t print_live_profile_summary(
    const emit_spirv_iree_hal_state_t* state) {
  iree_string_view_t device_id = iree_hal_device_id(state->device);
  if (iree_string_view_is_empty(device_id)) {
    device_id = iree_make_cstring_view("(unnamed)");
  }
  loomc_spirv_environment_value_t max_spirv = {0};
  loomc_spirv_limit_value_t flat_workgroup_size = {0};
  loomc_spirv_limit_value_t subgroup_size = {0};
  LOOMC_RETURN_IF_ERROR(loomc_spirv_target_profile_query_environment(
      state->target_profile, LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
      &max_spirv));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_target_profile_query_limit(
      state->target_profile, LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE,
      &flat_workgroup_size));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_target_profile_query_limit(
      state->target_profile, LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE, &subgroup_size));
  printf("device %.*s max_spirv=%" PRIu32 ".%" PRIu32 " max_workgroup=%" PRIu64
         " subgroup=%" PRIu64 "\n",
         (int)device_id.size, device_id.data,
         spirv_version_major(max_spirv.value),
         spirv_version_minor(max_spirv.value), flat_workgroup_size.value,
         subgroup_size.value);
  return loomc_ok_status();
}

static loomc_status_t create_compiler_and_target_pipeline(
    emit_spirv_iree_hal_state_t* state) {
  loomc_status_t status = loomc_compiler_create(
      state->context, NULL, loomc_allocator_system(), &state->compiler);
  loomc_target_selection_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_selection = state->target_selection,
  };
  loomc_target_pipeline_options_t pipeline_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      .structure_size = sizeof(pipeline_options),
      .next = &target_options,
      .identifier = loomc_make_cstring_view("iree-hal-spirv-prepared-low"),
      .kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      .source_to_low_max_errors = 20,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_create_from_target_pipeline(
        state->context, &pipeline_options, loomc_allocator_system(),
        &state->pass_program, &state->result);
  }
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result,
                                       "target pipeline preparation failed");
  }
  if (loomc_status_is_ok(status)) {
    emit_spirv_iree_hal_state_reset_result(state);
  }
  return status;
}

static loomc_status_t create_resources(emit_spirv_iree_hal_state_t* state) {
  loomc_status_t status = create_target_environment_and_context(state);
  if (loomc_status_is_ok(status)) {
    status = create_workspace_and_source(state);
  }
  if (loomc_status_is_ok(status)) {
    status = create_target_profile_and_selection(state);
  }
  if (loomc_status_is_ok(status) && !state->skipped) {
    status = check_live_profile_support(state);
  }
  if (loomc_status_is_ok(status) && !state->skipped) {
    status = print_live_profile_summary(state);
  }
  if (loomc_status_is_ok(status) && !state->skipped) {
    status = create_compiler_and_target_pipeline(state);
  }
  return status;
}

static loomc_status_t deserialize_source(emit_spirv_iree_hal_state_t* state) {
  loomc_status_t status = loomc_module_deserialize_from_source(
      state->context, state->workspace, state->source, NULL,
      loomc_allocator_system(), &state->module, &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result,
                                       "source deserialization failed");
  }
  if (loomc_status_is_ok(status)) {
    emit_spirv_iree_hal_state_reset_result(state);
  }
  return status;
}

static loomc_status_t compile_module_to_prepared_low(
    emit_spirv_iree_hal_state_t* state) {
  loomc_target_selection_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_selection = state->target_selection,
  };
  loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = &target_options,
      .module_name = loomc_make_cstring_view("double_i32_at_byte_offset"),
  };
  loomc_status_t status = loomc_compile_module(
      state->compiler, state->workspace, state->pass_program, state->module,
      &compile_options, loomc_allocator_system(), &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result, "compilation failed");
  }
  if (loomc_status_is_ok(status)) {
    emit_spirv_iree_hal_state_reset_result(state);
  }
  return status;
}

static loomc_status_t emit_spirv_artifact(emit_spirv_iree_hal_state_t* state) {
  loomc_target_selection_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_selection = state->target_selection,
  };
  loomc_spirv_emit_options_t spirv_options = {
      .type = LOOMC_STRUCTURE_TYPE_SPIRV_EMIT_OPTIONS,
      .structure_size = sizeof(spirv_options),
      .next = &target_options,
  };
  const loomc_option_entry_t emit_entries[] = {
      {
          .key = loomc_make_cstring_view(LOOMC_EMIT_OPTION_KEY_IDENTIFIER),
          .value = loomc_make_cstring_view("double_i32_at_byte_offset.spv"),
      },
  };
  loomc_option_dict_t option_dict = {
      .type = LOOMC_STRUCTURE_TYPE_OPTION_DICT,
      .structure_size = sizeof(option_dict),
      .next = &spirv_options,
      .entries = emit_entries,
      .entry_count = 1,
  };
  loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = &option_dict,
      .artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  loomc_status_t status = loomc_emit_module(
      state->target_environment, state->workspace, state->module, &emit_options,
      loomc_allocator_system(), &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result, "SPIR-V emission failed");
  }
  return status;
}

static loomc_status_t summarize_and_maybe_write_artifact(
    emit_spirv_iree_hal_state_t* state) {
  const loomc_artifact_t* artifact = find_result_artifact(
      state->result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV));
  if (artifact == NULL) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "SPIR-V executable artifact was not produced");
  }
  if (artifact->contents.data_length < sizeof(uint32_t)) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "SPIR-V artifact is too small");
  }

  uint32_t magic = 0;
  memcpy(&magic, artifact->contents.data, sizeof(magic));
  printf("artifact %.*s format=%.*s bytes=%zu magic=0x%08" PRIx32 "\n",
         (int)artifact->identifier.size, artifact->identifier.data,
         (int)artifact->format.size, artifact->format.data,
         (size_t)artifact->contents.data_length, magic);

  if (state->output_path == NULL) {
    return loomc_ok_status();
  }
  loomc_status_t status = loomc_artifact_write_to_path(
      artifact, loomc_make_cstring_view(state->output_path),
      loomc_allocator_system());
  if (loomc_status_is_ok(status)) {
    printf("wrote %s\n", state->output_path);
  }
  return status;
}

static loomc_status_t run_emit_spirv_iree_hal_example(const char* device_uri,
                                                      const char* output_path,
                                                      bool* out_skipped) {
  emit_spirv_iree_hal_state_t state;
  emit_spirv_iree_hal_state_initialize(&state, device_uri, output_path);

  loomc_status_t status = create_hal_device_and_cache(&state);
  if (loomc_status_is_ok(status) && !state.skipped) {
    status = create_resources(&state);
  }
  if (loomc_status_is_ok(status) && !state.skipped) {
    status = deserialize_source(&state);
  }
  if (loomc_status_is_ok(status) && !state.skipped) {
    status = compile_module_to_prepared_low(&state);
  }
  if (loomc_status_is_ok(status) && !state.skipped) {
    status = emit_spirv_artifact(&state);
  }
  if (loomc_status_is_ok(status) && !state.skipped) {
    status = summarize_and_maybe_write_artifact(&state);
  }

  *out_skipped = state.skipped;
  if (state.skipped) {
    fprintf(stderr, "skipping live IREE HAL SPIR-V example: %s\n",
            state.skip_message);
  }
  emit_spirv_iree_hal_state_deinitialize(&state);
  return status;
}

int main(int argc, char** argv) {
  const char* device_uri = kDefaultDeviceUri;
  const char* output_path = NULL;
  loomc_status_t status = loomc_ok_status();
  if (argc > 3) {
    status = loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "usage: emit_spirv_iree_hal [device-uri] [output.spv]");
  } else {
    device_uri = argc > 1 ? argv[1] : kDefaultDeviceUri;
    output_path = argc > 2 ? argv[2] : NULL;
  }

  bool skipped = false;
  if (loomc_status_is_ok(status)) {
    status = run_emit_spirv_iree_hal_example(device_uri, output_path, &skipped);
  }
  if (loomc_status_is_ok(status)) {
    return 0;
  }
  print_status(status);
  loomc_status_free(status);
  return skipped ? 0 : 1;
}
