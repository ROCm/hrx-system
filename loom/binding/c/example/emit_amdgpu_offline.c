// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"

static const char kSourceText[] =
    "kernel.def export(\"targetless_store_i32\") @targetless_store_i32() {\n"
    "  %unit = index.constant 1 : index\n"
    "  kernel.launch.config workgroups(%unit, %unit, %unit) "
    "workgroup_size(%unit, %unit, %unit) : index\n"
    "} launch(%output: buffer) {\n"
    "  %zero_offset = index.constant 0 : offset\n"
    "  %zero_index = index.constant 0 : index\n"
    "  %value = scalar.constant 42 : i32\n"
    "  %global = buffer.assume.memory_space<global> %output : buffer\n"
    "  %view = buffer.view %global[%zero_offset] : buffer -> view<1xi32, "
    "#dense>\n"
    "  view.store %value, %view[%zero_index] : i32, view<1xi32, #dense>\n"
    "  kernel.return\n"
    "}\n";

typedef struct emit_amdgpu_offline_state_t {
  // AMDGPU processor key, such as `gfx11-generic`, `gfx1100`, or `gfx942`.
  const char* processor;

  // Optional output path supplied by the caller.
  const char* output_path;

  // Optional artifact manifest output path supplied by the caller.
  const char* manifest_output_path;

  // AMDGPU target package linked into this embedding binary.
  loomc_target_environment_t* target_environment;

  // Shared API context with the AMDGPU target dialect registered.
  loomc_context_t* context;

  // Per-worker scratch storage used by deserialize, compile, and emit.
  loomc_workspace_t* workspace;

  // Immutable source containing the Loom kernel module.
  loomc_source_t* source;

  // Mutable module compiled and emitted by this invocation.
  loomc_module_t* module;

  // Offline AMDGPU processor target profile.
  loomc_target_profile_t* target_profile;

  // Invocation-ready target selection derived from the profile.
  loomc_target_selection_t* target_selection;

  // Immutable prepared compiler handle.
  loomc_compiler_t* compiler;

  // Prepared target pipeline shared across invocations.
  loomc_pass_program_t* pass_program;

  // Last operation result, reset between phases.
  loomc_result_t* result;
} emit_amdgpu_offline_state_t;

static void print_status(loomc_status_t status) {
  char buffer[1024] = {0};
  loomc_host_size_t length = 0;
  loomc_status_format(status, sizeof(buffer), buffer, &length);
  fprintf(stderr, "%.*s\n", (int)length, buffer);
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

static void print_usage(FILE* file) {
  fprintf(file,
          "Usage: emit_amdgpu_offline [processor [output.hsaco "
          "[manifest.json]]]\n");
  fprintf(
      file,
      "  processor    AMDGPU processor key, such as gfx11-generic, gfx1100, "
      "or gfx942.\n");
  fprintf(file,
          "  output.hsaco Optional path for the emitted AMDGPU HSACO ELF "
          "artifact.\n");
  fprintf(file,
          "  manifest.json Optional path for the emitted artifact manifest "
          "JSON sidecar.\n");
  fprintf(file,
          "Omitting processor uses gfx11-generic. The example always compiles "
          "the embedded targetless_store_i32 kernel.\n");
}

static void emit_amdgpu_offline_state_initialize(
    emit_amdgpu_offline_state_t* state, const char* processor,
    const char* output_path, const char* manifest_output_path) {
  memset(state, 0, sizeof(*state));
  state->processor = processor;
  state->output_path = output_path;
  state->manifest_output_path = manifest_output_path;
}

static void emit_amdgpu_offline_state_deinitialize(
    emit_amdgpu_offline_state_t* state) {
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
}

static void emit_amdgpu_offline_state_reset_result(
    emit_amdgpu_offline_state_t* state) {
  loomc_result_release(state->result);
  state->result = NULL;
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

static loomc_status_t create_target_environment_and_context(
    emit_amdgpu_offline_state_t* state) {
  loomc_status_t status = loomc_target_environment_create_amdgpu(
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
    emit_amdgpu_offline_state_t* state) {
  loomc_status_t status =
      loomc_workspace_create(NULL, loomc_allocator_system(), &state->workspace);
  loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("targetless_store_i32.loom"),
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
    emit_amdgpu_offline_state_t* state) {
  loomc_amdgpu_profile_options_t profile_options = {
      .type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      .structure_size = sizeof(profile_options),
      .identifier = loomc_make_cstring_view("offline-amdgpu"),
      .processor = loomc_make_cstring_view(state->processor),
  };
  loomc_status_t status = loomc_target_profile_create_amdgpu(
      state->target_environment, &profile_options, loomc_allocator_system(),
      &state->target_profile);
  if (loomc_status_is_ok(status)) {
    status = loomc_target_selection_create_from_profile(
        state->target_profile, loomc_allocator_system(),
        &state->target_selection);
  }
  return status;
}

static loomc_status_t create_compiler_and_target_pipeline(
    emit_amdgpu_offline_state_t* state) {
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
      .identifier = loomc_make_cstring_view("offline-amdgpu-prepared-low"),
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
    emit_amdgpu_offline_state_reset_result(state);
  }
  return status;
}

static loomc_status_t create_resources(emit_amdgpu_offline_state_t* state) {
  loomc_status_t status = create_target_environment_and_context(state);
  if (loomc_status_is_ok(status)) {
    status = create_workspace_and_source(state);
  }
  if (loomc_status_is_ok(status)) {
    status = create_target_profile_and_selection(state);
  }
  if (loomc_status_is_ok(status)) {
    status = create_compiler_and_target_pipeline(state);
  }
  return status;
}

static loomc_status_t deserialize_source(emit_amdgpu_offline_state_t* state) {
  loomc_status_t status = loomc_module_deserialize_from_source(
      state->context, state->workspace, state->source, NULL,
      loomc_allocator_system(), &state->module, &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result,
                                       "source deserialization failed");
  }
  if (loomc_status_is_ok(status)) {
    emit_amdgpu_offline_state_reset_result(state);
  }
  return status;
}

static loomc_status_t compile_module_to_prepared_low(
    emit_amdgpu_offline_state_t* state) {
  loomc_target_selection_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_selection = state->target_selection,
  };
  loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = &target_options,
      .module_name = loomc_make_cstring_view("targetless_store_i32"),
  };
  loomc_status_t status = loomc_compile_module(
      state->compiler, state->workspace, state->pass_program, state->module,
      &compile_options, loomc_allocator_system(), &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result, "compilation failed");
  }
  if (loomc_status_is_ok(status)) {
    emit_amdgpu_offline_state_reset_result(state);
  }
  return status;
}

static loomc_status_t emit_amdgpu_artifact(emit_amdgpu_offline_state_t* state) {
  loomc_target_selection_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_selection = state->target_selection,
  };
  loomc_amdgpu_emit_options_t amdgpu_options = {
      .type = LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
      .structure_size = sizeof(amdgpu_options),
      .next = &target_options,
  };
  const loomc_option_entry_t emit_entries[] = {
      {
          .key = loomc_make_cstring_view(LOOMC_EMIT_OPTION_KEY_IDENTIFIER),
          .value = loomc_make_cstring_view("targetless_store_i32.hsaco"),
      },
  };
  loomc_option_dict_t option_dict = {
      .type = LOOMC_STRUCTURE_TYPE_OPTION_DICT,
      .structure_size = sizeof(option_dict),
      .next = &amdgpu_options,
      .entries = emit_entries,
      .entry_count = 1,
  };
  loomc_artifact_manifest_options_t manifest_options = {
      .type = LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS,
      .structure_size = sizeof(manifest_options),
      .next = &option_dict,
      .mode = LOOMC_ARTIFACT_MANIFEST_MODE_DETAILS,
      .identifier = loomc_string_view_empty(),
  };
  loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = state->manifest_output_path != NULL ? &manifest_options
                                                  : (const void*)&option_dict,
      .artifact_format =
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  loomc_status_t status = loomc_emit_module(
      state->target_environment, state->workspace, state->module, &emit_options,
      loomc_allocator_system(), &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result, "AMDGPU emission failed");
  }
  return status;
}

static loomc_status_t summarize_and_maybe_write_manifest(
    emit_amdgpu_offline_state_t* state) {
  if (state->manifest_output_path == NULL) {
    return loomc_ok_status();
  }
  const loomc_artifact_t* manifest = find_result_artifact(
      state->result, LOOMC_ARTIFACT_KIND_REPORT,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_ARTIFACT_MANIFEST_JSON));
  if (manifest == NULL) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "artifact manifest sidecar was not produced");
  }
  if (manifest->contents.data_length == 0 ||
      manifest->contents.data[0] != '{') {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "artifact manifest sidecar is not JSON");
  }

  printf("manifest %.*s format=%.*s bytes=%zu magic=JSON\n",
         (int)manifest->identifier.size, manifest->identifier.data,
         (int)manifest->format.size, manifest->format.data,
         (size_t)manifest->contents.data_length);
  loomc_status_t status = loomc_artifact_write_to_path(
      manifest, loomc_make_cstring_view(state->manifest_output_path),
      loomc_allocator_system());
  if (loomc_status_is_ok(status)) {
    printf("wrote %s\n", state->manifest_output_path);
  }
  return status;
}

static loomc_status_t summarize_and_maybe_write_artifact(
    emit_amdgpu_offline_state_t* state) {
  const loomc_artifact_t* artifact = find_result_artifact(
      state->result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO));
  if (artifact == NULL) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "AMDGPU HSACO artifact was not produced");
  }
  const uint8_t elf_magic[] = {0x7f, 'E', 'L', 'F'};
  if (artifact->contents.data_length < sizeof(elf_magic) ||
      memcmp(artifact->contents.data, elf_magic, sizeof(elf_magic)) != 0) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "AMDGPU HSACO artifact is not an ELF image");
  }

  printf("artifact %.*s format=%.*s bytes=%zu magic=ELF\n",
         (int)artifact->identifier.size, artifact->identifier.data,
         (int)artifact->format.size, artifact->format.data,
         (size_t)artifact->contents.data_length);

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

static loomc_status_t run_emit_amdgpu_offline_example(
    const char* processor, const char* output_path, const char* manifest_path) {
  emit_amdgpu_offline_state_t state;
  emit_amdgpu_offline_state_initialize(&state, processor, output_path,
                                       manifest_path);

  loomc_status_t status = create_resources(&state);
  if (loomc_status_is_ok(status)) {
    status = deserialize_source(&state);
  }
  if (loomc_status_is_ok(status)) {
    status = compile_module_to_prepared_low(&state);
  }
  if (loomc_status_is_ok(status)) {
    status = emit_amdgpu_artifact(&state);
  }
  if (loomc_status_is_ok(status)) {
    status = summarize_and_maybe_write_artifact(&state);
  }
  if (loomc_status_is_ok(status)) {
    status = summarize_and_maybe_write_manifest(&state);
  }

  emit_amdgpu_offline_state_deinitialize(&state);
  return status;
}

int main(int argc, char** argv) {
  if (argc > 1 &&
      (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    print_usage(stdout);
    return 0;
  }
  if (argc > 4) {
    print_usage(stderr);
    return 1;
  }

  const char* processor = argc > 1 ? argv[1] : "gfx11-generic";
  const char* output_path = argc > 2 ? argv[2] : NULL;
  const char* manifest_path = argc > 3 ? argv[3] : NULL;
  loomc_status_t status =
      run_emit_amdgpu_offline_example(processor, output_path, manifest_path);
  if (loomc_status_is_ok(status)) {
    return 0;
  }
  print_status(status);
  loomc_status_free(status);
  return 1;
}
