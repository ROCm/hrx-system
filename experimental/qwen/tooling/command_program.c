// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/tooling/command_program.h"

#include <inttypes.h>
#include <string.h>

#include "iree/io/parameter_provider.h"
#include "loomc/iree.h"
#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/amdgpu/iree_hal.h"
#include "loomc/target/cmd/hal.h"
#include "loomc/target/cmd/program_plan.h"

typedef struct qwen_tooling_parameter_request_t {
  // Concrete GGUF tensor key borrowed from the loaded command package.
  iree_string_view_t key;
  // Source and packed target range for the tensor payload.
  iree_io_parameter_span_t span;
} qwen_tooling_parameter_request_t;

// One physical immutable parameter slab shared by all selected program roots.
typedef struct qwen_tooling_parameter_pack_t {
  // Single host allocation containing all trailing metadata arrays.
  void* storage;
  // Device-local slab populated by the parameter provider.
  iree_hal_buffer_t* buffer;
  // Signals 1 after all unique parameter payloads have been gathered.
  iree_hal_semaphore_t* ready_semaphore;
  // Unique parameter requests packed into |buffer|.
  qwen_tooling_parameter_request_t* requests;
  // Number of entries populated in |requests|.
  iree_host_size_t request_count;
  // Root-local fixed-buffer views across all selected programs.
  iree_hal_buffer_ref_t* fixed_buffer_refs;
  // Number of entries populated in |fixed_buffer_refs|.
  iree_host_size_t fixed_buffer_ref_count;
  // Per-program starting offsets in |fixed_buffer_refs| plus one sentinel.
  iree_host_size_t* program_fixed_buffer_offsets;
  // Exact physical slab byte length.
  iree_device_size_t byte_length;
  // Minimum alignment required for the physical slab allocation.
  iree_device_size_t minimum_alignment;
} qwen_tooling_parameter_pack_t;

struct qwen_tooling_command_program_t {
  // Materialized reusable command buffer and retained fixed resources.
  loomc_cmd_hal_program_t* hal_program;
  // Shared launch-config evaluator borrowed from the owning program set.
  loomc_launch_config_program_t* launch_config_program;
  // Program-local token for the selected public command root.
  loomc_launch_config_function_t launch_config_function;
  // Immutable root ABI borrowed from package.
  loomc_cmd_program_info_t info;
};

struct qwen_tooling_command_program_set_t {
  // Allocator used to release this object.
  iree_allocator_t host_allocator;
  // Package retained so strings borrowed by root info remain valid.
  loomc_cmd_program_package_t* package;
  // Shared evaluator for issue-time indirect dispatch and dynamic constants.
  loomc_launch_config_program_t* launch_config_program;
  // Materialized roots stored inline in this allocation.
  qwen_tooling_command_program_t* programs;
  // Number of entries in |programs|.
  iree_host_size_t program_count;
  // Packed fixed-parameter root size in bytes.
  iree_device_size_t parameter_byte_length;
};

static iree_status_t qwen_tooling_status_from_loomc(loomc_status_t status) {
  return iree_status_from_loomc(status);
}

static iree_status_t qwen_tooling_require_result(iree_string_view_t phase,
                                                 const loomc_result_t* result) {
  if (result && loomc_result_succeeded(result)) return iree_ok_status();
  iree_string_view_t message = IREE_SV("Loom operation failed");
  if (result && loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    if (diagnostic) message = iree_string_view_from_loomc(diagnostic->message);
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "%.*s: %.*s",
                          (int)phase.size, phase.data, (int)message.size,
                          message.data);
}

static const loomc_artifact_t* qwen_tooling_find_artifact(
    const loomc_result_t* result, loomc_artifact_kind_t kind,
    const char* format) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact && artifact->kind == kind &&
        loomc_string_view_equal(artifact->format,
                                loomc_make_cstring_view(format))) {
      return artifact;
    }
  }
  return NULL;
}

static iree_status_t qwen_tooling_parameter_enumerate(
    void* user_data, iree_host_size_t index, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  const qwen_tooling_parameter_request_t* requests =
      (const qwen_tooling_parameter_request_t*)user_data;
  *out_key = requests[index].key;
  *out_span = requests[index].span;
  return iree_ok_status();
}

static iree_status_t qwen_tooling_allocate_buffer(
    iree_hal_device_t* device, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t usage, iree_device_size_t minimum_alignment,
    iree_device_size_t byte_length, iree_hal_buffer_t** out_buffer) {
  const iree_hal_buffer_params_t params = {
      .usage = usage,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = memory_type,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = minimum_alignment,
  };
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, byte_length, out_buffer);
}

static iree_status_t qwen_tooling_load_executable(
    iree_hal_device_t* device, const loomc_artifact_t* artifact,
    iree_hal_executable_t** out_executable) {
  const iree_hal_executable_target_selection_t selection = {
      .family = IREE_SV("amdgpu"),
      .target_key = iree_string_view_empty(),
      .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      .physical_device_affinity = 0,
  };
  const iree_hal_executable_target_selection_result_t selected =
      iree_hal_device_spec_select_executable_target(
          iree_hal_device_spec(device), &selection);
  if (selected.outcome !=
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AMDGPU HAL target selection failed");
  }

  iree_hal_executable_load_params_t params;
  iree_hal_executable_load_params_initialize(&params);
  params.executable_data = iree_make_const_byte_span(
      artifact->contents.data, artifact->contents.data_length);
  return iree_hal_device_load_executable(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                         selected.target, &params,
                                         out_executable);
}

static iree_status_t qwen_tooling_query_kernels(
    loomc_module_t* module, iree_allocator_t host_allocator,
    loomc_module_function_t** out_functions,
    iree_host_size_t* out_function_count) {
  *out_functions = NULL;
  *out_function_count = 0;
  const loomc_module_function_query_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
      .structure_size = sizeof(options),
      .kind = LOOMC_MODULE_FUNCTION_KIND_KERNEL,
  };

  loomc_result_t* result = NULL;
  iree_status_t status =
      qwen_tooling_status_from_loomc(loomc_module_query_functions(
          module, &options, loomc_allocator_from_iree(host_allocator), 0, NULL,
          out_function_count, &result));
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_require_result(IREE_SV("kernel query"), result);
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status) && *out_function_count == 0) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Qwen command source contains no kernels");
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, *out_function_count,
                                         sizeof(**out_functions),
                                         (void**)out_functions);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_status_from_loomc(loomc_module_query_functions(
        module, &options, loomc_allocator_from_iree(host_allocator),
        *out_function_count, *out_functions, out_function_count, &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_require_result(IREE_SV("kernel query"), result);
  }
  loomc_result_release(result);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, *out_functions);
    *out_functions = NULL;
    *out_function_count = 0;
  }
  return status;
}

static iree_status_t qwen_tooling_compile_unit(
    const loomc_program_plan_t* plan, loomc_compiler_t* compiler,
    loomc_workspace_t* workspace, loomc_program_plan_unit_t unit,
    const loomc_pass_program_t* pass_program, iree_string_view_t phase,
    iree_allocator_t host_allocator, loomc_result_t** out_result) {
  *out_result = NULL;
  iree_status_t status =
      qwen_tooling_status_from_loomc(loomc_program_plan_compile_unit(
          plan, compiler, workspace, unit, pass_program, /*options=*/NULL,
          loomc_allocator_from_iree(host_allocator), out_result));
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_require_result(phase, *out_result);
  }
  if (!iree_status_is_ok(status)) {
    loomc_result_release(*out_result);
    *out_result = NULL;
  }
  return status;
}

static void qwen_tooling_parameter_pack_deinitialize(
    iree_allocator_t host_allocator, qwen_tooling_parameter_pack_t* pack) {
  iree_hal_semaphore_release(pack->ready_semaphore);
  iree_hal_buffer_release(pack->buffer);
  iree_allocator_free(host_allocator, pack->storage);
  memset(pack, 0, sizeof(*pack));
}

static iree_host_size_t qwen_tooling_parameter_pack_find(
    const qwen_tooling_parameter_pack_t* pack, iree_string_view_t key) {
  for (iree_host_size_t i = 0; i < pack->request_count; ++i) {
    if (iree_string_view_equal(pack->requests[i].key, key)) return i;
  }
  return IREE_HOST_SIZE_MAX;
}

static bool qwen_tooling_parameter_ranges_overlap(
    iree_device_size_t lhs_offset, iree_device_size_t lhs_length,
    iree_device_size_t rhs_offset, iree_device_size_t rhs_length) {
  return lhs_offset < rhs_offset ? lhs_length > rhs_offset - lhs_offset
                                 : rhs_length > lhs_offset - rhs_offset;
}

// Packs every selected program's logical parameter roots into one physical
// slab. A shared parameter key fixes a root's base relative to an existing
// placement. Roots with no shared keys append once. Conflicting relative
// layouts cannot be represented by fixed-buffer subranges and fail loud.
static iree_status_t qwen_tooling_parameter_pack_build(
    loomc_cmd_program_package_t* package,
    const loomc_cmd_program_export_t* program_exports,
    const loomc_cmd_program_info_t* program_infos,
    iree_host_size_t program_count, iree_allocator_t host_allocator,
    qwen_tooling_parameter_pack_t* out_pack) {
  memset(out_pack, 0, sizeof(*out_pack));

  iree_host_size_t request_capacity = 0;
  iree_host_size_t fixed_buffer_ref_capacity = 0;
  for (iree_host_size_t i = 0; i < program_count; ++i) {
    if (program_infos[i].parameter_count == 0 ||
        program_infos[i].parameter_root_count == 0 ||
        program_infos[i].parameter_root_count !=
            program_infos[i].fixed_buffer_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "command root '%.*s' must source every fixed buffer from parameters",
          (int)program_infos[i].name.size, program_infos[i].name.data);
    }
    if (!iree_host_size_checked_add(request_capacity,
                                    program_infos[i].parameter_count,
                                    &request_capacity) ||
        !iree_host_size_checked_add(fixed_buffer_ref_capacity,
                                    program_infos[i].fixed_buffer_count,
                                    &fixed_buffer_ref_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Qwen parameter metadata count overflow");
    }
  }

  iree_host_size_t program_offset_count = 0;
  if (!iree_host_size_checked_add(program_count, 1, &program_offset_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen program count overflow");
  }
  iree_host_size_t storage_size = 0;
  iree_host_size_t requests_offset = 0;
  iree_host_size_t fixed_buffer_refs_offset = 0;
  iree_host_size_t program_offsets_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &storage_size,
      IREE_STRUCT_FIELD_ALIGNED(
          request_capacity, qwen_tooling_parameter_request_t,
          iree_alignof(qwen_tooling_parameter_request_t), &requests_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          fixed_buffer_ref_capacity, iree_hal_buffer_ref_t,
          iree_alignof(iree_hal_buffer_ref_t), &fixed_buffer_refs_offset),
      IREE_STRUCT_FIELD_ALIGNED(program_offset_count, iree_host_size_t,
                                iree_alignof(iree_host_size_t),
                                &program_offsets_offset)));
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, storage_size, &out_pack->storage));
  memset(out_pack->storage, 0, storage_size);
  out_pack->requests =
      (qwen_tooling_parameter_request_t*)((uint8_t*)out_pack->storage +
                                          requests_offset);
  out_pack->fixed_buffer_refs =
      (iree_hal_buffer_ref_t*)((uint8_t*)out_pack->storage +
                               fixed_buffer_refs_offset);
  out_pack->program_fixed_buffer_offsets =
      (iree_host_size_t*)((uint8_t*)out_pack->storage + program_offsets_offset);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t program_i = 0;
       program_i < program_count && iree_status_is_ok(status); ++program_i) {
    const loomc_cmd_program_info_t* program_info = &program_infos[program_i];
    out_pack->program_fixed_buffer_offsets[program_i] =
        out_pack->fixed_buffer_ref_count;
    out_pack->fixed_buffer_ref_count += program_info->fixed_buffer_count;

    for (iree_host_size_t root_i = 0;
         root_i < program_info->parameter_root_count &&
         iree_status_is_ok(status);
         ++root_i) {
      loomc_cmd_program_parameter_root_info_t root_info = {
          .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO,
          .structure_size = sizeof(root_info),
      };
      status = qwen_tooling_status_from_loomc(
          loomc_cmd_program_package_parameter_root_info(
              package, program_exports[program_i], root_i, &root_info));
      if (!iree_status_is_ok(status)) break;

      bool root_base_known = false;
      iree_device_size_t root_base = 0;
      for (iree_host_size_t parameter_i = 0;
           parameter_i < program_info->parameter_count &&
           iree_status_is_ok(status);
           ++parameter_i) {
        loomc_cmd_program_parameter_info_t parameter_info = {
            .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_INFO,
            .structure_size = sizeof(parameter_info),
        };
        status = qwen_tooling_status_from_loomc(
            loomc_cmd_program_package_parameter_info(
                package, program_exports[program_i], parameter_i,
                &parameter_info));
        if (!iree_status_is_ok(status)) break;
        if (parameter_info.fixed_buffer_index != root_info.fixed_buffer_index) {
          continue;
        }
        const iree_host_size_t existing_i = qwen_tooling_parameter_pack_find(
            out_pack, iree_string_view_from_loomc(parameter_info.key));
        if (existing_i == IREE_HOST_SIZE_MAX) continue;
        const qwen_tooling_parameter_request_t* existing =
            &out_pack->requests[existing_i];
        if (existing->span.length != parameter_info.byte_length ||
            existing->span.buffer_offset < parameter_info.byte_offset) {
          status = iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "parameter '%.*s' in command root '%.*s' has length %" PRIu64
              " at root offset %" PRIu64
              "; the packed occurrence has length %" PRIu64
              " at slab offset %" PRIu64,
              (int)parameter_info.key.size, parameter_info.key.data,
              (int)program_info->name.size, program_info->name.data,
              parameter_info.byte_length, parameter_info.byte_offset,
              existing->span.length, existing->span.buffer_offset);
          break;
        }
        const iree_device_size_t candidate_base =
            existing->span.buffer_offset - parameter_info.byte_offset;
        if (root_base_known && candidate_base != root_base) {
          status = iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "command root '%.*s' cannot map to one parameter subrange",
              (int)program_info->name.size, program_info->name.data);
          break;
        }
        root_base = candidate_base;
        root_base_known = true;
      }
      if (!iree_status_is_ok(status)) break;

      if (!root_base_known &&
          !iree_device_size_checked_align(
              out_pack->byte_length, root_info.minimum_alignment, &root_base)) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "Qwen parameter-pack alignment overflow");
        break;
      }
      if (root_base % root_info.minimum_alignment != 0) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "command root '%.*s' requires incompatible parameter alignment",
            (int)program_info->name.size, program_info->name.data);
        break;
      }

      for (iree_host_size_t parameter_i = 0;
           parameter_i < program_info->parameter_count &&
           iree_status_is_ok(status);
           ++parameter_i) {
        loomc_cmd_program_parameter_info_t parameter_info = {
            .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_INFO,
            .structure_size = sizeof(parameter_info),
        };
        status = qwen_tooling_status_from_loomc(
            loomc_cmd_program_package_parameter_info(
                package, program_exports[program_i], parameter_i,
                &parameter_info));
        if (!iree_status_is_ok(status)) break;
        if (parameter_info.fixed_buffer_index != root_info.fixed_buffer_index) {
          continue;
        }

        iree_device_size_t parameter_offset = 0;
        if (!iree_device_size_checked_add(root_base, parameter_info.byte_offset,
                                          &parameter_offset)) {
          status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                    "Qwen parameter offset overflow");
          break;
        }
        const iree_string_view_t key =
            iree_string_view_from_loomc(parameter_info.key);
        const iree_host_size_t existing_i =
            qwen_tooling_parameter_pack_find(out_pack, key);
        if (existing_i != IREE_HOST_SIZE_MAX) {
          const qwen_tooling_parameter_request_t* existing =
              &out_pack->requests[existing_i];
          if (existing->span.buffer_offset != parameter_offset ||
              existing->span.length != parameter_info.byte_length ||
              parameter_offset % parameter_info.minimum_alignment != 0) {
            status = iree_make_status(
                IREE_STATUS_FAILED_PRECONDITION,
                "parameter '%.*s' has incompatible command-root placement",
                (int)key.size, key.data);
          }
          continue;
        }

        for (iree_host_size_t existing_i = 0;
             existing_i < out_pack->request_count; ++existing_i) {
          const qwen_tooling_parameter_request_t* existing =
              &out_pack->requests[existing_i];
          if (qwen_tooling_parameter_ranges_overlap(
                  parameter_offset, parameter_info.byte_length,
                  existing->span.buffer_offset, existing->span.length)) {
            status = iree_make_status(
                IREE_STATUS_FAILED_PRECONDITION,
                "parameter '%.*s' in command root '%.*s' requires slab range "
                "{offset=%" PRIu64 ", length=%" PRIu64 "}, overlapping "
                "parameter '%.*s' range {offset=%" PRIu64 ", length=%" PRIu64
                "}",
                (int)key.size, key.data, (int)program_info->name.size,
                program_info->name.data, parameter_offset,
                parameter_info.byte_length,
                (int)existing->key.size, existing->key.data,
                existing->span.buffer_offset, existing->span.length);
            break;
          }
        }
        if (!iree_status_is_ok(status)) break;
        out_pack->requests[out_pack->request_count++] =
            (qwen_tooling_parameter_request_t){
                .key = key,
                .span =
                    {
                        .parameter_offset = 0,
                        .buffer_offset = parameter_offset,
                        .length = parameter_info.byte_length,
                    },
            };
      }

      iree_device_size_t root_end = 0;
      if (iree_status_is_ok(status) &&
          !iree_device_size_checked_add(
              root_base, root_info.required_byte_length, &root_end)) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "Qwen parameter-root range overflow");
      }
      if (!iree_status_is_ok(status)) break;
      if (root_end > out_pack->byte_length) out_pack->byte_length = root_end;
      if (root_info.minimum_alignment > out_pack->minimum_alignment) {
        out_pack->minimum_alignment = root_info.minimum_alignment;
      }
      const iree_host_size_t fixed_buffer_ref_i =
          out_pack->program_fixed_buffer_offsets[program_i] +
          root_info.fixed_buffer_index;
      out_pack->fixed_buffer_refs[fixed_buffer_ref_i] =
          iree_hal_make_buffer_ref(/*buffer=*/NULL, root_base,
                                   root_info.required_byte_length);
    }
  }
  out_pack->program_fixed_buffer_offsets[program_count] =
      out_pack->fixed_buffer_ref_count;
  if (!iree_status_is_ok(status)) {
    qwen_tooling_parameter_pack_deinitialize(host_allocator, out_pack);
  }
  return status;
}

static iree_status_t qwen_tooling_parameter_pack_begin_gather(
    qwen_tooling_parameter_pack_t* pack,
    iree_io_parameter_provider_t* parameter_provider,
    iree_hal_device_t* device) {
  iree_status_t status = qwen_tooling_allocate_buffer(
      device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      pack->minimum_alignment, pack->byte_length, &pack->buffer);
  for (iree_host_size_t i = 0;
       i < pack->fixed_buffer_ref_count && iree_status_is_ok(status); ++i) {
    pack->fixed_buffer_refs[i].buffer = pack->buffer;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &pack->ready_semaphore);
  }
  uint64_t ready_value = 1;
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_t* ready_semaphore = pack->ready_semaphore;
    const iree_hal_semaphore_list_t signals = {
        .count = 1,
        .semaphores = &ready_semaphore,
        .payload_values = &ready_value,
    };
    status = iree_io_parameter_provider_gather(
        parameter_provider, device, IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), signals, iree_string_view_empty(),
        pack->buffer, pack->request_count,
        (iree_io_parameter_enumerator_t){
            .fn = qwen_tooling_parameter_enumerate,
            .user_data = pack->requests,
        });
  }
  return status;
}

iree_status_t qwen_tooling_command_program_set_create(
    qwen_tooling_runtime_context_t* runtime_context,
    const qwen_tooling_command_program_set_options_t* options,
    iree_allocator_t host_allocator,
    qwen_tooling_command_program_set_t** out_program_set) {
  if (!runtime_context || !options || !out_program_set ||
      iree_string_view_is_empty(options->source_identifier) ||
      iree_const_byte_span_is_empty(options->source_contents) ||
      options->root_count == 0 || !options->root_names) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program options must be complete");
  }
  for (iree_host_size_t i = 0; i < options->root_count; ++i) {
    if (iree_string_view_is_empty(options->root_names[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command program root names must be nonempty");
    }
  }
  *out_program_set = NULL;
  const loomc_allocator_t loom_allocator =
      loomc_allocator_from_iree(host_allocator);
  iree_hal_device_t* device =
      qwen_tooling_runtime_context_device(runtime_context);

  loomc_target_environment_t* target_environment = NULL;
  loomc_context_t* context = NULL;
  loomc_workspace_t* workspace = NULL;
  loomc_source_t* source = NULL;
  loomc_module_t* module = NULL;
  loomc_target_profile_t* target_profile = NULL;
  loomc_compiler_t* compiler = NULL;
  loomc_pass_program_t* empty_pass_program = NULL;
  loomc_pass_program_t* executable_pass_program = NULL;
  loomc_module_function_t* kernel_functions = NULL;
  loomc_target_specialization_t* specializations = NULL;
  iree_host_size_t kernel_function_count = 0;
  loomc_string_view_t* root_names = NULL;
  loomc_program_plan_t* plan = NULL;
  loomc_cmd_program_plan_root_info_t* root_plan_infos = NULL;
  loomc_cmd_program_package_t* package = NULL;
  loomc_cmd_program_export_t* program_exports = NULL;
  loomc_cmd_program_info_t* program_infos = NULL;
  loomc_launch_config_program_t* launch_config_program = NULL;
  loomc_result_t* result = NULL;
  qwen_tooling_parameter_pack_t parameter_pack = {0};
  bool parameter_gather_submitted = false;
  iree_hal_executable_t** executables_by_unit = NULL;
  iree_host_size_t unit_count = 0;
  iree_hal_executable_t** root_executables = NULL;
  iree_host_size_t root_executable_capacity = 0;
  qwen_tooling_command_program_set_t* program_set = NULL;

  iree_status_t status =
      qwen_tooling_status_from_loomc(loomc_target_environment_create_amdgpu(
          loom_allocator, &target_environment));
  if (iree_status_is_ok(status)) {
    const loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .target_environment = target_environment,
    };
    const loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next = &target_options,
    };
    status = qwen_tooling_status_from_loomc(
        loomc_context_create(&context_options, loom_allocator, &context));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_status_from_loomc(loomc_workspace_create(
        /*options=*/NULL, loom_allocator, &workspace));
  }
  if (iree_status_is_ok(status)) {
    const loomc_source_options_t source_options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        .structure_size = sizeof(source_options),
        .format = LOOMC_SOURCE_FORMAT_TEXT,
        .identifier = loomc_string_view_from_iree(options->source_identifier),
        .contents = loomc_make_byte_span(options->source_contents.data,
                                         options->source_contents.data_length),
        .storage = LOOMC_SOURCE_STORAGE_BORROWED,
    };
    status = qwen_tooling_status_from_loomc(
        loomc_source_create(&source_options, loom_allocator, &source));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_tooling_status_from_loomc(loomc_module_deserialize_from_source(
            context, workspace, source,
            /*options=*/NULL, loom_allocator, &module, &result));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_tooling_require_result(IREE_SV("command source parse"), result);
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status)) {
    const loomc_amdgpu_iree_hal_profile_options_t profile_options = {
        .type = LOOMC_STRUCTURE_TYPE_AMDGPU_IREE_HAL_PROFILE_OPTIONS,
        .structure_size = sizeof(profile_options),
        .identifier = loomc_make_cstring_view("qwen38-live-amdgpu"),
        .device = device,
        .physical_device_affinity = 0,
    };
    status = qwen_tooling_status_from_loomc(
        loomc_target_profile_create_amdgpu_iree_hal(
            target_environment, &profile_options, loom_allocator,
            &target_profile, &result));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_tooling_require_result(IREE_SV("AMDGPU target profile"), result);
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status)) {
    status = qwen_tooling_query_kernels(
        module, host_allocator, &kernel_functions, &kernel_function_count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, kernel_function_count,
                                         sizeof(*specializations),
                                         (void**)&specializations);
  }
  for (iree_host_size_t i = 0;
       i < kernel_function_count && iree_status_is_ok(status); ++i) {
    specializations[i] = (loomc_target_specialization_t){
        .function_symbol = kernel_functions[i].symbol_name,
        .target_profile = target_profile,
    };
  }

  if (iree_status_is_ok(status)) {
    status = qwen_tooling_status_from_loomc(loomc_compiler_create(
        context, /*options=*/NULL, loom_allocator, &compiler));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_status_from_loomc(loomc_pass_program_create_empty(
        context, /*options=*/NULL, loom_allocator, &empty_pass_program));
  }
  if (iree_status_is_ok(status)) {
    const loomc_target_pipeline_options_t pipeline_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
        .structure_size = sizeof(pipeline_options),
        .identifier = loomc_make_cstring_view("qwen38-prepared-low"),
        .kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
        .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
        .source_to_low_max_errors = 20,
    };
    status = qwen_tooling_status_from_loomc(
        loomc_pass_program_create_from_target_pipeline(
            context, &pipeline_options, loom_allocator,
            &executable_pass_program, &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_require_result(IREE_SV("AMDGPU pipeline"), result);
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(host_allocator, options->root_count,
                                    sizeof(*root_names), (void**)&root_names);
  }
  for (iree_host_size_t i = 0;
       i < options->root_count && iree_status_is_ok(status); ++i) {
    root_names[i] = loomc_string_view_from_iree(options->root_names[i]);
  }

  if (iree_status_is_ok(status)) {
    const loomc_target_specialization_options_t specialization_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        .structure_size = sizeof(specialization_options),
        .specializations = specializations,
        .specialization_count = kernel_function_count,
    };
    const loomc_compile_options_t compile_options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        .structure_size = sizeof(compile_options),
        .next = &specialization_options,
        .module_name = root_names[0],
        .config = options->config,
    };
    status = qwen_tooling_status_from_loomc(
        loomc_compile_module(compiler, workspace, empty_pass_program, module,
                             &compile_options, loom_allocator, &result));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_tooling_require_result(IREE_SV("target specialization"), result);
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status)) {
    status = qwen_tooling_status_from_loomc(loomc_program_plan_prepare(
        workspace, module, root_names, options->root_count, /*options=*/NULL,
        loom_allocator, &plan, &result));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_tooling_require_result(IREE_SV("command program plan"), result);
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, options->root_count,
                                         sizeof(*root_plan_infos),
                                         (void**)&root_plan_infos);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, options->root_count,
                                         sizeof(*program_exports),
                                         (void**)&program_exports);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, options->root_count,
                                         sizeof(*program_infos),
                                         (void**)&program_infos);
  }
  for (iree_host_size_t i = 0;
       i < options->root_count && iree_status_is_ok(status); ++i) {
    const loomc_program_plan_root_t root = loomc_program_plan_root_at(plan, i);
    status = qwen_tooling_status_from_loomc(
        loomc_cmd_program_plan_root_info(plan, root, &root_plan_infos[i]));
    if (iree_status_is_ok(status) &&
        root_plan_infos[i].executable_requirement_count >
            root_executable_capacity) {
      root_executable_capacity =
          root_plan_infos[i].executable_requirement_count;
    }
  }

  loomc_program_plan_unit_t launch_config_unit =
      loomc_program_plan_unit_invalid();
  for (iree_host_size_t i = 0; i < options->root_count; ++i) {
    if (root_plan_infos && loomc_program_plan_unit_is_valid(
                               root_plan_infos[i].launch_config_unit)) {
      launch_config_unit = root_plan_infos[i].launch_config_unit;
      break;
    }
  }
  if (iree_status_is_ok(status) &&
      loomc_program_plan_unit_is_valid(launch_config_unit)) {
    status = qwen_tooling_compile_unit(
        plan, compiler, workspace, launch_config_unit, empty_pass_program,
        IREE_SV("command launch config compile"), host_allocator, &result);
  }
  if (iree_status_is_ok(status) &&
      loomc_program_plan_unit_is_valid(launch_config_unit)) {
    const loomc_artifact_t* artifact = qwen_tooling_find_artifact(
        result, LOOMC_ARTIFACT_KIND_COMMAND_LAUNCH_CONFIG,
        LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE);
    if (!artifact) {
      status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                "command launch config artifact is absent");
    } else {
      status = qwen_tooling_status_from_loomc(loomc_launch_config_program_load(
          artifact, /*release=*/NULL, /*release_user_data=*/NULL,
          loom_allocator, &launch_config_program));
    }
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status)) {
    status = qwen_tooling_compile_unit(
        plan, compiler, workspace, root_plan_infos[0].package_unit,
        empty_pass_program, IREE_SV("command package compile"), host_allocator,
        &result);
  }
  if (iree_status_is_ok(status)) {
    const loomc_artifact_t* artifact =
        qwen_tooling_find_artifact(result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
                                   LOOMC_ARTIFACT_FORMAT_COMMAND_PACKAGE);
    if (!artifact) {
      status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                "command package artifact is absent");
    } else {
      status = qwen_tooling_status_from_loomc(loomc_cmd_program_package_load(
          artifact, /*release=*/NULL, /*release_user_data=*/NULL,
          loom_allocator, &package));
    }
  }
  loomc_result_release(result);
  result = NULL;

  for (iree_host_size_t i = 0;
       i < options->root_count && iree_status_is_ok(status); ++i) {
    status =
        qwen_tooling_status_from_loomc(loomc_cmd_program_package_lookup_export(
            package, root_names[i], &program_exports[i]));
    if (iree_status_is_ok(status)) {
      program_infos[i] = (loomc_cmd_program_info_t){
          .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_INFO,
          .structure_size = sizeof(program_infos[i]),
      };
      status =
          qwen_tooling_status_from_loomc(loomc_cmd_program_package_export_info(
              package, program_exports[i], &program_infos[i]));
    }
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_parameter_pack_build(
        package, program_exports, program_infos, options->root_count,
        host_allocator, &parameter_pack);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_parameter_pack_begin_gather(
        &parameter_pack, runtime_context->parameter_provider, device);
    parameter_gather_submitted = iree_status_is_ok(status);
  }

  unit_count = loomc_program_plan_unit_count(plan);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, unit_count,
                                         sizeof(*executables_by_unit),
                                         (void**)&executables_by_unit);
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < unit_count; ++i) {
        executables_by_unit[i] = NULL;
      }
    }
  }
  for (iree_host_size_t i = 0;
       i < options->root_count && iree_status_is_ok(status); ++i) {
    const loomc_cmd_program_plan_root_info_t* root_info = &root_plan_infos[i];
    for (iree_host_size_t j = 0; j < root_info->executable_requirement_count &&
                                 iree_status_is_ok(status);
         ++j) {
      const loomc_program_plan_unit_t unit =
          root_info->executable_requirements[j].unit;
      if (!loomc_program_plan_unit_is_valid(unit)) {
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "Qwen command programs do not yet accept external executables");
        break;
      }
      const iree_host_size_t unit_index = (iree_host_size_t)unit.value;
      if (executables_by_unit[unit_index]) continue;
      status = qwen_tooling_compile_unit(
          plan, compiler, workspace, unit, executable_pass_program,
          IREE_SV("AMDGPU executable compile"), host_allocator, &result);
      if (iree_status_is_ok(status)) {
        const loomc_artifact_t* artifact =
            qwen_tooling_find_artifact(result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
                                       LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
        if (!artifact) {
          status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                    "AMDGPU executable artifact is absent");
        } else {
          status = qwen_tooling_load_executable(
              device, artifact, &executables_by_unit[unit_index]);
        }
      }
      loomc_result_release(result);
      result = NULL;
    }
  }

  if (iree_status_is_ok(status) && root_executable_capacity != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, root_executable_capacity, sizeof(*root_executables),
        (void**)&root_executables);
  }

  iree_host_size_t program_set_size = 0;
  iree_host_size_t programs_offset = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        iree_sizeof_struct(*program_set), &program_set_size,
        IREE_STRUCT_FIELD_ALIGNED(
            options->root_count, qwen_tooling_command_program_t,
            iree_alignof(qwen_tooling_command_program_t), &programs_offset));
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, program_set_size,
                                   (void**)&program_set);
    if (iree_status_is_ok(status)) {
      memset(program_set, 0, program_set_size);
    }
  }
  if (iree_status_is_ok(status)) {
    program_set->host_allocator = host_allocator;
    program_set->programs =
        (qwen_tooling_command_program_t*)((uint8_t*)program_set +
                                          programs_offset);
    program_set->program_count = options->root_count;
    for (iree_host_size_t i = 0;
         i < options->root_count && iree_status_is_ok(status); ++i) {
      qwen_tooling_command_program_t* program = &program_set->programs[i];
      program->launch_config_program = launch_config_program;
      program->launch_config_function = loomc_launch_config_function_invalid();
      const loomc_cmd_program_plan_root_info_t* root_info = &root_plan_infos[i];
      for (iree_host_size_t j = 0; j < root_info->executable_requirement_count;
           ++j) {
        const iree_host_size_t unit_index =
            (iree_host_size_t)root_info->executable_requirements[j].unit.value;
        root_executables[j] = executables_by_unit[unit_index];
      }
      const loomc_cmd_hal_program_options_t materialization_options = {
          .type = LOOMC_STRUCTURE_TYPE_CMD_HAL_PROGRAM_OPTIONS,
          .structure_size = sizeof(materialization_options),
          .command_buffer_mode = runtime_context->command_buffer_mode,
          .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
          .fixed_buffers = parameter_pack.fixed_buffer_refs +
                           parameter_pack.program_fixed_buffer_offsets[i],
          .fixed_buffer_count = program_infos[i].fixed_buffer_count,
          .executables = root_executables,
          .executable_count = root_info->executable_requirement_count,
      };
      status = qwen_tooling_status_from_loomc(loomc_cmd_hal_program_create(
          package, program_exports[i], device, &materialization_options,
          loom_allocator, &program->hal_program));
      if (iree_status_is_ok(status) &&
          loomc_program_plan_unit_is_valid(root_info->launch_config_unit)) {
        status = qwen_tooling_status_from_loomc(
            loomc_launch_config_program_lookup_function(
                launch_config_program, root_names[i],
                &program->launch_config_function));
      }
      if (iree_status_is_ok(status)) program->info = program_infos[i];
    }
  }

  if (parameter_gather_submitted) {
    const iree_status_t gather_status = iree_hal_semaphore_wait(
        parameter_pack.ready_semaphore, 1, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE);
    status = iree_status_join(status, gather_status);
  }
  if (iree_status_is_ok(status)) {
    program_set->package = package;
    package = NULL;
    program_set->launch_config_program = launch_config_program;
    launch_config_program = NULL;
    program_set->parameter_byte_length = parameter_pack.byte_length;
    *out_program_set = program_set;
    program_set = NULL;
  }

  qwen_tooling_command_program_set_release(program_set);
  for (iree_host_size_t i = 0; i < unit_count; ++i) {
    iree_hal_executable_release(executables_by_unit ? executables_by_unit[i]
                                                    : NULL);
  }
  iree_allocator_free(host_allocator, root_executables);
  iree_allocator_free(host_allocator, executables_by_unit);
  qwen_tooling_parameter_pack_deinitialize(host_allocator, &parameter_pack);
  loomc_result_release(result);
  loomc_launch_config_program_release(launch_config_program);
  loomc_cmd_program_package_release(package);
  iree_allocator_free(host_allocator, program_infos);
  iree_allocator_free(host_allocator, program_exports);
  iree_allocator_free(host_allocator, root_plan_infos);
  loomc_program_plan_release(plan);
  iree_allocator_free(host_allocator, root_names);
  iree_allocator_free(host_allocator, specializations);
  iree_allocator_free(host_allocator, kernel_functions);
  loomc_pass_program_release(executable_pass_program);
  loomc_pass_program_release(empty_pass_program);
  loomc_compiler_release(compiler);
  loomc_target_profile_release(target_profile);
  loomc_module_release(module);
  loomc_source_release(source);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(target_environment);
  return status;
}

void qwen_tooling_command_program_set_release(
    qwen_tooling_command_program_set_t* program_set) {
  if (!program_set) return;
  const iree_allocator_t host_allocator = program_set->host_allocator;
  for (iree_host_size_t i = 0; i < program_set->program_count; ++i) {
    loomc_cmd_hal_program_release(program_set->programs[i].hal_program);
  }
  loomc_launch_config_program_release(program_set->launch_config_program);
  loomc_cmd_program_package_release(program_set->package);
  iree_allocator_free(host_allocator, program_set);
}

qwen_tooling_command_program_t* qwen_tooling_command_program_set_lookup(
    const qwen_tooling_command_program_set_t* program_set,
    iree_string_view_t root_name) {
  if (!program_set) return NULL;
  for (iree_host_size_t i = 0; i < program_set->program_count; ++i) {
    if (iree_string_view_equal(
            iree_string_view_from_loomc(program_set->programs[i].info.name),
            root_name)) {
      return &program_set->programs[i];
    }
  }
  return NULL;
}

const loomc_cmd_program_info_t* qwen_tooling_command_program_info(
    const qwen_tooling_command_program_t* program) {
  return program ? &program->info : NULL;
}

iree_hal_command_buffer_t* qwen_tooling_command_program_command_buffer(
    const qwen_tooling_command_program_t* program) {
  return program ? loomc_cmd_hal_program_command_buffer(program->hal_program)
                 : NULL;
}

iree_status_t qwen_tooling_command_program_populate_config(
    qwen_tooling_command_program_t* program, const uint64_t* argument_bits,
    iree_host_size_t argument_count, iree_byte_span_t config_data) {
  if (!program || !program->launch_config_program ||
      !loomc_launch_config_function_is_valid(program->launch_config_function)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command program has no dynamic launch config");
  }
  if (config_data.data_length != program->info.config.required_byte_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "config storage has byte length %" PRIhsz ", expected %" PRIu64,
        config_data.data_length,
        (uint64_t)program->info.config.required_byte_length);
  }
  loomc_cmd_launch_config_t launch_config = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_LAUNCH_CONFIG,
      .structure_size = sizeof(launch_config),
      .data = loomc_make_mutable_byte_span(config_data.data,
                                           config_data.data_length),
  };
  return qwen_tooling_status_from_loomc(loomc_launch_config_program_invoke_cmd(
      program->launch_config_program, program->launch_config_function,
      argument_bits, argument_count, &launch_config));
}

iree_device_size_t qwen_tooling_command_program_set_parameter_byte_length(
    const qwen_tooling_command_program_set_t* program_set) {
  return program_set ? program_set->parameter_byte_length : 0;
}
