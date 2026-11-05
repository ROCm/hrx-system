// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/stage_set.h"
#include "loom/serve/service/text/model_instance.h"

struct loom_serve_text_variant_t {
  // Allocator for the candidate and its stage records.
  iree_allocator_t host_allocator;
  // Borrowed model identity; candidates never cross parameter/state owners.
  loom_serve_text_model_t* owner;
  // Copied diagnostics and compilation counts, including failed candidates.
  loom_serve_compilation_report_t compilation;
  // Number of allocated slots, including partial compilation on failure.
  iree_host_size_t stage_count;
  // Owned compiled stages, commands and retained original parameter roots.
  loom_serve_text_stage_t* stages;
  // Owned descriptors borrowing this candidate's commands.
  loom_serve_stage_t* bindings;
  // Private peaks per actual pool, retained with the recorded bindings.
  uint64_t* workspace_bytes;
  // True only after every stage compiled, validated and recorded successfully.
  bool ready;
};

typedef struct loom_serve_text_variant_job_t {
  // Private candidate transferred on every terminal result.
  loom_serve_text_variant_t* variant;
  // Terminal ownership transfer, invoked after all compiler work retires.
  loom_serve_variant_completion_fn_t completion;
  // Borrowed terminal notification payload.
  void* user_data;
} loom_serve_text_variant_job_t;

void loom_serve_text_variant_free(loom_serve_text_variant_t* variant) {
  if (!variant) {
    return;
  }
  for (iree_host_size_t i = 0; i < variant->stage_count; ++i) {
    loom_serve_text_stage_t* stage = &variant->stages[i];
    iree_hal_command_buffer_release(stage->command);
    for (uint32_t j = 0; j < stage->fixed_buffer_count; ++j) {
      iree_hal_buffer_release(stage->fixed_buffers[j]);
    }
    iree_allocator_free(variant->host_allocator, stage->fixed_buffers);
    loom_serve_jit_stage_free(stage->compiled);
  }
  iree_allocator_free(variant->host_allocator, variant);
}

const loom_serve_compilation_report_t* loom_serve_text_variant_report(
    const loom_serve_text_variant_t* variant) {
  return &variant->compilation;
}

iree_status_t loom_serve_text_model_select_variant(
    loom_serve_text_model_t* model, const loom_serve_text_variant_t* variant) {
  if (variant && (variant->owner != model || !variant->ready)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "variant is not ready for this model");
  }
  const uint64_t* workspace_bytes =
      variant ? variant->workspace_bytes : model->residency.workspace_bytes;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < model->residency.count && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_residency_check_private_peak(model->residency.values[i],
                                                     workspace_bytes[i]);
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_serve_module_bind_stages(
      model->control.native_module, model->stage_count,
      variant ? variant->bindings : model->stage_bindings));
  for (uint32_t i = 0; i < model->residency.count; ++i) {
    loom_serve_residency_set_private_peak(model->residency.values[i],
                                          workspace_bytes[i]);
  }
  return iree_ok_status();
}

static void loom_serve_text_variant_recorded(void* user_data,
                                             iree_status_t status) {
  loom_serve_text_variant_job_t* job = user_data;
  loom_serve_text_variant_t* variant = job->variant;
  const loom_serve_text_model_t* model = variant->owner;
  uint64_t execution_bytes[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT] = {0};
  for (iree_host_size_t i = 0;
       i < variant->stage_count && iree_status_is_ok(status); ++i) {
    variant->bindings[i] = loom_serve_text_stage_binding(
        &variant->stages[i],
        loom_serve_preparation_stage(model->source.preparation, i)
            ->execution_index);
    const uint32_t execution = variant->bindings[i].execution_index;
    execution_bytes[execution] =
        iree_max(execution_bytes[execution],
                 variant->bindings[i].transient.required_byte_length);
  }
  for (uint32_t i = 0; i < model->device_count && iree_status_is_ok(status);
       ++i) {
    uint64_t* bytes =
        &variant->workspace_bytes[model->residency.pool_indices[i]];
    if (!iree_checked_add_u64(*bytes, execution_bytes[i], bytes)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "combined command workspace exceeds byte accounting");
    }
  }
  variant->ready = iree_status_is_ok(status);
  const loom_serve_variant_completion_fn_t completion = job->completion;
  void* completion_user_data = job->user_data;
  iree_allocator_free(variant->host_allocator, job);
  completion(completion_user_data, status, variant, &variant->compilation);
}

static void loom_serve_text_variant_compiled(void* user_data,
                                             iree_status_t status,
                                             loom_serve_stage_set_t* stages) {
  loom_serve_text_variant_job_t* job = user_data;
  const loom_serve_text_model_t* model = job->variant->owner;
  job->variant->compilation = stages->compilation;
  for (iree_host_size_t i = 0; i < stages->count; ++i) {
    job->variant->stages[i].compiled = stages->values[i];
    if (iree_status_is_ok(status)) {
      job->variant->stages[i].program =
          *loom_serve_jit_stage_program(stages->values[i]);
    }
    stages->values[i] = NULL;
  }
  loom_serve_stage_set_free(stages);
  loom_serve_text_variant_t* variant = job->variant;
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_stages_rebind(
        model->source.preparation, model->source.shared_stage_count,
        variant->stage_count, model->stages, variant->stages,
        variant->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_host_t* host = loom_serve_device_host(model->devices[0]);
    status = loom_serve_text_stages_record_async(
        loom_serve_host_preparation_sink(host),
        loomc_task_pool_worker_count(loom_serve_host_task_pool(host)),
        model->command_mode, variant->stage_count, variant->stages,
        loom_serve_text_variant_recorded, job, variant->host_allocator);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_text_variant_recorded(job, status);
  }
}

iree_status_t loom_serve_text_model_compile_variant_async(
    loom_serve_text_model_t* model, const loom_serve_sources_t* sources,
    const loomc_sanitizer_options_t* sanitizer,
    loom_serve_variant_completion_fn_t completion, void* user_data) {
  const iree_allocator_t host_allocator = model->host_allocator;
  iree_host_size_t total = 0, stages = 0, bindings = 0, workspace = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_text_variant_t), &total,
      IREE_STRUCT_FIELD(model->stage_count, loom_serve_text_stage_t, &stages),
      IREE_STRUCT_FIELD(model->stage_count, loom_serve_stage_t, &bindings),
      IREE_STRUCT_FIELD_ALIGNED(model->residency.count, uint64_t,
                                iree_alignof(uint64_t), &workspace)));
  loom_serve_text_variant_t* variant = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total, (void**)&variant));
  variant->host_allocator = host_allocator;
  variant->owner = model;
  variant->stage_count = model->stage_count;
  variant->stages = (loom_serve_text_stage_t*)((uint8_t*)variant + stages);
  variant->bindings = (loom_serve_stage_t*)((uint8_t*)variant + bindings);
  variant->workspace_bytes = (uint64_t*)((uint8_t*)variant + workspace);
  loom_serve_text_variant_job_t* job = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*job), (void**)&job);
  if (iree_status_is_ok(status)) {
    job->variant = variant;
    job->completion = completion;
    job->user_data = user_data;
    status = loom_serve_stage_set_compile_async(
        model->devices, model->source.preparation, sources, sanitizer,
        loom_serve_text_variant_compiled, job, host_allocator);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, job);
    loom_serve_text_variant_free(variant);
  }
  return status;
}
