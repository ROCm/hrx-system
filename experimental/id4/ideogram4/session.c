// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/session.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "experimental/id4/ideogram4/session_generation.h"
#include "experimental/id4/ideogram4/session_state.h"
#include "experimental/id4/ideogram4/session_support.h"
#include "experimental/id4/pipeline/binding.h"
#include "experimental/id4/stages/ideogram4_decode.h"
#include "experimental/id4/stages/ideogram4_dit.h"
#include "experimental/id4/stages/qwen3_vl.h"
#include "experimental/id4/stages/sampler.h"

static iree_status_t id4_ideogram4_validate_generation_prepare_options(
    const id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_prepare_options_t* options) {
  if (!session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session is required");
  }
  if (!session->is_loaded) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Ideogram 4 session must be loaded before generation preparation");
  }
  if (!plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation plan is required");
  }
  if (plan->session != session) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation plan was created by a different session");
  }
  if (!plan->qwen_plan || !plan->dit_conditioned_plan ||
      !plan->dit_unconditioned_plan || !plan->sampler_noise_plan ||
      !plan->sampler_denoise_plan || !plan->decode_plan) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation plan must contain every coarse stage plan");
  }
  if (!options) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation prepare options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation prepare")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Ideogram 4 generation prepare extension "
                            "structures are not supported");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_prepare_residency_options(
          options->residency_mode, options->resident_stage_mask));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("Ideogram 4 generation prepare "
                                            "wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list,
      IREE_SV("Ideogram 4 generation prepare signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation prepare final signal is required");
  }
  if (!options->parameter_providers.qwen) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation Qwen parameter provider is required");
  }
  if (!options->parameter_providers.dit_conditioned) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation conditioned DiT parameter provider is required");
  }
  if (!options->parameter_providers.dit_unconditioned) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation unconditioned DiT "
                            "parameter provider is required");
  }
  if (!options->parameter_providers.vae) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation VAE parameter provider is required");
  }
  if (!options->kernel_library) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation prepare kernel library is required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 generation prepare")));
  return iree_ok_status();
}

static void id4_ideogram4_generation_stage_slot_deinitialize(
    id4_ideogram4_generation_stage_slot_t* slot) {
  id4_pipeline_buffer_binding_set_deinitialize(&slot->diagnostic_tap_bindings);
  id4_pipeline_buffer_binding_set_deinitialize(&slot->boundary_bindings);
  id4_pipeline_plan_release(slot->plan);
  memset(slot, 0, sizeof(*slot));
}

static void id4_ideogram4_generation_parameter_providers_retain(
    const id4_ideogram4_generation_parameter_providers_t* providers) {
  iree_io_parameter_provider_retain(providers->qwen);
  iree_io_parameter_provider_retain(providers->dit_conditioned);
  iree_io_parameter_provider_retain(providers->dit_unconditioned);
  iree_io_parameter_provider_retain(providers->vae);
}

static void id4_ideogram4_generation_parameter_providers_release(
    id4_ideogram4_generation_parameter_providers_t* providers) {
  iree_io_parameter_provider_release(providers->vae);
  iree_io_parameter_provider_release(providers->dit_unconditioned);
  iree_io_parameter_provider_release(providers->dit_conditioned);
  iree_io_parameter_provider_release(providers->qwen);
  memset(providers, 0, sizeof(*providers));
}

static void id4_ideogram4_generation_bundle_capture_prepare_resources(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_prepare_options_t* options) {
  bundle->parameter_providers = options->parameter_providers;
  id4_ideogram4_generation_parameter_providers_retain(
      &bundle->parameter_providers);
  bundle->kernel_library = options->kernel_library;
  id4_pipeline_kernel_library_retain(bundle->kernel_library);
  bundle->command_buffer_mode = options->command_buffer_mode;
  bundle->residency_mode = options->residency_mode;
  switch (options->residency_mode) {
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES:
      bundle->resident_stage_mask = ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_ALL;
      break;
    default:
      bundle->resident_stage_mask = options->resident_stage_mask;
      break;
  }
}

static void id4_ideogram4_generation_bundle_assign_slot(
    id4_ideogram4_generation_stage_slot_t* slot, id4_pipeline_stage_t* stage,
    id4_pipeline_plan_t* plan) {
  slot->stage = stage;
  slot->plan = plan;
  id4_pipeline_plan_retain(plan);
}

static iree_status_t id4_ideogram4_generation_bundle_create(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_t* plan,
    iree_allocator_t host_allocator,
    id4_ideogram4_generation_bundle_t** out_bundle) {
  *out_bundle = NULL;
  id4_ideogram4_generation_bundle_t* bundle = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*bundle), (void**)&bundle));
  memset(bundle, 0, sizeof(*bundle));
  iree_atomic_ref_count_init(&bundle->ref_count);
  bundle->host_allocator = host_allocator;
  bundle->session = session;
  bundle->summary = plan->summary;
  id4_ideogram4_session_retain(session);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_NOISE],
      session->sampler_noise_stage, plan->sampler_noise_plan);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_QWEN], session->qwen_stage,
      plan->qwen_plan);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED],
      session->dit_conditioned_stage, plan->dit_conditioned_plan);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED],
      session->dit_unconditioned_stage, plan->dit_unconditioned_plan);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER],
      session->sampler_denoise_stage, plan->sampler_denoise_plan);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DECODE],
      session->decode_stage, plan->decode_plan);
  *out_bundle = bundle;
  return iree_ok_status();
}

static void id4_ideogram4_generation_bundle_retain(
    id4_ideogram4_generation_bundle_t* bundle) {
  if (!bundle) return;
  iree_atomic_ref_count_inc(&bundle->ref_count);
}

static iree_status_t id4_ideogram4_generation_stage_plan_placement(
    const id4_pipeline_plan_t* plan, iree_host_size_t* out_device_index,
    iree_hal_queue_affinity_t* out_queue_affinity) {
  const iree_host_size_t placement_count =
      id4_pipeline_plan_placement_count(plan);
  if (placement_count != 1) {
    iree_string_view_t stage_name = id4_pipeline_plan_stage_name(plan);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation stage %.*s has %" PRIhsz
        " placements; generation preparation currently requires one "
        "placement per coarse stage",
        (int)stage_name.size, stage_name.data, placement_count);
  }
  const id4_pipeline_device_placement_t* placement =
      id4_pipeline_plan_placement_at(plan, 0);
  if (!placement) {
    iree_string_view_t stage_name = id4_pipeline_plan_stage_name(plan);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation stage %.*s placement is missing",
        (int)stage_name.size, stage_name.data);
  }
  *out_device_index = placement->device_index;
  *out_queue_affinity = placement->queue_affinity;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_bundle_select_placement(
    id4_ideogram4_generation_bundle_t* bundle) {
  iree_host_size_t device_index = 0;
  iree_hal_queue_affinity_t queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_plan_placement(
      bundle->stages[0].plan, &device_index, &queue_affinity));
  iree_hal_device_group_t* device_group =
      id4_pipeline_plan_device_group(bundle->stages[0].plan);
  for (iree_host_size_t i = 1; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    iree_host_size_t stage_device_index = 0;
    iree_hal_queue_affinity_t stage_queue_affinity =
        IREE_HAL_QUEUE_AFFINITY_ANY;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_plan_placement(
        bundle->stages[i].plan, &stage_device_index, &stage_queue_affinity));
    if (id4_pipeline_plan_device_group(bundle->stages[i].plan) !=
        device_group) {
      iree_string_view_t stage_name =
          id4_pipeline_plan_stage_name(bundle->stages[i].plan);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation stage %.*s was planned for a different "
          "device group",
          (int)stage_name.size, stage_name.data);
    }
    if (stage_device_index != device_index ||
        stage_queue_affinity != queue_affinity) {
      iree_string_view_t stage_name =
          id4_pipeline_plan_stage_name(bundle->stages[i].plan);
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Ideogram 4 generation stage %.*s placement differs from the first "
          "stage; multi-placement generation preparation is not implemented",
          (int)stage_name.size, stage_name.data);
    }
  }
  bundle->device = iree_hal_device_group_device_at(device_group, device_index);
  if (!bundle->device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation selected device %" PRIhsz
                            " is missing",
                            device_index);
  }
  bundle->queue_affinity = queue_affinity;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_bundle_allocate_bindings(
    id4_ideogram4_generation_bundle_t* bundle) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT && iree_status_is_ok(status);
       ++i) {
    id4_ideogram4_generation_stage_slot_t* slot = &bundle->stages[i];
    status = id4_pipeline_allocate_boundary_bindings(
        bundle->device, bundle->queue_affinity, slot->plan,
        bundle->host_allocator, &slot->boundary_bindings);
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_allocate_diagnostic_tap_bindings(
          bundle->device, bundle->queue_affinity, slot->plan,
          bundle->host_allocator, &slot->diagnostic_tap_bindings);
    }
  }
  return status;
}

static bool id4_ideogram4_generation_shapes_equal(
    id4_pipeline_tensor_shape_t lhs, id4_pipeline_tensor_shape_t rhs) {
  if (lhs.rank != rhs.rank) return false;
  for (uint32_t i = 0; i < lhs.rank; ++i) {
    if (lhs.dims[i] != rhs.dims[i]) return false;
  }
  return true;
}

static iree_status_t id4_ideogram4_generation_find_boundary_tensor(
    const id4_pipeline_plan_t* plan, iree_string_view_t name,
    const id4_pipeline_boundary_tensor_plan_t** out_boundary) {
  *out_boundary = NULL;
  const iree_host_size_t boundary_count =
      id4_pipeline_plan_boundary_tensor_count(plan);
  for (iree_host_size_t i = 0; i < boundary_count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (boundary && iree_string_view_equal(boundary->layout.name, name)) {
      *out_boundary = boundary;
      return iree_ok_status();
    }
  }
  iree_string_view_t stage_name = id4_pipeline_plan_stage_name(plan);
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "Ideogram 4 generation stage %.*s has no boundary tensor `%.*s`",
      (int)stage_name.size, stage_name.data, (int)name.size, name.data);
}

static iree_status_t id4_ideogram4_generation_validate_boundary_alias(
    const id4_ideogram4_generation_stage_slot_t* source_slot,
    iree_string_view_t source_name,
    const id4_ideogram4_generation_stage_slot_t* target_slot,
    iree_string_view_t target_name) {
  const id4_pipeline_boundary_tensor_plan_t* source_boundary = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_find_boundary_tensor(
      source_slot->plan, source_name, &source_boundary));
  const id4_pipeline_boundary_tensor_plan_t* target_boundary = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_find_boundary_tensor(
      target_slot->plan, target_name, &target_boundary));

  iree_string_view_t source_stage_name =
      id4_pipeline_plan_stage_name(source_slot->plan);
  iree_string_view_t target_stage_name =
      id4_pipeline_plan_stage_name(target_slot->plan);
  if (!iree_all_bits_set(source_boundary->flags,
                         ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation alias source %.*s/%.*s is not exported",
        (int)source_stage_name.size, source_stage_name.data,
        (int)source_name.size, source_name.data);
  }
  if (!iree_all_bits_set(target_boundary->flags,
                         ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_IMPORTED)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation alias target %.*s/%.*s is not imported",
        (int)target_stage_name.size, target_stage_name.data,
        (int)target_name.size, target_name.data);
  }
  const id4_pipeline_tensor_layout_t* source_layout = &source_boundary->layout;
  const id4_pipeline_tensor_layout_t* target_layout = &target_boundary->layout;
  if (source_layout->dtype != target_layout->dtype ||
      !id4_ideogram4_generation_shapes_equal(source_layout->shape,
                                             target_layout->shape) ||
      source_layout->byte_length != target_layout->byte_length ||
      source_layout->alignment != target_layout->alignment) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation boundary alias %.*s/%.*s to %.*s/%.*s has "
        "incompatible tensor layouts",
        (int)source_stage_name.size, source_stage_name.data,
        (int)source_name.size, source_name.data, (int)target_stage_name.size,
        target_stage_name.data, (int)target_name.size, target_name.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_bundle_apply_boundary_alias(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_boundary_alias_t* alias) {
  id4_ideogram4_generation_stage_slot_t* source_slot =
      &bundle->stages[alias->source_stage];
  id4_ideogram4_generation_stage_slot_t* target_slot =
      &bundle->stages[alias->target_stage];
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_validate_boundary_alias(
      source_slot, alias->source_name, target_slot, alias->target_name));

  iree_hal_buffer_binding_t replacement;
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      source_slot->plan, &source_slot->boundary_bindings, alias->source_name,
      &replacement));
  return id4_pipeline_replace_boundary_binding(target_slot->plan,
                                               &target_slot->boundary_bindings,
                                               alias->target_name, replacement);
}

static iree_status_t id4_ideogram4_generation_bundle_apply_boundary_aliases(
    id4_ideogram4_generation_bundle_t* bundle) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_boundary_aliases); ++i) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_bundle_apply_boundary_alias(
        bundle, &id4_ideogram4_generation_boundary_aliases[i]));
  }
  return iree_ok_status();
}

static iree_io_parameter_provider_t*
id4_ideogram4_generation_prepare_stage_parameter_provider(
    const id4_ideogram4_generation_parameter_providers_t* providers,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal) {
  iree_io_parameter_provider_t*
      stage_providers[ID4_IDEOGRAM4_GENERATION_STAGE_COUNT] = {
          NULL,
          providers->qwen,
          providers->dit_conditioned,
          providers->dit_unconditioned,
          NULL,
          providers->vae,
      };
  if (stage_ordinal >= ID4_IDEOGRAM4_GENERATION_STAGE_COUNT) return NULL;
  return stage_providers[stage_ordinal];
}

void id4_ideogram4_generation_bundle_release(
    id4_ideogram4_generation_bundle_t* bundle) {
  if (IREE_LIKELY(bundle) &&
      iree_atomic_ref_count_dec(&bundle->ref_count) == 1) {
    for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT;
         ++i) {
      id4_pipeline_bundle_release(bundle->resident_stage_bundles[i]);
    }
    for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT;
         ++i) {
      id4_ideogram4_generation_stage_slot_deinitialize(&bundle->stages[i]);
    }
    id4_pipeline_kernel_library_release(bundle->kernel_library);
    id4_ideogram4_generation_parameter_providers_release(
        &bundle->parameter_providers);
    id4_ideogram4_session_release(bundle->session);
    iree_allocator_t host_allocator = bundle->host_allocator;
    iree_allocator_free(host_allocator, bundle);
  }
}

static iree_status_t id4_ideogram4_validate_generation_issue_options(
    const id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_issue_options_t* options) {
  if (!session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session is required");
  }
  if (!session->is_loaded) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Ideogram 4 session must be loaded before generation issue");
  }
  if (!bundle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation bundle is required");
  }
  if (bundle->session != session) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation bundle was prepared by a different session");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_bundle_residency(
      bundle->residency_mode, bundle->resident_stage_mask));
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation issue options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation issue")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation issue extension structures are not supported");
  }
  switch (options->issue_policy) {
    case ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_PHASE_CONCURRENT:
    case ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_STAGE_SERIAL:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram 4 generation issue policy %" PRIu32
                              " is invalid",
                              (uint32_t)options->issue_policy);
  }
  if (options->issue_policy ==
          ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_STAGE_SERIAL &&
      bundle->residency_mode ==
          ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_STAGE_BUNDLES) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 phase-stage-bundle residency requires phase-concurrent "
        "generation issue");
  }
  if (!options->tokenizer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue tokenizer is required");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_request(session, options->request));
  if (options->request->generation.denoise_step_count !=
      bundle->summary.denoise_step_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue step count does not match prepared "
        "bundle");
  }
  id4_pipeline_program_shape_t request_latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  if (request_latent_shape.rank !=
      bundle->summary.diffusion_latent_shape.rank) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue latent shape does not match prepared "
        "bundle");
  }
  for (uint32_t i = 0; i < request_latent_shape.rank; ++i) {
    if (request_latent_shape.dims[i] !=
        bundle->summary.diffusion_latent_shape.dims[i]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation issue latent shape does not match prepared "
          "bundle");
    }
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("Ideogram 4 generation issue "
                                            "wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list,
      IREE_SV("Ideogram 4 generation issue signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue final signal is required");
  }
  return id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 generation issue"));
}

static iree_status_t id4_ideogram4_validate_generation_begin_options(
    const id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_begin_options_t* options) {
  if (!session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session is required");
  }
  if (!session->is_loaded) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Ideogram 4 session must be loaded before generation begin");
  }
  if (!bundle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation bundle is required");
  }
  if (bundle->session != session) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation bundle was prepared by a different session");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_bundle_residency(
      bundle->residency_mode, bundle->resident_stage_mask));
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation begin options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation begin")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation begin extension structures are not supported");
  }
  if (!options->tokenizer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation begin tokenizer is required");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_request(session, options->request));
  if (options->request->generation.denoise_step_count !=
      bundle->summary.denoise_step_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation begin step count does not match prepared "
        "bundle");
  }
  id4_pipeline_program_shape_t request_latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  if (request_latent_shape.rank !=
      bundle->summary.diffusion_latent_shape.rank) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation begin latent shape does not match prepared "
        "bundle");
  }
  for (uint32_t i = 0; i < request_latent_shape.rank; ++i) {
    if (request_latent_shape.dims[i] !=
        bundle->summary.diffusion_latent_shape.dims[i]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation begin latent shape does not match prepared "
          "bundle");
    }
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("Ideogram 4 generation begin "
                                            "wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list,
      IREE_SV("Ideogram 4 generation begin signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation begin signal is required");
  }
  return id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 generation begin"));
}

static iree_status_t id4_ideogram4_validate_single_generation_phase_mask(
    id4_ideogram4_generation_phase_mask_t phase_mask) {
  switch (phase_mask) {
    case ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING:
    case ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE:
    case ID4_IDEOGRAM4_GENERATION_PHASE_DECODE:
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation phase mask 0x%x does not identify one "
          "generation phase",
          phase_mask);
  }
}

static iree_status_t id4_ideogram4_validate_generation_phase_prepare_options(
    const id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_phase_prepare_options_t* options) {
  if (!bundle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation bundle is required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_bundle_residency(
      bundle->residency_mode, bundle->resident_stage_mask));
  if (!options) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation phase prepare options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation phase prepare")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Ideogram 4 generation phase prepare extension "
                            "structures are not supported");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_single_generation_phase_mask(options->phase_mask));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("Ideogram 4 generation phase "
                                            "prepare wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list,
      IREE_SV("Ideogram 4 generation phase prepare signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation phase prepare signal is required");
  }
  return id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink,
      IREE_SV("Ideogram 4 generation phase prepare"));
}

static iree_status_t id4_ideogram4_validate_generation_phase_issue_options(
    const id4_ideogram4_generation_execution_t* execution,
    const id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    const id4_ideogram4_generation_phase_issue_options_t* options) {
  if (!execution) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation execution is required");
  }
  if (!phase_bundle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation phase bundle is required");
  }
  if (phase_bundle->generation_bundle &&
      phase_bundle->generation_bundle != execution->bundle) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation phase bundle does not belong to execution "
        "bundle");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_single_generation_phase_mask(
      phase_bundle->phase_mask));
  if (!options) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation phase issue options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation phase issue")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Ideogram 4 generation phase issue extension "
                            "structures are not supported");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list,
      IREE_SV("Ideogram 4 generation phase issue wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list,
      IREE_SV("Ideogram 4 generation phase issue signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation phase issue signal is required");
  }
  return id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 generation phase issue"));
}

static iree_status_t id4_ideogram4_generation_execution_allocate(
    id4_ideogram4_generation_bundle_t* bundle, iree_allocator_t host_allocator,
    id4_ideogram4_generation_execution_t** out_execution) {
  *out_execution = NULL;
  id4_ideogram4_generation_execution_t* execution = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*execution),
                                             (void**)&execution));
  memset(execution, 0, sizeof(*execution));
  execution->host_allocator = host_allocator;
  execution->bundle = bundle;
  id4_ideogram4_generation_bundle_retain(bundle);
  *out_execution = execution;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_execution_create_semaphores(
    id4_ideogram4_generation_execution_t* execution) {
  id4_ideogram4_generation_bundle_t* bundle = execution->bundle;
  iree_status_t status = iree_hal_semaphore_create(
      bundle->device, bundle->queue_affinity, 0, IREE_HAL_SEMAPHORE_FLAG_NONE,
      &execution->upload_semaphore);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(bundle->device, bundle->queue_affinity,
                                       0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &execution->qwen_done_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(bundle->device, bundle->queue_affinity,
                                       0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &execution->noise_done_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        bundle->device, bundle->queue_affinity, 0, IREE_HAL_SEMAPHORE_FLAG_NONE,
        &execution->dit_conditioned_done_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        bundle->device, bundle->queue_affinity, 0, IREE_HAL_SEMAPHORE_FLAG_NONE,
        &execution->dit_unconditioned_done_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(bundle->device, bundle->queue_affinity,
                                       0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &execution->sampler_done_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(bundle->device, bundle->queue_affinity,
                                       0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &execution->decode_done_semaphore);
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_upload_boundary_tensor(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    iree_string_view_t binding_name, const void* source_data,
    iree_host_size_t source_length,
    iree_hal_semaphore_list_t initial_wait_semaphore_list) {
  id4_ideogram4_generation_bundle_t* bundle = execution->bundle;
  id4_ideogram4_generation_stage_slot_t* slot = &bundle->stages[stage_ordinal];
  id4_ideogram4_boundary_upload_context_t upload_context = {
      .device = bundle->device,
      .queue_affinity = bundle->queue_affinity,
      .plan = slot->plan,
      .boundary_bindings = &slot->boundary_bindings,
      .semaphore = execution->upload_semaphore,
      .payload_value = &execution->upload_payload_value,
  };
  return id4_ideogram4_upload_boundary_tensor(&upload_context, binding_name,
                                              source_data, source_length,
                                              initial_wait_semaphore_list);
}

static iree_status_t id4_ideogram4_generation_upload_noise_seed(
    id4_ideogram4_generation_execution_t* execution, uint64_t seed,
    iree_hal_semaphore_list_t initial_wait_semaphore_list) {
  int32_t seed_words[2] = {
      // Low 32 bits of the request seed.
      (int32_t)(seed & 0xFFFFFFFFull),
      // High 32 bits of the request seed.
      (int32_t)(seed >> 32),
  };
  return id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_NOISE, IREE_SV("seed"),
      seed_words, sizeof(seed_words), initial_wait_semaphore_list);
}

static iree_status_t id4_ideogram4_generation_upload_qwen_inputs(
    id4_ideogram4_generation_execution_t* execution,
    const id4_ideogram4_qwen_inputs_t* inputs,
    iree_hal_semaphore_list_t initial_wait_semaphore_list) {
  const iree_host_size_t token_ids_length =
      inputs->token_count * (iree_host_size_t)sizeof(inputs->token_ids[0]);
  const iree_host_size_t attention_mask_length =
      inputs->token_count * (iree_host_size_t)inputs->token_count *
      sizeof(inputs->attention_mask[0]);
  const iree_host_size_t token_weights_length =
      inputs->token_count * (iree_host_size_t)sizeof(inputs->token_weights[0]);
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN, IREE_SV("token_ids"),
      inputs->token_ids, token_ids_length, initial_wait_semaphore_list));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN, IREE_SV("attention_mask"),
      inputs->attention_mask, attention_mask_length,
      iree_hal_semaphore_list_empty()));
  return id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN, IREE_SV("token_weights"),
      inputs->token_weights, token_weights_length,
      iree_hal_semaphore_list_empty());
}

static iree_status_t id4_ideogram4_generation_upload_dit_branch_inputs(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    const id4_ideogram4_dit_branch_inputs_t* inputs) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, stage_ordinal, IREE_SV("image_indicator"),
      inputs->image_indicator, inputs->image_indicator_byte_length,
      iree_hal_semaphore_list_empty()));
  return id4_ideogram4_generation_upload_boundary_tensor(
      execution, stage_ordinal, IREE_SV("position_embedding"),
      inputs->position_embedding, inputs->position_embedding_byte_length,
      iree_hal_semaphore_list_empty());
}

static iree_status_t id4_ideogram4_generation_upload_denoise_step(
    id4_ideogram4_generation_execution_t* execution,
    const id4_ideogram4_denoise_step_t* step) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
      IREE_SV("timestep"), &step->timestep, sizeof(step->timestep),
      iree_hal_semaphore_list_empty()));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
      IREE_SV("timestep"), &step->timestep, sizeof(step->timestep),
      iree_hal_semaphore_list_empty()));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER, IREE_SV("scalings"),
      step->scalings, sizeof(step->scalings), iree_hal_semaphore_list_empty()));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER, IREE_SV("sigmas"),
      step->sigmas, sizeof(step->sigmas), iree_hal_semaphore_list_empty()));
  return id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER, IREE_SV("guidance"),
      step->guidance, sizeof(step->guidance), iree_hal_semaphore_list_empty());
}

static iree_hal_semaphore_list_t id4_ideogram4_generation_make_wait_list(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_host_size_t count) {
  iree_hal_semaphore_list_t list = {
      // Number of semaphore edges in this stack-backed list.
      .count = count,
      // Stack-backed semaphore handles.
      .semaphores = semaphore_storage,
      // Stack-backed payload values.
      .payload_values = payload_storage,
  };
  return list;
}

static void id4_ideogram4_generation_push_wait(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_host_size_t* inout_count, iree_hal_semaphore_t* semaphore,
    uint64_t payload_value) {
  if (!semaphore) return;
  semaphore_storage[*inout_count] = semaphore;
  payload_storage[*inout_count] = payload_value;
  ++*inout_count;
}

static iree_host_size_t id4_ideogram4_generation_stage_bundle_readiness_count(
    id4_pipeline_bundle_t* stage_bundle) {
  if (!stage_bundle) return 0;
  const iree_hal_semaphore_list_t readiness_list =
      id4_pipeline_bundle_readiness_semaphore_list(stage_bundle);
  if (readiness_list.count != 0) return readiness_list.count;
  id4_pipeline_parameter_slab_set_t* parameter_slabs =
      id4_pipeline_bundle_parameter_slabs(stage_bundle);
  if (id4_pipeline_parameter_slab_set_has_deferred_load_context(
          parameter_slabs)) {
    return 0;
  }
  return id4_pipeline_parameter_slab_set_load_group_count(parameter_slabs);
}

static iree_status_t id4_ideogram4_generation_append_stage_bundle_readiness(
    id4_pipeline_bundle_t* stage_bundle, iree_hal_semaphore_t** semaphores,
    uint64_t* payload_values, iree_host_size_t* inout_count) {
  if (!stage_bundle) return iree_ok_status();
  const iree_hal_semaphore_list_t readiness_list =
      id4_pipeline_bundle_readiness_semaphore_list(stage_bundle);
  if (readiness_list.count != 0) {
    for (iree_host_size_t i = 0; i < readiness_list.count; ++i) {
      semaphores[*inout_count] = readiness_list.semaphores[i];
      payload_values[*inout_count] = readiness_list.payload_values[i];
      ++*inout_count;
    }
    return iree_ok_status();
  }
  id4_pipeline_parameter_slab_set_t* parameter_slabs =
      id4_pipeline_bundle_parameter_slabs(stage_bundle);
  if (id4_pipeline_parameter_slab_set_has_deferred_load_context(
          parameter_slabs)) {
    return iree_ok_status();
  }
  const iree_host_size_t load_group_count =
      id4_pipeline_parameter_slab_set_load_group_count(parameter_slabs);
  for (iree_host_size_t i = 0; i < load_group_count; ++i) {
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_set_load_group_ready_at(
        parameter_slabs, i, &semaphores[*inout_count],
        &payload_values[*inout_count]));
    ++*inout_count;
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_wait_stage_bundle_readiness(
    id4_pipeline_bundle_t* stage_bundle) {
  if (!stage_bundle) return iree_ok_status();
  const iree_host_size_t readiness_count =
      id4_ideogram4_generation_stage_bundle_readiness_count(stage_bundle);
  if (readiness_count == 0) return iree_ok_status();
  iree_hal_semaphore_t** readiness_semaphores =
      (iree_hal_semaphore_t**)iree_alloca(readiness_count *
                                          sizeof(readiness_semaphores[0]));
  uint64_t* readiness_payload_values = (uint64_t*)iree_alloca(
      readiness_count * sizeof(readiness_payload_values[0]));
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_append_stage_bundle_readiness(
      stage_bundle, readiness_semaphores, readiness_payload_values, &count));
  if (count != readiness_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Ideogram 4 generation stage bundle readiness count changed from "
        "%" PRIhsz " to %" PRIhsz,
        readiness_count, count);
  }
  return iree_hal_semaphore_list_wait(
      (iree_hal_semaphore_list_t){
          // Number of readiness edges retained by the stage bundle.
          .count = count,
          // Stack-backed readiness semaphore handles.
          .semaphores = readiness_semaphores,
          // Stack-backed readiness payload values.
          .payload_values = readiness_payload_values,
      },
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_release_stage_bundle_ref(
    id4_ideogram4_generation_stage_bundle_ref_t* stage_bundle_ref) {
  if (!stage_bundle_ref->bundle || !stage_bundle_ref->owns_bundle) {
    stage_bundle_ref->bundle = NULL;
    stage_bundle_ref->owns_bundle = false;
    stage_bundle_ref->was_issued = false;
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  if (!stage_bundle_ref->was_issued) {
    status = id4_ideogram4_generation_wait_stage_bundle_readiness(
        stage_bundle_ref->bundle);
  }
  id4_pipeline_bundle_release(stage_bundle_ref->bundle);
  stage_bundle_ref->bundle = NULL;
  stage_bundle_ref->owns_bundle = false;
  stage_bundle_ref->was_issued = false;
  return status;
}

static iree_status_t id4_ideogram4_generation_release_stage_bundle_refs(
    id4_ideogram4_generation_stage_bundle_ref_t* stage_bundle_refs) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    status = iree_status_join(status,
                              id4_ideogram4_generation_release_stage_bundle_ref(
                                  &stage_bundle_refs[i]));
  }
  return status;
}

static void id4_ideogram4_generation_phase_bundle_initialize(
    id4_ideogram4_generation_phase_mask_t phase_mask,
    id4_ideogram4_generation_phase_bundle_t* out_phase_bundle) {
  memset(out_phase_bundle->stage_bundle_refs, 0,
         sizeof(out_phase_bundle->stage_bundle_refs));
  out_phase_bundle->phase_mask = phase_mask;
}

static iree_status_t id4_ideogram4_generation_phase_bundle_deinitialize(
    id4_ideogram4_generation_phase_bundle_t* phase_bundle) {
  iree_status_t status = id4_ideogram4_generation_release_stage_bundle_refs(
      phase_bundle->stage_bundle_refs);
  phase_bundle->phase_mask = ID4_IDEOGRAM4_GENERATION_PHASE_NONE;
  return status;
}

static const id4_ideogram4_generation_phase_descriptor_t*
id4_ideogram4_generation_phase_descriptor_for_mask(
    id4_ideogram4_generation_phase_mask_t phase_mask) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_phase_descriptors); ++i) {
    const id4_ideogram4_generation_phase_descriptor_t* phase =
        &id4_ideogram4_generation_phase_descriptors[i];
    if (phase->phase_mask == phase_mask) return phase;
  }
  return NULL;
}

static iree_status_t id4_ideogram4_generation_prepare_stage_bundle(
    id4_ideogram4_generation_bundle_t* bundle,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    id4_ideogram4_generation_stage_prepare_mode_t prepare_mode,
    iree_hal_semaphore_list_t wait_semaphore_list,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_bundle_t** out_stage_bundle) {
  IREE_ASSERT_ARGUMENT(out_stage_bundle);
  *out_stage_bundle = NULL;
  id4_ideogram4_generation_stage_slot_t* slot = &bundle->stages[stage_ordinal];
  const bool has_parameter_slabs =
      id4_pipeline_plan_parameter_slab_count(slot->plan) != 0;
  const bool defer_parameter_loads_to_issue =
      has_parameter_slabs &&
      prepare_mode ==
          ID4_IDEOGRAM4_GENERATION_STAGE_PREPARE_MODE_DEFER_PARAMETERS;
  const bool materialize_parameter_slabs =
      has_parameter_slabs &&
      (prepare_mode ==
           ID4_IDEOGRAM4_GENERATION_STAGE_PREPARE_MODE_MATERIALIZE_PARAMETERS ||
       prepare_mode ==
           ID4_IDEOGRAM4_GENERATION_STAGE_PREPARE_MODE_RETAIN_PARAMETERS);
  const bool retain_parameter_slabs =
      materialize_parameter_slabs &&
      prepare_mode ==
          ID4_IDEOGRAM4_GENERATION_STAGE_PREPARE_MODE_RETAIN_PARAMETERS;

  id4_pipeline_parameter_slab_set_t* resident_parameter_slabs =
      bundle->session->resident_stage_parameter_slabs[stage_ordinal];
  const bool reuse_parameter_slabs =
      retain_parameter_slabs && resident_parameter_slabs != NULL;
  if (reuse_parameter_slabs) {
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_validate_parameter_slabs(
        slot->plan, resident_parameter_slabs));
  }

  iree_hal_semaphore_t* materialize_semaphore = NULL;
  uint64_t materialize_payload_value = 1;
  if (materialize_parameter_slabs && !reuse_parameter_slabs) {
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
        bundle->device, bundle->queue_affinity, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &materialize_semaphore));
  } else if (
      prepare_mode !=
          ID4_IDEOGRAM4_GENERATION_STAGE_PREPARE_MODE_DEFER_PARAMETERS &&
      prepare_mode !=
          ID4_IDEOGRAM4_GENERATION_STAGE_PREPARE_MODE_MATERIALIZE_PARAMETERS &&
      prepare_mode !=
          ID4_IDEOGRAM4_GENERATION_STAGE_PREPARE_MODE_RETAIN_PARAMETERS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation stage prepare mode %" PRIu32
                            " is invalid",
                            (uint32_t)prepare_mode);
  }

  iree_hal_semaphore_list_t signal_semaphore_list =
      materialize_semaphore
          ? (iree_hal_semaphore_list_t){
                // One internal readiness edge retained by the prepared bundle.
                .count = 1,
                // Internal readiness semaphore signaled by parameter loading.
                .semaphores = &materialize_semaphore,
                // Payload value paired with the internal readiness semaphore.
                .payload_values = &materialize_payload_value,
            }
          : iree_hal_semaphore_list_empty();

  id4_pipeline_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  if (defer_parameter_loads_to_issue) {
    prepare_options.flags =
        ID4_PIPELINE_STAGE_PREPARE_FLAG_DEFER_PARAMETER_LOADS_TO_ISSUE;
  } else if (reuse_parameter_slabs) {
    prepare_options.flags =
        ID4_PIPELINE_STAGE_PREPARE_FLAG_REUSE_PARAMETER_SLABS;
  }
  prepare_options.parameter_provider =
      reuse_parameter_slabs
          ? NULL
          : id4_ideogram4_generation_prepare_stage_parameter_provider(
                &bundle->parameter_providers, stage_ordinal);
  prepare_options.parameter_slabs =
      reuse_parameter_slabs ? resident_parameter_slabs : NULL;
  prepare_options.kernel_library = bundle->kernel_library;
  prepare_options.wait_semaphore_list =
      has_parameter_slabs && !reuse_parameter_slabs
          ? wait_semaphore_list
          : iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = signal_semaphore_list;
  prepare_options.command_buffer_mode = bundle->command_buffer_mode;
  prepare_options.kernel_diagnostic_artifact_flags =
      kernel_diagnostic_artifact_flags;
  prepare_options.diagnostics_sink = diagnostics_sink;
  iree_status_t status = id4_pipeline_stage_prepare(
      slot->stage, slot->plan, &prepare_options, out_stage_bundle);
  if (iree_status_is_ok(status) && retain_parameter_slabs &&
      !reuse_parameter_slabs) {
    id4_pipeline_parameter_slab_set_t* parameter_slabs =
        id4_pipeline_bundle_parameter_slabs(*out_stage_bundle);
    if (!parameter_slabs) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "resident stage bundle did not produce parameter slabs");
    } else {
      status = id4_pipeline_plan_validate_parameter_slabs(slot->plan,
                                                          parameter_slabs);
      if (iree_status_is_ok(status)) {
        id4_pipeline_parameter_slab_set_retain(parameter_slabs);
        bundle->session->resident_stage_parameter_slabs[stage_ordinal] =
            parameter_slabs;
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, id4_ideogram4_generation_wait_stage_bundle_readiness(
                    *out_stage_bundle));
    id4_pipeline_bundle_release(*out_stage_bundle);
    *out_stage_bundle = NULL;
  }
  iree_hal_semaphore_release(materialize_semaphore);
  return status;
}

static iree_status_t id4_ideogram4_generation_bundle_signal_prepared(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_prepare_options_t* options) {
  return iree_hal_device_queue_barrier(
      bundle->device, bundle->queue_affinity, options->wait_semaphore_list,
      options->signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_bundle_wait_resident_bundles(
    id4_ideogram4_generation_bundle_t* bundle) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    status = iree_status_join(
        status, id4_ideogram4_generation_wait_stage_bundle_readiness(
                    bundle->resident_stage_bundles[i]));
  }
  return status;
}

static iree_host_size_t
id4_ideogram4_generation_bundle_resident_readiness_count(
    id4_ideogram4_generation_bundle_t* bundle) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    count += id4_ideogram4_generation_stage_bundle_readiness_count(
        bundle->resident_stage_bundles[i]);
  }
  return count;
}

static iree_status_t id4_ideogram4_generation_bundle_resident_readiness_list(
    id4_ideogram4_generation_bundle_t* bundle,
    iree_hal_semaphore_t** semaphores, uint64_t* payload_values,
    iree_hal_semaphore_list_t* out_readiness_list) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_append_stage_bundle_readiness(
        bundle->resident_stage_bundles[i], semaphores, payload_values, &count));
  }
  *out_readiness_list = (iree_hal_semaphore_list_t){
      // Number of readiness edges retained by resident stage bundles.
      .count = count,
      // Stack-backed readiness semaphore handles.
      .semaphores = count == 0 ? NULL : semaphores,
      // Stack-backed readiness payload values.
      .payload_values = count == 0 ? NULL : payload_values,
  };
  return iree_ok_status();
}

static iree_status_t
id4_ideogram4_generation_bundle_signal_resident_bundles_prepared(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_prepare_options_t* options) {
  const iree_host_size_t readiness_count =
      id4_ideogram4_generation_bundle_resident_readiness_count(bundle);
  iree_host_size_t wait_count = 0;
  if (!iree_host_size_checked_add(options->wait_semaphore_list.count,
                                  readiness_count, &wait_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram 4 generation resident prepare wait "
                            "list count overflow");
  }
  iree_hal_semaphore_t** readiness_semaphores = NULL;
  uint64_t* readiness_payload_values = NULL;
  if (wait_count != 0) {
    readiness_semaphores = (iree_hal_semaphore_t**)iree_alloca(
        wait_count * sizeof(readiness_semaphores[0]));
    readiness_payload_values = (uint64_t*)iree_alloca(
        wait_count * sizeof(readiness_payload_values[0]));
  }
  for (iree_host_size_t i = 0; i < options->wait_semaphore_list.count; ++i) {
    readiness_semaphores[i] = options->wait_semaphore_list.semaphores[i];
    readiness_payload_values[i] =
        options->wait_semaphore_list.payload_values[i];
  }
  iree_hal_semaphore_t** resident_readiness_semaphores =
      wait_count == 0
          ? NULL
          : readiness_semaphores + options->wait_semaphore_list.count;
  uint64_t* resident_readiness_payload_values =
      wait_count == 0
          ? NULL
          : readiness_payload_values + options->wait_semaphore_list.count;
  iree_hal_semaphore_list_t readiness_list = iree_hal_semaphore_list_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_bundle_resident_readiness_list(
      bundle, resident_readiness_semaphores, resident_readiness_payload_values,
      &readiness_list));
  readiness_list.count = wait_count;
  readiness_list.semaphores = wait_count == 0 ? NULL : readiness_semaphores;
  readiness_list.payload_values =
      wait_count == 0 ? NULL : readiness_payload_values;
  return iree_hal_device_queue_barrier(
      bundle->device, bundle->queue_affinity, readiness_list,
      options->signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static void id4_ideogram4_session_trim_resident_parameter_slabs(
    id4_ideogram4_session_t* session,
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors); ++i) {
    const id4_ideogram4_generation_stage_descriptor_t* descriptor =
        &id4_ideogram4_generation_stage_descriptors[i];
    if (iree_any_bit_set(resident_stage_mask, descriptor->resident_stage_bit)) {
      continue;
    }
    const id4_ideogram4_generation_stage_ordinal_t stage_ordinal =
        descriptor->ordinal;
    id4_pipeline_parameter_slab_set_release(
        session->resident_stage_parameter_slabs[stage_ordinal]);
    session->resident_stage_parameter_slabs[stage_ordinal] = NULL;
  }
}

static bool id4_ideogram4_generation_bundle_stage_is_resident(
    const id4_ideogram4_generation_bundle_t* bundle,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal) {
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor(stage_ordinal);
  return descriptor && iree_any_bit_set(bundle->resident_stage_mask,
                                        descriptor->resident_stage_bit);
}

static iree_status_t id4_ideogram4_generation_bundle_prepare_resident_stages(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_prepare_options_t* options) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors) &&
       iree_status_is_ok(status);
       ++i) {
    const id4_ideogram4_generation_stage_descriptor_t* descriptor =
        &id4_ideogram4_generation_stage_descriptors[i];
    if (!iree_any_bit_set(bundle->resident_stage_mask,
                          descriptor->resident_stage_bit)) {
      continue;
    }
    const id4_ideogram4_generation_stage_ordinal_t stage_ordinal =
        descriptor->ordinal;
    if (bundle->resident_stage_bundles[stage_ordinal]) continue;
    status = id4_ideogram4_generation_prepare_stage_bundle(
        bundle, stage_ordinal,
        ID4_IDEOGRAM4_GENERATION_STAGE_PREPARE_MODE_RETAIN_PARAMETERS,
        options->wait_semaphore_list, options->kernel_diagnostic_artifact_flags,
        options->diagnostics_sink,
        &bundle->resident_stage_bundles[stage_ordinal]);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_bundle_signal_resident_bundles_prepared(
        bundle, options);
  }
  return status;
}

iree_status_t id4_ideogram4_session_prepare_generation(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_prepare_options_t* options,
    id4_ideogram4_generation_bundle_t** out_bundle) {
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_prepare_options(
      session, plan, options));

  id4_ideogram4_generation_bundle_t* bundle = NULL;
  iree_status_t status = id4_ideogram4_generation_bundle_create(
      session, plan, session->host_allocator, &bundle);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_bundle_select_placement(bundle);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_bundle_allocate_bindings(bundle);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_bundle_apply_boundary_aliases(bundle);
  }
  if (iree_status_is_ok(status)) {
    id4_ideogram4_generation_bundle_capture_prepare_resources(bundle, options);
    id4_ideogram4_session_trim_resident_parameter_slabs(
        session, bundle->resident_stage_mask);
  }
  if (iree_status_is_ok(status)) {
    switch (options->residency_mode) {
      case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ISSUE_PHASES:
        status =
            id4_ideogram4_generation_bundle_signal_prepared(bundle, options);
        break;
      case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_STAGE_BUNDLES:
        status =
            id4_ideogram4_generation_bundle_signal_prepared(bundle, options);
        break;
      case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_SELECTED_STAGE_BUNDLES:
        status = id4_ideogram4_generation_bundle_prepare_resident_stages(
            bundle, options);
        break;
      case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES:
        status = id4_ideogram4_generation_bundle_prepare_resident_stages(
            bundle, options);
        break;
      default:
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 generation residency mode %" PRIu32 " is invalid",
            (uint32_t)options->residency_mode);
        break;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_bundle = bundle;
  } else {
    status = iree_status_join(
        status, id4_ideogram4_generation_bundle_wait_resident_bundles(bundle));
    id4_ideogram4_generation_bundle_release(bundle);
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_acquire_stage_bundle_ref(
    id4_ideogram4_generation_bundle_t* bundle,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    iree_hal_semaphore_list_t wait_semaphore_list,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_ideogram4_generation_stage_bundle_ref_t* out_stage_bundle_ref) {
  memset(out_stage_bundle_ref, 0, sizeof(*out_stage_bundle_ref));
  if (id4_ideogram4_generation_bundle_stage_is_resident(bundle,
                                                        stage_ordinal)) {
    if (!bundle->resident_stage_bundles[stage_ordinal]) {
      IREE_RETURN_IF_ERROR(id4_ideogram4_generation_prepare_stage_bundle(
          bundle, stage_ordinal,
          ID4_IDEOGRAM4_GENERATION_STAGE_PREPARE_MODE_RETAIN_PARAMETERS,
          wait_semaphore_list, kernel_diagnostic_artifact_flags,
          diagnostics_sink, &bundle->resident_stage_bundles[stage_ordinal]));
    }
    out_stage_bundle_ref->bundle =
        bundle->resident_stage_bundles[stage_ordinal];
    out_stage_bundle_ref->owns_bundle = false;
    return iree_ok_status();
  }
  const id4_ideogram4_generation_stage_prepare_mode_t prepare_mode =
      bundle->residency_mode ==
              ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_STAGE_BUNDLES
          ? ID4_IDEOGRAM4_GENERATION_STAGE_PREPARE_MODE_MATERIALIZE_PARAMETERS
          : ID4_IDEOGRAM4_GENERATION_STAGE_PREPARE_MODE_DEFER_PARAMETERS;
  iree_status_t status = id4_ideogram4_generation_prepare_stage_bundle(
      bundle, stage_ordinal, prepare_mode, wait_semaphore_list,
      kernel_diagnostic_artifact_flags, diagnostics_sink,
      &out_stage_bundle_ref->bundle);
  out_stage_bundle_ref->owns_bundle = iree_status_is_ok(status);
  return status;
}

static iree_status_t id4_ideogram4_generation_prepare_phase_bundle(
    id4_ideogram4_generation_bundle_t* bundle,
    id4_ideogram4_generation_phase_mask_t phase_mask,
    iree_hal_semaphore_list_t wait_semaphore_list,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_ideogram4_generation_phase_bundle_t* out_phase_bundle) {
  id4_ideogram4_generation_phase_bundle_initialize(phase_mask,
                                                   out_phase_bundle);
  const id4_ideogram4_generation_phase_descriptor_t* phase =
      id4_ideogram4_generation_phase_descriptor_for_mask(phase_mask);
  if (!phase) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation phase mask 0x%x does not "
                            "identify one generation phase",
                            phase_mask);
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < phase->stage_count && iree_status_is_ok(status); ++i) {
    const id4_ideogram4_generation_stage_ordinal_t stage_ordinal =
        phase->stage_ordinals[i];
    status = id4_ideogram4_generation_acquire_stage_bundle_ref(
        bundle, stage_ordinal, wait_semaphore_list,
        kernel_diagnostic_artifact_flags, diagnostics_sink,
        &out_phase_bundle->stage_bundle_refs[stage_ordinal]);
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status,
        id4_ideogram4_generation_phase_bundle_deinitialize(out_phase_bundle));
  }
  return status;
}

static id4_pipeline_bundle_t* id4_ideogram4_generation_phase_stage_bundle(
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal) {
  return phase_bundle->stage_bundle_refs[stage_ordinal].bundle;
}

static iree_host_size_t id4_ideogram4_generation_phase_bundle_readiness_count(
    id4_ideogram4_generation_phase_bundle_t* phase_bundle) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    id4_pipeline_bundle_t* stage_bundle =
        phase_bundle->stage_bundle_refs[i].bundle;
    count +=
        id4_ideogram4_generation_stage_bundle_readiness_count(stage_bundle);
  }
  return count;
}

static iree_status_t id4_ideogram4_generation_phase_bundle_readiness_list(
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    iree_hal_semaphore_t** semaphores, uint64_t* payload_values,
    iree_hal_semaphore_list_t* out_readiness_list) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    id4_pipeline_bundle_t* stage_bundle =
        phase_bundle->stage_bundle_refs[i].bundle;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_append_stage_bundle_readiness(
        stage_bundle, semaphores, payload_values, &count));
  }
  *out_readiness_list = (iree_hal_semaphore_list_t){
      // Number of readiness edges retained by the phase bundle.
      .count = count,
      // Stack-backed readiness semaphore handles.
      .semaphores = count == 0 ? NULL : semaphores,
      // Stack-backed readiness payload values.
      .payload_values = count == 0 ? NULL : payload_values,
  };
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_phase_bundle_signal_prepared(
    id4_ideogram4_generation_bundle_t* bundle,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  const iree_host_size_t readiness_count =
      id4_ideogram4_generation_phase_bundle_readiness_count(phase_bundle);
  iree_hal_semaphore_t** readiness_semaphores =
      readiness_count == 0
          ? NULL
          : (iree_hal_semaphore_t**)iree_alloca(
                readiness_count * sizeof(readiness_semaphores[0]));
  uint64_t* readiness_payload_values =
      readiness_count == 0
          ? NULL
          : (uint64_t*)iree_alloca(readiness_count *
                                   sizeof(readiness_payload_values[0]));
  iree_hal_semaphore_list_t readiness_list = iree_hal_semaphore_list_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_phase_bundle_readiness_list(
      phase_bundle, readiness_semaphores, readiness_payload_values,
      &readiness_list));
  return iree_hal_device_queue_barrier(bundle->device, bundle->queue_affinity,
                                       readiness_list, signal_semaphore_list,
                                       IREE_HAL_EXECUTE_FLAG_NONE);
}

iree_status_t id4_ideogram4_generation_phase_bundle_release(
    id4_ideogram4_generation_phase_bundle_t* phase_bundle) {
  if (!phase_bundle) return iree_ok_status();
  iree_status_t status =
      id4_ideogram4_generation_phase_bundle_deinitialize(phase_bundle);
  id4_ideogram4_generation_bundle_release(phase_bundle->generation_bundle);
  iree_allocator_t host_allocator = phase_bundle->host_allocator;
  if (!iree_allocator_is_null(host_allocator)) {
    iree_allocator_free(host_allocator, phase_bundle);
  }
  return status;
}

iree_status_t id4_ideogram4_generation_bundle_prepare_phase(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_phase_prepare_options_t* options,
    id4_ideogram4_generation_phase_bundle_t** out_phase_bundle) {
  IREE_ASSERT_ARGUMENT(out_phase_bundle);
  *out_phase_bundle = NULL;
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_phase_prepare_options(bundle, options));

  id4_ideogram4_generation_phase_bundle_t* phase_bundle = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      bundle->host_allocator, sizeof(*phase_bundle), (void**)&phase_bundle));
  memset(phase_bundle, 0, sizeof(*phase_bundle));
  phase_bundle->host_allocator = bundle->host_allocator;
  phase_bundle->generation_bundle = bundle;
  id4_ideogram4_generation_bundle_retain(bundle);

  iree_status_t status = id4_ideogram4_generation_prepare_phase_bundle(
      bundle, options->phase_mask, options->wait_semaphore_list,
      options->kernel_diagnostic_artifact_flags, options->diagnostics_sink,
      phase_bundle);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_phase_bundle_signal_prepared(
        bundle, phase_bundle, options->signal_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    *out_phase_bundle = phase_bundle;
  } else {
    status = iree_status_join(
        status, id4_ideogram4_generation_phase_bundle_release(phase_bundle));
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_chain_upload_after_sampler(
    id4_ideogram4_generation_execution_t* execution,
    uint64_t sampler_payload_value) {
  iree_hal_semaphore_t* wait_semaphores[2];
  uint64_t wait_payload_values[2];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     execution->upload_payload_value);
  id4_ideogram4_generation_push_wait(
      wait_semaphores, wait_payload_values, &wait_count,
      execution->sampler_done_semaphore, sampler_payload_value);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  iree_hal_semaphore_t* signal_semaphore = NULL;
  uint64_t signal_payload_value = execution->upload_payload_value + 1;
  iree_hal_semaphore_list_t signal_list = id4_ideogram4_single_semaphore_list(
      &signal_semaphore, &signal_payload_value, execution->upload_semaphore,
      signal_payload_value);
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_barrier(
      execution->bundle->device, execution->bundle->queue_affinity, wait_list,
      signal_list, IREE_HAL_EXECUTE_FLAG_NONE));
  execution->upload_payload_value = signal_payload_value;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_chain_phase_issue_wait(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_mask_t phase_mask,
    iree_hal_semaphore_list_t phase_wait_list) {
  if (phase_wait_list.count == 0) return iree_ok_status();
  iree_host_size_t wait_count_capacity = 0;
  if (!iree_host_size_checked_add(phase_wait_list.count, 1,
                                  &wait_count_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram 4 generation phase issue wait list count overflow");
  }
  iree_hal_semaphore_t** wait_semaphores = (iree_hal_semaphore_t**)iree_alloca(
      wait_count_capacity * sizeof(wait_semaphores[0]));
  uint64_t* wait_payload_values = (uint64_t*)iree_alloca(
      wait_count_capacity * sizeof(wait_payload_values[0]));
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     execution->upload_payload_value);
  for (iree_host_size_t i = 0; i < phase_wait_list.count; ++i) {
    id4_ideogram4_generation_push_wait(
        wait_semaphores, wait_payload_values, &wait_count,
        phase_wait_list.semaphores[i], phase_wait_list.payload_values[i]);
  }
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  iree_hal_semaphore_t* signal_semaphore = NULL;
  uint64_t signal_payload_value = execution->upload_payload_value + 1;
  iree_hal_semaphore_list_t signal_list = id4_ideogram4_single_semaphore_list(
      &signal_semaphore, &signal_payload_value, execution->upload_semaphore,
      signal_payload_value);
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_barrier(
      execution->bundle->device, execution->bundle->queue_affinity, wait_list,
      signal_list, IREE_HAL_EXECUTE_FLAG_NONE));
  execution->upload_payload_value = signal_payload_value;
  if (phase_mask == ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING) {
    execution->qwen_upload_payload_value = signal_payload_value;
    execution->seed_upload_payload_value = signal_payload_value;
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_issue_stage(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    id4_pipeline_bundle_t* stage_bundle,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_t* signal_semaphore, uint64_t signal_payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_slot_t* slot =
      &execution->bundle->stages[stage_ordinal];
  iree_hal_semaphore_t* signal_semaphore_storage = NULL;
  uint64_t signal_payload_storage = 0;
  iree_hal_semaphore_list_t signal_list = id4_ideogram4_single_semaphore_list(
      &signal_semaphore_storage, &signal_payload_storage, signal_semaphore,
      signal_payload_value);

  id4_pipeline_stage_issue_options_t issue_options;
  memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.boundary_binding_count = slot->boundary_bindings.count;
  issue_options.boundary_bindings = slot->boundary_bindings.bindings;
  issue_options.diagnostic_tap_binding_count =
      slot->diagnostic_tap_bindings.count;
  issue_options.diagnostic_tap_bindings =
      slot->diagnostic_tap_bindings.bindings;
  id4_pipeline_parameter_slab_set_t* parameter_slabs =
      id4_pipeline_bundle_parameter_slabs(stage_bundle);
  if (parameter_slabs &&
      id4_pipeline_parameter_slab_set_has_deferred_load_context(
          parameter_slabs)) {
    issue_options.parameter_load_prefetch_region_distance =
        execution->parameter_load_prefetch_region_distance;
  }
  issue_options.wait_semaphore_list = wait_semaphore_list;
  issue_options.signal_semaphore_list = signal_list;
  issue_options.diagnostics_sink = diagnostics_sink;
  return id4_pipeline_stage_issue(slot->stage, stage_bundle, &issue_options);
}

static iree_status_t id4_ideogram4_generation_issue_qwen(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_bundle_t* qwen_bundle, uint64_t qwen_upload_payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphores[2];
  uint64_t wait_payload_values[2];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     qwen_upload_payload_value);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  return id4_ideogram4_generation_issue_stage(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN, qwen_bundle, wait_list,
      execution->qwen_done_semaphore, 1, diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_noise(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_bundle_t* noise_bundle, uint64_t seed_upload_payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphores[2];
  uint64_t wait_payload_values[2];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     seed_upload_payload_value);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  return id4_ideogram4_generation_issue_stage(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_NOISE, noise_bundle, wait_list,
      execution->noise_done_semaphore, 1, diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_dit_branch(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    id4_pipeline_bundle_t* branch_bundle,
    iree_hal_semaphore_t* latent_semaphore, uint64_t latent_payload_value,
    iree_hal_semaphore_t* condition_semaphore, uint64_t condition_payload_value,
    iree_hal_semaphore_t* branch_done_semaphore, uint64_t branch_payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphores[4];
  uint64_t wait_payload_values[4];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     execution->upload_payload_value);
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, latent_semaphore,
                                     latent_payload_value);
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, condition_semaphore,
                                     condition_payload_value);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  return id4_ideogram4_generation_issue_stage(
      execution, stage_ordinal, branch_bundle, wait_list, branch_done_semaphore,
      branch_payload_value, diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_sampler(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_bundle_t* sampler_bundle, uint64_t payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphores[3];
  uint64_t wait_payload_values[3];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     execution->upload_payload_value);
  id4_ideogram4_generation_push_wait(
      wait_semaphores, wait_payload_values, &wait_count,
      execution->dit_conditioned_done_semaphore, payload_value);
  id4_ideogram4_generation_push_wait(
      wait_semaphores, wait_payload_values, &wait_count,
      execution->dit_unconditioned_done_semaphore, payload_value);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  return id4_ideogram4_generation_issue_stage(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER, sampler_bundle,
      wait_list, execution->sampler_done_semaphore, payload_value,
      diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_decode(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_bundle_t* decode_bundle, uint64_t sampler_payload,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphores[3];
  uint64_t wait_payload_values[3];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     execution->upload_payload_value);
  id4_ideogram4_generation_push_wait(
      wait_semaphores, wait_payload_values, &wait_count,
      execution->sampler_done_semaphore, sampler_payload);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_issue_stage(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_DECODE, decode_bundle,
      wait_list, execution->decode_done_semaphore, 1, diagnostics_sink));
  iree_hal_semaphore_t* wait_semaphore = NULL;
  uint64_t wait_payload_value = 1;
  wait_list = id4_ideogram4_single_semaphore_list(
      &wait_semaphore, &wait_payload_value, execution->decode_done_semaphore,
      wait_payload_value);
  return iree_hal_device_queue_barrier(
      execution->bundle->device, execution->bundle->queue_affinity, wait_list,
      final_signal_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_find_outputs(
    id4_ideogram4_generation_execution_t* execution) {
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED]
          .plan,
      &execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED]
           .boundary_bindings,
      IREE_SV("velocity"), &execution->conditioned_velocity_binding));
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      execution->bundle
          ->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED]
          .plan,
      &execution->bundle
           ->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED]
           .boundary_bindings,
      IREE_SV("velocity"), &execution->unconditioned_velocity_binding));
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER].plan,
      &execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER]
           .boundary_bindings,
      IREE_SV("denoised"), &execution->denoised_latent_binding));
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER].plan,
      &execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER]
           .boundary_bindings,
      IREE_SV("x_next"), &execution->final_latent_binding));
  return id4_pipeline_find_boundary_binding(
      execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DECODE].plan,
      &execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DECODE]
           .boundary_bindings,
      IREE_SV("media.image.decoded"), &execution->decoded_image_binding);
}

static iree_status_t id4_ideogram4_generation_begin_execution(
    id4_ideogram4_session_t* session, id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_request_t* request, const iree_tokenizer_t* tokenizer,
    iree_tokenizer_encode_flags_t tokenizer_flags,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    id4_ideogram4_generation_execution_t** out_execution) {
  id4_ideogram4_generation_execution_t* execution = NULL;
  iree_status_t status = id4_ideogram4_generation_execution_allocate(
      bundle, session->host_allocator, &execution);
  if (iree_status_is_ok(status)) {
    id4_ideogram4_qwen_lowering_options_t qwen_lowering_options;
    memset(&qwen_lowering_options, 0, sizeof(qwen_lowering_options));
    qwen_lowering_options.structure_size = sizeof(qwen_lowering_options);
    qwen_lowering_options.tokenizer = tokenizer;
    qwen_lowering_options.request = request;
    qwen_lowering_options.tokenizer_flags = tokenizer_flags;
    qwen_lowering_options.max_token_count = session->qwen_model.max_token_count;
    qwen_lowering_options.vocab_size = session->qwen_model.vocab_size;
    status = id4_ideogram4_request_lower_qwen_inputs(&qwen_lowering_options,
                                                     execution->host_allocator,
                                                     &execution->qwen_inputs);
  }
  if (iree_status_is_ok(status) &&
      execution->qwen_inputs.token_count != bundle->summary.qwen_token_count) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue Qwen token count %" PRIu32
        " does not match prepared bundle count %" PRIu32,
        execution->qwen_inputs.token_count, bundle->summary.qwen_token_count);
  }
  if (iree_status_is_ok(status)) {
    id4_ideogram4_dit_lowering_options_t dit_lowering_options;
    memset(&dit_lowering_options, 0, sizeof(dit_lowering_options));
    dit_lowering_options.structure_size = sizeof(dit_lowering_options);
    dit_lowering_options.generation = &request->generation;
    dit_lowering_options.text_token_count = execution->qwen_inputs.token_count;
    dit_lowering_options.attention_head_size =
        session->dit_model.hidden_size /
        session->dit_model.attention_head_count;
    status = id4_ideogram4_request_lower_dit_inputs(&dit_lowering_options,
                                                    execution->host_allocator,
                                                    &execution->dit_inputs);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_request_generation_lower_denoise_schedule(
        &request->generation, execution->host_allocator,
        &execution->denoise_schedule);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_execution_create_semaphores(execution);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_wait(wait_semaphore_list,
                                          iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_upload_qwen_inputs(
        execution, &execution->qwen_inputs, iree_hal_semaphore_list_empty());
  }
  if (iree_status_is_ok(status)) {
    execution->qwen_upload_payload_value = execution->upload_payload_value;
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_upload_noise_seed(
        execution, request->generation.seed, iree_hal_semaphore_list_empty());
  }
  if (iree_status_is_ok(status)) {
    execution->seed_upload_payload_value = execution->upload_payload_value;
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_upload_dit_branch_inputs(
        execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
        &execution->dit_inputs.conditioned);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_upload_dit_branch_inputs(
        execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
        &execution->dit_inputs.unconditioned);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_upload_denoise_step(
        execution, &execution->denoise_schedule.steps[0]);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_find_outputs(execution);
  }
  if (iree_status_is_ok(status) && signal_semaphore_list.count != 0) {
    iree_hal_semaphore_t* wait_semaphore = NULL;
    uint64_t wait_payload_value = execution->upload_payload_value;
    iree_hal_semaphore_list_t upload_wait_list =
        id4_ideogram4_single_semaphore_list(
            &wait_semaphore, &wait_payload_value, execution->upload_semaphore,
            wait_payload_value);
    status = iree_hal_device_queue_barrier(
        bundle->device, bundle->queue_affinity, upload_wait_list,
        signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    *out_execution = execution;
  } else {
    id4_ideogram4_generation_execution_release(execution);
  }
  return status;
}

iree_status_t id4_ideogram4_session_begin_generation(
    id4_ideogram4_session_t* session, id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_begin_options_t* options,
    id4_ideogram4_generation_execution_t** out_execution) {
  IREE_ASSERT_ARGUMENT(out_execution);
  *out_execution = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_begin_options(
      session, bundle, options));
  return id4_ideogram4_generation_begin_execution(
      session, bundle, options->request, options->tokenizer,
      options->tokenizer_flags, options->wait_semaphore_list,
      options->signal_semaphore_list, out_execution);
}

static iree_status_t id4_ideogram4_generation_issue_conditioning_phase(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = id4_ideogram4_generation_issue_qwen(
      execution,
      id4_ideogram4_generation_phase_stage_bundle(
          phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN),
      execution->qwen_upload_payload_value, diagnostics_sink);
  phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_QWEN]
      .was_issued = iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_noise(
        execution,
        id4_ideogram4_generation_phase_stage_bundle(
            phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_NOISE),
        execution->seed_upload_payload_value, diagnostics_sink);
    phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_NOISE]
        .was_issued = iree_status_is_ok(status);
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_wait_conditioning_phase(
    id4_ideogram4_generation_execution_t* execution) {
  iree_hal_semaphore_t* qwen_done_semaphore = execution->qwen_done_semaphore;
  uint64_t qwen_done_payload_value = 1;
  iree_hal_semaphore_list_t qwen_done_list = {
      // One Qwen completion edge required before allocating DiT weights.
      .count = 1,
      // Stack-backed Qwen completion semaphore.
      .semaphores = &qwen_done_semaphore,
      // Stack-backed Qwen completion payload value.
      .payload_values = &qwen_done_payload_value,
  };
  return iree_hal_semaphore_list_wait(qwen_done_list, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_wait_one(
    iree_hal_semaphore_t* semaphore, uint64_t payload_value) {
  iree_hal_semaphore_t* wait_semaphore = NULL;
  uint64_t wait_payload_value = payload_value;
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_single_semaphore_list(
      &wait_semaphore, &wait_payload_value, semaphore, payload_value);
  return iree_hal_semaphore_list_wait(wait_list, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_issue_qwen_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_bundle_ref_t qwen_ref = {0};
  iree_status_t status = id4_ideogram4_generation_acquire_stage_bundle_ref(
      execution->bundle, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN,
      iree_hal_semaphore_list_empty(), kernel_diagnostic_artifact_flags,
      diagnostics_sink, &qwen_ref);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_qwen(
        execution, qwen_ref.bundle, execution->qwen_upload_payload_value,
        diagnostics_sink);
    qwen_ref.was_issued = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_ideogram4_generation_wait_one(execution->qwen_done_semaphore, 1);
  }
  status = iree_status_join(
      status, id4_ideogram4_generation_release_stage_bundle_ref(&qwen_ref));
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_noise_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_bundle_ref_t noise_ref = {0};
  iree_status_t status = id4_ideogram4_generation_acquire_stage_bundle_ref(
      execution->bundle, ID4_IDEOGRAM4_GENERATION_STAGE_NOISE,
      iree_hal_semaphore_list_empty(), kernel_diagnostic_artifact_flags,
      diagnostics_sink, &noise_ref);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_noise(
        execution, noise_ref.bundle, execution->seed_upload_payload_value,
        diagnostics_sink);
    noise_ref.was_issued = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_ideogram4_generation_wait_one(execution->noise_done_semaphore, 1);
  }
  status = iree_status_join(
      status, id4_ideogram4_generation_release_stage_bundle_ref(&noise_ref));
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_dit_branch_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    iree_hal_semaphore_t* latent_semaphore, uint64_t latent_payload_value,
    iree_hal_semaphore_t* condition_semaphore, uint64_t condition_payload_value,
    iree_hal_semaphore_t* branch_done_semaphore, uint64_t branch_payload_value,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_bundle_ref_t branch_ref = {0};
  iree_status_t status = id4_ideogram4_generation_acquire_stage_bundle_ref(
      execution->bundle, stage_ordinal, iree_hal_semaphore_list_empty(),
      kernel_diagnostic_artifact_flags, diagnostics_sink, &branch_ref);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_dit_branch(
        execution, stage_ordinal, branch_ref.bundle, latent_semaphore,
        latent_payload_value, condition_semaphore, condition_payload_value,
        branch_done_semaphore, branch_payload_value, diagnostics_sink);
    branch_ref.was_issued = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_wait_one(branch_done_semaphore,
                                               branch_payload_value);
  }
  status = iree_status_join(
      status, id4_ideogram4_generation_release_stage_bundle_ref(&branch_ref));
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_sampler_stage_serial(
    id4_ideogram4_generation_execution_t* execution, uint64_t payload_value,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_bundle_ref_t sampler_ref = {0};
  iree_status_t status = id4_ideogram4_generation_acquire_stage_bundle_ref(
      execution->bundle, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
      iree_hal_semaphore_list_empty(), kernel_diagnostic_artifact_flags,
      diagnostics_sink, &sampler_ref);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_sampler(
        execution, sampler_ref.bundle, payload_value, diagnostics_sink);
    sampler_ref.was_issued = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_wait_one(
        execution->sampler_done_semaphore, payload_value);
  }
  status = iree_status_join(
      status, id4_ideogram4_generation_release_stage_bundle_ref(&sampler_ref));
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_decode_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_bundle_ref_t decode_ref = {0};
  iree_status_t status = id4_ideogram4_generation_acquire_stage_bundle_ref(
      execution->bundle, ID4_IDEOGRAM4_GENERATION_STAGE_DECODE,
      iree_hal_semaphore_list_empty(), kernel_diagnostic_artifact_flags,
      diagnostics_sink, &decode_ref);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_decode(
        execution, decode_ref.bundle, execution->denoise_schedule.step_count,
        final_signal_list, diagnostics_sink);
    decode_ref.was_issued = iree_status_is_ok(status);
  }
  status = iree_status_join(
      status, id4_ideogram4_generation_release_stage_bundle_ref(&decode_ref));
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_conditioning_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_issue_qwen_stage_serial(
      execution, kernel_diagnostic_artifact_flags, diagnostics_sink));
  return id4_ideogram4_generation_issue_noise_stage_serial(
      execution, kernel_diagnostic_artifact_flags, diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_denoise_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < execution->denoise_schedule.step_count && iree_status_is_ok(status);
       ++i) {
    const uint64_t payload_value = (uint64_t)i + 1;
    iree_hal_semaphore_t* latent_semaphore =
        i == 0 ? execution->noise_done_semaphore
               : execution->sampler_done_semaphore;
    const uint64_t latent_payload_value = i == 0 ? 1 : (uint64_t)i;
    if (i > 0) {
      status = id4_ideogram4_generation_chain_upload_after_sampler(execution,
                                                                   (uint64_t)i);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_upload_denoise_step(
            execution, &execution->denoise_schedule.steps[i]);
      }
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_dit_branch_stage_serial(
          execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
          latent_semaphore, latent_payload_value,
          execution->qwen_done_semaphore, 1,
          execution->dit_conditioned_done_semaphore, payload_value,
          kernel_diagnostic_artifact_flags, diagnostics_sink);
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_dit_branch_stage_serial(
          execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
          latent_semaphore, latent_payload_value, NULL, 0,
          execution->dit_unconditioned_done_semaphore, payload_value,
          kernel_diagnostic_artifact_flags, diagnostics_sink);
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_sampler_stage_serial(
          execution, payload_value, kernel_diagnostic_artifact_flags,
          diagnostics_sink);
    }
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_issue_conditioning_stage_serial(
      execution, kernel_diagnostic_artifact_flags, diagnostics_sink));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_issue_denoise_stage_serial(
      execution, kernel_diagnostic_artifact_flags, diagnostics_sink));
  return id4_ideogram4_generation_issue_decode_stage_serial(
      execution, final_signal_list, kernel_diagnostic_artifact_flags,
      diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_denoise_phase(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < execution->denoise_schedule.step_count && iree_status_is_ok(status);
       ++i) {
    const uint64_t payload_value = (uint64_t)i + 1;
    iree_hal_semaphore_t* latent_semaphore =
        i == 0 ? execution->noise_done_semaphore
               : execution->sampler_done_semaphore;
    const uint64_t latent_payload_value = i == 0 ? 1 : (uint64_t)i;
    if (i > 0) {
      status = id4_ideogram4_generation_chain_upload_after_sampler(execution,
                                                                   (uint64_t)i);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_upload_denoise_step(
            execution, &execution->denoise_schedule.steps[i]);
      }
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_dit_branch(
          execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
          id4_ideogram4_generation_phase_stage_bundle(
              phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED),
          latent_semaphore, latent_payload_value,
          execution->qwen_done_semaphore, 1,
          execution->dit_conditioned_done_semaphore, payload_value,
          diagnostics_sink);
      phase_bundle
          ->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED]
          .was_issued |= iree_status_is_ok(status);
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_dit_branch(
          execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
          id4_ideogram4_generation_phase_stage_bundle(
              phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED),
          latent_semaphore, latent_payload_value, NULL, 0,
          execution->dit_unconditioned_done_semaphore, payload_value,
          diagnostics_sink);
      phase_bundle
          ->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED]
          .was_issued |= iree_status_is_ok(status);
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_sampler(
          execution,
          id4_ideogram4_generation_phase_stage_bundle(
              phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER),
          payload_value, diagnostics_sink);
      phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER]
          .was_issued |= iree_status_is_ok(status);
    }
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_decode_phase(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = id4_ideogram4_generation_issue_decode(
      execution,
      id4_ideogram4_generation_phase_stage_bundle(
          phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_DECODE),
      execution->denoise_schedule.step_count, final_signal_list,
      diagnostics_sink);
  phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_DECODE]
      .was_issued = iree_status_is_ok(status);
  return status;
}

static iree_status_t id4_ideogram4_generation_signal_conditioning_phase(
    id4_ideogram4_generation_execution_t* execution,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_hal_semaphore_t* wait_semaphores[2];
  uint64_t wait_payload_values[2];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count,
                                     execution->qwen_done_semaphore, 1);
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count,
                                     execution->noise_done_semaphore, 1);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  return iree_hal_device_queue_barrier(
      execution->bundle->device, execution->bundle->queue_affinity, wait_list,
      signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_signal_denoise_phase(
    id4_ideogram4_generation_execution_t* execution,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_hal_semaphore_t* wait_semaphore = NULL;
  uint64_t wait_payload_value = execution->denoise_schedule.step_count;
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_single_semaphore_list(
      &wait_semaphore, &wait_payload_value, execution->sampler_done_semaphore,
      wait_payload_value);
  return iree_hal_device_queue_barrier(
      execution->bundle->device, execution->bundle->queue_affinity, wait_list,
      signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

iree_status_t id4_ideogram4_generation_execution_issue_phase(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    const id4_ideogram4_generation_phase_issue_options_t* options) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_phase_issue_options(
      execution, phase_bundle, options));
  execution->parameter_load_prefetch_region_distance =
      options->parameter_load_prefetch_region_distance;
  iree_status_t status = id4_ideogram4_generation_chain_phase_issue_wait(
      execution, phase_bundle->phase_mask, options->wait_semaphore_list);
  if (!iree_status_is_ok(status)) return status;
  switch (phase_bundle->phase_mask) {
    case ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING:
      status = id4_ideogram4_generation_issue_conditioning_phase(
          execution, phase_bundle, options->diagnostics_sink);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_signal_conditioning_phase(
            execution, options->signal_semaphore_list);
      }
      return status;
    case ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE:
      status = id4_ideogram4_generation_issue_denoise_phase(
          execution, phase_bundle, options->diagnostics_sink);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_signal_denoise_phase(
            execution, options->signal_semaphore_list);
      }
      return status;
    case ID4_IDEOGRAM4_GENERATION_PHASE_DECODE:
      return id4_ideogram4_generation_issue_decode_phase(
          execution, phase_bundle, options->signal_semaphore_list,
          options->diagnostics_sink);
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation phase mask 0x%x does not identify one "
          "generation phase",
          phase_bundle->phase_mask);
  }
}

iree_status_t id4_ideogram4_session_issue_generation(
    id4_ideogram4_session_t* session, id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_issue_options_t* options,
    id4_ideogram4_generation_execution_t** out_execution) {
  IREE_ASSERT_ARGUMENT(out_execution);
  *out_execution = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_issue_options(
      session, bundle, options));

  id4_ideogram4_generation_execution_t* execution = NULL;
  if (options->issue_policy ==
      ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_STAGE_SERIAL) {
    iree_status_t status = id4_ideogram4_generation_begin_execution(
        session, bundle, options->request, options->tokenizer,
        options->tokenizer_flags, options->wait_semaphore_list,
        iree_hal_semaphore_list_empty(), &execution);
    if (iree_status_is_ok(status)) {
      execution->parameter_load_prefetch_region_distance =
          options->parameter_load_prefetch_region_distance;
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_stage_serial(
          execution, options->signal_semaphore_list,
          options->kernel_diagnostic_artifact_flags, options->diagnostics_sink);
    }
    if (iree_status_is_ok(status)) {
      *out_execution = execution;
    } else {
      id4_ideogram4_generation_execution_release(execution);
    }
    return status;
  }

  id4_ideogram4_generation_phase_bundle_t conditioning_phase = {0};
  id4_ideogram4_generation_phase_bundle_initialize(
      ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING, &conditioning_phase);
  id4_ideogram4_generation_phase_bundle_t denoise_phase = {0};
  id4_ideogram4_generation_phase_bundle_initialize(
      ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE, &denoise_phase);
  id4_ideogram4_generation_phase_bundle_t decode_phase = {0};
  id4_ideogram4_generation_phase_bundle_initialize(
      ID4_IDEOGRAM4_GENERATION_PHASE_DECODE, &decode_phase);

  iree_status_t status = id4_ideogram4_generation_begin_execution(
      session, bundle, options->request, options->tokenizer,
      options->tokenizer_flags, options->wait_semaphore_list,
      iree_hal_semaphore_list_empty(), &execution);
  if (iree_status_is_ok(status)) {
    execution->parameter_load_prefetch_region_distance =
        options->parameter_load_prefetch_region_distance;
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_prepare_phase_bundle(
        bundle, ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING,
        iree_hal_semaphore_list_empty(),
        options->kernel_diagnostic_artifact_flags, options->diagnostics_sink,
        &conditioning_phase);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_conditioning_phase(
        execution, &conditioning_phase, options->diagnostics_sink);
  }
  status = iree_status_join(
      status,
      id4_ideogram4_generation_phase_bundle_deinitialize(&conditioning_phase));
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_wait_conditioning_phase(execution);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_prepare_phase_bundle(
        bundle, ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE,
        iree_hal_semaphore_list_empty(),
        options->kernel_diagnostic_artifact_flags, options->diagnostics_sink,
        &denoise_phase);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_denoise_phase(
        execution, &denoise_phase, options->diagnostics_sink);
  }
  status = iree_status_join(
      status,
      id4_ideogram4_generation_phase_bundle_deinitialize(&denoise_phase));
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_prepare_phase_bundle(
        bundle, ID4_IDEOGRAM4_GENERATION_PHASE_DECODE,
        iree_hal_semaphore_list_empty(),
        options->kernel_diagnostic_artifact_flags, options->diagnostics_sink,
        &decode_phase);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_decode_phase(
        execution, &decode_phase, options->signal_semaphore_list,
        options->diagnostics_sink);
  }
  status = iree_status_join(
      status,
      id4_ideogram4_generation_phase_bundle_deinitialize(&decode_phase));
  status = iree_status_join(
      status,
      id4_ideogram4_generation_phase_bundle_deinitialize(&conditioning_phase));
  status = iree_status_join(
      status,
      id4_ideogram4_generation_phase_bundle_deinitialize(&denoise_phase));
  if (iree_status_is_ok(status)) {
    *out_execution = execution;
  } else {
    id4_ideogram4_generation_execution_release(execution);
  }
  return status;
}

void id4_ideogram4_generation_execution_release(
    id4_ideogram4_generation_execution_t* execution) {
  if (!execution) return;
  id4_ideogram4_denoise_schedule_deinitialize(&execution->denoise_schedule,
                                              execution->host_allocator);
  id4_ideogram4_dit_inputs_deinitialize(&execution->dit_inputs,
                                        execution->host_allocator);
  id4_ideogram4_qwen_inputs_deinitialize(&execution->qwen_inputs,
                                         execution->host_allocator);
  iree_hal_semaphore_release(execution->decode_done_semaphore);
  iree_hal_semaphore_release(execution->sampler_done_semaphore);
  iree_hal_semaphore_release(execution->dit_unconditioned_done_semaphore);
  iree_hal_semaphore_release(execution->dit_conditioned_done_semaphore);
  iree_hal_semaphore_release(execution->noise_done_semaphore);
  iree_hal_semaphore_release(execution->qwen_done_semaphore);
  iree_hal_semaphore_release(execution->upload_semaphore);
  id4_ideogram4_generation_bundle_release(execution->bundle);
  iree_allocator_t host_allocator = execution->host_allocator;
  iree_allocator_free(host_allocator, execution);
}

iree_status_t id4_ideogram4_generation_execution_result(
    const id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_result_t* out_result) {
  if (!execution || !out_result) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation execution and result output are required");
  }
  out_result->conditioned_velocity_binding =
      execution->conditioned_velocity_binding;
  out_result->unconditioned_velocity_binding =
      execution->unconditioned_velocity_binding;
  out_result->denoised_latent_binding = execution->denoised_latent_binding;
  out_result->final_latent_binding = execution->final_latent_binding;
  out_result->decoded_image_binding = execution->decoded_image_binding;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_find_diagnostic_tap_plan(
    const id4_pipeline_plan_t* plan, iree_string_view_t tap_name,
    const id4_pipeline_diagnostic_tap_plan_t** out_tap) {
  *out_tap = NULL;
  const iree_host_size_t tap_count =
      id4_pipeline_plan_diagnostic_tap_count(plan);
  for (iree_host_size_t i = 0; i < tap_count; ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    if (tap && iree_string_view_equal(tap->name, tap_name)) {
      *out_tap = tap;
      return iree_ok_status();
    }
  }
  iree_string_view_t stage_name = id4_pipeline_plan_stage_name(plan);
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "Ideogram 4 generation stage %.*s has no diagnostic tap `%.*s`",
      (int)stage_name.size, stage_name.data, (int)tap_name.size, tap_name.data);
}

iree_status_t id4_ideogram4_generation_execution_find_diagnostic_tap(
    const id4_ideogram4_generation_execution_t* execution,
    iree_string_view_t stage_key, iree_string_view_t tap_name,
    const id4_pipeline_tensor_layout_t** out_layout,
    iree_hal_buffer_binding_t* out_binding) {
  if (!execution || iree_string_view_is_empty(stage_key) ||
      iree_string_view_is_empty(tap_name) || !out_layout || !out_binding) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation diagnostic tap lookup requires execution, "
        "stage key, tap name, layout output, and binding output");
  }
  *out_layout = NULL;
  memset(out_binding, 0, sizeof(*out_binding));
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor_for_key(stage_key);
  if (!descriptor) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "Ideogram 4 generation has no stage `%.*s`",
                            (int)stage_key.size, stage_key.data);
  }
  const id4_ideogram4_generation_stage_slot_t* slot =
      &execution->bundle->stages[descriptor->ordinal];
  const id4_pipeline_diagnostic_tap_plan_t* tap = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_find_diagnostic_tap_plan(
      slot->plan, tap_name, &tap));
  IREE_RETURN_IF_ERROR(id4_pipeline_find_diagnostic_tap_binding(
      slot->plan, &slot->diagnostic_tap_bindings, tap_name, out_binding));
  *out_layout = &tap->layout;
  return iree_ok_status();
}
