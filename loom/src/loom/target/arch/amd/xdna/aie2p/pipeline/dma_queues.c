// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_queues.h"

#include "loom/analysis/pipeline/lifetime_layout.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/input.h"

typedef struct loom_aie2p_native_shim_queue_t {
  // Physical shim selected for one complete logical queue lifetime class.
  loom_aie2p_native_tile_t* tile;
  // Queue index in the selected transfer direction.
  uint8_t engine;
} loom_aie2p_native_shim_queue_t;

// Shim placement is independent of the issuing core and local payload storage.
// Preserve preferred-column demand before placing overflow, then use the
// nearest available column. Lower columns win ties to limit partition growth.
static loom_aie2p_native_tile_t* loom_aie2p_native_find_shim_queue(
    loom_aie2p_native_context_t* context, uint16_t preferred, bool receive) {
  for (uint16_t distance = 1; distance < context->family->column_count;
       ++distance) {
    for (unsigned side = 0; side < 2; ++side) {
      const int column = preferred + (side ? distance : -distance);
      if (column < 0 || column >= context->family->column_count) {
        continue;
      }
      loom_aie2p_native_tile_t* tile =
          &context->tiles[column * context->family->row_count];
      const uint8_t count = receive ? tile->s2mm_count : tile->mm2s_count;
      if (count < tile->facts->dma.channel_count_per_direction) {
        return tile;
      }
    }
  }
  return NULL;
}

static void loom_aie2p_native_remap_shim_queue(
    loom_aie2p_native_shim_queue_t* const* remaps,
    loom_aie2p_native_tile_t** shim, bool receive, uint8_t* engine) {
  const uint8_t capacity = (*shim)->facts->dma.channel_count_per_direction;
  if (*engine < capacity) {
    return;
  }
  const loom_aie2p_native_shim_queue_t* selected =
      &remaps[(*shim)->coordinate.column * 2 + receive][*engine - capacity];
  *shim = selected->tile;
  *engine = selected->engine;
}

static void loom_aie2p_native_request_dma_queue(
    const loom_aie2p_native_context_t* context, iree_host_size_t lifetime,
    const loom_aie2p_native_tile_t* tile, bool receive, uint8_t* engine,
    loom_pipeline_lifetime_request_t* requests, uint8_t** engines,
    iree_host_size_t* count) {
  requests[*count] = (loom_pipeline_lifetime_request_t){
      .lifetime = lifetime,
      .pool = (tile - context->tiles) * 2 + receive,
      .length = 1,
      .alignment = 1};
  engines[(*count)++] = engine;
}

iree_status_t loom_aie2p_native_assign_dma_queues(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    const iree_host_size_t* distribution_workers,
    iree_host_size_t distribution_count, bool* out_valid) {
  *out_valid = false;
  if (!context->dma_queue_count) {
    *out_valid = true;
    return iree_ok_status();
  }
  loom_pipeline_lifetime_request_t* requests;
  uint8_t** engines;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->pass->arena, context->dma_queue_count,
                                sizeof(*requests), (void**)&requests));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->pass->arena, context->dma_queue_count,
                                sizeof(*engines), (void**)&engines));
  const loom_pipeline_lifetimes_t* lifetimes =
      &realization->invocations.lifetimes;
  iree_host_size_t count = 0;
  for (iree_host_size_t d = 0; d < distribution_count; ++d) {
    const iree_host_size_t i = distribution_workers[d];
    loom_aie2p_native_worker_t* worker = &context->workers[i];
    const iree_host_size_t lifetime =
        lifetimes
            ->owners[realization->invocations.workers[i].invocation->ordinal];
    for (loom_aie2p_native_distribution_t* distribution =
             worker->distribution_plan->geometries;
         distribution; distribution = distribution->next_demand) {
      if (distribution->path->first != distribution) {
        continue;
      }
      loom_aie2p_native_request_dma_queue(
          context, lifetime, distribution->sender.tile, false,
          &distribution->sender.engine, requests, engines, &count);
      for (iree_host_size_t j = 0; j < distribution->receiver_count; ++j) {
        loom_aie2p_native_distribution_endpoint_t* receiver =
            &distribution->receivers[j];
        loom_aie2p_native_request_dma_queue(context, lifetime, receiver->tile,
                                            true, &receiver->engine, requests,
                                            engines, &count);
      }
    }
  }
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    loom_aie2p_native_worker_t* worker = &context->workers[i];
    const iree_host_size_t lifetime =
        lifetimes
            ->owners[realization->invocations.workers[i].invocation->ordinal];
    for (loom_aie2p_native_dma_path_t* path = worker->transfer_paths; path;
         path = path->next_demand) {
      loom_aie2p_native_request_dma_queue(context, lifetime, path->local,
                                          path->ingress, &path->local_engine,
                                          requests, engines, &count);
      loom_aie2p_native_request_dma_queue(context, lifetime, path->shim,
                                          !path->ingress, &path->shim_engine,
                                          requests, engines, &count);
    }
    loom_aie2p_native_input_t* input = worker->input;
    if (input) {
      loom_aie2p_native_request_dma_queue(context, lifetime, input->path->shim,
                                          false, &input->path->input_engine,
                                          requests, engines, &count);
      const iree_host_size_t receiver_count =
          input->window_capacity ? 1 : input->source->phase_count;
      for (iree_host_size_t j = 0; j < receiver_count; ++j) {
        loom_aie2p_native_input_phase_t* phase = &input->phases[j];
        loom_aie2p_native_request_dma_queue(
            context, lifetime,
            context->pool_tiles[phase->destination->pool_index], true,
            &phase->receiver.engine, requests, engines, &count);
      }
    }
  }
  const iree_host_size_t tile_count =
      (iree_host_size_t)context->family->column_count *
      context->family->row_count;
  const loom_pipeline_lifetime_extent_t* extents;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_lifetime_layout(lifetimes, requests, count, tile_count * 2,
                                    context->pass->arena, &extents));
  iree_host_size_t overflow_count = 0;
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    loom_aie2p_native_tile_t* tile = &context->tiles[i];
    const uint8_t capacity = tile->facts->dma.channel_count_per_direction;
    if (tile->facts->kind == LOOM_XDNA_TILE_KIND_SHIM_NOC) {
      for (unsigned direction = 0; direction < 2; ++direction) {
        const uint64_t required = extents[i * 2 + direction].length;
        overflow_count += required > capacity ? required - capacity : 0;
      }
    } else if (extents[i * 2].length > capacity ||
               extents[i * 2 + 1].length > capacity) {
      return loom_aie2p_native_reject(
          context, realization->function.op,
          IREE_SV(
              "available DMA queues for simultaneous construction lifetimes"));
    }
    tile->mm2s_count = (uint8_t)iree_min(extents[i * 2].length, capacity);
    tile->s2mm_count = (uint8_t)iree_min(extents[i * 2 + 1].length, capacity);
  }
  loom_aie2p_native_shim_queue_t** remaps = NULL;
  if (overflow_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, context->family->column_count * 2,
        sizeof(*remaps), (void**)&remaps));
    loom_aie2p_native_shim_queue_t* next;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, overflow_count, sizeof(*next), (void**)&next));
    for (uint16_t column = 0; column < context->family->column_count;
         ++column) {
      const iree_host_size_t tile_index = column * context->family->row_count;
      const uint8_t capacity =
          context->tiles[tile_index].facts->dma.channel_count_per_direction;
      for (unsigned direction = 0; direction < 2; ++direction) {
        remaps[column * 2 + direction] = next;
        const uint64_t required = extents[tile_index * 2 + direction].length;
        for (uint64_t engine = capacity; engine < required; ++engine) {
          loom_aie2p_native_tile_t* shim = loom_aie2p_native_find_shim_queue(
              context, column, direction != 0);
          if (!shim) {
            return loom_aie2p_native_reject(
                context, realization->function.op,
                IREE_SV("available DMA queues for simultaneous construction "
                        "lifetimes"));
          }
          uint8_t* assigned = direction ? &shim->s2mm_count : &shim->mm2s_count;
          *next++ = (loom_aie2p_native_shim_queue_t){.tile = shim,
                                                     .engine = (*assigned)++};
        }
      }
    }
  }
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    loom_aie2p_native_tile_t* tile = &context->tiles[i];
    for (unsigned direction = 0; direction < 2; ++direction) {
      const iree_host_size_t queues =
          direction ? tile->s2mm_count : tile->mm2s_count;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          context->pass->arena, queues, sizeof(*tile->dma_controls[direction]),
          (void**)&tile->dma_controls[direction]));
      for (iree_host_size_t queue = 0; queue < queues; ++queue) {
        tile->dma_controls[direction][queue] =
            (loom_aie2p_native_dma_control_t){.controller = UINT8_MAX};
      }
    }
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    *engines[i] = (uint8_t)requests[i].offset;
  }
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    loom_aie2p_native_worker_t* worker = &context->workers[i];
    if (remaps) {
      for (loom_aie2p_native_dma_path_t* path = worker->transfer_paths; path;
           path = path->next_demand) {
        loom_aie2p_native_remap_shim_queue(remaps, &path->shim, !path->ingress,
                                           &path->shim_engine);
      }
      if (worker->input) {
        loom_aie2p_native_input_path_t* path = worker->input->path;
        loom_aie2p_native_remap_shim_queue(remaps, &path->shim, false,
                                           &path->input_engine);
      }
    }
    for (loom_aie2p_native_distribution_t* distribution =
             worker->distribution_plan ? worker->distribution_plan->geometries
                                       : NULL;
         distribution; distribution = distribution->next_demand) {
      const loom_aie2p_native_distribution_t* first = distribution->path->first;
      distribution->sender.engine = first->sender.engine;
      for (iree_host_size_t j = 0; j < distribution->receiver_count; ++j) {
        distribution->receivers[j].engine = first->receivers[j].engine;
      }
    }
    loom_aie2p_native_input_t* input = worker->input;
    if (input && input->window_capacity) {
      for (iree_host_size_t j = 1; j < input->source->phase_count; ++j) {
        input->phases[j].receiver.engine = input->phases[0].receiver.engine;
      }
    }
  }
  *out_valid = true;
  return iree_ok_status();
}

static void loom_aie2p_native_select_dma_controller(
    loom_aie2p_native_tile_t* tile, bool receive, uint8_t engine,
    uint8_t controller) {
  loom_aie2p_native_dma_control_t* control =
      &tile->dma_controls[receive][engine];
  if (control->controller == UINT8_MAX) {
    control->controller = controller;
  } else {
    control->dynamic |= control->controller != controller;
  }
}

void loom_aie2p_native_select_dma_controllers(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization) {
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    for (const loom_aie2p_native_dma_path_t* path = context->workers[i].paths;
         path; path = path->next) {
      if (!path->completion_tokens) {
        continue;
      }
      loom_aie2p_native_select_dma_controller(path->shim, !path->ingress,
                                              path->shim_engine,
                                              path->completion_packet);
      if (path->completion_tokens == 2) {
        loom_aie2p_native_select_dma_controller(path->local, path->ingress,
                                                path->local_engine,
                                                path->completion_packet);
      }
    }
  }
}

loom_xdna_register_field_id_t loom_aie2p_native_dma_controller_field(
    const loom_aie2p_native_tile_t* tile, loom_xdna_dma_direction_t direction) {
  if (tile->memory_layout) {
    return tile->memory_layout->directions[direction - 1].controller;
  }
  return direction == LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM
             ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_CONTROL_CONTROLLER_ID
             : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_CONTROL_CONTROLLER_ID;
}
