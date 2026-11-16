// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_queues.h"

#include "iree/base/internal/math.h"
#include "loom/analysis/pipeline/lifetime_layout.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/input.h"
#include "loom/util/adaptive_sort.h"

typedef struct loom_aie2p_native_dma_queue_binding_t {
  // Demand field receiving the lifetime allocator's logical queue index.
  uint8_t* engine;
  // Physical core issuing shim commands, or IREE_HOST_SIZE_MAX for local or
  // configuration-owned endpoints that need no core-to-shim control route.
  iree_host_size_t controller;
} loom_aie2p_native_dma_queue_binding_t;

typedef struct loom_aie2p_native_shim_queue_t {
  // Physical shim selected for one complete logical queue lifetime class.
  loom_aie2p_native_tile_t* tile;
  // Queue index in the selected transfer direction.
  uint8_t engine;
} loom_aie2p_native_shim_queue_t;

typedef struct loom_aie2p_native_receive_queue_t {
  // Receiver whose queue preference follows its physical sender.
  loom_aie2p_native_distribution_endpoint_t* receiver;
  // Independently allocated memory-to-stream endpoint.
  const loom_aie2p_native_distribution_endpoint_t* sender;
  // Construction owning this receive demand.
  iree_host_size_t lifetime;
  // Original distribution priority, preserving ties deterministically.
  iree_host_size_t ordinal;
} loom_aie2p_native_receive_queue_t;

static bool loom_aie2p_native_receive_queue_before(
    const loom_aie2p_native_receive_queue_t* a,
    const loom_aie2p_native_receive_queue_t* b) {
  if (a->receiver->tile != b->receiver->tile) {
    return a->receiver->tile < b->receiver->tile;
  }
  const uint16_t column = a->receiver->tile->coordinate.column;
  const uint16_t first = a->sender->tile->coordinate.column;
  const uint16_t second = b->sender->tile->coordinate.column;
  if ((first == column) != (second == column)) {
    return first == column;
  }
  if (a->sender->engine != b->sender->engine) {
    return a->sender->engine < b->sender->engine;
  }
  const uint16_t da = first > column ? first - column : column - first;
  const uint16_t db = second > column ? second - column : column - second;
  if (da != db) {
    return da < db;
  }
  if (a->sender->tile != b->sender->tile) {
    return a->sender->tile < b->sender->tile;
  }
  return a->ordinal < b->ordinal;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_aie2p_native_sort_receive_queues,
                          loom_aie2p_native_receive_queue_t,
                          loom_aie2p_native_receive_queue_before)

// Shim placement is independent of the issuing core and local payload storage.
// Preserve preferred-column demand before placing overflow, then use the
// nearest column whose payload and command traffic fit its southbound links.
// The cost is a pressure estimate: physical routing still owns admission, since
// a payload can also descend through its source column and turn at the shim.
// Lower columns win ties to limit partition growth.
static loom_aie2p_native_tile_t* loom_aie2p_native_find_shim_queue(
    loom_aie2p_native_context_t* context, uint16_t preferred, bool receive,
    const uint64_t* controllers, const uint64_t* column_controllers,
    iree_host_size_t word_count, uint8_t southbound_capacity) {
  loom_aie2p_native_tile_t* best = NULL;
  iree_host_size_t best_excess = IREE_HOST_SIZE_MAX;
  for (uint16_t distance = 1; distance < context->family->column_count;
       ++distance) {
    for (unsigned side = 0; side < 2; ++side) {
      const int column = preferred + (side ? distance : -distance);
      if (column < 0 || column >= context->family->column_count) {
        continue;
      }
      loom_aie2p_native_tile_t* tile =
          &context->tiles[loom_xdna_array_physical_tile_index(
              context->family,
              (loom_xdna_tile_coordinate_t){(uint16_t)column, 0})];
      const uint8_t count = receive ? tile->s2mm_count : tile->mm2s_count;
      if (count >= tile->facts->dma.channel_count_per_direction) {
        continue;
      }
      iree_host_size_t demand = tile->s2mm_count + receive;
      for (iree_host_size_t word = 0; word < word_count; ++word) {
        demand += iree_math_count_ones_u64(
            column_controllers[column * word_count + word] | controllers[word]);
      }
      if (demand <= southbound_capacity) {
        return tile;
      }
      const iree_host_size_t excess = demand - southbound_capacity;
      if (excess < best_excess) {
        best = tile;
        best_excess = excess;
      }
    }
  }
  return best;
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
    iree_host_size_t controller, loom_pipeline_lifetime_request_t* requests,
    loom_aie2p_native_dma_queue_binding_t* bindings, iree_host_size_t* count) {
  requests[*count] = (loom_pipeline_lifetime_request_t){
      .lifetime = lifetime,
      .pool = (tile - context->tiles) * 2 + receive,
      .length = 1,
      .alignment = 1};
  bindings[(*count)++] = (loom_aie2p_native_dma_queue_binding_t){
      .engine = engine, .controller = controller};
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
  loom_aie2p_native_dma_queue_binding_t* bindings;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->pass->arena, context->dma_queue_count,
                                sizeof(*requests), (void**)&requests));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->pass->arena, context->dma_queue_count,
                                sizeof(*bindings), (void**)&bindings));
  const loom_pipeline_lifetimes_t* lifetimes =
      &realization->invocations.lifetimes;
  iree_host_size_t count = 0;
  const iree_host_size_t tile_count =
      loom_xdna_array_physical_tile_count(context->family);
  const loom_pipeline_lifetime_extent_t* extents[2];
  // Send and receive directions are independent allocation pools. Establish
  // physical senders first, in producer order, so serial receivers can preserve
  // source identity instead of connecting independent outputs through an
  // arbiter merely because different phases enumerate their receivers alike.
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    loom_aie2p_native_worker_t* worker = &context->workers[i];
    const iree_host_size_t lifetime =
        lifetimes
            ->owners[realization->invocations.workers[i].invocation->ordinal];
    for (loom_aie2p_native_distribution_t* distribution =
             worker->distribution_plan ? worker->distribution_plan->geometries
                                       : NULL;
         distribution; distribution = distribution->next_demand) {
      if (distribution->path->first == distribution) {
        loom_aie2p_native_request_dma_queue(
            context, lifetime, distribution->sender.tile, false,
            &distribution->sender.engine, IREE_HOST_SIZE_MAX, requests,
            bindings, &count);
      }
    }
    for (loom_aie2p_native_dma_path_t* path = worker->transfer_paths; path;
         path = path->next_demand) {
      loom_aie2p_native_request_dma_queue(
          context, lifetime, path->ingress ? path->shim : path->local, false,
          path->ingress ? &path->shim_engine : &path->local_engine,
          path->ingress && worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_CORE
              ? context->core_engines[i]
              : IREE_HOST_SIZE_MAX,
          requests, bindings, &count);
    }
    loom_aie2p_native_input_t* input = worker->input;
    if (input) {
      loom_aie2p_native_request_dma_queue(
          context, lifetime, input->path->shim, false,
          &input->path->input_engine, context->core_engines[input->controller],
          requests, bindings, &count);
    }
  }
  const iree_host_size_t sender_count = count;
  IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_layout(
      lifetimes, requests, sender_count, tile_count * 2, context->pass->arena,
      &extents[0]));
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    const loom_aie2p_native_tile_t* tile = &context->tiles[i];
    if (tile->facts->kind != LOOM_XDNA_TILE_KIND_SHIM_NOC &&
        extents[0][i * 2].length >
            tile->facts->dma.channel_count_per_direction) {
      return loom_aie2p_native_reject(
          context, realization->function.op,
          IREE_SV(
              "available DMA queues for simultaneous construction lifetimes"));
    }
  }
  for (iree_host_size_t i = 0; i < sender_count; ++i) {
    // Shim overflow is placed below. On-chip senders are now fully assigned.
    if (context->tiles[requests[i].pool / 2].facts->kind !=
        LOOM_XDNA_TILE_KIND_SHIM_NOC) {
      *bindings[i].engine = (uint8_t)requests[i].offset;
    }
  }
  loom_aie2p_native_receive_queue_t* receivers;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, context->dma_queue_count - sender_count,
      sizeof(*receivers), (void**)&receivers));
  iree_host_size_t receiver_count = 0;
  for (iree_host_size_t d = 0; d < distribution_count; ++d) {
    const iree_host_size_t i = distribution_workers[d];
    const iree_host_size_t lifetime =
        lifetimes
            ->owners[realization->invocations.workers[i].invocation->ordinal];
    for (loom_aie2p_native_distribution_t* distribution =
             context->workers[i].distribution_plan->geometries;
         distribution; distribution = distribution->next_demand) {
      if (distribution->path->first != distribution) {
        continue;
      }
      for (iree_host_size_t j = 0; j < distribution->receiver_count; ++j) {
        receivers[receiver_count] = (loom_aie2p_native_receive_queue_t){
            .receiver = &distribution->receivers[j],
            .sender = &distribution->sender,
            .lifetime = lifetime,
            .ordinal = receiver_count};
        ++receiver_count;
      }
    }
  }
  // Prefer the local column's source queues, then crossing senders. Common
  // lifetime layout preserves this preference while keeping concurrent users
  // distinct. Unphased pipelines retain their original distribution priority.
  if (lifetimes->has_serial) {
    loom_aie2p_native_sort_receive_queues(receivers, receiver_count);
  }
  for (iree_host_size_t i = 0; i < receiver_count; ++i) {
    loom_aie2p_native_distribution_endpoint_t* receiver = receivers[i].receiver;
    loom_aie2p_native_request_dma_queue(
        context, receivers[i].lifetime, receiver->tile, true, &receiver->engine,
        IREE_HOST_SIZE_MAX, requests, bindings, &count);
  }
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    loom_aie2p_native_worker_t* worker = &context->workers[i];
    const iree_host_size_t lifetime =
        lifetimes
            ->owners[realization->invocations.workers[i].invocation->ordinal];
    for (loom_aie2p_native_dma_path_t* path = worker->transfer_paths; path;
         path = path->next_demand) {
      loom_aie2p_native_request_dma_queue(
          context, lifetime, path->ingress ? path->local : path->shim, true,
          path->ingress ? &path->local_engine : &path->shim_engine,
          !path->ingress &&
                  worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_CORE
              ? context->core_engines[i]
              : IREE_HOST_SIZE_MAX,
          requests, bindings, &count);
    }
    loom_aie2p_native_input_t* input = worker->input;
    if (input) {
      const iree_host_size_t phase_count =
          input->window_capacity ? 1 : input->source->phase_count;
      for (iree_host_size_t j = 0; j < phase_count; ++j) {
        loom_aie2p_native_input_phase_t* phase = &input->phases[j];
        loom_aie2p_native_request_dma_queue(
            context, lifetime,
            context->pool_tiles[phase->destination->pool_index], true,
            &phase->receiver.engine, IREE_HOST_SIZE_MAX, requests, bindings,
            &count);
      }
    }
  }
  IREE_RETURN_IF_ERROR(loom_pipeline_lifetime_layout(
      lifetimes, requests + sender_count, count - sender_count, tile_count * 2,
      context->pass->arena, &extents[1]));
  iree_host_size_t overflow_count = 0;
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    loom_aie2p_native_tile_t* tile = &context->tiles[i];
    const uint8_t capacity = tile->facts->dma.channel_count_per_direction;
    if (tile->facts->kind == LOOM_XDNA_TILE_KIND_SHIM_NOC) {
      for (unsigned direction = 0; direction < 2; ++direction) {
        const uint64_t required = extents[direction][i * 2 + direction].length;
        overflow_count += required > capacity ? required - capacity : 0;
      }
    } else if (extents[1][i * 2 + 1].length > capacity) {
      return loom_aie2p_native_reject(
          context, realization->function.op,
          IREE_SV(
              "available DMA queues for simultaneous construction lifetimes"));
    }
    tile->mm2s_count = (uint8_t)iree_min(extents[0][i * 2].length, capacity);
    tile->s2mm_count =
        (uint8_t)iree_min(extents[1][i * 2 + 1].length, capacity);
  }
  loom_aie2p_native_shim_queue_t** remaps = NULL;
  if (overflow_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, context->family->column_count * 2,
        sizeof(*remaps), (void**)&remaps));
    loom_aie2p_native_shim_queue_t* overflow;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(context->pass->arena, overflow_count,
                                  sizeof(*overflow), (void**)&overflow));
    loom_aie2p_native_shim_queue_t* next = overflow;
    for (uint16_t column = 0; column < context->family->column_count;
         ++column) {
      const iree_host_size_t tile_index = loom_xdna_array_physical_tile_index(
          context->family, (loom_xdna_tile_coordinate_t){column, 0});
      const uint8_t capacity =
          context->tiles[tile_index].facts->dma.channel_count_per_direction;
      for (unsigned direction = 0; direction < 2; ++direction) {
        remaps[column * 2 + direction] = next;
        const uint64_t required =
            extents[direction][tile_index * 2 + direction].length;
        next += required > capacity ? required - capacity : 0;
      }
    }
    // Requests already retain their physical command source. Preserve the
    // union for every shared lifetime class, then merge it at final placement.
    const iree_host_size_t word_count = (tile_count + 63) / 64;
    const iree_host_size_t mask_count =
        context->family->column_count + overflow_count;
    uint64_t* controllers;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(context->pass->arena, mask_count * word_count,
                                  sizeof(*controllers), (void**)&controllers));
    memset(controllers, 0, mask_count * word_count * sizeof(*controllers));
    uint64_t* overflow_controllers =
        controllers + context->family->column_count * word_count;
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (bindings[i].controller == IREE_HOST_SIZE_MAX) {
        continue;
      }
      const loom_pipeline_lifetime_request_t* request = &requests[i];
      const loom_aie2p_native_tile_t* shim = &context->tiles[request->pool / 2];
      const uint8_t capacity = shim->facts->dma.channel_count_per_direction;
      uint64_t* mask = controllers + shim->coordinate.column * word_count;
      if (request->offset >= capacity) {
        const loom_aie2p_native_shim_queue_t* queue =
            &remaps[shim->coordinate.column * 2 + request->pool % 2]
                   [request->offset - capacity];
        mask = overflow_controllers + (queue - overflow) * word_count;
      }
      const iree_host_size_t controller = bindings[i].controller;
      mask[controller / 64] |= UINT64_C(1) << (controller % 64);
    }
    const uint8_t southbound_capacity = iree_min(
        loom_xdna_array_stream_port_range(
            context->family, LOOM_XDNA_TILE_KIND_MEMORY,
            LOOM_XDNA_STREAM_DIRECTION_MASTER, LOOM_XDNA_STREAM_PORT_SOUTH)
            ->count,
        loom_xdna_array_stream_port_range(
            context->family, LOOM_XDNA_TILE_KIND_SHIM_NOC,
            LOOM_XDNA_STREAM_DIRECTION_SLAVE, LOOM_XDNA_STREAM_PORT_NORTH)
            ->count);
    for (uint16_t column = 0; column < context->family->column_count;
         ++column) {
      const iree_host_size_t tile_index = loom_xdna_array_physical_tile_index(
          context->family, (loom_xdna_tile_coordinate_t){column, 0});
      const uint8_t capacity =
          context->tiles[tile_index].facts->dma.channel_count_per_direction;
      for (unsigned direction = 0; direction < 2; ++direction) {
        next = remaps[column * 2 + direction];
        const uint64_t required =
            extents[direction][tile_index * 2 + direction].length;
        for (uint64_t engine = capacity; engine < required; ++engine) {
          loom_aie2p_native_tile_t* shim = loom_aie2p_native_find_shim_queue(
              context, column, direction != 0,
              overflow_controllers + (next - overflow) * word_count,
              controllers, word_count, southbound_capacity);
          if (!shim) {
            return loom_aie2p_native_reject(
                context, realization->function.op,
                IREE_SV("available DMA queues for simultaneous construction "
                        "lifetimes"));
          }
          uint8_t* assigned = direction ? &shim->s2mm_count : &shim->mm2s_count;
          for (iree_host_size_t word = 0; word < word_count; ++word) {
            controllers[shim->coordinate.column * word_count + word] |=
                overflow_controllers[(next - overflow) * word_count + word];
          }
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
    *bindings[i].engine = (uint8_t)requests[i].offset;
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
