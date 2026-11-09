// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_resources.h"

#include "loom/target/arch/amd/xdna/aie2p/call_abi.h"

// Callers compare transfers sharing one admitted physical resource owner.
// Common execution proves distinct worker occurrences have disjoint lifetimes,
// even when repeated calls share source operations. Within one occurrence,
// async legality retires every group in its block, and the retained completion
// edge determines when that block can reuse the resource.
bool loom_aie2p_native_transfer_retired(
    const loom_aie2p_native_transfer_t* transfer, iree_host_size_t worker_index,
    const loom_op_t* issue) {
  return transfer->worker_index != worker_index ||
         transfer->completion->parent_block != issue->parent_block ||
         transfer->completion->block_ordinal <= issue->block_ordinal;
}

void loom_aie2p_native_task_descriptor_words(
    const loom_aie2p_native_transfer_task_t* task,
    const loom_aie2p_native_dma_descriptor_t* descriptor,
    const loom_aie2p_native_transfer_t* transfer, uint32_t out_words[8]) {
  memcpy(out_words,
         descriptor->shim ? transfer->shim_words : transfer->local_words,
         sizeof(transfer->local_words));
  if (transfer->descriptor_ordinal + 1 == task->transfer_count) {
    return;
  }
  const loom_aie2p_native_memory_layout_t* layout =
      descriptor->shim ? NULL : task->path->local->memory_layout;
  const uint8_t word = layout ? layout->chain.word : 7;
  const loom_xdna_register_field_id_t next =
      layout ? layout->chain.next
             : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_NEXT_BD;
  const loom_xdna_register_field_id_t enable =
      layout ? layout->chain.enable
             : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_USE_NEXT_BD;
  out_words[word] |= loom_xdna_register_field_encode(
                         next, descriptor->sequence.descriptor +
                                   transfer->descriptor_ordinal + 1) |
                     loom_xdna_register_field_encode(enable, 1);
}

static bool loom_aie2p_native_task_descriptor_matches(
    const loom_aie2p_native_transfer_task_t* task,
    const loom_aie2p_native_dma_descriptor_t* descriptor, uint32_t address_mask,
    bool* out_address_differs) {
  *out_address_differs = false;
  if (descriptor->sequence.descriptor_count != task->transfer_count) {
    return false;
  }
  for (const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
       transfer; transfer = transfer->task_next) {
    uint32_t words[8];
    loom_aie2p_native_task_descriptor_words(task, descriptor, transfer, words);
    const uint32_t* previous =
        descriptor->sequence.words[transfer->descriptor_ordinal];
    if (address_mask) {
      const uint8_t word = task->path->local->memory_layout->address_word;
      const uint32_t difference = (words[word] ^ previous[word]) & address_mask;
      *out_address_differs |= difference != 0;
      words[word] ^= difference;
    }
    if (memcmp(previous, words, sizeof(words)) != 0) {
      return false;
    }
  }
  return true;
}

static bool loom_aie2p_native_task_address_differs(
    const loom_aie2p_native_transfer_task_t* task,
    const loom_aie2p_native_transfer_t* previous, bool shim) {
  const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
  for (; transfer && previous;
       transfer = transfer->task_next, previous = previous->task_next) {
    if (shim) {
      if (transfer->binding != previous->binding ||
          transfer->external_view->begin_byte_offset.constant !=
              previous->external_view->begin_byte_offset.constant) {
        return true;
      }
    } else {
      const uint8_t word = task->path->local->memory_layout->address_word;
      // Other bits in this word already match by descriptor admission.
      if (transfer->local_words[word] != previous->local_words[word]) {
        return true;
      }
    }
  }
  return transfer || previous;
}

// Each task endpoint has an immutable descriptor sequence and latest retained
// completion. Local and shim sequences can be reused independently.
loom_aie2p_native_dma_descriptor_t* loom_aie2p_native_reusable_descriptor(
    const loom_aie2p_native_worker_t* worker,
    const loom_aie2p_native_transfer_task_t* task, bool shim) {
  if (worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_DMA) {
    // Autonomous descriptors retain iteration state until invocation
    // retirement.
    return NULL;
  }
  const loom_op_t* issue = task->first_transfer->source->request.op;
  const iree_host_size_t worker_index = task->first_transfer->worker_index;
  uint32_t address_mask = 0;
  if (!shim) {
    const loom_xdna_register_field_info_t address =
        loom_xdna_register_field_info(
            task->path->local->memory_layout->address);
    address_mask = (UINT32_MAX >> (32 - address.bit_width))
                   << address.least_significant_bit;
  }
  for (loom_aie2p_native_dma_descriptor_t* candidate = worker->descriptors;
       candidate; candidate = candidate->next) {
    const loom_aie2p_native_transfer_t* previous = candidate->last_transfer;
    bool address_differs;
    // Shim descriptors belong to the tile's shared bank. Direction and
    // completion routing belong to the issuing channel, so a retired sequence
    // can serve another path on that shim. Local descriptors retain their
    // path-specific completion semaphore and address aperture.
    const bool same_bank = shim ? candidate->path->shim == task->path->shim
                                : candidate->path == task->path;
    if (candidate->shim != shim || !same_bank ||
        !loom_aie2p_native_transfer_retired(previous, worker_index, issue) ||
        (candidate->completion &&
         !loom_aie2p_native_transfer_retired(
             candidate->completion->last_transfer, worker_index, issue)) ||
        !loom_aie2p_native_task_descriptor_matches(
            task, candidate, address_mask, &address_differs)) {
      continue;
    }
    candidate->address_required |=
        shim ? task->shim_address_dynamic : task->local_address_dynamic;
    if (shim) {
      candidate->address_dynamic |=
          task->shim_address_dynamic ||
          loom_aie2p_native_task_address_differs(task, previous, true);
    } else if (task->local_address_dynamic || address_differs) {
      candidate->address_dynamic = true;
      // All users patch their own address after the preceding task retires.
      // Keep length, dimensions, and chain links in the shared template.
      const uint8_t word = task->path->local->memory_layout->address_word;
      for (uint16_t i = 0; i < candidate->sequence.descriptor_count; ++i) {
        candidate->sequence.words[i][word] &= ~address_mask;
      }
    }
    return candidate;
  }
  return NULL;
}

// Both core and configuration waits consume arrival credits, so a retired
// credit can serve another descriptor geometry on the same tile. The search is
// bounded by the physical lock count.
loom_aie2p_native_dma_completion_t* loom_aie2p_native_reusable_completion(
    const loom_aie2p_native_worker_t* worker,
    const loom_aie2p_native_tile_t* tile,
    const loom_aie2p_native_transfer_t* next) {
  if (worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_DMA) {
    return NULL;
  }
  for (loom_aie2p_native_dma_completion_t* completion = worker->dma_completions;
       completion; completion = completion->next) {
    if (completion->tile == tile &&
        loom_aie2p_native_transfer_retired(completion->last_transfer,
                                           next->worker_index,
                                           next->source->request.op)) {
      return completion;
    }
  }
  return NULL;
}

// The caller has reserved a contiguous range in this endpoint bank.
static iree_status_t loom_aie2p_native_create_task_descriptor(
    loom_aie2p_native_context_t* context,
    const loom_aie2p_native_transfer_task_t* task, bool shim,
    uint16_t first_descriptor,
    loom_aie2p_native_dma_descriptor_t** out_descriptor) {
  const iree_host_size_t words_bytes =
      task->transfer_count * sizeof(uint32_t[8]);
  loom_aie2p_native_dma_descriptor_t* descriptor;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(context->pass->arena,
                                           sizeof(*descriptor) + words_bytes,
                                           (void**)&descriptor));
  *descriptor = (loom_aie2p_native_dma_descriptor_t){
      .path = task->path,
      .first_transfer = task->first_transfer,
      .sequence =
          {
              .descriptor = first_descriptor,
              .descriptor_count = (uint16_t)task->transfer_count,
              .words = (uint32_t (*)[8])(descriptor + 1),
          },
      .shim = shim,
      .address_dynamic =
          shim ? task->shim_address_dynamic : task->local_address_dynamic,
      .address_required =
          shim ? task->shim_address_dynamic : task->local_address_dynamic,
  };
  for (const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
       transfer; transfer = transfer->task_next) {
    loom_aie2p_native_task_descriptor_words(
        task, descriptor, transfer,
        descriptor->sequence.words[transfer->descriptor_ordinal]);
  }
  *out_descriptor = descriptor;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_allocate_task_descriptor(
    loom_aie2p_native_context_t* context, loom_aie2p_native_worker_t* worker,
    loom_aie2p_native_worker_t* resource_owner,
    loom_aie2p_native_transfer_task_t* task, bool shim,
    loom_aie2p_native_dma_descriptor_t*** descriptor_tail,
    loom_aie2p_native_dma_completion_t*** completion_tail,
    loom_aie2p_native_dma_descriptor_t** out_descriptor) {
  loom_aie2p_native_tile_t* tile = shim ? task->path->shim : task->path->local;
  const uint8_t engine =
      shim ? task->path->shim_engine : task->path->local_engine;
  uint16_t descriptor_base, descriptor_count;
  uint16_t* next_descriptor = loom_aie2p_native_descriptor_bank(
      tile, engine, &descriptor_base, &descriptor_count);
  if (task->transfer_count > descriptor_count - *next_descriptor) {
    return loom_aie2p_native_reject(
        context, task->first_transfer->source->request.op,
        shim ? IREE_SV("available native shim descriptors for one async group")
             : IREE_SV("available native descriptors in the DMA channel's bank "
                       "for one async group"));
  }
  loom_aie2p_native_dma_descriptor_t* descriptor;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_create_task_descriptor(
      context, task, shim, descriptor_base + *next_descriptor, &descriptor));
  *next_descriptor += (uint16_t)task->transfer_count;
  if (!shim && task->path->ingress && !task->path->completion_tokens) {
    descriptor->completion = loom_aie2p_native_reusable_completion(
        resource_owner, tile, task->first_transfer);
    if (!descriptor->completion) {
      if (tile->next_lock == tile->facts->lock_count) {
        return loom_aie2p_native_reject(
            context, task->first_transfer->source->request.op,
            IREE_SV("an available native ingress completion semaphore"));
      }
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          context->pass->arena, sizeof(*descriptor->completion),
          (void**)&descriptor->completion));
      *descriptor->completion = (loom_aie2p_native_dma_completion_t){
          .tile = tile, .lock = tile->next_lock++};
      if (worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
        const loom_aie2p_native_channel_t* channel =
            task->first_transfer->local_channel;
        const uint32_t channel_index = (uint32_t)(channel - context->channels);
        const loom_aie2p_native_channel_access_t* access =
            &worker->channels.accesses[worker->channels.indices[channel_index]];
        descriptor->completion->selector =
            access->window->lock_selector_base + descriptor->completion->lock;
      }
      **completion_tail = descriptor->completion;
      *completion_tail = &descriptor->completion->next;
    }
  }
  **descriptor_tail = descriptor;
  *descriptor_tail = &descriptor->next;
  *out_descriptor = descriptor;
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_select_task_descriptors(
    loom_aie2p_native_context_t* context, loom_aie2p_native_worker_t* worker,
    loom_aie2p_native_worker_t* resource_owner,
    loom_aie2p_native_transfer_task_t* task, bool autonomous,
    loom_aie2p_native_dma_descriptor_t*** descriptor_tail,
    loom_aie2p_native_dma_completion_t*** completion_tail, bool* out_valid) {
  *out_valid = false;
  if (autonomous) {
    // Autonomous repetition admits one source transfer and builds its immutable
    // ring program through the sequence planner.
    const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
    for (unsigned endpoint = 0; endpoint < 2; ++endpoint) {
      const bool shim = endpoint != 0;
      loom_aie2p_native_tile_t* tile =
          shim ? task->path->shim : task->path->local;
      loom_aie2p_native_dma_descriptor_t* descriptor;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          context->pass->arena, sizeof(*descriptor), (void**)&descriptor));
      *descriptor = (loom_aie2p_native_dma_descriptor_t){
          .path = task->path,
          .first_transfer = task->first_transfer,
          .shim = shim,
      };
      iree_string_view_t reason;
      bool valid;
      if (shim) {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_select_sequence(
            context, tile, task->path->shim_engine,
            LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM, NULL,
            worker->repetition.shim_repetitions, 1, 0, 0, 1,
            transfer->shim_words, &descriptor->sequence, &reason, &valid));
      } else {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_select_sequence(
            context, tile, task->path->local_engine,
            LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY, transfer->local_channel,
            worker->repetition.count,
            transfer->local_channel->replay.record_count
                ? 1
                : transfer->local_channel->source->capacity,
            transfer->local_channel->byte_offset +
                (uint32_t)transfer->local_view->projection_byte_offset.constant,
            transfer->local_channel->byte_stride, 1, transfer->local_words,
            &descriptor->sequence, &reason, &valid));
      }
      if (!valid) {
        return loom_aie2p_native_reject(
            context, task->first_transfer->source->request.op, reason);
      }
      **descriptor_tail = descriptor;
      *descriptor_tail = &descriptor->next;
      if (shim) {
        task->shim_descriptor = descriptor;
      } else {
        task->local_descriptor = descriptor;
      }
    }
  } else {
    task->local_descriptor =
        loom_aie2p_native_reusable_descriptor(resource_owner, task, false);
    if (!task->local_descriptor) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_allocate_task_descriptor(
          context, worker, resource_owner, task, false, descriptor_tail,
          completion_tail, &task->local_descriptor));
      if (!task->local_descriptor) {
        return iree_ok_status();
      }
    }
    task->shim_descriptor =
        loom_aie2p_native_reusable_descriptor(resource_owner, task, true);
    if (!task->shim_descriptor) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_allocate_task_descriptor(
          context, worker, resource_owner, task, true, descriptor_tail,
          completion_tail, &task->shim_descriptor));
      if (!task->shim_descriptor) {
        return iree_ok_status();
      }
    }
  }
  task->local_descriptor->last_transfer = task->first_transfer;
  task->shim_descriptor->last_transfer = task->first_transfer;
  if (task->local_descriptor->completion) {
    task->local_descriptor->completion->last_transfer = task->first_transfer;
  }
  for (loom_aie2p_native_transfer_t* transfer = task->first_transfer; transfer;
       transfer = transfer->task_next) {
    transfer->local_descriptor = task->local_descriptor;
    transfer->shim_descriptor = task->shim_descriptor;
  }
  *out_valid = true;
  return iree_ok_status();
}

// Admission first packs all required geometries, preserving the programs that
// fit by rewriting shared addresses. Only then may spare descriptors remove
// issue-time patches. Each variant inherits the original ordered lifetimes and
// arrival credit; it needs no new route, semaphore, or payload storage.
iree_status_t loom_aie2p_native_refine_transfer_addresses(
    loom_aie2p_native_context_t* context, iree_host_size_t worker_count) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < worker_count;
       ++i) {
    loom_aie2p_native_worker_t* worker = &context->workers[i];
    if (worker->execution != LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
      continue;
    }
    loom_aie2p_native_worker_t* owner =
        &context->workers[context->execution.workers[context->core_engines[i]]];
    for (loom_aie2p_native_transfer_task_t* task = worker->transfer_tasks;
         iree_status_is_ok(status) && task; task = task->next) {
      for (uint8_t endpoint = 0; iree_status_is_ok(status) && endpoint < 2;
           ++endpoint) {
        const bool shim = endpoint != 0;
        loom_aie2p_native_dma_descriptor_t* original =
            shim ? task->shim_descriptor : task->local_descriptor;
        if (!original->address_dynamic ||
            (shim ? task->shim_address_dynamic : task->local_address_dynamic)) {
          original->last_transfer = task->first_transfer;
          continue;
        }
        if (!original->address_required &&
            !loom_aie2p_native_task_address_differs(
                task, original->first_transfer, shim)) {
          original->last_transfer = task->first_transfer;
          continue;
        }
        loom_aie2p_native_dma_descriptor_t* variant =
            original->address_variants;
        while (variant && loom_aie2p_native_task_address_differs(
                              task, variant->first_transfer, shim)) {
          variant = variant->address_variants;
        }
        if (!variant) {
          loom_aie2p_native_tile_t* tile =
              shim ? task->path->shim : task->path->local;
          const uint8_t engine =
              shim ? task->path->shim_engine : task->path->local_engine;
          uint16_t descriptor_base, descriptor_count;
          uint16_t* next_descriptor = loom_aie2p_native_descriptor_bank(
              tile, engine, &descriptor_base, &descriptor_count);
          if (task->transfer_count > descriptor_count - *next_descriptor) {
            original->address_required = true;
            original->last_transfer = task->first_transfer;
            continue;
          }
          status = loom_aie2p_native_create_task_descriptor(
              context, task, shim, descriptor_base + *next_descriptor,
              &variant);
          if (!iree_status_is_ok(status)) {
            continue;
          }
          *next_descriptor += (uint16_t)task->transfer_count;
          variant->completion = original->completion;
          variant->next = owner->descriptors;
          owner->descriptors = variant;
          variant->address_variants = original->address_variants;
          variant->address_origin = original;
          original->address_variants = variant;
        }
        variant->last_transfer = task->first_transfer;
        if (shim) {
          task->shim_descriptor = variant;
        } else {
          task->local_descriptor = variant;
        }
        for (loom_aie2p_native_transfer_t* transfer = task->first_transfer;
             transfer; transfer = transfer->task_next) {
          if (shim) {
            transfer->shim_descriptor = variant;
          } else {
            transfer->local_descriptor = variant;
          }
        }
      }
    }
  }
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < worker_count;
       ++i) {
    if (context->workers[i].execution != LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
      continue;
    }
    for (loom_aie2p_native_dma_descriptor_t* descriptor =
             context->workers[i].descriptors;
         descriptor; descriptor = descriptor->next) {
      if (!descriptor->address_dynamic || descriptor->address_required) {
        continue;
      }
      descriptor->address_dynamic = false;
      const loom_aie2p_native_transfer_task_t* task =
          descriptor->first_transfer->task;
      for (const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
           transfer; transfer = transfer->task_next) {
        loom_aie2p_native_task_descriptor_words(
            task, descriptor, transfer,
            descriptor->sequence.words[transfer->descriptor_ordinal]);
      }
    }
  }
  return status;
}

// Final descriptor policies define the submission signature. A local
// descriptor family proves matching routes, geometry, and local patch words.
// Fixed variants share one body by passing their local queue word; the dynamic
// original keeps a separate body with its constant descriptor patch address.
// Its physical core owns the complete program, including ordered occurrences
// that reuse the endpoint after the preceding phase has retired.
iree_status_t loom_aie2p_native_group_submissions(
    loom_aie2p_native_context_t* context, iree_host_size_t worker_count) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < worker_count;
       ++i) {
    loom_aie2p_native_worker_t* worker = &context->workers[i];
    if (worker->execution != LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
      continue;
    }
    for (loom_aie2p_native_transfer_task_t* task = worker->transfer_tasks;
         iree_status_is_ok(status) && task; task = task->next) {
      // Queue-only and local-only patches are too short to reliably amortize
      // native call overhead. Their constants also share within the caller.
      if (!task->shim_descriptor->address_dynamic) {
        continue;
      }
      loom_aie2p_native_dma_descriptor_t* descriptor = task->local_descriptor;
      loom_aie2p_native_dma_descriptor_t* origin =
          descriptor->address_origin ? descriptor->address_origin : descriptor;
      const bool local_queue_argument =
          !descriptor->address_dynamic && origin->address_variants;
      const loom_aie2p_call_abi_requirements_t requirements = {
          .scalar_count = (uint16_t)(task->transfer_count *
                                         (2 + descriptor->address_dynamic) +
                                     1 + local_queue_argument),
          .pointer_count = (uint16_t)task->transfer_count,
      };
      if (!loom_aie2p_call_abi_requirements_fit(requirements)) {
        continue;
      }
      loom_aie2p_native_dma_submit_t* group =
          origin->submit_groups[descriptor->address_dynamic];
      if (!group) {
        status = iree_arena_allocate(context->pass->arena, sizeof(*group),
                                     (void**)&group);
        if (iree_status_is_ok(status)) {
          *group = (loom_aie2p_native_dma_submit_t){
              .local_queue_argument = local_queue_argument,
              .function = loom_symbol_ref_null(),
          };
          origin->submit_groups[descriptor->address_dynamic] = group;
        }
      }
      if (iree_status_is_ok(status)) {
        ++group->use_count;
        task->submit_group = group;
      }
    }
  }
  return status;
}
