// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sanitizer/msan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#if !defined(IREE_SANITIZER_MEMORY)
#error "This test requires MemorySanitizer instrumentation."
#endif

// A native C entry point keeps allocator qualification independent of the C++
// standard library's instrumentation. Inspect shadow without reading undefined
// payload bytes or intentionally triggering an MSAN report.
#define CHECK(expression)                                              \
  do {                                                                 \
    if (!(expression)) {                                               \
      fprintf(stderr, "%s:%d: %s\n", __func__, __LINE__, #expression); \
      abort();                                                         \
    }                                                                  \
  } while (0)

static void require_ok(iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
  }
}

static void expect_uninitialized(const void* pointer, iree_host_size_t length) {
  for (iree_host_size_t i = 0; i < length; ++i) {
    CHECK(__msan_test_shadow((const uint8_t*)pointer + i, 1) == 0);
  }
  if (length && __msan_get_track_origins()) {
    CHECK(__msan_get_origin(pointer) != 0);
  }
}

static void expect_initialized(const void* pointer, iree_host_size_t length,
                               uint8_t value) {
  CHECK(__msan_test_shadow(pointer, length) == -1);
  for (iree_host_size_t i = 0; i < length; ++i) {
    CHECK(((const uint8_t*)pointer)[i] == value);
  }
}

static void initialize(void* pointer, iree_host_size_t length) {
  memset(pointer, 0xAB, length);
  // A later allocation must install its own origin, not retain an earlier one.
  __msan_set_origin(pointer, length, 0);
}

// Returning initialized storage is a valid backing-allocator policy. Arena
// allocation lifetimes must not depend on the backing allocator's shadow.
static iree_status_t zeroing_allocator_ctl(void* self,
                                           iree_allocator_command_t command,
                                           const void* params,
                                           void** inout_pointer) {
  iree_allocator_t allocator = iree_allocator_system();
  if (command == IREE_ALLOCATOR_COMMAND_MALLOC) {
    command = IREE_ALLOCATOR_COMMAND_CALLOC;
  }
  iree_status_t status =
      allocator.ctl(allocator.self, command, params, inout_pointer);
  if (iree_status_is_ok(status) && command == IREE_ALLOCATOR_COMMAND_CALLOC) {
    const iree_allocator_alloc_params_t* alloc_params = params;
    __msan_set_origin(*inout_pointer, alloc_params->byte_length, 0);
  }
  return status;
}

typedef struct test_arena_t {
  // Shared pool, with deliberately initialized backing allocations.
  iree_arena_block_pool_t pool;
  // Arena under test; individual cases may share the pool with other arenas.
  iree_arena_allocator_t arena;
} test_arena_t;

static void* allocate(iree_arena_allocator_t* arena, iree_host_size_t length) {
  void* pointer = NULL;
  require_ok(iree_arena_allocate(arena, length, &pointer));
  return pointer;
}

static void raw_blocks(test_arena_t* test) {
  for (int preallocate = 0; preallocate < 2; ++preallocate) {
    if (preallocate) {
      iree_arena_block_pool_trim(&test->pool);
      require_ok(iree_arena_block_pool_preallocate(&test->pool, 2));
    }
    iree_arena_block_t* block = NULL;
    void* pointer = NULL;
    require_ok(iree_arena_block_pool_acquire(&test->pool, &block, &pointer));
    expect_uninitialized(pointer, test->pool.usable_block_size);
    CHECK(block->next == NULL);
    initialize(pointer, test->pool.usable_block_size);
    iree_arena_block_pool_release(&test->pool, block, block);
    void* reused_pointer = NULL;
    require_ok(
        iree_arena_block_pool_acquire(&test->pool, &block, &reused_pointer));
    CHECK(reused_pointer == pointer);
    expect_uninitialized(reused_pointer, test->pool.usable_block_size);
    CHECK(block->next == NULL);
    iree_arena_block_pool_release(&test->pool, block, block);
  }
}

static void same_block_checkpoint(test_arena_t* test) {
  void* retained = allocate(&test->arena, 32);
  initialize(retained, 32);
  iree_arena_checkpoint_t checkpoint = iree_arena_checkpoint_save(&test->arena);
  // Restoring a checkpoint stored in the discarded region must finish reading
  // its fields before marking that region uninitialized.
  iree_arena_checkpoint_t* stored_checkpoint =
      allocate(&test->arena, sizeof(checkpoint));
  *stored_checkpoint = checkpoint;
  void* discarded = allocate(&test->arena, 37);
  initialize(discarded, 37);
  iree_arena_checkpoint_restore(stored_checkpoint);
  expect_initialized(retained, 32, 0xAB);
  CHECK(allocate(&test->arena, sizeof(checkpoint)) == stored_checkpoint);
  expect_uninitialized(stored_checkpoint, sizeof(checkpoint));
  CHECK(allocate(&test->arena, 37) == discarded);
  expect_uninitialized(discarded, 37);
  initialize(discarded, 37);
  checkpoint = iree_arena_checkpoint_save(&test->arena);
  iree_arena_checkpoint_restore(&checkpoint);
  expect_initialized(discarded, 37, 0xAB);
}

static void acquired_storage_checkpoint(test_arena_t* test) {
  // Exercise block acquisition, oversized acquisition, and both together.
  for (int mode = 1; mode <= 3; ++mode) {
    iree_arena_reset(&test->arena);
    void* retained = allocate(&test->arena, 32);
    initialize(retained, 32);
    iree_arena_checkpoint_t checkpoint =
        iree_arena_checkpoint_save(&test->arena);
    void* discarded = allocate(&test->arena, 48);
    initialize(discarded, 48);
    if (mode & 1) {
      void* block = allocate(
          &test->arena, iree_arena_block_pool_max_allocation_size(&test->pool));
      initialize(block, iree_arena_block_pool_max_allocation_size(&test->pool));
    }
    if (mode & 2) {
      initialize(allocate(&test->arena, 8192), 8192);
    }
    iree_arena_checkpoint_t* stored_checkpoint =
        allocate(&test->arena, sizeof(checkpoint));
    *stored_checkpoint = checkpoint;
    iree_arena_checkpoint_restore(stored_checkpoint);
    expect_initialized(retained, 32, 0xAB);
    CHECK(allocate(&test->arena, 48) == discarded);
    expect_uninitialized(discarded, 48);
    void* next_block = allocate(
        &test->arena, iree_arena_block_pool_max_allocation_size(&test->pool));
    expect_uninitialized(
        next_block, iree_arena_block_pool_max_allocation_size(&test->pool));
  }
}

static void reset_and_share(test_arena_t* test) {
  void* pointer = allocate(&test->arena, 64);
  expect_uninitialized(pointer, 64);
  initialize(pointer, 64);
  iree_arena_reset(&test->arena);
  CHECK(allocate(&test->arena, 64) == pointer);
  expect_uninitialized(pointer, 64);
  initialize(pointer, 64);
  iree_arena_reset(&test->arena);
  iree_arena_allocator_t other;
  iree_arena_initialize(&test->pool, &other);
  CHECK(allocate(&other, 64) == pointer);
  expect_uninitialized(pointer, 64);
  iree_arena_deinitialize(&other);
}

static void aligned_arrays(test_arena_t* test) {
  const iree_host_size_t lengths[] = {0, 1, 17, 128, 4096, 8193};
  const iree_host_size_t alignments[] = {iree_max_align_t, 64, 4096};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(lengths); ++i) {
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(alignments); ++j) {
      for (int reuse = 0; reuse < 2; ++reuse) {
        void* retained = allocate(&test->arena, 16);
        initialize(retained, 16);
        void* pointer = NULL;
        require_ok(iree_arena_allocate_array_aligned(
            &test->arena, lengths[i], 1, alignments[j], &pointer));
        CHECK(((uintptr_t)pointer % alignments[j]) == 0);
        expect_uninitialized(pointer, lengths[i]);
        initialize(pointer, lengths[i]);
        expect_initialized(retained, 16, 0xAB);
        iree_arena_reset(&test->arena);
      }
    }
  }
}

static void growing_array(test_arena_t* test) {
  iree_host_size_t capacity = 64;
  void* original = NULL;
  require_ok(iree_arena_allocate_array(&test->arena, capacity, 1, &original));
  initialize(original, 32);
  void* grown = original;
  require_ok(
      iree_arena_grow_array(&test->arena, 64, 128, 1, &capacity, &grown));
  CHECK(grown != original);
  CHECK(capacity == 128);
  expect_initialized(grown, 32, 0xAB);
  expect_uninitialized((uint8_t*)grown + 32, capacity - 32);
  expect_initialized(original, 32, 0xAB);
  expect_uninitialized((uint8_t*)original + 32, 32);
}

static void allocator_adapter(test_arena_t* test) {
  iree_allocator_t allocator = iree_arena_allocator(&test->arena);
  const iree_host_size_t lengths[] = {0, 32, 8192};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(lengths); ++i) {
    void* pointer = NULL;
    require_ok(iree_allocator_malloc(allocator, lengths[i], &pointer));
    expect_initialized(pointer, lengths[i], 0);
    require_ok(
        iree_allocator_malloc_uninitialized(allocator, lengths[i], &pointer));
    expect_uninitialized(pointer, lengths[i]);
  }
}

static void transfer(test_arena_t* test) {
  iree_arena_allocator_t target;
  iree_arena_initialize(&test->pool, &target);
  void* retained = allocate(&target, 32);
  initialize(retained, 32);
  iree_arena_checkpoint_t checkpoint = iree_arena_checkpoint_save(&target);
  void* source = allocate(&test->arena, 64);
  initialize(source, 32);
  void* oversized = allocate(&test->arena, 8192);
  initialize(oversized, 32);
  iree_arena_transfer(&test->arena, &target);
  expect_initialized(retained, 32, 0xAB);
  expect_initialized(source, 32, 0xAB);
  expect_uninitialized((uint8_t*)source + 32, 32);
  expect_initialized(oversized, 32, 0xAB);
  expect_uninitialized((uint8_t*)oversized + 32, 8192 - 32);
  expect_uninitialized(allocate(&target, 32), 32);
  expect_uninitialized(allocate(&test->arena, 64), 64);
  expect_initialized(source, 32, 0xAB);
  iree_arena_checkpoint_restore(&checkpoint);
  expect_initialized(retained, 32, 0xAB);
  expect_uninitialized(allocate(&target, 32), 32);
  iree_arena_deinitialize(&target);
}

int main(void) {
  void (*const tests[])(test_arena_t*) = {
      raw_blocks,
      same_block_checkpoint,
      acquired_storage_checkpoint,
      reset_and_share,
      aligned_arrays,
      growing_array,
      allocator_adapter,
      transfer,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(tests); ++i) {
    test_arena_t test;
    const iree_allocator_t backing_allocator = {
        .self = NULL,
        .ctl = zeroing_allocator_ctl,
    };
    iree_arena_block_pool_initialize(4096, backing_allocator, &test.pool);
    iree_arena_initialize(&test.pool, &test.arena);
    tests[i](&test);
    iree_arena_deinitialize(&test.arena);
    iree_arena_block_pool_deinitialize(&test.pool);
    printf("arena MSAN case %" PRIhsz " passed\n", i);
  }
  return 0;
}
