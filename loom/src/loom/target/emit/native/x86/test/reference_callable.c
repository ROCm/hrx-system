// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/vm/ref.h"

typedef float vector8_t __attribute__((vector_size(32)));
typedef struct object_t {
  // Managed objects use the ordinary runtime's intrusive prefix.
  iree_vm_ref_object_t ref_object;
  // Payload remains readable after the source owner is consumed.
  uint64_t value;
} object_t;
typedef struct result_t {
  // Scalar result must survive final destruction alongside the vector.
  uint64_t value;
  // Natural C alignment also exercises the native stack realignment path.
  vector8_t wide;
} result_t;
typedef struct escape_result_t {
  // Earlier fields remain untouched when a later result cannot escape.
  uint64_t marker;
  // A successful return promotes the managed pointer to a real owner.
  iree_vm_ref_t pointer;
} escape_result_t;
typedef struct identity_result_t {
  // One owner returned after consuming the incoming owner.
  iree_vm_ref_t owner;
  // Exact object identity, independent of reference state.
  bool same;
  // Nullness does not inspect object storage.
  bool missing;
} identity_result_t;
typedef struct null_result_t {
  // Null owning result.
  iree_vm_ref_t owner;
  // Null pointer result.
  iree_vm_ref_t pointer;
} null_result_t;
typedef struct storage_result_t {
  // First owner survives retirement of the private storage plane.
  iree_vm_ref_t first;
  // Second owner may be null.
  iree_vm_ref_t second;
  // An untouched cell must be null before any store.
  uint32_t missing;
} storage_result_t;

extern iree_status_t object_forward(result_t* result, uint32_t failure,
                                    uint64_t seed);
extern iree_status_t object_escape(escape_result_t* result,
                                   const iree_vm_ref_t* pointer);
extern iree_status_t object_identity(identity_result_t* result,
                                     iree_vm_ref_t* owner,
                                     const iree_vm_ref_t* pointer);
extern iree_status_t object_select(iree_vm_ref_t* result, bool condition,
                                   const iree_vm_ref_t* left,
                                   const iree_vm_ref_t* right);
extern iree_status_t object_null(null_result_t* result);
extern iree_status_t object_twice(null_result_t* result,
                                  const iree_vm_ref_t* pointer);
extern iree_status_t object_stack(
    iree_vm_ref_t* result, const iree_vm_ref_t* a, const iree_vm_ref_t* b,
    const iree_vm_ref_t* c, const iree_vm_ref_t* d, const iree_vm_ref_t* e,
    const iree_vm_ref_t* f, const iree_vm_ref_t* g);

extern iree_status_t object_drop_input(iree_vm_ref_t* owner);
extern iree_status_t object_drop_result(void);

extern iree_status_t object_preserve_vectors(iree_vm_ref_t* owner,
                                             const int32_t* left,
                                             const int32_t* right,
                                             int32_t* output);

extern iree_status_t object_rotate(null_result_t* result, iree_vm_ref_t* first,
                                   iree_vm_ref_t* second, uint32_t count);

extern iree_status_t object_storage(iree_vm_ref_t* result, iree_vm_ref_t* first,
                                    iree_vm_ref_t* second, uint32_t failure);
extern iree_status_t object_storage_borrowed(uint32_t* result,
                                             const iree_vm_ref_t* pointer);
extern iree_status_t object_storage_group(storage_result_t* result,
                                          iree_vm_ref_t* first,
                                          iree_vm_ref_t* second);
extern iree_status_t object_view_reuse(uint32_t* result, iree_vm_ref_t* owner,
                                       uint64_t count);
extern iree_status_t object_view_slice(iree_vm_ref_t* result, uint64_t count,
                                       const iree_vm_ref_t* values,
                                       uint64_t position);
extern iree_status_t object_view_replace(iree_vm_ref_t* owner,
                                         iree_vm_ref_t* values,
                                         uint64_t position);
extern iree_status_t object_view_call(uint64_t count,
                                      const iree_vm_ref_t* values,
                                      uint32_t failure);
extern iree_status_t object_storage_clear(iree_vm_ref_t* cells);
extern iree_status_t object_storage_clear_many(iree_vm_ref_t* cells);

// Host observations are independent of generated ownership bookkeeping.
static unsigned destroyed;
// Enables wider clobbers only when execution hardware admits AVX-512.
static bool wide_clobbers;

__attribute__((target("avx512f"))) static void clobber_wide_vectors(void) {
  __asm__ volatile(
      "vpxord %%zmm0, %%zmm0, %%zmm0\n\t"
      "vpxord %%zmm1, %%zmm1, %%zmm1\n\t"
      "vpxord %%zmm2, %%zmm2, %%zmm2\n\t"
      "vpxord %%zmm3, %%zmm3, %%zmm3\n\t"
      "vpxord %%zmm4, %%zmm4, %%zmm4\n\t"
      "vpxord %%zmm5, %%zmm5, %%zmm5\n\t"
      "vpxord %%zmm6, %%zmm6, %%zmm6\n\t"
      "vpxord %%zmm7, %%zmm7, %%zmm7\n\t"
      "vpxord %%zmm8, %%zmm8, %%zmm8\n\t"
      "vpxord %%zmm9, %%zmm9, %%zmm9\n\t"
      "vpxord %%zmm10, %%zmm10, %%zmm10\n\t"
      "vpxord %%zmm11, %%zmm11, %%zmm11\n\t"
      "vpxord %%zmm12, %%zmm12, %%zmm12\n\t"
      "vpxord %%zmm13, %%zmm13, %%zmm13\n\t"
      "vpxord %%zmm14, %%zmm14, %%zmm14\n\t"
      "vpxord %%zmm15, %%zmm15, %%zmm15\n\t"
      "vpxord %%zmm16, %%zmm16, %%zmm16\n\t"
      "vpxord %%zmm17, %%zmm17, %%zmm17\n\t"
      "vpxord %%zmm18, %%zmm18, %%zmm18\n\t"
      "vpxord %%zmm19, %%zmm19, %%zmm19\n\t"
      "vpxord %%zmm20, %%zmm20, %%zmm20\n\t"
      "vpxord %%zmm21, %%zmm21, %%zmm21\n\t"
      "vpxord %%zmm22, %%zmm22, %%zmm22\n\t"
      "vpxord %%zmm23, %%zmm23, %%zmm23\n\t"
      "vpxord %%zmm24, %%zmm24, %%zmm24\n\t"
      "vpxord %%zmm25, %%zmm25, %%zmm25\n\t"
      "vpxord %%zmm26, %%zmm26, %%zmm26\n\t"
      "vpxord %%zmm27, %%zmm27, %%zmm27\n\t"
      "vpxord %%zmm28, %%zmm28, %%zmm28\n\t"
      "vpxord %%zmm29, %%zmm29, %%zmm29\n\t"
      "vpxord %%zmm30, %%zmm30, %%zmm30\n\t"
      "vpxord %%zmm31, %%zmm31, %%zmm31\n\t"
      "kxorw %%k0, %%k0, %%k0\n\t"
      "kxorw %%k1, %%k1, %%k1\n\t"
      "kxorw %%k2, %%k2, %%k2\n\t"
      "kxorw %%k3, %%k3, %%k3\n\t"
      "kxorw %%k4, %%k4, %%k4\n\t"
      "kxorw %%k5, %%k5, %%k5\n\t"
      "kxorw %%k6, %%k6, %%k6\n\t"
      "kxorw %%k7, %%k7, %%k7\n\t"
      :
      :
      : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8",
        "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16",
        "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24",
        "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31", "k0",
        "k1", "k2", "k3", "k4", "k5", "k6", "k7");
}
// Number of C adapters reached before propagation.
static unsigned calls;
// C observes the exact collection geometry and reference state passed by Loom.
static uint64_t inspected_count;
// Non-null rows observed independently of the source's view construction.
static uint64_t inspected_non_null_count;
// One status payload whose exact identity must survive every cleanup path.
static iree_status_t failure_status;

static void destroy_object(void* object) {
  ++destroyed;
  iree_allocator_free(iree_allocator_system(), object);
  if (wide_clobbers) {
    clobber_wide_vectors();
  }
  // Exercise the ordinary C ABI's complete volatile register permission.
  __asm__ volatile(
      "xor %%eax, %%eax\n\t"
      "xor %%ecx, %%ecx\n\t"
      "xor %%edx, %%edx\n\t"
      "xor %%esi, %%esi\n\t"
      "xor %%edi, %%edi\n\t"
      "xor %%r8d, %%r8d\n\t"
      "xor %%r9d, %%r9d\n\t"
      "xor %%r10d, %%r10d\n\t"
      "xor %%r11d, %%r11d\n\t"
      "vzeroall"
      :
      :
      : "rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "ymm0",
        "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9",
        "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15", "cc");
}

static const iree_vm_ref_type_table_t managed_types = {
    .structure_size = sizeof(iree_vm_ref_type_table_t),
    .namespace_name = IREE_SVL("native"),
};
static const iree_vm_ref_type_descriptor_t object_type = {
    .destroy = destroy_object,
    .table = &managed_types,
    .type_name = IREE_SVL("object"),
};
static const iree_vm_ref_type_table_t borrowed_types = {
    .structure_size = sizeof(iree_vm_ref_type_table_t),
    .flags = IREE_VM_REF_TYPE_TABLE_FLAG_BORROWED_ONLY,
    .namespace_name = IREE_SVL("native_borrowed"),
};
static const iree_vm_ref_type_descriptor_t borrowed_type = {
    .table = &borrowed_types,
    .type_name = IREE_SVL("object"),
};

iree_status_t object_create(iree_vm_ref_t* result, uint32_t failure) {
  ++calls;
  if (failure == 1) {
    return failure_status;
  }
  object_t* object = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(iree_allocator_system(),
                                             sizeof(*object), (void**)&object));
  iree_vm_ref_object_initialize(&object->ref_object);
  object->value = 123;
  void* pointer = object;
  *result = iree_vm_ref_from_ptr_move(&pointer, &object_type);
  return iree_ok_status();
}

iree_status_t object_consume(iree_vm_ref_t* owner, uint32_t failure) {
  ++calls;
  iree_vm_ref_reset(owner);
  return failure == 2 ? failure_status : iree_ok_status();
}

iree_status_t object_inspect(uint64_t count, const iree_vm_ref_t* values,
                             uint32_t failure) {
  inspected_count = count;
  inspected_non_null_count = 0;
  for (uint64_t i = 0; i < count; ++i) {
    if (!iree_vm_ref_is_null(values[i])) {
      if (!(values[i].type_and_state & IREE_VM_REF_STATE_BORROWED)) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "borrowed column acquired an owner");
      }
      ++inspected_non_null_count;
    }
  }
  return failure ? failure_status : iree_ok_status();
}

iree_status_t object_read(result_t* result, const iree_vm_ref_t* pointer,
                          uint64_t seed, uint32_t failure) {
  ++calls;
  if (failure == 3) {
    return failure_status;
  }
  void* object = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_ptr_from_ref_borrowed(*pointer, &object_type, &object));
  result->value = ((object_t*)object)->value + seed;
  for (unsigned i = 0; i < 8; ++i) {
    result->wide[i] = (float)(seed * 7 + i);
  }
  return iree_ok_status();
}

#define CHECK(condition)                                      \
  do {                                                        \
    if (!(condition)) {                                       \
      fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
      passed = false;                                         \
    }                                                         \
  } while (0)

int main(void) {
  bool passed = true;
  failure_status =
      iree_make_status(IREE_STATUS_DATA_LOSS, "owned failure payload");
  for (unsigned failure = 0; failure <= 3; ++failure) {
    calls = destroyed = 0;
    result_t result;
    memset(&result, 0xCD, sizeof(result));
    unsigned char sentinel[sizeof(result)];
    memset(sentinel, 0xCD, sizeof(sentinel));
    iree_status_t status = object_forward(&result, failure, 31);
    CHECK(status == (failure ? failure_status : iree_ok_status()));
    CHECK(calls == (failure ? failure : 3));
    CHECK(destroyed == (failure == 1 ? 0 : 1));
    if (failure) {
      CHECK(memcmp(&result, sentinel, sizeof(result)) == 0);
    } else {
      CHECK(result.value == 154);
      for (unsigned i = 0; i < 8; ++i) {
        CHECK(result.wide[i] == (float)(217 + i));
      }
    }
  }
  iree_status_ignore(failure_status);
  failure_status = iree_ok_status();

  destroyed = 0;
  iree_vm_ref_t owner = iree_vm_ref_null();
  CHECK(iree_status_is_ok(object_create(&owner, 0)));
  const iree_vm_ref_t borrowed =
      iree_vm_ref_from_ptr_borrowed(owner.object, &object_type);
  identity_result_t identity;
  CHECK(iree_status_is_ok(object_identity(&identity, &owner, &borrowed)));
  CHECK(iree_vm_ref_is_null(owner));
  CHECK(identity.same && !identity.missing && destroyed == 0);
  escape_result_t escaped;
  CHECK(iree_status_is_ok(object_escape(&escaped, &borrowed)));
  CHECK(escaped.marker == 42 &&
        escaped.pointer.object == identity.owner.object);
  iree_vm_ref_reset(&identity.owner);
  CHECK(destroyed == 0);
  iree_vm_ref_reset(&escaped.pointer);
  CHECK(destroyed == 1);

  _Alignas(16) unsigned char external[8];
  const iree_vm_ref_t scoped =
      iree_vm_ref_from_ptr_borrowed(external + 1, &borrowed_type);
  memset(&escaped, 0xCD, sizeof(escaped));
  unsigned char sentinel[sizeof(escaped)];
  memset(sentinel, 0xCD, sizeof(sentinel));
  iree_status_t status = object_escape(&escaped, &scoped);
  CHECK(iree_status_code(status) == IREE_STATUS_FAILED_PRECONDITION);
  iree_status_ignore(status);
  CHECK(memcmp(&escaped, sentinel, sizeof(escaped)) == 0);

  CHECK(iree_status_is_ok(object_create(&owner, 0)));
  iree_vm_ref_t selected = iree_vm_ref_null();
  CHECK(iree_status_is_ok(object_select(&selected, true, &owner, &scoped)));
  CHECK(selected.object == owner.object);
  iree_vm_ref_reset(&selected);
  status = object_select(&selected, false, &owner, &scoped);
  CHECK(iree_status_code(status) == IREE_STATUS_FAILED_PRECONDITION);
  iree_status_ignore(status);
  CHECK(iree_vm_ref_is_null(selected));
  CHECK(iree_status_is_ok(object_stack(&selected, &scoped, &scoped, &scoped,
                                       &scoped, &scoped, &scoped, &owner)));
  CHECK(selected.object == owner.object);
  iree_vm_ref_reset(&selected);
  null_result_t pair;
  CHECK(iree_status_is_ok(object_twice(&pair, &owner)));
  CHECK(pair.owner.object == owner.object &&
        pair.pointer.object == owner.object);
  iree_vm_ref_reset(&owner);
  CHECK(destroyed == 1);
  iree_vm_ref_reset(&pair.owner);
  CHECK(destroyed == 1);
  iree_vm_ref_reset(&pair.pointer);
  CHECK(destroyed == 2);

  CHECK(iree_status_is_ok(object_create(&owner, 0)));
  CHECK(iree_status_is_ok(object_drop_input(&owner)));
  CHECK(iree_vm_ref_is_null(owner) && destroyed == 3);
  CHECK(iree_status_is_ok(object_drop_result()));
  CHECK(destroyed == 4);

  null_result_t empty;
  CHECK(iree_status_is_ok(object_null(&empty)));
  CHECK(iree_vm_ref_is_null(empty.owner) && iree_vm_ref_is_null(empty.pointer));
  if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
      __builtin_cpu_supports("avx512dq") &&
      __builtin_cpu_supports("avx512vl")) {
    int32_t left[16], right[16], output[16];
    for (unsigned i = 0; i < 16; ++i) {
      left[i] = (int32_t)i * 3 - 20;
      right[i] = 14 - (int32_t)i;
    }
    CHECK(iree_status_is_ok(object_create(&owner, 0)));
    wide_clobbers = true;
    CHECK(iree_status_is_ok(
        object_preserve_vectors(&owner, left, right, output)));
    CHECK(iree_vm_ref_is_null(owner) && destroyed == 5);
    for (unsigned i = 0; i < 16; ++i) {
      CHECK(output[i] == (left[i] < right[i] ? left[i] : right[i]));
    }
  } else {
    fprintf(stderr, "AVX-512 execution skipped: host lacks the required ISA\n");
  }
  for (unsigned rotations = 0; rotations < 5; ++rotations) {
    const unsigned destroyed_before = destroyed;
    iree_vm_ref_t first = iree_vm_ref_null(), second = iree_vm_ref_null();
    CHECK(iree_status_is_ok(object_create(&first, 0)));
    CHECK(iree_status_is_ok(object_create(&second, 0)));
    void* first_object = first.object;
    void* second_object = second.object;
    CHECK(iree_status_is_ok(object_rotate(&pair, &first, &second, rotations)));
    CHECK(iree_vm_ref_is_null(first) && iree_vm_ref_is_null(second));
    CHECK(pair.owner.object == (rotations % 2 ? second_object : first_object));
    CHECK(pair.pointer.object ==
          (rotations % 2 ? first_object : second_object));
    CHECK(destroyed == destroyed_before);
    iree_vm_ref_reset(&pair.owner);
    iree_vm_ref_reset(&pair.pointer);
    CHECK(destroyed == destroyed_before + 2);
  }
  failure_status =
      iree_make_status(IREE_STATUS_DATA_LOSS, "storage cleanup failure");
  for (unsigned failure = 0; failure <= 2; failure += 2) {
    const unsigned destroyed_before = destroyed;
    iree_vm_ref_t first = iree_vm_ref_null(), second = iree_vm_ref_null();
    iree_vm_ref_t result = iree_vm_ref_null();
    CHECK(iree_status_is_ok(object_create(&first, 0)));
    CHECK(iree_status_is_ok(object_create(&second, 0)));
    void* first_object = first.object;
    iree_status_t storage_status =
        object_storage(&result, &first, &second, failure);
    CHECK(storage_status == (failure ? failure_status : iree_ok_status()));
    CHECK(iree_vm_ref_is_null(first) && iree_vm_ref_is_null(second));
    CHECK(destroyed == destroyed_before + (failure ? 2 : 1));
    if (failure) {
      CHECK(iree_vm_ref_is_null(result));
    } else {
      CHECK(result.object == first_object);
      uint32_t same = 0;
      CHECK(iree_status_is_ok(object_storage_borrowed(&same, &result)));
      CHECK(same == 1 && destroyed == destroyed_before + 1);
      iree_vm_ref_reset(&result);
      CHECK(destroyed == destroyed_before + 2);
    }
  }
  iree_status_ignore(failure_status);
  uint32_t same = 0;
  CHECK(iree_status_is_ok(object_storage_borrowed(&same, &scoped)));
  CHECK(same == 1);
  for (unsigned count = 1; count <= 2; ++count) {
    const unsigned destroyed_before = destroyed;
    iree_vm_ref_t first = iree_vm_ref_null(), second = iree_vm_ref_null();
    CHECK(iree_status_is_ok(object_create(&first, 0)));
    if (count == 2) {
      CHECK(iree_status_is_ok(object_create(&second, 0)));
    }
    void* first_object = first.object;
    void* second_object = second.object;
    storage_result_t result = {0};
    CHECK(iree_status_is_ok(object_storage_group(&result, &first, &second)));
    CHECK(iree_vm_ref_is_null(first) && iree_vm_ref_is_null(second));
    CHECK(result.first.object == first_object);
    CHECK(result.second.object == second_object);
    CHECK(result.missing == 1 && destroyed == destroyed_before);
    iree_vm_ref_reset(&result.first);
    iree_vm_ref_reset(&result.second);
    CHECK(destroyed == destroyed_before + count);
  }
  for (unsigned count = 0; count < 8; ++count) {
    const unsigned destroyed_before = destroyed;
    CHECK(iree_status_is_ok(object_create(&owner, 0)));
    uint32_t observed = UINT32_MAX;
    CHECK(iree_status_is_ok(object_view_reuse(&observed, &owner, count)));
    CHECK(observed == count && destroyed == destroyed_before + 1);
    CHECK(iree_vm_ref_is_null(owner));
  }
  {
    const unsigned destroyed_before = destroyed;
    CHECK(iree_status_is_ok(object_create(&owner, 0)));
    const iree_vm_ref_t values[] = {
        iree_vm_ref_null(), scoped,
        iree_vm_ref_from_ptr_borrowed(owner.object, &object_type)};
    for (unsigned position = 0; position < 4; ++position) {
      iree_vm_ref_t result = iree_vm_ref_null();
      iree_status_t slice_status =
          object_view_slice(&result, 3, values, position);
      if (position == 1) {
        CHECK(iree_status_is_failed_precondition(slice_status));
        iree_status_ignore(slice_status);
        CHECK(iree_vm_ref_is_null(result));
      } else {
        CHECK(iree_status_is_ok(slice_status));
        CHECK(result.object == (position == 2 ? owner.object : NULL));
      }
      iree_vm_ref_reset(&result);
      CHECK(destroyed == destroyed_before);
    }
    iree_vm_ref_t cells[3] = {0};
    for (unsigned position = 0; position < 3; ++position) {
      CHECK(iree_status_is_ok(object_create(&cells[position], 0)));
    }
    void* replacement = owner.object;
    CHECK(iree_status_is_ok(object_view_replace(&owner, cells, 1)));
    CHECK(iree_vm_ref_is_null(owner) && cells[1].object == replacement);
    CHECK(destroyed == destroyed_before + 1);
    CHECK(iree_status_is_ok(object_storage_clear(cells)));
    CHECK(iree_vm_ref_is_null(cells[0]) && destroyed == destroyed_before + 2);
    CHECK(iree_status_is_ok(object_storage_clear_many(cells)));
    CHECK(destroyed == destroyed_before + 4);
    for (unsigned position = 0; position < 3; ++position) {
      CHECK(iree_vm_ref_is_null(cells[position]));
    }
  }
  failure_status =
      iree_make_status(IREE_STATUS_DATA_LOSS, "counted call failure");
  iree_vm_ref_t borrowed_cells[67];
  for (unsigned i = 0; i < IREE_ARRAYSIZE(borrowed_cells); ++i) {
    borrowed_cells[i] = i % 2 ? iree_vm_ref_null() : scoped;
  }
  const uint64_t counts[] = {0, 1, 17, 67};
  for (unsigned i = 0; i < IREE_ARRAYSIZE(counts); ++i) {
    for (unsigned failure = 0; failure < 2; ++failure) {
      iree_status_t call_status = object_view_call(
          counts[i], counts[i] ? borrowed_cells : NULL, failure);
      CHECK(call_status == (failure ? failure_status : iree_ok_status()));
      CHECK(inspected_count == counts[i]);
      CHECK(inspected_non_null_count == (counts[i] + 1) / 2);
    }
  }
  iree_status_ignore(failure_status);
  return passed ? 0 : 1;
}
