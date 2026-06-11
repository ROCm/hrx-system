// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef NULL
#define NULL ((void*)0)
#endif  // NULL

#ifndef IREE_LIKELY
#define IREE_LIKELY(x) (x)
#endif  // IREE_LIKELY

#ifndef IREE_ASSERT_ARGUMENT
#define IREE_ASSERT_ARGUMENT(x) ((void)(x))
#endif  // IREE_ASSERT_ARGUMENT

typedef int iree_atomic_ref_count_t;
typedef int iree_status_t;

typedef struct iree_clang_tidy_refcounted_t {
  iree_atomic_ref_count_t ref_count;
} iree_clang_tidy_refcounted_t;

typedef struct iree_clang_tidy_refcounted_misaligned_t {
  int state;
  iree_atomic_ref_count_t ref_count;
} iree_clang_tidy_refcounted_misaligned_t;

typedef struct iree_clang_tidy_refcount_misused_counter_t {
  iree_atomic_ref_count_t pending_submissions;
} iree_clang_tidy_refcount_misused_counter_t;

typedef struct iree_clang_tidy_refcounted_with_counter_t {
  iree_atomic_ref_count_t ref_count;
  iree_atomic_ref_count_t queued_callbacks;
} iree_clang_tidy_refcounted_with_counter_t;

typedef struct iree_clang_tidy_not_refcount_anchored_t {
  int state;
  iree_atomic_ref_count_t side_counter;
} iree_clang_tidy_not_refcount_anchored_t;

void iree_atomic_ref_count_inc(iree_atomic_ref_count_t* ref_count);
int iree_atomic_ref_count_dec(iree_atomic_ref_count_t* ref_count);
iree_status_t iree_ok_status(void);

iree_status_t iree_clang_tidy_refcount_status_retain(
    iree_clang_tidy_refcounted_t* resource) {
  iree_atomic_ref_count_inc(&resource->ref_count);
  return iree_ok_status();
}

iree_status_t iree_clang_tidy_refcount_status_release(
    iree_clang_tidy_refcounted_t* resource) {
  if (iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
    return iree_ok_status();
  }
  return iree_ok_status();
}

void iree_clang_tidy_refcount_void_retain(
    iree_clang_tidy_refcounted_t* resource) {
  iree_atomic_ref_count_inc(&resource->ref_count);
}

void iree_clang_tidy_refcount_void_release(
    iree_clang_tidy_refcounted_t* resource) {
  if (!resource) {
    return;
  }
  if (iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
  }
}

void iree_clang_tidy_refcount_observe(iree_clang_tidy_refcounted_t* resource) {
  (void)resource;
}

void iree_clang_tidy_refcount_unguarded_release(
    iree_clang_tidy_refcounted_t* resource) {
  if (iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
  }
}

void iree_clang_tidy_refcount_asserting_release(
    iree_clang_tidy_refcounted_t* resource) {
  IREE_ASSERT_ARGUMENT(resource);
  if (resource && iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
  }
}

void iree_clang_tidy_refcount_ignored_dec(
    iree_clang_tidy_refcounted_t* resource) {
  (void)iree_atomic_ref_count_dec(&resource->ref_count);
}

void iree_clang_tidy_refcount_early_null_release(
    iree_clang_tidy_refcounted_t* resource) {
  if (!resource) {
    return;
  }
  if (iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
  }
}

void iree_clang_tidy_refcount_inline_null_release(
    iree_clang_tidy_refcounted_t* resource) {
  if (resource && iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
  }
}

void iree_clang_tidy_refcount_likely_null_release(
    iree_clang_tidy_refcounted_t* resource) {
  if (IREE_LIKELY(resource) &&
      iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
  }
}

void iree_clang_tidy_refcount_release_then_deref(
    iree_clang_tidy_refcounted_t* resource) {
  iree_clang_tidy_refcount_void_release(resource);
  resource->ref_count = 0;
}

void iree_clang_tidy_refcount_release_then_indirect_deref(
    iree_clang_tidy_refcounted_t* resource) {
  iree_clang_tidy_refcount_void_release(resource);
  (*resource).ref_count = 0;
}

void iree_clang_tidy_refcount_release_then_array_use(
    iree_clang_tidy_refcounted_t* resources) {
  iree_clang_tidy_refcount_void_release(resources);
  resources[0].ref_count = 0;
}

void iree_clang_tidy_refcount_release_then_release(
    iree_clang_tidy_refcounted_t* resource) {
  iree_clang_tidy_refcount_void_release(resource);
  iree_clang_tidy_refcount_void_release(resource);
}

void iree_clang_tidy_refcount_release_then_retain(
    iree_clang_tidy_refcounted_t* retained_resource) {
  iree_clang_tidy_refcount_void_release(retained_resource);
  iree_clang_tidy_refcount_void_retain(retained_resource);
}

void iree_clang_tidy_refcount_release_then_call(
    iree_clang_tidy_refcounted_t* resource) {
  iree_clang_tidy_refcount_void_release(resource);
  iree_clang_tidy_refcount_observe(resource);
}

void iree_clang_tidy_refcount_release_then_cast_call(
    iree_clang_tidy_refcounted_t* cast_resource) {
  iree_clang_tidy_refcount_void_release(cast_resource);
  iree_clang_tidy_refcount_observe(
      (iree_clang_tidy_refcounted_t*)cast_resource);
}

void iree_clang_tidy_refcount_release_then_assign(
    iree_clang_tidy_refcounted_t* assigned_resource,
    iree_clang_tidy_refcounted_t** out_resource) {
  iree_clang_tidy_refcount_void_release(assigned_resource);
  *out_resource = assigned_resource;
}

void iree_clang_tidy_refcount_release_then_alias(
    iree_clang_tidy_refcounted_t* aliased_resource) {
  iree_clang_tidy_refcount_void_release(aliased_resource);
  iree_clang_tidy_refcounted_t* alias = aliased_resource;
  (void)alias;
}

iree_clang_tidy_refcounted_t* iree_clang_tidy_refcount_release_then_return(
    iree_clang_tidy_refcounted_t* returned_resource) {
  iree_clang_tidy_refcount_void_release(returned_resource);
  return returned_resource;
}

void iree_clang_tidy_refcount_retain_then_double_release(
    iree_clang_tidy_refcounted_t* resource) {
  iree_clang_tidy_refcount_void_retain(resource);
  iree_clang_tidy_refcount_void_release(resource);
  iree_clang_tidy_refcount_void_release(resource);
}

void iree_clang_tidy_refcount_retain_then_double_release_then_deref(
    iree_clang_tidy_refcounted_t* resource) {
  iree_clang_tidy_refcount_void_retain(resource);
  iree_clang_tidy_refcount_void_release(resource);
  iree_clang_tidy_refcount_void_release(resource);
  resource->ref_count = 0;
}

void iree_clang_tidy_refcount_release_then_clear(
    iree_clang_tidy_refcounted_t* resource) {
  iree_clang_tidy_refcount_void_release(resource);
  resource = NULL;
}

void iree_clang_tidy_refcount_release_then_replace(
    iree_clang_tidy_refcounted_t* resource,
    iree_clang_tidy_refcounted_t* replacement) {
  iree_clang_tidy_refcount_void_release(resource);
  resource = replacement;
  resource->ref_count = 0;
}

void iree_clang_tidy_refcount_guarded_release(
    iree_clang_tidy_refcounted_t* guarded_resource) {
  if (guarded_resource) {
    iree_clang_tidy_refcount_void_release(guarded_resource);
  }
}

void iree_clang_tidy_refcount_guarded_release_null_comparison(
    iree_clang_tidy_refcounted_t* null_compared_resource) {
  if (null_compared_resource != NULL) {
    iree_clang_tidy_refcount_void_release(null_compared_resource);
  }
}

void iree_clang_tidy_refcount_guarded_release_reversed_null_comparison(
    iree_clang_tidy_refcounted_t* reversed_null_compared_resource) {
  if (NULL != reversed_null_compared_resource) {
    iree_clang_tidy_refcount_void_release(reversed_null_compared_resource);
  }
}

void iree_clang_tidy_refcount_conditional_release(
    iree_clang_tidy_refcounted_t* conditionally_released_resource,
    int condition) {
  if (condition) {
    iree_clang_tidy_refcount_void_release(conditionally_released_resource);
  }
}

void iree_clang_tidy_refcount_release_in_branch_then_use(
    iree_clang_tidy_refcounted_t* resource, int condition) {
  if (condition) {
    iree_clang_tidy_refcount_void_release(resource);
  }
  resource->ref_count = 0;
}

void iree_clang_tidy_refcount_local_counter_release(
    iree_clang_tidy_refcounted_t* resource) {
  iree_atomic_ref_count_t* ref_count = &resource->ref_count;
  if (iree_atomic_ref_count_dec(ref_count) == 1) {
  }
}

iree_status_t iree_clang_tidy_refcount_lookup_retain(
    iree_clang_tidy_refcounted_t* resource) {
  (void)resource;
  return iree_ok_status();
}

iree_status_t iree_clang_tidy_virtual_memory_release(void* memory) {
  (void)memory;
  return iree_ok_status();
}

void iree_clang_tidy_refcount_status_release_then_use(
    iree_clang_tidy_refcounted_t* status_released_resource) {
  (void)iree_clang_tidy_virtual_memory_release(status_released_resource);
  iree_clang_tidy_refcount_observe(status_released_resource);
}

void iree_clang_tidy_refcount_guarded_status_release(
    iree_clang_tidy_refcounted_t* guarded_status_released_resource) {
  if (guarded_status_released_resource) {
    (void)iree_clang_tidy_virtual_memory_release(
        guarded_status_released_resource);
  }
}
