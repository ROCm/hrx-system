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

int iree_clang_tidy_style_direct_goto(int flag) {
  if (flag) goto cleanup;
  return 0;
cleanup:
  return 1;
}

#define IREE_CLANG_TIDY_STYLE_LOCAL_GOTO_MACRO goto macro_cleanup
int iree_clang_tidy_style_local_goto_macro(int flag) {
  if (flag) {
    IREE_CLANG_TIDY_STYLE_LOCAL_GOTO_MACRO;
  }
  return 0;
macro_cleanup:
  return 1;
}

#if defined(__GNUC__)
int iree_clang_tidy_style_computed_goto(int index) {
  static void* labels[] = {&&dispatch_zero, &&dispatch_one};
  goto* labels[index & 1];
dispatch_zero:
  return 0;
dispatch_one:
  return 1;
}
#endif  // defined(__GNUC__)

typedef struct iree_clang_tidy_style_resource_t
    iree_clang_tidy_style_resource_t;
typedef struct iree_clang_tidy_style_pool_t iree_clang_tidy_style_pool_t;
typedef unsigned long iree_clang_tidy_style_handle_t;
typedef int iree_status_t;

typedef struct {
  int value;
} iree_clang_tidy_style_c_designated_init_t;

iree_clang_tidy_style_c_designated_init_t
    iree_clang_tidy_style_c_designated_init = {
        .value = 1,
};

void iree_clang_tidy_style_resource_release(
    iree_clang_tidy_style_resource_t* resource);
void iree_clang_tidy_style_resource_deinitialize(
    iree_clang_tidy_style_resource_t* resource);
void iree_clang_tidy_style_resource_destroy(
    iree_clang_tidy_style_resource_t* resource);
void iree_clang_tidy_style_pool_release(iree_clang_tidy_style_pool_t* pool,
                                        void* entry);
void iree_clang_tidy_style_handle_release(
    iree_clang_tidy_style_handle_t handle);
void iree_clang_tidy_style_extra_cleanup(void);

typedef void (*iree_clang_tidy_style_release_fn_t)(
    iree_clang_tidy_style_resource_t* resource);

void iree_clang_tidy_style_guarded_release(
    iree_clang_tidy_style_resource_t* resource) {
  if (resource) iree_clang_tidy_style_resource_release(resource);
}

void iree_clang_tidy_style_guarded_release_with_clear(
    iree_clang_tidy_style_resource_t* resource) {
  if (resource != NULL) {
    iree_clang_tidy_style_resource_release(resource);
    resource = NULL;
  }
}

void iree_clang_tidy_style_null_guard_ignored(
    iree_clang_tidy_style_resource_t* resource) {
  if (resource == NULL) {
    iree_clang_tidy_style_resource_release(resource);
  }
}

void iree_clang_tidy_style_different_predicate_ignored(
    int owns_resource, iree_clang_tidy_style_resource_t* resource) {
  if (owns_resource) {
    iree_clang_tidy_style_resource_release(resource);
  }
}

void iree_clang_tidy_style_extra_behavior_ignored(
    iree_clang_tidy_style_resource_t* resource) {
  if (resource) {
    iree_clang_tidy_style_resource_release(resource);
    iree_clang_tidy_style_extra_cleanup();
  }
}

void iree_clang_tidy_style_deinitialize_ignored(
    iree_clang_tidy_style_resource_t* resource) {
  if (resource) {
    iree_clang_tidy_style_resource_deinitialize(resource);
  }
}

void iree_clang_tidy_style_destroy_ignored(
    iree_clang_tidy_style_resource_t* resource) {
  if (resource) {
    iree_clang_tidy_style_resource_destroy(resource);
  }
}

void iree_clang_tidy_style_multi_argument_release_ignored(
    iree_clang_tidy_style_pool_t* pool, void* entry) {
  if (pool) {
    iree_clang_tidy_style_pool_release(pool, entry);
  }
}

void iree_clang_tidy_style_non_pointer_release_ignored(
    iree_clang_tidy_style_handle_t handle) {
  if (handle) {
    iree_clang_tidy_style_handle_release(handle);
  }
}

void iree_clang_tidy_style_indirect_release_ignored(
    iree_clang_tidy_style_resource_t* resource,
    iree_clang_tidy_style_release_fn_t release_fn) {
  if (resource) {
    release_fn(resource);
  }
}

#define EXPECT_TRUE(expr) ((void)(expr))
#define ASSERT_TRUE(expr) ((void)(expr))
#define EXPECT_FALSE(expr) ((void)(expr))
#define ASSERT_FALSE(expr) ((void)(expr))

int iree_status_is_ok(iree_status_t status);
int iree_clang_tidy_style_some_predicate(void);

void iree_clang_tidy_style_raw_status_test_predicates_are_invalid(
    iree_status_t status) {
  EXPECT_TRUE(iree_status_is_ok(status));
  ASSERT_TRUE(iree_status_is_ok(status));
  EXPECT_FALSE(iree_status_is_ok(status));
  ASSERT_FALSE(iree_status_is_ok(status));
}

void iree_clang_tidy_style_boolean_test_predicates_are_allowed(void) {
  EXPECT_TRUE(iree_clang_tidy_style_some_predicate());
  ASSERT_FALSE(iree_clang_tidy_style_some_predicate());
}
