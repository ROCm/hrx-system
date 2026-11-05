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

#ifndef IREE_ASSERT
#define IREE_ASSERT(condition, ...) \
  while (0 && (condition)) {        \
  }
#endif  // IREE_ASSERT
#ifndef IREE_ASSERT_ARGUMENT
#define IREE_ASSERT_ARGUMENT(x) IREE_ASSERT(x)
#endif  // IREE_ASSERT_ARGUMENT
#ifndef IREE_ASSERT_TRUE
#define IREE_ASSERT_TRUE(x, ...) IREE_ASSERT(!!(x), __VA_ARGS__)
#endif  // IREE_ASSERT_TRUE
#ifndef IREE_ASSERT_EQ
#define IREE_ASSERT_EQ(x, y, ...) IREE_ASSERT((x) == (y), __VA_ARGS__)
#endif  // IREE_ASSERT_EQ

void output_order_declaration(int* out_declaration, int input);

void output_order_definition(int* out_definition, int* state) {
  *out_definition = *state;
}

typedef void (*output_order_callback_t)(void* user_data, int* out_callback,
                                        void* scope);
typedef void output_order_function_t(int* out_function_type, int input);

struct output_order_vtable_t {
  void (*fn)(int* out_field, int input);
};

void output_order_nested(void (*callback)(int* out_nested, void* scope));
void (*output_order_variable)(int* out_variable, int input);
void output_order_array(int out_array[4], int input);
void output_order_unnamed(int* out_unnamed, int);
void output_order_interleaved(int* out_first, int input, int* out_second);

#define OUTPUT_ORDER_DECLARATION(name) void name(int* out_macro, int input)
OUTPUT_ORDER_DECLARATION(output_order_macro);

void output_order_valid_inputs(int input, int* state);
void output_order_valid_empty(void);
void output_order_valid_single(int* out_value);
void output_order_valid_multiple(void* user_data, int* state, int flags,
                                 int* out_wait, int* out_copy);
typedef void (*output_order_valid_callback_t)(void* user_data, void* scope,
                                              int* out_wait, int* out_copy);
typedef void (*output_order_valid_unnamed_t)(void*, int*, int*);

int iree_clang_tidy_style_direct_goto(int flag) {
  if (flag) {
    goto cleanup;
  }
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

int iree_clang_tidy_style_query(int input, int* out_output);
int iree_clang_tidy_style_const_query(const int* input);
int iree_clang_tidy_style_mutable_query(int* state);

int iree_clang_tidy_style_assert_output_call(int input) {
  int output = 0;
  IREE_ASSERT(iree_clang_tidy_style_query(input, &output));
  IREE_ASSERT_TRUE(iree_clang_tidy_style_query(input, &output));
  IREE_ASSERT_EQ(iree_clang_tidy_style_query(input, &output), 1);
  return output;
}

int iree_clang_tidy_style_assert_caller_output_call(int input,
                                                    int* out_output) {
  IREE_ASSERT_TRUE(iree_clang_tidy_style_query(input, out_output));
  return *out_output;
}

int iree_clang_tidy_style_assert_after_output_call(int input) {
  int output = 0;
  int matched = iree_clang_tidy_style_query(input, &output);
  IREE_ASSERT_TRUE(matched);
  return output;
}

int iree_clang_tidy_style_assert_const_call(int input) {
  IREE_ASSERT_TRUE(iree_clang_tidy_style_const_query(&input));
  return input;
}

int iree_clang_tidy_style_assert_mutable_non_output_call(int state) {
  IREE_ASSERT_TRUE(iree_clang_tidy_style_mutable_query(&state));
  return state;
}

int iree_clang_tidy_style_assert_null_output_call(int input) {
  IREE_ASSERT_TRUE(iree_clang_tidy_style_query(input, NULL));
  return input;
}

typedef struct {
  int value;
} iree_clang_tidy_style_c_designated_init_t;

iree_clang_tidy_style_c_designated_init_t
    iree_clang_tidy_style_c_designated_init = {
        .value = 1,
};

void iree_clang_tidy_style_resource_release(
    iree_clang_tidy_style_resource_t* resource);
void iree_clang_tidy_style_resource_await_release(
    iree_clang_tidy_style_resource_t* resource);
void iree_clang_tidy_style_resource_wait_release(
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

// Exercises fixes for an unbraced release guard.
// clang-format off
void iree_clang_tidy_style_guarded_release(
    iree_clang_tidy_style_resource_t* resource) {
  if (resource) iree_clang_tidy_style_resource_release(resource);
}
// clang-format on

void iree_clang_tidy_style_guarded_release_with_clear(
    iree_clang_tidy_style_resource_t* resource) {
  if (resource != NULL) {
    iree_clang_tidy_style_resource_release(resource);
    resource = NULL;
  }
}

void iree_clang_tidy_style_guarded_wait_ignored(
    iree_clang_tidy_style_resource_t* awaited_resource,
    iree_clang_tidy_style_resource_t* waited_resource) {
  if (awaited_resource) {
    iree_clang_tidy_style_resource_await_release(awaited_resource);
  }
  if (waited_resource) {
    iree_clang_tidy_style_resource_wait_release(waited_resource);
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

typedef struct iree_clang_tidy_style_lease_t {
  // Whether the caller-owned storage currently holds a scratch lease.
  int acquired;
} iree_clang_tidy_style_lease_t;
void iree_clang_tidy_style_lease_release(iree_clang_tidy_style_lease_t* lease);
void iree_clang_tidy_style_guarded_lease(iree_clang_tidy_style_lease_t* lease) {
  if (lease) {
    iree_clang_tidy_style_lease_release(lease);
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
