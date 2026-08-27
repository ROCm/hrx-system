// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

static int direct_recursion(int value) {
  return value == 0 ? 0 : direct_recursion(value - 1);
}

static int mutual_recursion_b(int value);

static int mutual_recursion_a(int value) {
  return value == 0 ? 0 : mutual_recursion_b(value - 1);
}

static int mutual_recursion_b(int value) {
  return value == 0 ? 0 : mutual_recursion_a(value - 1);
}

typedef int (*recursion_callback_t)(int value);

static int invoke_callback(recursion_callback_t callback, int value) {
  return callback(value);
}

static int parameter_callback_b(int value);

static int parameter_callback_a(int value) {
  return value == 0 ? 0 : invoke_callback(parameter_callback_b, value - 1);
}

static int parameter_callback_b(int value) {
  return value == 0 ? 0 : invoke_callback(parameter_callback_a, value - 1);
}

static int second_parameter_callback_b(int value);

static int second_parameter_callback_a(int value) {
  return value == 0 ? 0
                    : invoke_callback(second_parameter_callback_b, value - 1);
}

static int second_parameter_callback_b(int value) {
  return value == 0 ? 0
                    : invoke_callback(second_parameter_callback_a, value - 1);
}

typedef struct callback_record_t {
  recursion_callback_t fn;
  const void* user_data;
} callback_record_t;

static int invoke_record(callback_record_t callback, int value) {
  return callback.fn(value);
}

static int aggregate_callback_b(int value);

static int aggregate_callback_a(int value) {
  callback_record_t callback = {
      .fn = aggregate_callback_b,
  };
  return value == 0 ? 0 : invoke_record(callback, value - 1);
}

static int aggregate_callback_b(int value) {
  callback_record_t callback = {
      .fn = aggregate_callback_a,
  };
  return value == 0 ? 0 : invoke_record(callback, value - 1);
}

static int table_callback_a(int value);
static int table_callback_b(int value);

static const recursion_callback_t kCallbackTable[] = {
    table_callback_a,
    table_callback_b,
};

static int dispatch_table_callback(unsigned index, int value) {
  return kCallbackTable[index](value);
}

static int table_callback_a(int value) {
  return value == 0 ? 0 : dispatch_table_callback(1, value - 1);
}

static int table_callback_b(int value) {
  return value == 0 ? 0 : dispatch_table_callback(0, value - 1);
}

static int explicit_worklist(int value) {
  int result = 0;
  while (value-- > 0) {
    ++result;
  }
  return result;
}

int recursion_check_entry(int value) {
  return direct_recursion(value) + mutual_recursion_a(value) +
         parameter_callback_a(value) + second_parameter_callback_a(value) +
         aggregate_callback_a(value) + table_callback_a(value) +
         explicit_worklist(value);
}
