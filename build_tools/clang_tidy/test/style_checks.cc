// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct OutputOrder {
  OutputOrder(int* out_constructor, int input);
  void Method(int* out_method, int input);
  void (*callback)(int* out_member, int input);
};

using OutputOrderAlias = void (*)(int* out_alias, int input);

auto output_order_lambda = [](int* out_lambda, int input) {
  *out_lambda = input;
};

auto output_order_generic = [](auto* out_generic, auto input) {
  *out_generic = input;
};

template <typename T>
void OutputOrderTemplate(T* out_template, T input) {
  *out_template = input;
}

void InstantiateOutputOrder(int* value) {
  output_order_generic(value, 1);
  OutputOrderTemplate(value, 2);
}

void OutputOrderValid(int input, int& out_result, int* out_optional = nullptr);
auto output_order_valid_lambda = [](int input, int* out_result) {
  *out_result = input;
};
