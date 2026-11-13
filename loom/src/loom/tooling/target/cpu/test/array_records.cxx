// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Command {
  // Independently copied source words.
  unsigned words[2];
};
static unsigned change(Command value, unsigned index) {
  value.words[index] += 3u;
  return value.words[index];
}
[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void array_records_kernel(const Command* input, Command* output) {
  Command saved = input[0];
  unsigned changed = change(saved, input[1].words[0] & 1u);
  saved.words[0] += 5u;
  output[0] = saved;
  output[1].words[0] = changed;
  output[1].words[1] = input[0].words[1];
}
