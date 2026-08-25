// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#define HRX_CTS_TEST_ATTRIBUTE_KERNEL \
  [[clang::amdgpu_kernel, gnu::visibility("protected")]]

HRX_CTS_TEST_ATTRIBUTE_KERNEL void hrx_noop(void) {}

HRX_CTS_TEST_ATTRIBUTE_KERNEL void hrx_store_output(unsigned int* output,
                                                    unsigned int value) {
  output[0] = value;
}

typedef struct hrx_nested_pointer_arguments_t {
  unsigned int* input;
  unsigned int* output;
} hrx_nested_pointer_arguments_t;

// Exercises HIP's native argument packing with device pointers nested inside
// one by-value kernel argument.
HRX_CTS_TEST_ATTRIBUTE_KERNEL void hrx_transform_nested_pointers(
    hrx_nested_pointer_arguments_t pointers, unsigned int scale,
    unsigned int offset) {
  for (unsigned int i = 0; i < 4; ++i) {
    pointers.output[i] = pointers.input[i] * scale + offset;
  }
}

// Occupies the device for a duration scaling with |iterations| by stepping a
// linear congruential recurrence. Each step consumes the previous accumulator,
// so the chain cannot be vectorized or reassociated: it retires no faster than
// one step per engine cycle however wide the device is.
HRX_CTS_TEST_ATTRIBUTE_KERNEL void hrx_spin_dependent_chain(
    unsigned long long* output, unsigned long long iterations) {
  unsigned long long accumulator = 1;
  for (unsigned long long i = 0; i < iterations; ++i) {
    accumulator = accumulator * 6364136223846793005ULL + 1442695040888963407ULL;
  }
  output[0] = accumulator;
}
