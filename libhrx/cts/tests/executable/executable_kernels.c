// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#define HRX_CTS_TEST_ATTRIBUTE_KERNEL \
  [[clang::amdgpu_kernel, gnu::visibility("protected")]]

HRX_CTS_TEST_ATTRIBUTE_KERNEL void hrx_noop(void) {}

HRX_CTS_TEST_ATTRIBUTE_KERNEL void hrx_store_output(unsigned int* output,
                                                    unsigned int value) {
  output[0] = value;
}

// The leading 16-bit scalar forces native ABI padding before the first
// pointer. Metadata queries and argument-array launches must agree on that
// padded layout.
HRX_CTS_TEST_ATTRIBUTE_KERNEL void hrx_transform_padded_parameters(
    unsigned short bias, unsigned int* input, unsigned int scale,
    unsigned int* output) {
  output[0] = input[0] * scale + bias;
}

// Records every work-item instantiated by an exact-size dispatch. A launch of
// 100 work-items with a local size of 64 proves that the final workgroup has 36
// active lanes instead of being rejected or rounded up to 128 lanes.
HRX_CTS_TEST_ATTRIBUTE_KERNEL void hrx_store_exact_workitem_indices(
    unsigned int local_size, unsigned int* output) {
  const unsigned int index = __builtin_amdgcn_workgroup_id_x() * local_size +
                             __builtin_amdgcn_workitem_id_x();
  output[index] = index + 1;
}

// Reports the packet geometry so callers can distinguish an exact partial
// workgroup from a launch whose requested workgroup size was silently reduced.
HRX_CTS_TEST_ATTRIBUTE_KERNEL void hrx_report_dispatch_size(
    unsigned int* output) {
  if (__builtin_amdgcn_workgroup_id_x() == 0 &&
      __builtin_amdgcn_workitem_id_x() == 0) {
    output[0] = __builtin_amdgcn_grid_size_x();
    output[1] = __builtin_amdgcn_workgroup_size_x();
  }
}

[[gnu::used, gnu::visibility("protected")]] unsigned int hrx_device_global = 17;

typedef struct hrx_launch_gate_t {
  // Set once the gated kernel begins execution.
  unsigned int entered;
  // Set by the host to allow the gated kernel to finish.
  unsigned int released;
} hrx_launch_gate_t;

// Marks entry through coherent host memory and remains active until the host
// releases the launch. System-scope atomics make the gate observable across the
// host and every participating device.
HRX_CTS_TEST_ATTRIBUTE_KERNEL void hrx_gated_store_output(
    hrx_launch_gate_t* gate, unsigned int* output, unsigned int value) {
  __atomic_store_n(&gate->entered, 1, __ATOMIC_RELEASE);
  while (!__atomic_load_n(&gate->released, __ATOMIC_ACQUIRE)) {
    __builtin_amdgcn_s_sleep(1);
  }
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
