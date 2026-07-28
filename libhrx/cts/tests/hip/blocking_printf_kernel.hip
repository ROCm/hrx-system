// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

extern "C" __attribute__((device)) int printf(const char* format, ...);

// Exercises LLVM's default blocking printf lowering and records both its return
// value and proof that the work-item resumed after the call.
extern "C" __attribute__((global)) void hrx_blocking_printf(unsigned int value,
                                                            int* result) {
  result[0] = printf("hrx cts printf value=%u\n", value);
  result[1] = 0xC0FFEE;
}

// Produces a valid printf result larger than one MiB.
extern "C" __attribute__((global)) void hrx_blocking_printf_large_output(
    int* result) {
  result[0] = printf("%1048577u", 7u);
}

#define HRX_PRINTF_ZEROS_64 \
  "0000000000000000000000000000000000000000000000000000000000000000"
#define HRX_PRINTF_ZEROS_1024                                                 \
  HRX_PRINTF_ZEROS_64 HRX_PRINTF_ZEROS_64 HRX_PRINTF_ZEROS_64                 \
      HRX_PRINTF_ZEROS_64 HRX_PRINTF_ZEROS_64 HRX_PRINTF_ZEROS_64             \
          HRX_PRINTF_ZEROS_64 HRX_PRINTF_ZEROS_64 HRX_PRINTF_ZEROS_64         \
              HRX_PRINTF_ZEROS_64 HRX_PRINTF_ZEROS_64 HRX_PRINTF_ZEROS_64     \
                  HRX_PRINTF_ZEROS_64 HRX_PRINTF_ZEROS_64 HRX_PRINTF_ZEROS_64 \
                      HRX_PRINTF_ZEROS_64

// Forces every work-item through a message requiring a 2048-byte vector in
// CLR's power-of-two representation while producing only one output byte.
extern "C" __attribute__((global)) void hrx_blocking_printf_many_workitems(
    unsigned int workgroup_size, unsigned int workitem_count, int* results) {
  const unsigned int index =
      __builtin_amdgcn_workgroup_id_x() * workgroup_size +
      __builtin_amdgcn_workitem_id_x();
  if (index >= workitem_count) return;
  results[index] = printf("%" HRX_PRINTF_ZEROS_1024 "1u", 7u);
}

#undef HRX_PRINTF_ZEROS_1024
#undef HRX_PRINTF_ZEROS_64
