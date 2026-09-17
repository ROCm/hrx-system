// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>
#include <hip/hip_ext.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" hipError_t hipExtModuleLaunchKernel(
    hipFunction_t f, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t lx,
    uint32_t ly, uint32_t lz, size_t sm, hipStream_t stream, void** params,
    void** extra, hipEvent_t start, hipEvent_t stop, uint32_t flags) {
  static auto real = (decltype(&hipExtModuleLaunchKernel))dlsym(
      RTLD_NEXT, "hipExtModuleLaunchKernel");
  if (!real) std::abort();
  static int count = 0;
  if (count++ == 0) {
    const char* name = hipKernelNameRef(f);
    printf("CAPTURE name=%s global=%u,%u,%u local=%u,%u,%u shared=%zu\n", name,
           gx, gy, gz, lx, ly, lz, sm);
    void* data = nullptr;
    size_t size = 0;
    if (extra) {
      for (int i = 0; extra[i] != HIP_LAUNCH_PARAM_END; i += 2) {
        if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_POINTER) data = extra[i + 1];
        if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_SIZE)
          size = *(size_t*)extra[i + 1];
      }
    }
    if (!data || size != 176 || sm != 0) {
      fprintf(stderr, "Unsupported GEMM launch ABI\n");
      std::abort();
    }
    if (data && size) {
      // Do not persist process addresses. The preparation tool verifies these
      // offsets against code-object metadata before accepting the template.
      std::vector<unsigned char> arguments(size);
      memcpy(arguments.data(), data, size);
      memset(arguments.data() + 32, 0, 32);
      FILE* out = fopen("captured-args.bin", "wb");
      if (!out || fwrite(arguments.data(), 1, size, out) != size) std::abort();
      fclose(out);
    }
    FILE* out = fopen("captured-kernel.txt", "w");
    if (!out) std::abort();
    fprintf(out, "%s\n%u %u %u %u %u %u %zu\n", name, gx, gy, gz, lx, ly, lz,
            size);
    fclose(out);
  }
  return real(f, gx, gy, gz, lx, ly, lz, sm, stream, params, extra, start, stop,
              flags);
}
extern "C" hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
  static auto real =
      (decltype(&hipModuleLoad))dlsym(RTLD_NEXT, "hipModuleLoad");
  if (!real) std::abort();
  printf("MODULE_FILE %s\n", fname);
  return real(module, fname);
}
