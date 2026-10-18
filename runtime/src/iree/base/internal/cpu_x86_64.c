// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTE: must be first before any system includes.
#define _GNU_SOURCE

#include "iree/base/internal/cpu_x86_64.h"

#define IREE_COPY_BITS(dst_val, dst_mask, src_val, src_mask) \
  ((dst_val) |= (iree_all_bits_set((src_val), (src_mask)) ? (dst_mask) : 0))

uint64_t iree_cpu_x86_64_decode_features(
    const iree_cpu_x86_64_capabilities_t* capabilities) {
  // Bits are given by bit position not by hex value because this is how they
  // are described in the Intel Architectures Software Developer's Manual,
  // Table 3-8, "Information Returned by CPUID Instruction".

  uint64_t features = 0;
  IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_SSE3, capabilities->leaf1.ecx,
                 1 << 0);
  IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_SSSE3, capabilities->leaf1.ecx,
                 1 << 9);
  IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_SSE41, capabilities->leaf1.ecx,
                 1 << 19);
  IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_SSE42, capabilities->leaf1.ecx,
                 1 << 20);
  IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_SSE4A,
                 capabilities->extended_leaf1.ecx, 1 << 6);
  IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_BMI, capabilities->leaf7_0.ebx,
                 1 << 3);
  IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_BMI2,
                 capabilities->leaf7_0.ebx, 1 << 8);
  IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_POPCNT,
                 capabilities->leaf1.ecx, 1 << 23);
  IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_LZCNT,
                 capabilities->extended_leaf1.ecx, 1 << 5);

  const uint32_t xsave_bits = (1u << 26) | (1u << 27);
  const bool has_xsave = iree_all_bits_set(capabilities->leaf1.ecx, xsave_bits);
  const bool has_avx = has_xsave &&
                       iree_all_bits_set(capabilities->leaf1.ecx, 1u << 28) &&
                       iree_all_bits_set(capabilities->enabled_xstate, 0x6);

  // Features that depend on AVX and OS support for YMM state.
  if (has_avx) {
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVX, capabilities->leaf1.ecx,
                   1 << 28);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_FMA, capabilities->leaf1.ecx,
                   1 << 12);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_FMA4,
                   capabilities->extended_leaf1.ecx, 1 << 16);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_XOP,
                   capabilities->extended_leaf1.ecx, 1 << 11);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_F16C,
                   capabilities->leaf1.ecx, 1 << 29);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVX2,
                   capabilities->leaf7_0.ebx, 1 << 5);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVXVNNI,
                   capabilities->leaf7_1.eax, 1 << 4);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVXVNNIINT8,
                   capabilities->leaf7_1.edx, 1 << 4);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVXVNNIINT16,
                   capabilities->leaf7_1.edx, 1 << 10);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVXNECONVERT,
                   capabilities->leaf7_1.edx, 1 << 5);
  }

  // Features that depend on ZMM registers being enabled by the OS.
  if (has_avx && iree_all_bits_set(capabilities->leaf7_0.ebx, 1u << 16) &&
      iree_all_bits_set(capabilities->enabled_xstate, 0xE6)) {
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVX512F,
                   capabilities->leaf7_0.ebx, 1 << 16);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVX512CD,
                   capabilities->leaf7_0.ebx, 1 << 28);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVX512VL,
                   capabilities->leaf7_0.ebx, 1u << 31);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVX512DQ,
                   capabilities->leaf7_0.ebx, 1 << 17);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVX512BW,
                   capabilities->leaf7_0.ebx, 1 << 30);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVX512IFMA,
                   capabilities->leaf7_0.ebx, 1 << 21);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVX512VBMI,
                   capabilities->leaf7_0.ecx, 1 << 1);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVX512VPOPCNTDQ,
                   capabilities->leaf7_0.ecx, 1 << 14);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVX512VNNI,
                   capabilities->leaf7_0.ecx, 1 << 11);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVX512VBMI2,
                   capabilities->leaf7_0.ecx, 1 << 6);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVX512BITALG,
                   capabilities->leaf7_0.ecx, 1 << 12);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVX512BF16,
                   capabilities->leaf7_1.eax, 1 << 5);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AVX512FP16,
                   capabilities->leaf7_0.edx, 1 << 23);
  }

  // AMX also requires permission for dynamically enabled tile state.
  if (has_xsave && iree_all_bits_set(capabilities->leaf7_0.edx, 1u << 24) &&
      iree_all_bits_set(
          capabilities->enabled_xstate & capabilities->permitted_xstate,
          0x60000)) {
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AMXTILE,
                   capabilities->leaf7_0.edx, 1 << 24);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AMXINT8,
                   capabilities->leaf7_0.edx, 1 << 25);
    IREE_COPY_BITS(features, IREE_CPU_DATA0_X86_64_AMXBF16,
                   capabilities->leaf7_0.edx, 1 << 22);
  }

  return features;
}

#if defined(IREE_ARCH_X86_64)

#if defined(__GNUC__)
#include <cpuid.h>
#elif defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(IREE_PLATFORM_APPLE)
#include <sys/sysctl.h>
#endif

#if defined(__GNUC__)
static iree_cpu_x86_64_registers_t iree_cpu_x86_64_cpuid(uint32_t eax,
                                                         uint32_t ecx) {
  iree_cpu_x86_64_registers_t registers;
  __cpuid_count(eax, ecx, registers.eax, registers.ebx, registers.ecx,
                registers.edx);
  return registers;
}
#elif defined(_MSC_VER)
// The noinline is a tentative work-around for what might be a MSVC miscompile.
// The symptom is that MSVC builds incorrectly report some CPU features as
// supported. This only happens for CPU feature bits in the EDX output register.
__declspec(noinline) static iree_cpu_x86_64_registers_t iree_cpu_x86_64_cpuid(
    uint32_t eax, uint32_t ecx) {
  int eax_int;
  int ecx_int;
  memcpy(&eax_int, &eax, sizeof eax);
  memcpy(&ecx_int, &ecx, sizeof ecx);
  int register_values[4];
  __cpuidex(register_values, eax_int, ecx_int);
  iree_cpu_x86_64_registers_t registers;
  memcpy(&registers, register_values, sizeof registers);
  return registers;
}
#else
#error What is the __cpuidex built-in for this compiler?
#endif

// The caller establishes XSAVE and OSXSAVE before executing XGETBV.
static uint64_t iree_cpu_x86_64_read_xcr0(void) {
#if defined(__GNUC__) || defined(__clang__)
  uint32_t eax;
  uint32_t edx;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return ((uint64_t)edx << 32) | eax;
#else
  return _xgetbv(0);
#endif
}

uint64_t iree_cpu_x86_64_query_features(void) {
  iree_cpu_x86_64_capabilities_t capabilities = {0};
  const uint32_t max_leaf = iree_cpu_x86_64_cpuid(0, 0).eax;
  const uint32_t max_extended_leaf = iree_cpu_x86_64_cpuid(0x80000000u, 0).eax;
  if (max_leaf >= 1) {
    capabilities.leaf1 = iree_cpu_x86_64_cpuid(1, 0);
  }
  if (max_leaf >= 7) {
    capabilities.leaf7_0 = iree_cpu_x86_64_cpuid(7, 0);
    if (capabilities.leaf7_0.eax >= 1) {
      capabilities.leaf7_1 = iree_cpu_x86_64_cpuid(7, 1);
    }
  }
  if (max_extended_leaf >= 0x80000001u) {
    capabilities.extended_leaf1 = iree_cpu_x86_64_cpuid(0x80000001u, 0);
  }

  if (iree_all_bits_set(capabilities.leaf1.ecx, (1u << 26) | (1u << 27))) {
    capabilities.enabled_xstate = iree_cpu_x86_64_read_xcr0();
#if defined(IREE_PLATFORM_APPLE)
    // Darwin advertises AVX-512 through sysctl and promotes a thread's save
    // area on first use. XCR0 can omit the ZMM bits before that promotion.
    int avx512_available = 0;
    size_t value_size = sizeof(avx512_available);
    if (sysctlbyname("hw.optional.avx512f", &avx512_available, &value_size,
                     NULL, 0) == 0 &&
        avx512_available) {
      capabilities.enabled_xstate |= 0xE0;
    }
#endif
// AMX needs process-wide permission. Windows optional state can be enabled for
// individual threads; querying the caller cannot establish permission for
// runtime workers. Platforms without a process query leave permissions unset.
#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)
    if (iree_all_bits_set(capabilities.enabled_xstate, 0x60000)) {
      // Linux enables tile state in XCR0 before granting process permission.
      // Keep the UAPI value local so older kernel headers can build this query.
      const int arch_get_xcomp_perm = 0x1022;
      uint64_t permitted_xstate = 0;
      if (syscall(SYS_arch_prctl, arch_get_xcomp_perm, &permitted_xstate) ==
          0) {
        capabilities.permitted_xstate = permitted_xstate;
      }
    }
#endif
  }
  return iree_cpu_x86_64_decode_features(&capabilities);
}

#endif  // IREE_ARCH_X86_64
