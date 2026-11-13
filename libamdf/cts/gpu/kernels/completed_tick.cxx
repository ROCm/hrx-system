// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>
#include <loomcxx/low.h>
#include <loomcxx/target/amdgpu.h>

namespace {

// Samples the constant-frequency reference clock after the resident program's
// prior vector loads and stores complete. Release/acquire operations in the
// caller separately own visibility to the peer device.
LOOM_TEMPLATE_DECL("completed_tick") unsigned completed_tick();
LOOM_TEMPLATE_DECL("completed_tick64") unsigned long long completed_tick64();

constexpr loom::amdgpu::target clock_gfx11{.kind = "gfx11-generic"};
constexpr loom::amdgpu::target clock_gfx12{.kind = "gfx12-generic"};
constexpr loom::amdgpu::target clock_gfx125{.kind = "gfx12-5-generic"};
constexpr loom::amdgpu::target clock_gfx1250_strict{.kind = "gfx1250-strict"};
constexpr loom::amdgpu::target clock_gfx1170{.kind = "gfx1170"};
constexpr loom::amdgpu::target clock_gfx1171{.kind = "gfx1171"};
constexpr loom::amdgpu::target clock_gfx1172{.kind = "gfx1172"};

struct [[loom::representation("amdgpu.gfx11.generic.core")]] Gfx11Core {};
struct [[loom::representation("amdgpu.gfx12.generic.core")]] Gfx12Core {};
struct [[loom::representation("amdgpu.gfx12_5.generic.core")]] Gfx125Core {};
struct [[loom::representation("amdgpu.rdna4.gfx1250_a0.core")]]
Gfx1250StrictCore {};
struct [[loom::representation("amdgpu.rdna4m.core")]] Rdna4mCore {};

// Message 0x83 is RTN_GET_REALTIME. The returned SGPR is asynchronous and
// becomes available through LGKM on gfx11 and RDNA4m processors.
template <class Contract>
LOOM_INLINE static unsigned sample_legacy_tick() {
  return loom::low::assembly<Contract, unsigned>(R"loom(
      () -> (reg<amdgpu.sgpr>) {
        s_waitcnt {vmcnt = 0}
        s_waitcnt_vscnt {vscnt = 0}
        %tick = s_sendmsg_rtn_b32 {message = 131}
        s_waitcnt {lgkmcnt = 0}
        return %tick
      }
  )loom");
}

template <class Contract>
LOOM_INLINE static unsigned long long sample_legacy_tick64() {
  return loom::low::assembly<Contract, unsigned long long>(R"loom(
      () -> (reg<amdgpu.sgpr x2>) {
        s_waitcnt {vmcnt = 0}
        s_waitcnt_vscnt {vscnt = 0}
        %tick = s_sendmsg_rtn_b64 {message = 131}
        s_waitcnt {lgkmcnt = 0}
        return %tick
      }
  )loom");
}

// gfx12 processors expose the same reference clock through the split load,
// store, and KM counters.
template <class Contract>
LOOM_INLINE static unsigned sample_modern_tick() {
  return loom::low::assembly<Contract, unsigned>(R"loom(
      () -> (reg<amdgpu.sgpr>) {
        s_wait_loadcnt {loadcnt = 0}
        s_wait_storecnt {storecnt = 0}
        %tick = s_sendmsg_rtn_b32 {message = 131}
        s_wait_kmcnt {kmcnt = 0}
        return %tick
      }
  )loom");
}

template <class Contract>
LOOM_INLINE static unsigned long long sample_modern_tick64() {
  return loom::low::assembly<Contract, unsigned long long>(R"loom(
      () -> (reg<amdgpu.sgpr x2>) {
        s_wait_loadcnt {loadcnt = 0}
        s_wait_storecnt {storecnt = 0}
        %tick = s_sendmsg_rtn_b64 {message = 131}
        s_wait_kmcnt {kmcnt = 0}
        return %tick
      }
  )loom");
}

LOOM_TEMPLATE_DEF(completed_tick)
[[loom::target(clock_gfx11)]] static unsigned tick_gfx11() {
  return sample_legacy_tick<Gfx11Core>();
}

LOOM_TEMPLATE_DEF(completed_tick)
[[loom::target(clock_gfx12)]] static unsigned tick_gfx12() {
  return sample_modern_tick<Gfx12Core>();
}

LOOM_TEMPLATE_DEF(completed_tick)
[[loom::target(clock_gfx125)]] static unsigned tick_gfx125() {
  return sample_modern_tick<Gfx125Core>();
}

LOOM_TEMPLATE_DEF(completed_tick)
[[loom::target(clock_gfx1250_strict)]] static unsigned tick_gfx1250_strict() {
  return sample_modern_tick<Gfx1250StrictCore>();
}

LOOM_TEMPLATE_DEF(completed_tick)
[[loom::target(clock_gfx1170)]] static unsigned tick_gfx1170() {
  return sample_legacy_tick<Rdna4mCore>();
}

LOOM_TEMPLATE_DEF(completed_tick)
[[loom::target(clock_gfx1171)]] static unsigned tick_gfx1171() {
  return sample_legacy_tick<Rdna4mCore>();
}

LOOM_TEMPLATE_DEF(completed_tick)
[[loom::target(clock_gfx1172)]] static unsigned tick_gfx1172() {
  return sample_legacy_tick<Rdna4mCore>();
}

LOOM_TEMPLATE_DEF(completed_tick64)
[[loom::target(clock_gfx11)]] static unsigned long long tick64_gfx11() {
  return sample_legacy_tick64<Gfx11Core>();
}

LOOM_TEMPLATE_DEF(completed_tick64)
[[loom::target(clock_gfx12)]] static unsigned long long tick64_gfx12() {
  return sample_modern_tick64<Gfx12Core>();
}

LOOM_TEMPLATE_DEF(completed_tick64)
[[loom::target(clock_gfx125)]] static unsigned long long tick64_gfx125() {
  return sample_modern_tick64<Gfx125Core>();
}

LOOM_TEMPLATE_DEF(completed_tick64)
[[loom::target(clock_gfx1250_strict)]] static unsigned long long
tick64_gfx1250_strict() {
  return sample_modern_tick64<Gfx1250StrictCore>();
}

LOOM_TEMPLATE_DEF(completed_tick64)
[[loom::target(clock_gfx1170)]] static unsigned long long tick64_gfx1170() {
  return sample_legacy_tick64<Rdna4mCore>();
}

LOOM_TEMPLATE_DEF(completed_tick64)
[[loom::target(clock_gfx1171)]] static unsigned long long tick64_gfx1171() {
  return sample_legacy_tick64<Rdna4mCore>();
}

LOOM_TEMPLATE_DEF(completed_tick64)
[[loom::target(clock_gfx1172)]] static unsigned long long tick64_gfx1172() {
  return sample_legacy_tick64<Rdna4mCore>();
}

}  // namespace
