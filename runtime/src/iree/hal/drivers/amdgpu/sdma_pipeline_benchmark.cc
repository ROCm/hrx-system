// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Double-buffered host-to-device weight transfers overlapping optimized GEMMs.
// Emits raw GPU intervals for the companion run.py/plot.py tools. The GPU timer
// excludes recording/publication; CPU submission timings are reported
// separately. Kernel preparation and usage: sdma_pipeline_benchmark/prepare.py
// --help.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/drivers/amdgpu/util/aql_ring.h"
#include "iree/hal/drivers/amdgpu/util/pm4_barrier.h"
#include "iree/hal/drivers/amdgpu/util/pm4_dispatch.h"
#include "iree/hal/drivers/amdgpu/util/pm4_program.h"
#include "iree/hal/drivers/amdgpu/util/sdma_emitter.h"
#include "iree/hal/drivers/amdgpu/util/sdma_program.h"
#include "iree/hal/drivers/amdgpu/util/sdma_queue.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
IREE_FLAG(string, sdma_data_directory, "",
          "Prepared benchmark data directory; see "
          "sdma_pipeline_benchmark/prepare.py.");
IREE_FLAG(string, sdma_copy_mode, "sdma-ring",
          "Copy implementation: sdma-ring, sdma-ib, or shader. Compute always "
          "uses a PM4 IB.");
IREE_FLAG(bool, sdma_allow_experimental_ib, false,
          "Allow experimental IB execution on a kernel with KFD IB_ENABLE set. "
          "Does not change runtime capabilities.");
IREE_FLAG(int32_t, sdma_stages, 8,
          "Number of double-buffered copy/GEMM stages (1..32).");
IREE_FLAG(int32_t, sdma_iterations, 24,
          "Number of pipeline replays (1..1000).");
IREE_FLAG(int32_t, sdma_engine, 0,
          "SDMA engine ordinal; -1 lets the runtime select one.");

namespace {

using Clock = std::chrono::steady_clock;

void Check(bool condition, const char* message = "benchmark invariant failed") {
  if (!condition) {
    IREE_CHECK_OK(
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "%s", message));
  }
}

std::string DataPath(const char* name) {
  return std::string(FLAG_sdma_data_directory) + "/" + name;
}

struct alignas(64) Word {
  uint64_t value;
  uint64_t pad[7];
};

struct Stage {
  Word ready, started, consumed, copy_start, copy_end, compute_start,
      compute_end;
};

struct BlitArgs {
  void* src;
  void* dst;
  uint64_t count;
};

struct Kernel {
  uint64_t object;
  uint32_t wg;
  iree_hal_amdgpu_pm4_dispatch_launch_state_t launch;
};

struct Context {
  iree_hal_amdgpu_libhsa_t lib{};
  iree_hal_amdgpu_topology_t topology{};
  hsa_agent_t gpu{}, cpu{};
  hsa_amd_memory_pool_t fine{}, device{};
  hsa_executable_t exe{};
  hsa_queue_t* cq[2]{};
  iree_hal_amdgpu_aql_ring_t rings[2]{};
  iree_hal_amdgpu_sdma_queue_t sdma{};
  std::vector<void*> allocations;
  std::vector<iree_hal_amdgpu_pm4_program_t> programs;
  uint32_t cus = 0;

  explicit Context(const char* path) {
    IREE_CHECK_OK(iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        iree_allocator_system(), &lib));
    IREE_CHECK_OK(
        iree_hal_amdgpu_topology_initialize_with_defaults(&lib, &topology));
    Check(topology.gpu_agent_count && topology.cpu_agent_count);
    gpu = topology.gpu_agents[0];
    cpu = topology.cpu_agents[0];
    char name[64]{};
    IREE_CHECK_OK(iree_hsa_agent_get_info(IREE_LIBHSA(&lib), gpu,
                                          HSA_AGENT_INFO_NAME, name));
    Check(!strcmp(name, "gfx1201"),
          "This benchmark launch recipe requires gfx1201");
    IREE_CHECK_OK(iree_hsa_agent_get_info(
        IREE_LIBHSA(&lib), gpu,
        (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &cus));
    IREE_CHECK_OK(
        iree_hal_amdgpu_find_fine_global_memory_pool(&lib, cpu, &fine));
    IREE_CHECK_OK(
        iree_hal_amdgpu_find_coarse_global_memory_pool(&lib, gpu, &device));
    iree_hal_amdgpu_aql_queue_execution_mode_t mode;
    IREE_CHECK_OK(
        iree_hal_amdgpu_query_aql_queue_execution_mode(&lib, gpu, &mode));
    Check(mode != IREE_HAL_AMDGPU_AQL_QUEUE_EXECUTION_MODE_PM4_EMULATED);
    for (int i = 0; i < 2; i++) {
      IREE_CHECK_OK(iree_hsa_queue_create(
          IREE_LIBHSA(&lib), gpu, 64, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr,
          UINT32_MAX, UINT32_MAX, &cq[i]));
      iree_hal_amdgpu_aql_ring_initialize(
          &lib, reinterpret_cast<iree_amd_queue_t*>(cq[i]), mode, &rings[i]);
    }
    iree_hal_amdgpu_sdma_queue_params_t p{};
    p.libhsa = &lib;
    p.agent = gpu;
    p.gfxip_version = {12, 0, 1};
    p.capacity_bytes = 65536;
    p.engine_id = FLAG_sdma_engine < 0 ? UINT32_MAX : FLAG_sdma_engine;
    p.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
    IREE_CHECK_OK(iree_hal_amdgpu_sdma_queue_initialize(&p, &sdma));
    std::ifstream file(path, std::ios::binary);
    Check(file.good(), "Cannot open blit code object; run prepare.py first");
    std::vector<char> code((std::istreambuf_iterator<char>(file)), {});
    hsa_code_object_reader_t reader{};
    hsa_loaded_code_object_t loaded{};
    IREE_CHECK_OK(iree_hsa_code_object_reader_create_from_memory(
        IREE_LIBHSA(&lib), code.data(), code.size(), &reader));
    IREE_CHECK_OK(iree_hsa_executable_create_alt(
        IREE_LIBHSA(&lib), HSA_PROFILE_FULL,
        HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &exe));
    IREE_CHECK_OK(iree_hsa_executable_load_agent_code_object(
        IREE_LIBHSA(&lib), exe, gpu, reader, nullptr, &loaded));
    {
      std::ifstream f(DataPath("optimized.hsaco"), std::ios::binary);
      Check(f.good(),
            "Cannot open optimized code object; run prepare.py first");
      std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
      hsa_code_object_reader_t r{};
      IREE_CHECK_OK(iree_hsa_code_object_reader_create_from_memory(
          IREE_LIBHSA(&lib), bytes.data(), bytes.size(), &r));
      IREE_CHECK_OK(iree_hsa_executable_load_agent_code_object(
          IREE_LIBHSA(&lib), exe, gpu, r, nullptr, &loaded));
      IREE_CHECK_OK(iree_hsa_code_object_reader_destroy(IREE_LIBHSA(&lib), r));
    }
    IREE_CHECK_OK(iree_hsa_executable_freeze(IREE_LIBHSA(&lib), exe, nullptr));
    IREE_CHECK_OK(
        iree_hsa_code_object_reader_destroy(IREE_LIBHSA(&lib), reader));
  }

  ~Context() {
    for (auto q : cq) {
      IREE_CHECK_OK(iree_hsa_queue_destroy(IREE_LIBHSA(&lib), q));
    }
    iree_hal_amdgpu_sdma_queue_deinitialize(&sdma);
    for (auto& p : programs) iree_hal_amdgpu_pm4_program_deinitialize(&p);
    for (auto a : allocations) {
      IREE_CHECK_OK(iree_hsa_amd_memory_pool_free(IREE_LIBHSA(&lib), a));
    }
    IREE_CHECK_OK(iree_hsa_executable_destroy(IREE_LIBHSA(&lib), exe));
    iree_hal_amdgpu_topology_deinitialize(&topology);
    iree_hal_amdgpu_libhsa_deinitialize(&lib);
  }

  void* Allocate(size_t size, bool vram = false) {
    void* p = nullptr;
    IREE_CHECK_OK(iree_hsa_amd_memory_pool_allocate(
        IREE_LIBHSA(&lib), vram ? device : fine, size, 0, &p));
    IREE_CHECK_OK(iree_hsa_amd_agents_allow_access(IREE_LIBHSA(&lib), 1, &gpu,
                                                   nullptr, p));
    allocations.push_back(p);
    return p;
  }

  void Upload(void* dst, const void* src, size_t size) {
    IREE_CHECK_OK(iree_hsa_memory_copy(IREE_LIBHSA(&lib), dst, src, size));
  }

  Kernel LoadKernel(const char* name, uint32_t wg_size = 256) {
    Kernel k{};
    k.wg = wg_size;
    hsa_executable_symbol_t sym{};
    std::string symbol_name = std::string(name) + ".kd";
    const char* buf = symbol_name.c_str();
    IREE_CHECK_OK(iree_hsa_executable_get_symbol_by_name(IREE_LIBHSA(&lib), exe,
                                                         buf, &gpu, &sym));
    IREE_CHECK_OK(iree_hsa_executable_symbol_get_info(
        IREE_LIBHSA(&lib), sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
        &k.object));
    uint32_t priv = 0;
    IREE_CHECK_OK(iree_hsa_executable_symbol_get_info(
        IREE_LIBHSA(&lib), sym,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &priv));
    Check(priv == 0);
    uint16_t wg[] = {(uint16_t)wg_size, 1, 1};
    IREE_CHECK_OK(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize(
        {12, 0, 1},
        reinterpret_cast<const iree_hal_amdgpu_kernel_descriptor_t*>(k.object),
        k.object, wg, IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE,
        &k.launch));
    return k;
  }

  void SubmitCompute(int index, const std::vector<uint32_t>& words) {
    iree_hal_amdgpu_pm4_program_t p{};
    IREE_CHECK_OK(iree_hal_amdgpu_pm4_program_initialize(
        &lib, gpu, device, words.data(), words.size(), &p));
    programs.push_back(p);
    auto& ring = rings[index];
    uint64_t pos = iree_hal_amdgpu_aql_ring_reserve(&ring, 1);
    auto* packet = iree_hal_amdgpu_aql_ring_packet(&ring, pos);
    uint16_t setup = 0;
    auto hdr = iree_hal_amdgpu_aql_emit_pm4_ib_dwords(
        &packet->pm4_ib, p.dwords, p.dword_count,
        iree_hal_amdgpu_aql_packet_control_barrier_system(),
        iree_hsa_signal_null(), &setup);
    iree_hal_amdgpu_aql_ring_commit(packet, hdr, setup);
    iree_hal_amdgpu_aql_ring_doorbell(&ring, pos);
  }

  void SubmitSdma(const std::vector<uint32_t>& words, bool indirect) {
    if (!indirect) {
      // Publish the identical packet stream inline, with one doorbell. The
      // whole stream must fit: it waits on a gate released after submission.
      Check(words.size() * sizeof(uint32_t) < sdma.ring.capacity);
      uint32_t* dst = nullptr;
      IREE_CHECK_OK(iree_hal_amdgpu_sdma_ring_try_reserve(
          &sdma.ring, words.size() * sizeof(uint32_t), &dst));
      memcpy(dst, words.data(), words.size() * sizeof(uint32_t));
      iree_hal_amdgpu_sdma_ring_commit(&sdma.ring);
      return;
    }
    uint32_t count = (words.size() + 7) & ~7u;
    std::vector<uint32_t> padded = words;
    padded.resize(count);
    void* ib = Allocate(count * 4, true);
    Upload(ib, padded.data(), count * 4);
    uint32_t* dst = nullptr;
    IREE_CHECK_OK(iree_hal_amdgpu_sdma_ring_try_reserve(&sdma.ring, 64, &dst));
    memset(dst, 0, 64);
    Check(iree_hal_amdgpu_sdma_emit_indirect(16, dst, dst - sdma.ring.base,
                                             (uint64_t)ib, count, 0, 0));
    iree_hal_amdgpu_sdma_ring_commit(&sdma.ring);
  }
  // Check execution before submitting compute that would wait indefinitely if
  // the kernel consumed an INDIRECT packet without executing its body.
  void VerifyIndirectExecution() {
    auto* body_done = static_cast<Word*>(Allocate(sizeof(Word)));
    auto* ring_done = static_cast<Word*>(Allocate(sizeof(Word)));
    memset(body_done, 0, sizeof(Word));
    memset(ring_done, 0, sizeof(Word));
    uint32_t body[8] = {};
    Check(iree_hal_amdgpu_sdma_emit_fence32(
        8, body, reinterpret_cast<uint64_t>(body_done), 1));
    void* ib = Allocate(sizeof(body), true);
    Upload(ib, body, sizeof(body));
    uint32_t* words = nullptr;
    IREE_CHECK_OK(
        iree_hal_amdgpu_sdma_ring_try_reserve(&sdma.ring, 64, &words));
    memset(words, 0, 64);
    uint32_t count = iree_hal_amdgpu_sdma_emit_indirect(
        16, words, words - sdma.ring.base, reinterpret_cast<uint64_t>(ib), 8, 0,
        0);
    Check(count);
    Check(iree_hal_amdgpu_sdma_emit_fence32(
        16 - count, words + count, reinterpret_cast<uint64_t>(ring_done), 1));
    iree_hal_amdgpu_sdma_ring_commit(&sdma.ring);
    Wait(ring_done);
    Check(__atomic_load_n(&body_done->value, __ATOMIC_ACQUIRE) == 1,
          "SDMA queue did not execute the IB; check KFD IB_ENABLE");
  }

  void Wait(Word* w) {
    auto deadline = Clock::now() + std::chrono::seconds(30);
    while (__atomic_load_n(&w->value, __ATOMIC_ACQUIRE) != 1) {
      Check(Clock::now() < deadline);
    }
  }

  double ToMicroseconds(uint64_t t) {
    uint64_t sys = 0, freq = 0;
    IREE_CHECK_OK(iree_hsa_amd_profiling_convert_tick_to_system_domain(
        IREE_LIBHSA(&lib), gpu, t, &sys));
    IREE_CHECK_OK(iree_hsa_system_get_info(
        IREE_LIBHSA(&lib), HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq));
    return double(sys) * 1e6 / double(freq);
  }
};

struct Pm4ProgramBuilder {
  std::vector<uint32_t> d;
  template <class F>
  void Emit(F f) {
    uint32_t buf[256], n = 0;
    IREE_CHECK_OK(f(buf, &n));
    d.insert(d.end(), buf, buf + n);
  }

  void Barrier() {
    Emit([](uint32_t* b, uint32_t* n) {
      auto caps =
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM_GFX10;
      Check(iree_hal_amdgpu_pm4_barrier_emit(
          caps, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
          IREE_HSA_FENCE_SCOPE_SYSTEM, IREE_HSA_FENCE_SCOPE_SYSTEM, 256, b, n));
      return iree_ok_status();
    });
  }

  void Wait(Word* w) {
    iree_hal_amdgpu_pm4_ib_slot_t slot;
    uint32_t* buf = slot.dwords;
    iree_hal_amdgpu_pm4_ib_builder_t b;
    iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &b);
    Check(iree_hal_amdgpu_pm4_ib_builder_emit_wait_memory64(
        &b, w, IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_FUNC_EQUAL, 1, UINT64_MAX));
    d.insert(d.end(), buf, buf + b.dword_count);
  }

  void Signal(Word* w) {
    uint64_t a = (uint64_t)w;
    d.insert(d.end(),
             {iree_hal_amdgpu_pm4_make_header(
                  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA, 5),
              IREE_HAL_AMDGPU_PM4_WRITE_DATA_DST_SEL_TC_L2 |
                  IREE_HAL_AMDGPU_PM4_WRITE_DATA_WR_CONFIRM_WAIT_CONFIRMATION,
              (uint32_t)a, (uint32_t)(a >> 32), 1});
  }

  void Timestamp(Word* w) {
    iree_hal_amdgpu_pm4_ib_slot_t slot;
    uint32_t* buf = slot.dwords;
    iree_hal_amdgpu_pm4_ib_builder_t b;
    iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &b);
    Check(iree_hal_amdgpu_pm4_ib_builder_emit_copy_timestamp_to_memory(
        &b, IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU, w));
    d.insert(d.end(), buf, buf + b.dword_count);
  }

  void Dispatch(const Kernel& k, void* args, uint32_t gx, uint32_t gy) {
    Emit([&](uint32_t* b, uint32_t* n) {
      return iree_hal_amdgpu_pm4_dispatch_emit_setup(&k.launch, 256, b, n);
    });
    Emit([&](uint32_t* b, uint32_t* n) {
      return iree_hal_amdgpu_pm4_dispatch_emit_user_data(
          &k.launch, (uint64_t)args, nullptr, 256, b, n);
    });
    d.insert(d.end(),
             {iree_hal_amdgpu_pm4_make_compute_header(
                  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_DISPATCH_DIRECT, 5),
              gx * k.wg, gy, 1, k.launch.dispatch_initiator});
  }
};
}  // namespace

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc != 1 || !FLAG_sdma_data_directory[0]) {
    fprintf(stderr, "Specify --sdma_data_directory; run --help for options.\n");
    return 2;
  }
  const char* mode = FLAG_sdma_copy_mode;
  bool indirect = !strcmp(mode, "sdma-ib");
  bool sdma = indirect || !strcmp(mode, "sdma-ring");
  if ((!sdma && strcmp(mode, "shader")) || FLAG_sdma_stages < 1 ||
      FLAG_sdma_stages > 32 || FLAG_sdma_iterations < 1 ||
      FLAG_sdma_iterations > 1000 || FLAG_sdma_engine < -1) {
    fprintf(stderr, "Invalid mode, stage count, iteration count, or engine.\n");
    return 2;
  }
  if (indirect && !FLAG_sdma_allow_experimental_ib) {
    fprintf(stderr,
            "SDMA IBs are disabled: use --sdma_allow_experimental_ib=true "
            "only with an IB-enabled kernel.\n");
    return 2;
  }
  constexpr uint32_t m = 4096, n = 4096, k = 4096;
  uint32_t stages = FLAG_sdma_stages, iterations = FLAG_sdma_iterations;
  Context ctx(DataPath("blit.hsaco").c_str());
  if (indirect) ctx.VerifyIndirectExecution();
  auto blit = ctx.LoadKernel("blit");
  std::ifstream kernel_name(DataPath("captured-kernel.txt"));
  std::string name;
  std::getline(kernel_name, name);
  Check(!name.empty());
  auto mm = ctx.LoadKernel(name.c_str(), 128);
  std::ifstream args(DataPath("optimized-args-template.bin"), std::ios::binary);
  std::vector<char> optimized_args((std::istreambuf_iterator<char>(args)), {});
  Check(optimized_args.size() == 176);

  size_t ab = size_t(m) * k * 2, wb = size_t(n) * k * 2, cb = size_t(m) * n * 4;
  auto* ah = (_Float16*)ctx.Allocate(ab);
  auto* wh = (_Float16*)ctx.Allocate(wb * stages);
  for (size_t i = 0; i < ab / 2; i++) ah[i] = (_Float16)(int(i % 17) - 8) / 16;
  for (size_t i = 0; i < wb / 2 * stages; i++)
    wh[i] = (_Float16)(int((i * 7 + i / (wb / 2)) % 19) - 9) / 16;
  void* a = ctx.Allocate(ab, true);
  ctx.Upload(a, ah, ab);
  void* w[2] = {ctx.Allocate(wb, true), ctx.Allocate(wb, true)};
  // Separate outputs let validation detect early buffer reuse and wrong stages.
  std::vector<void*> outputs;
  for (uint32_t i = 0; i < stages; i++)
    outputs.push_back(ctx.Allocate(cb, true));
  printf(
      "CONFIG mode=%s m=%u n=%u k=%u stages=%u cus=%u "
      "groups=%u weight_bytes=%zu sdma_engine=%u\n",
      mode, m, n, k, stages, ctx.cus, (m / 128) * (n / 128), wb,
      ctx.sdma.engine_id);
  for (uint32_t iter = 0; iter < iterations; iter++) {
    auto* s = (Stage*)ctx.Allocate(sizeof(Stage) * stages);
    memset(s, 0, sizeof(Stage) * stages);
    auto* gate = (Word*)ctx.Allocate(sizeof(Word));
    memset(gate, 0, sizeof(Word));
    Pm4ProgramBuilder compute, copy;
    compute.Wait(gate);
    copy.Wait(gate);
    std::vector<uint32_t> dma;
    auto poll = [&](Word* word) {
      uint32_t b[6];
      Check(iree_hal_amdgpu_sdma_emit_poll32(6, b, (uint64_t)word, 1));
      dma.insert(dma.end(), b, b + 6);
    };
    poll(gate);
    for (uint32_t i = 0; i < stages; i++) {
      // The preparation tool verifies this fixed-shape library-kernel ABI.
      auto* launch_args = static_cast<char*>(ctx.Allocate(256));
      memset(launch_args, 0, 256);
      memcpy(launch_args, optimized_args.data(), 176);
      memcpy(launch_args + 32, &outputs[i], 8);
      memcpy(launch_args + 40, &outputs[i], 8);
      memcpy(launch_args + 48, &w[i % 2], 8);
      memcpy(launch_args + 56, &a, 8);
      compute.Wait(&s[i].ready);
      compute.Barrier();
      compute.Timestamp(&s[i].compute_start);
      compute.Signal(&s[i].started);
      compute.Dispatch(mm, launch_args, 1024, 1);
      compute.Barrier();
      compute.Timestamp(&s[i].compute_end);
      compute.Signal(&s[i].consumed);
      if (i) {
        copy.Wait(&s[i - 1].started);
        poll(&s[i - 1].started);
      }
      if (i > 1) {
        copy.Wait(&s[i - 2].consumed);
        poll(&s[i - 2].consumed);
      }
      void* src = wh + (wb / 2) * i;
      if (sdma) {
        iree_hal_amdgpu_sdma_copy_t c{(uint64_t)src, (uint64_t)w[i % 2], wb};
        iree_hal_amdgpu_sdma_program_params_t p{};
        p.capabilities = ctx.sdma.capabilities;
        p.copy_count = 1;
        p.copies = &c;
        p.completion_address = (uint64_t)&s[i].ready;
        p.completion_value = 1;
        p.start_timestamp_address = (uint64_t)&s[i].copy_start;
        p.end_timestamp_address = (uint64_t)&s[i].copy_end;
        uint32_t len = 0;
        IREE_CHECK_OK(iree_hal_amdgpu_sdma_program_measure(&p, &len));
        size_t pos = dma.size();
        dma.resize(pos + len);
        IREE_CHECK_OK(
            iree_hal_amdgpu_sdma_program_emit(&p, len, dma.data() + pos, &len));
      } else {
        auto* ba = (BlitArgs*)ctx.Allocate(128);
        memset(ba, 0, 128);
        *ba = {src, w[i % 2], wb / 8};
        copy.Barrier();
        copy.Timestamp(&s[i].copy_start);
        copy.Dispatch(blit, ba, 1024, 1);
        copy.Barrier();
        copy.Timestamp(&s[i].copy_end);
        copy.Signal(&s[i].ready);
      }
    }
    auto submit_start = Clock::now();
    ctx.SubmitCompute(0, compute.d);
    auto copy_submit_start = Clock::now();
    if (sdma)
      ctx.SubmitSdma(dma, indirect);
    else
      ctx.SubmitCompute(1, copy.d);
    auto start = Clock::now();
    double submit_us =
        std::chrono::duration<double, std::micro>(start - submit_start).count();
    double copy_submit_us =
        std::chrono::duration<double, std::micro>(start - copy_submit_start)
            .count();
    __atomic_store_n(&gate->value, 1, __ATOMIC_RELEASE);
    ctx.Wait(&s[stages - 1].consumed);
    double host_us =
        std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    double first = ctx.ToMicroseconds(s[0].copy_start.value),
           last = ctx.ToMicroseconds(s[stages - 1].compute_end.value);
    printf(
        "RESULT iter=%u mode=%s gpu_us=%.3f host_us=%.3f "
        "submit_us=%.3f copy_submit_us=%.3f\n",
        iter, mode, last - first, host_us, submit_us, copy_submit_us);
    for (uint32_t i = 0; i < stages; i++)
      printf(
          "STAGE iter=%u stage=%u copy_start_us=%.3f copy_end_us=%.3f "
          "compute_start_us=%.3f compute_end_us=%.3f\n",
          iter, i, ctx.ToMicroseconds(s[i].copy_start.value) - first,
          ctx.ToMicroseconds(s[i].copy_end.value) - first,
          ctx.ToMicroseconds(s[i].compute_start.value) - first,
          ctx.ToMicroseconds(s[i].compute_end.value) - first);
    if (iter == 0) {
      std::vector<float> result(size_t(m) * n);
      for (uint32_t stage = 0; stage < stages; stage++) {
        ctx.Upload(result.data(), outputs[stage], cb);
        double table[17][19];
        for (uint32_t row = 0; row < 17; row++)
          for (uint32_t col = 0; col < 19; col++) {
            double expected = 0;
            for (uint32_t q = 0; q < k; q++)
              expected += (float)ah[size_t(row) * k + q] *
                          (float)wh[(wb / 2) * stage + size_t(col) * k + q];
            table[row][col] = expected;
          }
        double max_error = 0;
        for (uint32_t row = 0; row < m; row++)
          for (uint32_t col = 0; col < n; col++) {
            double error =
                fabs(result[size_t(row) * n + col] - table[row % 17][col % 19]);
            Check(std::isfinite(error));
            max_error = std::max(max_error, error);
          }
        printf("VALIDATE stage=%u max_abs_error=%g\n", stage, max_error);
        Check(max_error < 0.02);
      }
    }
  }
}
