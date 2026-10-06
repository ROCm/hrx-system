// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/kernels/file_latency.h"

#include <drm/amdgpu_drm.h>
#include <fcntl.h>
#include <immintrin.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest-spi.h"
#include "libamdf/cts/gpu/kernels/file_demand.h"
#if defined(AMDF_FILE_IO_CXX_KERNELS)
#include "libamdf/cts/gpu/kernels/file_demand_cxx_kernels.h"
#include "libamdf/cts/gpu/kernels/file_latency_cxx_kernels.h"
#else
#include "libamdf/cts/gpu/kernels/file_demand_kernels.h"
#include "libamdf/cts/gpu/kernels/file_latency_kernels.h"
#endif
#include "libamdf/cts/gpu/linux/io_uring/file_io_fixture.h"

namespace {

namespace protocol = kernels::file_latency;
#if defined(AMDF_FILE_IO_CXX_KERNELS)
namespace latency_products = kernels::file_latency_cxx;
#else
namespace latency_products = kernels::file_latency;
#endif
constexpr uint32_t kGuard = 0x9d372be5u;
constexpr uint32_t kGuardWords = 16;
enum class InputFault { kNone, kShortFile, kAbsentFile };

uint32_t Hash(uint32_t value) {
  value ^= value << 13;
  value ^= value >> 17;
  return value ^ (value << 5);
}

uint32_t InputWord(uint32_t block, uint32_t word) {
  return Hash(0x5a7fc321u + block * 0x9e3779b9u + word * 0x85ebca6bu);
}

uint64_t WallNanoseconds() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void ThreadNanoseconds(uint64_t* result) {
  timespec value = {};
  ASSERT_EQ(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value), 0);
  *result = static_cast<uint64_t>(value.tv_sec) * 1000000000 +
            static_cast<uint64_t>(value.tv_nsec);
}

void EnvironmentCount(const char* name, uint32_t maximum, uint32_t* value) {
  const char* text = std::getenv(name);
  if (!text) {
    return;
  }
  char* end = nullptr;
  errno = 0;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  ASSERT_EQ(errno, 0) << name;
  ASSERT_NE(end, text) << name;
  ASSERT_EQ(*end, 0) << name;
  ASSERT_GE(parsed, 1u) << name;
  ASSERT_LE(parsed, maximum) << name;
  *value = static_cast<uint32_t>(parsed);
}

// Each profile describes bytes and dependencies, not an application speedup.
struct Profile {
  // Bytes in one aligned read or write.
  uint32_t byte_length;
  // Independent completion-driven streams sharing one native ring.
  uint32_t depth;
  // One for lookups or three for read/write/reload.
  uint32_t phases;
  // Device-side arrival gap after consumption, excluding the final round.
  uint32_t gap_microseconds;
  // Fixed optimized-run consumers per stream; correctness uses eight.
  uint32_t rounds;
};

void CheckTickInterval(uint32_t begin_low, uint32_t begin_high,
                       uint32_t end_low, uint32_t end_high) {
  const uint64_t begin = (uint64_t{begin_high} << 32) | begin_low;
  const uint64_t end = (uint64_t{end_high} << 32) | end_low;
  ASSERT_GE(end, begin) << "shader reference clock moved backwards";
  ASSERT_LT(end - begin, UINT32_MAX)
      << "shader interval exceeds compact request timestamp width";
}

TEST(FileClockTest, SameShaderIntervalAllowsLowWordWrap) {
  ASSERT_NO_FATAL_FAILURE(CheckTickInterval(0xfffffff0u, 7, 16, 8));
  ASSERT_NO_FATAL_FAILURE(CheckTickInterval(1, 7, 0xffffffffu, 7));
}

TEST(FileClockTest, BackwardsIntervalRemainsAFailure) {
  EXPECT_FATAL_FAILURE(CheckTickInterval(16, 8, 15, 8),
                       "shader reference clock moved backwards");
}

TEST(FileClockTest, AmbiguousCompactIntervalRemainsAFailure) {
  EXPECT_FATAL_FAILURE(
      CheckTickInterval(1, 7, 0, 8),
      "shader interval exceeds compact request timestamp width");
  EXPECT_FATAL_FAILURE(
      CheckTickInterval(1, 7, 2, 9),
      "shader interval exceeds compact request timestamp width");
}

struct PollerAccounting {
  // Accumulated CPU time, absent when the kernel cannot expose the sample.
  std::optional<uint64_t> microseconds;
  // SQPOLL thread identity, or -1 when fdinfo cannot observe its owner.
  int thread = -1;
  // Last reported CPU, not an affinity guarantee; -1 means unavailable.
  int cpu = -1;
};

void ReadPollerAccounting(std::istream& stream, PollerAccounting* accounting) {
  std::string line;
  while (std::getline(stream, line)) {
    if (line.starts_with("SqTotalTime:")) {
      std::istringstream value(line.substr(12));
      uint64_t microseconds = 0;
      ASSERT_TRUE(value >> microseconds);
      accounting->microseconds = microseconds;
    } else if (line.starts_with("SqThread:")) {
      std::istringstream value(line.substr(9));
      ASSERT_TRUE(value >> accounting->thread);
    } else if (line.starts_with("SqThreadCpu:")) {
      std::istringstream value(line.substr(12));
      ASSERT_TRUE(value >> accounting->cpu);
    }
  }
  ASSERT_FALSE(stream.bad());
  // Older kernels omit CPU time. Some fdinfo implementations also return
  // sentinel ownership and zero counters when their nonblocking lock fails.
  if (accounting->thread <= 0) {
    accounting->microseconds.reset();
  }
}

std::string CounterJson(std::optional<uint64_t> value) {
  return value ? std::to_string(*value) : "null";
}

TEST(PollerAccountingTest, KernelWithoutCpuTimePreservesThreadIdentity) {
  std::istringstream stream("SqThread:\t37\nSqThreadCpu:\t5\nUserFiles:\t1\n");
  PollerAccounting accounting;
  ASSERT_NO_FATAL_FAILURE(ReadPollerAccounting(stream, &accounting));
  EXPECT_EQ(accounting.thread, 37);
  EXPECT_EQ(accounting.cpu, 5);
  EXPECT_FALSE(accounting.microseconds.has_value());
  EXPECT_EQ(CounterJson(accounting.microseconds), "null");
}

TEST(PollerAccountingTest, KernelCpuTimeIncludesAValidZero) {
  for (uint64_t microseconds : {0u, 123456u}) {
    std::istringstream stream("SqThread:\t37\nSqThreadCpu:\t5\nSqTotalTime:\t" +
                              std::to_string(microseconds) + "\n");
    PollerAccounting accounting;
    ASSERT_NO_FATAL_FAILURE(ReadPollerAccounting(stream, &accounting));
    ASSERT_TRUE(accounting.microseconds.has_value());
    EXPECT_EQ(*accounting.microseconds, microseconds);
    EXPECT_EQ(CounterJson(accounting.microseconds),
              std::to_string(microseconds));
  }
}

TEST(PollerAccountingTest, UnobservedOwnerDoesNotReportZeroCpuTime) {
  std::istringstream stream(
      "SqThread:\t-1\nSqThreadCpu:\t-1\nSqTotalTime:\t0\n");
  PollerAccounting accounting;
  ASSERT_NO_FATAL_FAILURE(ReadPollerAccounting(stream, &accounting));
  EXPECT_EQ(accounting.thread, -1);
  EXPECT_EQ(accounting.cpu, -1);
  EXPECT_FALSE(accounting.microseconds.has_value());
}

TEST(PollerAccountingTest, MalformedAccountingRemainsAFailure) {
  EXPECT_FATAL_FAILURE(
      {
        std::istringstream stream("SqThread:\t37\nSqTotalTime:\tinvalid\n");
        PollerAccounting accounting;
        ReadPollerAccounting(stream, &accounting);
      },
      "value >> microseconds");
}

TEST(PollerAccountingTest, FailedReadRemainsAFailure) {
  EXPECT_FATAL_FAILURE(
      {
        std::istringstream stream;
        stream.setstate(std::ios::badbit);
        PollerAccounting accounting;
        ReadPollerAccounting(stream, &accounting);
      },
      "stream.bad()");
}

// Process CPU clocks can include other threads. This record deliberately
// accounts for the service thread and SQPOLL separately, without double count.
struct Sample {
  // Host publication through observed final GPU completion.
  uint64_t wall_nanoseconds = 0;
  // Userspace service thread CPU time across that same interval.
  uint64_t host_nanoseconds = 0;
  // SQPOLL CPU time through final completion, absent if not observable.
  std::optional<uint64_t> poller_microseconds;
  // Additional SQPOLL CPU time until sleep, absent if not observable.
  std::optional<uint64_t> poller_tail_microseconds;
  // Idle wake syscalls, coalesced by the published native tail.
  uint64_t wake_calls = 0;
  // Ordinary submission/task-work service calls, with or without waiting.
  uint64_t enter_calls = 0;
  // Userspace service owner's native thread ID.
  int host_thread = 0;
  // CPU executing the service owner before measured publication.
  int host_start_cpu = -1;
  // CPU executing the service owner after measured completion.
  int host_end_cpu = -1;
};

class GpuFileLatencyTest : public GpuFileIoFixture,
                           public ::testing::WithParamInterface<FileMode> {
 protected:
  void TearDown() override {
    ASSERT_NO_FATAL_FAILURE(GpuFileIoFixture::TearDown());
    if (clock_file_ >= 0) {
      ASSERT_EQ(close(clock_file_), 0);
      clock_file_ = -1;
    }
  }

  void OpenClock() {
    amdf_endpoint_info_t endpoint_info = {};
    endpoint_info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    endpoint_info.structure_size = sizeof(endpoint_info);
    ASSERT_EQ(api_->endpoint_query_info(endpoint_, &endpoint_info),
              AMDF_STATUS_OK);
    ASSERT_EQ(endpoint_info.native_identity.type,
              AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE);
    const auto& identity = endpoint_info.native_identity.value.linux_device;
    const std::string path =
        "/dev/dri/renderD" + std::to_string(identity.minor);
    clock_file_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    ASSERT_GE(clock_file_, 0) << std::strerror(errno);
    struct stat info = {};
    ASSERT_EQ(fstat(clock_file_, &info), 0);
    ASSERT_EQ(major(info.st_rdev), identity.major);
    ASSERT_EQ(minor(info.st_rdev), identity.minor);
    drm_amdgpu_info_device device_info = {};
    drm_amdgpu_info query = {};
    query.return_pointer = reinterpret_cast<uintptr_t>(&device_info);
    query.return_size = sizeof(device_info);
    query.query = AMDGPU_INFO_DEV_INFO;
    ASSERT_EQ(ioctl(clock_file_, DRM_IOCTL_AMDGPU_INFO, &query), 0);
    frequency_khz_ = device_info.gpu_counter_freq;
    ASSERT_GT(frequency_khz_, 0u);
    RecordProperty("io_reference_clock_khz", frequency_khz_);
    RecordProperty("io_clock_source", "shader_get_realtime");
  }

  void PollerCpu(std::optional<uint64_t>* microseconds) {
    if (!(ring_->parameters.flags & IORING_SETUP_SQPOLL)) {
      *microseconds = 0;
      poller_thread_ = -1;
      poller_cpu_ = -1;
      return;
    }
    std::ifstream stream("/proc/self/fdinfo/" + std::to_string(ring_->file));
    ASSERT_TRUE(stream.is_open());
    PollerAccounting accounting;
    ASSERT_NO_FATAL_FAILURE(ReadPollerAccounting(stream, &accounting));
    *microseconds = accounting.microseconds;
    poller_thread_ = accounting.thread;
    poller_cpu_ = accounting.cpu;
    if (!microseconds->has_value()) {
      RecordProperty("io_sqpoll_cpu_accounting", "unavailable");
    }
  }

  void WaitIdle() {
    while (
        (ring_->parameters.flags & IORING_SETUP_SQPOLL) &&
        !(GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.flags)) &
          IORING_SQ_NEED_WAKEUP)) {
      std::this_thread::yield();
    }
  }

  void RunEpoch(GpuCommandQueue* queue, Pm4CommandWriter* commands,
                const Pm4ComputeProgram& program, GpuMemory* arguments,
                GpuMemory* completion, Sample* sample) {
    commands->SystemBarrier();
    commands->BindCompute(program, arguments->device_address);
    commands->DispatchWave32(1, 1, 1);
    commands->SystemBarrier();
    commands->WriteData32(completion->device_address, 1);
    commands->PadToEightWords();
    ASSERT_LT(commands->word_count(), queue->words().size());
    WaitIdle();
    std::optional<uint64_t> poller_before;
    uint64_t host_before = 0;
    ASSERT_NO_FATAL_FAILURE(PollerCpu(&poller_before));
    sample->host_thread = static_cast<int>(syscall(__NR_gettid));
    sample->host_start_cpu = sched_getcpu();
    uint32_t wake_tail =
        GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.tail));
    ASSERT_NO_FATAL_FAILURE(ThreadNanoseconds(&host_before));
    const uint64_t wall_before = WallNanoseconds();
    ASSERT_NO_FATAL_FAILURE(
        queue->Publish(api_, gpu_api_, commands->word_count()));
    const uintptr_t completion_address =
        reinterpret_cast<uintptr_t>(completion->host.pointer);
    while (GpuLoadAcquire<uint32_t>(completion_address) != 1) {
      RelayFileIo();
      const uint32_t tail =
          GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.tail));
      if (!(ring_->parameters.flags & IORING_SETUP_SQPOLL)) {
        ASSERT_NO_FATAL_FAILURE(ServiceHostIo(&sample->enter_calls));
      } else if (tail != wake_tail && (GpuLoadAcquire<uint32_t>(RingWord(
                                           ring_->parameters.sq_off.flags)) &
                                       IORING_SQ_NEED_WAKEUP)) {
        const long result = syscall(__NR_io_uring_enter, ring_->file, 0, 0,
                                    IORING_ENTER_SQ_WAKEUP, nullptr, 0);
        if (result < 0 && errno == EINTR) {
          continue;
        }
        ASSERT_EQ(result, 0) << std::strerror(errno);
        // A wake owns this published tail until the poller consumes it. A
        // later tail can need another wake; a still-set flag cannot by itself.
        wake_tail = tail;
        ++sample->wake_calls;
      }
      if (service_microseconds_ == 0) {
        _mm_pause();
      } else {
        // This is a polling policy, not a deadline on valid I/O. Actual
        // scheduling and timer slack can exceed the requested interval.
        std::this_thread::sleep_for(
            std::chrono::microseconds(service_microseconds_));
      }
    }
    sample->wall_nanoseconds = WallNanoseconds() - wall_before;
    uint64_t host_after = 0;
    ASSERT_NO_FATAL_FAILURE(ThreadNanoseconds(&host_after));
    sample->host_nanoseconds = host_after - host_before;
    sample->host_end_cpu = sched_getcpu();
    std::optional<uint64_t> poller_after;
    ASSERT_NO_FATAL_FAILURE(PollerCpu(&poller_after));
    if (poller_before && poller_after) {
      ASSERT_GE(*poller_after, *poller_before);
      sample->poller_microseconds = *poller_after - *poller_before;
    }
    ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    WaitIdle();
    std::optional<uint64_t> poller_idle;
    ASSERT_NO_FATAL_FAILURE(PollerCpu(&poller_idle));
    if (poller_after && poller_idle) {
      ASSERT_GE(*poller_idle, *poller_after);
      sample->poller_tail_microseconds = *poller_idle - *poller_after;
    }
    if (ring_->parameters.flags & IORING_SETUP_SQPOLL) {
      EXPECT_GT(sample->wake_calls, 0u);
      EXPECT_EQ(sample->enter_calls, 0u);
    } else {
      EXPECT_GT(sample->enter_calls, 0u);
      EXPECT_EQ(sample->wake_calls, 0u);
      EXPECT_EQ(sample->poller_microseconds, 0u);
      EXPECT_EQ(sample->poller_tail_microseconds, 0u);
    }
  }

  void CheckEpoch(const Profile& profile, const protocol::Arguments& arguments,
                  GpuMemory* payload, GpuMemory* state, GpuMemory* records,
                  std::vector<uint32_t>* expected_file) {
    const auto* summary =
        static_cast<const protocol::Summary*>(state->host.pointer);
    const uint32_t count =
        profile.depth * arguments.round_count * profile.phases;
    ASSERT_EQ(summary->status, 0);
    ASSERT_GE(summary->submitted, count);
    ASSERT_EQ(summary->completed, summary->submitted);
    const uint32_t position = arguments.initial_position + summary->submitted;
    for (const uint32_t offset :
         {ring_->parameters.sq_off.head, ring_->parameters.sq_off.tail,
          ring_->parameters.cq_off.head, ring_->parameters.cq_off.tail}) {
      ASSERT_EQ(GpuLoadAcquire<uint32_t>(RingWord(offset)), position);
    }
    ASSERT_EQ(
        GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.dropped)),
        0u);
    ASSERT_EQ(
        GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.overflow)),
        0u);
    ASSERT_NO_FATAL_FAILURE(
        CheckTickInterval(summary->begin_tick, summary->begin_tick_high,
                          summary->end_tick, summary->end_tick_high));
    ASSERT_EQ(summary->reserved, 0u);
    const auto* rows = reinterpret_cast<const protocol::Record*>(
        static_cast<const uint32_t*>(records->host.pointer) + kGuardWords);
    const auto* slots = reinterpret_cast<const protocol::Slot*>(summary + 1);
    const auto* words = static_cast<const uint32_t*>(payload->host.pointer);
    const uint32_t word_count = arguments.word_count;
    const uint32_t stride_words = arguments.payload_stride / sizeof(uint32_t);
    std::vector<bool> seen(summary->submitted);
    for (uint32_t slot = 0; slot < profile.depth; ++slot) {
      uint32_t cause = arguments.seed + slot;
      uint32_t last_key = 0;
      uint32_t previous_end = summary->begin_tick;
      for (uint32_t round = 0; round < arguments.round_count; ++round) {
        const uint32_t key = cause & arguments.file_block_mask;
        const uint32_t first = InputWord(key, 0);
        const uint32_t selected = InputWord(key, key & (word_count - 1));
        const uint32_t last = InputWord(key, word_count - 1);
        for (uint32_t phase = 0; phase < profile.phases; ++phase) {
          const auto& row =
              rows[(slot * arguments.round_count + round) * profile.phases +
                   phase];
          ASSERT_EQ(row.slot, slot);
          ASSERT_EQ(row.round, round);
          ASSERT_EQ(row.phase, phase);
          ASSERT_EQ(row.key, key);
          ASSERT_EQ(row.cause, cause);
          ASSERT_EQ(row.result, profile.byte_length);
          ASSERT_EQ(row.first, phase == 1 ? 0 : first);
          ASSERT_EQ(row.selected, phase == 1 ? 0 : selected);
          ASSERT_EQ(row.last, phase == 1 ? 0 : last);
          const uint32_t ticket = row.ticket - arguments.initial_position;
          ASSERT_LT(ticket, summary->submitted);
          ASSERT_FALSE(seen[ticket]);
          seen[ticket] = true;
          ASSERT_LE(row.begin_tick - summary->begin_tick,
                    row.end_tick - summary->begin_tick);
          ASSERT_LE(row.end_tick - summary->begin_tick,
                    summary->end_tick - summary->begin_tick);
          ASSERT_LE(previous_end - summary->begin_tick,
                    row.begin_tick - summary->begin_tick);
          if (phase == 0 && round != 0) {
            ASSERT_GE(row.begin_tick - previous_end, arguments.gap_ticks);
          }
          previous_end = row.end_tick;
        }
        cause = Hash(((first ^ selected) ^ last) + cause);
        last_key = key;
      }
      EXPECT_EQ(slots[slot].cause, cause);
      EXPECT_EQ(slots[slot].round, arguments.round_count);
      EXPECT_EQ(slots[slot].phase, 0u);
      EXPECT_EQ(slots[slot].progress, 0u);
      for (uint32_t word = 0; word < word_count; ++word) {
        const uint32_t expected = InputWord(last_key, word);
        ASSERT_EQ(words[page_byte_length_ / 4 + slot * stride_words + word],
                  expected);
        const uint32_t reload = profile.phases == 3 ? expected : kGuard;
        ASSERT_EQ(words[page_byte_length_ / 4 +
                        (slot + profile.depth) * stride_words + word],
                  reload);
        if (profile.phases == 3) {
          (*expected_file)[(arguments.file_block_mask + 1 + slot * 3) *
                               word_count +
                           word] = expected;
        }
      }
    }
    const auto* state_words = static_cast<const uint32_t*>(state->host.pointer);
    for (size_t word = (sizeof(protocol::Summary) +
                        profile.depth * sizeof(protocol::Slot)) /
                       4;
         word < state->host.byte_length / 4; ++word) {
      ASSERT_EQ(state_words[word], 0u);
    }
    const uint32_t guard_words = page_byte_length_ / 4;
    for (uint32_t window = 0; window <= 2 * profile.depth; ++window) {
      for (uint32_t word = 0; word < guard_words; ++word) {
        ASSERT_EQ(words[window * stride_words + word], kGuard);
      }
    }
    const auto* record_words =
        static_cast<const uint32_t*>(records->host.pointer);
    for (uint32_t word = 0; word < kGuardWords; ++word) {
      ASSERT_EQ(record_words[word], kGuard);
      ASSERT_EQ(record_words[kGuardWords +
                             count * sizeof(protocol::Record) / 4 + word],
                kGuard);
    }
  }

  void PrintSample(const Profile& profile, const protocol::Arguments& arguments,
                   FileIoPath path, uint32_t epoch,
                   uint32_t poller_idle_milliseconds, const Sample& sample,
                   GpuMemory* state, GpuMemory* records) {
    const auto* summary =
        static_cast<const protocol::Summary*>(state->host.pointer);
    const auto* rows = reinterpret_cast<const protocol::Record*>(
        static_cast<const uint32_t*>(records->host.pointer) + kGuardWords);
    const uint32_t count =
        profile.depth * arguments.round_count * profile.phases;
    std::ostringstream output;
    output << "AMDF_IO_SAMPLE {\"path\":\"" << FileIoPathName(path)
           << "\",\"mode\":\""
           << (GetParam() == FileMode::kDirect ? "direct" : "buffered")
           << "\",\"epoch\":" << epoch << ",\"bytes\":" << profile.byte_length
           << ",\"depth\":" << profile.depth << ",\"phases\":" << profile.phases
           << ",\"rounds\":" << arguments.round_count
           << ",\"gap_us\":" << profile.gap_microseconds
           << ",\"clock_khz\":" << frequency_khz_
           << ",\"idle_ms\":" << ring_->parameters.sq_thread_idle
           << ",\"poller_idle_ms\":" << poller_idle_milliseconds
           << ",\"setup_flags\":" << ring_->parameters.flags
           << ",\"service_us\":" << service_microseconds_
           << ",\"seed\":" << arguments.seed
           << ",\"physical_requests\":" << summary->submitted
           << ",\"wall_ns\":" << sample.wall_nanoseconds
           << ",\"host_cpu_ns\":" << sample.host_nanoseconds
           << ",\"host_thread\":" << sample.host_thread
           << ",\"host_start_cpu\":" << sample.host_start_cpu
           << ",\"host_end_cpu\":" << sample.host_end_cpu
           << ",\"sqpoll_thread\":" << poller_thread_
           << ",\"sqpoll_last_cpu\":" << poller_cpu_
           << ",\"host_waits_for_io\":"
           << (path == FileIoPath::kHostWait ? "true" : "false")
           << ",\"sqpoll_cpu_us\":" << CounterJson(sample.poller_microseconds)
           << ",\"sqpoll_tail_cpu_us\":"
           << CounterJson(sample.poller_tail_microseconds)
           << ",\"wake_calls\":" << sample.wake_calls
           << ",\"submit_calls\":" << sample.enter_calls << ",\"device_ticks\":"
           << uint32_t(summary->end_tick - summary->begin_tick)
           << ",\"request_ticks\":[";
    for (uint32_t i = 0; i < count; ++i) {
      if (i) {
        output << ',';
      }
      output << uint32_t(rows[i].end_tick - rows[i].begin_tick);
    }
    output << "]}";
    std::puts(output.str().c_str());
  }

  void CheckFailure(const Profile& profile, const protocol::Arguments& values,
                    InputFault fault, GpuMemory* state, GpuMemory* records,
                    GpuMemory* payload) {
    const auto* summary =
        static_cast<const protocol::Summary*>(state->host.pointer);
    ASSERT_EQ(summary->status,
              fault == InputFault::kShortFile ? -ENODATA : -EBADF);
    ASSERT_NO_FATAL_FAILURE(
        CheckTickInterval(summary->begin_tick, summary->begin_tick_high,
                          summary->end_tick, summary->end_tick_high));
    ASSERT_EQ(summary->reserved, 0u);
    ASSERT_EQ(summary->submitted, summary->completed);
    ASSERT_GE(summary->submitted, profile.depth);
    ASSERT_LE(summary->submitted, 2 * profile.depth);
    if (fault == InputFault::kAbsentFile) {
      ASSERT_EQ(summary->submitted, profile.depth);
    }
    for (const uint32_t offset :
         {ring_->parameters.sq_off.head, ring_->parameters.sq_off.tail,
          ring_->parameters.cq_off.head, ring_->parameters.cq_off.tail}) {
      ASSERT_EQ(GpuLoadAcquire<uint32_t>(RingWord(offset)),
                values.initial_position + summary->submitted);
    }
    const auto* rows = reinterpret_cast<const protocol::Record*>(
        static_cast<const uint32_t*>(records->host.pointer) + kGuardWords);
    for (uint32_t slot = 0; slot < profile.depth; ++slot) {
      for (uint32_t round = 0; round < values.round_count; ++round) {
        const auto& row = rows[slot * values.round_count + round];
        EXPECT_EQ(row.result, 0);
        EXPECT_EQ(row.first, 0u);
        EXPECT_EQ(row.selected, 0u);
        EXPECT_EQ(row.last, 0u);
        EXPECT_EQ(row.end_tick, 0u);
      }
    }
    const auto* words = static_cast<const uint32_t*>(payload->host.pointer);
    const size_t guard_words = page_byte_length_ / 4;
    const size_t stride_words = values.payload_stride / 4;
    for (size_t word = 0; word < payload->host.byte_length / 4; ++word) {
      const size_t window = word / stride_words;
      const size_t local = word % stride_words;
      const bool partial = fault == InputFault::kShortFile &&
                           window < profile.depth && local >= guard_words &&
                           local < guard_words + values.word_count / 2;
      // iomap can transfer a complete EOF-containing block and trim only the
      // reported count to i_size. Those unreported bytes remain inside the
      // submitted destination, not its guards, and are never consumed here.
      const bool unreported = fault == InputFault::kShortFile &&
                              GetParam() == FileMode::kDirect &&
                              window < profile.depth &&
                              local >= guard_words + values.word_count / 2 &&
                              local < guard_words + values.word_count;
      if (unreported) {
        continue;
      }
      ASSERT_EQ(words[word],
                partial ? InputWord(0, local - guard_words) : kGuard);
    }
  }

  void Run(const Profile& profile, InputFault fault = InputFault::kNone) {
    const auto* kernel = latency_products::kKernels.Find(gpu_endpoint_info_);
    ASSERT_NE(kernel, nullptr);
    ASSERT_NO_FATAL_FAILURE(OpenClock());
    uint32_t repetitions = 1;
    uint32_t idle_milliseconds = 1;
    ASSERT_NO_FATAL_FAILURE(
        EnvironmentCount("AMDF_IO_REPETITIONS", 31, &repetitions));
    ASSERT_NO_FATAL_FAILURE(
        EnvironmentCount("AMDF_IO_IDLE_MS", 100, &idle_milliseconds));
    ASSERT_NO_FATAL_FAILURE(
        EnvironmentCount("AMDF_IO_SERVICE_US", 1000, &service_microseconds_));
    const bool measurement = std::getenv("AMDF_IO_REPETITIONS") != nullptr &&
                             fault == InputFault::kNone;
    if (measurement) {
      ASSERT_NE(std::getenv("BENCHMARK_LOCK_LEASE_ID"), nullptr)
          << "measurements require the machine benchmark lease";
    }
    const uint32_t rounds = measurement ? profile.rounds : 8;
    const uint32_t word_count = profile.byte_length / 4;
    const uint32_t blocks = fault == InputFault::kShortFile
                                ? 1
                                : (profile.byte_length >= 1048576 ? 16 : 256);
    std::vector<uint32_t> expected_file((blocks + 16) * word_count, kGuard);
    for (uint32_t block = 0; block < blocks; ++block) {
      for (uint32_t word = 0; word < word_count; ++word) {
        expected_file[block * word_count + word] = InputWord(block, word);
      }
    }
    if (fault == InputFault::kShortFile) {
      expected_file.resize(word_count / 2);
    }
    ASSERT_NO_FATAL_FAILURE(CreateFile(expected_file, GetParam()));
    if (IsSkipped()) {
      return;
    }
    ASSERT_EQ(fdatasync(data_file_), 0) << std::strerror(errno);
    const uint32_t stride = profile.byte_length + page_byte_length_;
    GpuMemory* payload = nullptr;
    GpuMemory* state = nullptr;
    GpuMemory* records = nullptr;
    GpuMemory* arguments = nullptr;
    GpuMemory* completion = nullptr;
    const size_t record_count = profile.depth * rounds * profile.phases;
    const size_t record_bytes = 2 * kGuardWords * sizeof(uint32_t) +
                                record_count * sizeof(protocol::Record);
    ASSERT_NO_FATAL_FAILURE(CreateRegisteredPages(
        2 * profile.depth * stride + page_byte_length_, kGuard, &payload));
    ASSERT_NO_FATAL_FAILURE(
        CreateRing(payload, FileIoPath::kHostRelay, idle_milliseconds));
    Ring* poll_ring = ring_;
    ASSERT_NO_FATAL_FAILURE(CreateRing(payload, FileIoPath::kHostWait));
    Ring* wait_ring = ring_;
    ASSERT_NO_FATAL_FAILURE(CreateRing(payload, FileIoPath::kHostPoll));
    Ring* host_poll_ring = ring_;
    ASSERT_NO_FATAL_FAILURE(CreateMemory(
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &state));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                     record_bytes, &records));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
    Pm4ComputeProgram program = {0,
                                 kernel->program.resource1,
                                 kernel->program.resource2,
                                 kernel->program.resource3,
                                 kernel->group_segment_byte_length,
                                 {1, 1, 1}};
    ASSERT_NO_FATAL_FAILURE(PrepareProgram(kernel->executable,
                                           kernel->entry_byte_offset, &program,
                                           "file_latency"));
    // Each epoch owns a short finite command stream, outside timing. Release
    // after completion prevents live queue accumulation across repetitions.
    constexpr std::array paths = {FileIoPath::kDevice, FileIoPath::kHostRelay,
                                  FileIoPath::kHostWait, FileIoPath::kHostPoll};
    for (uint32_t epoch = 0; epoch <= repetitions; ++epoch) {
      for (uint32_t order = 0; order < paths.size(); ++order) {
        // Rotate the starting transport and reverse direction every cycle.
        // Eight measured epochs balance every path across every position,
        // once in each direction. This is not all twenty-four permutations.
        const uint32_t direction = ((epoch / paths.size()) & 1) ? 3 : 1;
        const FileIoPath path =
            paths[(epoch + direction * order) % paths.size()];
        ring_ = path == FileIoPath::kHostWait   ? wait_ring
                : path == FileIoPath::kHostPoll ? host_poll_ring
                                                : poll_ring;
        device_ring_memory_ =
            path == FileIoPath::kDevice ? ring_->memory : ring_->relay_memory;
        const uint32_t position =
            GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.tail));
        if (path != FileIoPath::kDevice) {
          std::memset(device_ring_memory_->host.pointer, 0,
                      device_ring_memory_->host.byte_length);
          const uintptr_t control =
              reinterpret_cast<uintptr_t>(device_ring_memory_->host.pointer) +
              ring_->control_offset;
          for (const uint32_t offset :
               {ring_->parameters.sq_off.head, ring_->parameters.sq_off.tail,
                ring_->parameters.cq_off.head, ring_->parameters.cq_off.tail}) {
            GpuStoreRelease<uint32_t>(control + offset, position);
          }
        }
        std::fill_n(static_cast<uint32_t*>(payload->host.pointer),
                    payload->host.byte_length / 4, kGuard);
        std::memset(state->host.pointer, 0, state->host.byte_length);
        std::fill_n(static_cast<uint32_t*>(records->host.pointer),
                    record_bytes / 4, kGuard);
        std::memset(static_cast<uint32_t*>(records->host.pointer) + kGuardWords,
                    0, record_count * sizeof(protocol::Record));
        std::memset(completion->host.pointer, 0, completion->host.byte_length);
        const protocol::Arguments values = {
            .submission_entries = device_ring_memory_->device_address,
            .submission_tail = RingAddress(ring_->parameters.sq_off.tail),
            .completion_entries = RingAddress(ring_->parameters.cq_off.cqes),
            .completion_head = RingAddress(ring_->parameters.cq_off.head),
            .completion_tail = RingAddress(ring_->parameters.cq_off.tail),
            .payload = payload->device_address + page_byte_length_,
            .state = state->device_address,
            .records = records->device_address + kGuardWords * 4,
            .host_payload = reinterpret_cast<uintptr_t>(payload->host.pointer) +
                            page_byte_length_,
            .submission_mask = ring_->parameters.sq_entries - 1,
            .completion_mask = ring_->parameters.cq_entries - 1,
            .initial_position = position,
            .depth = profile.depth,
            .round_count = rounds,
            .word_count = word_count,
            .file_block_mask = blocks - 1,
            .seed = Hash(0x81234567u + epoch),
            .payload_stride = stride,
            .phase_count = profile.phases,
            .gap_ticks = uint32_t(uint64_t{frequency_khz_} *
                                  profile.gap_microseconds / 1000),
            .file_index = fault == InputFault::kAbsentFile ? 1u : 0u,
        };
        std::memset(arguments->host.pointer, 0, arguments->host.byte_length);
        std::memcpy(arguments->host.pointer, &values, sizeof(values));
        GpuCommandQueue* queue = nullptr;
        ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
        Pm4CommandWriter commands(queue->words().data(), *pm4_profile_);
        Sample sample;
        ASSERT_NO_FATAL_FAILURE(RunEpoch(queue, &commands, program, arguments,
                                         completion, &sample));
        ASSERT_TRUE(queue->Release(api_));
        if (fault == InputFault::kNone) {
          ASSERT_NO_FATAL_FAILURE(CheckEpoch(profile, values, payload, state,
                                             records, &expected_file));
        } else {
          ASSERT_NO_FATAL_FAILURE(
              CheckFailure(profile, values, fault, state, records, payload));
        }
        ASSERT_EQ(std::memcmp(arguments->host.pointer, &values, sizeof(values)),
                  0);
        // Flush outside timing so a later sample does not inherit buffered
        // writeback from this one. Timed writes still make no durability claim.
        if (profile.phases == 3) {
          ASSERT_EQ(fdatasync(data_file_), 0);
        }
        if (measurement && epoch != 0) {
          PrintSample(profile, values, path, epoch, idle_milliseconds, sample,
                      state, records);
        }
      }
    }
    ASSERT_NO_FATAL_FAILURE(VerifyFile(expected_file, GetParam()));
  }

 protected:
  // Read-only DRM descriptor matched to the active endpoint's device identity.
  int clock_file_ = -1;
  // Nominal reference-counter frequency from AMDGPU_INFO_DEV_INFO, in kHz.
  uint32_t frequency_khz_ = 0;
  // Requested delay between host service passes; zero selects busy polling.
  uint32_t service_microseconds_ = 0;
  // SQPOLL thread ID, or -1 for ordinary submission or an unobserved owner.
  int poller_thread_ = -1;
  // SQPOLL's last reported CPU, not a promise of fixed affinity.
  int poller_cpu_ = -1;
};

TEST_P(GpuFileLatencyTest, DependentLookup4KiB) { Run({4096, 1, 1, 0, 1024}); }
TEST_P(GpuFileLatencyTest, FourIndependentLookups4KiB) {
  Run({4096, 4, 1, 0, 256});
}
TEST_P(GpuFileLatencyTest, BlockRoundTrip64KiB) { Run({65536, 1, 3, 0, 256}); }
TEST_P(GpuFileLatencyTest, FourBlockRoundTrips4MiB) {
  Run({4194304, 4, 3, 0, 8});
}
TEST_P(GpuFileLatencyTest, SparseLookup200us) { Run({4096, 1, 1, 200, 128}); }
TEST_P(GpuFileLatencyTest, SparseLookup2ms) { Run({4096, 1, 1, 2000, 64}); }
TEST_P(GpuFileLatencyTest, FailedInputDrainsAcceptedReads) {
  Run({4096, 4, 1, 0, 8}, InputFault::kAbsentFile);
}
TEST_P(GpuFileLatencyTest, ShortInputThenEofDrainsWithoutConsumption) {
  Run({4096, 4, 1, 0, 8}, InputFault::kShortFile);
}

INSTANTIATE_TEST_SUITE_P(FileModes, GpuFileLatencyTest,
                         ::testing::Values(FileMode::kBuffered,
                                           FileMode::kDirect),
                         [](const auto& info) {
                           return info.param == FileMode::kDirect ? "Direct"
                                                                  : "Buffered";
                         });

namespace demand = kernels::file_demand;
#if defined(AMDF_FILE_IO_CXX_KERNELS)
namespace demand_products = kernels::file_demand_cxx;
#else
namespace demand_products = kernels::file_demand;
#endif

// Offered bursts are independent of the transport's completion rate. Credits
// bound backing, not demand count; duplicate readers retain the same credit.
struct DemandProfile {
  // Stable workload name in retained measurement records.
  const char* name;
  // Bytes in one physical operation; an expert tile is not a whole expert.
  uint32_t byte_length;
  // Logical readers offered in each burst.
  uint32_t burst;
  // Maximum independently owned payload windows.
  uint32_t credits;
  // Distinct keys before the offered manifest repeats a key.
  uint32_t distinct_keys;
  // Microseconds between bursts; zero offers the entire manifest at once.
  uint32_t interval_microseconds;
  // Every seventh reader retains backing for this interval after its probe.
  uint32_t hold_microseconds;
  // One for reads, three for scattered read/write/reload.
  uint32_t phases;
  // Enables the independent GPU arithmetic instance.
  bool background;
};

class GpuFileDemandTest : public GpuFileLatencyTest {
 protected:
  uint32_t Ticks(uint32_t microseconds) const {
    return uint64_t{frequency_khz_} * microseconds / 1000;
  }

  void CheckDemand(const DemandProfile& profile,
                   const demand::Arguments& arguments, InputFault fault,
                   GpuMemory* payload, GpuMemory* state, GpuMemory* records,
                   GpuMemory* keys, const std::vector<demand::Record>& offered,
                   std::vector<uint32_t>* expected_file) {
    const auto* summary =
        static_cast<const demand::Summary*>(state->host.pointer);
    const auto* slots = reinterpret_cast<const demand::Slot*>(summary + 1);
    const auto* rows = reinterpret_cast<const demand::Record*>(
        static_cast<const uint32_t*>(records->host.pointer) + kGuardWords);
    ASSERT_EQ(summary->status, fault == InputFault::kNone         ? 0
                               : fault == InputFault::kAbsentFile ? -EBADF
                                                                  : -ENODATA);
    ASSERT_EQ(summary->completed, summary->submitted);
    ASSERT_GT(summary->submitted, 0u);
    ASSERT_LE(summary->peak_outstanding, arguments.credit_count);
    ASSERT_LE(summary->peak_credits, arguments.credit_count);
    ASSERT_EQ(summary->live_credits, 0u);
    ASSERT_EQ(summary->ready_head, 0u);
    ASSERT_EQ(summary->ready_tail, 0u);
    ASSERT_EQ(summary->issue, 0u);
    const uint32_t position = arguments.initial_position + summary->submitted;
    for (const uint32_t offset :
         {ring_->parameters.sq_off.head, ring_->parameters.sq_off.tail,
          ring_->parameters.cq_off.head, ring_->parameters.cq_off.tail}) {
      ASSERT_EQ(GpuLoadAcquire<uint32_t>(RingWord(offset)), position);
    }
    ASSERT_EQ(
        GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.dropped)),
        0u);
    ASSERT_EQ(
        GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.overflow)),
        0u);
    ASSERT_NO_FATAL_FAILURE(
        CheckTickInterval(summary->begin_tick, summary->begin_tick_high,
                          summary->end_tick, summary->end_tick_high));
    ASSERT_EQ(summary->reserved, (std::array<uint32_t, 2>{0, 0}));
    const auto elapsed = [summary](uint32_t tick) {
      return uint32_t(tick - summary->begin_tick);
    };
    struct Generation {
      // First logical reader that allocated the physical backing.
      const demand::Record* leader = nullptr;
      // Latest consumer release, relative to the experiment origin.
      uint32_t release = 0;
    };
    std::vector<std::vector<Generation>> generations(profile.credits);
    uint32_t duplicates = 0;
    for (uint32_t i = 0; i < arguments.demand_count; ++i) {
      SCOPED_TRACE(i);
      const auto& row = rows[i];
      ASSERT_EQ(row.arrival_ticks, offered[i].arrival_ticks);
      ASSERT_EQ(row.key_seed, offered[i].key_seed);
      ASSERT_EQ(row.hold_ticks, offered[i].hold_ticks);
      if (fault != InputFault::kNone) {
        ASSERT_EQ(row.observed, 0u);
        ASSERT_EQ(row.ready_tick, 0u);
        ASSERT_EQ(row.end_tick, 0u);
        ASSERT_EQ(row.result, 0u);
        continue;
      }
      ASSERT_LT(row.slot, profile.credits);
      ASSERT_GT(row.generation, 0u);
      ASSERT_LE(row.generation, arguments.demand_count);
      ASSERT_EQ(row.key, Hash(row.key_seed) & arguments.file_block_mask);
      ASSERT_EQ(row.observed, 1u);
      const uint32_t first = InputWord(row.key, 0);
      const uint32_t selected =
          InputWord(row.key, row.key & (arguments.word_count - 1));
      const uint32_t last = InputWord(row.key, arguments.word_count - 1);
      ASSERT_EQ(row.first, first);
      ASSERT_EQ(row.selected, selected);
      ASSERT_EQ(row.last, last);
      ASSERT_EQ(row.result, Hash((first ^ selected ^ last) + row.key_seed));
      ASSERT_GE(elapsed(row.admitted_tick), row.arrival_ticks);
      ASSERT_GE(elapsed(row.ready_tick), elapsed(row.admitted_tick));
      ASSERT_GE(elapsed(row.end_tick), elapsed(row.ready_tick));
      ASSERT_GE(uint32_t(row.end_tick - row.ready_tick), row.hold_ticks);
      ASSERT_LE(elapsed(row.end_tick), elapsed(summary->end_tick));
      ASSERT_LE(elapsed(row.read_begin), elapsed(row.read_end));
      ASSERT_LE(elapsed(row.read_end), elapsed(row.ready_tick));
      if (profile.phases == 3) {
        ASSERT_EQ(row.shared, 0u);
        ASSERT_LE(elapsed(row.read_end), elapsed(row.write_begin));
        ASSERT_LE(elapsed(row.write_begin), elapsed(row.write_end));
        ASSERT_LE(elapsed(row.write_end), elapsed(row.reload_begin));
        ASSERT_LE(elapsed(row.reload_begin), elapsed(row.reload_end));
        ASSERT_LE(elapsed(row.reload_end), elapsed(row.ready_tick));
      } else {
        ASSERT_EQ(row.write_begin, 0u);
        ASSERT_EQ(row.write_end, 0u);
        ASSERT_EQ(row.reload_begin, 0u);
        ASSERT_EQ(row.reload_end, 0u);
      }
      ASSERT_LT(uint32_t(row.ticket - arguments.initial_position),
                summary->submitted);
      auto& owners = generations[row.slot];
      if (owners.size() < row.generation) {
        ASSERT_EQ(owners.size() + 1, row.generation);
        owners.emplace_back();
      }
      auto& owner = owners[row.generation - 1];
      if (!row.shared) {
        ASSERT_EQ(owner.leader, nullptr);
        owner.leader = &row;
        ASSERT_LE(elapsed(row.admitted_tick), elapsed(row.read_begin));
      } else {
        ASSERT_EQ(row.shared, 1u);
        ASSERT_NE(owner.leader, nullptr);
        ASSERT_EQ(row.key, owner.leader->key);
        ASSERT_EQ(row.read_begin, owner.leader->read_begin);
        ASSERT_EQ(row.read_end, owner.leader->read_end);
        ASSERT_EQ(row.ticket, owner.leader->ticket);
        ++duplicates;
      }
      owner.release = std::max(owner.release, elapsed(row.end_tick));
    }
    if (fault == InputFault::kNone) {
      ASSERT_EQ(summary->admitted, arguments.demand_count);
      ASSERT_EQ(summary->consumed, arguments.demand_count);
      ASSERT_EQ(summary->deduplicated, duplicates);
      ASSERT_GE(summary->submitted,
                (arguments.demand_count - duplicates) * profile.phases);
      std::vector<bool> free_slots(profile.credits);
      uint32_t free = summary->free_head;
      for (uint32_t i = 0; i < profile.credits; ++i) {
        ASSERT_GE(free, 1u);
        ASSERT_LE(free, profile.credits);
        ASSERT_FALSE(free_slots[free - 1]);
        free_slots[free - 1] = true;
        free = slots[free - 1].next;
      }
      ASSERT_EQ(free, 0u);
      for (uint32_t i = 0; i < arguments.demand_count; ++i) {
        const auto& row = rows[i];
        const uint32_t previous =
            row.generation == 1
                ? 0
                : summary->begin_tick +
                      generations[row.slot][row.generation - 2].release;
        ASSERT_EQ(row.previous_release, previous);
        if (row.generation != 1) {
          ASSERT_GE(elapsed(row.admitted_tick), elapsed(previous));
        }
      }
      const auto* map = static_cast<const uint32_t*>(keys->host.pointer);
      for (uint32_t i = 0; i <= arguments.file_block_mask; ++i) {
        ASSERT_EQ(map[i], 0u);
      }
    } else {
      ASSERT_EQ(summary->consumed, 0u);
      ASSERT_LE(summary->admitted, arguments.demand_count);
    }
    const auto* words = static_cast<const uint32_t*>(payload->host.pointer);
    const uint32_t guard_words = page_byte_length_ / 4;
    const uint32_t stride_words = arguments.payload_stride / 4;
    for (uint32_t slot = 0; slot < profile.credits; ++slot) {
      ASSERT_EQ(slots[slot].readers, 0u);
      ASSERT_EQ(slots[slot].first_reader, 0u);
      ASSERT_EQ(slots[slot].last_reader, 0u);
      if (fault == InputFault::kNone) {
        ASSERT_EQ(slots[slot].generation, generations[slot].size());
        if (!generations[slot].empty()) {
          ASSERT_EQ(elapsed(slots[slot].last_release),
                    generations[slot].back().release);
          const auto& leader = *generations[slot].back().leader;
          ASSERT_EQ(slots[slot].submit_tick, profile.phases == 3
                                                 ? leader.reload_begin
                                                 : leader.read_begin);
          ASSERT_EQ(slots[slot].ready_tick,
                    profile.phases == 3 ? leader.reload_end : leader.read_end);
          ASSERT_EQ(slots[slot].phase_end_tick, slots[slot].ready_tick);
          ASSERT_EQ(slots[slot].ticket, leader.ticket);
        }
      }
      for (uint32_t word = 0; word < arguments.word_count; ++word) {
        const bool populated = slots[slot].generation != 0;
        uint32_t expected =
            populated ? InputWord(slots[slot].key, word) : kGuard;
        if (fault == InputFault::kAbsentFile) {
          expected = kGuard;
        } else if (fault == InputFault::kShortFile &&
                   word >= arguments.word_count / 2) {
          // The direct EOF block can contain unreported bytes inside the
          // submitted destination. None becomes a logical consumer result.
          if (GetParam() == FileMode::kDirect && populated) {
            continue;
          }
          expected = kGuard;
        }
        ASSERT_EQ(words[guard_words + slot * stride_words + word], expected);
        ASSERT_EQ(
            words[guard_words + (slot + profile.credits) * stride_words + word],
            profile.phases == 3 && populated ? expected : kGuard);
        if (profile.phases == 3 && populated) {
          (*expected_file)[(arguments.file_block_mask + 1 + slot * 3) *
                               arguments.word_count +
                           word] = expected;
        }
      }
    }
    for (uint32_t window = 0; window <= 2 * profile.credits; ++window) {
      for (uint32_t word = 0; word < guard_words; ++word) {
        ASSERT_EQ(words[window * stride_words + word], kGuard);
      }
    }
    const auto* record_words =
        static_cast<const uint32_t*>(records->host.pointer);
    for (uint32_t word = 0; word < kGuardWords; ++word) {
      ASSERT_EQ(record_words[word], kGuard);
      ASSERT_EQ(
          record_words[kGuardWords +
                       arguments.demand_count * sizeof(demand::Record) / 4 +
                       word],
          kGuard);
    }
  }

  void PrintDemand(const DemandProfile& profile,
                   const demand::Arguments& arguments, FileIoPath path,
                   uint32_t epoch, const Sample& sample, GpuMemory* state,
                   GpuMemory* records, const demand::Background& background) {
    const auto* summary =
        static_cast<const demand::Summary*>(state->host.pointer);
    const auto* rows = reinterpret_cast<const demand::Record*>(
        static_cast<const uint32_t*>(records->host.pointer) + kGuardWords);
    const auto elapsed = [summary](uint32_t tick) {
      return uint32_t(tick - summary->begin_tick);
    };
    std::ostringstream output;
    output << "AMDF_IO_DEMAND {\"profile\":\"" << profile.name
           << "\",\"path\":\"" << FileIoPathName(path) << "\",\"mode\":\""
           << (GetParam() == FileMode::kDirect ? "direct" : "buffered")
           << "\",\"epoch\":" << epoch << ",\"bytes\":" << profile.byte_length
           << ",\"burst\":" << profile.burst
           << ",\"credits\":" << profile.credits
           << ",\"demands\":" << arguments.demand_count
           << ",\"distinct_keys\":" << profile.distinct_keys
           << ",\"interval_us\":" << profile.interval_microseconds
           << ",\"hold_us\":" << profile.hold_microseconds
           << ",\"phases\":" << profile.phases
           << ",\"clock_khz\":" << frequency_khz_
           << ",\"service_us\":" << service_microseconds_
           << ",\"idle_ms\":" << ring_->parameters.sq_thread_idle
           << ",\"setup_flags\":" << ring_->parameters.flags
           << ",\"physical_requests\":" << summary->submitted
           << ",\"deduplicated\":" << summary->deduplicated
           << ",\"peak_outstanding\":" << summary->peak_outstanding
           << ",\"peak_credits\":" << summary->peak_credits
           << ",\"wall_ns\":" << sample.wall_nanoseconds
           << ",\"host_cpu_ns\":" << sample.host_nanoseconds
           << ",\"host_thread\":" << sample.host_thread
           << ",\"host_start_cpu\":" << sample.host_start_cpu
           << ",\"host_end_cpu\":" << sample.host_end_cpu
           << ",\"sqpoll_thread\":" << poller_thread_
           << ",\"sqpoll_last_cpu\":" << poller_cpu_
           << ",\"host_waits_for_io\":"
           << (path == FileIoPath::kHostWait ? "true" : "false")
           << ",\"sqpoll_cpu_us\":" << CounterJson(sample.poller_microseconds)
           << ",\"sqpoll_tail_cpu_us\":"
           << CounterJson(sample.poller_tail_microseconds)
           << ",\"wake_calls\":" << sample.wake_calls
           << ",\"submit_calls\":" << sample.enter_calls
           << ",\"device_ticks\":" << elapsed(summary->end_tick)
           << ",\"background_iterations\":" << background.iterations
           << ",\"columns\":[\"arrival\",\"admitted\",\"ready\",\"released\","
              "\"shared\",\"slot\",\"generation\",\"read_begin\",\"read_end\","
              "\"write_begin\",\"write_end\",\"reload_begin\",\"reload_end\"]"
           << ",\"requests\":[";
    for (uint32_t i = 0; i < arguments.demand_count; ++i) {
      if (i) {
        output << ',';
      }
      const auto& row = rows[i];
      output << '[' << row.arrival_ticks << ',' << elapsed(row.admitted_tick)
             << ',' << elapsed(row.ready_tick) << ',' << elapsed(row.end_tick)
             << ',' << row.shared << ',' << row.slot << ',' << row.generation
             << ',' << elapsed(row.read_begin) << ',' << elapsed(row.read_end);
      for (uint32_t tick :
           {row.write_begin, row.write_end, row.reload_begin, row.reload_end}) {
        output << ',' << (profile.phases == 3 ? elapsed(tick) : 0);
      }
      output << ']';
    }
    output << "]}";
    std::puts(output.str().c_str());
  }

  void RunDemand(const DemandProfile& profile,
                 InputFault fault = InputFault::kNone) {
    ASSERT_LE(profile.credits, demand::kMaximumCredits);
    const auto* kernel = demand_products::kKernels.Find(gpu_endpoint_info_);
    ASSERT_NE(kernel, nullptr);
    ASSERT_NO_FATAL_FAILURE(OpenClock());
    uint32_t repetitions = 1;
    ASSERT_NO_FATAL_FAILURE(
        EnvironmentCount("AMDF_IO_REPETITIONS", 31, &repetitions));
    const bool measurement = std::getenv("AMDF_IO_REPETITIONS") != nullptr &&
                             fault == InputFault::kNone;
    if (measurement) {
      ASSERT_NE(std::getenv("BENCHMARK_LOCK_LEASE_ID"), nullptr);
    }
    const uint32_t count = profile.burst * 4;
    ASSERT_LE(count, demand::kMaximumDemands);
    const uint32_t word_count = profile.byte_length / 4;
    const uint32_t blocks = fault == InputFault::kShortFile ? 1
                            : profile.byte_length >= 262144 ? 64
                                                            : 256;
    ASSERT_LE(blocks, demand::kMaximumFileBlocks);
    std::vector<uint32_t> expected_file(
        (blocks + 3 * profile.credits) * word_count, kGuard);
    for (uint32_t block = 0; block < blocks; ++block) {
      for (uint32_t word = 0; word < word_count; ++word) {
        expected_file[block * word_count + word] = InputWord(block, word);
      }
    }
    if (fault == InputFault::kShortFile) {
      expected_file.resize(word_count / 2);
    }
    ASSERT_NO_FATAL_FAILURE(CreateFile(expected_file, GetParam()));
    if (IsSkipped()) {
      return;
    }
    ASSERT_EQ(fdatasync(data_file_), 0);
    const uint32_t stride = profile.byte_length + page_byte_length_;
    const size_t record_bytes =
        2 * kGuardWords * 4 + count * sizeof(demand::Record);
    GpuMemory* payload = nullptr;
    GpuMemory* state = nullptr;
    GpuMemory* records = nullptr;
    GpuMemory* keys = nullptr;
    GpuMemory* arguments = nullptr;
    GpuMemory* completion = nullptr;
    GpuMemory* background = nullptr;
    GpuMemory* background_arguments = nullptr;
    GpuMemory* background_completion = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateRegisteredPages(
        2 * profile.credits * stride + page_byte_length_, kGuard, &payload));
    ASSERT_NO_FATAL_FAILURE(
        CreateRing(payload, FileIoPath::kHostRelay, 1, 512));
    Ring* poll_ring = ring_;
    ASSERT_NO_FATAL_FAILURE(CreateRing(payload, FileIoPath::kHostWait, 1, 512));
    Ring* wait_ring = ring_;
    ASSERT_NO_FATAL_FAILURE(CreateRing(payload, FileIoPath::kHostPoll, 1, 512));
    Ring* host_poll_ring = ring_;
    const amdf_memory_access_t access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    ASSERT_NO_FATAL_FAILURE(CreateMemory(
        access,
        sizeof(demand::Summary) + profile.credits * sizeof(demand::Slot),
        &state));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(access, record_bytes, &records));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(access, blocks * 4, &keys));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(access, 4096, &completion));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(access, 4096, &background));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &background_arguments));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(access, 4096, &background_completion));
    Pm4ComputeProgram program = {0,
                                 kernel->program.resource1,
                                 kernel->program.resource2,
                                 kernel->program.resource3,
                                 kernel->group_segment_byte_length,
                                 {1, 1, 1}};
    ASSERT_NO_FATAL_FAILURE(PrepareProgram(kernel->executable,
                                           kernel->entry_byte_offset, &program,
                                           "file_demand"));
    struct Transport {
      // Native submission and completion ownership.
      FileIoPath path;
      // Requested host service interval; zero is a busy owner.
      uint32_t service_microseconds;
    };
    constexpr std::array transports = {Transport{FileIoPath::kDevice, 0},
                                       Transport{FileIoPath::kDevice, 50},
                                       Transport{FileIoPath::kHostRelay, 0},
                                       Transport{FileIoPath::kHostWait, 0},
                                       Transport{FileIoPath::kHostPoll, 0}};
    std::vector<uint32_t> seeds(blocks, 0);
    uint32_t remaining = blocks;
    for (uint32_t seed = 1; remaining; ++seed) {
      const uint32_t key = Hash(seed) & (blocks - 1);
      if (!seeds[key]) {
        seeds[key] = seed;
        --remaining;
      }
    }
    for (uint32_t epoch = 0; epoch <= repetitions; ++epoch) {
      std::vector<demand::Record> offered(count);
      for (uint32_t i = 0; i < count; ++i) {
        auto& row = offered[i];
        row.arrival_ticks =
            Ticks((i / profile.burst) * profile.interval_microseconds);
        const uint32_t key =
            ((i % profile.distinct_keys) * 73 + epoch * 17) & (blocks - 1);
        row.key_seed = seeds[key];
        row.hold_ticks = i % 7 == 0 ? Ticks(profile.hold_microseconds) : 0;
      }
      for (uint32_t order = 0; order < transports.size(); ++order) {
        const uint32_t direction = epoch / transports.size() % 2 ? 4 : 1;
        const auto transport =
            transports[(epoch + direction * order) % transports.size()];
        const FileIoPath path = transport.path;
        service_microseconds_ = transport.service_microseconds;
        ring_ = path == FileIoPath::kHostWait   ? wait_ring
                : path == FileIoPath::kHostPoll ? host_poll_ring
                                                : poll_ring;
        device_ring_memory_ =
            path == FileIoPath::kDevice ? ring_->memory : ring_->relay_memory;
        const uint32_t position =
            GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.tail));
        if (path != FileIoPath::kDevice) {
          std::memset(device_ring_memory_->host.pointer, 0,
                      device_ring_memory_->host.byte_length);
          const uintptr_t control =
              reinterpret_cast<uintptr_t>(device_ring_memory_->host.pointer) +
              ring_->control_offset;
          for (uint32_t offset :
               {ring_->parameters.sq_off.head, ring_->parameters.sq_off.tail,
                ring_->parameters.cq_off.head, ring_->parameters.cq_off.tail}) {
            GpuStoreRelease<uint32_t>(control + offset, position);
          }
        }
        std::fill_n(static_cast<uint32_t*>(payload->host.pointer),
                    payload->host.byte_length / 4, kGuard);
        std::memset(state->host.pointer, 0, state->host.byte_length);
        std::memset(keys->host.pointer, 0, keys->host.byte_length);
        std::fill_n(static_cast<uint32_t*>(records->host.pointer),
                    record_bytes / 4, kGuard);
        std::memcpy(static_cast<uint32_t*>(records->host.pointer) + kGuardWords,
                    offered.data(), count * sizeof(demand::Record));
        for (GpuMemory* memory :
             {completion, background, background_completion}) {
          std::memset(memory->host.pointer, 0, memory->host.byte_length);
        }
        const demand::Arguments values = {
            .submission_entries = device_ring_memory_->device_address,
            .submission_tail = RingAddress(ring_->parameters.sq_off.tail),
            .completion_entries = RingAddress(ring_->parameters.cq_off.cqes),
            .completion_head = RingAddress(ring_->parameters.cq_off.head),
            .completion_tail = RingAddress(ring_->parameters.cq_off.tail),
            .payload = payload->device_address + page_byte_length_,
            .state = state->device_address,
            .records = records->device_address + kGuardWords * 4,
            .keys = keys->device_address,
            .background = background->device_address,
            .host_payload = reinterpret_cast<uintptr_t>(payload->host.pointer) +
                            page_byte_length_,
            .submission_mask = ring_->parameters.sq_entries - 1,
            .completion_mask = ring_->parameters.cq_entries - 1,
            .initial_position = position,
            .credit_count = profile.credits,
            .demand_count = count,
            .word_count = word_count,
            .file_block_mask = blocks - 1,
            .payload_stride = stride,
            .phase_count = profile.phases,
            .file_index = fault == InputFault::kAbsentFile ? 1u : 0u,
            .role = 0,
            .background_enabled = profile.background ? 1u : 0u};
        std::memcpy(arguments->host.pointer, &values, sizeof(values));
        GpuCommandQueue* queue = nullptr;
        ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
        GpuCommandQueue* background_queue = nullptr;
        if (profile.background) {
          demand::Arguments background_values = values;
          background_values.role = 1;
          std::memcpy(background_arguments->host.pointer, &background_values,
                      sizeof(background_values));
          ASSERT_NO_FATAL_FAILURE(CreateQueue(&background_queue));
          Pm4CommandWriter background_commands(background_queue->words().data(),
                                               *pm4_profile_);
          background_commands.SystemBarrier();
          background_commands.BindCompute(program,
                                          background_arguments->device_address);
          background_commands.DispatchWave32(1, 1, 1);
          background_commands.SystemBarrier();
          background_commands.WriteData32(background_completion->device_address,
                                          1);
          background_commands.PadToEightWords();
          ASSERT_NO_FATAL_FAILURE(background_queue->Publish(
              api_, gpu_api_, background_commands.word_count()));
          const uintptr_t started =
              reinterpret_cast<uintptr_t>(background->host.pointer) +
              offsetof(demand::Background, started);
          while (!GpuLoadAcquire<uint32_t>(started)) {
            std::this_thread::yield();
          }
        }
        Pm4CommandWriter commands(queue->words().data(), *pm4_profile_);
        Sample sample;
        RunEpoch(queue, &commands, program, arguments, completion, &sample);
        if (background_queue) {
          if (HasFatalFailure()) {
            // The host owns cancellation if I/O service cannot finish. Keep
            // the arithmetic queue's storage alive until its explicit stop.
            const uintptr_t work =
                reinterpret_cast<uintptr_t>(background->host.pointer);
            GpuStoreRelease<uint32_t>(work + offsetof(demand::Background, run),
                                      1);
            GpuStoreRelease<uint32_t>(work + offsetof(demand::Background, stop),
                                      1);
          }
          const uintptr_t done =
              reinterpret_cast<uintptr_t>(background_completion->host.pointer);
          while (!GpuLoadAcquire<uint32_t>(done)) {
            std::this_thread::yield();
          }
          ASSERT_NO_FATAL_FAILURE(background_queue->WaitRetired(api_));
          ASSERT_TRUE(background_queue->Release(api_));
        }
        ASSERT_FALSE(HasFatalFailure());
        ASSERT_TRUE(queue->Release(api_));
        const auto* work =
            static_cast<const demand::Background*>(background->host.pointer);
        ASSERT_EQ(work->started, profile.background ? 1u : 0u);
        ASSERT_EQ(work->run, 1u);
        ASSERT_EQ(work->stop, 1u);
        uint32_t result = profile.background ? 0x1234567 : 0;
        if (profile.background) {
          ASSERT_GT(work->iterations, 0u);
        }
        for (uint32_t i = 0; i < work->iterations; ++i) {
          result = Hash(result + i);
        }
        ASSERT_EQ(work->result, result);
        ASSERT_NO_FATAL_FAILURE(CheckDemand(profile, values, fault, payload,
                                            state, records, keys, offered,
                                            &expected_file));
        ASSERT_EQ(std::memcmp(arguments->host.pointer, &values, sizeof(values)),
                  0);
        if (profile.phases == 3) {
          ASSERT_EQ(fdatasync(data_file_), 0);
        }
        if (measurement && epoch != 0) {
          PrintDemand(profile, values, path, epoch, sample, state, records,
                      *work);
        }
      }
    }
    ASSERT_NO_FATAL_FAILURE(VerifyFile(expected_file, GetParam()));
  }
};

TEST_P(GpuFileDemandTest, LookupBursts32) {
  RunDemand({"lookup32", 4096, 32, 32, 256, 1000, 0, 1, false});
}
TEST_P(GpuFileDemandTest, LookupBursts64) {
  RunDemand({"lookup64", 4096, 64, 64, 256, 1000, 0, 1, false});
}
TEST_P(GpuFileDemandTest, LookupBursts128) {
  RunDemand({"lookup128", 4096, 128, 128, 256, 1000, 0, 1, false});
}
TEST_P(GpuFileDemandTest, LookupBursts256) {
  RunDemand({"lookup256", 4096, 256, 256, 256, 1000, 0, 1, false});
}
TEST_P(GpuFileDemandTest, LookupBacklog256On32Credits) {
  RunDemand({"lookup_backlog", 4096, 256, 32, 256, 1000, 200, 1, false});
}
TEST_P(GpuFileDemandTest, StaggeredLookupWithGpuWork) {
  RunDemand({"lookup_staggered", 4096, 32, 64, 256, 40, 0, 1, true});
}
TEST_P(GpuFileDemandTest, SharedExpertTiles256KiB) {
  RunDemand({"expert_tiles", 262144, 256, 32, 16, 1000, 2000, 1, false});
}
TEST_P(GpuFileDemandTest, ExpertTileCreditPressureWithGpuWork) {
  RunDemand({"expert_pressure", 262144, 256, 8, 64, 1000, 200, 1, true});
}
TEST_P(GpuFileDemandTest, ScatteredKvBlocks64KiB) {
  RunDemand({"kv_blocks", 65536, 256, 64, 256, 1000, 200, 3, false});
}
TEST_P(GpuFileDemandTest, InvalidFileDrainsConcurrentDemands) {
  RunDemand({"invalid", 4096, 256, 64, 256, 1000, 0, 1, false},
            InputFault::kAbsentFile);
}
TEST_P(GpuFileDemandTest, PartialInputDrainsHeldDemands) {
  RunDemand({"partial", 4096, 32, 1, 1, 1000, 200, 1, false},
            InputFault::kShortFile);
}

INSTANTIATE_TEST_SUITE_P(FileModes, GpuFileDemandTest,
                         ::testing::Values(FileMode::kBuffered,
                                           FileMode::kDirect),
                         [](const auto& info) {
                           return info.param == FileMode::kDirect ? "Direct"
                                                                  : "Buffered";
                         });

}  // namespace
