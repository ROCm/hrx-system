// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fcntl.h>
#include <sys/vfs.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "libamdf/cts/util/mapped_memory.h"
#include "libamdf/cts/xdna/programs/resident_file_channel.h"
#include "libamdf/cts/xdna/util/executable.h"
#include "libamdf/cts/xdna/util/execution.h"
#include "libamdf/cts/xdna/util/resident_transaction.h"
#include "libamdf/cts/xdna/xdna_device_fixture.h"

namespace {

constexpr size_t kControlOffset = 64;
constexpr size_t kPayloadOffset = 4096;
constexpr size_t kStorageLength = 3 * 4096;
constexpr uint32_t kFileRecords = 64;
constexpr uint32_t kSeed = 0xe719bc53;
constexpr uint32_t kGuardWord = 0xa5a5a5a5;
constexpr uint32_t kFaultGeneration = 3;

enum Buffer : size_t {
  kConfiguration,
  kControl,
  kRequest,
  kResponse,
  kTerminal,
  kBufferCount,
};
enum class FileFault { kNone, kEndOfFile, kReadOnly };
enum class Startup : uint32_t { kRun = 1, kAbort = 2 };
enum class Consumer { kCpu, kNpu };

struct Shape {
  // Words per individual pread and pwrite record.
  uint32_t record_words;
  // Records serviced before returning the batch credit to the NPU.
  uint32_t batch_records;
  // Native allocation or registration of separately retained host storage.
  amdf_memory_profile_roles_t backing;

  uint32_t words() const { return record_words * batch_records; }
  uint32_t bytes() const { return words() * sizeof(uint32_t); }
};

uint32_t InputWord(uint32_t record, uint32_t word) {
  uint32_t value = 0x917af531u + record * 0x9e3779b9u + word * 0x85ebca6bu;
  value ^= value << 13;
  value ^= value >> 17;
  return value ^ (value << 5);
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

void StoreWord(std::span<uint8_t> bytes, size_t offset, uint32_t value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void CopyWords(std::span<uint8_t> bytes, size_t offset,
               std::span<const uint32_t> words) {
  std::memcpy(bytes.data() + offset, words.data(), words.size_bytes());
}

// Positive short transfers advance the exact file and memory extents. EOF and
// zero-progress writes terminate the channel instead of retrying indefinitely.
int ReadAll(int descriptor, std::span<uint8_t> bytes, off_t offset) {
  while (!bytes.empty()) {
    const ssize_t count = pread(descriptor, bytes.data(), bytes.size(), offset);
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -errno;
    }
    if (count == 0) {
      return -ENODATA;
    }
    bytes = bytes.subspan(static_cast<size_t>(count));
    offset += count;
  }
  return 0;
}

int WriteAll(int descriptor, std::span<const uint8_t> bytes, off_t offset) {
  while (!bytes.empty()) {
    const ssize_t count =
        pwrite(descriptor, bytes.data(), bytes.size(), offset);
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -errno;
    }
    if (count == 0) {
      return -EIO;
    }
    bytes = bytes.subspan(static_cast<size_t>(count));
    offset += count;
  }
  return 0;
}

class FileRecords {
 public:
  void Create(const Shape& shape, FileFault fault) {
    ASSERT_NO_FATAL_FAILURE(CreateUnlinked(&descriptor_));
    record_bytes_ = shape.record_words * sizeof(uint32_t);
    initial_.resize(2 * kFileRecords * record_bytes_, 0xa5);
    for (uint32_t record = 0; record < kFileRecords; ++record) {
      for (uint32_t word = 0; word < shape.record_words; ++word) {
        StoreWord(initial_, record * record_bytes_ + word * sizeof(uint32_t),
                  InputWord(record, word));
      }
    }
    if (fault == FileFault::kEndOfFile) {
      ASSERT_NO_FATAL_FAILURE(CreateUnlinked(&fault_descriptor_));
    } else if (fault == FileFault::kReadOnly) {
      const std::string path = "/proc/self/fd/" + std::to_string(descriptor_);
      fault_descriptor_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
      ASSERT_GE(fault_descriptor_, 0) << std::strerror(errno);
    }
    struct statfs filesystem = {};
    ASSERT_EQ(fstatfs(descriptor_, &filesystem), 0);
    filesystem_type_ = filesystem.f_type;
  }

  void Close() {
    if (fault_descriptor_ >= 0) {
      EXPECT_EQ(close(fault_descriptor_), 0);
      fault_descriptor_ = -1;
    }
    if (descriptor_ >= 0) {
      EXPECT_EQ(close(descriptor_), 0);
      descriptor_ = -1;
    }
  }

  void Reset() { ASSERT_EQ(WriteAll(descriptor_, initial_, 0), 0); }

  // The request's first word selects each file record. The immutable input
  // bank and output bank are disjoint even when several keys coincide.
  int Exchange(std::span<const uint32_t> request, std::span<uint32_t> response,
               const Shape& shape, uint32_t generation, FileFault fault) {
    const int descriptor =
        fault != FileFault::kNone && generation == kFaultGeneration
            ? fault_descriptor_
            : descriptor_;
    for (uint32_t record = 0; record < shape.batch_records; ++record) {
      const size_t word_offset = record * shape.record_words;
      const uint32_t key = request[word_offset] % kFileRecords;
      auto returned = response.subspan(word_offset, shape.record_words);
      const int read_result = ReadAll(
          descriptor,
          {reinterpret_cast<uint8_t*>(returned.data()), returned.size_bytes()},
          key * record_bytes_);
      if (read_result != 0) {
        return read_result;
      }
      const auto written = request.subspan(word_offset, shape.record_words);
      const int write_result =
          WriteAll(descriptor,
                   {reinterpret_cast<const uint8_t*>(written.data()),
                    written.size_bytes()},
                   (kFileRecords + key) * record_bytes_);
      if (write_result != 0) {
        return write_result;
      }
    }
    return 0;
  }

  void Check(std::span<const uint8_t> expected) {
    std::vector<uint8_t> observed(expected.size());
    ASSERT_EQ(ReadAll(descriptor_, observed, 0), 0);
    EXPECT_TRUE(std::equal(observed.begin(), observed.end(), expected.begin()));
    uint8_t beyond = 0;
    ASSERT_EQ(pread(descriptor_, &beyond, 1, expected.size()), 0);
  }

  std::span<const uint8_t> initial() const { return initial_; }
  int64_t filesystem_type() const { return filesystem_type_; }

 private:
  static void CreateUnlinked(int* descriptor) {
    const char* directory = std::getenv("TEST_TMPDIR");
    std::string path =
        std::string(directory ? directory : "/tmp") + "/amdf-npu-file-XXXXXX";
    *descriptor = mkstemp(path.data());
    ASSERT_GE(*descriptor, 0) << std::strerror(errno);
    ASSERT_EQ(unlink(path.c_str()), 0) << std::strerror(errno);
  }

  // Owned private file, unlinked immediately after creation.
  int descriptor_ = -1;
  // Owned empty or read-only file used to exercise an actual syscall failure.
  int fault_descriptor_ = -1;
  // Complete immutable input bank followed by guarded output slots.
  std::vector<uint8_t> initial_;
  // Byte length of a single file record.
  uint32_t record_bytes_ = 0;
  // Observed filesystem magic; a buffered tmpfs run is not an NVMe result.
  int64_t filesystem_type_ = 0;
};

struct Sample {
  // Complete invocation, including startup and final native completion.
  uint64_t wall_nanoseconds = 0;
  // CPU service thread time, including cache maintenance and polling.
  uint64_t thread_nanoseconds = 0;
  // CPU-observed ready-to-ready batch cycles, not divided by record count.
  std::vector<uint64_t> cycles;
  // Every complete emitted batch, including the closing payload.
  std::vector<uint32_t> transcript;
  // Number of batches captured in transcript, including a failing request.
  uint32_t request_count = 0;
  // Negative errno from actual file I/O, or zero on success.
  int file_error = 0;
};

class NpuFileChannelTest : public XdnaDeviceFixture,
                           public ::testing::WithParamInterface<Shape> {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(XdnaDeviceFixture::SetUp());
    amdf_xdna_endpoint_info_t endpoint_info = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO,
        .structure_size = sizeof(endpoint_info)};
    ASSERT_EQ(xdna_api_->endpoint_query_info(endpoint_, &endpoint_info),
              AMDF_STATUS_OK);
    RecordProperty("amdf_xdna_target", endpoint_info.target_id);
    amdf_xdna_device_info_t device_info = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO,
        .structure_size = sizeof(device_info)};
    ASSERT_EQ(xdna_api_->device_query_info(device_, &device_info),
              AMDF_STATUS_OK);
    ASSERT_TRUE(FindXdnaKernelQueueFamily(api_, endpoint_, &family_ordinal_));
    const iree_file_toc_t* image = nullptr;
    const std::string_view target = endpoint_info.target_id;
    if (target == "amd.xdna.strix_halo.17f0_11") {
      image = &amdf_cts_xdna_resident_file_channel_create()[1];
    } else if (target == "amd.xdna.strix.17f0_10" ||
               target == "amd.xdna.krackan.17f0_20") {
      image = &amdf_cts_xdna_resident_file_channel_create()[0];
    } else {
      GTEST_SKIP() << "no resident file-channel fixture for " << target;
    }
    constexpr std::array<amdf_memory_access_t, 2> accesses = {
        AMDF_MEMORY_ACCESS_READ, AMDF_MEMORY_ACCESS_WRITE};
    ASSERT_TRUE(executable_.Initialize(
        {reinterpret_cast<const uint8_t*>(image->data), image->size},
        endpoint_info, device_info, 1, accesses));
    memory_access_.requirements.address_kinds = UINT64_C(1)
                                                << AMDF_MEMORY_ADDRESS_XDNA_DMA;
  }

  void TearDown() override {
    // Failed native retirement retains addressed owners rather than recycling
    // storage while a service might still reach it.
    if (!execution_.Release(api_, xdna_api_)) {
      return;
    }
    for (auto& buffer : buffers_) {
      if (!buffer.Release(api_)) {
        return;
      }
    }
    for (auto& backing : backing_) {
      if (!backing.Release(api_)) {
        return;
      }
    }
    file_.Close();
  }

  void Prepare(FileFault fault) {
    const auto shape = GetParam();
    const uint32_t ordinal = FindMemoryProfileOrdinal(
        shape.backing | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        AMDF_MEMORY_FLAG_HOST_VISIBLE);
    ASSERT_NE(ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    amdf_memory_profile_t profile = {.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
                                     .structure_size = sizeof(profile)};
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities)};
    ASSERT_EQ(QueryMemoryProfile(ordinal, &profile, &capabilities),
              AMDF_STATUS_OK);
    const auto& geometry = shape.backing == AMDF_MEMORY_PROFILE_ROLE_REGISTER
                               ? profile.registration
                               : profile.allocation;
    ASSERT_GT(geometry.byte_length_granularity, 0u);
    for (size_t i = 0; i < buffers_.size(); ++i) {
      amdf_memory_create_info_t create =
          {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment order
               // differs from declaration order.
      create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
      create.structure_size = sizeof(create);
      create.memory_profile_ordinal = ordinal;
      create.access_count = 1;
      create.accesses = &memory_access_;
      create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
      create.byte_length =
          (kStorageLength + geometry.byte_length_granularity - 1) /
          geometry.byte_length_granularity * geometry.byte_length_granularity;
      create.minimum_alignment = geometry.minimum_alignment;
      if (shape.backing == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
        amdf_memory_create_info_t host = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
            .structure_size = sizeof(host),
            .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
            .byte_length = create.byte_length,
            .minimum_alignment = geometry.registered_host_pointer_alignment};
        ASSERT_NO_FATAL_FAILURE(backing_[i].Create(api_, system_scope_, host));
        ASSERT_EQ(backing_[i].host.cacheability,
                  geometry.registered_host_cacheability);
        create.registered_host_pointer = backing_[i].host.pointer;
        create.registered_host_cacheability =
            geometry.registered_host_cacheability;
      }
      ASSERT_NO_FATAL_FAILURE(buffers_[i].Create(api_, system_scope_, create));
      ASSERT_EQ(api_->memory_query_address(buffers_[i].memory, 0,
                                           AMDF_MEMORY_ADDRESS_XDNA_DMA,
                                           &addresses_[i]),
                AMDF_STATUS_OK);
      ASSERT_LE(buffers_[i].host.cache_line_size, kControlOffset);
      const auto host = buffers_[i].HostSite();
      const auto device = buffers_[i].DeviceSite(0, family_ordinal_);
      amdf_memory_pair_info_t ingress = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
          .structure_size = sizeof(ingress)};
      ASSERT_EQ(api_->memory_query_pair_info(&host, &device, &ingress),
                AMDF_STATUS_OK);
      amdf_memory_pair_info_t egress = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
          .structure_size = sizeof(egress)};
      ASSERT_EQ(api_->memory_query_pair_info(&device, &host, &egress),
                AMDF_STATUS_OK);
      ASSERT_NE(ingress.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
                0u);
      ASSERT_NE(egress.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
                0u);
      ASSERT_EQ(ingress.release.host_operation,
                AMDF_HOST_CACHE_OPERATION_FLUSH);
      ASSERT_EQ(ingress.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
      ASSERT_EQ(egress.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
      ASSERT_EQ(egress.acquire.host_operation,
                AMDF_HOST_CACHE_OPERATION_INVALIDATE);
    }
    const std::array<uint64_t, 2> bindings = {
        addresses_[kConfiguration] + kControlOffset,
        addresses_[kTerminal] + kControlOffset};
    std::vector<uint8_t> image(executable_.allocation_byte_length());
    executable_.Load(image);
    ASSERT_TRUE(executable_.Bind(image, bindings));
    // Utility names follow physical NPU input/output, not channel initiator.
    const std::array<ResidentNpuSlot, 1> slots = {ResidentNpuSlot{
        addresses_[kResponse] + kControlOffset,
        addresses_[kResponse] + kPayloadOffset,
        addresses_[kRequest] + kPayloadOffset,
        addresses_[kRequest] + kControlOffset,
    }};
    const std::array<ResidentNpuAddresses, 1> services = {ResidentNpuAddresses{
        addresses_[kControl] + kControlOffset, slots,
        addresses_[kControl] + kResidentFinalAckByteOffset}};
    std::vector<uint8_t> commands;
    ASSERT_TRUE(BuildResidentTransaction(
        executable_.ResolveInvocation(image), services, shape.bytes(),
        ResidentResponsePath::kChained, &commands));
    ASSERT_NO_FATAL_FAILURE(
        execution_.Prepare(api_, xdna_api_, device_, family_ordinal_, 1,
                           commands, executable_.allocation_alignment()));
    ASSERT_EQ(
        api_->host_mapping_cache_control(
            execution_.instructions.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
            execution_.instructions.host.byte_length),
        AMDF_STATUS_OK);
    original_commands_.assign(execution_.instructions.bytes().begin(),
                              execution_.instructions.bytes().end());
    ASSERT_NO_FATAL_FAILURE(file_.Create(shape, fault));
    RecordProperty("file_system_type", std::to_string(file_.filesystem_type()));
  }

  void Reset(uint32_t rounds) {
    ASSERT_NO_FATAL_FAILURE(file_.Reset());
    for (auto& buffer : buffers_) {
      std::fill(buffer.bytes().begin(), buffer.bytes().end(), 0xa5);
    }
    const auto configuration = buffers_[kConfiguration].bytes();
    std::fill_n(configuration.begin() + kControlOffset, 64, 0);
    StoreWord(configuration, kControlOffset, rounds);
    StoreWord(configuration, kControlOffset + 4, GetParam().words());
    StoreWord(configuration, kControlOffset + 8, 1);
    StoreWord(configuration, kControlOffset + 12, kSeed);
    for (Buffer ordinal : {kControl, kRequest, kResponse}) {
      StoreWord(buffers_[ordinal].bytes(), kControlOffset, 0);
    }
    StoreWord(buffers_[kControl].bytes(), kResidentFinalAckByteOffset, 0);
    for (size_t i = 0; i < buffers_.size(); ++i) {
      const auto bytes = buffers_[i].bytes();
      initial_[i].assign(bytes.begin(), bytes.end());
      ASSERT_EQ(Cache(i, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, bytes.size()),
                AMDF_STATUS_OK);
    }
  }

  amdf_status_t Cache(size_t ordinal, amdf_host_cache_operation_t operation,
                      uint64_t offset, uint64_t length) {
    return api_->host_mapping_cache_control(buffers_[ordinal].mapping,
                                            operation, offset, length);
  }

  amdf_status_t Publish(size_t ordinal, size_t offset, uint32_t value) {
    StoreWord(buffers_[ordinal].bytes(), offset, value);
    return Cache(ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH, offset,
                 sizeof(value));
  }

  amdf_status_t AcquireRequest(uint32_t generation) {
    const auto* ready = reinterpret_cast<volatile const uint32_t*>(
        buffers_[kRequest].bytes().data() + kControlOffset);
    // This line has no live CPU writes. Invalidation cannot overwrite a
    // neighboring NPU field with dirty CPU data, and credits protect payloads.
    do {
      const amdf_status_t status =
          Cache(kRequest, AMDF_HOST_CACHE_OPERATION_INVALIDATE, kControlOffset,
                sizeof(uint32_t));
      if (status != AMDF_STATUS_OK) {
        return status;
      }
    } while (*ready != generation);
    return AMDF_STATUS_OK;
  }

  std::span<uint32_t> Payload(size_t ordinal) {
    return {reinterpret_cast<uint32_t*>(buffers_[ordinal].bytes().data() +
                                        kPayloadOffset),
            GetParam().words()};
  }

  Sample AllocateSample(uint32_t rounds) {
    Sample result;
    result.cycles.reserve(rounds);
    result.transcript.resize(size_t{rounds + 1} * GetParam().words());
    return result;
  }

  void RunNpu(uint32_t rounds, Startup startup, FileFault fault,
              Sample* sample) {
    uint64_t thread_begin = 0;
    ASSERT_NO_FATAL_FAILURE(ThreadNanoseconds(&thread_begin));
    const uint64_t wall_begin = WallNanoseconds();
    amdf_xdna_kernel_queue_submission_info_t submit = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
        .structure_size = sizeof(submit),
        .command_count = 1,
        .commands = &execution_.command};
    uint64_t point = 0;
    ASSERT_EQ(xdna_api_->kernel_queue_submit(execution_.queue, &submit, &point),
              AMDF_STATUS_OK);
    ASSERT_EQ(Publish(kControl, kControlOffset, static_cast<uint32_t>(startup)),
              AMDF_STATUS_OK);
    uint64_t previous_ready = 0;
    if (startup == Startup::kRun && rounds != 0) {
      for (uint32_t generation = 1; generation <= rounds + 1; ++generation) {
        ASSERT_EQ(AcquireRequest(generation), AMDF_STATUS_OK);
        const uint64_t ready = WallNanoseconds();
        if (previous_ready != 0) {
          sample->cycles.push_back(ready - previous_ready);
        }
        previous_ready = ready;
        ASSERT_EQ(Cache(kRequest, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                        kPayloadOffset, GetParam().bytes()),
                  AMDF_STATUS_OK);
        const auto request = Payload(kRequest);
        std::copy(request.begin(), request.end(),
                  sample->transcript.begin() +
                      size_t{sample->request_count++} * request.size());
        if (generation > rounds) {
          break;
        }
        sample->file_error = file_.Exchange(request, Payload(kResponse),
                                            GetParam(), generation, fault);
        if (sample->file_error != 0) {
          ASSERT_EQ(Publish(kResponse, kControlOffset,
                            static_cast<uint32_t>(sample->file_error)),
                    AMDF_STATUS_OK);
          break;
        }
        ASSERT_EQ(Cache(kResponse, AMDF_HOST_CACHE_OPERATION_FLUSH,
                        kPayloadOffset, GetParam().bytes()),
                  AMDF_STATUS_OK);
        ASSERT_EQ(Publish(kResponse, kControlOffset, generation),
                  AMDF_STATUS_OK);
      }
    }
    if (startup == Startup::kRun && sample->file_error == 0) {
      ASSERT_EQ(Publish(kControl, kResidentFinalAckByteOffset, 1),
                AMDF_STATUS_OK);
    }
    // Even failed I/O takes the device's terminal path and drains DMA before
    // any error assertion or addressed-owner release.
    ASSERT_EQ(api_->kernel_queue_wait(execution_.queue, point,
                                      AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
    sample->wall_nanoseconds = WallNanoseconds() - wall_begin;
    uint64_t thread_end = 0;
    ASSERT_NO_FATAL_FAILURE(ThreadNanoseconds(&thread_end));
    sample->thread_nanoseconds = thread_end - thread_begin;
    amdf_kernel_queue_status_t status = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
        .structure_size = sizeof(status)};
    ASSERT_EQ(api_->kernel_queue_query_status(execution_.queue, &status),
              AMDF_STATUS_OK);
    ASSERT_EQ(status.retired_submission, point);
    ASSERT_EQ(status.terminal_status, AMDF_STATUS_OK);
  }

  void RunCpu(uint32_t rounds, Sample* sample) {
    std::vector<uint32_t> request(GetParam().words());
    std::vector<uint32_t> response(request.size());
    uint64_t thread_begin = 0;
    ASSERT_NO_FATAL_FAILURE(ThreadNanoseconds(&thread_begin));
    const uint64_t wall_begin = WallNanoseconds();
    for (uint32_t i = 0; i < request.size(); ++i) {
      request[i] = kSeed + 257u + 17u * i;
    }
    uint64_t previous_ready = 0;
    for (uint32_t generation = 1; generation <= rounds + 1; ++generation) {
      const uint64_t ready = WallNanoseconds();
      if (previous_ready != 0) {
        sample->cycles.push_back(ready - previous_ready);
      }
      previous_ready = ready;
      std::copy(request.begin(), request.end(),
                sample->transcript.begin() +
                    size_t{sample->request_count++} * request.size());
      if (generation > rounds) {
        break;
      }
      sample->file_error = file_.Exchange(request, response, GetParam(),
                                          generation, FileFault::kNone);
      if (sample->file_error != 0) {
        break;
      }
      for (uint32_t i = 0; i < request.size(); ++i) {
        request[i] = response[i] + 257u * (generation + 1) + 17u * i;
      }
    }
    sample->wall_nanoseconds = WallNanoseconds() - wall_begin;
    uint64_t thread_end = 0;
    ASSERT_NO_FATAL_FAILURE(ThreadNanoseconds(&thread_end));
    sample->thread_nanoseconds = thread_end - thread_begin;
  }

  void Verify(const Sample& sample, uint32_t rounds, Startup startup,
              FileFault fault, Consumer consumer) {
    const auto shape = GetParam();
    const bool has_payload = startup == Startup::kRun && rounds != 0;
    const int expected_error = fault == FileFault::kEndOfFile  ? -ENODATA
                               : fault == FileFault::kReadOnly ? -EBADF
                                                               : 0;
    const uint32_t request_count = !has_payload     ? 0
                                   : expected_error ? kFaultGeneration
                                                    : rounds + 1;
    EXPECT_EQ(sample.file_error, expected_error);
    ASSERT_EQ(sample.request_count, request_count);
    EXPECT_EQ(sample.cycles.size(), request_count == 0 ? 0 : request_count - 1);
    std::vector<uint32_t> request(shape.words());
    std::vector<uint32_t> response(shape.words(), kGuardWord);
    std::vector<uint32_t> transcript(sample.transcript.size());
    std::vector<uint8_t> expected_file(file_.initial().begin(),
                                       file_.initial().end());
    for (uint32_t i = 0; i < request.size(); ++i) {
      request[i] = kSeed + 257u + 17u * i;
    }
    for (uint32_t generation = 1; generation <= request_count; ++generation) {
      std::copy(request.begin(), request.end(),
                transcript.begin() + size_t{generation - 1} * shape.words());
      if (generation > rounds) {
        break;
      }
      const bool failing = expected_error && generation == kFaultGeneration;
      for (uint32_t record = 0; record < shape.batch_records; ++record) {
        if (failing && fault == FileFault::kEndOfFile) {
          break;
        }
        const size_t start = record * shape.record_words;
        const uint32_t key = request[start] % kFileRecords;
        for (uint32_t word = 0; word < shape.record_words; ++word) {
          response[start + word] = InputWord(key, word);
        }
        if (failing) {
          break;
        }
        CopyWords(expected_file,
                  (kFileRecords + key) * shape.record_words * sizeof(uint32_t),
                  std::span(request).subspan(start, shape.record_words));
      }
      if (failing) {
        break;
      }
      for (uint32_t i = 0; i < request.size(); ++i) {
        request[i] = response[i] + 257u * (generation + 1) + 17u * i;
      }
    }
    EXPECT_EQ(sample.transcript, transcript);
    ASSERT_NO_FATAL_FAILURE(file_.Check(expected_file));
    if (consumer == Consumer::kCpu) {
      return;
    }

    auto expected = initial_;
    StoreWord(expected[kControl], kControlOffset,
              static_cast<uint32_t>(startup));
    if (startup == Startup::kRun && expected_error == 0) {
      StoreWord(expected[kControl], kResidentFinalAckByteOffset, 1);
    }
    std::array<uint32_t, 16> terminal = {};
    terminal[0] = startup == Startup::kAbort ? 2 : expected_error ? 3 : 1;
    terminal[1] = startup == Startup::kAbort ? 0
                  : expected_error           ? kFaultGeneration - 1
                                             : rounds;
    terminal[2] = shape.words();
    terminal[6] = static_cast<uint32_t>(expected_error);
    if (has_payload) {
      terminal[3] = request.front();
      terminal[4] = request.back();
      for (uint32_t word : request) {
        terminal[5] += word;
      }
      CopyWords(expected[kRequest], kPayloadOffset, request);
      StoreWord(expected[kRequest], kControlOffset, request_count);
      CopyWords(expected[kResponse], kPayloadOffset, response);
      StoreWord(
          expected[kResponse], kControlOffset,
          expected_error ? static_cast<uint32_t>(expected_error) : rounds);
    }
    CopyWords(expected[kTerminal], kControlOffset, terminal);
    for (size_t i = 0; i < buffers_.size(); ++i) {
      SCOPED_TRACE(i);
      const auto bytes = buffers_[i].bytes();
      ASSERT_EQ(Cache(i, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0, bytes.size()),
                AMDF_STATUS_OK);
      const auto mismatch =
          std::mismatch(bytes.begin(), bytes.end(), expected[i].begin());
      EXPECT_EQ(mismatch.first, bytes.end())
          << "allocation byte " << (mismatch.first - bytes.begin());
    }
    ASSERT_EQ(api_->host_mapping_cache_control(
                  execution_.instructions.mapping,
                  AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                  execution_.instructions.host.byte_length),
              AMDF_STATUS_OK);
    EXPECT_TRUE(std::equal(original_commands_.begin(), original_commands_.end(),
                           execution_.instructions.bytes().begin()));
  }

  void Exercise(uint32_t rounds, Startup startup, FileFault fault) {
    ASSERT_NO_FATAL_FAILURE(Prepare(fault));
    ASSERT_NO_FATAL_FAILURE(Reset(rounds));
    auto sample = AllocateSample(rounds);
    ASSERT_NO_FATAL_FAILURE(RunNpu(rounds, startup, fault, &sample));
    ASSERT_NO_FATAL_FAILURE(
        Verify(sample, rounds, startup, fault, Consumer::kNpu));
  }

  void Compare() {
    uint32_t phases = 2;
    uint32_t rounds = 17;
    const char* repetitions = std::getenv("AMDF_NPU_FILE_REPETITIONS");
    if (repetitions) {
      ASSERT_NE(std::getenv("BENCHMARK_LOCK_LEASE_ID"), nullptr)
          << "measurement requires the benchmark broker lease";
      char* end = nullptr;
      errno = 0;
      const unsigned long value = std::strtoul(repetitions, &end, 10);
      ASSERT_EQ(errno, 0);
      ASSERT_NE(end, repetitions);
      ASSERT_EQ(*end, 0);
      ASSERT_GE(value, 2u);
      ASSERT_LE(value, 63u);
      phases = static_cast<uint32_t>(value);
      rounds = 1024;
    }
    ASSERT_NO_FATAL_FAILURE(Prepare(FileFault::kNone));
    // The first CPU and NPU phases warm the actual file and native execution.
    // Every subsequent phase alternates A/B; all phases verify the same data.
    for (uint32_t phase = 0; phase < phases + 2 && !HasFailure(); ++phase) {
      const Consumer consumer =
          phase % 2 == 0 ? Consumer::kCpu : Consumer::kNpu;
      ASSERT_NO_FATAL_FAILURE(Reset(rounds));
      auto sample = AllocateSample(rounds);
      if (consumer == Consumer::kCpu) {
        ASSERT_NO_FATAL_FAILURE(RunCpu(rounds, &sample));
      } else {
        ASSERT_NO_FATAL_FAILURE(
            RunNpu(rounds, Startup::kRun, FileFault::kNone, &sample));
      }
      ASSERT_NO_FATAL_FAILURE(
          Verify(sample, rounds, Startup::kRun, FileFault::kNone, consumer));
      if (!repetitions || phase < 2 || HasFailure()) {
        continue;
      }
      const auto shape = GetParam();
      std::printf(
          "AMDF_NPU_FILE {\"consumer\":\"%s\",\"phase\":%u,"
          "\"record_bytes\":%u,\"batch_records\":%u,\"rounds\":%u,"
          "\"registered\":%s,\"filesystem_type\":%lld,"
          "\"wall_ns\":%llu,\"thread_ns\":%llu,\"cycles_ns\":[",
          consumer == Consumer::kCpu ? "cpu" : "npu", phase - 2,
          shape.record_words * 4, shape.batch_records, rounds,
          shape.backing == AMDF_MEMORY_PROFILE_ROLE_REGISTER ? "true" : "false",
          static_cast<long long>(file_.filesystem_type()),
          static_cast<unsigned long long>(sample.wall_nanoseconds),
          static_cast<unsigned long long>(sample.thread_nanoseconds));
      for (size_t i = 0; i < sample.cycles.size(); ++i) {
        std::printf("%s%llu", i == 0 ? "" : ",",
                    static_cast<unsigned long long>(sample.cycles[i]));
      }
      std::printf("]}\n");
    }
  }

  // Native transaction family; no per-request native submissions are made.
  uint32_t family_ordinal_ = UINT32_MAX;
  // Admitted build-generated program borrowing its immutable image.
  XdnaExecutable executable_;
  // Case-owned native queue, context and immutable command storage.
  XdnaExecution execution_;
  // Optional CPU allocations retained through registration teardown.
  std::array<CtsMappedMemory, kBufferCount> backing_;
  // Disjoint physical-writer allocations, retained through native retirement.
  std::array<CtsMappedMemory, kBufferCount> buffers_;
  // Independently queried NPU DMA addresses, not assumed CPU virtual addresses.
  std::array<uint64_t, kBufferCount> addresses_ = {};
  // Cold snapshots for whole-allocation guard and payload oracles.
  std::array<std::vector<uint8_t>, kBufferCount> initial_;
  // Exact command allocation including its guards, checked after native join.
  std::vector<uint8_t> original_commands_;
  // Private pread/pwrite backing; no persistent user file is opened.
  FileRecords file_;
};

TEST_P(NpuFileChannelTest, ReturnsEveryFileWord) {
  Exercise(17, Startup::kRun, FileFault::kNone);
}

TEST_P(NpuFileChannelTest, ZeroRounds) {
  Exercise(0, Startup::kRun, FileFault::kNone);
}

TEST_P(NpuFileChannelTest, AbortBeforeRequests) {
  Exercise(17, Startup::kAbort, FileFault::kNone);
}

TEST_P(NpuFileChannelTest, EndOfFileRetiresResidentProgram) {
  Exercise(17, Startup::kRun, FileFault::kEndOfFile);
}

TEST_P(NpuFileChannelTest, WriteErrorRetiresResidentProgram) {
  Exercise(17, Startup::kRun, FileFault::kReadOnly);
}

TEST_P(NpuFileChannelTest, CompareCpu) { Compare(); }

INSTANTIATE_TEST_SUITE_P(
    BackingAndBatch, NpuFileChannelTest,
    ::testing::Values(Shape{16, 1, AMDF_MEMORY_PROFILE_ROLE_CREATE},
                      Shape{16, 8, AMDF_MEMORY_PROFILE_ROLE_CREATE},
                      Shape{16, 64, AMDF_MEMORY_PROFILE_ROLE_CREATE},
                      Shape{1024, 1, AMDF_MEMORY_PROFILE_ROLE_CREATE},
                      Shape{16, 1, AMDF_MEMORY_PROFILE_ROLE_REGISTER},
                      Shape{16, 8, AMDF_MEMORY_PROFILE_ROLE_REGISTER},
                      Shape{16, 64, AMDF_MEMORY_PROFILE_ROLE_REGISTER},
                      Shape{1024, 1, AMDF_MEMORY_PROFILE_ROLE_REGISTER}),
    [](const auto& info) {
      const auto& shape = info.param;
      return std::string(shape.backing == AMDF_MEMORY_PROFILE_ROLE_CREATE
                             ? "Allocated"
                             : "Registered") +
             "Records" + std::to_string(shape.batch_records) + "Bytes" +
             std::to_string(shape.record_words * 4);
    });

}  // namespace
