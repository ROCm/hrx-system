// Bounded experiment using the real serving command JIT and queue owner.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

#include "iree/base/tooling/flags.h"
#include "iree/hal/utils/statistics_sink.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/program/jit.h"
#include "loom/serve/testing/compiler.h"
#include "loom/serve/testing/device.h"

IREE_FLAG(string, jpeg_fixture, "",
          "Original-byte fixture and independent expectations.");
IREE_FLAG(bool, jpeg_access, false, "Compile device access assertions.");
IREE_FLAG(bool, jpeg_benchmark, false,
          "Measure the authored batch per submission.");
IREE_FLAG(bool, jpeg_profile, false,
          "Timestamp one separate batch replay; no counters.");
IREE_FLAG(int32_t, jpeg_repetitions, 8,
          "Decode repetitions in the authored batch, checked against its "
          "dispatch count.");
IREE_FLAG(string, jpeg_regions, "",
          "Explicit verification regions for unpublished partial outputs of "
          "rejected images.");
IREE_FLAG(string, jpeg_initial, "",
          "Directory of explicit per-binding initial NPY values.");
IREE_FLAG(string, jpeg_capture, "",
          "Directory for raw post-command binding captures.");

namespace {

struct Region {
  // First byte of the independently specified output contract.
  size_t offset;
  // Byte extent checked against that contract.
  size_t length;
  // One u32 must be an intentional body-error code rather than a sentinel.
  bool body_error;
};

struct Binding {
  // Source-level binding name for an exact failure location.
  std::string name;
  // Original header/data input; other bindings begin with distinct sentinels.
  bool input = false;
  // Element width used by the original fixture's sentinel.
  size_t element_bytes = 0;
  // Independent full expected bytes, including untouched padding and guards.
  std::vector<uint8_t> expected;
  // Upload payload kept alive through accepted transfer completion.
  std::vector<uint8_t> initial;
  // Exact output ranges, or the invalid image's defined terminal failure.
  std::vector<Region> regions;
  // Owned persistent buffer with storage, transfer, and indirect usage.
  iree_hal_buffer_t* buffer = nullptr;
};

struct Compilation {
  // Publishes the completed compiler result to the outer thread.
  std::mutex mutex;
  // Readiness notification; no timeout or worker-side blocking.
  std::condition_variable condition;
  // Callback has finished accessing source/configuration state.
  bool complete = false;
  // Owned compilation status transferred to the caller.
  iree_status_t status = iree_ok_status();
  // Owned compiled command stage transferred to the caller.
  loom_serve_jit_stage_t* stage = nullptr;

  static void Complete(void* pointer, iree_status_t status,
                       loom_serve_jit_stage_t* stage) {
    auto* self = static_cast<Compilation*>(pointer);
    std::lock_guard<std::mutex> lock(self->mutex);
    self->status = status;
    self->stage = stage;
    self->complete = true;
    self->condition.notify_one();
  }
};

class JpegCommandTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_GT(FLAG_jpeg_repetitions, 0);
    ASSERT_LE(FLAG_jpeg_repetitions, 128);
    IREE_ASSERT_OK(loom_serve_test_host_create(4, allocator_, &host_));
    iree_hal_driver_t* driver = nullptr;
    IREE_ASSERT_OK(loom_serve_test_driver_create(host_, allocator_, &driver));
    const loom_serve_device_options_t options = {
        .driver = driver,
        .uri = IREE_SV("amdgpu"),
        .backing = LOOM_SERVE_DEVICE_BACKING_FIXED};
    auto status = loom_serve_device_group_allocate(host_, 1, &options,
                                                   allocator_, &group_);
    iree_hal_driver_release(driver);
    IREE_ASSERT_OK(status);
    owner_ = loom_serve_device_group_at(group_, 0);
    IREE_ASSERT_OK(loom_serve_device_create_execution(owner_, &execution_));
    directory_ = FLAG_jpeg_fixture;
    const iree_string_view_t source = IREE_SV("summary.loombc");
    const loom_serve_sources_t sources = {
        .base_directory = iree_make_cstring_view(directory_.c_str()),
        .count = 1,
        .values = &source};
    const loomc_sanitizer_options_t sanitizer = {
        .type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
        .structure_size = sizeof(loomc_sanitizer_options_t),
        .checks = LOOMC_SANITIZER_CHECK_ACCESS};
    IREE_ASSERT_OK(loom_serve_jit_create(
        loom_serve_device_handle(owner_),
        loom_serve_device_dispatch_queue(owner_), host_, &sources,
        FLAG_jpeg_access ? &sanitizer : nullptr, &report_, allocator_, &jit_));
    std::ifstream config_file(std::filesystem::path(directory_) /
                              "config.json");
    ASSERT_TRUE(config_file.good());
    configuration_.assign(std::istreambuf_iterator<char>(config_file), {});
  }

  void TearDown() override {
    if (execution_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(execution_));
    }
    iree_hal_command_buffer_release(command_);
    for (auto& binding : bindings_) {
      iree_hal_buffer_release(binding.buffer);
    }
    iree_hal_buffer_release(transient_);
    loom_serve_jit_release(jit_);
    loom_serve_execution_release(execution_);
    if (owner_) {
      IREE_EXPECT_OK(loom_serve_test_device_trim_and_drain(owner_, 0));
    }
    IREE_EXPECT_OK(loom_serve_device_group_free(group_));
    loom_serve_host_release(host_);
  }

  void Prepare() {
    std::ifstream manifest(std::filesystem::path(directory_) / "bindings.txt");
    ASSERT_TRUE(manifest.good());
    std::string name, relative;
    unsigned input = 0;
    size_t width = 0, count = 0;
    while (manifest >> name >> input >> width >> count >> relative) {
      ASSERT_TRUE(width == 1 || width == 4);
      Binding binding;
      binding.name = name;
      binding.input = input != 0;
      binding.element_bytes = width;
      std::ifstream file(std::filesystem::path(directory_) / relative,
                         std::ios::binary);
      ASSERT_TRUE(file.good()) << relative;
      uint8_t prefix[10] = {};
      file.read(reinterpret_cast<char*>(prefix), sizeof(prefix));
      ASSERT_EQ(std::memcmp(prefix, "\x93NUMPY\x01\x00", 8), 0);
      size_t header_bytes = size_t(prefix[8]) | (size_t(prefix[9]) << 8);
      file.seekg(static_cast<std::streamoff>(10 + header_bytes));
      binding.expected.resize(count * width);
      file.read(reinterpret_cast<char*>(binding.expected.data()),
                binding.expected.size());
      ASSERT_EQ(size_t(file.gcount()), binding.expected.size()) << relative;
      ASSERT_EQ(file.peek(), std::char_traits<char>::eof());
      if (FLAG_jpeg_initial[0]) {
        std::ifstream initial(std::filesystem::path(directory_) /
                                  FLAG_jpeg_initial / (binding.name + ".npy"),
                              std::ios::binary);
        ASSERT_TRUE(initial.good()) << binding.name;
        uint8_t initial_prefix[10] = {};
        initial.read(reinterpret_cast<char*>(initial_prefix),
                     sizeof(initial_prefix));
        ASSERT_EQ(std::memcmp(initial_prefix, "\x93NUMPY\x01\x00", 8), 0);
        const size_t initial_header_bytes =
            size_t(initial_prefix[8]) | (size_t(initial_prefix[9]) << 8);
        initial.seekg(static_cast<std::streamoff>(10 + initial_header_bytes));
        binding.initial.resize(binding.expected.size());
        initial.read(reinterpret_cast<char*>(binding.initial.data()),
                     binding.initial.size());
        ASSERT_EQ(size_t(initial.gcount()), binding.initial.size())
            << binding.name;
        ASSERT_EQ(initial.peek(), std::char_traits<char>::eof());
      } else if (binding.input) {
        binding.initial = binding.expected;
      } else {
        binding.initial.resize(binding.expected.size(), 0);
        for (size_t byte = 0; byte < binding.initial.size(); byte += width) {
          binding.initial[byte] = 17;
        }
      }
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          loom_serve_device_storage_pool(owner_),
          iree_hal_make_pool_reservation_request(binding.expected.size(), 16),
          iree_infinite_timeout(), &binding.buffer));
      bindings_.push_back(std::move(binding));
    }
    ASSERT_TRUE(manifest.eof());
    ASSERT_FALSE(bindings_.empty());
    if (FLAG_jpeg_regions[0]) {
      std::ifstream regions(std::filesystem::path(directory_) /
                            FLAG_jpeg_regions);
      ASSERT_TRUE(regions.good());
      std::string mode;
      size_t offset = 0, length = 0;
      while (regions >> name >> offset >> length >> mode) {
        auto binding = std::find_if(
            bindings_.begin(), bindings_.end(),
            [&](const Binding& value) { return value.name == name; });
        ASSERT_NE(binding, bindings_.end()) << name;
        ASSERT_GT(length, 0u);
        ASSERT_LE(offset, binding->expected.size());
        ASSERT_LE(length, binding->expected.size() - offset);
        ASSERT_TRUE(mode == "exact" || mode == "body_error");
        if (mode == "body_error") {
          ASSERT_EQ(name, "validity");
          ASSERT_EQ(offset % 4, 0u);
          ASSERT_EQ(length, 4u);
        }
        binding->regions.push_back({offset, length, mode == "body_error"});
      }
      ASSERT_TRUE(regions.eof());
      for (const auto& binding : bindings_) {
        ASSERT_FALSE(binding.regions.empty()) << binding.name;
      }
    } else {
      for (auto& binding : bindings_) {
        binding.regions.push_back({0, binding.expected.size(), false});
      }
    }
    std::vector<iree_hal_transfer_operation_t> uploads(bindings_.size());
    for (size_t i = 0; i < bindings_.size(); ++i) {
      auto& upload = uploads[i];
      upload.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
      upload.upload.source = bindings_[i].initial.data();
      upload.upload.target_buffer = bindings_[i].buffer;
      upload.upload.length = bindings_[i].initial.size();
    }
    uint64_t completion = 0;
    IREE_ASSERT_OK(loom_serve_execution_transfer(execution_, uploads.size(),
                                                 uploads.data(), &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution_, completion));
  }

  void Compile(const char* name, bool retain_profile_metadata = false) {
    iree_hal_command_buffer_release(command_);
    command_ = nullptr;
    iree_hal_buffer_release(transient_);
    transient_ = nullptr;
    const loomc_config_options_t config = {
        .json_object = loomc_make_cstring_view(configuration_.c_str())};
    Compilation compilation;
    IREE_ASSERT_OK(loom_serve_jit_compile_async(
        jit_, iree_make_cstring_view(name), &config, Compilation::Complete,
        &compilation));
    {
      std::unique_lock<std::mutex> lock(compilation.mutex);
      compilation.condition.wait(lock, [&] { return compilation.complete; });
    }
    IREE_ASSERT_OK(compilation.status);
    const auto* program = loom_serve_jit_stage_program(compilation.stage);
    const uint32_t dispatch_count = program->commands.count;
    size_t scratch_bytes = program->requirements.transient.required_byte_length;
    std::printf(
        "{\"phase\":\"command\",\"root\":\"%s\",\"scratch_bytes\":%zu,"
        "\"commands\":%u,\"entries\":%u,\"kinds\":[",
        name, scratch_bytes, program->commands.count,
        program->requirements.entry_count);
    for (uint32_t i = 0; i < program->commands.count; ++i) {
      std::printf("%s%u", i ? "," : "",
                  unsigned(loom_cmd_program_command_at(program, i).kind));
    }
    std::printf("]}\n");
    table_.clear();
    for (auto& binding : bindings_) {
      table_.push_back({binding.buffer, 0, binding.expected.size()});
    }
    ASSERT_EQ(program->requirements.rebindable_binding_count,
              table_.size() + (scratch_bytes ? 1u : 0u));
    if (scratch_bytes) {
      ASSERT_EQ(program->requirements.transient.binding_index, table_.size());
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          loom_serve_device_storage_pool(owner_),
          iree_hal_make_pool_reservation_request(
              scratch_bytes, program->requirements.transient.minimum_alignment),
          iree_infinite_timeout(), &transient_));
      table_.push_back({transient_, 0, scratch_bytes});
    }
    auto status = loom_serve_jit_stage_record(
        compilation.stage,
        iree_hal_queue_family(loom_serve_device_dispatch_queue(owner_)),
        retain_profile_metadata
            ? IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA
            : IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        nullptr, nullptr, &command_);
    loom_serve_jit_stage_free(compilation.stage);
    IREE_ASSERT_OK(status);
    if (std::strcmp(name, "jpeg_decode") == 0) {
      decode_dispatch_count_ = dispatch_count;
    } else {
      ASSERT_GT(decode_dispatch_count_, 0u);
      ASSERT_EQ(dispatch_count, decode_dispatch_count_ * FLAG_jpeg_repetitions);
    }
  }

  void Memory() {
    uint64_t logical = 0, reserved = 0;
    std::vector<const iree_hal_buffer_t*> allocations;
    auto account = [&](iree_hal_buffer_t* buffer) {
      if (!buffer) {
        return;
      }
      logical += iree_hal_buffer_byte_length(buffer);
      auto* allocation = iree_hal_buffer_allocated_buffer(buffer);
      if (std::find(allocations.begin(), allocations.end(), allocation) ==
          allocations.end()) {
        allocations.push_back(allocation);
        reserved += iree_hal_buffer_allocation_size(allocation);
      }
    };
    for (auto& binding : bindings_) {
      account(binding.buffer);
    }
    account(transient_);
    std::printf(
        "{\"phase\":\"memory\",\"logical_buffer_bytes\":%llu,\"unique_backing_"
        "bytes\":%llu,\"backings\":%zu",
        static_cast<unsigned long long>(logical),
        static_cast<unsigned long long>(reserved), allocations.size());
#if IREE_STATISTICS_ENABLE
    iree_hal_allocator_statistics_t statistics = {};
    iree_hal_allocator_query_statistics(
        iree_hal_device_allocator(loom_serve_device_handle(owner_)),
        &statistics);
    std::printf(
        ",\"device_peak\":%llu,\"device_live\":%llu,\"host_peak\":%llu,\"host_"
        "live\":%llu",
        static_cast<unsigned long long>(statistics.device_bytes_peak),
        static_cast<unsigned long long>(statistics.device_bytes_allocated -
                                        statistics.device_bytes_freed),
        static_cast<unsigned long long>(statistics.host_bytes_peak),
        static_cast<unsigned long long>(statistics.host_bytes_allocated -
                                        statistics.host_bytes_freed));
#endif
    std::printf("}\n");
  }

  void Execute() {
    uint64_t completion = 0;
    IREE_ASSERT_OK(loom_serve_execution_execute(
        execution_, command_, {table_.size(), table_.data()}, &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution_, completion));
  }

  void Verify() {
    std::vector<uint8_t> actual;
    for (const auto& binding : bindings_) {
      actual.resize(binding.expected.size());
      iree_hal_transfer_operation_t transfer = {};
      transfer.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
      transfer.download.source_buffer = binding.buffer;
      transfer.download.target = actual.data();
      transfer.download.length = actual.size();
      uint64_t completion = 0;
      IREE_ASSERT_OK(
          loom_serve_execution_feedback(execution_, 1, &transfer, &completion));
      IREE_ASSERT_OK(
          loom_serve_execution_feedback_wait(execution_, completion));
      if (FLAG_jpeg_capture[0]) {
        std::ofstream capture(
            std::filesystem::path(FLAG_jpeg_capture) / (binding.name + ".bin"),
            std::ios::binary);
        ASSERT_TRUE(capture.good()) << binding.name;
        capture.write(reinterpret_cast<const char*>(actual.data()),
                      actual.size());
        capture.close();
        ASSERT_TRUE(capture.good()) << binding.name;
      }
      size_t checked_bytes = 0;
      for (const auto& region : binding.regions) {
        if (region.body_error) {
          uint32_t value = 0;
          std::memcpy(&value, actual.data() + region.offset, sizeof(value));
          ASSERT_TRUE(value == 256 || value == 512 || value == 768 ||
                      value == 1024 || value == 1280)
              << binding.name << " byte=" << region.offset
              << " value=" << value;
          std::printf("{\"phase\":\"body_error\",\"image\":%zu,\"value\":%u}\n",
                      region.offset / 4, value);
        } else {
          auto end = actual.begin() + region.offset + region.length;
          auto mismatch =
              std::mismatch(actual.begin() + region.offset, end,
                            binding.expected.begin() + region.offset);
          ASSERT_EQ(mismatch.first, end)
              << binding.name << " byte=" << (mismatch.first - actual.begin())
              << " actual=" << unsigned(*mismatch.first)
              << " expected=" << unsigned(*mismatch.second);
        }
        checked_bytes += region.length;
      }
      std::printf(
          "{\"phase\":\"%s\",\"binding\":\"%s\",\"bytes\":%zu,\"checked_"
          "bytes\":%zu}\n",
          FLAG_jpeg_regions[0] ? "regions" : "exact", binding.name.c_str(),
          actual.size(), checked_bytes);
    }
  }

  void Benchmark() {
    ASSERT_FALSE(FLAG_jpeg_access);
    for (size_t i = 0; i < 2; ++i) {
      Execute();
      ASSERT_FALSE(HasFatalFailure());
    }
    std::vector<int64_t> durations;
    for (size_t i = 0; i < 9; ++i) {
      const auto begin = std::chrono::steady_clock::now();
      Execute();
      const auto end = std::chrono::steady_clock::now();
      ASSERT_FALSE(HasFatalFailure());
      durations.push_back(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
              .count());
    }
    std::printf(
        "{\"phase\":\"measurement\",\"operations_per_submission\":%d,\"batch_"
        "ns\":[",
        FLAG_jpeg_repetitions);
    for (size_t i = 0; i < durations.size(); ++i) {
      std::printf("%s%lld", i ? "," : "", static_cast<long long>(durations[i]));
    }
    std::sort(durations.begin(), durations.end());
    std::printf("],\"median_operation_ns\":%lld}\n",
                static_cast<long long>(durations[durations.size() / 2] /
                                       FLAG_jpeg_repetitions));
  }

  void Profile() {
    ASSERT_FALSE(FLAG_jpeg_access);
    Compile("jpeg_batch", true);
    ASSERT_FALSE(HasFatalFailure());
    for (size_t i = 0; i < 2; ++i) {
      Execute();
      ASSERT_FALSE(HasFatalFailure());
    }
    iree_hal_profile_statistics_sink_t* sink = nullptr;
    IREE_ASSERT_OK(iree_hal_profile_statistics_sink_create(allocator_, &sink));
    const iree_hal_device_profiling_options_t options = {
        .data_families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                         IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA,
        .sink = iree_hal_profile_statistics_sink_base(sink)};
    auto* device = loom_serve_device_handle(owner_);
    auto status = iree_hal_device_profiling_begin(device, &options);
    if (iree_status_is_ok(status)) {
      Execute();
      status = iree_hal_device_profiling_end(device);
    }
    if (iree_status_is_ok(status) && !HasFatalFailure()) {
      EXPECT_GT(iree_hal_profile_statistics_sink_row_count(sink), 0);
      EXPECT_EQ(iree_hal_profile_statistics_sink_dropped_record_count(sink), 0);
      status = iree_hal_profile_statistics_sink_for_each_row(
          sink,
          {.fn =
               [](void* pointer, const iree_hal_profile_statistics_row_t* row) {
                 auto* statistics =
                     static_cast<iree_hal_profile_statistics_sink_t*>(pointer);
                 if (row->row_type !=
                     IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_FUNCTION) {
                   return iree_ok_status();
                 }
                 iree_string_view_t name = iree_string_view_empty();
                 EXPECT_TRUE(
                     iree_hal_profile_statistics_sink_find_function_name(
                         statistics, row->executable_id, row->function_ordinal,
                         &name));
                 uint64_t total = 0;
                 EXPECT_TRUE(
                     iree_hal_profile_statistics_sink_scale_duration_to_ns(
                         statistics, row, row->total_duration, &total));
                 EXPECT_EQ(row->invalid_sample_count, 0);
                 std::printf(
                     "{\"phase\":\"profile\",\"function\":\"%.*s\",\"samples\":"
                     "%llu,\"total_ns\":%llu}\n",
                     static_cast<int>(name.size), name.data,
                     static_cast<unsigned long long>(row->sample_count),
                     static_cast<unsigned long long>(total));
                 return iree_ok_status();
               },
           .user_data = sink});
    }
    iree_hal_profile_statistics_sink_release(sink);
    IREE_ASSERT_OK(status);
  }

  // Host policy and owned resources; teardown joins accepted work first.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Shared host services through device retirement.
  loom_serve_host_t* host_ = nullptr;
  // Owned application device group.
  loom_serve_device_group_t* group_ = nullptr;
  // Borrowed device owner in group.
  loom_serve_device_t* owner_ = nullptr;
  // Owned ordered execution timeline.
  loom_serve_execution_t* execution_ = nullptr;
  // Owned source catalog and compiled native cache.
  loom_serve_jit_t* jit_ = nullptr;
  // Compiler evidence through all callback completions.
  loom_serve_compilation_report_t report_ = {};
  // Owned command buffer for the current root.
  iree_hal_command_buffer_t* command_ = nullptr;
  // Owned transient count storage for the current command.
  iree_hal_buffer_t* transient_ = nullptr;
  // Reflected single-decode dispatch count checks the authored batch
  // multiplier.
  uint32_t decode_dispatch_count_ = 0;
  // Fixture location and stable specialization JSON.
  std::string directory_;
  // Configuration outliving asynchronous compilation.
  std::string configuration_;
  // Input/output owners through all uploads and downloads.
  std::vector<Binding> bindings_;
  // Issue-time binding views; the queue captures them synchronously.
  std::vector<iree_hal_buffer_binding_t> table_;
};

TEST_F(JpegCommandTest, CompressedToRgb) {
  Prepare();
  ASSERT_FALSE(HasFatalFailure());
  Compile("jpeg_decode");
  ASSERT_FALSE(HasFatalFailure());
  Execute();
  ASSERT_FALSE(HasFatalFailure());
  Memory();
  Verify();
  ASSERT_FALSE(HasFatalFailure());
  if (FLAG_jpeg_benchmark || FLAG_jpeg_profile) {
    Compile("jpeg_batch");
    ASSERT_FALSE(HasFatalFailure());
    if (FLAG_jpeg_benchmark) {
      Benchmark();
      ASSERT_FALSE(HasFatalFailure());
    }
    if (FLAG_jpeg_profile) {
      Profile();
      ASSERT_FALSE(HasFatalFailure());
    }
    Memory();
    Verify();
  }
}

}  // namespace
