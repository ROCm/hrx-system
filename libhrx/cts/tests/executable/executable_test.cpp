// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <string>

#include "hrx_test_fixture.hpp"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

namespace {

// Kernels defined by libhrx/cts/tests/executable/executable_kernels.c, the
// source compiled into the HSACO this test loads. Nothing else in that source
// becomes an export, so these names are the loaded executable's whole export
// table.
constexpr const char* kFixtureKernelNames[] = {
    "hrx_noop",
    "hrx_read_repeated",
    "hrx_store_output",
    "hrx_transform_padded_parameters",
    "hrx_store_exact_workitem_indices",
    "hrx_report_dispatch_size",
    "hrx_gated_store_output",
    "hrx_transform_nested_pointers",
};

std::string gpu_architecture(hrx_device_t device) {
  std::array<char, 64> arch = {};
  REQUIRE_OK(hrx().device_get_property(device, HRX_DEVICE_PROPERTY_ARCHITECTURE,
                                       arch.data(), arch.size()));
  return std::string(arch.data());
}

class GraphVisibilityTest : public HrxTestFixture {
 protected:
  void Initialize() {
    if (!is_gpu()) {
      SKIP("Native executable dispatch is GPU-only");
    }
    const auto hsaco =
        hrx_cts::FindAmdgpuExecutableTestImage(gpu_architecture(device_));
    if (!hsaco.file) {
      SKIP("No build-time HSACO test asset for this GPU");
    }
    REQUIRE_OK(hrx().executable_load_data(
        device_, hsaco.file->data, hsaco.file->size, "amdgpu",
        hsaco.target_key.c_str(), &executable_));
    REQUIRE_OK(hrx().executable_lookup_export_by_name(
        executable_, "hrx_transform_padded_parameters", &ordinal_));
    REQUIRE_OK(hrx().executable_export_info(executable_, ordinal_, &info_));
    REQUIRE_OK(hrx().stream_create(
        FindProvisionedQueue(device_,
                             HRX_QUEUE_ROLE_DISPATCH | HRX_QUEUE_ROLE_TRANSFER),
        0, &stream_));
    const hrx_buffer_params_t params = {
        .type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_HOST_COHERENT |
                HRX_MEMORY_TYPE_DEVICE_VISIBLE,
        .access = HRX_MEMORY_ACCESS_ALL,
        .usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
    };
    REQUIRE_OK(hrx().allocator_allocate_buffer(
        hrx().device_allocator(device_), params, kWordCount * sizeof(uint32_t),
        &buffer_));
    void* mapped = nullptr;
    REQUIRE_OK(hrx().buffer_map(buffer_, HRX_MAP_READ | HRX_MAP_WRITE, 0,
                                kWordCount * sizeof(uint32_t), &mapped));
    words_ = static_cast<uint32_t*>(mapped);
    callback_.words = words_;
    REQUIRE_OK(hrx().graph_create(device_, 0, &graph_));
  }

  ~GraphVisibilityTest() {
    if (gate_) {
      ReportHrxTestFailure("Release pending graph work",
                           hrx().semaphore_signal(gate_, 1));
    }
    if (other_stream_) {
      ReportHrxTestFailure("Other stream graph completion",
                           hrx().stream_synchronize(other_stream_));
    }
    if (stream_) {
      ReportHrxTestFailure("Graph completion",
                           hrx().stream_synchronize(stream_));
    }
    hrx().graph_exec_release(exec_);
    hrx().graph_release(graph_);
    if (words_) {
      ReportHrxTestFailure("Buffer unmap", hrx().buffer_unmap(buffer_));
    }
    hrx().buffer_release(buffer_);
    hrx().stream_release(stream_);
    hrx().stream_release(other_stream_);
    hrx().executable_release(executable_);
    hrx().semaphore_release(gate_);
  }

  void CreateOtherStream() {
    const uint32_t family_ordinal =
        hrx().queue_get_family_ordinal(hrx().stream_get_queue(stream_));
    hrx_queue_family_info_t family;
    REQUIRE_OK(
        hrx().device_query_queue_family(device_, family_ordinal, &family));
    if (!family.supports_acquisition) {
      SKIP("compute queue family does not support acquiring another queue");
    }
    hrx_queue_t queue = nullptr;
    REQUIRE_OK(hrx().queue_acquire(device_, family_ordinal, nullptr, &queue));
    hrx_status_t status = hrx().stream_create(queue, 0, &other_stream_);
    hrx().queue_release(queue);
    REQUIRE_OK(status);
  }

  hrx_graph_node_t AddTransform(size_t input_word, size_t output_word,
                                hrx_graph_node_t dependency = nullptr) {
    // HAL constants are densely packed independently of native ABI padding.
    const uint16_t bias = 7;
    const uint32_t scale = 3;
    std::array<uint8_t, sizeof(bias) + sizeof(scale)> constants;
    std::memcpy(constants.data(), &bias, sizeof(bias));
    std::memcpy(constants.data() + sizeof(bias), &scale, sizeof(scale));
    const hrx_buffer_ref_t bindings[] = {
        {buffer_, input_word * sizeof(uint32_t), sizeof(uint32_t)},
        {buffer_, output_word * sizeof(uint32_t), sizeof(uint32_t)},
    };
    const hrx_graph_kernel_node_attrs_t attrs = {
        executable_,
        ordinal_,
        {{1, 1, 1},
         {info_.workgroup_size[0], info_.workgroup_size[1],
          info_.workgroup_size[2]},
         0},
        constants.data(),
        constants.size(),
        bindings,
        std::size(bindings),
        HRX_DISPATCH_FLAG_NONE,
    };
    hrx_graph_node_t node = nullptr;
    REQUIRE_OK(hrx().graph_add_kernel_node(graph_,
                                           dependency ? &dependency : nullptr,
                                           dependency ? 1 : 0, &attrs, &node));
    return node;
  }

  static constexpr size_t kWordCount = 256;
  // Code object shared by every node in this fixture's graph.
  hrx_executable_t executable_ = nullptr;
  // Export computing output = input * 3 + 7.
  uint32_t ordinal_ = 0;
  // Reflected workgroup dimensions for the loaded export.
  hrx_executable_export_info_t info_ = {};
  // Timeline joined before inspecting or releasing mapped storage.
  hrx_stream_t stream_ = nullptr;
  // Optional independent stream exercising executable reuse across timelines.
  hrx_stream_t other_stream_ = nullptr;
  // Optional host-controlled dependency, released before teardown joins work.
  hrx_semaphore_t gate_ = nullptr;
  // Coherent host allocation retained through all replays.
  hrx_buffer_t buffer_ = nullptr;
  // Persistent mapping, kept live until after the last host observation.
  uint32_t* words_ = nullptr;
  // Recorded graph owning its captured arguments.
  hrx_graph_t graph_ = nullptr;
  // Reusable executable released before the graph and its buffers.
  hrx_graph_exec_t exec_ = nullptr;
  // Callback state outliving submitted work even when an assertion fails.
  struct {
    // Persistent input/output mapping, owned by the test fixture.
    uint32_t* words = nullptr;
    // Value read before any subsequent GPU work can publish it.
    uint32_t observed = 0;
    // Number of callback invocations, checked after stream completion.
    uint32_t calls = 0;
  } callback_;
};

}  // namespace

TEST_CASE_METHOD(HrxTestFixture, "executable_load_lookup_dispatch") {
  if (!is_gpu()) {
    SUCCEED("Native executable dispatch is GPU-only");
    return;
  }

  std::string arch = gpu_architecture(device_);
  if (arch.empty() || arch == "unknown") {
    SUCCEED("Skipping native executable CTS: unknown GPU architecture");
    return;
  }

  hrx_cts::AmdgpuExecutableTestImage hsaco =
      hrx_cts::FindAmdgpuExecutableTestImage(arch);
  if (!hsaco.file) {
    SUCCEED("Skipping native executable CTS: no build-time HSACO test asset");
    return;
  }

  hrx_executable_t executable = nullptr;
  REQUIRE_OK(hrx().executable_load_data(device_, hsaco.file->data,
                                        hsaco.file->size, "amdgpu",
                                        hsaco.target_key.c_str(), &executable));
  REQUIRE(executable != nullptr);

  hrx().executable_retain(executable);
  hrx().executable_release(executable);

  size_t export_count = 0;
  REQUIRE_OK(hrx().executable_export_count(executable, &export_count));
  REQUIRE(export_count == std::size(kFixtureKernelNames));

  // A count says nothing about which kernels it counted. Resolving every
  // fixture name to an ordinal of its own pairs with it to say the export table
  // is exactly this set.
  std::array<bool, std::size(kFixtureKernelNames)> ordinal_seen = {};
  for (const char* kernel_name : kFixtureKernelNames) {
    INFO("kernel " << kernel_name);
    uint32_t ordinal = UINT32_MAX;
    REQUIRE_OK(hrx().executable_lookup_export_by_name(executable, kernel_name,
                                                      &ordinal));
    REQUIRE(ordinal < export_count);
    REQUIRE(!ordinal_seen[ordinal]);
    ordinal_seen[ordinal] = true;
  }

  uint32_t noop_ordinal = UINT32_MAX;
  REQUIRE_OK(hrx().executable_lookup_export_by_name(executable, "hrx_noop",
                                                    &noop_ordinal));
  REQUIRE(noop_ordinal < export_count);

  uint32_t store_ordinal = UINT32_MAX;
  REQUIRE_OK(hrx().executable_lookup_export_by_name(
      executable, "hrx_store_output", &store_ordinal));
  REQUIRE(store_ordinal < export_count);
  REQUIRE(store_ordinal != noop_ordinal);

  hrx_executable_export_info_t noop_info = {};
  REQUIRE_OK(
      hrx().executable_export_info(executable, noop_ordinal, &noop_info));
  REQUIRE(noop_info.name != nullptr);
  REQUIRE(std::string(noop_info.name) == "hrx_noop");
  REQUIRE(noop_info.constant_byte_length == 0);
  REQUIRE(noop_info.binding_count == 0);
  REQUIRE(noop_info.workgroup_size[0] >= 1);
  REQUIRE(noop_info.workgroup_size[1] >= 1);
  REQUIRE(noop_info.workgroup_size[2] >= 1);

  hrx_executable_export_info_t store_info = {};
  REQUIRE_OK(
      hrx().executable_export_info(executable, store_ordinal, &store_info));
  REQUIRE(store_info.name != nullptr);
  REQUIRE(std::string(store_info.name) == "hrx_store_output");
  REQUIRE(store_info.constant_byte_length == sizeof(uint32_t));
  REQUIRE(store_info.binding_count == 1);
  REQUIRE(store_info.workgroup_size[0] >= 1);
  REQUIRE(store_info.workgroup_size[1] >= 1);
  REQUIRE(store_info.workgroup_size[2] >= 1);

  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(
      FindProvisionedQueue(device_,
                           HRX_QUEUE_ROLE_DISPATCH | HRX_QUEUE_ROLE_TRANSFER),
      0, &stream));
  REQUIRE(stream != nullptr);

  hrx_dispatch_config_t noop_config = {
      /* .workgroup_count = */ {1, 1, 1},
      /* .workgroup_size = */
      {
          noop_info.workgroup_size[0],
          noop_info.workgroup_size[1],
          noop_info.workgroup_size[2],
      },
      /* .subgroup_size = */ 0,
  };
  REQUIRE_OK(hrx().stream_dispatch(stream, executable, noop_ordinal,
                                   &noop_config,
                                   /*constants=*/nullptr, /*constants_size=*/0,
                                   /*bindings=*/nullptr, /*binding_count=*/0,
                                   HRX_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS));

  hrx_buffer_t output = nullptr;
  REQUIRE_OK(hrx().buffer_allocate(stream, sizeof(uint32_t),
                                   HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                   HRX_BUFFER_USAGE_DEFAULT, &output));
  REQUIRE(output != nullptr);

  const uint32_t expected = 0xFEED1234u;
  const hrx_buffer_ref_t bindings[] = {{
      /* .buffer = */ output,
      /* .offset = */ 0,
      /* .length = */ sizeof(uint32_t),
  }};
  hrx_dispatch_config_t store_config = {
      /* .workgroup_count = */ {1, 1, 1},
      /* .workgroup_size = */
      {
          store_info.workgroup_size[0],
          store_info.workgroup_size[1],
          store_info.workgroup_size[2],
      },
      /* .subgroup_size = */ 0,
  };
  REQUIRE_OK(hrx().stream_dispatch(
      stream, executable, store_ordinal, &store_config, &expected,
      sizeof(expected), bindings, std::size(bindings), HRX_DISPATCH_FLAG_NONE));
  REQUIRE_OK(hrx().stream_synchronize(stream));

  uint32_t actual = 0;
  REQUIRE_OK(hrx().synchronous_d2h(transfer_queue_, output, 0, &actual,
                                   sizeof(actual)));
  REQUIRE(actual == expected);

  // The direct API targets an exact queue without joining the stream timeline.
  // The completed stream work above supplies the initial dependency boundary.
  hrx_queue_t queue = FindProvisionedQueue(device_, HRX_QUEUE_ROLE_DISPATCH);
  hrx_semaphore_t completion = nullptr;
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &completion));
  uint64_t completion_value = 1;
  const hrx_semaphore_list_t signals = {&completion, &completion_value, 1};
  const uint32_t direct_expected = 0xABCD4321u;
  const hrx_barrier_list_t no_payload_acquire = {};
  const hrx_memory_barrier_t store_access = {
      .source_scope = HRX_ACCESS_SCOPE_DISPATCH_WRITE,
      .target_scope = HRX_ACCESS_SCOPE_MEMORY_READ,
  };
  const hrx_barrier_t release = {
      .source_stage_mask = HRX_EXECUTION_STAGE_DISPATCH,
      .flags = HRX_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
      .memory_barrier_count = 1,
      .memory_barriers = &store_access,
  };
  const hrx_barrier_list_t after = {1, &release};
  const hrx_queue_barriers_t barriers = {&no_payload_acquire, &after};
  REQUIRE_OK(hrx().queue_dispatch(
      queue, nullptr, &signals, executable, store_ordinal, &store_config,
      &direct_expected, sizeof(direct_expected), bindings, std::size(bindings),
      &barriers, HRX_DISPATCH_FLAG_NONE));
  REQUIRE_OK(hrx().semaphore_wait(completion, completion_value, UINT64_MAX));
  REQUIRE_OK(hrx().synchronous_d2h(transfer_queue_, output, 0, &actual,
                                   sizeof(actual)));
  REQUIRE(actual == direct_expected);

  // Invalid public bindings must unwind both inline and overflow translation
  // storage before native acceptance, without advancing the caller's timeline.
  ++completion_value;
  for (size_t binding_count : {size_t{1}, size_t{16}, size_t{17}}) {
    CAPTURE(binding_count);
    std::array<hrx_buffer_ref_t, 17> invalid_bindings;
    invalid_bindings.fill(bindings[0]);
    invalid_bindings[binding_count - 1].buffer = nullptr;
    hrx_status_t status = hrx().queue_dispatch(
        queue, nullptr, &signals, executable, store_ordinal, &store_config,
        &direct_expected, sizeof(direct_expected), invalid_bindings.data(),
        binding_count, nullptr, HRX_DISPATCH_FLAG_NONE);
    const hrx_status_code_t code = hrx().status_code(status);
    hrx().status_ignore(status);
    if (code == HRX_STATUS_OK) {
      // Retire unexpectedly accepted work before reporting the failure.
      REQUIRE_OK(
          hrx().semaphore_wait(completion, completion_value, UINT64_MAX));
    }
    CHECK(code == HRX_STATUS_INVALID_ARGUMENT);
    uint64_t current_value = 0;
    REQUIRE_OK(hrx().semaphore_query(completion, &current_value));
    CHECK(current_value == 1);
  }
  hrx().semaphore_release(completion);

  hrx().buffer_release(output);
  hrx().stream_release(stream);
  hrx().executable_release(executable);
}

TEST_CASE_METHOD(GraphVisibilityTest,
                 "graph_kernel_acquires_coherent_host_input") {
  Initialize();
  const size_t input_word = GENERATE(size_t{0}, size_t{64});
  const size_t output_word = input_word + 128;
  AddTransform(input_word, output_word);
  REQUIRE_OK(hrx().graph_instantiate(graph_, hrx().stream_get_queue(stream_), 0,
                                     &exec_));
  for (uint32_t replay = 0; replay < 16; ++replay) {
    CAPTURE(input_word, replay);
    const uint32_t sentinel = 0x12345678u + replay;
    const uint32_t input = 0x100u + replay;
    std::fill_n(words_, kWordCount, sentinel);
    words_[input_word] = input;
    REQUIRE_OK(hrx().graph_exec_launch(exec_, stream_));
    REQUIRE_OK(hrx().stream_synchronize(stream_));
    for (size_t i = 0; i < kWordCount; ++i) {
      CAPTURE(i);
      const uint32_t expected = i == input_word    ? input
                                : i == output_word ? input * 3 + 7
                                                   : sentinel;
      REQUIRE(words_[i] == expected);
    }
  }
}

TEST_CASE_METHOD(GraphVisibilityTest,
                 "graph_replays_chain_coherent_data_without_host_waits") {
  Initialize();
  AddTransform(0, 0);
  REQUIRE_OK(hrx().graph_instantiate(graph_, hrx().stream_get_queue(stream_), 0,
                                     &exec_));
  for (uint32_t batch = 0; batch < 8; ++batch) {
    CAPTURE(batch);
    const uint32_t sentinel = 0x12345678u + batch;
    std::fill_n(words_, kWordCount, sentinel);
    uint32_t expected = 0x100u + batch;
    words_[0] = expected;
    for (uint32_t replay = 0; replay < 16; ++replay) {
      REQUIRE_OK(hrx().graph_exec_launch(exec_, stream_));
      expected = expected * 3 + 7;
    }
    REQUIRE_OK(hrx().stream_synchronize(stream_));
    REQUIRE(words_[0] == expected);
    for (size_t i = 1; i < kWordCount; ++i) {
      CAPTURE(i);
      REQUIRE(words_[i] == sentinel);
    }
  }
}

TEST_CASE_METHOD(GraphVisibilityTest,
                 "graph_orders_dependent_transfers_and_dispatches") {
  Initialize();
  constexpr uint32_t kInput = 0x102u;
  const hrx_graph_fill_buffer_node_attrs_t fill_attrs = {
      .dst = {buffer_, 0, sizeof(uint32_t)},
      .pattern = kInput,
      .pattern_size = sizeof(uint32_t),
  };
  hrx_graph_node_t fill = nullptr;
  REQUIRE_OK(
      hrx().graph_add_fill_buffer_node(graph_, nullptr, 0, &fill_attrs, &fill));
  const hrx_graph_copy_buffer_node_attrs_t input_copy_attrs = {
      .src = {buffer_, 0, sizeof(uint32_t)},
      .dst = {buffer_, 64 * sizeof(uint32_t), sizeof(uint32_t)},
  };
  hrx_graph_node_t input_copy = nullptr;
  REQUIRE_OK(hrx().graph_add_copy_buffer_node(graph_, &fill, 1,
                                              &input_copy_attrs, &input_copy));
  const auto first_dispatch = AddTransform(64, 128, input_copy);
  const auto second_dispatch = AddTransform(128, 192, first_dispatch);
  const hrx_graph_copy_buffer_node_attrs_t output_copy_attrs = {
      .src = {buffer_, 192 * sizeof(uint32_t), sizeof(uint32_t)},
      .dst = {buffer_, 224 * sizeof(uint32_t), sizeof(uint32_t)},
  };
  REQUIRE_OK(hrx().graph_add_copy_buffer_node(graph_, &second_dispatch, 1,
                                              &output_copy_attrs, nullptr));
  REQUIRE_OK(hrx().graph_instantiate(graph_, hrx().stream_get_queue(stream_), 0,
                                     &exec_));
  for (uint32_t replay = 0; replay < 16; ++replay) {
    const uint32_t sentinel = 0x12345678u + replay;
    std::fill_n(words_, kWordCount, sentinel);
    REQUIRE_OK(hrx().graph_exec_launch(exec_, stream_));
    REQUIRE_OK(hrx().stream_synchronize(stream_));
    for (size_t i = 0; i < kWordCount; ++i) {
      CAPTURE(replay, i);
      const uint32_t expected = i == 0 || i == 64 ? kInput
                                : i == 128        ? kInput * 3 + 7
                                : i == 192 || i == 224
                                    ? (kInput * 3 + 7) * 3 + 7
                                    : sentinel;
      REQUIRE(words_[i] == expected);
    }
  }
}

TEST_CASE_METHOD(GraphVisibilityTest,
                 "graph_parallel_replays_order_every_workstream") {
  Initialize();
  const size_t width = GENERATE(size_t{17}, size_t{64});
  const bool cross_stream = GENERATE(false, true);
  if (cross_stream) {
    CreateOtherStream();
  }
  std::array<uint32_t, kWordCount> expected;
  for (size_t i = 0; i < kWordCount; ++i) {
    words_[i] = expected[i] = static_cast<uint32_t>(i + 1);
  }
  for (size_t i = 0; i < width; ++i) {
    AddTransform(i, i);
  }
  REQUIRE_OK(hrx().graph_instantiate(graph_, hrx().stream_get_queue(stream_), 0,
                                     &exec_));
  for (size_t replay = 0; replay < 8; ++replay) {
    auto stream = cross_stream && (replay & 1) ? other_stream_ : stream_;
    REQUIRE_OK(hrx().graph_exec_launch(exec_, stream));
    for (size_t i = 0; i < width; ++i) {
      expected[i] = expected[i] * 3 + 7;
    }
  }
  // The last launch includes every branch of every earlier replay, even when
  // those replays came from another stream. No intermediate host wait is used.
  REQUIRE_OK(hrx().stream_synchronize(cross_stream ? other_stream_ : stream_));
  for (size_t i = 0; i < kWordCount; ++i) {
    CAPTURE(width, cross_stream, i);
    REQUIRE(words_[i] == expected[i]);
  }
}

TEST_CASE_METHOD(GraphVisibilityTest,
                 "graph_update_preserves_pending_dispatches") {
  Initialize();
  const bool cross_stream = GENERATE(false, true);
  if (cross_stream) {
    CreateOtherStream();
  }
  std::fill_n(words_, kWordCount, 0u);
  words_[0] = 13;
  AddTransform(0, 64);
  REQUIRE_OK(hrx().graph_instantiate(graph_, hrx().stream_get_queue(stream_), 0,
                                     &exec_));
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &gate_));
  REQUIRE_OK(hrx().stream_wait_on(stream_, {gate_, 1}));
  REQUIRE_OK(hrx().graph_exec_launch(exec_, stream_));

  hrx().graph_release(graph_);
  graph_ = nullptr;
  REQUIRE_OK(hrx().graph_create(device_, 0, &graph_));
  AddTransform(64, 128);
  REQUIRE_OK(hrx().graph_exec_update(exec_, graph_));
  auto consumer = cross_stream ? other_stream_ : stream_;
  REQUIRE_OK(hrx().graph_exec_launch(exec_, consumer));

  // Only accepted native command buffers retain code and recorded arguments
  // before releasing the producer. The output allocation remains caller-owned.
  hrx().graph_exec_release(exec_);
  exec_ = nullptr;
  hrx().graph_release(graph_);
  graph_ = nullptr;
  hrx().executable_release(executable_);
  executable_ = nullptr;
  REQUIRE_OK(hrx().semaphore_signal(gate_, 1));
  hrx().semaphore_release(gate_);
  gate_ = nullptr;
  REQUIRE_OK(hrx().stream_synchronize(consumer));
  CHECK(words_[64] == 13 * 3 + 7);
  CHECK(words_[128] == (13 * 3 + 7) * 3 + 7);
}

TEST_CASE_METHOD(GraphVisibilityTest,
                 "graph_host_callback_exchanges_coherent_data") {
  Initialize();
  // Reuse the same input address before and after the callback so the consumer
  // must discard the first dispatch's cached copy of the host-written input.
  const hrx_graph_node_t producer = AddTransform(0, 64);
  const hrx_graph_host_call_node_attrs_t host_attrs = {
      [](void* user_data) -> hrx_status_t {
        auto& state = *static_cast<decltype(callback_)*>(user_data);
        state.observed = state.words[64];
        state.words[0] = state.observed + 11;
        ++state.calls;
        return hrx_ok_status();
      },
      &callback_,
  };
  hrx_graph_node_t host_node = nullptr;
  REQUIRE_OK(hrx().graph_add_host_call_node(graph_, &producer, 1, &host_attrs,
                                            &host_node));
  AddTransform(0, 128, host_node);
  REQUIRE_OK(hrx().graph_instantiate(graph_, hrx().stream_get_queue(stream_), 0,
                                     &exec_));
  for (uint32_t replay = 0; replay < 16; ++replay) {
    CAPTURE(replay);
    const uint32_t sentinel = 0x12345678u + replay;
    const uint32_t input = 0x100u + replay;
    std::fill_n(words_, kWordCount, sentinel);
    words_[0] = input;
    REQUIRE_OK(hrx().graph_exec_launch(exec_, stream_));
    REQUIRE_OK(hrx().stream_synchronize(stream_));
    const uint32_t produced = input * 3 + 7;
    const uint32_t consumed = produced + 11;
    REQUIRE(callback_.observed == produced);
    REQUIRE(callback_.calls == replay + 1);
    for (size_t i = 0; i < kWordCount; ++i) {
      CAPTURE(i);
      const uint32_t expected = i == 0     ? consumed
                                : i == 64  ? produced
                                : i == 128 ? consumed * 3 + 7
                                           : sentinel;
      REQUIRE(words_[i] == expected);
    }
  }
}

TEST_CASE_METHOD(HrxTestFixture,
                 "graph_kernel_publishes_coherent_host_output") {
  if (!is_gpu()) {
    SKIP("Native executable dispatch is GPU-only");
  }
  const auto hsaco =
      hrx_cts::FindAmdgpuExecutableTestImage(gpu_architecture(device_));
  if (!hsaco.file) {
    SKIP("No build-time HSACO test asset for this GPU");
  }

  hrx_executable_t executable = nullptr;
  REQUIRE_OK(hrx().executable_load_data(device_, hsaco.file->data,
                                        hsaco.file->size, "amdgpu",
                                        hsaco.target_key.c_str(), &executable));
  uint32_t ordinal = 0;
  REQUIRE_OK(hrx().executable_lookup_export_by_name(
      executable, "hrx_store_output", &ordinal));
  hrx_executable_export_info_t info = {};
  REQUIRE_OK(hrx().executable_export_info(executable, ordinal, &info));
  hrx_stream_t stream = nullptr;
  REQUIRE_OK(hrx().stream_create(
      FindProvisionedQueue(device_,
                           HRX_QUEUE_ROLE_DISPATCH | HRX_QUEUE_ROLE_TRANSFER),
      0, &stream));

  constexpr size_t kWordCount = 256;
  const hrx_buffer_params_t params = {
      .type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_HOST_COHERENT |
              HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      .access = HRX_MEMORY_ACCESS_ALL,
      .usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED |
               HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
  };
  hrx_buffer_t output = nullptr;
  REQUIRE_OK(
      hrx().allocator_allocate_buffer(hrx().device_allocator(device_), params,
                                      kWordCount * sizeof(uint32_t), &output));
  void* mapped = nullptr;
  REQUIRE_OK(hrx().buffer_map(output, HRX_MAP_READ | HRX_MAP_WRITE, 0,
                              kWordCount * sizeof(uint32_t), &mapped));
  auto* words = static_cast<uint32_t*>(mapped);

  for (size_t word_offset : {size_t{0}, size_t{64}}) {
    CAPTURE(word_offset);
    const uint32_t expected = 0xFEED1234u;
    const hrx_buffer_ref_t binding = {output, word_offset * sizeof(uint32_t),
                                      sizeof(uint32_t)};
    const hrx_graph_kernel_node_attrs_t attrs = {
        executable,
        ordinal,
        {{1, 1, 1},
         {info.workgroup_size[0], info.workgroup_size[1],
          info.workgroup_size[2]},
         0},
        &expected,
        sizeof(expected),
        &binding,
        1,
        HRX_DISPATCH_FLAG_NONE,
    };
    hrx_graph_t graph = nullptr;
    REQUIRE_OK(hrx().graph_create(device_, 0, &graph));
    hrx_graph_node_t node = nullptr;
    REQUIRE_OK(hrx().graph_add_kernel_node(graph, nullptr, 0, &attrs, &node));
    hrx_graph_exec_t exec = nullptr;
    REQUIRE_OK(hrx().graph_instantiate(graph, hrx().stream_get_queue(stream), 0,
                                       &exec));

    for (uint32_t replay = 0; replay < 16; ++replay) {
      CAPTURE(replay);
      const uint32_t sentinel = 0x12345678u + replay;
      std::fill_n(words, kWordCount, sentinel);
      REQUIRE_OK(hrx().graph_exec_launch(exec, stream));
      REQUIRE_OK(hrx().stream_synchronize(stream));
      // Read before any transfer or graph teardown can publish cached writes.
      for (size_t i = 0; i < kWordCount; ++i) {
        CAPTURE(i);
        REQUIRE(words[i] == (i == word_offset ? expected : sentinel));
      }
    }
    hrx().graph_exec_release(exec);
    hrx().graph_release(graph);
  }

  REQUIRE_OK(hrx().buffer_unmap(output));
  hrx().buffer_release(output);
  hrx().stream_release(stream);
  hrx().executable_release(executable);
}
