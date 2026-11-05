// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "iree/hal/drivers/amd/fabric/cts/virtual_buffer_test_base.h"
#include "iree/hal/drivers/amd/xdna/context.h"
#include "iree/hal/drivers/amd/xdna/device.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"

namespace iree::hal::amd::cts {

using fabric::cts::FabricVirtualBufferTest;

enum class HostProjection { kBackingOnly, kStableReservation };

// Compose Task and XDNA HAL execution with native GPU SDMA. This suite checks
// shared backing and stable addresses independently of GPU HAL queue execution.
class FabricXdnaVirtualBufferTest : public FabricVirtualBufferTest {
 protected:
  void AddDevices(iree_hal_device_group_builder_t* builder) override {
    const amdf_endpoint_id_t* endpoint_id = nullptr;
    for (uint32_t i = 0; i < fabric_->endpoint_count; ++i) {
      if (fabric_->endpoints[i].engine_kind == AMDF_ENGINE_KIND_XDNA) {
        endpoint_id = &fabric_->endpoints[i].id;
        break;
      }
    }
    if (!endpoint_id) {
      GTEST_SKIP() << "no XDNA endpoint in the shared fabric";
    }
    iree_hal_amd_xdna_context_t* context = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_xdna_context_allocate(
        fabric_, endpoint_id, 0, nullptr, iree_hal_device_event_sink_stderr(),
        allocator_, &context));
    amdf_memory_reservation_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    const auto native_status =
        api_->device_query_memory_reservation_capabilities(
            context->device,
            context->data_source.capabilities.device_address
                .address_domain_ordinal,
            &capabilities);
    if (native_status != AMDF_STATUS_OK) {
      iree_hal_amd_xdna_context_free(context);
      if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        GTEST_SKIP() << "native XDNA address domain does not support VMM";
      }
      ASSERT_EQ(native_status, AMDF_STATUS_OK);
    }
    auto status = iree_hal_amd_xdna_device_create(
        context, IREE_SV("npu"), context_.params(), allocator_, &npu_);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_xdna_context_free(context);
    }
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(builder, npu_));
  }

  void TearDown() override {
    // The sealed group keeps the device alive until base teardown releases it,
    // before the shared progress services and native instance are destroyed.
    iree_hal_device_release(npu_);
    FabricVirtualBufferTest::TearDown();
  }

  void RunReplacement(HostProjection host_projection,
                      iree_hal_pool_placement_t placement = {});

  // Real HAL execution device sharing the GPU's native instance.
  iree_hal_device_t* npu_ = nullptr;
};

void FabricXdnaVirtualBufferTest::RunReplacement(
    HostProjection host_projection, iree_hal_pool_placement_t placement) {
  using iree::hal::cts::Ref;
  const bool host_alias = host_projection == HostProjection::kStableReservation;
  const auto* npu_family = iree_hal_device_queue_family(npu_, 0);
  const auto* cpu_family = iree_hal_device_queue_family(cpu_, 0);
  const uint64_t source_interface = UINT64_C(1)
                                    << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  // NPU is first in reservation order, while CPU is first in the sealed group.
  // Shared-group updates must resolve backing for the canonical native consumer
  // independently of the HAL scope used to name each replacement.
  std::array<iree_hal_pool_family_access_t, 3> families = {{
      {
          .family = npu_family,
          .usage =
              IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
          .interfaces = source_interface,
      },
      {
          .family = access_.family,
          .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
          .interfaces = source_interface,
      },
      {
          .family = cpu_family,
          .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
          .interfaces = source_interface,
      },
  }};
  iree_hal_pool_scope_t scope = {
      host_alias ? families.size() : families.size() - 1,
      families.data(),
      {IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_MAPPING_MODE_PERSISTENT,
       IREE_HAL_HOST_MEMORY_PROPERTY_NONE}};
  iree_hal_slab_pool_options_t pool_options;
  iree_hal_slab_pool_options_initialize(&pool_options);
  pool_options.placement = placement;
  Ref<iree_hal_pool_t> pool;
  auto pool_status = iree_hal_slab_pool_create(group_, scope, &pool_options,
                                               allocator_, pool.out());
  if (iree_status_is_unavailable(pool_status)) {
    const iree::Status unavailable(std::move(pool_status));
    GTEST_SKIP()
        << "no backing profile satisfies the requested mapping sources: "
        << iree::StatusCodeToString(unavailable.code());
  }
  IREE_ASSERT_OK(pool_status);
  if (placement.mode == IREE_HAL_POOL_PLACEMENT_REQUIRED) {
    iree_hal_pool_capabilities_t pool_capabilities;
    iree_hal_pool_query_capabilities(pool, &pool_capabilities);
    EXPECT_EQ(pool_capabilities.placement.mode, placement.mode);
    EXPECT_EQ(pool_capabilities.placement.node, placement.node);
  }
  const auto source_requirements =
      iree_hal_pool_query_mapping_requirements(pool);
  families[0].interfaces = UINT64_C(1)
                           << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA;
  families[1].interfaces = UINT64_C(1)
                           << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  families[2].interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_HOST;
  // Native shared permissions may be wider than this public host contract.
  scope.host =
      host_alias
          ? iree_hal_pool_host_access_t{IREE_HAL_MEMORY_ACCESS_READ,
                                        IREE_HAL_MAPPING_MODE_PERSISTENT,
                                        IREE_HAL_HOST_MEMORY_PROPERTY_NONE}
          : iree_hal_pool_host_access_t{};
  iree_hal_pool_t* source_pool = pool.get();
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, {1, &source_pool}, allocator_, &capabilities));
  ASSERT_NE(capabilities.max_byte_length, 0u);
  const auto unit = std::max(capabilities.reservation_granularity,
                             capabilities.mapping_granularity);
  const size_t stride = unit / sizeof(uint32_t);
  std::array<Ref<iree_hal_buffer_t>, 2> backings;
  std::array<iree_byte_span_t, 2> original_spans;
  std::array<std::vector<uint32_t>, 2> expected;
  for (uint32_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool,
        iree_hal_make_pool_reservation_request(
            6 * unit, source_requirements.minimum_granularity),
        iree_infinite_timeout(), backings[i].out()));
    IREE_ASSERT_OK(iree_hal_buffer_native_host_span(backings[i], 0, 6 * unit,
                                                    &original_spans[i]));
    expected[i].assign(6 * stride, 0xabab0000u + i);
    std::memcpy(original_spans[i].data, expected[i].data(), 6 * unit);
    IREE_ASSERT_OK(iree_hal_buffer_flush_host_range(backings[i], 0, 6 * unit));
  }
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = 3 * unit,
      .mapping_granularity = unit,
  };
  Ref<iree_hal_buffer_t> root;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, scope, {1, &source_pool}, &options, allocator_, root.out()));
  EXPECT_EQ(root.get()->memory.contract->host.access,
            host_alias ? IREE_HAL_MEMORY_ACCESS_READ : 0u);
  iree_hal_memory_scope_t cpu_scope, npu_scope, host_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_QUEUE, cpu_family}, &cpu_scope));
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_PROGRAM, npu_family}, &npu_scope));
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_HOST, nullptr}, &host_scope));
  uint32_t cpu_projection = UINT32_MAX, npu_projection, gpu_projection;
  if (host_alias) {
    IREE_ASSERT_OK(iree_hal_virtual_buffer_resolve_projection(root, cpu_scope,
                                                              &cpu_projection));
  }
  IREE_ASSERT_OK(iree_hal_virtual_buffer_resolve_projection(root, npu_scope,
                                                            &npu_projection));
  IREE_ASSERT_OK(iree_hal_virtual_buffer_resolve_projection(root, scope_,
                                                            &gpu_projection));
  EXPECT_NE(npu_projection, gpu_projection);
  const iree_hal_buffer_native_binding_slot_t gpu_slot = {
      .index = root.get()
                   ->memory.contract->scopes[scope_.id]
                   .bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS],
      .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  const iree_hal_buffer_native_binding_slot_t npu_slot = {
      .index = root.get()
                   ->memory.contract->scopes[npu_scope.id]
                   .bindings[IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA],
      .type = IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
  };
  std::array<Ref<iree_hal_buffer_t>, 3> views;
  std::array<Ref<iree_hal_buffer_t>, 3> dispatch_views;
  std::array<iree_byte_span_t, 3> host_views;
  std::array<uint64_t, 3> gpu_addresses, npu_addresses;
  std::array<iree_hal_buffer_ref_t, 3> bindings;
  for (uint32_t i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(iree_hal_buffer_subspan(root, i * unit, unit, allocator_,
                                           views[i].out()));
    if (host_alias) {
      IREE_ASSERT_OK(
          iree_hal_buffer_native_host_span(views[i], 256, 64, &host_views[i]));
    }
    gpu_addresses[i] =
        iree_hal_buffer_native_binding(views[i], gpu_slot).device_address;
    npu_addresses[i] =
        iree_hal_buffer_native_binding(views[i], npu_slot).device_address;
    // This executable admits binding-relative offset zero. An ordinary HAL
    // subspan supplies that ABI while retaining the interior virtual address.
    IREE_ASSERT_OK(iree_hal_buffer_subspan(views[i], 256, 64, allocator_,
                                           dispatch_views[i].out()));
    bindings[i] = iree_hal_make_buffer_ref(dispatch_views[i], 0, 64);
  }
  root.reset();
  const auto* targets =
      iree_hal_device_spec_executables(iree_hal_device_spec(npu_));
  ASSERT_EQ(targets->target_count, 1u);
  const bool halo = iree_string_view_equal(
      targets->targets[0].target_key, IREE_SV("amd.xdna.strix_halo.17f0_11"));
  const auto* toc = halo ? iree_hal_amd_xdna_test_mul_i32_create()
                         : iree_hal_amd_xdna_test_mul_i32_npu4_create();
  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  load_params.executable_data =
      iree_make_const_byte_span(toc[0].data, toc[0].size);
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(iree_hal_executable_load(npu_family, &targets->targets[0],
                                          &load_params, executable.out()));
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV("mul_i32"), &function));
  Ref<iree_hal_semaphore_t> completed;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      npu_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, completed.out()));
  auto* completed_semaphore = completed.get();
  const auto staging_address =
      iree_hal_buffer_native_binding(backing_, source_slot_).device_address;
  auto* staging = reinterpret_cast<uint32_t*>(host_mapping_.contents.data);
  const bool cache_control =
      native_family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR;
  for (uint32_t epoch = 0; epoch < 3; ++epoch) {
    const std::array<iree_hal_memory_scope_t, 3> shared_scopes = {
        cpu_scope, npu_scope, host_scope};
    std::vector<iree_hal_memory_mapping_t> mappings;
    for (uint32_t i = 0; i < 3; ++i) {
      const iree_hal_memory_mapping_t mapping = {
          .target = views[i],
          .scope =
              cpu_projection == npu_projection ? shared_scopes[i] : npu_scope,
          .target_range = {0, unit},
          .source = backings[(epoch + i) % 2],
          .source_offset = (1 + epoch + i) * unit,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .operation = epoch ? IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE
                             : IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
          .previous_source =
              epoch ? backings[(epoch - 1 + i) % 2].get() : nullptr,
          .previous_source_offset = epoch ? (epoch + i) * unit : 0,
      };
      mappings.push_back(mapping);
      mappings.push_back(mapping);
      mappings.back().scope = scope_;
      if (host_alias && cpu_projection != npu_projection) {
        mappings.push_back(mapping);
        mappings.back().scope = cpu_scope;
      }
    }
    // Both participants submit through their existing host progress service.
    auto* mapping_queue = iree_hal_device_queue(epoch == 1 ? npu_ : cpu_, 0, 0);
    ++sequence_;
    IREE_ASSERT_OK(iree_hal_queue_map(mapping_queue, {},
                                      {1, &done_, &sequence_}, mappings.size(),
                                      mappings.data(), 0));
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(done_, sequence_, iree_infinite_timeout(), 0));
    auto* rhs = reinterpret_cast<uint32_t*>(
        host_alias
            ? host_views[1].data
            : original_spans[(epoch + 1) % 2].data + (epoch + 2) * unit + 256);
    std::array<uint32_t, 16> product;
    for (uint32_t i = 0; i < 16; ++i) {
      staging[i] = 3 + epoch * 17 + i;
      rhs[i] = 19 + epoch * 29 + i;
      product[i] = staging[i] * rhs[i];
    }
    IREE_ASSERT_OK(iree_hal_buffer_flush_host_range(
        host_alias ? views[1].get() : backings[(epoch + 1) % 2].get(),
        host_alias ? 256 : (epoch + 2) * unit + 256, 64));
    SdmaCommandWriter upload(
        reinterpret_cast<uint32_t*>(queue_host_.ring_address) + ring_words_,
        native_family_.format_features);
    if (cache_control) {
      upload.AcquireFromSystem();
    }
    upload.CopyLinear(staging_address, gpu_addresses[0] + 256, 64);
    if (cache_control) {
      upload.ReleaseToSystem();
    }
    ASSERT_NO_FATAL_FAILURE(SubmitAndWait(upload));
    uint64_t completed_value = epoch + 1;
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        iree_hal_device_queue(npu_, 0, 0), {1, &done_, &sequence_},
        {1, &completed_semaphore, &completed_value}, executable, function,
        iree_hal_make_static_dispatch_config(1, 1, 1),
        iree_const_byte_span_empty(), {bindings.size(), bindings.data()},
        nullptr, IREE_HAL_DISPATCH_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(completed, completed_value,
                                           iree_infinite_timeout(), 0));
    // Observe the stable CPU alias when requested; device-only reservations
    // leave CPU accesses on the original backing's ordinary host mapping.
    IREE_ASSERT_OK(iree_hal_buffer_invalidate_host_range(
        host_alias ? views[2].get() : backings[epoch % 2].get(),
        host_alias ? 256 : (epoch + 3) * unit + 256, 64));
    EXPECT_EQ(std::memcmp(host_alias ? host_views[2].data
                                     : original_spans[epoch % 2].data +
                                           (epoch + 3) * unit + 256,
                          product.data(), 64),
              0);
    SdmaCommandWriter readback(
        reinterpret_cast<uint32_t*>(queue_host_.ring_address) + ring_words_,
        native_family_.format_features);
    if (cache_control) {
      readback.AcquireFromSystem();
    }
    readback.CopyLinear(gpu_addresses[2] + 256, staging_address + granularity_,
                        64);
    if (cache_control) {
      readback.ReleaseToSystem();
    }
    ASSERT_NO_FATAL_FAILURE(SubmitAndWait(readback));
    EXPECT_EQ(std::memcmp(host_mapping_.contents.data + granularity_,
                          product.data(), 64),
              0);
    const std::array<const uint32_t*, 3> outputs = {staging, rhs,
                                                    product.data()};
    for (uint32_t i = 0; i < 3; ++i) {
      std::copy_n(
          outputs[i], 16,
          expected[(epoch + i) % 2].data() + (1 + epoch + i) * stride + 64);
      if (host_alias) {
        iree_byte_span_t current;
        IREE_ASSERT_OK(
            iree_hal_buffer_native_host_span(views[i], 256, 64, &current));
        EXPECT_EQ(current.data, host_views[i].data);
      }
      EXPECT_EQ(
          iree_hal_buffer_native_binding(views[i], gpu_slot).device_address,
          gpu_addresses[i]);
      EXPECT_EQ(
          iree_hal_buffer_native_binding(views[i], npu_slot).device_address,
          npu_addresses[i]);
    }
    for (uint32_t i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(
          iree_hal_buffer_invalidate_host_range(backings[i], 0, 6 * unit));
      EXPECT_EQ(
          std::memcmp(original_spans[i].data, expected[i].data(), 6 * unit), 0);
    }
  }
  auto* owner = &iree_hal_buffer_virtual_view(views[0]).root->buffer;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(
      owner, {iree_hal_device_group_memory_domain(group_),
              IREE_HAL_MEMORY_SCOPE_ANY}));
}

TEST_P(FabricXdnaVirtualBufferTest, CpuGpuNpuExecuteAcrossBackingReplacement) {
  RunReplacement(HostProjection::kStableReservation);
}

TEST_P(FabricXdnaVirtualBufferTest, CpuGpuNpuExecuteWithRequiredHostPlacement) {
  const auto* topology = iree_hal_device_group_topology(group_);
  for (iree_host_size_t i = 0; i < iree_hal_topology_node_count(topology);
       ++i) {
    const auto* node = iree_hal_topology_node_at(topology, i);
    if (node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_HOST_NUMA) {
      RunReplacement(HostProjection::kStableReservation,
                     {IREE_HAL_POOL_PLACEMENT_REQUIRED, node->ordinal});
      return;
    }
  }
  GTEST_SKIP() << "no host NUMA allocation domain in the device group";
}

TEST_P(FabricXdnaVirtualBufferTest,
       DeviceMappingsUseOrdinaryHostBackingAccess) {
  RunReplacement(HostProjection::kBackingOnly);
}

INSTANTIATE_TEST_SUITE_P(
    NativeLifetime, FabricXdnaVirtualBufferTest,
    ::testing::Values(AMDF_NATIVE_LIFETIME_PROCESS,
                      AMDF_NATIVE_LIFETIME_INSTANCE),
    [](const ::testing::TestParamInfo<amdf_native_lifetime_t>& info) {
      return info.param == AMDF_NATIVE_LIFETIME_PROCESS ? "Process"
                                                        : "Instance";
    });

}  // namespace iree::hal::amd::cts
