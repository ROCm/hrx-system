// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/kernel_request_producer.h"

#include <memory>
#include <vector>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/link/module_index.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

struct IndexDeleter {
  void operator()(loom_link_module_index_t* index) const {
    loom_link_module_index_free(index);
  }
};
using IndexPtr = std::unique_ptr<loom_link_module_index_t, IndexDeleter>;

struct ProducerDeleter {
  void operator()(loom_kernel_request_producer_t* producer) const {
    loom_kernel_request_producer_free(producer);
  }
};
using ProducerPtr =
    std::unique_ptr<loom_kernel_request_producer_t, ProducerDeleter>;

typedef struct CapturedRequest {
  iree_host_size_t source_symbol_ordinal;
  loom_decision_class_ordinal_t class_ordinal;
  iree_host_size_t member_count;
  loom_kernel_class_product_t product;
} CapturedRequest;

typedef struct RequestCapture {
  iree_arena_block_pool_t* block_pool;
  std::vector<CapturedRequest> requests;
} RequestCapture;

static iree_status_t CaptureRequest(void* user_data,
                                    const loom_kernel_request_t* request) {
  RequestCapture* capture = static_cast<RequestCapture*>(user_data);
  loom_kernel_class_product_t product = {};
  IREE_RETURN_IF_ERROR(loom_kernel_request_materialize(
      request, capture->block_pool, iree_allocator_system(), &product));
  capture->requests.push_back(CapturedRequest{
      request->source_symbol_ordinal,
      request->class_ordinal,
      request->member_count,
      product,
  });
  return iree_ok_status();
}

typedef struct RejectRequestState {
  iree_host_size_t call_count;
} RejectRequestState;

static iree_status_t RejectRequest(void* user_data,
                                   const loom_kernel_request_t* request) {
  (void)request;
  RejectRequestState* state = static_cast<RejectRequestState*>(user_data);
  ++state->call_count;
  return iree_make_status(IREE_STATUS_ABORTED, "request sink stopped");
}

class KernelRequestProducerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseSource() {
    const iree_string_view_t source = IREE_SV(R"(
template.decl @request.schedule(%size: index)

template.def<@request.schedule> priority(10) @large(%size: index) where [ge(%size, 128)] {
  template.return
}

template.def<@request.schedule> priority(1) @small(%size: index) {
  template.return
}

kernel.def @classified() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%size: index) {
  template.apply<@request.schedule>(%size) : (index)
  kernel.return
}
)");
    loom_text_parse_options_t parse_options = {};
    parse_options.diagnostic_sink.fn = loom_diagnostic_stderr_sink;
    parse_options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(
        loom_text_parse(source, IREE_SV("kernel_request_producer_test.loom"),
                        &context_, &block_pool_, &parse_options, &module));
    ModulePtr module_ptr(module);
    Verify(module);
    return module_ptr;
  }

  void Verify(const loom_module_t* module) {
    loom_verify_options_t verify_options = {};
    verify_options.sink.fn = loom_diagnostic_stderr_sink;
    verify_options.max_errors = 20;
    loom_verify_result_t verify_result = {};
    IREE_CHECK_OK(loom_verify_module(module, &verify_options, &verify_result));
    EXPECT_EQ(verify_result.error_count, 0u);
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    EXPECT_NE(name_id, LOOM_STRING_ID_INVALID);
    if (name_id == LOOM_STRING_ID_INVALID) return LOOM_SYMBOL_ID_INVALID;
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    EXPECT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  std::vector<uint8_t> WriteModule(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(loom_bytecode_write_module(
        module, stream, /*options=*/nullptr, &block_pool_));
    const iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytecode(length);
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(iree_io_stream_read(stream, bytecode.size(), bytecode.data(),
                                      /*out_buffer_length=*/nullptr));
    iree_io_stream_release(stream);
    return bytecode;
  }

  IndexPtr CreateIndex() {
    loom_link_module_index_t* index = nullptr;
    IREE_CHECK_OK(loom_link_module_index_allocate(
        &context_, &block_pool_, iree_allocator_system(), &index));
    return IndexPtr(index);
  }

  iree_host_size_t AddMaterializedSource(loom_link_module_index_t* index,
                                         const loom_module_t* module) {
    const loom_link_module_index_add_options_t options = {
        /*.provider_name=*/IREE_SV("kernel-source"),
        /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
    };
    iree_host_size_t provider_ordinal = 0;
    IREE_CHECK_OK(loom_link_module_index_add_materialized(
        index, module, &options, &provider_ordinal));
    return SourceSymbolOrdinal(index, provider_ordinal, module);
  }

  iree_host_size_t AddBytecodeSource(loom_link_module_index_t* index,
                                     const std::vector<uint8_t>& bytecode,
                                     const loom_module_t* source_module) {
    const loom_link_module_index_add_options_t options = {
        /*.provider_name=*/IREE_SV("kernel-source"),
        /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
    };
    iree_host_size_t provider_ordinal = 0;
    IREE_CHECK_OK(loom_link_module_index_add_bytecode(
        index, iree_make_const_byte_span(bytecode.data(), bytecode.size()),
        IREE_SV("kernel-source.loombc"), /*index_options=*/nullptr, &options,
        &provider_ordinal));
    return SourceSymbolOrdinal(index, provider_ordinal, source_module);
  }

  iree_host_size_t SourceSymbolOrdinal(const loom_link_module_index_t* index,
                                       iree_host_size_t provider_ordinal,
                                       const loom_module_t* source_module) {
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_provider_at(index, provider_ordinal);
    EXPECT_NE(provider, nullptr);
    if (provider == nullptr) return LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    const loom_link_module_index_module_t* indexed_module =
        loom_link_module_index_module_at(index, provider->module_start_ordinal);
    EXPECT_NE(indexed_module, nullptr);
    if (indexed_module == nullptr) {
      return LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    }
    return indexed_module->symbol_start_ordinal +
           FindSymbol(source_module, IREE_SV("classified"));
  }

  loom_link_plan_materialization_environment_t MaterializationEnvironment() {
    return loom_link_plan_materialization_environment_t{
        /*.context=*/&context_,
        /*.block_pool=*/&block_pool_,
        /*.low_repr_environment=*/{},
        /*.diagnostic_sink=*/nullptr,
        /*.prepare_module=*/nullptr,
        /*.user_data=*/nullptr,
        /*.allocator=*/iree_allocator_system(),
    };
  }

  ProducerPtr CreateProducer(const loom_link_module_index_t* index) {
    const loom_link_plan_materialization_environment_t environment =
        MaterializationEnvironment();
    loom_kernel_request_producer_t* producer = nullptr;
    IREE_CHECK_OK(
        loom_kernel_request_producer_allocate(index, &environment, &producer));
    return ProducerPtr(producer);
  }

  void BuildSites(iree_arena_allocator_t* arena,
                  loom_value_fact_table_t* out_facts,
                  loom_kernel_class_site_t out_sites[2],
                  loom_value_id_t out_argument_values[2]) {
    IREE_CHECK_OK(
        loom_value_fact_table_initialize(out_facts, arena, /*capacity=*/2));
    out_facts->count = 2;
    out_facts->entries[0] = loom_value_facts_exact_i64(64);
    out_facts->entries[1] = loom_value_facts_exact_i64(256);
    out_argument_values[0] = 0;
    out_argument_values[1] = 1;
    out_sites[0] = {
        /*.facts=*/out_facts,
        /*.argument_values=*/&out_argument_values[0],
    };
    out_sites[1] = {
        /*.facts=*/out_facts,
        /*.argument_values=*/&out_argument_values[1],
    };
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
};

TEST_F(KernelRequestProducerTest,
       PublishesBytecodeBackedClassesWithIndependentOwnership) {
  ModulePtr source_module = ParseSource();
  ASSERT_NE(source_module, nullptr);
  std::vector<uint8_t> bytecode = WriteModule(source_module.get());
  IndexPtr index = CreateIndex();
  const iree_host_size_t source_symbol_ordinal =
      AddBytecodeSource(index.get(), bytecode, source_module.get());
  ProducerPtr producer = CreateProducer(index.get());

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  loom_value_fact_table_t site_facts = {};
  loom_kernel_class_site_t sites[2] = {};
  loom_value_id_t argument_values[2] = {};
  BuildSites(&scratch_arena, &site_facts, sites, argument_values);

  RequestCapture capture = {
      /*.block_pool=*/&block_pool_,
  };
  const loom_kernel_class_collection_options_t collection_options =
      loom_kernel_class_collection_options_default();
  loom_kernel_class_collection_t collection = {};
  const loom_link_plan_materialization_environment_t environment =
      MaterializationEnvironment();
  IREE_ASSERT_OK(loom_kernel_request_producer_publish(
      producer.get(), &environment, source_symbol_ordinal, sites,
      IREE_ARRAYSIZE(sites), &collection_options,
      (loom_kernel_request_sink_t){
          /*.publish=*/CaptureRequest,
          /*.user_data=*/&capture,
      },
      &scratch_arena, &collection));
  ASSERT_EQ(collection.class_count, 2u);
  EXPECT_NE(collection.site_classes[0], collection.site_classes[1]);

  producer.reset();
  index.reset();
  source_module.reset();
  bytecode.clear();
  iree_arena_deinitialize(&scratch_arena);

  ASSERT_EQ(capture.requests.size(), 2u);
  for (const CapturedRequest& request : capture.requests) {
    EXPECT_EQ(request.source_symbol_ordinal, source_symbol_ordinal);
    EXPECT_EQ(request.member_count, 1u);
    ASSERT_NE(request.product.module, nullptr);
    ASSERT_TRUE(loom_symbol_ref_is_valid(request.product.kernel));
    ASSERT_LT(request.product.kernel.symbol_id,
              request.product.module->symbols.count);
    EXPECT_TRUE(
        loom_kernel_def_isa(request.product.module->symbols
                                .entries[request.product.kernel.symbol_id]
                                .defining_op));
    Verify(request.product.module);
  }
  EXPECT_NE(capture.requests[0].class_ordinal,
            capture.requests[1].class_ordinal);
  EXPECT_NE(capture.requests[0].product.module,
            capture.requests[1].product.module);
  for (CapturedRequest& request : capture.requests) {
    loom_kernel_class_product_deinitialize(&request.product);
  }
}

TEST_F(KernelRequestProducerTest, StopsAfterSinkFailureBeforeMaterialization) {
  ModulePtr source_module = ParseSource();
  ASSERT_NE(source_module, nullptr);
  IndexPtr index = CreateIndex();
  const iree_host_size_t source_symbol_ordinal =
      AddMaterializedSource(index.get(), source_module.get());

  ProducerPtr producer = CreateProducer(index.get());

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  loom_value_fact_table_t site_facts = {};
  loom_kernel_class_site_t sites[2] = {};
  loom_value_id_t argument_values[2] = {};
  BuildSites(&scratch_arena, &site_facts, sites, argument_values);

  RejectRequestState state = {};
  const loom_kernel_class_collection_options_t collection_options =
      loom_kernel_class_collection_options_default();
  loom_kernel_class_collection_t collection = {};
  const loom_link_plan_materialization_environment_t environment =
      MaterializationEnvironment();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        loom_kernel_request_producer_publish(
                            producer.get(), &environment, source_symbol_ordinal,
                            sites, IREE_ARRAYSIZE(sites), &collection_options,
                            (loom_kernel_request_sink_t){
                                /*.publish=*/RejectRequest,
                                /*.user_data=*/&state,
                            },
                            &scratch_arena, &collection));
  EXPECT_EQ(state.call_count, 1u);
  EXPECT_EQ(collection.class_count, 0u);

  iree_arena_deinitialize(&scratch_arena);
}

}  // namespace
}  // namespace loom
