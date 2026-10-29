// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/kernel_request_producer.h"

#include <cstring>
#include <memory>
#include <string>
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
#include "loom/ops/scalar/ops.h"
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

class KernelRequestProducerTest : public ::testing::TestWithParam<bool> {
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
    const std::string source =
        std::string(R"(
template.decl @request.schedule(%size: index)

template.def<@request.schedule> priority(10) @large(%size: index) where [ge(%size, 128)] {
  template.return
}

template.def<@request.schedule> priority(1) @small(%size: index) {
  template.return
}

)") +
        (GetParam() ? "target.decl @device\nkernel.def target(@device) "
                    : "kernel.def ") +
        R"(@classified() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%size: index) {
  template.apply<@request.schedule>(%size) : (index)
  kernel.return
}
)";
    return ParseText(source);
  }

  ModulePtr ParseText(const std::string& source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.diagnostic_sink.fn = loom_diagnostic_stderr_sink;
    parse_options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source.c_str()),
                                  IREE_SV("kernel_request_producer_test.loom"),
                                  &context_, &block_pool_, &parse_options,
                                  &module));
    ModulePtr module_ptr(module);
    if (!module) {
      return module_ptr;
    }
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
    if (name_id == LOOM_STRING_ID_INVALID) {
      return LOOM_SYMBOL_ID_INVALID;
    }
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
        .provider_name = IREE_SV("kernel-source"),
        .role = LOOM_LINK_PROVIDER_ROLE_INPUT,
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
        .provider_name = IREE_SV("kernel-source"),
        .role = LOOM_LINK_PROVIDER_ROLE_INPUT,
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
    if (provider == nullptr) {
      return LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    }
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
        .context = &context_,
        .block_pool = &block_pool_,
        .low_repr_environment = {},
        .diagnostic_sink = nullptr,
        .prepare_module = nullptr,
        .user_data = nullptr,
        .allocator = iree_allocator_system(),
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
        .facts = out_facts,
        .argument_values = &out_argument_values[0],
    };
    out_sites[1] = {
        .facts = out_facts,
        .argument_values = &out_argument_values[1],
    };
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
};

TEST_P(KernelRequestProducerTest, BindsScalarAndVectorLiteralFamilies) {
  const char* scalar_types[] = {"index", "offset", "i1",     "i8",     "i16",
                                "i32",   "i64",    "f8E4M3", "f8E5M2", "f16",
                                "bf16",  "f32",    "f64"};
  for (const char* scalar_type : scalar_types) {
    for (const char* shape : {"", "1x", "4x", "2x3x5x", "0x"}) {
      const bool vector = shape[0] != '\0';
      const bool empty = std::strcmp(shape, "0x") == 0;
      const bool boolean = std::strcmp(scalar_type, "i1") == 0;
      const std::string type =
          vector ? std::string("vector<") + shape + scalar_type + ">"
                 : scalar_type;
      SCOPED_TRACE(type);
      const bool floating = scalar_type[0] == 'f' || scalar_type[0] == 'b';
      const std::string operation =
          empty    ? "vector.empty"
          : vector ? (floating ? "vector.splatf" : "vector.splati")
          : (std::strcmp(scalar_type, "index") == 0 ||
             std::strcmp(scalar_type, "offset") == 0)
              ? "index.constant"
              : "scalar.constant";
      const std::string first = empty      ? ""
                                : boolean  ? (vector ? " 1" : " true")
                                : floating ? " 0.0"
                                           : " 1";
      const std::string second = empty      ? ""
                                 : boolean  ? (vector ? " 0" : " false")
                                 : floating ? " -0.0"
                                            : " 2";
      const std::string source =
          "func.decl @observe(%value: " + type + ")\n" +
          (GetParam()
               ? "target.decl @device\npipeline.def<kernel> target(@device) "
               : "pipeline.def<kernel> ") +
          "@classified(%value: " + type +
          ") run(%output: buffer) {\n"
          "func.call @observe(%value) : (" +
          type +
          ")\npipeline.finish\n}\n"
          "func.def @sites(%output: buffer) {\n%first = " +
          operation + first + " : " + type + "\n%second = " + operation +
          second + " : " + type + "\n%equal = " + operation + first + " : " +
          type + "\nfunc.return\n}\n";
      ModulePtr source_module = ParseText(source);
      ASSERT_NE(source_module, nullptr);
      std::vector<uint8_t> bytecode = WriteModule(source_module.get());
      IndexPtr index = CreateIndex();
      const iree_host_size_t source_symbol_ordinal =
          AddBytecodeSource(index.get(), bytecode, source_module.get());
      ProducerPtr producer = CreateProducer(index.get());
      iree_arena_allocator_t scratch_arena;
      iree_arena_initialize(&block_pool_, &scratch_arena);
      const auto site_function = loom_func_like_cast(
          source_module.get(),
          source_module->symbols
              .entries[FindSymbol(source_module.get(), IREE_SV("sites"))]
              .defining_op);
      loom_value_fact_table_t facts = {};
      IREE_ASSERT_OK(
          loom_value_fact_table_initialize(&facts, &scratch_arena, 0));
      IREE_ASSERT_OK(loom_value_fact_table_compute(&facts, source_module.get(),
                                                   site_function));
      loom_value_id_t arguments[3][2] = {};
      loom_kernel_class_site_t sites[3] = {};
      const loom_op_t* literal =
          loom_region_entry_block(loom_func_like_body(site_function))->first_op;
      for (size_t i = 0; i < 3; ++i) {
        arguments[i][0] = loom_op_results((loom_op_t*)literal)[0];
        arguments[i][1] =
            loom_region_entry_block(loom_func_like_body(site_function))
                ->arg_ids[0];
        sites[i] = {.facts = &facts, .argument_values = arguments[i]};
        literal = literal->next_op;
      }
      RequestCapture capture = {.block_pool = &block_pool_};
      const auto environment = MaterializationEnvironment();
      auto options = loom_kernel_class_collection_options_default();
      loom_kernel_class_collection_t collection = {};
      if (!empty) {
        options.class_limit = 1;
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            loom_kernel_request_producer_publish(
                producer.get(), &environment, source_symbol_ordinal,
                source_module.get(), sites, std::size(sites), &options,
                {.publish = CaptureRequest, .user_data = &capture},
                &scratch_arena, &collection));
        EXPECT_TRUE(capture.requests.empty());
        options = loom_kernel_class_collection_options_default();
      }
      IREE_ASSERT_OK(loom_kernel_request_producer_publish(
          producer.get(), &environment, source_symbol_ordinal,
          source_module.get(), sites, std::size(sites), &options,
          {.publish = CaptureRequest, .user_data = &capture}, &scratch_arena,
          &collection));
      EXPECT_EQ(collection.class_count, empty ? 1u : 2u);
      EXPECT_EQ(collection.site_classes[0], collection.site_classes[2]);
      producer.reset();
      index.reset();
      source_module.reset();
      bytecode.clear();
      iree_arena_deinitialize(&scratch_arena);
      for (CapturedRequest& request : capture.requests) {
        Verify(request.product.module);
        const auto function =
            loom_func_like_cast(request.product.module,
                                request.product.module->symbols
                                    .entries[request.product.kernel.symbol_id]
                                    .defining_op);
        uint16_t argument_count = 0;
        loom_func_like_arg_ids(function, &argument_count);
        EXPECT_EQ(argument_count, 1u);
        EXPECT_EQ(loom_func_like_specialization_count(function), 0);
        const loom_op_t* bound =
            loom_region_entry_block(loom_func_like_body(function))->first_op;
        EXPECT_TRUE(iree_all_bits_set(bound->traits, LOOM_TRAIT_CONSTANT_LIKE));
        const loom_type_t bound_type = loom_module_value_type(
            request.product.module, loom_op_results((loom_op_t*)bound)[0]);
        EXPECT_EQ(loom_type_is_vector(bound_type), vector);
        if (vector) {
          EXPECT_EQ(loom_type_rank(bound_type),
                    std::strcmp(shape, "2x3x5x") == 0 ? 3u : 1u);
        }
        loom_kernel_class_product_deinitialize(&request.product);
      }
    }
  }
}

TEST_P(KernelRequestProducerTest,
       PreservesMandatoryClassesAndResidualContracts) {
  for (const char* scalar_type :
       {"f8E4M3", "f8E5M2", "f16", "bf16", "f32", "f64"}) {
    SCOPED_TRACE(scalar_type);
    const std::string type = scalar_type;
    const std::string source =
        R"(
template.decl @schedule(%size: index)
template.def<@schedule> priority(10) @large(%size: index) where [ge(%size, 512)] {
  template.return
}
template.def<@schedule> priority(1) @small(%size: index) {
  template.return
}
)" + std::string("func.decl @observe(%value: ") +
        type + ")\n" +
        (GetParam()
             ? "target.decl @device\npipeline.def<kernel> target(@device) "
             : "pipeline.def<kernel> ") +
        "@classified(%width: index, %reference: " + type +
        ") "
        "run(%size: index, %value: " +
        type +
        ", %output: buffer) "
        "where [multiple_of(%width, 16), ge(%size, %width), "
        "eq(%value, %reference), finite(%reference), finite(%value)] {\n"
        "template.apply<@schedule>(%size) : (index)\n"
        "func.call @observe(%value) : (" +
        type +
        ")\npipeline.finish\n}\n"
        "func.def @sites(%output: buffer) {\n"
        "%narrow = index.constant 64 : index\n"
        "%wide = index.constant 128 : index\n"
        "%small = index.constant 256 : index\n"
        "%large = index.constant 512 : index\n"
        "%reference = scalar.constant 1.25 : " +
        type + "\nfunc.return\n}\n";
    ModulePtr source_module = ParseText(source);
    ASSERT_NE(source_module, nullptr);
    auto bytecode = WriteModule(source_module.get());
    IndexPtr index = CreateIndex();
    const auto source_symbol_ordinal =
        AddBytecodeSource(index.get(), bytecode, source_module.get());
    ProducerPtr producer = CreateProducer(index.get());
    iree_arena_allocator_t scratch_arena;
    iree_arena_initialize(&block_pool_, &scratch_arena);
    const auto site_function = loom_func_like_cast(
        source_module.get(),
        source_module->symbols
            .entries[FindSymbol(source_module.get(), IREE_SV("sites"))]
            .defining_op);
    loom_value_fact_table_t facts = {};
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &scratch_arena, 0));
    IREE_ASSERT_OK(loom_value_fact_table_compute(&facts, source_module.get(),
                                                 site_function));
    const auto* block =
        loom_region_entry_block(loom_func_like_body(site_function));
    loom_value_id_t constants[5] = {};
    const auto* literal = block->first_op;
    for (auto& value : constants) {
      value = loom_op_const_results(literal)[0];
      literal = literal->next_op;
    }
    loom_value_id_t arguments[4][5] = {};
    loom_kernel_class_site_t sites[4] = {};
    for (size_t i = 0; i < 4; ++i) {
      arguments[i][0] = constants[i / 2];
      arguments[i][1] = constants[4];
      arguments[i][2] = constants[2 + i % 2];
      arguments[i][3] = constants[4];
      arguments[i][4] = block->arg_ids[0];
      sites[i] = {.facts = &facts, .argument_values = arguments[i]};
    }
    RequestCapture capture = {.block_pool = &block_pool_};
    const auto environment = MaterializationEnvironment();
    auto options = loom_kernel_class_collection_options_default();
    options.class_limit = 2;
    loom_kernel_class_collection_t collection = {};
    IREE_ASSERT_OK(loom_kernel_request_producer_publish(
        producer.get(), &environment, source_symbol_ordinal,
        source_module.get(), sites, std::size(sites), &options,
        {.publish = CaptureRequest, .user_data = &capture}, &scratch_arena,
        &collection));
    EXPECT_EQ(collection.class_count, 2u);
    EXPECT_EQ(collection.accepted_decision_count, 0u);
    EXPECT_EQ(collection.skipped_decision_count, 1u);
    EXPECT_EQ(collection.decision_results[0].state,
              LOOM_KERNEL_CLASS_DECISION_SKIPPED_CLASS_LIMIT);
    EXPECT_EQ(collection.site_classes[0], collection.site_classes[1]);
    EXPECT_EQ(collection.site_classes[2], collection.site_classes[3]);
    EXPECT_NE(collection.site_classes[0], collection.site_classes[2]);
    producer.reset();
    index.reset();
    source_module.reset();
    bytecode.clear();
    iree_arena_deinitialize(&scratch_arena);
    ASSERT_EQ(capture.requests.size(), 2u);
    for (auto& request : capture.requests) {
      Verify(request.product.module);
      const auto function = loom_func_like_cast(
          request.product.module, request.product.module->symbols
                                      .entries[request.product.kernel.symbol_id]
                                      .defining_op);
      uint16_t predicate_count = 0;
      const auto* predicates =
          loom_func_like_predicates(function, &predicate_count);
      ASSERT_GE(predicate_count, 2u);
      EXPECT_EQ(predicates[0].kind, LOOM_PREDICATE_GE);
      EXPECT_EQ(predicates[0].arg_tags[1], LOOM_PRED_ARG_CONST);
      EXPECT_EQ(predicates[0].args[1], request.class_ordinal == 0 ? 64 : 128);
      EXPECT_EQ(predicates[1].kind, LOOM_PREDICATE_FINITE);
      iree_arena_allocator_t query_arena;
      iree_arena_initialize(&block_pool_, &query_arena);
      loom_value_fact_table_t retained = {};
      IREE_ASSERT_OK(loom_value_fact_table_initialize(
          &retained, &query_arena, request.product.module->values.count));
      IREE_ASSERT_OK(loom_value_fact_table_compute(
          &retained, request.product.module, function));
      const auto* body = loom_region_entry_block(loom_func_like_body(function));
      const auto size_facts =
          loom_value_fact_table_lookup(&retained, body->arg_ids[0]);
      EXPECT_EQ(size_facts.range_lo, 256);
      EXPECT_EQ(size_facts.range_hi, 512);
      EXPECT_EQ(size_facts.known_divisor, 256);
      iree_arena_deinitialize(&query_arena);
      const auto* first =
          loom_region_entry_block(loom_func_like_body(function))->first_op;
      const auto* assume = first->next_op->next_op;
      ASSERT_TRUE(loom_scalar_assume_isa(assume));
      const auto assumption = loom_scalar_assume_predicates(assume);
      ASSERT_EQ(assumption.count, 1u);
      EXPECT_EQ(assumption.predicate_list[0].kind, LOOM_PREDICATE_EQ);
      EXPECT_EQ(assume->operand_count, 2u);
      loom_kernel_class_product_deinitialize(&request.product);
    }
  }
}

TEST_P(KernelRequestProducerTest, SharesStructuralBodiesAcrossDistinctOffsets) {
  for (bool use_bytecode : {false, true}) {
    SCOPED_TRACE(use_bytecode);
    const std::string source_text =
        R"(
func.def inline @extent(%rows: index, %unused: offset) -> (index) {
  func.return %rows : index
}
template.decl @write(%rows: index, %offset: offset, %output: buffer)
template.def<@write> @write_impl(%rows: index, %offset: offset, %output: buffer) {
  func.call @compute(%rows, %offset, %output) : (index, offset, buffer)
  template.return
}
func.def inline @compute(%rows: index, %offset: offset, %output: buffer) {
  %extent = func.call @extent(%rows, %offset) : (index, offset) -> (index)
  %values = vector.splatf 1.0 : vector<[%extent]xf32>
  %view = buffer.view %output[%offset] : buffer -> view<[%extent]xf32>
  vector.store %values, %view[0] : vector<[%extent]xf32>, view<[%extent]xf32>
  func.return
}
template.decl @schedule(%rows: index, %fixed: index, %offset: offset, %output: buffer)
template.def<@schedule> priority(10) @fixed(%rows: index, %fixed: index, %offset: offset, %output: buffer) where [eq(%fixed, 1)] {
  %width = index.constant 4 : index
  func.call @compute(%width, %offset, %output) : (index, offset, buffer)
  template.return
}
template.def<@schedule> priority(1) @variable(%rows: index, %fixed: index, %offset: offset, %output: buffer) {
  template.apply<@write>(%rows, %offset, %output) : (index, offset, buffer)
  template.return
}
)" +
        std::string(GetParam()
                        ? "target.decl @device\nkernel.def target(@device) "
                        : "kernel.def ") +
        R"(@classified() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%rows: index, %padding: index, %fixed: index, %offset: offset, %output: buffer) {
  %width = index.add %rows, %padding : index
  template.apply<@schedule>(%width, %fixed, %offset, %output) : (index, index, offset, buffer)
  kernel.return
}
)";
    ModulePtr source = ParseText(source_text);
    ASSERT_NE(source, nullptr);
    auto bytes = WriteModule(source.get());
    auto index = CreateIndex();
    const auto source_ordinal =
        use_bytecode ? AddBytecodeSource(index.get(), bytes, source.get())
                     : AddMaterializedSource(index.get(), source.get());
    auto producer = CreateProducer(index.get());
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    constexpr size_t kSites = 999;
    std::vector<loom_value_facts_t> facts(kSites * 5);
    std::vector<loom_value_id_t> arguments(kSites * 5);
    std::vector<loom_kernel_class_site_t> sites(kSites);
    const loom_value_fact_table_t observations = {
        .entries = facts.data(),
        .count = facts.size(),
        .capacity = facts.size(),
    };
    for (size_t i = 0; i < kSites; ++i) {
      const int64_t width = i % 3 == 0 ? 8 : i % 3 == 1 ? 16 : 32 + i;
      const int64_t padding = i % 4;
      facts[i * 5] = loom_value_facts_exact_i64(width - padding);
      facts[i * 5 + 1] = loom_value_facts_exact_i64(padding);
      facts[i * 5 + 2] = loom_value_facts_exact_i64(i % 3 == 2);
      facts[i * 5 + 3] = loom_value_facts_exact_i64(256 * i);
      facts[i * 5 + 4] = loom_value_facts_unknown();
      for (size_t j = 0; j < 5; ++j) {
        arguments[i * 5 + j] = i * 5 + j;
      }
      sites[i] = {
          .facts = &observations,
          .argument_values = arguments.data() + i * 5,
      };
    }
    RequestCapture capture = {.block_pool = &block_pool_};
    const auto environment = MaterializationEnvironment();
    auto options = loom_kernel_class_collection_options_default();
    loom_kernel_class_collection_t collection = {};
    options.class_limit = 2;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_kernel_request_producer_publish(
            producer.get(), &environment, source_ordinal, source.get(),
            sites.data(), sites.size(), &options,
            {.publish = CaptureRequest, .user_data = &capture}, &arena,
            &collection));
    EXPECT_TRUE(capture.requests.empty());
    options = loom_kernel_class_collection_options_default();
    IREE_ASSERT_OK(loom_kernel_request_producer_publish(
        producer.get(), &environment, source_ordinal, source.get(),
        sites.data(), sites.size(), &options,
        {.publish = CaptureRequest, .user_data = &capture}, &arena,
        &collection));
    ASSERT_EQ(capture.requests.size(), 3u);
    for (size_t i = 3; i < kSites; ++i) {
      EXPECT_EQ(collection.site_classes[i], collection.site_classes[i % 3]);
    }
    producer.reset();
    index.reset();
    source.reset();
    bytes.clear();
    iree_arena_deinitialize(&arena);
    size_t fixed_count = 0;
    size_t width8_count = 0;
    size_t width16_count = 0;
    for (auto& request : capture.requests) {
      auto* module = request.product.module;
      Verify(module);
      EXPECT_EQ(request.member_count, kSites / 3);
      const auto kernel = loom_func_like_cast(
          module, module->symbols.entries[request.product.kernel.symbol_id]
                      .defining_op);
      iree_arena_initialize(&block_pool_, &arena);
      loom_value_fact_table_t retained = {};
      IREE_ASSERT_OK(loom_value_fact_table_initialize(&retained, &arena,
                                                      module->values.count));
      IREE_ASSERT_OK(loom_value_fact_table_compute(&retained, module, kernel));
      auto* body = loom_region_entry_block(loom_func_like_body(kernel));
      loom_op_t* operation = nullptr;
      loom_block_for_each_op(body, operation) {
        const auto call = loom_call_like_cast(module, operation);
        if (!loom_call_like_isa(call)) {
          continue;
        }
        const auto callee = loom_call_like_callee(call);
        const auto name = loom_string_table_get(
            &module->strings,
            module->symbols.entries[callee.symbol_id].name_id);
        if (iree_string_view_equal(name, IREE_SV("fixed"))) {
          ++fixed_count;
          continue;
        }
        EXPECT_TRUE(iree_string_view_equal(name, IREE_SV("variable")));
        const auto values = loom_call_like_operands(call);
        const auto width =
            loom_value_fact_table_lookup(&retained, values.values[0]);
        EXPECT_TRUE(loom_value_facts_is_exact(width));
        if (width.range_lo == 8) {
          ++width8_count;
        }
        if (width.range_lo == 16) {
          ++width16_count;
        }
        const auto offset =
            loom_value_fact_table_lookup(&retained, values.values[2]);
        EXPECT_FALSE(loom_value_facts_is_exact(offset));
        EXPECT_EQ(offset.known_divisor % 256, 0);
      }
      iree_arena_deinitialize(&arena);
      loom_kernel_class_product_deinitialize(&request.product);
    }
    EXPECT_EQ(fixed_count, 1u);
    EXPECT_EQ(width8_count, 1u);
    EXPECT_EQ(width16_count, 1u);
  }
}

TEST_P(KernelRequestProducerTest,
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
      .block_pool = &block_pool_,
  };
  const loom_kernel_class_collection_options_t collection_options =
      loom_kernel_class_collection_options_default();
  loom_kernel_class_collection_t collection = {};
  const loom_link_plan_materialization_environment_t environment =
      MaterializationEnvironment();
  IREE_ASSERT_OK(loom_kernel_request_producer_publish(
      producer.get(), &environment, source_symbol_ordinal, nullptr, sites,
      IREE_ARRAYSIZE(sites), &collection_options,
      loom_kernel_request_sink_t{
          .publish = CaptureRequest,
          .user_data = &capture,
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
    const auto kernel = loom_func_like_cast(
        request.product.module, request.product.module->symbols
                                    .entries[request.product.kernel.symbol_id]
                                    .defining_op);
    EXPECT_EQ(loom_symbol_ref_is_valid(loom_func_like_target(kernel)),
              GetParam());
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

TEST_P(KernelRequestProducerTest, StopsAfterSinkFailureBeforeMaterialization) {
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
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      loom_kernel_request_producer_publish(
          producer.get(), &environment, source_symbol_ordinal, nullptr, sites,
          IREE_ARRAYSIZE(sites), &collection_options,
          loom_kernel_request_sink_t{
              .publish = RejectRequest,
              .user_data = &state,
          },
          &scratch_arena, &collection));
  EXPECT_EQ(state.call_count, 1u);
  EXPECT_EQ(collection.class_count, 0u);

  iree_arena_deinitialize(&scratch_arena);
}

INSTANTIATE_TEST_SUITE_P(TargetDeclaration, KernelRequestProducerTest,
                         ::testing::Bool());

}  // namespace
}  // namespace loom
