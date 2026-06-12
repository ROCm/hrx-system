// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer/printer.h"

#include <math.h>
#include <string.h>

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/target/registers.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

static const loom_encoding_vtable_t kQ8_0EncodingVtable = {
    /*.name=*/IREE_SV("q8_0"),
};

static const loom_encoding_vtable_t kDenseEncodingVtable = {
    /*.name=*/IREE_SV("dense"),
    /*.role=*/LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
};

// Helper to print a type and return the output as a std::string.
std::string print_type(loom_type_t type,
                       const loom_module_t* module = nullptr) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  iree_status_t status = loom_text_print_type(type, module, &stream);
  std::string result;
  if (iree_status_is_ok(status)) {
    result = std::string(iree_string_builder_buffer(&builder),
                         iree_string_builder_size(&builder));
  }
  IREE_EXPECT_OK(status);
  iree_string_builder_deinitialize(&builder);
  return result;
}

struct CapturedPrintField {
  loom_print_field_ref_t field_ref;
  iree_host_size_t start;
  iree_host_size_t end;
};

static void CapturePrintFieldCallback(void* user_data,
                                      loom_print_field_ref_t field_ref,
                                      iree_host_size_t start,
                                      iree_host_size_t end) {
  auto* fields = static_cast<std::vector<CapturedPrintField>*>(user_data);
  fields->push_back(CapturedPrintField{
      /*.field_ref=*/field_ref,
      /*.start=*/start,
      /*.end=*/end,
  });
}

static const CapturedPrintField* FindCapturedPrintField(
    const std::vector<CapturedPrintField>& fields,
    loom_print_field_kind_t field_kind, uint16_t field_index) {
  for (const CapturedPrintField& field : fields) {
    if (field.field_ref.kind == field_kind &&
        field.field_ref.index == field_index) {
      return &field;
    }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Type printing
//===----------------------------------------------------------------------===//

TEST(PrintType, ScalarF32) {
  EXPECT_EQ(print_type(loom_type_scalar(LOOM_SCALAR_TYPE_F32)), "f32");
}

TEST(PrintType, ScalarI32) {
  EXPECT_EQ(print_type(loom_type_scalar(LOOM_SCALAR_TYPE_I32)), "i32");
}

TEST(PrintType, ScalarI8) {
  EXPECT_EQ(print_type(loom_type_scalar(LOOM_SCALAR_TYPE_I8)), "i8");
}

TEST(PrintType, ScalarI64) {
  EXPECT_EQ(print_type(loom_type_scalar(LOOM_SCALAR_TYPE_I64)), "i64");
}

TEST(PrintType, ScalarBF16) {
  EXPECT_EQ(print_type(loom_type_scalar(LOOM_SCALAR_TYPE_BF16)), "bf16");
}

TEST(PrintType, ScalarF16) {
  EXPECT_EQ(print_type(loom_type_scalar(LOOM_SCALAR_TYPE_F16)), "f16");
}

TEST(PrintType, ScalarF64) {
  EXPECT_EQ(print_type(loom_type_scalar(LOOM_SCALAR_TYPE_F64)), "f64");
}

TEST(PrintType, ScalarIndex) {
  EXPECT_EQ(print_type(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)), "index");
}

TEST(PrintType, ScalarI1) {
  EXPECT_EQ(print_type(loom_type_scalar(LOOM_SCALAR_TYPE_I1)), "i1");
}

TEST(PrintType, ScalarI16) {
  EXPECT_EQ(print_type(loom_type_scalar(LOOM_SCALAR_TYPE_I16)), "i16");
}

TEST(PrintType, Tile1D) {
  loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                         loom_dim_pack_static(4), 0);
  EXPECT_EQ(print_type(type), "tile<4xf32>");
}

TEST(PrintType, Tile2D) {
  loom_type_t type =
      loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), loom_dim_pack_static(4), 0);
  EXPECT_EQ(print_type(type), "tile<4x4xf32>");
}

TEST(PrintType, Tile0D) {
  loom_type_t type =
      loom_type_shaped_0d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, 0);
  EXPECT_EQ(print_type(type), "tile<f32>");
}

TEST(PrintType, Tensor1D) {
  loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                                         loom_dim_pack_static(256), 0);
  EXPECT_EQ(print_type(type), "tensor<256xf32>");
}

TEST(PrintType, Vector1D) {
  loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                                         loom_dim_pack_static(16), 0);
  EXPECT_EQ(print_type(type), "vector<16xf32>");
}

TEST(PrintType, Vector2D) {
  loom_type_t type =
      loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), loom_dim_pack_static(16), 0);
  EXPECT_EQ(print_type(type), "vector<4x16xf32>");
}

TEST(PrintType, View1D) {
  loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                                         loom_dim_pack_static(256), 0);
  EXPECT_EQ(print_type(type), "view<256xf32>");
}

TEST(PrintType, Buffer) { EXPECT_EQ(print_type(loom_type_buffer()), "buffer"); }

TEST(PrintType, GroupWorkgroup) {
  loom_type_t type;
  memset(&type, 0, sizeof(type));
  type.header = loom_type_make_raw_header(LOOM_TYPE_GROUP,
                                          LOOM_GROUP_SCOPE_WORKGROUP, 0, 0);
  EXPECT_EQ(print_type(type), "group<workgroup>");
}

TEST(PrintType, GroupSubgroup) {
  loom_type_t type;
  memset(&type, 0, sizeof(type));
  type.header = loom_type_make_raw_header(LOOM_TYPE_GROUP,
                                          LOOM_GROUP_SCOPE_SUBGROUP, 0, 0);
  EXPECT_EQ(print_type(type), "group<subgroup>");
}

TEST(PrintType, NoneType) {
  loom_type_t type;
  memset(&type, 0, sizeof(type));
  EXPECT_EQ(print_type(type), "none");
}

TEST(PrintType, TileLargeStatic) {
  loom_type_t type = loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F16,
                                         loom_dim_pack_static(256),
                                         loom_dim_pack_static(256), 0);
  EXPECT_EQ(print_type(type), "tile<256x256xf16>");
}

TEST(DimPacking, DynamicRoundTrip) {
  uint64_t packed = loom_dim_pack_dynamic(42);
  EXPECT_TRUE(loom_dim_is_dynamic(packed));
  EXPECT_EQ(loom_dim_value_id(packed), 42u);
}

TEST(DimPacking, StaticNotDynamic) {
  uint64_t packed = loom_dim_pack_static(256);
  EXPECT_FALSE(loom_dim_is_dynamic(packed));
  EXPECT_EQ(loom_dim_static_size(packed), 256);
}

// RAII wrapper for function types in tests. Builds via
// loom_type_function_build and frees the backing data on destruction.
class FuncType {
 public:
  FuncType(std::initializer_list<loom_type_t> args,
           std::initializer_list<loom_type_t> results) {
    IREE_CHECK_OK(loom_type_function_build(
        args.begin(), (uint16_t)args.size(), results.begin(),
        (uint16_t)results.size(), iree_allocator_system(), &type_));
  }
  ~FuncType() {
    iree_allocator_free(iree_allocator_system(),
                        (void*)(uintptr_t)type_.dims[0]);
  }
  operator loom_type_t() const { return type_; }

 private:
  loom_type_t type_;
};

TEST(PrintType, FunctionType) {
  FuncType type({loom_type_scalar(LOOM_SCALAR_TYPE_F32),
                 loom_type_scalar(LOOM_SCALAR_TYPE_I32)},
                {loom_type_scalar(LOOM_SCALAR_TYPE_F64)});
  EXPECT_EQ(print_type(type), "(f32, i32) -> (f64)");
}

TEST(PrintType, FunctionTypeNoArgs) {
  FuncType type({}, {loom_type_scalar(LOOM_SCALAR_TYPE_F32)});
  EXPECT_EQ(print_type(type), "() -> (f32)");
}

TEST(PrintType, FunctionTypeNoResults) {
  FuncType type({loom_type_scalar(LOOM_SCALAR_TYPE_I32)}, {});
  EXPECT_EQ(print_type(type), "(i32) -> ()");
}

TEST(PrintType, FunctionTypeEmpty) {
  FuncType type({}, {});
  EXPECT_EQ(print_type(type), "() -> ()");
}

TEST(PrintType, FunctionTypeMultipleResults) {
  FuncType type({loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)},
                {loom_type_scalar(LOOM_SCALAR_TYPE_F32),
                 loom_type_scalar(LOOM_SCALAR_TYPE_I32)});
  EXPECT_EQ(print_type(type), "(index) -> (f32, i32)");
}

TEST(PrintType, FunctionTypeWithShapedTypes) {
  FuncType type({loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                     loom_dim_pack_static(4), 0)},
                {loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_I8,
                                     loom_dim_pack_static(256), 0)});
  EXPECT_EQ(print_type(type), "(tile<4xf32>) -> (tensor<256xi8>)");
}

TEST(PrintType, EncodingType) {
  EXPECT_EQ(print_type(loom_type_encoding()), "encoding");
  EXPECT_EQ(print_type(loom_type_encoding_with_role(
                LOOM_ENCODING_ROLE_ADDRESS_LAYOUT)),
            "encoding<layout>");
  EXPECT_EQ(print_type(loom_type_encoding_with_role(
                LOOM_ENCODING_ROLE_STORAGE_SCHEMA)),
            "encoding<schema>");
  EXPECT_EQ(print_type(loom_type_encoding_with_role(
                LOOM_ENCODING_ROLE_PHYSICAL_STORAGE)),
            "encoding<storage>");
  EXPECT_EQ(print_type(loom_type_encoding_with_role(
                LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM)),
            "encoding<transform>");
}

TEST(PrintType, PoolStaticBlockSize) {
  loom_type_t type = loom_type_pool(loom_dim_pack_static(65536));
  EXPECT_EQ(print_type(type), "pool<65536>");
}

TEST(PrintType, PoolStaticSmallBlockSize) {
  loom_type_t type = loom_type_pool(loom_dim_pack_static(4096));
  EXPECT_EQ(print_type(type), "pool<4096>");
}

TEST(PrintType, PoolDynamicBlockSize) {
  // Dynamic dims without module context print as [%?].
  loom_type_t type = loom_type_pool(loom_dim_pack_dynamic(42));
  EXPECT_EQ(print_type(type), "pool<[%?]>");
}

TEST(PrintType, PoolTypeKindCheck) {
  loom_type_t type = loom_type_pool(loom_dim_pack_static(4096));
  EXPECT_TRUE(loom_type_is_pool(type));
  EXPECT_FALSE(loom_type_is_tile(type));
  EXPECT_FALSE(loom_type_is_tensor(type));
  EXPECT_FALSE(loom_type_is_scalar(type));
}

TEST(PrintType, PoolDimAccessor) {
  loom_type_t type = loom_type_pool(loom_dim_pack_static(65536));
  EXPECT_EQ(loom_type_rank(type), 1);
  EXPECT_FALSE(loom_type_dim_is_dynamic_at(type, 0));
  EXPECT_EQ(loom_type_dim_static_size_at(type, 0), 65536);
}

TEST(PrintType, PoolDynamicDimAccessor) {
  loom_type_t type = loom_type_pool(loom_dim_pack_dynamic(7));
  EXPECT_EQ(loom_type_rank(type), 1);
  EXPECT_TRUE(loom_type_dim_is_dynamic_at(type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(type, 0), 7u);
}

TEST(PrintType, UnknownTypeKindErrors) {
  loom_type_t type;
  memset(&type, 0, sizeof(type));
  type.header =
      loom_type_make_header((loom_type_kind_t)99, (loom_scalar_type_t)0, 0, 0);
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_text_print_type(type, nullptr, &stream));
  iree_string_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Op printing
//===----------------------------------------------------------------------===//

class PrintOpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t test_count = 0;
    const loom_op_vtable_t* const* test_vtables =
        loom_test_dialect_vtables(&test_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, test_vtables, (uint16_t)test_count));

    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kQ8_0EncodingVtable));
    IREE_ASSERT_OK(loom_context_register_encoding_vtable(
        &context_, &kDenseEncodingVtable));

    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  std::string print_op(loom_op_t* op, loom_text_print_flags_t flags) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    iree_status_t status =
        loom_text_print_operation_to_builder(module_, op, &builder, flags);
    std::string result;
    if (iree_status_is_ok(status)) {
      result = std::string(iree_string_builder_buffer(&builder),
                           iree_string_builder_size(&builder));
    }
    IREE_EXPECT_OK(status);
    iree_string_builder_deinitialize(&builder);
    return result;
  }

  std::string PrintOpWithFields(loom_op_t* op, loom_text_print_flags_t flags,
                                std::vector<CapturedPrintField>* fields) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_print_field_callback_t callback = {
        /*.fn=*/CapturePrintFieldCallback,
        /*.user_data=*/fields,
    };
    iree_status_t status = loom_text_print_operation_with_field_callback(
        module_, op, &builder, flags, callback);
    std::string result;
    if (iree_status_is_ok(status)) {
      result = std::string(iree_string_builder_buffer(&builder),
                           iree_string_builder_size(&builder));
    }
    IREE_EXPECT_OK(status);
    iree_string_builder_deinitialize(&builder);
    return result;
  }

  // Returns the status from printing without consuming it (for error tests).
  iree_status_t print_op_status(loom_op_t* op, loom_text_print_flags_t flags) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    iree_status_t status =
        loom_text_print_operation_to_builder(module_, op, &builder, flags);
    iree_string_builder_deinitialize(&builder);
    return status;
  }

  // Defines a value in the module and returns its ID.
  loom_value_id_t def(loom_type_t type) {
    loom_value_id_t id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(module_, type, &id));
    return id;
  }

  // Interns a string and returns its ID.
  loom_string_id_t intern(const char* str) {
    loom_string_id_t id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_module_intern_string(module_, iree_make_cstring_view(str), &id));
    return id;
  }

  // Assigns a display name to a value.
  void set_value_name(loom_value_id_t value, const char* name) {
    IREE_CHECK_OK(loom_module_set_value_name(module_, value, intern(name)));
  }

  // Creates a symbol with the given name and returns a local reference.
  loom_symbol_ref_t make_symbol(const char* name) {
    loom_string_id_t name_id = intern(name);
    uint16_t symbol_id = (uint16_t)module_->symbols.count;
    EXPECT_LT(symbol_id, (uint16_t)module_->symbols.capacity);
    loom_symbol_t* symbol = &module_->symbols.entries[symbol_id];
    memset(symbol, 0, sizeof(*symbol));
    symbol->name_id = name_id;
    module_->symbols.count++;
    return {0, symbol_id};
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(PrintOpTest, PrintRegisterType) {
  loom_type_t type = loom_low_register_type(
      /*descriptor_set_stable_id=*/0x2a, /*register_class_id=*/7,
      /*unit_count=*/4);
  EXPECT_EQ(print_type(type, module_), "reg<0x2a:7 x4>");
}

//===----------------------------------------------------------------------===//
// Simple ops (no regions)
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, BinaryOp) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t lhs = def(i32);
  loom_value_id_t rhs = def(i32);

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, lhs, rhs, i32,
                                      LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%2 = test.addi %0, %1 : i32\n");
}

TEST_F(PrintOpTest, UnaryOp) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, input, f32, LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT), "%1 = test.neg %0 : f32\n");
}

TEST_F(PrintOpTest, CastOp) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(i32);

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_cast_build(&builder_, input, i32, f32,
                                      LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%1 = test.cast %0 : i32 to f32\n");
}

TEST_F(PrintOpTest, ConstantI64) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%0 = test.constant 42 : i32\n");
}

TEST_F(PrintOpTest, ConstantNegative) {
  loom_type_t i64 = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(-1), i64,
                                          LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%0 = test.constant -1 : i64\n");
}

TEST_F(PrintOpTest, ConstantZero) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder_, loom_attr_i64(0), index_type, LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%0 = test.constant 0 : index\n");
}

TEST_F(PrintOpTest, ConstantSpecialFloatValues) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_op_t* nan_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_f64(NAN), f32,
                                          LOOM_LOCATION_UNKNOWN, &nan_op));
  EXPECT_EQ(print_op(nan_op, LOOM_TEXT_PRINT_DEFAULT),
            "%0 = test.constant nan : f32\n");

  loom_op_t* inf_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_f64(INFINITY),
                                          f32, LOOM_LOCATION_UNKNOWN, &inf_op));
  EXPECT_EQ(print_op(inf_op, LOOM_TEXT_PRINT_DEFAULT),
            "%1 = test.constant inf : f32\n");

  loom_op_t* negative_inf_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_f64(-INFINITY),
                                          f32, LOOM_LOCATION_UNKNOWN,
                                          &negative_inf_op));
  EXPECT_EQ(print_op(negative_inf_op, LOOM_TEXT_PRINT_DEFAULT),
            "%2 = test.constant -inf : f32\n");
}

TEST_F(PrintOpTest, ComparisonOp) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t lhs = def(i32);
  loom_value_id_t rhs = def(i32);

  loom_op_t* op = NULL;
  // predicate=0 → "eq" from the enum case name table.
  IREE_ASSERT_OK(loom_test_cmp_build(&builder_, 0, lhs, rhs, i32, i32,
                                     LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%2 = test.cmp eq, %0, %1 : i32\n");
}

//===----------------------------------------------------------------------===//
// Variadic ops
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, YieldSingleOperand) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t value = def(f32);

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(
      loom_test_yield_build(&builder_, &value, 1, LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT), "test.yield %0 : f32\n");
}

TEST_F(PrintOpTest, YieldMultipleOperands) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t values[] = {def(f32), def(i32)};

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(
      loom_test_yield_build(&builder_, values, 2, LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "test.yield %0, %1 : f32, i32\n");
}

TEST_F(PrintOpTest, YieldNoOperands) {
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(
      loom_test_yield_build(&builder_, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  // Zero operands: the OptionalGroup (Refs, COLON, TypesOf) is absent,
  // so the op name stands alone.
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT), "test.yield\n");
}

TEST_F(PrintOpTest, SegmentedOperands) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t root = def(i32);
  loom_value_id_t guard = def(i32);
  loom_value_id_t lhs0 = def(i32);
  loom_value_id_t lhs1 = def(i32);
  loom_value_id_t rhs = def(i32);
  set_value_name(root, "root");
  set_value_name(guard, "guard");
  set_value_name(lhs0, "lhs0");
  set_value_name(lhs1, "lhs1");
  set_value_name(rhs, "rhs");

  loom_value_id_t lhs[] = {lhs0, lhs1};
  loom_value_id_t rhs_values[] = {rhs};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_segmented_build(
      &builder_, LOOM_TEST_SEGMENTED_BUILD_FLAG_HAS_GUARD, root, guard, lhs,
      IREE_ARRAYSIZE(lhs), rhs_values, IREE_ARRAYSIZE(rhs_values), i32,
      LOOM_LOCATION_UNKNOWN, &op));
  set_value_name(loom_test_segmented_result(op), "result");

  std::vector<CapturedPrintField> fields;
  std::string output = PrintOpWithFields(op, LOOM_TEXT_PRINT_DEFAULT, &fields);
  EXPECT_EQ(output,
            "%result = test.segmented %root base %guard values %lhs0, %lhs1 "
            "expected %rhs : i32 -> i32\n");

  const CapturedPrintField* root_field =
      FindCapturedPrintField(fields, LOOM_PRINT_FIELD_OPERAND, 0);
  ASSERT_NE(root_field, nullptr);
  EXPECT_EQ(
      output.substr(root_field->start, root_field->end - root_field->start),
      "%root");
  const CapturedPrintField* lhs1_field =
      FindCapturedPrintField(fields, LOOM_PRINT_FIELD_OPERAND, 3);
  ASSERT_NE(lhs1_field, nullptr);
  EXPECT_EQ(
      output.substr(lhs1_field->start, lhs1_field->end - lhs1_field->start),
      "%lhs1");
}

TEST_F(PrintOpTest, SegmentedOperandsAbsentOptionalAndEmptySpan) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t root = def(i32);
  loom_value_id_t rhs0 = def(i32);
  loom_value_id_t rhs1 = def(i32);
  set_value_name(root, "root");
  set_value_name(rhs0, "rhs0");
  set_value_name(rhs1, "rhs1");

  loom_value_id_t rhs[] = {rhs0, rhs1};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_segmented_build(
      &builder_, 0, root, LOOM_VALUE_ID_INVALID, NULL, 0, rhs,
      IREE_ARRAYSIZE(rhs), i32, LOOM_LOCATION_UNKNOWN, &op));
  set_value_name(loom_test_segmented_result(op), "result");

  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%result = test.segmented %root values expected %rhs0, %rhs1 : "
            "i32 -> i32\n");
}

TEST_F(PrintOpTest, SuccessorReferenceSynthesizesBlockLabels) {
  loom_symbol_ref_t callee = make_symbol("cfg");

  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, callee, NULL, 0, NULL,
                                      0, NULL, 0, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &func_op));

  loom_region_t* body = loom_test_func_body(func_op);
  ASSERT_NE(body, nullptr);
  loom_block_t* exit_block = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &exit_block));
  ASSERT_NE(exit_block, nullptr);

  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, func_op, body);
  loom_op_t* branch_op = NULL;
  IREE_ASSERT_OK(loom_test_br_build(&builder_, exit_block,
                                    LOOM_LOCATION_UNKNOWN, &branch_op));
  loom_builder_restore(&builder_, saved);
  ASSERT_EQ(loom_test_br_dest(branch_op), exit_block);

  std::vector<CapturedPrintField> fields;
  std::string output =
      PrintOpWithFields(func_op, LOOM_TEXT_PRINT_DEFAULT, &fields);
  EXPECT_EQ(output,
            "test.func @cfg() {\n"
            "  test.br ^_bb1\n"
            "^_bb1:\n"
            "}\n");

  const CapturedPrintField* successor_field =
      FindCapturedPrintField(fields, LOOM_PRINT_FIELD_SUCCESSOR, 0);
  ASSERT_NE(successor_field, nullptr);
  EXPECT_EQ(output.substr(successor_field->start,
                          successor_field->end - successor_field->start),
            "^_bb1");
}

TEST_F(PrintOpTest, SyntheticEntryBlockLabelOmitsParentDeclaredArgs) {
  loom_symbol_ref_t callee = make_symbol("cycle");
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_type_t arg_types[] = {f32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, callee, arg_types, 1,
                                      NULL, 0, NULL, 0, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &func_op));

  loom_region_t* body = loom_test_func_body(func_op);
  ASSERT_NE(body, nullptr);
  loom_block_t* entry_block = loom_region_entry_block(body);
  ASSERT_NE(entry_block, nullptr);
  loom_block_t* backedge_block = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &backedge_block));
  ASSERT_NE(backedge_block, nullptr);

  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, func_op, body);
  loom_builder_set_block(&builder_, backedge_block);
  loom_op_t* branch_op = NULL;
  IREE_ASSERT_OK(loom_test_br_build(&builder_, entry_block,
                                    LOOM_LOCATION_UNKNOWN, &branch_op));
  loom_builder_restore(&builder_, saved);

  EXPECT_EQ(print_op(func_op, LOOM_TEXT_PRINT_DEFAULT),
            "test.func @cycle(%0: f32) {\n"
            "^_bb0:\n"
            "^_bb1:\n"
            "  test.br ^_bb0\n"
            "}\n");
}

//===----------------------------------------------------------------------===//
// Index list ops
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, SliceAllStatic) {
  loom_type_t tile_type = loom_type_shaped_2d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F16, loom_dim_pack_static(64),
      loom_dim_pack_static(64), 0);
  loom_type_t result_type = loom_type_shaped_2d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F16, loom_dim_pack_static(16),
      loom_dim_pack_static(16), 0);
  loom_value_id_t source = def(tile_type);

  int64_t static_offsets[] = {0, 32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_slice_build(&builder_, source, NULL, 0,
                                       static_offsets, 2, result_type, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(
      print_op(op, LOOM_TEXT_PRINT_DEFAULT),
      "%1 = test.slice %0[0, 32] : tile<64x64xf16> -> (tile<16x16xf16>)\n");
}

TEST_F(PrintOpTest, SliceMixedStaticDynamic) {
  loom_type_t tile_type = loom_type_shaped_2d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F16, loom_dim_pack_static(64),
      loom_dim_pack_static(64), 0);
  loom_type_t result_type = loom_type_shaped_2d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F16, loom_dim_pack_static(16),
      loom_dim_pack_static(16), 0);
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t source = def(tile_type);
  loom_value_id_t offset = def(index_type);

  // INT64_MIN = dynamic sentinel, 0 = static.
  int64_t static_offsets[] = {INT64_MIN, 0};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_slice_build(&builder_, source, &offset, 1,
                                       static_offsets, 2, result_type, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(
      print_op(op, LOOM_TEXT_PRINT_DEFAULT),
      "%2 = test.slice %0[%1, 0] : tile<64x64xf16> -> (tile<16x16xf16>)\n");
}

TEST_F(PrintOpTest, UpdateWithTiedResult) {
  loom_type_t tile_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t tensor_type = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(256), 0);
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t source = def(tile_type);
  loom_value_id_t target = def(tensor_type);
  loom_value_id_t offset = def(index_type);

  // Build manually to test printer with specific tied result layout.
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_builder_allocate_op(
      &builder_, LOOM_OP_TEST_UPDATE,
      /*operand_count=*/3, /*result_count=*/1,
      /*region_count=*/0, /*tied_result_count=*/1,
      /*attribute_count=*/1, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = source;
  loom_op_operands(op)[1] = target;
  loom_op_operands(op)[2] = offset;
  int64_t static_offsets[] = {INT64_MIN};
  loom_op_attrs(op)[0] = loom_attr_i64_array(static_offsets, 1);
  loom_value_id_t result_id = def(tensor_type);
  loom_op_results(op)[0] = result_id;
  loom_tied_result_t tied = {0, 1};
  loom_op_tied_results(op)[0] = tied;

  EXPECT_EQ(
      print_op(op, LOOM_TEXT_PRINT_DEFAULT),
      "%3 = test.update %0, %1[%2] : tile<4xf32> -> (%1 as tensor<256xf32>)\n");
}

//===----------------------------------------------------------------------===//
// Symbol ref and glue
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, InvokeWithSymbolRef) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t input = def(f32);

  loom_symbol_ref_t callee = make_symbol("my_func");
  loom_type_t result_types[] = {index_type};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_invoke_build(&builder_, callee, &input, 1,
                                        result_types, 1, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &op));
  // Format: @callee(%input) : (f32) -> (index)
  // The Glue element suppresses the space between @callee and (.
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%1 = test.invoke @my_func(%0) : (f32) -> (index)\n");
}

TEST_F(PrintOpTest, InvokeMultipleOperandsAndResults) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t inputs[] = {def(f32), def(index_type)};

  loom_symbol_ref_t callee = make_symbol("compute");
  loom_type_t result_types[] = {f32, index_type};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_invoke_build(&builder_, callee, inputs, 2,
                                        result_types, 2, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(
      print_op(op, LOOM_TEXT_PRINT_DEFAULT),
      "%2, %3 = test.invoke @compute(%0, %1) : (f32, index) -> (f32, index)\n");
}

//===----------------------------------------------------------------------===//
// Region ops
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, BranchEmptyRegions) {
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t condition = def(i1);

  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  // Both regions are always auto-created by the builder.
  IREE_ASSERT_OK(loom_test_branch_build(&builder_, condition, result_types, 1,
                                        NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_region_t* then_region = loom_test_branch_then_region(op);
  loom_region_t* else_region = loom_test_branch_else_region(op);
  ASSERT_NE(then_region, nullptr);
  ASSERT_NE(else_region, nullptr);
  ASSERT_EQ(loom_region_entry_block(then_region)->op_count, 0u);
  ASSERT_EQ(loom_region_entry_block(else_region)->op_count, 0u);
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%1 = test.branch %0 -> (f32) {\n"
            "} else {\n"
            "}\n");
}

TEST_F(PrintOpTest, BranchThenElse) {
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t condition = def(i1);

  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_branch_build(&builder_, condition, result_types, 1,
                                        NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%1 = test.branch %0 -> (f32) {\n"
            "} else {\n"
            "}\n");
}

TEST_F(PrintOpTest, BranchWithBodyOps) {
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t condition = def(i1);

  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_branch_build(&builder_, condition, result_types, 1,
                                        NULL, 0, LOOM_LOCATION_UNKNOWN, &op));

  // Add an op inside the auto-created then region.
  loom_value_id_t const_val = def(f32);
  loom_op_t* yield_op = NULL;
  loom_region_t* then_region = loom_test_branch_then_region(op);
  loom_builder_ip_t saved =
      loom_builder_enter_region(&builder_, op, then_region);
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &const_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved);

  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%1 = test.branch %0 -> (f32) {\n"
            "  test.yield %2 : f32\n"
            "} else {\n"
            "}\n");
}

TEST_F(PrintOpTest, MapWithBindingList) {
  loom_type_t tile_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_value_id_t input = def(tile_type);

  loom_op_t* op = NULL;
  // Builder auto-creates body region with one block arg per input
  // (element type derived from the input tile type).
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &input, 1, tile_type, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));
  // BindingList(element): (%block_arg = %operand : type)
  // The block arg is value %1 (element), operand is %0 (input).
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%2 = test.map(%1 = %0 : tile<4xf32>) {\n"
            "} -> (tile<4xf32>)\n");
}

TEST_F(PrintOpTest, LoopWithIterArgs) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t lower = def(index_type);
  loom_value_id_t upper = def(index_type);
  loom_value_id_t step = def(index_type);
  loom_value_id_t init = def(f32);

  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  // Builder auto-creates body region with IV block arg (index) and one
  // block arg per iter_arg (type looked up from value table).
  IREE_ASSERT_OK(loom_test_loop_build(&builder_, lower, upper, step, &init, 1,
                                      result_types, 1, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &op));
  // IV is block arg 0 (%4), accumulator is block arg 1 (%5).
  // OperandRef(255) prints the IV.
  // BindingList(capture, start=3): (%5 = %3 : f32)
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%6 = test.loop %4 = %0 to %1 step %2 iter_args(%5 = %3 : f32) -> "
            "(f32) {\n"
            "}\n");
}

TEST_F(PrintOpTest, LoopWithoutIterArgs) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t lower = def(index_type);
  loom_value_id_t upper = def(index_type);
  loom_value_id_t step = def(index_type);

  loom_op_t* op = NULL;
  // Builder auto-creates body region with IV block arg only (no iter_args).
  IREE_ASSERT_OK(loom_test_loop_build(&builder_, lower, upper, step, NULL, 0,
                                      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN,
                                      &op));
  // No iter_args, no results: both OptionalGroups skip their content.
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "test.loop %3 = %0 to %1 step %2 {\n"
            "}\n");
}

//===----------------------------------------------------------------------===//
// Function-like ops
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, FuncDeclaration) {
  loom_symbol_ref_t callee = make_symbol("identity");
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  // No arg_types (NULL, 0). Body region is auto-created (empty).
  IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, callee, NULL, 0,
                                      result_types, 1, NULL, 0, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &op));
  // SYMBOL_DEFINE trait: no LHS results. Optional visibility/cc absent
  // because no build flags were set. Body always present.
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "test.func @identity() -> (f32) {\n"
            "}\n");
}

TEST_F(PrintOpTest, FuncWithBody) {
  loom_symbol_ref_t callee = make_symbol("negate");
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_type_t arg_types[] = {f32};
  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  // Builder auto-creates body region with block args from arg_types.
  IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, callee, arg_types, 1,
                                      result_types, 1, NULL, 0, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &op));
  // Body present, one arg. FuncArgs prints (%block_arg: f32).
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "test.func @negate(%0: f32) -> (f32) {\n"
            "}\n");
}

TEST_F(PrintOpTest, DuplicateExplicitValueNamesPrintUniquely) {
  loom_symbol_ref_t callee = make_symbol("dupes");
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_type_t arg_types[] = {f32, f32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, callee, arg_types, 2,
                                      NULL, 0, NULL, 0, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &func_op));
  loom_region_t* body = loom_test_func_body(func_op);
  ASSERT_NE(body, nullptr);
  loom_block_t* entry_block = loom_region_entry_block(body);
  ASSERT_NE(entry_block, nullptr);
  set_value_name(loom_block_arg_id(entry_block, 0), "input");
  set_value_name(loom_block_arg_id(entry_block, 1), "input");

  EXPECT_EQ(print_op(func_op, LOOM_TEXT_PRINT_DEFAULT),
            "test.func @dupes(%input$0: f32, %input$1: f32) {\n"
            "}\n");
}

TEST_F(PrintOpTest, DuplicateExplicitValueNamesInSiblingRegionsArePreserved) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t arg_types[] = {f32};

  loom_op_t* first_func = NULL;
  IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, make_symbol("first"),
                                      arg_types, 1, NULL, 0, NULL, 0, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &first_func));
  loom_region_t* first_body = loom_test_func_body(first_func);
  ASSERT_NE(first_body, nullptr);
  set_value_name(loom_block_arg_id(loom_region_entry_block(first_body), 0),
                 "input");

  loom_op_t* second_func = NULL;
  IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, make_symbol("second"),
                                      arg_types, 1, NULL, 0, NULL, 0, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &second_func));
  loom_region_t* second_body = loom_test_func_body(second_func);
  ASSERT_NE(second_body, nullptr);
  set_value_name(loom_block_arg_id(loom_region_entry_block(second_body), 0),
                 "input");

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_text_print_module_to_builder(module_, &builder,
                                                   LOOM_TEXT_PRINT_DEFAULT));
  std::string output(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_EQ(output,
            "test.func @first(%input: f32) {\n"
            "}\n"
            "\n"
            "test.func @second(%input: f32) {\n"
            "}\n");
}

TEST_F(PrintOpTest, GeneratedValueNameAvoidsExplicitNumericName) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* explicit_numeric_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(7), i32,
                                          LOOM_LOCATION_UNKNOWN,
                                          &explicit_numeric_op));
  set_value_name(loom_test_constant_result(explicit_numeric_op), "1");
  loom_op_t* generated_numeric_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(8), i32,
                                          LOOM_LOCATION_UNKNOWN,
                                          &generated_numeric_op));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_text_print_module_to_builder(module_, &builder,
                                                   LOOM_TEXT_PRINT_DEFAULT));
  std::string output(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_EQ(output,
            "%1 = test.constant 7 : i32\n"
            "%1$1 = test.constant 8 : i32\n");
}

TEST_F(PrintOpTest, FuncWithVisibility) {
  loom_symbol_ref_t callee = make_symbol("main");
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_type_t arg_types[] = {f32};
  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  // Visibility present (enum value 1 = "public"), cc absent.
  IREE_ASSERT_OK(
      loom_test_func_build(&builder_, LOOM_TEST_FUNC_BUILD_FLAG_HAS_VISIBILITY,
                           1, 0, callee, arg_types, 1, result_types, 1, NULL, 0,
                           NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "test.func public @main(%0: f32) -> (f32) {\n"
            "}\n");
}

TEST_F(PrintOpTest, FuncDefTiedResultPrintsEntryArgName) {
  loom_symbol_ref_t callee = make_symbol("identity");
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_type_t arg_types[] = {f32};
  loom_type_t result_types[] = {f32};
  loom_tied_result_t tied_results[] = {
      {/*.result_index=*/0, /*.operand_index=*/0, /*.has_type_change=*/false},
  };
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder_, 0, 0, 0, callee, arg_types, 1, result_types, 1, tied_results,
      IREE_ARRAYSIZE(tied_results), NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "test.func @identity(%0: f32) -> (%0 as f32) {\n"
            "}\n");
}

TEST_F(PrintOpTest, FuncDeclTiedResultPrintsArgOperandName) {
  loom_symbol_ref_t callee = make_symbol("identity");
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_type_t arg_types[] = {f32};
  loom_type_t result_types[] = {f32};
  loom_tied_result_t tied_results[] = {
      {/*.result_index=*/0, /*.operand_index=*/0, /*.has_type_change=*/false},
  };
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_decl_build(
      &builder_, 0, 0, 0, callee, arg_types, 1, result_types, 1, tied_results,
      IREE_ARRAYSIZE(tied_results), LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "test.decl @identity(%0: f32) -> (%0 as f32)\n");
}

//===----------------------------------------------------------------------===//
// Attr dict
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, AttrsOpEmptyDict) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, input,
                                       loom_make_named_attr_slice(NULL, 0), f32,
                                       LOOM_LOCATION_UNKNOWN, &op));
  // The optional dict builder leaves an empty slice absent.
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%1 = test.attrs %0 : f32\n");
}

TEST_F(PrintOpTest, AttrsOpExplicitEmptyDictCanonicalizesToAbsent) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, input,
                                       loom_make_named_attr_slice(NULL, 0), f32,
                                       LOOM_LOCATION_UNKNOWN, &op));
  loom_op_attrs(op)[0] = loom_make_canonical_attr_dict(NULL, 0);

  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%1 = test.attrs %0 : f32\n");
}

TEST_F(PrintOpTest, AttrsOpWithDictEntries) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);

  loom_string_id_t axis_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t label_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t foo_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("axis"), &axis_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("label"), &label_id));
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("foo"), &foo_id));

  loom_named_attr_t entries[2] = {
      {/*.name_id=*/axis_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(0)},
      {/*.name_id=*/label_id, /*.reserved=*/{},
       /*.value=*/loom_attr_string(foo_id)},
  };
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &builder_, input,
      loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)), f32,
      LOOM_LOCATION_UNKNOWN, &op));

  std::string output = print_op(op, LOOM_TEXT_PRINT_DEFAULT);
  EXPECT_NE(output.find("{axis = 0, label = \"foo\"}"), std::string::npos)
      << "Expected attr dict in output, got: " << output;
}

TEST_F(PrintOpTest, AttrsOpStringAttrsUseCanonicalEscapes) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);

  loom_string_id_t label_id = intern("label");
  loom_string_id_t value_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(
      module_, IREE_SV("has \"quotes\" and \\slashes\\\n\t\b\f\r and \xCE\xBB"),
      &value_id));

  loom_named_attr_t entries[1] = {
      {/*.name_id=*/label_id, /*.reserved=*/{},
       /*.value=*/loom_attr_string(value_id)},
  };
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &builder_, input,
      loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)), f32,
      LOOM_LOCATION_UNKNOWN, &op));

  std::string output = print_op(op, LOOM_TEXT_PRINT_DEFAULT);
  EXPECT_NE(output.find("{label = \"has \\\"quotes\\\" and \\\\slashes\\\\"
                        "\\n\\t\\b\\f\\r and \xCE\xBB\"}"),
            std::string::npos)
      << "Expected canonical string escapes, got: " << output;
}

TEST_F(PrintOpTest, AttrsOpStringAttrsRejectInvalidUtf8) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);

  loom_string_id_t label_id = intern("label");
  loom_string_id_t invalid_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("\xFF"), &invalid_id));

  loom_named_attr_t entries[1] = {
      {/*.name_id=*/label_id, /*.reserved=*/{},
       /*.value=*/loom_attr_string(invalid_id)},
  };
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &builder_, input,
      loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)), f32,
      LOOM_LOCATION_UNKNOWN, &op));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        print_op_status(op, LOOM_TEXT_PRINT_DEFAULT));
}

TEST_F(PrintOpTest, AttrsOpWithNestedDictEntries) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);

  loom_string_id_t axis_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t phase_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t empty_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t alpha_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t zeta_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("axis"), &axis_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("phase"), &phase_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("empty"), &empty_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("alpha"), &alpha_id));
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("zeta"), &zeta_id));

  loom_named_attr_t nested_entries[2] = {
      {/*.name_id=*/zeta_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(2)},
      {/*.name_id=*/alpha_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(1)},
  };
  loom_attribute_t nested_dict = {0};
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module_,
      loom_make_named_attr_slice(nested_entries,
                                 IREE_ARRAYSIZE(nested_entries)),
      &nested_dict));

  loom_named_attr_t entries[3] = {
      {/*.name_id=*/phase_id, /*.reserved=*/{}, /*.value=*/nested_dict},
      {/*.name_id=*/axis_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(0)},
      {/*.name_id=*/empty_id,
       /*.reserved=*/{},
       /*.value=*/loom_make_canonical_attr_dict(/*entries=*/NULL, /*count=*/0)},
  };
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &builder_, input,
      loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)), f32,
      LOOM_LOCATION_UNKNOWN, &op));

  std::string output = print_op(op, LOOM_TEXT_PRINT_DEFAULT);
  EXPECT_NE(
      output.find("{axis = 0, empty = {}, phase = {alpha = 1, zeta = 2}}"),
      std::string::npos)
      << "Expected nested attr dict in output, got: " << output;
}

TEST_F(PrintOpTest,
       AttrsOpWithMalformedNonEmptyNullDictReturnsInvalidArgument) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, input,
                                       loom_make_named_attr_slice(NULL, 0), f32,
                                       LOOM_LOCATION_UNKNOWN, &op));

  loom_op_attrs(op)[0] =
      loom_make_canonical_attr_dict(/*entries=*/NULL, /*count=*/1);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        print_op_status(op, LOOM_TEXT_PRINT_DEFAULT));
}

TEST_F(PrintOpTest, AttrsOpWithWrongDictAttrKindReturnsInvalidArgument) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, input,
                                       loom_make_named_attr_slice(NULL, 0), f32,
                                       LOOM_LOCATION_UNKNOWN, &op));

  loom_op_attrs(op)[0] = loom_attr_i64(1);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        print_op_status(op, LOOM_TEXT_PRINT_DEFAULT));
}

//===----------------------------------------------------------------------===//
// Result dim references
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, DeflateWithResultDimReference) {
  // test.deflate with result 0's dim referencing result 1 (the length).
  // Uses loom_builder_reserve_results so the result value_ids are known
  // before constructing the result types.
  loom_type_t tensor_dyn = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(0), 0);
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t input = def(tensor_dyn);

  // Reserve 2 result value_ids so we can reference result[1] in
  // result[0]'s type.
  loom_value_id_t result_ids[2];
  IREE_ASSERT_OK(loom_builder_reserve_results(&builder_, 2, result_ids));
  loom_type_t tensor_ref =
      loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(result_ids[1]), 0);
  loom_type_t result_types[] = {tensor_ref, index_type};

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_deflate_build(&builder_, input, result_types, 2,
                                         NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  std::string output = print_op(op, LOOM_TEXT_PRINT_DEFAULT);
  EXPECT_EQ(output,
            "%1, %2 = test.deflate %0 : tensor<[%0]xf32>"
            " -> (tensor<[%2]xf32>, index)\n");
}

//===----------------------------------------------------------------------===//
// Spacing edge cases
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, SpacingCommaNoLeadingSpace) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t lhs = def(i32);
  loom_value_id_t rhs = def(i32);

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, lhs, rhs, i32,
                                      LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%2 = test.addi %0, %1 : i32\n");
}

TEST_F(PrintOpTest, SpacingColonHasLeadingSpace) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, input, f32, LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT), "%1 = test.neg %0 : f32\n");
}

TEST_F(PrintOpTest, SpacingIndexListGluesToPrecedingToken) {
  loom_type_t tile_type = loom_type_shaped_2d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F16, loom_dim_pack_static(64),
      loom_dim_pack_static(64), 0);
  loom_type_t result_type = loom_type_shaped_2d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F16, loom_dim_pack_static(16),
      loom_dim_pack_static(16), 0);
  loom_value_id_t source = def(tile_type);
  int64_t static_offsets[] = {0, 32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_slice_build(&builder_, source, NULL, 0,
                                       static_offsets, 2, result_type, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%1 = test.slice %0[0, 32] : tile<64x64xf16>"
            " -> (tile<16x16xf16>)\n");
}

TEST_F(PrintOpTest, SpacingParensGlue) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);
  loom_symbol_ref_t callee = make_symbol("f");
  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_invoke_build(&builder_, callee, &input, 1,
                                        result_types, 1, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%1 = test.invoke @f(%0) : (f32) -> (f32)\n");
}

TEST_F(PrintOpTest, SpacingGlueElementSuppressesSpace) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);
  loom_symbol_ref_t callee = make_symbol("g");
  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_invoke_build(&builder_, callee, &input, 1,
                                        result_types, 1, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%1 = test.invoke @g(%0) : (f32) -> (f32)\n");
}

TEST_F(PrintOpTest, SpacingResultTypeListParens) {
  loom_type_t tile_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_value_id_t source = def(tile_type);
  int64_t static_offsets[] = {0};
  loom_type_t result_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_slice_build(&builder_, source, NULL, 0,
                                       static_offsets, 1, result_type, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &op));
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%1 = test.slice %0[0] : tile<4xf32> -> (tile<4xf32>)\n");
}

//===----------------------------------------------------------------------===//
// Module printing
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, PrintModule) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t lhs = def(i32);
  loom_value_id_t rhs = def(i32);

  loom_op_t* op1 = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, lhs, rhs, i32,
                                      LOOM_LOCATION_UNKNOWN, &op1));
  loom_op_t* op2 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(7), i32,
                                          LOOM_LOCATION_UNKNOWN, &op2));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_text_print_module_to_builder(module_, &builder,
                                                   LOOM_TEXT_PRINT_DEFAULT));
  std::string output(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_EQ(output,
            "%2 = test.addi %0, %1 : i32\n"
            "%3 = test.constant 7 : i32\n");
}

//===----------------------------------------------------------------------===//
// Stream byte offset tracking
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, StreamOffsetTracking) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &op));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);

  IREE_ASSERT_OK(
      loom_text_print_operation(module_, op, &stream, LOOM_TEXT_PRINT_DEFAULT));
  // The stream offset should equal the number of bytes written.
  EXPECT_EQ(stream.offset, iree_string_builder_size(&builder));
  EXPECT_GT(stream.offset, (iree_host_size_t)0);

  iree_string_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Field callbacks
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, FieldCallbackPreservesWideVariadicOperandIndex) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  std::vector<loom_value_id_t> inputs;
  inputs.reserve(65);
  for (uint16_t i = 0; i < 65; ++i) {
    inputs.push_back(def(i32));
  }

  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_reduce_build(&builder_, inputs.data(), inputs.size(),
                                        i32, LOOM_LOCATION_UNKNOWN, &op));

  std::vector<CapturedPrintField> fields;
  std::string output = PrintOpWithFields(op, LOOM_TEXT_PRINT_DEFAULT, &fields);

  const CapturedPrintField* result_field =
      FindCapturedPrintField(fields, LOOM_PRINT_FIELD_RESULT, 0);
  ASSERT_NE(result_field, nullptr);
  EXPECT_EQ(output.substr(result_field->start,
                          result_field->end - result_field->start),
            "%65");

  const CapturedPrintField* operand_field =
      FindCapturedPrintField(fields, LOOM_PRINT_FIELD_OPERAND, 64);
  ASSERT_NE(operand_field, nullptr);
  EXPECT_EQ(output.substr(operand_field->start,
                          operand_field->end - operand_field->start),
            "%64");
}

TEST_F(PrintOpTest, FieldCallbackReportsAttrAndRegionSpans) {
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t condition = def(i1);
  loom_value_id_t input = def(f32);

  loom_string_id_t axis_id = intern("axis");
  loom_string_id_t label_id = intern("label");
  loom_string_id_t foo_id = intern("foo");
  loom_named_attr_t entries[2] = {
      {/*.name_id=*/axis_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(0)},
      {/*.name_id=*/label_id, /*.reserved=*/{},
       /*.value=*/loom_attr_string(foo_id)},
  };
  loom_op_t* attrs_op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &builder_, input,
      loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)), f32,
      LOOM_LOCATION_UNKNOWN, &attrs_op));

  std::vector<CapturedPrintField> attr_fields;
  std::string attrs_output =
      PrintOpWithFields(attrs_op, LOOM_TEXT_PRINT_DEFAULT, &attr_fields);

  const CapturedPrintField* attr_field =
      FindCapturedPrintField(attr_fields, LOOM_PRINT_FIELD_ATTR, 0);
  ASSERT_NE(attr_field, nullptr);
  EXPECT_EQ(attrs_output.substr(attr_field->start,
                                attr_field->end - attr_field->start),
            "{axis = 0, label = \"foo\"}");

  loom_type_t result_types[] = {f32};
  loom_op_t* branch_op = NULL;
  IREE_ASSERT_OK(loom_test_branch_build(&builder_, condition, result_types, 1,
                                        NULL, 0, LOOM_LOCATION_UNKNOWN,
                                        &branch_op));

  std::vector<CapturedPrintField> region_fields;
  std::string branch_output =
      PrintOpWithFields(branch_op, LOOM_TEXT_PRINT_DEFAULT, &region_fields);

  const CapturedPrintField* then_field =
      FindCapturedPrintField(region_fields, LOOM_PRINT_FIELD_REGION, 0);
  ASSERT_NE(then_field, nullptr);
  EXPECT_EQ(branch_output.substr(then_field->start,
                                 then_field->end - then_field->start),
            "{\n}");

  const CapturedPrintField* else_field =
      FindCapturedPrintField(region_fields, LOOM_PRINT_FIELD_REGION, 1);
  ASSERT_NE(else_field, nullptr);
  EXPECT_EQ(branch_output.substr(else_field->start,
                                 else_field->end - else_field->start),
            "{\n}");
}

//===----------------------------------------------------------------------===//
// Null stream (offset-only computation)
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, NullStreamDiscardsOutput) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &op));

  loom_output_stream_t stream;
  loom_output_stream_null(&stream);

  IREE_ASSERT_OK(
      loom_text_print_operation(module_, op, &stream, LOOM_TEXT_PRINT_DEFAULT));
  // Offset should still track bytes even though output is discarded.
  EXPECT_GT(stream.offset, (iree_host_size_t)0);
}

//===----------------------------------------------------------------------===//
// Nested region indentation
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, NestedBranchInBranch) {
  // Outer branch with an inner branch in the then body.
  // Tests 2-level indentation.
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t outer_cond = def(i1);
  loom_value_id_t inner_cond = def(i1);

  // Build outer branch — auto-creates then/else regions.
  loom_op_t* outer_branch = NULL;
  IREE_ASSERT_OK(loom_test_branch_build(&builder_, outer_cond, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &outer_branch));

  // Build inner branch inside the outer then region.
  loom_region_t* outer_then = loom_test_branch_then_region(outer_branch);
  loom_builder_ip_t saved =
      loom_builder_enter_region(&builder_, outer_branch, outer_then);
  loom_op_t* inner_branch = NULL;
  IREE_ASSERT_OK(loom_test_branch_build(&builder_, inner_cond, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &inner_branch));

  // Build a yield in the inner then body.
  loom_value_id_t yield_val = def(f32);
  loom_region_t* inner_then = loom_test_branch_then_region(inner_branch);
  loom_builder_ip_t saved2 =
      loom_builder_enter_region(&builder_, inner_branch, inner_then);
  loom_op_t* yield_op = NULL;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &yield_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved2);
  loom_builder_restore(&builder_, saved);

  EXPECT_EQ(print_op(outer_branch, LOOM_TEXT_PRINT_DEFAULT),
            "test.branch %0 {\n"
            "  test.branch %1 {\n"
            "    test.yield %2 : f32\n"
            "  } else {\n"
            "  }\n"
            "} else {\n"
            "}\n");
}

TEST_F(PrintOpTest, NestedLoopInFunc) {
  // Function containing a loop — tests indentation at func > loop > yield.
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_symbol_ref_t callee = make_symbol("compute");

  // Define operands that the loop will consume.
  loom_value_id_t lower = def(index_type);  // %0
  loom_value_id_t upper = def(index_type);  // %1
  loom_value_id_t step = def(index_type);   // %2
  loom_value_id_t yield_val = def(f32);     // %3

  // Build the func (top-down). Builder auto-creates body region
  // with one block arg from arg_types.
  loom_type_t arg_types[] = {f32};
  loom_type_t result_types[] = {f32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, callee, arg_types, 1,
                                      result_types, 1, NULL, 0, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &func_op));
  // func block arg = %4, func result = %5

  // Build the loop inside the func body.
  loom_region_t* func_body = loom_test_func_body(func_op);
  loom_builder_ip_t saved =
      loom_builder_enter_region(&builder_, func_op, func_body);
  loom_op_t* loop_op = NULL;
  IREE_ASSERT_OK(loom_test_loop_build(&builder_, lower, upper, step, NULL, 0,
                                      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN,
                                      &loop_op));
  // loop IV block arg = %6

  // Build a yield inside the loop body.
  loom_region_t* loop_body = loom_test_loop_body(loop_op);
  loom_builder_ip_t saved2 =
      loom_builder_enter_region(&builder_, loop_op, loop_body);
  loom_op_t* yield_op = NULL;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &yield_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved2);
  loom_builder_restore(&builder_, saved);

  // Value IDs: %0=lower, %1=upper, %2=step, %3=yield_val,
  // %4=func arg, %5=func result, %6=IV.
  EXPECT_EQ(print_op(func_op, LOOM_TEXT_PRINT_DEFAULT),
            "test.func @compute(%4: f32) -> (f32) {\n"
            "  test.loop %6 = %0 to %1 step %2 {\n"
            "    test.yield %3 : f32\n"
            "  }\n"
            "}\n");
}

TEST_F(PrintOpTest, BranchThenElseWithBodies) {
  // Branch with ops in both then and else regions.
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t condition = def(i1);
  loom_value_id_t true_val = def(f32);
  loom_value_id_t false_val = def(f32);

  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_branch_build(&builder_, condition, result_types, 1,
                                        NULL, 0, LOOM_LOCATION_UNKNOWN, &op));

  // Yield in then.
  loom_region_t* then_region = loom_test_branch_then_region(op);
  loom_builder_ip_t saved =
      loom_builder_enter_region(&builder_, op, then_region);
  loom_op_t* then_yield = NULL;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &true_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &then_yield));
  loom_builder_restore(&builder_, saved);

  // Yield in else.
  loom_region_t* else_region = loom_test_branch_else_region(op);
  saved = loom_builder_enter_region(&builder_, op, else_region);
  loom_op_t* else_yield = NULL;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &false_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &else_yield));
  loom_builder_restore(&builder_, saved);

  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "%3 = test.branch %0 -> (f32) {\n"
            "  test.yield %1 : f32\n"
            "} else {\n"
            "  test.yield %2 : f32\n"
            "}\n");
}

//===----------------------------------------------------------------------===//
// Block labels
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, BlockLabel) {
  // A region with two blocks: entry (no label) and a labeled block.
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_value_id_t condition = def(i1);

  // Build the branch op (auto-creates both regions with 1 block each).
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_branch_build(&builder_, condition, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &op));

  // Replace the auto-created then_region with a 2-block region.
  loom_region_t* then_region = NULL;
  IREE_ASSERT_OK(loom_module_allocate_region(module_, 2, &then_region));
  loom_block_t* second_block = loom_region_block(then_region, 1);
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("next"),
                                           &second_block->label_id));
  IREE_ASSERT_OK(loom_block_add_arg(module_, second_block, def(f32)));
  loom_op_regions(op)[0] = then_region;

  std::string output = print_op(op, LOOM_TEXT_PRINT_DEFAULT);
  EXPECT_EQ(output,
            "test.branch %0 {\n"
            "^next(%1: f32):\n"
            "} else {\n"
            "}\n");
}

TEST_F(PrintOpTest, SkipRegionsFlag) {
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t condition = def(i1);

  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_branch_build(&builder_, condition, result_types, 1,
                                        NULL, 0, LOOM_LOCATION_UNKNOWN, &op));

  // Add an op to the then region so we can verify it's not printed.
  loom_value_id_t yield_val = def(f32);
  loom_region_t* then_region = loom_test_branch_then_region(op);
  loom_builder_ip_t saved =
      loom_builder_enter_region(&builder_, op, then_region);
  loom_op_t* yield_op = NULL;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &yield_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved);

  // Print with SKIP_REGIONS: body should not appear.
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_text_print_operation_to_builder(
      module_, op, &builder,
      LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_SKIP_REGIONS));
  std::string output(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_EQ(output, "%1 = test.branch %0 -> (f32) { ... } else { ... }\n");
}

//===----------------------------------------------------------------------===//
// Predicate types
//===----------------------------------------------------------------------===//

TEST(PredicateLayout, StructSize) {
  // Verify the predicate struct is exactly 32 bytes.
  EXPECT_EQ(sizeof(loom_predicate_t), 32u);
}

TEST(PredicateLayout, FieldOffsets) {
  // Verify fields are at expected offsets for correct packing.
  loom_predicate_t predicate = {0};
  predicate.kind = LOOM_PREDICATE_MUL;
  predicate.arg_count = 2;
  predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicate.arg_tags[1] = LOOM_PRED_ARG_CONST;
  predicate.arg_tags[2] = LOOM_PRED_ARG_NONE;
  predicate.args[0] = 42;
  predicate.args[1] = 16;
  predicate.args[2] = 0;

  EXPECT_EQ(predicate.kind, LOOM_PREDICATE_MUL);
  EXPECT_EQ(predicate.arg_count, 2);
  EXPECT_EQ(predicate.arg_tags[0], LOOM_PRED_ARG_VALUE);
  EXPECT_EQ(predicate.arg_tags[1], LOOM_PRED_ARG_CONST);
  EXPECT_EQ(predicate.arg_tags[2], LOOM_PRED_ARG_NONE);
  EXPECT_EQ(predicate.args[0], 42);
  EXPECT_EQ(predicate.args[1], 16);
  EXPECT_EQ(predicate.args[2], 0);
}

TEST(PredicateLayout, AllKindsValid) {
  // Verify all predicate kinds can be stored and retrieved.
  loom_predicate_kind_t kinds[] = {
      LOOM_PREDICATE_EQ,   LOOM_PREDICATE_NE,    LOOM_PREDICATE_LT,
      LOOM_PREDICATE_LE,   LOOM_PREDICATE_GT,    LOOM_PREDICATE_GE,
      LOOM_PREDICATE_MUL,  LOOM_PREDICATE_MIN,   LOOM_PREDICATE_MAX,
      LOOM_PREDICATE_POW2, LOOM_PREDICATE_RANGE,
  };
  for (int i = 0; i < (int)IREE_ARRAYSIZE(kinds); ++i) {
    loom_predicate_t predicate = {0};
    predicate.kind = (uint8_t)kinds[i];
    EXPECT_EQ(predicate.kind, (uint8_t)i);
  }
}

TEST(PredicateLayout, AllArgTagsValid) {
  // Verify all argument tag values can be stored.
  loom_predicate_arg_tag_t tags[] = {
      LOOM_PRED_ARG_NONE,
      LOOM_PRED_ARG_VALUE,
      LOOM_PRED_ARG_CONST,
  };
  loom_predicate_t predicate = {0};
  predicate.arg_count = 3;
  for (int i = 0; i < 3 && i < (int)(sizeof(tags) / sizeof(tags[0])); ++i) {
    predicate.arg_tags[i] = (uint8_t)tags[i];
  }
  EXPECT_EQ(predicate.arg_tags[0], LOOM_PRED_ARG_NONE);
  EXPECT_EQ(predicate.arg_tags[1], LOOM_PRED_ARG_VALUE);
  EXPECT_EQ(predicate.arg_tags[2], LOOM_PRED_ARG_CONST);
}

TEST(PredicateLayout, LargeConstants) {
  // Verify int64_t args handle large values.
  loom_predicate_t predicate = {0};
  predicate.kind = LOOM_PREDICATE_RANGE;
  predicate.arg_count = 3;
  predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicate.arg_tags[1] = LOOM_PRED_ARG_CONST;
  predicate.arg_tags[2] = LOOM_PRED_ARG_CONST;
  predicate.args[0] = 0;
  predicate.args[1] = INT64_MIN;
  predicate.args[2] = INT64_MAX;
  EXPECT_EQ(predicate.args[1], INT64_MIN);
  EXPECT_EQ(predicate.args[2], INT64_MAX);
}

//===----------------------------------------------------------------------===//
// Predicate list attribute
//===----------------------------------------------------------------------===//

TEST(PredicateAttr, Construction) {
  // Verify loom_attr_predicate_list constructs the right attribute.
  loom_predicate_t predicates[2] = {};
  predicates[0].kind = LOOM_PREDICATE_MUL;
  predicates[0].arg_count = 2;
  predicates[1].kind = LOOM_PREDICATE_POW2;
  predicates[1].arg_count = 1;

  loom_attribute_t attr = loom_attr_predicate_list(predicates, 2);
  EXPECT_EQ(attr.kind, LOOM_ATTR_PREDICATE_LIST);
  EXPECT_EQ(attr.count, 2);
  EXPECT_EQ(attr.predicate_list, predicates);
}

TEST(PredicateAttr, NullList) {
  // A zero-count predicate list attribute.
  loom_attribute_t attr = loom_attr_predicate_list(NULL, 0);
  EXPECT_EQ(attr.kind, LOOM_ATTR_PREDICATE_LIST);
  EXPECT_EQ(attr.count, 0);
  EXPECT_EQ(attr.predicate_list, nullptr);
}

TEST(PredicateAttr, SizeConstraint) {
  // The attribute must still be exactly 16 bytes.
  EXPECT_EQ(sizeof(loom_attribute_t), 16u);
}

//===----------------------------------------------------------------------===//
// Predicate list printing
//===----------------------------------------------------------------------===//

// A synthetic vtable and format for testing predicate list printing.
// The op has one attribute (a predicate list) and no operands or results.
// Format: "test.predtest" PredicateList
static const uint8_t kPredTestName[] =
    "\x0d\x04"
    "test.predtest";
static const loom_format_element_t kPredTestFormat[] = {
    {LOOM_FORMAT_KIND_PREDICATE_LIST, 0, 0},
};
static const uint8_t kPredTestPredicatesBname[] =
    "\x0a"
    "predicates";
static const loom_attr_descriptor_t kPredTestAttrDesc[] = {
    {
        /*.name=*/kPredTestPredicatesBname,
        /*.attr_kind=*/LOOM_ATTR_PREDICATE_LIST,
    },
};
static const loom_op_vtable_t kPredTestVtable = {
    // Cache line 1: hot path.
    /*.traits=*/LOOM_TRAIT_PURE,
    /*.fixed_operand_count=*/0,
    /*.fixed_result_count=*/0,
    /*.attribute_count=*/1,
    /*.region_count=*/0,
    /*.vtable_flags=*/0,
    /*.symbol_kind=*/LOOM_SYMBOL_NONE,
    /*.constraint_count=*/0,
    /*.operand_descriptor_count=*/{},
    /*.control_flow_flags=*/{},
    /*.control_flow_reserved=*/{},
    /*.successor_selector_operand_index=*/{},
    /*.canonicalize=*/NULL,
    /*.infer_facts=*/NULL,
    /*.effective_traits=*/NULL,
    /*.attr_descriptors=*/kPredTestAttrDesc,
    /*.operand_descriptors=*/NULL,
    // Cache line 2: verify + parse/print + diagnostics.
    /*.type_transfer=*/{},
    /*.result_descriptors=*/NULL,
    /*.region_descriptors=*/NULL,
    /*.constraints=*/NULL,
    /*.verify=*/NULL,
    /*.name=*/kPredTestName,
    /*.format_elements=*/kPredTestFormat,
    /*.instance_flags_case_names=*/NULL,
    /*.format_element_count=*/1,
    /*.instance_flags_case_count=*/0,
    // Cache line 3: interface pointers.
    /*.call_like=*/{},
    /*.func_like=*/NULL,
};

// Test fixture that registers both the test dialect and the synthetic
// predicate test vtable.
class PrintPredicateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t test_count = 0;
    const loom_op_vtable_t* const* test_vtables =
        loom_test_dialect_vtables(&test_count);

    // Build an extended vtable array that includes the generated test
    // ops plus our synthetic predicate test vtable at index 0xFF.
    memset(extended_test_vtables_, 0, sizeof(extended_test_vtables_));
    for (iree_host_size_t i = 0; i < test_count; ++i) {
      extended_test_vtables_[i] = test_vtables[i];
    }
    pred_test_kind_ = LOOM_OP_KIND(LOOM_DIALECT_TEST, 0xFF);
    extended_test_vtables_[0xFF] = &kPredTestVtable;

    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 extended_test_vtables_, 256));

    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  std::string print_op(loom_op_t* op, loom_text_print_flags_t flags) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    iree_status_t status =
        loom_text_print_operation_to_builder(module_, op, &builder, flags);
    std::string result;
    if (iree_status_is_ok(status)) {
      result = std::string(iree_string_builder_buffer(&builder),
                           iree_string_builder_size(&builder));
    }
    IREE_EXPECT_OK(status);
    iree_string_builder_deinitialize(&builder);
    return result;
  }

  // Defines a value in the module and returns its ID.
  loom_value_id_t def(loom_type_t type) {
    loom_value_id_t id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(module_, type, &id));
    return id;
  }

  // Interns a string and returns its ID.
  loom_string_id_t intern(const char* str) {
    loom_string_id_t id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_module_intern_string(module_, iree_make_cstring_view(str), &id));
    return id;
  }

  // Builds an op with the synthetic predicate test vtable and the given
  // predicate list attribute.
  loom_op_t* build_pred_op(loom_predicate_t* predicates, uint16_t count) {
    loom_op_t* op = NULL;
    IREE_CHECK_OK(loom_builder_allocate_op(
        &builder_, pred_test_kind_, /*operand_count=*/0,
        /*result_count=*/0, /*region_count=*/0, /*tied_result_count=*/0,
        /*attribute_count=*/1, LOOM_LOCATION_UNKNOWN, &op));
    loom_op_attrs(op)[0] = loom_attr_predicate_list(predicates, count);
    return op;
  }

  iree_arena_block_pool_t block_pool_;
  const loom_op_vtable_t* extended_test_vtables_[256] = {};
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
  uint16_t pred_test_kind_ = 0;
};

TEST_F(PrintPredicateTest, SingleMulPredicate) {
  // Define a value so we can reference it by name.
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t dim_value = def(index_type);

  // Name the value %M (string table stores bare name "M").
  loom_string_id_t name_id = intern("M");
  module_->values.entries[dim_value].name_id = name_id;

  // Build: mul(%M, 16)
  loom_predicate_t predicates[1] = {};
  predicates[0].kind = LOOM_PREDICATE_MUL;
  predicates[0].arg_count = 2;
  predicates[0].arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicates[0].arg_tags[1] = LOOM_PRED_ARG_CONST;
  predicates[0].args[0] = (int64_t)dim_value;
  predicates[0].args[1] = 16;

  loom_op_t* op = build_pred_op(predicates, 1);
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "test.predtest [mul(%M, 16)]\n");
}

TEST_F(PrintPredicateTest, MultipleMixedPredicates) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t k_value = def(index_type);

  loom_string_id_t k_name = intern("K");
  module_->values.entries[k_value].name_id = k_name;

  // Build: [mul(%K, 16), lt(%K, 1024)]
  loom_predicate_t predicates[2] = {};
  predicates[0].kind = LOOM_PREDICATE_MUL;
  predicates[0].arg_count = 2;
  predicates[0].arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicates[0].arg_tags[1] = LOOM_PRED_ARG_CONST;
  predicates[0].args[0] = (int64_t)k_value;
  predicates[0].args[1] = 16;

  predicates[1].kind = LOOM_PREDICATE_LT;
  predicates[1].arg_count = 2;
  predicates[1].arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicates[1].arg_tags[1] = LOOM_PRED_ARG_CONST;
  predicates[1].args[0] = (int64_t)k_value;
  predicates[1].args[1] = 1024;

  loom_op_t* op = build_pred_op(predicates, 2);
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "test.predtest [mul(%K, 16), lt(%K, 1024)]\n");
}

TEST_F(PrintPredicateTest, Pow2SingleArg) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t value = def(index_type);

  loom_string_id_t name = intern("N");
  module_->values.entries[value].name_id = name;

  // Build: [pow2(%N)]
  loom_predicate_t predicates[1] = {};
  predicates[0].kind = LOOM_PREDICATE_POW2;
  predicates[0].arg_count = 1;
  predicates[0].arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicates[0].args[0] = (int64_t)value;

  loom_op_t* op = build_pred_op(predicates, 1);
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "test.predtest [pow2(%N)]\n");
}

TEST_F(PrintPredicateTest, RangeThreeArgs) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t value = def(index_type);

  loom_string_id_t name = intern("M");
  module_->values.entries[value].name_id = name;

  // Build: [range(%M, 1, 4096)]
  loom_predicate_t predicates[1] = {};
  predicates[0].kind = LOOM_PREDICATE_RANGE;
  predicates[0].arg_count = 3;
  predicates[0].arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicates[0].arg_tags[1] = LOOM_PRED_ARG_CONST;
  predicates[0].arg_tags[2] = LOOM_PRED_ARG_CONST;
  predicates[0].args[0] = (int64_t)value;
  predicates[0].args[1] = 1;
  predicates[0].args[2] = 4096;

  loom_op_t* op = build_pred_op(predicates, 1);
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT),
            "test.predtest [range(%M, 1, 4096)]\n");
}

TEST_F(PrintPredicateTest, EmptyPredicateList) {
  loom_op_t* op = build_pred_op(NULL, 0);
  EXPECT_EQ(print_op(op, LOOM_TEXT_PRINT_DEFAULT), "test.predtest []\n");
}

TEST_F(PrintPredicateTest, AllPredicateKinds) {
  // Verify all predicate kinds print their correct name.
  const char* expected_names[] = {"eq",  "ne",  "lt",  "le",   "gt",   "ge",
                                  "mul", "min", "max", "pow2", "range"};
  for (int kind = 0; kind <= LOOM_PREDICATE_RANGE; ++kind) {
    loom_predicate_t predicates[1] = {};
    predicates[0].kind = (uint8_t)kind;
    predicates[0].arg_count = loom_predicate_kind_argument_count((uint8_t)kind);
    for (uint8_t argument_index = 0; argument_index < predicates[0].arg_count;
         ++argument_index) {
      predicates[0].arg_tags[argument_index] = LOOM_PRED_ARG_CONST;
      predicates[0].args[argument_index] = 99 + argument_index;
    }

    loom_op_t* op = build_pred_op(predicates, 1);
    std::string output = print_op(op, LOOM_TEXT_PRINT_DEFAULT);
    std::string expected =
        std::string("test.predtest [") + expected_names[kind] + "(99";
    if (predicates[0].arg_count > 1) expected += ", 100";
    if (predicates[0].arg_count > 2) expected += ", 101";
    expected += ")]\n";
    EXPECT_EQ(output, expected) << "Failed for predicate kind " << kind;
  }
}

// Tests using the generated test.assume vtable (not the synthetic one).
// Validates the full code-gen -> builder -> printer path.
TEST_F(PrintPredicateTest, GeneratedTestAssume) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);

  // Create named values %M and %K.
  loom_value_id_t m_id = def(index_type);
  loom_value_id_t k_id = def(index_type);
  module_->values.entries[m_id].name_id = intern("M");
  module_->values.entries[k_id].name_id = intern("K");

  // Build predicates for the where clause.
  loom_predicate_t predicates[2] = {};
  predicates[0].kind = LOOM_PREDICATE_MUL;
  predicates[0].arg_count = 2;
  predicates[0].arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicates[0].args[0] = m_id;
  predicates[0].arg_tags[1] = LOOM_PRED_ARG_CONST;
  predicates[0].args[1] = 16;
  predicates[1].kind = LOOM_PREDICATE_LT;
  predicates[1].arg_count = 2;
  predicates[1].arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicates[1].args[0] = k_id;
  predicates[1].arg_tags[1] = LOOM_PRED_ARG_CONST;
  predicates[1].args[1] = 1024;

  // Build test.assume with predicates passed through the builder.
  loom_value_id_t operands[] = {m_id, k_id};
  loom_type_t result_types[] = {index_type, index_type};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_assume_build(&builder_, operands, 2, predicates, 2,
                                        result_types, 2, LOOM_LOCATION_UNKNOWN,
                                        &op));
  ASSERT_NE(op, nullptr);

  std::string output = print_op(op, LOOM_TEXT_PRINT_DEFAULT);
  EXPECT_EQ(output,
            "%2, %3 = test.assume %M, %K [mul(%M, 16), lt(%K, 1024)]"
            " : index, index\n");
}

//===----------------------------------------------------------------------===//
// Static encoding printing
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, TypeWithStaticEncoding) {
  // Add an encoding to the module: #q8_0<block=32>.
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("q8_0"), &name_id));
  loom_string_id_t block_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("block"), &block_id));

  loom_named_attr_t param = {/*.name_id=*/block_id, /*.reserved=*/{},
                             /*.value=*/loom_attr_i64(32)};
  loom_encoding_t encoding = {
      /*.name_id=*/name_id,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/1,
      /*.reserved=*/{},
      /*.attributes=*/&param,
  };
  uint16_t encoding_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module_, &encoding, &encoding_id));

  // Create a tile type referencing this encoding.
  loom_type_t tile = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                         256, encoding_id);
  EXPECT_EQ(print_type(tile, module_), "tile<256xf32, #q8_0<block=32>>");
}

TEST_F(PrintOpTest, TypeWithEncodingAlias) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("q8_0"), &name_id));
  loom_string_id_t alias_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("enc"), &alias_id));

  loom_encoding_t encoding = {
      /*.name_id=*/name_id,
      /*.alias_id=*/alias_id,
      /*.attribute_count=*/0,
  };
  uint16_t encoding_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module_, &encoding, &encoding_id));

  loom_type_t tile = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                         256, encoding_id);
  EXPECT_EQ(print_type(tile, module_), "tile<256xf32, #enc>");
}

TEST_F(PrintOpTest, TypeWithEncodingNoParams) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("dense"), &name_id));

  loom_encoding_t encoding = {
      /*.name_id=*/name_id,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/0,
  };
  uint16_t encoding_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module_, &encoding, &encoding_id));

  loom_type_t tile = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                         256, encoding_id);
  EXPECT_EQ(print_type(tile, module_), "tile<256xf32, #dense>");
}

TEST_F(PrintOpTest, TypeWithEncodingNoModule) {
  // Without a module, static encodings fall back to #encoding_N.
  loom_type_t tile = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                         256, /*encoding_id=*/3);
  EXPECT_EQ(print_type(tile), "tile<256xf32, #encoding_3>");
}

TEST_F(PrintOpTest, DynamicDimNamedValue) {
  // Dynamic dim referencing a named value prints [%name].
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t dim_id = def(index_type);
  module_->values.entries[dim_id].name_id = intern("M");
  loom_type_t tensor = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(dim_id), 0);
  EXPECT_EQ(print_type(tensor, module_), "tensor<[%M]xf32>");
}

TEST_F(PrintOpTest, DynamicDimUnnamedValue) {
  // Dynamic dim referencing an unnamed value prints [%N] (auto-name).
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t dim_id = def(index_type);
  loom_type_t tile = loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F16,
                                         loom_dim_pack_dynamic(dim_id),
                                         loom_dim_pack_static(4), 0);
  EXPECT_EQ(print_type(tile, module_), "tile<[%0]x4xf16>");
}

TEST_F(PrintOpTest, TypeWithSSAEncoding) {
  // SSA encoding prints as %name when the value is named.
  loom_type_t encoding_type = loom_type_encoding();
  loom_value_id_t enc_id = def(encoding_type);
  module_->values.entries[enc_id].name_id = intern("enc");
  loom_type_t tile = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I8,
                                         loom_dim_pack_static(256),
                                         /*encoding_id=*/(uint16_t)enc_id);
  tile.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  EXPECT_EQ(print_type(tile, module_), "tile<256xi8, %enc>");
}

TEST_F(PrintOpTest, TypeWithSSAEncodingUnnamed) {
  // SSA encoding with unnamed value prints %N (auto-name).
  loom_type_t encoding_type = loom_type_encoding();
  loom_value_id_t enc_id = def(encoding_type);
  loom_type_t tile = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I8,
                                         loom_dim_pack_static(256),
                                         /*encoding_id=*/(uint16_t)enc_id);
  tile.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  EXPECT_EQ(print_type(tile, module_), "tile<256xi8, %0>");
}

//===----------------------------------------------------------------------===//
// Location printing
//===----------------------------------------------------------------------===//

TEST_F(PrintOpTest, LocationUnknown) {
  // LOOM_LOCATION_UNKNOWN (0) emits no location annotation even when
  // location printing is enabled.
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, input, f32, LOOM_LOCATION_UNKNOWN, &op));
  std::string output =
      print_op(op, LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_LOCATIONS);
  EXPECT_EQ(output.find("loc("), std::string::npos)
      << "UNKNOWN location should not print, got: " << output;
}

TEST_F(PrintOpTest, LocationFile) {
  // Register a source with the context.
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_register_source(module_, IREE_SV("model.loom"), &source_id));

  // Add a file location to the module.
  loom_location_entry_t file_loc =
      loom_location_file_range(/*source_id=*/0, 42, 3, 42, 58);
  loom_location_id_t loc_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_module_add_location(module_, file_loc, &loc_id));

  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_neg_build(&builder_, input, f32, loc_id, &op));
  std::string output =
      print_op(op, LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_LOCATIONS);
  EXPECT_NE(output.find("loc(\"model.loom\":42:3 to 42:58)"), std::string::npos)
      << "Expected file location, got: " << output;
}

TEST_F(PrintOpTest, LocationFileUsesCanonicalStringEscapes) {
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_register_source(
      module_, IREE_SV("model \"main\"\\v2\n.loom"), &source_id));

  loom_location_entry_t file_loc =
      loom_location_file_range(source_id, 7, 8, 9, 10);
  loom_location_id_t loc_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_module_add_location(module_, file_loc, &loc_id));

  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_neg_build(&builder_, input, f32, loc_id, &op));
  std::string output =
      print_op(op, LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_LOCATIONS);
  EXPECT_NE(output.find("loc(\"model \\\"main\\\"\\\\v2\\n.loom\":7:8 to "
                        "9:10)"),
            std::string::npos)
      << "Expected escaped file location, got: " << output;
}

TEST_F(PrintOpTest, LocationOpaque) {
  // Register a source tag with the context.
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_register_source(module_, IREE_SV("torch"), &source_id));

  // Add an opaque location.
  loom_location_entry_t opaque_loc = {/*.kind=*/LOOM_LOCATION_OPAQUE};
  opaque_loc.opaque.source_id = 0;
  const char* data = "node_id=42";
  opaque_loc.opaque.data = (const uint8_t*)data;
  opaque_loc.opaque.data_length = (uint32_t)strlen(data);
  loom_location_id_t loc_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_module_add_location(module_, opaque_loc, &loc_id));

  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_neg_build(&builder_, input, f32, loc_id, &op));
  std::string output =
      print_op(op, LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_LOCATIONS);
  EXPECT_NE(output.find("loc(opaque<\"torch\", \"node_id=42\">)"),
            std::string::npos)
      << "Expected opaque location, got: " << output;
}

TEST_F(PrintOpTest, LocationOpaqueUsesCanonicalStringEscapes) {
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_register_source(module_, IREE_SV("torch \"aten\""),
                                             &source_id));

  loom_location_entry_t opaque_loc = {/*.kind=*/LOOM_LOCATION_OPAQUE};
  opaque_loc.opaque.source_id = source_id;
  const char* data = "node\\id\n\x01";
  opaque_loc.opaque.data = (const uint8_t*)data;
  opaque_loc.opaque.data_length = (uint32_t)strlen(data);
  loom_location_id_t loc_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_module_add_location(module_, opaque_loc, &loc_id));

  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_neg_build(&builder_, input, f32, loc_id, &op));
  std::string output =
      print_op(op, LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_LOCATIONS);
  EXPECT_NE(output.find("loc(opaque<\"torch \\\"aten\\\"\", "
                        "\"node\\\\id\\n\\u0001\">)"),
            std::string::npos)
      << "Expected escaped opaque location, got: " << output;
}

TEST_F(PrintOpTest, LocationOpaqueRejectsInvalidUtf8Data) {
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_register_source(module_, IREE_SV("torch"), &source_id));

  loom_location_entry_t opaque_loc = {/*.kind=*/LOOM_LOCATION_OPAQUE};
  opaque_loc.opaque.source_id = source_id;
  const uint8_t data[] = {0xFF};
  opaque_loc.opaque.data = data;
  opaque_loc.opaque.data_length = IREE_ARRAYSIZE(data);
  loom_location_id_t loc_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_module_add_location(module_, opaque_loc, &loc_id));

  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_neg_build(&builder_, input, f32, loc_id, &op));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      print_op_status(op, LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_LOCATIONS));
}

TEST_F(PrintOpTest, LocationFusedPrintsNestedLocationBodies) {
  loom_source_id_t jax_source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_register_source(module_, IREE_SV("jax.py"), &jax_source_id));
  loom_location_id_t jax_loc_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_module_add_location(
      module_,
      loom_location_file_range(jax_source_id, /*start_line=*/7,
                               /*start_col=*/8, /*end_line=*/7,
                               /*end_col=*/8),
      &jax_loc_id));

  loom_source_id_t recipe_source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_register_source(module_, IREE_SV("recipe.loom"),
                                             &recipe_source_id));
  loom_location_id_t recipe_loc_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_module_add_location(
      module_,
      loom_location_file_range(recipe_source_id, /*start_line=*/1,
                               /*start_col=*/2, /*end_line=*/3,
                               /*end_col=*/4),
      &recipe_loc_id));

  loom_source_id_t torch_source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_register_source(module_, IREE_SV("torch"), &torch_source_id));
  loom_location_entry_t opaque_loc = {/*.kind=*/LOOM_LOCATION_OPAQUE};
  opaque_loc.opaque.source_id = torch_source_id;
  const char* data = "node\n42";
  opaque_loc.opaque.data = (const uint8_t*)data;
  opaque_loc.opaque.data_length = (uint32_t)strlen(data);
  loom_location_id_t opaque_loc_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_module_add_location(module_, opaque_loc, &opaque_loc_id));

  loom_location_id_t* nested_children = NULL;
  IREE_ASSERT_OK(iree_arena_allocate_array(
      &module_->arena, 2, sizeof(*nested_children), (void**)&nested_children));
  nested_children[0] = recipe_loc_id;
  nested_children[1] = opaque_loc_id;
  loom_location_entry_t nested_fused_loc = {/*.kind=*/LOOM_LOCATION_FUSED};
  nested_fused_loc.fused.count = 2;
  nested_fused_loc.fused.children = nested_children;
  loom_location_id_t nested_fused_loc_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_module_add_location(module_, nested_fused_loc,
                                          &nested_fused_loc_id));

  loom_location_id_t* root_children = NULL;
  IREE_ASSERT_OK(iree_arena_allocate_array(
      &module_->arena, 2, sizeof(*root_children), (void**)&root_children));
  root_children[0] = jax_loc_id;
  root_children[1] = nested_fused_loc_id;
  loom_location_entry_t fused_loc = {/*.kind=*/LOOM_LOCATION_FUSED};
  fused_loc.fused.count = 2;
  fused_loc.fused.children = root_children;
  loom_location_id_t fused_loc_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_module_add_location(module_, fused_loc, &fused_loc_id));

  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_neg_build(&builder_, input, f32, fused_loc_id, &op));

  std::string output =
      print_op(op, LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_LOCATIONS);
  EXPECT_NE(output.find("loc(fused<\"jax.py\":7:8, "
                        "fused<\"recipe.loom\":1:2 to 3:4, "
                        "opaque<\"torch\", \"node\\n42\">>>)"),
            std::string::npos)
      << "Expected nested fused location, got: " << output;
}

TEST_F(PrintOpTest, LocationFusedRejectsOutOfRangeChildId) {
  loom_location_id_t* children = NULL;
  IREE_ASSERT_OK(iree_arena_allocate_array(
      &module_->arena, 1, sizeof(*children), (void**)&children));
  children[0] = 42;
  loom_location_entry_t fused_loc = {/*.kind=*/LOOM_LOCATION_FUSED};
  fused_loc.fused.count = 1;
  fused_loc.fused.children = children;
  loom_location_id_t fused_loc_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_module_add_location(module_, fused_loc, &fused_loc_id));

  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_neg_build(&builder_, input, f32, fused_loc_id, &op));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      print_op_status(op, LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_LOCATIONS));
}

//===----------------------------------------------------------------------===//
// Bounds-check regression tests
//===----------------------------------------------------------------------===//

// Build a test.attrs op and corrupt attribute_count to 0. The ATTR_DICT
// format element should fail with INVALID_ARGUMENT instead of reading
// out of bounds.
TEST_F(PrintOpTest, BoundsCheckAttrDictOutOfRange) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t input = def(f32);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, input,
                                       loom_make_named_attr_slice(NULL, 0), f32,
                                       LOOM_LOCATION_UNKNOWN, &op));
  // Corrupt: pretend the op has no attributes.
  op->attribute_count = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        print_op_status(op, LOOM_TEXT_PRINT_DEFAULT));
}

// Build a test.branch op (2 regions) and corrupt region_count to 0.
// The OPTIONAL_GROUP with ANCHOR_REGION should treat the group as absent
// (present=false) rather than reading out of bounds.
TEST_F(PrintOpTest, BoundsCheckOptionalRegionOutOfRange) {
  loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_value_id_t condition = def(i1);
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_branch_build(&builder_, condition, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &op));
  // Save the actual region count so we can verify it had regions.
  ASSERT_GT(op->region_count, 0);
  // Corrupt: pretend the op has no regions.
  op->region_count = 0;
  // The optional group anchor should not read out of bounds. The non-optional
  // REGION format element will still hit its own bounds check and return
  // INVALID_ARGUMENT — that's expected and correct.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        print_op_status(op, LOOM_TEXT_PRINT_DEFAULT));
}

// Build a test.func op with a visibility attribute and corrupt
// attribute_count to be less than the vtable expects. The OPTIONAL_GROUP
// with ANCHOR_ATTR should treat the group as absent rather than reading
// the op's attribute array out of bounds.
TEST_F(PrintOpTest, BoundsCheckOptionalAttrMalformed) {
  loom_symbol_ref_t callee = make_symbol("f");
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t result_types[] = {f32};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder_, LOOM_TEST_FUNC_BUILD_FLAG_HAS_VISIBILITY, 1, 0, callee, &f32,
      1, result_types, 1, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  // Verify the op was built with attributes.
  ASSERT_GT(op->attribute_count, 0);
  // Corrupt: pretend the op has no attributes.
  op->attribute_count = 0;
  // The optional groups should treat their attrs as absent (present=false)
  // rather than reading out of bounds. The non-optional SYMBOL_REF for the
  // callee will hit its own bounds check and return INVALID_ARGUMENT.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        print_op_status(op, LOOM_TEXT_PRINT_DEFAULT));
}

// Build a test.update op with a tied result and corrupt operand_count
// to be less than the tied result's operand_index. The printer should
// return INVALID_ARGUMENT instead of reading out of bounds.
TEST_F(PrintOpTest, BoundsCheckTiedResultBadOperandIndex) {
  loom_type_t tile_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t tensor_type = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(256), 0);
  loom_value_id_t source = def(tile_type);
  loom_value_id_t target = def(tensor_type);
  int64_t static_offsets[] = {0};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_test_update_build(&builder_, source, target, NULL, 0,
                                        static_offsets, 1, tensor_type,
                                        LOOM_LOCATION_UNKNOWN, &op));
  // Verify it has a tied result.
  ASSERT_GT(op->tied_result_count, 0);
  // Corrupt: pretend the op has no operands. The tied result still
  // references operand_index=1, which is now out of range.
  op->operand_count = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        print_op_status(op, LOOM_TEXT_PRINT_DEFAULT));
}

// Build a test.update op with dynamic offsets (INDEX_LIST) and corrupt
// operand_count so the dynamic operand indices are out of range.
TEST_F(PrintOpTest, BoundsCheckIndexListDynamicOutOfRange) {
  loom_type_t tile_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t tensor_type = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(256), 0);
  loom_value_id_t source = def(tile_type);
  loom_value_id_t target = def(tensor_type);
  // INT64_MIN marks a dynamic operand slot in the static_offsets array.
  int64_t static_offsets[] = {INT64_MIN};
  loom_op_t* op = NULL;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_UPDATE,
                                          /*operand_count=*/2,
                                          /*result_count=*/1,
                                          /*region_count=*/0,
                                          /*tied_result_count=*/1,
                                          /*attribute_count=*/1,
                                          LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = source;
  loom_op_operands(op)[1] = target;
  loom_op_attrs(op)[0] = loom_attr_i64_array(static_offsets, 1);
  loom_op_results(op)[0] = def(tensor_type);
  loom_op_tied_results(op)[0] = (loom_tied_result_t){
      /*.result_index=*/0,
      /*.operand_index=*/1,
      /*.has_type_change=*/true,
  };
  IREE_ASSERT_OK(loom_builder_finalize_op(&builder_, op));
  // The index list starts dynamic operands at field index 2, but this malformed
  // op only has source and target operands.
  ASSERT_EQ(op->operand_count, 2);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        print_op_status(op, LOOM_TEXT_PRINT_DEFAULT));
}

}  // namespace
}  // namespace loom
