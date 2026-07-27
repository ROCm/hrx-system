// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/module_emitter.h"

#include <memory>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/llvmir/descriptors/low_registry.h"
#include "loom/target/emit/llvmir/target_presets.h"
#include "loom/target/emit/llvmir/text_writer.h"
#include "loom/target/emit/llvmir/verify.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnosticEmission;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::DiagnosticEmissionCapture;
using ModulePtr = ::loom::testing::ModulePtr;
using LlvmirModulePtr =
    std::unique_ptr<loom_llvmir_module_t, decltype(&loom_llvmir_module_free)>;

static const loom_llvmir_target_env_t kTestKernelTargetEnv = {
    /*.name=*/IREE_SVL("loom-kernel64-unknown-none"),
    /*.target_triple=*/IREE_SVL("loom-kernel64-unknown-none"),
    /*.data_layout=*/IREE_SVL("e-p:64:64-p1:64:64-i64:64-n8:16:32:64-S128"),
    /*.object_format=*/LOOM_LLVMIR_OBJECT_FORMAT_ELF,
    /*.default_pointer_bitwidth=*/64,
    /*.index_bitwidth=*/64,
    /*.offset_bitwidth=*/64,
    /*.address_spaces=*/
    {
        /*.generic=*/0,
        /*.global=*/1,
        /*.local=*/3,
        /*.constant=*/4,
        /*.private_memory=*/5,
        /*.buffer_resource=*/UINT32_MAX,
    },
};

static const loom_llvmir_target_profile_t kTestKernelProfile = {
    /*.name=*/IREE_SVL("loom-test-kernel"),
    /*.target_env=*/&kTestKernelTargetEnv,
    /*.kind=*/LOOM_LLVMIR_TARGET_PROFILE_KERNEL,
    /*.target_cpu=*/{},
    /*.target_features=*/{},
    /*.x86_packed_dot_feature_bits=*/{},
    /*.exported_linkage=*/LOOM_LLVMIR_LINKAGE_DEFAULT,
    /*.kernel=*/
    {
        /*.calling_convention=*/{},
        /*.required_workgroup_size_metadata_name=*/
        IREE_SVL("loom_test_workgroup_size"),
        /*.required_workgroup_size=*/{},
        /*.flat_workgroup_size_min=*/1,
        /*.flat_workgroup_size_max=*/1024,
        /*.subgroup_size=*/{},
        /*.binding_resource_flags=*/{},
        /*.flat_workgroup_size_attr_name=*/
        IREE_SVL("loom-test-flat-work-group-size"),
        /*.uniform_workgroup_size_attr_name=*/
        IREE_SVL("loom-test-uniform-workgroup-size"),
        /*.flags=*/LOOM_LLVMIR_KERNEL_PROFILE_FLAG_ALWAYSINLINE,
        /*.coordinate_intrinsics=*/
        {
            /*.workitem_id_x=*/IREE_SVL("llvm.loom.workitem.id.x"),
            /*.workitem_id_y=*/{},
            /*.workitem_id_z=*/{},
            /*.workgroup_id_x=*/{},
            /*.workgroup_id_y=*/{},
            /*.workgroup_id_z=*/{},
        },
        /*.binding_parameter_attrs=*/
        {
            {
                /*.kind=*/LOOM_LLVMIR_ATTR_NOUNDEF,
                /*.value=*/{},
                /*.value2=*/{},
                /*.type_id=*/LOOM_LLVMIR_TYPE_ID_INVALID,
            },
        },
        /*.binding_parameter_attr_count=*/1,
    },
};

static bool TestKernelProjectBundle(
    const loom_llvmir_target_profile_projection_request_t* request,
    const loom_llvmir_target_profile_t** out_profile) {
  *out_profile = nullptr;
  const loom_target_bundle_t* bundle = request->bundle;
  if (!iree_string_view_equal(request->target_triple,
                              kTestKernelTargetEnv.target_triple)) {
    return false;
  }
  if (bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return false;
  }
  *out_profile = &kTestKernelProfile;
  return true;
}

static const loom_llvmir_target_profile_t* const kTestKernelProfiles[] = {
    &kTestKernelProfile,
};

static const loom_llvmir_target_profile_provider_t kTestKernelProvider = {
    /*.name=*/IREE_SVL("loom-test-kernel"),
    /*.profiles=*/kTestKernelProfiles,
    /*.profile_count=*/IREE_ARRAYSIZE(kTestKernelProfiles),
    /*.llc_target_name=*/{},
    /*.project_bundle=*/TestKernelProjectBundle,
};

class LlvmirModuleEmitterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_llvmir_low_descriptor_registry_initialize(&low_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseModule(const char* source) {
    DiagnosticCapture parse_capture;
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/parse_capture.sink(),
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("llvmir_module_emitter_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    EXPECT_TRUE(parse_capture.diagnostics.empty());
    EXPECT_NE(module, nullptr);
    return ModulePtr(module);
  }

  iree_status_t EmitLowModule(loom_module_t* module,
                              DiagnosticEmissionCapture* capture,
                              LlvmirModulePtr* out_module) {
    iree_arena_allocator_t scratch_arena;
    iree_arena_initialize(&block_pool_, &scratch_arena);
    loom_llvmir_module_t* raw_module = nullptr;
    const loom_llvmir_target_profile_provider_t* providers[] = {
        &kTestKernelProvider,
    };
    const loom_llvmir_target_profile_registry_t target_profile_registry = {
        /*.default_profile=*/nullptr,
        /*.providers=*/providers,
        /*.provider_count=*/IREE_ARRAYSIZE(providers),
    };
    loom_llvmir_emit_low_module_options_t options = {};
    loom_llvmir_emit_low_module_options_initialize(&options);
    options.target_profile_registry = &target_profile_registry;
    iree_status_t status = loom_llvmir_emit_low_module(
        module, &low_registry_.registry, loom_target_selection_empty(),
        capture->emitter(), &scratch_arena, &options, &raw_module,
        iree_allocator_system());
    if (iree_status_is_ok(status) && raw_module != nullptr) {
      out_module->reset(raw_module);
    }
    iree_arena_deinitialize(&scratch_arena);
    return status;
  }

  iree_status_t WriteText(loom_llvmir_module_t* module, std::string* out_text) {
    IREE_RETURN_IF_ERROR(loom_llvmir_verify_module(module));
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    iree_status_t status = loom_llvmir_text_write_module(module, &stream);
    if (iree_status_is_ok(status)) {
      iree_string_view_t view = iree_string_builder_view(&builder);
      out_text->assign(view.data, view.size);
    }
    iree_string_builder_deinitialize(&builder);
    return status;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t low_registry_ = {};
};

TEST_F(LlvmirModuleEmitterTest, EmitsMultipleLowFunctionsInModuleOrder) {
  ModulePtr module = ParseModule(R"(
llvmir.target<object> @target {
  triple = "loom-direct64-unknown-none",
  data_layout = "e-p:64:64-i64:64-n8:16:32:64-S128"
}

low.func.def target(@target) abi(object_function) @first(%lhs: reg<llvmir.i32>, %rhs: reg<llvmir.i32>) -> (reg<llvmir.i32>) asm<llvmir.generic.core> {
  %sum = add.i32 %lhs, %rhs
  return %sum
}

low.func.def target(@target) abi(object_function) @second(%input_view: reg<llvmir.ptr>, %output_view: reg<llvmir.ptr>, %bounded_i: reg<llvmir.i64>) -> (reg<llvmir.i32>) asm<llvmir.generic.core> {
  %loaded = load.indexed.i32 %input_view, %bounded_i, 16, 4
  store.indexed.i32 %loaded, %output_view, %bounded_i, 32, 4
  return %loaded
}
)");

  DiagnosticEmissionCapture capture;
  LlvmirModulePtr llvmir_module(nullptr, loom_llvmir_module_free);
  IREE_ASSERT_OK(EmitLowModule(module.get(), &capture, &llvmir_module));
  ASSERT_NE(llvmir_module, nullptr);
  EXPECT_TRUE(capture.emissions.empty());

  std::string text;
  IREE_ASSERT_OK(WriteText(llvmir_module.get(), &text));
  const size_t first_position = text.find("@first(");
  const size_t second_position = text.find("@second(");
  ASSERT_NE(first_position, std::string::npos) << text;
  ASSERT_NE(second_position, std::string::npos) << text;
  EXPECT_LT(first_position, second_position) << text;
  EXPECT_NE(text.find("%sum = add i32 %lhs, %rhs"), std::string::npos) << text;
  EXPECT_NE(text.find("getelementptr i8, ptr %input_view"), std::string::npos)
      << text;
  EXPECT_NE(text.find("store i32 %loaded"), std::string::npos) << text;
}

TEST_F(LlvmirModuleEmitterTest, EmitsFusedMultiplyAddIntrinsics) {
  ModulePtr module = ParseModule(R"(
llvmir.target<object> @target {
  triple = "loom-direct64-unknown-none",
  data_layout = "e-p:64:64-i64:64-n8:16:32:64-S128"
}

low.func.def target(@target) abi(object_function) @fma_scalar(%a: reg<llvmir.f32>, %b: reg<llvmir.f32>, %c: reg<llvmir.f32>) -> (reg<llvmir.f32>) asm<llvmir.generic.core> {
  %result = fma.f32 %a, %b, %c
  return %result
}

low.func.def target(@target) abi(object_function) @fma_vector(%a: reg<llvmir.f32 x4>, %b: reg<llvmir.f32 x4>, %c: reg<llvmir.f32 x4>) -> (reg<llvmir.f32 x4>) asm<llvmir.generic.core> {
  %result = fma.v4f32 %a, %b, %c
  return %result
}
)");

  DiagnosticEmissionCapture capture;
  LlvmirModulePtr llvmir_module(nullptr, loom_llvmir_module_free);
  IREE_ASSERT_OK(EmitLowModule(module.get(), &capture, &llvmir_module));
  ASSERT_NE(llvmir_module, nullptr);
  EXPECT_TRUE(capture.emissions.empty());

  std::string text;
  IREE_ASSERT_OK(WriteText(llvmir_module.get(), &text));
  EXPECT_NE(text.find("%result = call float @llvm.fma.f32(float %a, float %b, "
                      "float %c)"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("%result = call <4 x float> @llvm.fma.v4f32(<4 x float> "
                      "%a, <4 x float> %b, <4 x float> %c)"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("declare float @llvm.fma.f32(float, float, float)"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("declare <4 x float> @llvm.fma.v4f32(<4 x float>, "
                      "<4 x float>, <4 x float>)"),
            std::string::npos)
      << text;
}

TEST_F(LlvmirModuleEmitterTest, EmitsUnaryFloatOps) {
  ModulePtr module = ParseModule(R"(
llvmir.target<object> @target {
  triple = "loom-direct64-unknown-none",
  data_layout = "e-p:64:64-i64:64-n8:16:32:64-S128"
}

low.func.def target(@target) abi(object_function) @unary_scalar(%input: reg<llvmir.f32>) -> (reg<llvmir.f32>) asm<llvmir.generic.core> {
  %negated = neg.f32 %input
  %result = abs.f32 %negated
  return %result
}

low.func.def target(@target) abi(object_function) @unary_vector(%input: reg<llvmir.f32 x4>) -> (reg<llvmir.f32 x4>) asm<llvmir.generic.core> {
  %negated = neg.v4f32 %input
  %result = abs.v4f32 %negated
  return %result
}
)");

  DiagnosticEmissionCapture capture;
  LlvmirModulePtr llvmir_module(nullptr, loom_llvmir_module_free);
  IREE_ASSERT_OK(EmitLowModule(module.get(), &capture, &llvmir_module));
  ASSERT_NE(llvmir_module, nullptr);
  EXPECT_TRUE(capture.emissions.empty());

  std::string text;
  IREE_ASSERT_OK(WriteText(llvmir_module.get(), &text));
  EXPECT_NE(text.find("%negated = fneg float %input"), std::string::npos)
      << text;
  EXPECT_NE(text.find("%result = call float @llvm.fabs.f32(float %negated)"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("%negated = fneg <4 x float> %input"), std::string::npos)
      << text;
  EXPECT_NE(text.find("%result = call <4 x float> @llvm.fabs.v4f32(<4 x float> "
                      "%negated)"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("declare float @llvm.fabs.f32(float)"), std::string::npos)
      << text;
  EXPECT_NE(text.find("declare <4 x float> @llvm.fabs.v4f32(<4 x float>)"),
            std::string::npos)
      << text;
}

TEST_F(LlvmirModuleEmitterTest, EmitsFloatMinNumMaxNumIntrinsics) {
  ModulePtr module = ParseModule(R"(
llvmir.target<object> @target {
  triple = "loom-direct64-unknown-none",
  data_layout = "e-p:64:64-i64:64-n8:16:32:64-S128"
}

low.func.def target(@target) abi(object_function) @minmax_scalar(%lhs: reg<llvmir.f32>, %rhs: reg<llvmir.f32>) -> (reg<llvmir.f32>) asm<llvmir.generic.core> {
  %min = minnum.f32 %lhs, %rhs
  %result = maxnum.f32 %min, %rhs
  return %result
}

low.func.def target(@target) abi(object_function) @minmax_vector(%lhs: reg<llvmir.f32 x2>, %rhs: reg<llvmir.f32 x2>) -> (reg<llvmir.f32 x2>) asm<llvmir.generic.core> {
  %min = minnum.v2f32 %lhs, %rhs
  %result = maxnum.v2f32 %min, %rhs
  return %result
}
)");

  DiagnosticEmissionCapture capture;
  LlvmirModulePtr llvmir_module(nullptr, loom_llvmir_module_free);
  IREE_ASSERT_OK(EmitLowModule(module.get(), &capture, &llvmir_module));
  ASSERT_NE(llvmir_module, nullptr);
  EXPECT_TRUE(capture.emissions.empty());

  std::string text;
  IREE_ASSERT_OK(WriteText(llvmir_module.get(), &text));
  EXPECT_NE(text.find("%min = call float @llvm.minnum.f32(float %lhs, float "
                      "%rhs)"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("%result = call float @llvm.maxnum.f32(float %min, float "
                      "%rhs)"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("%min = call <2 x float> @llvm.minnum.v2f32(<2 x float> "
                      "%lhs, <2 x float> %rhs)"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("%result = call <2 x float> @llvm.maxnum.v2f32(<2 x "
                      "float> %min, <2 x float> %rhs)"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("declare float @llvm.minnum.f32(float, float)"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("declare <2 x float> @llvm.maxnum.v2f32(<2 x float>, "
                      "<2 x float>)"),
            std::string::npos)
      << text;
}

TEST_F(LlvmirModuleEmitterTest, EmitsKernelAbiFromLowKernelFacts) {
  ModulePtr module = ParseModule(R"(
llvmir.target<object> @target {
  triple = "loom-kernel64-unknown-none"
}

low.kernel.def target(@target) workgroup_size(128, 2, 1) @dispatch() asm<llvmir.generic.core> {
  %input = resource<hal_binding> {index = 0, source_type = hal.buffer} : reg<llvmir.ptr>
  %value = load.i32 %input, 4
  return
}
)");

  DiagnosticEmissionCapture capture;
  LlvmirModulePtr llvmir_module(nullptr, loom_llvmir_module_free);
  IREE_ASSERT_OK(EmitLowModule(module.get(), &capture, &llvmir_module));
  ASSERT_NE(llvmir_module, nullptr);
  EXPECT_TRUE(capture.emissions.empty());

  std::string text;
  IREE_ASSERT_OK(WriteText(llvmir_module.get(), &text));
  EXPECT_NE(text.find("target triple = \"loom-kernel64-unknown-none\""),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("define void @dispatch(ptr addrspace(1) noundef %input) "
                      "#0 !loom_test_workgroup_size !0"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("getelementptr i8, ptr addrspace(1) %input, i64 4"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("load i32, ptr addrspace(1)"), std::string::npos) << text;
  EXPECT_NE(text.find("\"loom-test-flat-work-group-size\"=\"1,1024\""),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("\"loom-test-uniform-workgroup-size\""),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("!0 = !{i32 128, i32 2, i32 1}\n"), std::string::npos)
      << text;
}

TEST_F(LlvmirModuleEmitterTest, EmitsKernelScratchAllocaAddressSpace) {
  ModulePtr module = ParseModule(R"(
llvmir.target<object> @target {
  triple = "loom-kernel64-unknown-none"
}

low.kernel.def target(@target) workgroup_size(64, 1, 1) @dispatch() asm<llvmir.generic.core> {
  %tid = kernel.workitem_id.x
  %bytes = const.i64 256
  %scratch = alloca.workgroup.i8 %bytes, 16
  %one = const.i32 1
  store.indexed.i32 %one, %scratch, %tid, 0, 4
  %loaded = load.indexed.i32 %scratch, %tid, 0, 4
  store.indexed.i32 %loaded, %scratch, %tid, 0, 4
  return
}
)");

  DiagnosticEmissionCapture capture;
  LlvmirModulePtr llvmir_module(nullptr, loom_llvmir_module_free);
  IREE_ASSERT_OK(EmitLowModule(module.get(), &capture, &llvmir_module));
  ASSERT_NE(llvmir_module, nullptr);
  EXPECT_TRUE(capture.emissions.empty());

  std::string text;
  IREE_ASSERT_OK(WriteText(llvmir_module.get(), &text));
  EXPECT_NE(text.find("%scratch = alloca i8, i64 256, align 16, addrspace(3)"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("mul i64 %tid, 4"), std::string::npos) << text;
  EXPECT_NE(text.find("getelementptr i8, ptr addrspace(3) %scratch"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("store i32 1, ptr addrspace(3)"), std::string::npos)
      << text;
  EXPECT_NE(text.find("load i32, ptr addrspace(3)"), std::string::npos) << text;
}

TEST_F(LlvmirModuleEmitterTest, ReportsHalKernelWithoutWorkgroupSize) {
  ModulePtr module = ParseModule(R"(
llvmir.target<object> @target {
  triple = "loom-kernel64-unknown-none"
}

low.kernel.def target(@target) @dispatch() asm<llvmir.generic.core> {
  return
}
)");

  DiagnosticEmissionCapture capture;
  LlvmirModulePtr llvmir_module(nullptr, loom_llvmir_module_free);
  IREE_ASSERT_OK(EmitLowModule(module.get(), &capture, &llvmir_module));
  EXPECT_EQ(llvmir_module, nullptr);
  ASSERT_EQ(capture.emissions.size(), 1u);

  const CapturedDiagnosticEmission& emission = capture.emissions[0];
  EXPECT_EQ(emission.error, LOOM_ERR_TARGET_054);
  ASSERT_EQ(emission.string_params.size(), 3u);
  EXPECT_EQ(emission.string_params[0], "dispatch");
  EXPECT_EQ(emission.string_params[1], "llvmir.low");
  EXPECT_EQ(emission.string_params[2], "hal_kernel_workgroup_size");
  ASSERT_EQ(emission.u32_params.size(), 2u);
  EXPECT_EQ(emission.u32_params[0], 0u);
  EXPECT_EQ(emission.u32_params[1], 3u);
}

TEST_F(LlvmirModuleEmitterTest,
       ReportsMissingKernelCoordinateIntrinsicAsDiagnostic) {
  ModulePtr module = ParseModule(R"(
llvmir.target<object> @target {
  triple = "loom-kernel64-unknown-none"
}

low.kernel.def target(@target) workgroup_size(1, 1, 1) @dispatch() asm<llvmir.generic.core> {
  %tid = kernel.workitem_id.y
  return
}
)");

  DiagnosticEmissionCapture capture;
  LlvmirModulePtr llvmir_module(nullptr, loom_llvmir_module_free);
  IREE_ASSERT_OK(EmitLowModule(module.get(), &capture, &llvmir_module));
  EXPECT_EQ(llvmir_module, nullptr);
  ASSERT_EQ(capture.emissions.size(), 1u);

  const CapturedDiagnosticEmission& emission = capture.emissions[0];
  EXPECT_EQ(emission.error, LOOM_ERR_TARGET_053);
  ASSERT_EQ(emission.string_params.size(), 4u);
  EXPECT_EQ(emission.string_params[0], "dispatch");
  EXPECT_EQ(emission.string_params[1], "llvmir.kernel.workitem_id.y");
  EXPECT_EQ(emission.string_params[2], "llvmir.generic.core");
  EXPECT_EQ(emission.string_params[3], "llvmir.low");
}

TEST_F(LlvmirModuleEmitterTest, ReportsDirectHalKernelPointerArgument) {
  ModulePtr module = ParseModule(R"(
llvmir.target<object> @target {
  triple = "loom-kernel64-unknown-none"
}

low.kernel.def target(@target) workgroup_size(1, 1, 1) @dispatch(%input: reg<llvmir.ptr>) asm<llvmir.generic.core> {
  return
}
)");

  DiagnosticEmissionCapture capture;
  LlvmirModulePtr llvmir_module(nullptr, loom_llvmir_module_free);
  IREE_ASSERT_OK(EmitLowModule(module.get(), &capture, &llvmir_module));
  EXPECT_EQ(llvmir_module, nullptr);
  ASSERT_EQ(capture.emissions.size(), 1u);

  const CapturedDiagnosticEmission& emission = capture.emissions[0];
  EXPECT_EQ(emission.error, LOOM_ERR_TARGET_056);
  ASSERT_EQ(emission.string_params.size(), 5u);
  EXPECT_EQ(emission.string_params[0], "dispatch");
  EXPECT_EQ(emission.string_params[1], "parameter");
  EXPECT_EQ(emission.string_params[2], "input");
  EXPECT_EQ(emission.string_params[3], "llvmir.low");
  EXPECT_EQ(emission.string_params[4], "low.resource<hal_binding> pointer");
  ASSERT_EQ(emission.type_params.size(), 1u);
}

TEST_F(LlvmirModuleEmitterTest, ReportsUnsupportedAbiProjection) {
  ModulePtr module = ParseModule(R"(
llvmir.target<object> @target {
  abi = shader_entry_point,
  contract_set_key = "llvmir.generic.core",
  triple = "loom-kernel64-unknown-none"
}

low.func.def target(@target) @dispatch() asm<llvmir.generic.core> {
  return
}
)");

  DiagnosticEmissionCapture capture;
  LlvmirModulePtr llvmir_module(nullptr, loom_llvmir_module_free);
  IREE_ASSERT_OK(EmitLowModule(module.get(), &capture, &llvmir_module));
  EXPECT_EQ(llvmir_module, nullptr);
  ASSERT_EQ(capture.emissions.size(), 1u);

  const CapturedDiagnosticEmission& emission = capture.emissions[0];
  EXPECT_EQ(emission.error, LOOM_ERR_TARGET_036);
  ASSERT_EQ(emission.string_params.size(), 6u);
  EXPECT_EQ(emission.string_params[0], "target");
  EXPECT_EQ(emission.string_params[1], "dispatch");
  EXPECT_EQ(emission.string_params[2], "target");
  EXPECT_EQ(emission.string_params[3], "llvmir.low");
  EXPECT_EQ(emission.string_params[4], "llvmir");
  EXPECT_EQ(emission.string_params[5], "shader_entry_point");
}

TEST_F(LlvmirModuleEmitterTest, ReportsUnsupportedFunctionShapeAsDiagnostic) {
  ModulePtr module = ParseModule(R"(
llvmir.target<object> @target {
  triple = "loom-direct64-unknown-none",
  data_layout = "e-p:64:64-i64:64-n8:16:32:64-S128"
}

low.func.def target(@target) abi(object_function) @multi_result(%lhs: reg<llvmir.i32>, %rhs: reg<llvmir.i32>) -> (reg<llvmir.i32>, reg<llvmir.i32>) asm<llvmir.generic.core> {
  %sum = add.i32 %lhs, %rhs
  return %sum, %lhs
}
)");

  DiagnosticEmissionCapture capture;
  LlvmirModulePtr llvmir_module(nullptr, loom_llvmir_module_free);
  IREE_ASSERT_OK(EmitLowModule(module.get(), &capture, &llvmir_module));
  EXPECT_EQ(llvmir_module, nullptr);
  ASSERT_EQ(capture.emissions.size(), 1u);

  const CapturedDiagnosticEmission& emission = capture.emissions[0];
  EXPECT_EQ(emission.error, LOOM_ERR_TARGET_054);
  ASSERT_EQ(emission.string_params.size(), 3u);
  EXPECT_EQ(emission.string_params[0], "multi_result");
  EXPECT_EQ(emission.string_params[1], "llvmir.low");
  EXPECT_EQ(emission.string_params[2], "function_result");
  ASSERT_EQ(emission.u32_params.size(), 2u);
  EXPECT_EQ(emission.u32_params[0], 2u);
  EXPECT_EQ(emission.u32_params[1], 1u);
}

}  // namespace
}  // namespace loom
