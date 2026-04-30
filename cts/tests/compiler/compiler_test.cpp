// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_compiler_cxx.h"
#include "hrx_runtime_cxx.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

extern hrx_device_t g_test_device;

namespace {

void requireOk(hrx_status_t status) {
  if (hrx_status_is_ok(status)) {
    return;
  }

  char *message = nullptr;
  size_t length = 0;
  hrx_status_t format_status = hrx_status_to_string(status, &message, &length);
  INFO("hrx error: " << (message ? message : "?"));
  hrx_status_free_message(message);
  hrx_status_ignore(format_status);
  hrx_status_ignore(status);
  REQUIRE(false);
}

#define REQUIRE_HRX_OK(expr) requireOk((expr))

std::string loadMlir(const char *relative_path) {
  const char *source_dir = std::getenv("HRX_CTS_SOURCE_DIR");
  std::string path =
      std::string((source_dir && source_dir[0]) ? source_dir
                                                : HRX_CTS_SOURCE_DIR) +
      "/" + relative_path;
  std::ifstream file(path);
  REQUIRE(file.is_open());
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

bool cliCompilerConfigured() {
  const char *cli_path = std::getenv("HRX_IREE_COMPILER_CLI");
  if (!cli_path || !*cli_path) {
    return false;
  }
  std::ifstream file(cli_path);
  return file.good();
}

std::unique_ptr<hrx::compiler::HrxGraphCompiler>
createCompilerIfAvailable(hrx_compiler_backend_t backend) {
  try {
    return std::make_unique<hrx::compiler::HrxGraphCompiler>(backend);
  } catch (const std::runtime_error &e) {
    INFO("compiler backend unavailable: " << e.what());
    return nullptr;
  }
}

void verifyAddI64(hrx_device_t device,
                  const hrx::compiler::compiler_output_ptr &output) {
  REQUIRE(output);
  REQUIRE(hrx_compiler_output_data(output.get()) != nullptr);
  REQUIRE(hrx_compiler_output_size(output.get()) > 0);

  hrx::runtime::module_ptr module;
  REQUIRE_HRX_OK(hrx_module_load_compiler_output(device, output.get(),
                                                 module.for_output()));

  hrx::runtime::function_ptr function;
  REQUIRE_HRX_OK(hrx_module_lookup_function(module.get(), "module.add_i64",
                                            function.for_output()));

  hrx::runtime::value_list_ptr args;
  hrx::runtime::value_list_ptr rets;
  REQUIRE_HRX_OK(hrx_value_list_create(2, args.for_output()));
  REQUIRE_HRX_OK(hrx_value_list_create(1, rets.for_output()));
  REQUIRE_HRX_OK(hrx_value_list_push_i64(args.get(), 40));
  REQUIRE_HRX_OK(hrx_value_list_push_i64(args.get(), 2));
  REQUIRE_HRX_OK(hrx_function_invoke(module.get(), function.get(), args.get(),
                                     rets.get()));

  int64_t result = 0;
  REQUIRE_HRX_OK(hrx_value_list_get_i64(rets.get(), 0, &result));
  REQUIRE(result == 42);
}

void verifyBadMlirDiagnostics(hrx::compiler::HrxGraphCompiler &compiler) {
  auto session = compiler.createSession();
  session.setFlags({
      "--iree-hal-target-backends=llvm-cpu",
  });

  try {
    (void)session.compileMlir("module { util.func public @broken(");
    FAIL("expected malformed MLIR to fail");
  } catch (const std::runtime_error &e) {
    std::string message = e.what();
    INFO("diagnostics: " << message);
    REQUIRE(message.find("failed to parse MLIR") != std::string::npos);
  }
}

} // namespace

TEST_CASE("Compile MLIR with default dylib backend", "[compiler][dylib]") {
  auto compiler = createCompilerIfAvailable(HRX_COMPILER_BACKEND_DEFAULT);
  if (!compiler) {
    SKIP("IREE compiler dylib unavailable");
  }

  hrx::compiler::compiler_output_ptr output =
      compiler->compileMlir(loadMlir("testdata/add_i64.mlir"),
                            {"--iree-hal-target-backends=llvm-cpu"});
  verifyAddI64(g_test_device, output);
}

TEST_CASE("Compile MLIR with explicit CLI backend", "[compiler][cli]") {
  if (!cliCompilerConfigured()) {
    SKIP("HRX_IREE_COMPILER_CLI is not configured");
  }

  auto compiler = createCompilerIfAvailable(HRX_COMPILER_BACKEND_CLI);
  REQUIRE(compiler != nullptr);

  hrx::compiler::compiler_output_ptr output =
      compiler->compileMlir(loadMlir("testdata/add_i64.mlir"),
                            {"--iree-hal-target-backends=llvm-cpu"});
  verifyAddI64(g_test_device, output);
}

TEST_CASE("Compiler reports diagnostics for bad MLIR",
          "[compiler][diagnostics]") {
  auto compiler = createCompilerIfAvailable(HRX_COMPILER_BACKEND_DEFAULT);
  if (!compiler) {
    if (!cliCompilerConfigured()) {
      SKIP("no compiler backend available");
    }
    compiler = createCompilerIfAvailable(HRX_COMPILER_BACKEND_CLI);
    REQUIRE(compiler != nullptr);
  }

  verifyBadMlirDiagnostics(*compiler);
}
