// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_COMPILER_CXX_H_
#define HRX_COMPILER_CXX_H_

#include "hrx_compiler.h"
#include "hrx_runtime_cxx.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace hrx::compiler {

using compiler_ptr =
    runtime::hrx_ptr<hrx_compiler_t, hrx_compiler_retain, hrx_compiler_release>;
using compiler_session_ptr =
    runtime::hrx_ptr<hrx_compiler_session_t, hrx_compiler_session_retain,
                     hrx_compiler_session_release>;
using compiler_output_ptr =
    runtime::hrx_ptr<hrx_compiler_output_t, hrx_compiler_output_retain,
                     hrx_compiler_output_release>;

class HrxGraphCompiler {
public:
  class Session {
  public:
    Session() = default;
    explicit Session(compiler_session_ptr session)
        : session_(std::move(session)) {}

    void setFlags(const std::vector<std::string> &flags) {
      std::vector<const char *> c_flags;
      c_flags.reserve(flags.size());
      for (const std::string &flag : flags) {
        c_flags.push_back(flag.c_str());
      }
      hrx_status_t status = hrx_compiler_session_set_flags(
          session_.get(), c_flags.data(), c_flags.size());
      if (!hrx_status_is_ok(status)) {
        std::string message = runtime::format_status(status);
        hrx_status_ignore(status);
        throw std::runtime_error("hrx compiler: " + message);
      }
    }

    compiler_output_ptr compileMlir(std::string_view mlir) {
      compiler_output_ptr output;
      hrx_status_t status = hrx_compiler_session_compile_mlir(
          session_.get(), mlir.data(), mlir.size(), output.for_output());
      if (!hrx_status_is_ok(status)) {
        std::string message = runtime::format_status(status);
        hrx_status_ignore(status);
        throw std::runtime_error("hrx compiler: " + message);
      }
      return output;
    }

    hrx_compiler_session_t get() const { return session_.get(); }
    explicit operator bool() const { return static_cast<bool>(session_); }

  private:
    compiler_session_ptr session_;
  };

  explicit HrxGraphCompiler(
      hrx_compiler_backend_t backend = HRX_COMPILER_BACKEND_AUTO) {
    hrx_status_t status = hrx_compiler_create(backend, compiler_.for_output());
    if (!hrx_status_is_ok(status)) {
      std::string message = runtime::format_status(status);
      hrx_status_ignore(status);
      throw std::runtime_error("hrx compiler: " + message);
    }
  }

  Session createSession() const {
    compiler_session_ptr session;
    hrx_status_t status =
        hrx_compiler_session_create(compiler_.get(), session.for_output());
    if (!hrx_status_is_ok(status)) {
      std::string message = runtime::format_status(status);
      hrx_status_ignore(status);
      throw std::runtime_error("hrx compiler: " + message);
    }
    return Session(std::move(session));
  }

  compiler_output_ptr
  compileMlir(std::string_view mlir,
              const std::vector<std::string> &flags = {}) const {
    Session session = createSession();
    if (!flags.empty()) {
      session.setFlags(flags);
    }
    return session.compileMlir(mlir);
  }

  hrx_compiler_backend_t backend() const {
    return hrx_compiler_backend(compiler_.get());
  }

  hrx_compiler_t get() const { return compiler_.get(); }

private:
  compiler_ptr compiler_;
};

} // namespace hrx::compiler

#endif // HRX_COMPILER_CXX_H_
