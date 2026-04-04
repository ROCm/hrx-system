// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PYRE_COMPILER_CXX_H_
#define PYRE_COMPILER_CXX_H_

#include "pyre_compiler.h"
#include "pyre_runtime_cxx.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pyre::compiler {

using compiler_ptr =
    runtime::pyre_ptr<pyre_compiler_t, pyre_compiler_retain,
                      pyre_compiler_release>;
using compiler_session_ptr =
    runtime::pyre_ptr<pyre_compiler_session_t,
                      pyre_compiler_session_retain,
                      pyre_compiler_session_release>;
using compiler_output_ptr =
    runtime::pyre_ptr<pyre_compiler_output_t,
                      pyre_compiler_output_retain,
                      pyre_compiler_output_release>;

class PyreGraphCompiler {
 public:
  class Session {
   public:
    Session() = default;
    explicit Session(compiler_session_ptr session)
        : session_(std::move(session)) {}

    void setFlags(const std::vector<std::string>& flags) {
      std::vector<const char*> c_flags;
      c_flags.reserve(flags.size());
      for (const std::string& flag : flags) {
        c_flags.push_back(flag.c_str());
      }
      pyre_status_t status = pyre_compiler_session_set_flags(
          session_.get(), c_flags.data(), c_flags.size());
      if (!pyre_status_is_ok(status)) {
        std::string message = runtime::format_status(status);
        pyre_status_ignore(status);
        throw std::runtime_error("pyre compiler: " + message);
      }
    }

    compiler_output_ptr compileMlir(std::string_view mlir) {
      compiler_output_ptr output;
      pyre_status_t status = pyre_compiler_session_compile_mlir(
          session_.get(), mlir.data(), mlir.size(), output.for_output());
      if (!pyre_status_is_ok(status)) {
        std::string message = runtime::format_status(status);
        pyre_status_ignore(status);
        throw std::runtime_error("pyre compiler: " + message);
      }
      return output;
    }

    pyre_compiler_session_t get() const { return session_.get(); }
    explicit operator bool() const { return static_cast<bool>(session_); }

   private:
    compiler_session_ptr session_;
  };

  explicit PyreGraphCompiler(
      pyre_compiler_backend_t backend = PYRE_COMPILER_BACKEND_AUTO) {
    pyre_status_t status =
        pyre_compiler_create(backend, compiler_.for_output());
    if (!pyre_status_is_ok(status)) {
      std::string message = runtime::format_status(status);
      pyre_status_ignore(status);
      throw std::runtime_error("pyre compiler: " + message);
    }
  }

  Session createSession() const {
    compiler_session_ptr session;
    pyre_status_t status = pyre_compiler_session_create(
        compiler_.get(), session.for_output());
    if (!pyre_status_is_ok(status)) {
      std::string message = runtime::format_status(status);
      pyre_status_ignore(status);
      throw std::runtime_error("pyre compiler: " + message);
    }
    return Session(std::move(session));
  }

  compiler_output_ptr compileMlir(
      std::string_view mlir,
      const std::vector<std::string>& flags = {}) const {
    Session session = createSession();
    if (!flags.empty()) {
      session.setFlags(flags);
    }
    return session.compileMlir(mlir);
  }

  pyre_compiler_backend_t backend() const {
    return pyre_compiler_backend(compiler_.get());
  }

  pyre_compiler_t get() const { return compiler_.get(); }

 private:
  compiler_ptr compiler_;
};

}  // namespace pyre::compiler

#endif  // PYRE_COMPILER_CXX_H_
