// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/vm/bytecode/tooling/dump.h"

IREE_FLAG(string, module_name, "module",
          "Logical module name used to qualify reflected function types.");

static const char kIreeDumpVmModuleUsage[] =
    "Inspects an IREE VM bytecode module without loading runtime providers.\n"
    "\n"
    "The dumper verifies the module container and declarations, prints public\n"
    "reflection and private function storage, and disassembles every known\n"
    "instruction record. It never links or executes the module.\n"
    "\n"
    "Usage:\n"
    "  iree-dump-vm-module [--module_name=name] <module.vm>\n"
    "\n"
    "The binary format does not contain its deployment module name. The\n"
    "default logical name is 'module'; --module_name supplies the name used\n"
    "when rendering function-reference types.\n";

static iree_status_t iree_dump_vm_module_write_file(void* user_data,
                                                    iree_string_view_t text) {
  FILE* file = (FILE*)user_data;
  if (text.size == 0) return iree_ok_status();
  if (IREE_UNLIKELY(fwrite(text.data, 1, text.size, file) != text.size)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to write VM module dump output");
  }
  return iree_ok_status();
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;

  iree_flags_set_usage("iree-dump-vm-module", kIreeDumpVmModuleUsage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_ok_status();
  if (argc != 2) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected one VM bytecode module path argument");
  }

  iree_io_file_contents_t* file_contents = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_map(iree_make_cstring_view(argv[1]),
                                       IREE_IO_FILE_ACCESS_READ, host_allocator,
                                       &file_contents);
  }

  if (iree_status_is_ok(status)) {
    const iree_vm_bytecode_dump_write_callback_t write_callback = {
        .fn = iree_dump_vm_module_write_file,
        .user_data = stdout,
    };
    status = iree_vm_bytecode_module_dump(
        iree_make_cstring_view(FLAG_module_name), file_contents->const_buffer,
        write_callback, host_allocator);
  }

  iree_io_file_contents_free(file_contents);

  if (iree_status_is_ok(status) && fflush(stdout) != 0) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "failed to flush VM module dump output");
  }
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  fflush(stderr);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
