// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/io/stdio_util.h"
#include "iree/vm/bytecode/tooling/dump.h"

static const char kIreeDumpModuleUsage[] =
    "Inspects an IREE module without loading executable providers.\n"
    "\n"
    "The current implementation accepts VM bytecode modules. It verifies the\n"
    "container and declarations, prints public reflection and private "
    "function\n"
    "storage, and disassembles every known instruction record. It never links\n"
    "or executes the module.\n"
    "\n"
    "Usage:\n"
    "  iree-dump-module <module.vm>\n";

static iree_status_t iree_dump_module_write_file(void* user_data,
                                                 iree_string_view_t text) {
  FILE* file = (FILE*)user_data;
  if (text.size == 0) return iree_ok_status();
  if (IREE_UNLIKELY(fwrite(text.data, 1, text.size, file) != text.size)) {
    return iree_make_stdio_status("failed to write module dump output");
  }
  return iree_ok_status();
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;

  iree_flags_set_usage("iree-dump-module", kIreeDumpModuleUsage);
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
        .fn = iree_dump_module_write_file,
        .user_data = stdout,
    };
    status = iree_vm_bytecode_module_dump(IREE_SV("module"),
                                          file_contents->const_buffer,
                                          write_callback, host_allocator);
  }

  iree_io_file_contents_free(file_contents);

  if (iree_status_is_ok(status) && fflush(stdout) != 0) {
    status = iree_make_stdio_status("failed to flush module dump output");
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
