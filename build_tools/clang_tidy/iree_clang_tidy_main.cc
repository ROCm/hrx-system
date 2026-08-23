// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "clang-tidy/tool/ClangTidyMain.h"

// Windows LLVM distributions are statically linked and do not export the
// symbols required by loadable clang-tidy modules. Link the IREE checks into a
// dedicated tool instead. clangTidyMain references every built-in module
// anchor, but this tool only exposes the IREE policy checks, so satisfy those
// references without pulling hundreds of megabytes of unused check modules
// into the executable.
namespace clang::tidy {

volatile int AbseilModuleAnchorSource = 0;
volatile int AlteraModuleAnchorSource = 0;
volatile int AndroidModuleAnchorSource = 0;
volatile int BoostModuleAnchorSource = 0;
volatile int BugproneModuleAnchorSource = 0;
volatile int CERTModuleAnchorSource = 0;
volatile int ConcurrencyModuleAnchorSource = 0;
volatile int CppCoreGuidelinesModuleAnchorSource = 0;
volatile int CustomModuleAnchorSource = 0;
volatile int DarwinModuleAnchorSource = 0;
volatile int FuchsiaModuleAnchorSource = 0;
volatile int GoogleModuleAnchorSource = 0;
volatile int HICPPModuleAnchorSource = 0;
volatile int LinuxKernelModuleAnchorSource = 0;
volatile int LLVMLibcModuleAnchorSource = 0;
volatile int LLVMModuleAnchorSource = 0;
volatile int MPIModuleAnchorSource = 0;
volatile int MiscModuleAnchorSource = 0;
volatile int ModernizeModuleAnchorSource = 0;
volatile int ObjCModuleAnchorSource = 0;
volatile int OpenMPModuleAnchorSource = 0;
volatile int PerformanceModuleAnchorSource = 0;
volatile int PortabilityModuleAnchorSource = 0;
volatile int ReadabilityModuleAnchorSource = 0;
volatile int ZirconModuleAnchorSource = 0;

}  // namespace clang::tidy

int main(int argument_count, char** argument_values) {
  return clang::tidy::clangTidyMain(argument_count,
                                    const_cast<const char**>(argument_values));
}
