// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_KERNEL_REF_H_
#define LOOMCXX_KERNEL_REF_H_

// Host launch descriptors for single-source C++.
//
// Host code names a kernel K by odr-using loom::kernel::kernel_ref<&K>; the
// host compile only references that variable template specialization. The
// specializations host code references are the importer's roots: it exports
// their kernels K. The
// descriptor gives the host K's export and how to pack its kernel arguments,
// so the host can dispatch K without naming conventions or side tables.
// loom-cxx compiles host and device code from one parse and defines each
// descriptor the host references inside the host object, with local linkage.
// Each object resolves its own references to its own descriptors, which name
// the object's own embedded code object; no kernel_ref symbol is exported, so
// descriptors of different kernels with the same mangled name
// (internal-linkage lambdas, for example) never merge. Nothing registers or
// loads a code object at startup: a runtime loads it from its
// embedded_code_object record the first time it launches one of its kernels.
// The kernel argument layout comes from the compiled kernel.
//
// To launch, a host calls the descriptor's |pack| function to write the
// kernel's explicit argument segment, |kernarg_size| bytes, from the kernel's
// host parameter objects, and dispatches export |export_ordinal| of the
// executable it loaded from |code_object| with that buffer as the kernel's
// explicit arguments.
//
// This header has no other dependencies so that host code can include it.

namespace loom::kernel {

// Writes the explicit kernel argument segment of a compiled kernel: all of
// its descriptor's |kernarg_size| bytes at |kernarg|, from the kernel's host
// parameter objects, |parameters|[0..parameter_count). loom-cxx generates one
// per available kernel from the kernel argument layout the device compiler
// chose, with internal linkage. It copies each argument's payload: a device
// pointer for a buffer, or the low-address bytes of a by-value scalar (a
// char, short or bool occupies the low bytes of a wider argument slot). Every
// byte no argument covers, such as the rest of such a slot, is zeroed. It
// needs no particular alignment of |kernarg| or of the parameter objects.
using pack_function = void (*)(void* kernarg, const void* const* parameters);

// Version 3 names the kernel by its export ordinal instead of by name and its
// code object by an embedded_code_object record that a runtime loads lazily,
// and packs kernel arguments with a generated pack_function instead of a
// table of leaves.
inline constexpr unsigned int descriptor_version = 3;
// The kernel was imported and compiled.
inline constexpr unsigned int descriptor_flag_available = 1u << 0;

// The device code object an object file embeds, with what loading it needs.
// loom-cxx defines one per object that launches kernels, with internal
// linkage; every available descriptor of the object points to it. A runtime
// identifies the code object by this record's address.
struct embedded_code_object {
  // Device target of the code object, from loom-cxx's
  // --loom-target=FAMILY:SELECTOR: NUL-terminated family ("amdgpu") and
  // selector ("gfx90a").
  const char* target_family;
  const char* target_key;
  // The code object's bytes.
  const void* data;
  decltype(sizeof(0)) size;
};

// Launch facts of one kernel, one per kernel_ref specialization a translation
// unit references.
struct descriptor {
  // descriptor_version.
  unsigned int version;
  // descriptor_flag_* bits.
  unsigned int flags;
  // NUL-terminated export name of the kernel in the compiled executable, for
  // diagnostics: launches use |export_ordinal|.
  const char* name;
  // Workgroup size fixed by a loom::workgroup_size annotation, or zeros when
  // the kernel's launch configuration computes it at dispatch.
  unsigned int workgroup_size[3];
  // Byte size of the compiled kernel's explicit kernel argument segment.
  unsigned int kernarg_size;
  // Number of kernel parameters: host parameter objects |pack| reads.
  unsigned int parameter_count;
  // Ordinal of the kernel's export in |code_object|: its index among the
  // code object's kernels, in the order a loader numbers them (for an AMDGPU
  // code object, the order of its amdhsa.kernels metadata). Zero when the
  // kernel is unavailable.
  unsigned int export_ordinal;
  // The code object that exports the kernel: the object's embedded code
  // object. Null when the kernel is unavailable.
  const embedded_code_object* code_object;
  // Packs the kernel arguments. Null when the kernel is unavailable.
  pack_function pack;
};

// The descriptor of kernel K. Odr-use it, for example by taking its address,
// from host code that launches K. It has no definition in C++; loom-cxx
// defines each specialization.
template <auto Kernel>
[[loom::kernel_ref]] extern const descriptor kernel_ref;

}  // namespace loom::kernel

#endif  // LOOMCXX_KERNEL_REF_H_
