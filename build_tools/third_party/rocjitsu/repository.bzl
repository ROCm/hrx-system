# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Explicit local rocjitsu runtime packages with a checked ELF closure."""

load("//build_tools/bazel:elf_dynamic_library_closure.bzl", "resolve_elf_dynamic_library_closure")

_LIBRARY = "lib/librocjitsu.so"
_MEMORY_HELPER = "libexec/rocjitsu/rocjitsu-memory"

def _runtime_build(repository_ctx, arguments):
    repository_ctx.file("BUILD.bazel", """load(%r, "rocjitsu_runtime")
package(default_visibility = ["//visibility:public"])
rocjitsu_runtime(name = "runtime", %s)
""" % (str(repository_ctx.attr._runtime_rule), arguments))

def _rocjitsu_repository_impl(repository_ctx):
    package_path = repository_ctx.getenv("IREE_ROCJITSU_RUNTIME_PATH")
    if not package_path:
        _runtime_build(repository_ctx, "error = %r" % (
            "rocjitsu execution requires an explicit runtime package: " +
            "--repo_env=IREE_ROCJITSU_RUNTIME_PATH=/path/to/package. " +
            "See build_tools/third_party/rocjitsu/README.md."
        ))
        return

    if repository_ctx.os.name != "linux":
        fail("rocjitsu runtime packages require a Linux host")
    root = repository_ctx.path(package_path)
    manifest_path = repository_ctx.path(root.get_child("runtime.json"))
    if not manifest_path.exists:
        fail("rocjitsu package has no runtime.json: %s" % root)
    manifest = json.decode(repository_ctx.read(manifest_path))
    if type(manifest) != "dict":
        fail("rocjitsu runtime.json must contain a JSON object")
    expected = {
        "format": 2,
        "host_fault_recovery": False,
        "host_os": "linux",
        "interposer": False,
        "revision": repository_ctx.attr.revision,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            fail("rocjitsu runtime.json requires %s = %r; got %r" % (
                name,
                value,
                manifest.get(name),
            ))
    if manifest.get("host_cpu") not in ["x86-64", "x86-64-v3"]:
        fail("rocjitsu runtime.json has an unsupported host_cpu")
    if manifest.get("sanitizer") not in ["none", "address"]:
        fail("rocjitsu runtime.json requires sanitizer = 'none' or 'address'")

    files = manifest.get("files", {})
    if type(files) != "dict" or _LIBRARY not in files or _MEMORY_HELPER not in files:
        fail("rocjitsu runtime.json must declare %s, %s, and their non-glibc dependencies" % (_LIBRARY, _MEMORY_HELPER))
    sha256sum = repository_ctx.which("sha256sum")
    if not sha256sum:
        fail("Checking a local rocjitsu runtime package requires sha256sum")
    for name, digest in sorted(files.items()):
        parts = name.split("/")
        if name != _MEMORY_HELPER and (
            len(parts) != 2 or parts[0] != "lib" or parts[1] in ["", ".", ".."] or "\\" in name
        ):
            fail("rocjitsu runtime files must be libraries under lib/ or %s: %r" % (_MEMORY_HELPER, name))
        if type(digest) != "string" or len(digest) != 64:
            fail("rocjitsu runtime checksum must be a SHA256 digest: %r" % name)
        file = root.get_child(name)
        if not file.exists:
            fail("rocjitsu runtime package is missing %s" % file)
        repository_ctx.watch(file)
        repository_ctx.watch(file.realpath)
        result = repository_ctx.execute([sha256sum, "--", file], quiet = True)
        if result.return_code or result.stdout.split(" ")[0] != digest:
            fail("rocjitsu runtime checksum mismatch for %s: %s%s" % (
                file,
                result.stdout,
                result.stderr,
            ))

    closure = resolve_elf_dynamic_library_closure(
        repository_ctx,
        {"rocjitsu": str(root.get_child(_LIBRARY))},
        library_search_path = str(root.get_child("lib")),
    )

    # The simulator execs its helper without the application's loader variables.
    # Its dependencies must resolve from its own installed layout or be static.
    helper = root.get_child(_MEMORY_HELPER)
    executable = repository_ctx.execute(["test", "-x", helper], quiet = True)
    if executable.return_code:
        fail("rocjitsu memory helper is not executable: %s" % helper)
    helper_closure = resolve_elf_dynamic_library_closure(
        repository_ctx,
        {"rocjitsu-memory": str(helper)},
    )
    for name, path in helper_closure.items():
        if name in closure and closure[name] != path:
            fail("rocjitsu library and helper resolve %s to different files" % name)
        closure[name] = path
    for name, path in closure.items():
        relative_path = "lib/" + name
        if relative_path not in files or str(root.get_child(relative_path).realpath) != path:
            fail("rocjitsu package must include its non-glibc dependency %s (resolved to %s)" % (name, path))
    for name in files:
        if name not in [_LIBRARY, _MEMORY_HELPER] and name[len("lib/"):] not in closure:
            fail("rocjitsu runtime file %s is outside the runtime's non-glibc ELF closure" % name)

    repository_ctx.symlink(manifest_path, "runtime.json")
    for name in files:
        repository_ctx.symlink(root.get_child(name), name)
    _runtime_build(repository_ctx, """library = %r,
    memory_helper = %r,
    manifest = "runtime.json",
    srcs = %r,
    host_cpu = %r,
    sanitizer = %r,
""" % (_LIBRARY, _MEMORY_HELPER, sorted(files), manifest["host_cpu"], manifest["sanitizer"]))

rocjitsu_repository = repository_rule(
    implementation = _rocjitsu_repository_impl,
    attrs = {
        "revision": attr.string(mandatory = True),
        "_runtime_rule": attr.label(default = "//build_tools/third_party/rocjitsu:runtime.bzl"),
    },
    environ = ["IREE_ROCJITSU_RUNTIME_PATH"],
    doc = "Exposes an explicit local package as @rocjitsu_runtime//:runtime.",
)
