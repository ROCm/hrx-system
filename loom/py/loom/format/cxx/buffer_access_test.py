# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Immutable buffer boundaries in public C++ source projection."""

import pytest

from loom.format.cxx.errors import CxxRenderError
from loom.format.cxx.registry import load_module
from loom.format.cxx.renderer import render_module


def test_preserves_public_mutable_signature_with_immutable_internal_call() -> None:
    source = """\
func.def public @identity(%value: buffer) -> (buffer) {
  func.return %value : buffer
}
func.def public @literal() -> (buffer) {
  %data = buffer.constant bytes("0b") : buffer
  %result = func.call pure @identity(%data) : (buffer) -> (buffer)
  func.return %result : buffer
}
"""
    exported = render_module(load_module(source.encode()))
    assert (
        "\nloom::type::buffer<unsigned char> identity("
        "loom::type::buffer<unsigned char> value);"
    ) in exported
    assert (
        "static loom::type::buffer<const unsigned char> identity_buffer1("
        "loom::type::buffer<const unsigned char> value);"
    ) in exported
    assert "identity_buffer1(loom::buffer::root(data))" in exported


def test_rejects_writing_through_an_immutable_callable_result() -> None:
    source = """\
func.def @identity(%value: buffer) -> (buffer) {
  %aligned = buffer.assume.alignment %value {minimum_alignment = 1} : buffer
  func.return %aligned : buffer
}
func.def public @write() {
  %data = buffer.constant bytes("0b") : buffer
  %result = func.call pure @identity(%data) : (buffer) -> (buffer)
  %c0 = index.constant 0 : offset
  %view = buffer.view %result[%c0] : buffer -> view<1xi8>
  %value = scalar.constant 1 : i8
  view.store %value, %view[0] : i8, view<1xi8>
  func.return
}
"""
    with pytest.raises(CxxRenderError, match="immutable buffer storage is written"):
        render_module(load_module(source.encode()))


def test_rejects_immutable_argument_to_an_opaque_external_buffer_contract() -> None:
    source = """\
func.decl import("host") @consume(%data: buffer)
func.def public @run() {
  %data = buffer.constant bytes("0b") : buffer
  func.call @consume(%data) : (buffer)
  func.return
}
"""
    with pytest.raises(CxxRenderError, match="access-qualified external declaration"):
        render_module(load_module(source.encode()))


@pytest.mark.parametrize("literal_region", [0, 1, 2])
def test_rejects_writing_immutable_storage_selected_by_switch(
    literal_region: int,
) -> None:
    yielded = ["%mutable"] * 3
    yielded[literal_region] = "%literal"
    source = f"""\
func.def public @write(%selector: index, %mutable: buffer) {{
  %literal = buffer.constant bytes("0b") : buffer
  %selected = scf.switch %selector -> (buffer) {{
    case 0 {{ scf.yield {yielded[0]} : buffer }}
    case 1 {{ scf.yield {yielded[1]} : buffer }}
    default {{ scf.yield {yielded[2]} : buffer }}
  }}
  %zero = index.constant 0 : offset
  %view = buffer.view %selected[%zero] : buffer -> view<1xi8>
  %value = scalar.constant 1 : i8
  view.store %value, %view[0] : i8, view<1xi8>
  func.return
}}
"""
    with pytest.raises(CxxRenderError, match="immutable buffer storage is written"):
        render_module(load_module(source.encode()))
