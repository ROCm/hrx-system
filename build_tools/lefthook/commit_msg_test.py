# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

COMMIT_MSG_PATH = Path(__file__).with_name("commit_msg.py")
COMMIT_MSG_SPEC = importlib.util.spec_from_file_location("commit_msg", COMMIT_MSG_PATH)
if COMMIT_MSG_SPEC is None or COMMIT_MSG_SPEC.loader is None:
    raise RuntimeError(f"unable to load {COMMIT_MSG_PATH}")
commit_msg = importlib.util.module_from_spec(COMMIT_MSG_SPEC)
sys.modules[COMMIT_MSG_SPEC.name] = commit_msg
COMMIT_MSG_SPEC.loader.exec_module(commit_msg)


class CommitMessageTest(unittest.TestCase):
    def diagnostic_texts(
        self,
        message: str,
        *,
        changed_paths: list[str] | None = None,
    ) -> list[str]:
        return [
            diagnostic.text
            for diagnostic in commit_msg.validate_commit_message_text(
                message,
                changed_paths=changed_paths or [],
            )
        ]

    def diagnostic_messages(
        self,
        message: str,
        *,
        changed_paths: list[str] | None = None,
    ) -> list[str]:
        return [
            diagnostic.message
            for diagnostic in commit_msg.validate_commit_message_text(
                message,
                changed_paths=changed_paths or [],
            )
        ]

    def test_allows_normal_paragraphs_and_loom_tools(self):
        self.assertEqual(
            self.diagnostic_texts(
                "[Loom] Update loom-check help\n"
                "\n"
                "Document loom-compile and loom-opt workflows.\n"
            ),
            [],
        )

    def test_allows_git_autosquash_subjects(self):
        valid_subject = "[Loom] Update sanitizer race lowering"
        for prefix in ("fixup!", "squash!", "amend!", "fixup! squash!"):
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    self.diagnostic_texts(f"{prefix} {valid_subject}\n"),
                    [],
                )

    def test_allows_autosquash_prefix_to_exceed_subject_length(self):
        valid_subject = "[Infra] " + "x" * (commit_msg.LINE_LENGTH_LIMIT - 8)
        self.assertEqual(
            self.diagnostic_messages(f"fixup! {valid_subject}\n"),
            [],
        )

    def test_rejects_autosquash_target_without_subject_tag(self):
        self.assertEqual(
            self.diagnostic_messages("fixup! Update race lowering\n"),
            ["autosquash target subject must start with a bracketed subsystem tag"],
        )

    def test_rejects_long_autosquash_target_subject(self):
        subject = "[Infra] " + "x" * 80
        self.assertEqual(
            self.diagnostic_messages(f"fixup! {subject}\n"),
            ["subject line is 88 characters; keep it at or below 72"],
        )

    def test_rejects_literal_newline_escape(self):
        self.assertEqual(
            self.diagnostic_texts(
                "[Loom] Verify sanitizer race sync ordering\n"
                "\n"
                "First paragraph.\\n\\nSecond paragraph.\n"
            ),
            [r"\n"],
        )

    def test_rejects_literal_carriage_return_escape(self):
        self.assertEqual(
            self.diagnostic_texts("[Infra] Subject\n\nBody with \\r escape.\n"),
            [r"\r"],
        )

    def test_rejects_bead_ids(self):
        self.assertEqual(
            self.diagnostic_texts(
                "[Loom] Wire race checks\n"
                "\n"
                "Covers loom-20r1y, loom-qof73, loom-20r1y.1, and bd-123.\n"
            ),
            ["loom-20r1y", "loom-qof73", "loom-20r1y.1", "bd-123"],
        )

    def test_ignores_git_comment_lines(self):
        self.assertEqual(
            self.diagnostic_texts(
                "[Loom] Update checks\n"
                "\n"
                "# literal \\n and loom-20r1y in comments are ignored\n"
            ),
            [],
        )

    def test_validates_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            message_path = Path(temporary_directory) / "COMMIT_EDITMSG"
            message_path.write_text(
                "[Infra] Subject\n\nReferences loom-20r1y.\n",
                encoding="utf-8",
            )

            diagnostics = commit_msg.validate_commit_message_file(message_path)

        self.assertEqual(
            [diagnostic.text for diagnostic in diagnostics], ["loom-20r1y"]
        )

    def test_rejects_missing_subject_tag_with_path_suggestions(self):
        diagnostics = commit_msg.validate_commit_message_text(
            "Update race lowering\n\nBody.\n",
            changed_paths=[
                "loom/src/loom/target/arch/amdgpu/lower/sanitizer.c",
                "runtime/src/iree/hal/drivers/amdgpu/device.c",
            ],
        )

        self.assertEqual(
            [diagnostic.message for diagnostic in diagnostics],
            ["subject line must start with a bracketed subsystem tag"],
        )
        self.assertIn("[Loom/AMDGPU]", diagnostics[0].hint)
        self.assertIn("[HAL/AMDGPU]", diagnostics[0].hint)

    def test_rejects_missing_subject_description(self):
        self.assertEqual(
            self.diagnostic_messages("[Infra]\n\nBody.\n"),
            ["subject line must include a description after the tag"],
        )

    def test_rejects_long_subject(self):
        subject = "[Infra] " + "x" * 80
        self.assertEqual(
            self.diagnostic_messages(subject + "\n\nBody.\n"),
            ["subject line is 88 characters; keep it at or below 72"],
        )

    def test_rejects_long_body_lines_outside_code_blocks(self):
        long_line = "This body line is intentionally too long for commit prose " + (
            "because it should be wrapped."
        )
        diagnostics = commit_msg.validate_commit_message_text(
            f"[Infra] Subject\n\n{long_line}\n"
        )
        self.assertEqual(
            [diagnostic.message for diagnostic in diagnostics],
            [
                f"commit message body line is {len(long_line)} characters; "
                "wrap prose to 72 columns"
            ],
        )
        self.assertEqual([diagnostic.autofixable for diagnostic in diagnostics], [True])

    def test_allows_long_lines_inside_code_blocks(self):
        long_line = "x" * 120
        self.assertEqual(
            self.diagnostic_messages(
                f"[Infra] Subject\n\n```text\n{long_line}\n```\nWrapped prose.\n"
            ),
            [],
        )

    def test_allows_long_structured_body_lines(self):
        long_url = "See https://example.com/" + "x" * 90
        long_table = "| Column | " + "x" * 90 + " |"
        long_code = "    git commit -m " + "x" * 90
        long_trailer = "Co-authored-by: " + "x" * 90
        long_trailer_continuation = " " + "x" * 90

        self.assertEqual(
            self.diagnostic_messages(
                "[Infra] Subject\n"
                "\n"
                f"{long_url}\n"
                f"{long_table}\n"
                f"{long_code}\n"
                "\n"
                f"{long_trailer}\n"
                f"{long_trailer_continuation}\n"
            ),
            [],
        )

    def test_rejects_long_bullet_body_lines(self):
        long_bullet = (
            "- This bullet list item is intentionally too long for the commit "
            "message body width."
        )
        diagnostics = commit_msg.validate_commit_message_text(
            f"[Infra] Subject\n\n{long_bullet}\n"
        )
        self.assertEqual(
            [diagnostic.message for diagnostic in diagnostics],
            [
                f"commit message body line is {len(long_bullet)} characters; "
                "wrap prose to 72 columns"
            ],
        )
        self.assertEqual(
            [diagnostic.autofixable for diagnostic in diagnostics], [False]
        )

    def test_reflows_ordinary_body_paragraphs(self):
        self.assertEqual(
            commit_msg.reflow_commit_message_text(
                "[Infra] Subject\n"
                "\n"
                "This body paragraph is intentionally long enough to require "
                "wrapping while preserving all words in the original order.\n"
            ),
            "[Infra] Subject\n"
            "\n"
            "This body paragraph is intentionally long enough to require wrapping\n"
            "while preserving all words in the original order.\n",
        )

    def test_reflow_preserves_structured_body_lines(self):
        long_bullet = (
            "- This bullet list item is intentionally too long for automatic "
            "reflow because list indentation needs human intent."
        )
        long_url = "See https://example.com/" + "x" * 90
        long_code = "    git commit -m " + "x" * 90
        long_trailer = "Change-Id: " + "x" * 90
        message = (
            "[Infra] Subject\n"
            "\n"
            f"{long_bullet}\n"
            f"{long_url}\n"
            "```text\n"
            f"{'x' * 90}\n"
            "```\n"
            f"{long_code}\n"
            "\n"
            f"{long_trailer}\n"
        )

        self.assertEqual(commit_msg.reflow_commit_message_text(message), message)

    def test_writes_reflow_suggestion(self):
        message = (
            "[Infra] Subject\n"
            "\n"
            "This body paragraph is intentionally long enough to require "
            "wrapping while preserving all words in the original order.\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            old_repo_root = commit_msg.REPO_ROOT
            commit_msg.REPO_ROOT = Path(temporary_directory)
            try:
                suggestion_path = commit_msg.write_reflow_suggestion(
                    Path("COMMIT_EDITMSG"),
                    message,
                )
            finally:
                commit_msg.REPO_ROOT = old_repo_root

            self.assertIsNotNone(suggestion_path)
            assert suggestion_path is not None
            self.assertEqual(
                suggestion_path.read_text(encoding="utf-8"),
                "[Infra] Subject\n"
                "\n"
                "This body paragraph is intentionally long enough to require wrapping\n"
                "while preserving all words in the original order.\n",
            )

    def test_ranks_tag_suggestions_from_paths(self):
        self.assertEqual(
            commit_msg.tag_suggestions_for_paths(
                [
                    "loom/src/loom/target/arch/amdgpu/lower/sanitizer.c",
                    "loom/py/loom/dialect/sanitizer/defs.py",
                    "libhrx/src/libhrx/runtime.c",
                    "build_tools/lefthook/commit_msg.py",
                ]
            )[:4],
            ["[Loom]", "[Loom/AMDGPU]", "[HRX]", "[Infra]"],
        )


if __name__ == "__main__":
    unittest.main()
