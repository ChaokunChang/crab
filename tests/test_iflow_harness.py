from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from integrations.sandboxes.iflow.harness import (
    IFLOW_TOOL_OUTPUT_LIMIT_ENV,
    _patch_iflow_bundle_for_output_truncation,
)


class IFlowHarnessTests(unittest.TestCase):
    def test_patch_iflow_bundle_for_output_truncation_is_idempotent(self) -> None:
        original = (
            'prefix)}let T="";if(R.aborted)T=k.t("shellTool.messages.commandCancelled"),'
            'R.output.trim()?T+=` Below is the output (on stdout and stderr) before it was cancelled:\\n'
            '${R.output}`:T+=" There was no output before it was cancelled.";else{let Q=R.error?R.error.message.replace(m,e.command):"(none)";'
            'T=[`Command: ${e.command}`,`Directory: ${e.dir_path||"(root)"}`,'
            '`Stdout: ${R.stdout||"(empty)"}`,`Stderr: ${R.stderr||"(empty)"}`].join(`\\n`)}'
            'let N="";this.config.getDebugMode()?N=T:R.output.trim()?N=R.output:tail'
        )
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "iflow.js"
            bundle_path.write_text(original, encoding="utf-8")

            _patch_iflow_bundle_for_output_truncation(bundle_path)
            first_pass = bundle_path.read_text(encoding="utf-8")
            _patch_iflow_bundle_for_output_truncation(bundle_path)
            second_pass = bundle_path.read_text(encoding="utf-8")

        self.assertEqual(first_pass, second_pass)
        self.assertIn(IFLOW_TOOL_OUTPUT_LIMIT_ENV, first_pass)
        self.assertIn("agentCrTrimToolOutput", first_pass)
        self.assertIn("${agentCrTrimToolOutput(R.output)}", first_pass)
        self.assertIn('`Stdout: ${agentCrTrimToolOutput(R.stdout)||"(empty)"}`', first_pass)
        self.assertIn('`Stderr: ${agentCrTrimToolOutput(R.stderr)||"(empty)"}`', first_pass)
        self.assertIn("R.output.trim()?N=agentCrTrimToolOutput(R.output):", first_pass)


if __name__ == "__main__":
    unittest.main()
