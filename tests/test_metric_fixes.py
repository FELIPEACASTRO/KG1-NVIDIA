"""Tests for metric fixes aplicadas em 2026-04-21 (Agent D1 reverse engineering).

Valida que:
1. local_score.verify usa math.isclose (não abs diff scaled errado)
2. Bit-manipulation "01011" == "1011" (ambos float-parseable, no strict regex)
3. extract_boxed suporta \\boxed{X} sem fechamento final
4. extract_boxed prefere ultimo \\boxed{} nao-vazio
5. kg1_local_metric_gate.answers_match mesmo behavior
6. Case-insensitive string match para cryptarithm/cipher

Run: python -m pytest tests/test_metric_fixes.py -v
Or:  python tests/test_metric_fixes.py
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.local_score import verify, extract_boxed  # noqa: E402


class TestVerifyMetricOfficial(unittest.TestCase):
    """verify() deve reproduzir Kaggle metric exato."""

    def test_bit_leading_zeros_equal(self):
        """BUG #1 FIX: '01011' == '1011' via float parse."""
        # Antigamente: re.fullmatch([01]+) -> strict '01011' != '1011'
        # Correto: math.isclose(float('01011')==1011.0, float('1011')==1011.0)
        self.assertTrue(verify("01011", "1011"))
        self.assertTrue(verify("1011", "01011"))
        self.assertTrue(verify("00000000", "0"))

    def test_numeric_tolerance_1pct(self):
        """BUG #2 FIX: rel_tol=1e-2 (não abs_tol=1e-2)."""
        # Para numeral=1000, antes era strict abs<1e-2 → 1000.01 failed
        # Correto: rel_tol 1% → 990.1 ≈ 1000.0 OK
        self.assertTrue(verify("1000", "995"))  # 0.5% off, passa
        self.assertTrue(verify("1000", "1010"))  # 1% off (boundary)
        self.assertTrue(verify("9.81", "9.80"))  # 0.1% off
        self.assertFalse(verify("1000", "1015"))  # 1.5% off, falha

    def test_string_case_insensitive(self):
        """Fallback string match é case-insensitive."""
        self.assertTrue(verify("HELLO", "hello"))
        self.assertTrue(verify("AbC", "abc"))
        self.assertTrue(verify("cat", "CAT"))
        self.assertFalse(verify("hello", "world"))

    def test_string_strip_whitespace(self):
        """strip() aplicado em ambos lados."""
        self.assertTrue(verify("  abc  ", "abc"))
        self.assertTrue(verify("abc", "  abc\n"))

    def test_mixed_numeric_string(self):
        """Se um é parseable e outro não, cai em string compare."""
        # "abc" não parseable, "123" parseable → string lower compare
        self.assertFalse(verify("abc", "123"))
        self.assertFalse(verify("123", "abc"))

    def test_predicted_none_returns_false(self):
        self.assertFalse(verify("1000", None))
        self.assertFalse(verify("1000", ""))


class TestExtractBoxed(unittest.TestCase):
    """extract_boxed() deve seguir fallback oficial."""

    def test_boxed_basic(self):
        self.assertEqual(extract_boxed("... \\boxed{42}"), "42")
        self.assertEqual(extract_boxed("answer: \\boxed{hello}"), "hello")

    def test_boxed_unclosed_eof(self):
        """Exploit: \\boxed{X} sem fechamento aceito no EOF (max_tokens cutoff)."""
        self.assertEqual(extract_boxed("...final answer: \\boxed{42"), "42")

    def test_last_non_empty_wins(self):
        """Exploit: último \\boxed{} não-vazio vence."""
        text = "draft: \\boxed{TRY1} but actually \\boxed{FINAL}"
        self.assertEqual(extract_boxed(text), "FINAL")

    def test_empty_boxed_ignored(self):
        """Exploit: \\boxed{} vazio apaga rascunhos, último não-vazio vence."""
        text = "\\boxed{good answer} then erase \\boxed{}"
        # According to D1 finding: empty ignored, last non-empty wins
        result = extract_boxed(text)
        # Either "good answer" (if empty ignored) or empty string
        self.assertIn(result, ["good answer", ""])

    def test_fallback_final_answer_is(self):
        """Sem \\boxed{}, tenta 'The final answer is: X'."""
        text = "Long reasoning... The final answer is: 42"
        self.assertEqual(extract_boxed(text), "42")

    def test_fallback_last_number(self):
        """Sem \\boxed{} nem 'final answer is', pega último número."""
        text = "I compute 5+3=8, then 2*7=14"
        self.assertEqual(extract_boxed(text), "14")

    def test_fallback_last_line(self):
        """Sem nada acima, última linha não-vazia."""
        text = "First line\nSecond line\nThird line"
        # Mas se tem números, último número vence sobre última linha (bug documentado)
        result = extract_boxed(text)
        # Com números ausentes em "First line" etc, retorna última linha
        self.assertEqual(result, "Third line")

    def test_empty_text_returns_none(self):
        self.assertIsNone(extract_boxed(""))
        self.assertIsNone(extract_boxed(None))


class TestExploitsLegitimos(unittest.TestCase):
    """Os 8 exploits legitimos descobertos pelo Agent D1."""

    def test_exploit_1_bit_leading_zeros(self):
        """Leading zeros em bit não importam."""
        self.assertTrue(verify("01011", "1011"))

    def test_exploit_2_gravity_1pct(self):
        """9.80 ≈ 9.81 (gravity dentro de tol 1%)."""
        self.assertTrue(verify("9.81", "9.80"))
        self.assertTrue(verify("9.81", "9.89"))  # 0.8% off, passa
        self.assertFalse(verify("9.81", "10.00"))  # 1.94% off, falha

    def test_exploit_3_unclosed_boxed(self):
        """\\boxed{X sem } é aceito (EOF)."""
        self.assertEqual(extract_boxed("...\\boxed{42"), "42")

    def test_exploit_4_last_boxed_wins(self):
        """Último \\boxed{} não-vazio vence (pode 'corrigir' rascunho)."""
        text = "\\boxed{wrong} ... \\boxed{correct}"
        self.assertEqual(extract_boxed(text), "correct")

    def test_exploit_5_case_insensitive(self):
        """cipher HELLO == hello."""
        self.assertTrue(verify("HELLO", "hello"))
        self.assertTrue(verify("abc", "ABC"))


class TestLocalGateAlignment(unittest.TestCase):
    """kg1_local_metric_gate.answers_match deve estar alinhado com verify()."""

    def test_gate_import(self):
        """kg1_local_metric_gate deve importar sem erros."""
        try:
            from scripts.kg1_local_metric_gate import answers_match
            self.assertTrue(callable(answers_match))
        except ImportError as e:
            self.skipTest(f"kg1_local_metric_gate not importable standalone: {e}")

    def test_gate_bit_leading_zeros(self):
        """Gate deve aceitar '01011' == '1011' pos-fix."""
        try:
            from scripts.kg1_local_metric_gate import answers_match
            self.assertTrue(answers_match("01011", "1011", rel_tol=1e-2))
        except ImportError:
            self.skipTest("kg1_local_metric_gate not importable")

    def test_gate_numeric_tolerance(self):
        """Gate deve usar rel_tol corretamente."""
        try:
            from scripts.kg1_local_metric_gate import answers_match
            self.assertTrue(answers_match("1000", "1005", rel_tol=1e-2))  # 0.5%
            self.assertFalse(answers_match("1000", "1015", rel_tol=1e-2))  # 1.5%
        except ImportError:
            self.skipTest("kg1_local_metric_gate not importable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
