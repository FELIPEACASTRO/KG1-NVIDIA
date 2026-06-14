"""Desk-test for kg1_score_live.py with real + synthetic cases.
Uses real boxed strings (chr(92)+'boxed') to avoid any shell-escaping artifacts.
"""
import json
import sys
sys.path.insert(0, r"C:\tmp")
import kg1_score_live as K

B = chr(92) + "boxed{"  # literal backslash + boxed{   (no escaping ambiguity)

def box(payload):
    return B + payload + "}"

# -------------------------------------------------------------------
# 1. SYNTHETIC TAXONOMY (the required representative set)
# -------------------------------------------------------------------
cases = [
    # correct, single closed boxed, binary-strict family
    dict(id="c_ok_bin", family="bit_manipulation", answer="10110",
         raw_output="reasoning here\nThe answer is " + box("10110"),
         finish_reason="stop", completion_tokens=120),
    # correct numeric within tolerance, single boxed
    dict(id="c_ok_num", family="gravity_constant", answer="24.64",
         raw_output="work...\n" + box("24.6401"),
         finish_reason="stop", completion_tokens=200),
    # TRUNCATED: never reached a box, finish_reason length, ct==max
    dict(id="c_trunc", family="equation_transform", answer="84",
         raw_output="Let me think step by step. " * 500,
         finish_reason="length", completion_tokens=7680),
    # NO BOX: falls back to last number, WRONG
    dict(id="c_nobox", family="equation_transform", answer="84",
         raw_output="I think the result is probably 73 somewhere",
         finish_reason="stop", completion_tokens=60),
    # DOUBLE BOXED: two boxes, last (correct) wins but anomaly flagged
    dict(id="c_double", family="numeral_system", answer="XLVII",
         raw_output="first guess " + box("XLI") + " then corrected " + box("xlvii"),
         finish_reason="stop", completion_tokens=90),
    # UNCLOSED BOX: \boxed{ with no closing brace
    dict(id="c_unclosed", family="text_encryption", answer="HELLO",
         raw_output="the cipher decodes to " + B + "HELLO",
         finish_reason="stop", completion_tokens=70),
    # protected family, correct, single box
    dict(id="c_prot", family="unit_conversion", answer="5000",
         raw_output=box("5000"),
         finish_reason="stop", completion_tokens=40),
]

baseline = {"bit_manipulation": 1, "gravity_constant": 1, "equation_transform": 1,
            "numeral_system": 1, "text_encryption": 1, "unit_conversion": 1}
baseline_trunc = {k: 0.0 for k in baseline}

print("=" * 78)
print("CASE A: synthetic taxonomy WITH baseline (expect ABORT: protected+overall+trunc)")
teleA = K.compute_score_telemetry(cases, baseline_per_family=baseline,
                                  baseline_truncation_per_family=baseline_trunc,
                                  step=320, max_step=1600)
print(json.dumps(teleA, indent=2, ensure_ascii=False))

print("=" * 78)
print("MARKERS rendered:")
print(K.render_marker(teleA))

# -------------------------------------------------------------------
# 2. Per-case extraction sanity (proves extractor parity)
# -------------------------------------------------------------------
print("=" * 78)
print("PER-CASE extraction / box-status / correct:")
for c in cases:
    ext = K.extract_final_answer(c["raw_output"])
    ok = K.verify(c["answer"], ext)
    bs = K.box_status(c["raw_output"])
    tr = K.is_truncated(c.get("finish_reason"), c.get("completion_tokens"))
    print(f"  {c['id']:11s} box={bs:12s} trunc={int(tr)} extract={ext!r:20s} correct={ok}")

# -------------------------------------------------------------------
# 3. NO-baseline path -> WATCH + blocker
# -------------------------------------------------------------------
print("=" * 78)
print("CASE B: NO baseline -> decision should be WATCH with blocker baseline_per_item_missing")
teleB = K.compute_score_telemetry(cases, step=320, max_step=1600)
print("decision:", teleB["abort"]["decision"], "blockers:", teleB["abort"]["blockers"],
      "abort_reasons:", teleB["abort"]["abort_reasons"])

# -------------------------------------------------------------------
# 4. HEALTHY checkpoint -> OK  (improves weak, no regression, low trunc)
# -------------------------------------------------------------------
print("=" * 78)
print("CASE C: healthy improving checkpoint -> expect OK")
good = [
    dict(id="g1", family="bit_manipulation", answer="10110", raw_output=box("10110"),
         finish_reason="stop", completion_tokens=100),
    dict(id="g2", family="equation_transform", answer="84", raw_output=box("84"),
         finish_reason="stop", completion_tokens=100),
    dict(id="g3", family="gravity_constant", answer="24.64", raw_output=box("24.64"),
         finish_reason="stop", completion_tokens=100),
    dict(id="g4", family="unit_conversion", answer="5000", raw_output=box("5000"),
         finish_reason="stop", completion_tokens=100),
    dict(id="g5", family="numeral_system", answer="XLVII", raw_output=box("XLVII"),
         finish_reason="stop", completion_tokens=100),
    dict(id="g6", family="text_encryption", answer="HELLO", raw_output=box("HELLO"),
         finish_reason="stop", completion_tokens=100),
]
base_good = {"bit_manipulation": 0, "equation_transform": 0, "gravity_constant": 1,
             "unit_conversion": 1, "numeral_system": 1, "text_encryption": 1}
teleC = K.compute_score_telemetry(good, baseline_per_family=base_good,
                                  baseline_truncation_per_family={k: 0.0 for k in base_good},
                                  step=800, max_step=1600)
print("decision:", teleC["abort"]["decision"],
      "| overall_delta:", teleC["exact_delta"]["overall"],
      "| bit:", teleC["exact_delta"]["bit_manipulation"],
      "| eq:", teleC["exact_delta"]["equation_transform"],
      "| protected:", teleC["exact_delta"]["protected"],
      "| trunc:", teleC["truncation"]["rate"])
print("boxed:", teleC["boxed"])
print("STATUS line:", K.render_marker(teleC).splitlines()[-1])
