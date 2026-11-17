"""Builds PRESENTATION_meeting.pptx from the packet figures (python-pptx). Run: <tools_venv>/bin/python make_deck.py <packet_dir>
All numbers are copied from the guide / CSVs in the packet; sources are in each slide's footer and speaker notes."""
import sys
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

PK = Path(sys.argv[1]); FIG = PK / "figures"; SIM = PK / "simulator_evidence"
RED = RGBColor(0xC0, 0x39, 0x2B); INK = RGBColor(0x22, 0x22, 0x22); GREY = RGBColor(0x66, 0x66, 0x66); BLUE = RGBColor(0x1F, 0x4E, 0x79); LIGHT = RGBColor(0xF3, 0xF6, 0xFA)
prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5); BLANK = prs.slide_layouts[6]
BANNER = "Idealized 1D Coulomb dynamics • 1 seed • no observation noise • preliminary"


def tb(slide, x, y, w, h, text="", size=16, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = s.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    p = tf.paragraphs[0]; p.alignment = align; r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = "Calibri"
    return s


def bullets(slide, x, y, w, h, items, size=16, gap=6):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = s.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, it in enumerate(items):
        lvl = 0
        if isinstance(it, tuple): lvl, it = it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(gap)
        r = p.add_run(); r.text = (("" if it[:2] in ("1.","2.","3.") else "• ") if lvl == 0 else "– ") + it; r.font.size = Pt(size - 2 * lvl); r.font.color.rgb = INK if lvl == 0 else GREY; r.font.name = "Calibri"
        p.level = lvl
        if lvl: p.left_indent = Inches(0.3)
    return s


def title(slide, text, banner=True):
    tb(slide, 0.45, 0.25, 12.4, 0.9, text, size=27, bold=True, color=BLUE, anchor=MSO_ANCHOR.MIDDLE)
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.15), Inches(12.33), Emu(22000)); ln.fill.solid(); ln.fill.fore_color.rgb = RED; ln.line.fill.background()
    if banner: tb(slide, 6.2, 7.08, 6.9, 0.3, BANNER, size=10, color=RED, align=PP_ALIGN.RIGHT)


def footer(slide, text): tb(slide, 0.45, 6.98, 6.9, 0.45, text, size=9, color=GREY)


def image(slide, path, x, y, w, h):
    iw, ih = Image.open(path).size; r = min(w / iw, h / ih); ww, hh = iw * r, ih * r
    slide.shapes.add_picture(str(path), Inches(x + (w - ww) / 2), Inches(y + (h - hh) / 2), Inches(ww), Inches(hh))


def table(slide, x, y, w, rows, colw, size=12, rowh=0.34, header=True):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(rowh * len(rows))).table
    for j, cw in enumerate(colw): t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.text = str(val); c.margin_left = c.margin_right = Inches(0.05); c.margin_top = c.margin_bottom = Inches(0.02)
            for p in c.text_frame.paragraphs:
                for r in p.runs: r.font.size = Pt(size); r.font.name = "Calibri"; r.font.bold = (i == 0 and header); r.font.color.rgb = RGBColor(255, 255, 255) if (i == 0 and header) else INK
            c.fill.solid(); c.fill.fore_color.rgb = BLUE if (i == 0 and header) else (LIGHT if i % 2 == 0 else RGBColor(255, 255, 255))
    return t


def notes(slide, text): slide.notes_slide.notes_text_frame.text = text


def box(slide, x, y, w, h, text, size=15, fill=RGBColor(0xFD, 0xF2, 0xD8)):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); s.fill.solid(); s.fill.fore_color.rgb = fill; s.line.color.rgb = RGBColor(0xC9, 0xA2, 0x27)
    tf = s.text_frame; tf.word_wrap = True; tf.margin_left = tf.margin_right = Inches(0.12); p = tf.paragraphs[0]; r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.color.rgb = INK; r.font.bold = True; r.font.name = "Calibri"; tf.vertical_anchor = MSO_ANCHOR.MIDDLE


def crop(path, l=0.0, t=0.0, r=1.0, b=1.0, name=None):
    im = Image.open(path); w, h = im.size; out = PK / "guide_assets" / "slide_crops"; out.mkdir(exist_ok=True)
    f = out / (name or (Path(path).stem + f"_{int(l*100)}_{int(t*100)}_{int(r*100)}_{int(b*100)}.png")); im.crop((int(l*w), int(t*h), int(r*w), int(b*h))).save(f); return f


def new(t, banner=True):
    s = prs.slides.add_slide(BLANK); title(s, t, banner); return s


# ---- 1 ----
s = new("Does a transformer use the information in a push history — or just a longer one?", banner=False)
tb(s, 0.5, 1.3, 12.3, 0.5, "In-context learning of push dynamics — first end-to-end pilot (CS 182/282A ICL track)", size=18, color=GREY)
bullets(s, 0.6, 2.0, 12.1, 3.6, [
    "Task: predict how far a cube slides from a history of earlier pushes on the same hidden cube (unknown mass and friction).",
    "Weights are frozen at test time: any benefit has to come from reading the history.",
    "What we did: built and audited the pipeline; found a limitation in the original MuJoCo contact scene; trained one model on an idealized 1D simulator; ran E1–E4.",
    "Headline (preliminary): context helps from the very first push — median error 56% → 0.57% — then plateaus at 0.3–0.4% while exact references keep improving.",
], size=20, gap=12)
box(s, 0.6, 5.7, 12.1, 0.95, "Everything shown: idealized 1D Coulomb dynamics implemented in MuJoCo • one training seed • no observation noise • 50–100 test items per experiment", size=16)
footer(s, "Source: packet GUIDE_READ_FIRST.md §0")
notes(s, "SAY: The question is whether a transformer improves because the history carries information about the hidden cube, or only because it is longer. Weights never change at test time. Tonight's contribution is an audited end-to-end pipeline, an honest finding about our original simulator, and first evidence from one trained model. Everything is idealized 1D Coulomb dynamics with one seed and no noise — I will repeat this because it limits what we can claim.\nLIKELY QUESTIONS: 'Why idealized?' -> slide 3-4. 'Is one seed enough?' -> slide 9; no, it is preliminary.")

# ---- 2 ----
s = new("Experiment flow, with one worked example")
steps = ["1. Sample cube (m, μ) and push actions (F, k)", "2. Simulate clean outcomes (ideal 1D Coulomb, in MuJoCo)", "3. Build disjoint train / validation / test sets", "4. Encode histories as push / outcome tokens",
         "5. Train ONE causal transformer (150k steps, seed 0)", "6. Freeze weights; select checkpoint (step 118,000)", "7. Evaluate the SAME checkpoint on E1–E4", "8. Compare with oracle, analytic, physics fit, Bayes", "9. Interpret; choose the next experiment"]
for i, t_ in enumerate(steps):
    y = 1.32 + i * 0.64; sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(y), Inches(5.7), Inches(0.5)); sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor(0xFB, 0xE9, 0xE7) if i in (4, 5, 6) else RGBColor(0xE8, 0xF1, 0xFA); sh.line.color.rgb = GREY
    tf = sh.text_frame; tf.word_wrap = True; pp = tf.paragraphs[0]; pp.alignment = PP_ALIGN.CENTER; rr = pp.add_run(); rr.text = t_; rr.font.size = Pt(13); rr.font.color.rgb = INK; rr.font.name = "Calibri"; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    if i < len(steps) - 1:
        a = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(3.2), Inches(y + 0.5), Inches(0.3), Inches(0.14)); a.fill.solid(); a.fill.fore_color.rgb = GREY; a.line.fill.background()

bullets(s, 6.5, 1.35, 6.4, 5.6, [
    "Hidden: mass m, friction μ. Observed: pushes (F, T) and how far the cube slid d. Query: d for a new push.",
    "Ideal law: d = F T² (F − μ m g) / (2 m² μ g), moving only if F > μ m g.",
    (1, "Example m = 1.0 kg, μ = 0.45, F = 20 N, T = 0.15 s: a = 15.59 m/s² while pushing, −4.41 m/s² after; d = 0.7944 m (simulator 0.7944 m)."),
    "Same force ⇒ one push is enough: d_q = d₁ (T_q/T₁)².",
    "Different forces ⇒ y = d/(F T²) = A F − B is a line that identifies m and μ.",
    "Same-length histories can differ in information: 4 pushes at one force vs 4 at different forces.",
    "E1 = how much context • E2 = what kind • E3 = duplicates • E4 = heavier cubes.",
], size=16, gap=7)
footer(s, "Red = the single training run and its frozen use; weights never change while a history is read. Sources: csv/guide_facts.json (worked example); GUIDE §1–2")
notes(s, "SAY: Walk the pipeline top to bottom, stressing that the SAME frozen checkpoint is used for E1 to E4. Then the worked example: friction force 4.41 N, push acceleration 15.59, coast deceleration 4.41, total 0.7944 m — we checked it by hand, by the closed form, and with the simulator.\nWhy information differs: same-force pushes only rescale by T squared, distinct forces give a line in F whose two coefficients identify m and mu.\nLIKELY QUESTIONS: 'Why does T enter squared?' Both the push distance and the coast distance scale with T squared. 'What does the model see?' Tokens only; never m or mu.")

# ---- 3 ----
s = new("The original MuJoCo contact scene did not represent our assumptions")
image(s, crop(SIM / "contact_reproduction_5e0ae3f_v2" / "fig1_contact_failure_traces.png", 0, 0, 1, 0.485, "fig1_top_row.png"), 0.3, 1.3, 12.7, 3.2)
bullets(s, 0.5, 4.65, 12.3, 2.3, [
    "Verified: the cube hops and loses contact on 104–182 of 150–200 push steps; normal force reaches 8–41 × m g; 105 of 1000 random pushes tumble past 45°. Friction = μ·N whenever it is in contact — the normal load is what breaks.",
    "The hop occurs even with no applied force. ~30 solver settings and five scene redesigns failed; the armature workaround agreed with the formula only through contact chatter (contact 9–20% of steps) and was rejected.",
    "Hypothesis, not proven: the soft contact solver lifts N above m g when F > μN (first-step N matches (mg+μF)/(1+μ²) within 0.2%). A finding about this configuration — not “MuJoCo is wrong”.",
], size=15, gap=6)
footer(s, "REPRODUCTION at historical commit 5e0ae3f: m=0.5 kg, μ=0.6, F=26 N, k=200; yellow = force on; column 3 dashed line = N = m g. Both rows and saved historical evidence: backup slides / packet.")
notes(s, "SAY: Yellow is force on. Column 1: the cube leaps to 136 mm and spins about 180 degrees. Column 2: velocity falls in stair-steps because friction acts only in contact steps. Column 3: normal force over m g; the dashed line is what the ideal model assumes. Column 4: contact flips between 0, 2 and 4 contacts. Friction obeys Coulomb whenever there is contact; the normal load is what breaks.\nBe careful: say 'this configuration', not 'MuJoCo is wrong'. The mechanism is a hypothesis supported by a numerical match on the first step only.\nLIKELY QUESTIONS: 'Smaller time step?' dt was frozen by the protocol; not tried. 'Could contact be fixed?' Possibly with another formulation; a later project (slide 9).\nEVIDENCE: simulator_evidence/ (historical + reproduction), reviews/T01.")

# ---- 4 ----
s = new("Accepted simplification: idealized 1D Coulomb dynamics in MuJoCo")
box(s, 0.5, 1.3, 12.3, 0.9, "The learning pilot uses idealized 1D Coulomb dynamics implemented in MuJoCo. Agreement with the analytic formula is primarily an implementation check, because the simulator explicitly implements that law.", size=16)
bullets(s, 0.5, 2.35, 6.3, 4.6, [
    "One x-slide joint; push force and Coulomb friction applied explicitly each step; stick rule gives exact stiction (F ≤ μ m g ⇒ d = 0).",
    "No contact, rotation, hopping or tipping (max rotation is vacuously 0).",
    "Checks passed (2000 settings): |relative error| median 3.4e-7, 95th pct 2.4e-6; halving dt 1.8e-7; T² scaling 1e-6; hand-written event simulation agrees to 1e-13 m.",
    "Can support: does a model infer hidden parameters from context in a known, exactly solvable family?",
    "Cannot support: anything about realistic contact dynamics; oracle and analytic references are not independent of the simulator.",
], size=15, gap=8)
image(s, crop(FIG / "F03_worked_example.png", 0, 0, 0.5, 0.9, "F03_left.png"), 6.9, 2.35, 6.1, 3.4)
tb(s, 6.9, 5.75, 6.1, 1.2, "Velocity of the worked example (m=1.0, μ=0.45, F=20 N, T=0.15 s; yellow = force on). Solid: saved replay of the final simulator; dashed: hand-derived piecewise motion. Final displacement 0.79437 m (hand) vs 0.79435 m (replay). Agreement tests the implementation, not physics.", size=12, color=GREY)
footer(s, "Sources: simulator_evidence/ideal_simulator_H2_gate_packet; reviews/T01/f28ff1e; GUIDE §4")
notes(s, "SAY: After the contact scene failed, you approved replacing it by explicit 1D Coulomb dynamics inside MuJoCo. We rejected MuJoCo's built-in joint friction because it creeps below the stiction threshold (0.6-0.7 mm). Read the boxed sentence aloud. The 1e-6 agreement is expected and shows correct implementation.\nWHAT IT SUPPORTS: ICL on a known function class, like classic ICL studies, where we know exactly what information suffices.\nLIKELY QUESTIONS: 'Is this circular?' Yes for the physics; the information-content question does not depend on independence. 'Why keep MuJoCo?' Interface and later extension to contact.\nHOUSEKEEPING: the header comment in cube.xml is stale (says frictionloss).")

# ---- 5 ----
s = new("Data, model and training")
table(s, 0.5, 1.35, 6.7, [["Item", "Actual value"],
    ["Cubes", "1,000 train × 32 pushes; 100 val (1,000 fixed rows)"],
    ["Test sets (disjoint cubes)", "E1 100 • E2 50 • E3 50 • E4 50"],
    ["Distributions", "m log-U[0.5,1.5] kg; μ U[0.3,0.6]; F U[13,26] N; T = k·1 ms, k ∈ [100,200]"],
    ["Noise / seed", "none / training seed 0, one run"],
    ["Dataset", "pilot_v1, hash f722c512…"],
    ["Model", "GPT-2-style causal, 12 layers, d=256, 8 heads, 9,485,057 params, 22 tokens"],
    ["Optimization", "AdamW 1e-4, wd 0.01, batch 256, clip 1.0, warmup 2k + cosine"],
    ["Run", "150,000 steps, 33.6 min, 1 RTX 4090, 0.56 GPU-h"],
    ["Checkpoint", "best val 0.06137 at step 118,000 (selected)"]], [1.9, 4.8], size=13, rowh=0.52)
image(s, crop(FIG / "F05_training_curves.png", 0, 0, 0.5, 0.88, "F05_left.png"), 7.4, 1.3, 5.6, 3.1)
bullets(s, 7.4, 4.5, 5.6, 2.5, [
    "Train and validation track; no NaN, no early stop.",
    "Flat after ~3k steps: 99.95% of the validation loss is position 0 (no context).",
    "So “best validation loss” selects by no-context error; last.pt is a sensitivity check only.",
    "Disclosure: 150k steps exceeds PROTOCOL §7's 100k ceiling (set from the benchmark for ~34 min).",
], size=14, gap=5)
footer(s, "Sources: datasets/pilot_v1/meta.json; runs/pilot_v1_n0_s0/config.json, metrics.jsonl; figure F05 (full-run panel); GUIDE §5–6")
notes(s, "SAY: The numbers are read from meta.json and config.json. Normalization uses training data only. Causality, leakage, normalization and overfit checks passed before training. The loss is averaged over all 11 positions; position 0 has no context so it dominates the aggregate; that is why the curves look flat and why checkpoint selection is driven by the no-context error.\nBe upfront: 150,000 steps is above the protocol's 100k 'ceiling'; the pilot spec had no maximum and I set it from measured speed. I did not flag it at launch.\nLIKELY QUESTIONS: 'Why one seed?' Budget for a first look; run-to-run variability unknown. 'Why 22 tokens?' 11 pushes x (push, outcome).")

# ---- 6 ----
s = new("E1: context helps from the first push — and the one-push result survives an audit")
image(s, crop(FIG / "F07_E1_error_vs_context.png", 0, 0, 0.5, 0.88, "F07_left.png"), 0.3, 1.3, 6.4, 3.4)
image(s, crop(FIG / "F08_c1_audit.png", 0.5, 0, 1.0, 0.86, "F08_right.png"), 6.8, 1.3, 6.2, 3.4)
bullets(s, 0.5, 4.8, 12.3, 2.2, [
    "Median error 56% (c=0) → 0.57% (c=1) → 0.34% (c=10). Exact Bayes and the physics fit reach ~1e-7 by c=2; the transformer plateaus (cause untested).",
    "At c=1 exact Bayes gets 0.46% (MSE 8.2e-5 m²) vs the transformer's 0.57% (1.3e-4 m²): about 1.6× worse in MSE.",
    "One push is enough here because the narrow prior truncates the one-parameter family of consistent cubes. Audit: no leakage or indexing defect detected (worse-than-Bayes alone would not prove that).",
], size=15, gap=6)
footer(s, "Left: E1, standardized MSE vs context length c (log), 100 items, best.pt step 118,000. Right: median relative error at c=1 by force gap |F_q−F_c| (transformer, exact Bayes, Bayes posterior sd). Sources: results/v3, c1_audit.")
notes(s, "SAY: Left: standardized MSE vs context length, log axis; the transformer drops four orders of magnitude at c=1 and then stays flat, while Bayes and the physics fit keep dropping to 1e-7 and below (the full figure with median relative error is figure F07). Right: error by force gap — the transformer tracks exact Bayes but is systematically a bit worse; Bayes' own posterior sd (grey) shows how much uncertainty the prior leaves.\nWHY SO GOOD: In the ideal model one observation gives y1 = A F1 - B; the prior ranges are narrow, so the unresolved direction changes the query by under 1%. A property of the setup, not necessarily of the model.\nLEAKAGE: tested the token path, re-simulated stored pushes, shuffled contexts (error jumps to 71%), scaled d1 by 1% (elasticity 0.995 vs Bayes 1.001), wrote the true query outcome into its slot (prediction unchanged, exactly 0.0). 'No leakage detected' — not 'proof of absence'.\nLIKELY QUESTIONS: 'Worse than Bayes at c=0?' No: 1.50 vs 1.53 m2, within noise. 'Why plateau?' Not tested; hypotheses on slide 9.")

# ---- 7 ----
s = new("E2: Bayes shows the information effect; the transformer's accuracy floor hides it")
image(s, crop(FIG / "F09_E2_same_vs_distinct.png", 0, 0, 1, 0.86, "F09_nocap.png"), 0.3, 1.3, 7.9, 3.7)
bullets(s, 8.4, 1.35, 4.6, 3.7, [
    "New-force query, Bayes: same-force history stays ~1.4e-5; distinct forces fall to ~2e-14.",
    "Transformer at c=2 matches Bayes' contrast (+5.2e-5 vs +5.25e-5 m²).",
    "At c=4 it shows none (−9e-6 m², CI includes 0): its floor (2–3e-5) is as large as the effect.",
], size=15, gap=8)
table(s, 0.5, 5.15, 7.6, [["Standardized MSE (new-force query)", "c=2", "c=4"], ["Transformer, same-force (B)", "4.1e-5", "2.1e-5"], ["Transformer, distinct-force (D)", "1.9e-5", "2.5e-5"],
    ["Bayes, same-force (B)", "2.3e-5", "1.4e-5"], ["Bayes, distinct-force (D)", "1e-10", "2e-14"]], [4.1, 1.75, 1.75], size=12, rowh=0.34)
bullets(s, 8.4, 5.1, 4.6, 1.8, ["E2 as designed, without noise, cannot yet separate information from tokens for this model.", "50 paired items; intervals are over items, not seeds."], size=14, gap=5)
footer(s, "Solid = same-force context, dotted = distinct-force, every method; log axis. Sources: results/v3 e2_same_vs_distinct.csv, e2_paired.csv.")
notes(s, "SAY: Solid lines are same-force context, dotted are distinct-force, for every method. In the right panel the blue Bayes curves separate by ten orders of magnitude, so the design works. The red transformer curves stay together near 2e-5. At two observations the transformer's same-minus-distinct gap equals Bayes'; at four it vanishes. Its same-force result is near Bayes; it cannot reach Bayes' distinct-force accuracy because of its own floor.\nWHAT THIS MEANS: we cannot say the transformer ignores diversity. We can say this experiment is limited by precision, which motivates noise (raises Bayes' floor) and a floor diagnosis.\nPAIRED NUMBERS (new-force query, B minus D, squared error m^2): Bayes +5.25e-5 at c=2 (CI 2.5e-5 to 8.5e-5), +3.15e-5 at c=4 (1.9e-5 to 4.6e-5); transformer +5.2e-5 at c=2 (1.5e-5 to 1.0e-4), -9.3e-6 at c=4 (-6.6e-5 to +2.8e-5).\nLIKELY QUESTIONS: 'Did the transformer learn diversity?' At c=2 it behaves like Bayes. 'Why different at c=4?' Untested.")

# ---- 8 ----
s = new("E3 and E4 (both preliminary)")
image(s, crop(FIG / "F10_E3_redundancy.png", 0, 0, 0.5, 0.91, "F10_left.png"), 0.3, 1.3, 6.3, 3.2)
image(s, crop(FIG / "F12_e4_prior_comparison.png", 0.5, 0, 1.0, 0.9, "F12_right.png"), 6.8, 1.3, 6.2, 3.2)
tb(s, 6.8, 4.28, 6.2, 0.3, "red: transformer • blue: Bayes, training prior • green: Bayes, broad prior (y: median relative error, log)", size=11, color=GREY)
bullets(s, 0.5, 4.6, 6.1, 2.4, [
    "E3: appending exact copies leaves every reference unchanged but moves the transformer's prediction by 3.2 / 1.9 / 2.6 mm (k = 1/2/4, mean; max 11 mm); its error does not change materially.",
    "Preliminary no-noise redundancy test. This does not test the benefit of independent noisy repeated measurements.",
], size=14, gap=6)
bullets(s, 6.9, 4.65, 6.1, 2.35, [
    "E4: for heavy cubes (m 1.5–2.0 kg) the context gain persists (544% → 2.3%), but relative error is ~7× the in-distribution value (2.3% vs 0.34% at c=10).",
    "Training-prior Bayes is biased here (truncated at m = 1.5); broad-prior Bayes is ≈ exact. 50 cubes, one seed: preliminary heavy-OOD behaviour, not generalization.",
], size=14, gap=6)
footer(s, "Left: E3 mean |prediction change| vs appended copies (÷ train std of d). Right: E4 heavy cubes, median relative error by prior. Sources: results/v3 e3_*, e4_*; csv/supp_broad_prior_bayes.")
notes(s, "SAY: E3: with no noise a copy and a fresh repeat are identical numbers, so this only tests robustness to redundancy. The prediction shifts by a couple of millimetres — about a third of the model's typical error — without systematically hurting.\nE4: standardized MSE looks similar for heavy and normal cubes only because heavy cubes slide shorter; relative error is 7x worse. The training-prior Bayes is truncated at 1.5 kg so it is the wrong yardstick; the broad-prior Bayes is nearly exact, so the transformer is far from the best achievable. Do not say 'generalizes'.\nLIKELY QUESTIONS: 'What would a stronger E3 look like?' Noise, fresh vs copy. 'Which prior did you use?' Both shown, labelled.")

# ---- 9 ----
s = new("Limitations and recommended next experiments")
bullets(s, 0.5, 1.35, 5.9, 5.6, [
    "Idealized dynamics; realistic contact not tested (simulator failure is a finding).",
    "One training seed; intervals are across items, not seeds.",
    "No observation noise: E3's main question and E2's information effect are untested/hidden.",
    "Small tests (100/50/50/50); structured test histories differ from training histories.",
    "Checkpoint selection driven by the no-context position; last.pt shifts c ≥ 1 MSE by 1–18%.",
    "Short, small training (1,000 cubes, 34 min); plateau cause untested.",
    "Review gaps: control-authored scripts and the final packet were audited only partly; the Bayes reviewer's adversarial synthetic set was 19 of 23 cases complete.",
], size=14, gap=5)
tb(s, 6.7, 1.3, 6.2, 0.4, "Next experiments, ranked (resource figures are estimates)", size=16, bold=True, color=BLUE)
bullets(s, 6.7, 1.95, 6.2, 5.0, [
    "1. Observation noise + diagnose the accuracy floor. Question: does the model value independent information over copies/length? Noise lifts Bayes' floor above the transformer's. Needs the noise path, retraining; ~3.4 GPU-h for 3 seeds × 2 noise levels.",
    "2. Why the plateau? Per-position loss weights / log target, 10× more cubes, longer training, full-precision matmul. ~2.4 GPU-h for 4 ablations.",
    "3. Broader prior and action ranges so one push is not decisive. One dataset + one run ≈ 0.6 GPU-h.",
    "Not in the top three: repairing the contact simulator — answers realism, not information content; later extension.",
], size=14, gap=7)
footer(s, "Sources: GUIDE §9; rates from the measured run (150k steps = 33.6 min)")
notes(s, "SAY: Rank by what the findings say. Noise is first because Bayes' information effect (about 1e-5) is no larger than the transformer's floor, so without noise E2 cannot discriminate and E3 cannot separate copies from fresh repeats. The floor diagnosis gates the value of that experiment. A broader prior addresses the one-observation easiness.\nResource figures are estimates from measured rates: 150k steps = 33.6 min on one RTX 4090; dataset generation 8 s; Bayes evaluation about 30 s.\nBe honest on review gaps: the loader and figure code were audited and two label bugs fixed by me without re-review; the guide and deck were fact-checked only as stated in the packet.\nLIKELY QUESTIONS: 'Why not fix the simulator first?' Different question; unbounded time.")

# ---- 10 ----
s = new("Questions for guidance")
rows = [["#", "Question", "Decision it informs"],
 ["1", "Is idealized 1D Coulomb dynamics an acceptable primary setting, with contact as a later extension?", "Continue on the ideal model vs invest in simulator repair first"],
 ["2", "Is the narrow (m, μ) prior making one observation too decisive to expose the information question?", "Whether broader priors/actions go before or with noise"],
 ["3", "Noisy repeated measurements or broader parameter/action distributions first?", "Order of experiments; whether Bayes is re-validated at σ_d > 0"],
 ["4", "Which comparison best separates useful information from extra tokens?", "The main metric and figure"],
 ["5", "Should checkpoint selection report context-conditioned validation loss separately (and select on it)?", "Training-log change and selection rule"],
 ["6", "What baseline coverage and how many seeds are enough for the next milestone?", "Compute budget (seeds × noise conditions)"]]
table(s, 0.5, 1.4, 12.3, rows, [0.5, 7.4, 4.4], size=14, rowh=0.8)
footer(s, "Source: GUIDE §10")
notes(s, "SAY: These are the decisions I need guidance on. Question 1 and 3 together set the plan: ideal model plus noise is the cheapest path to a real test of the hypothesis. Question 5 is a concrete code change. Question 6 sets the budget.\nIf asked for my recommendation: noise + floor diagnosis, then broaden priors; keep contact as a later extension.")

# ---- backups ----
def backup(t): s = new("Backup — " + t); return s
s = backup("model and training hyperparameters")
table(s, 0.5, 1.35, 12.3, [["Item", "Value (source: model/gpt2.py, train/loop.py, config.json)"],
 ["Architecture", "pre-LN GPT-2 style; 12 layers; d=256; 8 heads (size 32); MLP 256→1024→256 GELU; learned positions (22); Linear(5,256) input; Linear(256,1) head; no dropout; no weight tying"],
 ["Parameters", "9,485,057"], ["Tokens / shapes", "push [z(F), z(T), 0,0,0]; outcome [0,0,z(d_obs), z(asinh(d_obs/0.01)), 1]; tokens (B,22,5) → readouts (B,22) → push readouts (B,11)"],
 ["Loss", "mean squared error of standardized clean displacement over all 11 push positions (equal weights)"],
 ["Optimizer", "AdamW lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0.01 (all parameters), batch 256, no accumulation, grad-clip norm 1.0"],
 ["Schedule", "linear warmup 2,000 steps, cosine to 0 over 150,000 steps; validation every 1,000 steps; best.pt on lowest validation loss"],
 ["Precision", "float32; TF32 matmul permitted; no autocast"], ["Run", "2,016 s wall, 2,012 s GPU, ≈420k tokens/s, 1 × RTX 4090, ≈ 845M tokens; source e6c51af (clean)"],
 ["Checkpoint", "best.pt step 118,000 sha256 6101448e…f230; last.pt step 150,000 sha256 f645625d…c85a"]], [2.0, 10.3], size=12, rowh=0.55)
notes(s, "Use if asked for exact hyperparameters. All values verified in code and config.json. Pre-training checks: shape, causality (max diff < 1e-6), train-only normalization, leakage, fixed-set overfit (lr 5e-4, 16k step cap — a plumbing test, not evidence of ICL).")
s = backup("token layout and why the query outcome is hidden"); image(s, FIG / "F02_token_diagram.png", 0.5, 1.4, 12.3, 5.4)
notes(s, "The readout at the query push token cannot attend to the query's own outcome because attention is causal and the outcome token comes after. Writing the true outcome there changes the prediction by exactly 0.0.")
s = backup("per-position validation loss and checkpoint sensitivity"); image(s, FIG / "F06_per_position_val_loss.png", 0.3, 1.3, 6.3, 3.4); image(s, FIG / "B01_E1_last_checkpoint_sensitivity.png", 6.7, 1.3, 6.4, 3.4)
bullets(s, 0.5, 4.9, 12.3, 2.1, ["Position 0 is 99.95% (best.pt) / 99.94% (last.pt) of the mean validation loss.", "Sensitivity (separate from the selected checkpoint): E1 with last.pt changes c ≥ 1 MSE by 1–18% in both directions, c=0 by a few percent.", "Report context-conditioned validation loss separately next time."], size=14)
notes(s, "last.pt results are in csv/results_v4_last_ckpt_sensitivity. The primary results use best.pt.")
s = backup("reference methods and what each knows")
table(s, 0.5, 1.35, 12.3, [["Reference", "What it is", "Knows true (m, μ)?"],
 ["Simulator oracle", "the clean simulator output itself (error 0 by definition)", "yes"], ["Analytic, true parameters", "closed form at the true (m, μ)", "yes"],
 ["Physics fit", "least squares of d on (F²T², −FT²); “not determined” if < 2 moving pushes spanning ≥ 3 N; same-force rule when the query force is in context", "no"],
 ["Training-prior Bayes", "posterior mean, prior m log-U[0.5,1.5], μ U[0.3,0.6], τ = 9.5e-7·f + 1e-4 m; adaptive nested quadrature", "no"],
 ["Broad-prior Bayes", "prior m log-U[0.35,2.0], μ U[0.2,0.7] (used for E4 comparison, outside the harness)", "no"],
 ["Exact ideal Bayes (audit)", "σ→0 posterior on the constraint curve, plain numpy, c=1 only; separate code", "no"]], [2.6, 8.2, 1.5], size=13, rowh=0.62)
notes(s, "Oracle and analytic are ceilings, not competitors. The transformer knows neither the law nor the parameters. Bayes was reviewed against an independent implementation (median 2e-15 relative difference over 80 real cases). Do not conflate the audit's exact Bayes with the project's discrepancy-aware Bayes.")
s = backup("contact failure: 1000 random pushes and the no-force test"); image(s, SIM / "contact_reproduction_5e0ae3f_v2" / "fig2_random_1000.png", 0.3, 1.3, 12.7, 3.0); image(s, SIM / "contact_reproduction_5e0ae3f_v2" / "fig3_no_force_sliding.png", 1.3, 4.35, 10.7, 2.6)
notes(s, "Reproduction at commit 5e0ae3f. 906/1000 rotate >1 deg, 188 >10, 105 >45; median |relative error| 3e-4, 95th percentile 9.7e-2, max 0.74. The no-force test shows hopping without any applied force.")
s = backup("attempted simulator fixes")
table(s, 0.5, 1.35, 12.3, [["Attempt", "Outcome", "Decision"],
 ["solimp tweak", "met resting-height tolerance; hop/tumble remained", "kept as softness choice, not a fix"], ["~30 contact/solver settings", "all ~180° rotation/no contact, or could not settle", "rejected"],
 ["pair solreffriction", "0.4–1.9° but viscous friction; d off 7%–1000%", "rejected"], ["x–y slide only", "N = 0 exactly ⇒ no friction", "rejected"],
 ["x–y–z slides (free/ranged/spring/damper)", "contact dropouts; N/mg 0–110", "rejected"], ["armature 1000 kg", "flat, d within 0.37% median; contact 9–20%, N spikes ≈100 m g", "rejected"],
 ["MuJoCo frictionloss = μ m g", "creeps 0.63–0.70 mm below threshold", "rejected"], ["explicit Coulomb force, 1D", "exact ideal law; all checks passed", "accepted"]], [3.6, 6.1, 2.6], size=13, rowh=0.55)
notes(s, "Numbers for fixes are the writers' saved measurements (docs/questions/T01.md, docs/evidence); only the failure of the original scene was reproduced for this packet.")
s = backup("parameter ambiguity: one observation cannot separate two cubes, yet predictions agree"); image(s, FIG / "F04_parameter_ambiguity.png", 0.5, 1.4, 12.3, 5.3)
notes(s, "Cube A (1.0 kg, 0.45) and cube B (1.2 kg, 0.3244) give identical displacement at F=20 N, T=0.15 s but differ by 0.75% at F=24 N. This is why one push can predict well while parameters stay ambiguous.")
s = backup("E4 detail"); image(s, FIG / "F11_E4_heavy_ood.png", 0.4, 1.4, 12.5, 5.3)
notes(s, "Standardized MSE in the figure; relative errors are in the table on slide 8 notes and the guide. 50 heavy cubes, 1 seed.")
prs.save(PK / "PRESENTATION_meeting.pptx"); print("saved", len(prs.slides), "slides")
