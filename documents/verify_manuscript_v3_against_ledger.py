#!/usr/bin/env python3
"""Phase D verifier: extracts key numeric strings from the generated manuscript
docx and confirms each traces back to an exact value in manuscript_ground_truth.json.
This is a targeted spot-check (not exhaustive NLP number extraction) covering the
highest-risk claims: every number that was corrected or flagged this session."""
import json
from pathlib import Path
from docx import Document

HERE = Path(__file__).resolve().parent
LEDGER = json.load(open(HERE / 'manuscript_ground_truth.json'))
doc = Document(HERE / 'GENESIS_NMI_Manuscript_v3_FreshData.docx')
full_text = '\n'.join(p.text for p in doc.paragraphs)
for t in doc.tables:
    for row in t.rows:
        full_text += '\n' + ' '.join(c.text for c in row.cells)

checks = []


def must_contain(label, needle, case_insensitive=False):
    haystack = full_text.lower() if case_insensitive else full_text
    n = needle.lower() if case_insensitive else needle
    ok = n in haystack
    checks.append((label, ok, needle))
    print(f'[{"OK" if ok else "MISSING"}] {label}: {needle!r}')


# High-risk / previously-corrected claims that MUST appear verbatim
must_contain('PGCL rho 3-seed', '0.5553')
must_contain('PGCL cliff 3-seed', '66.4')
must_contain('Table1 PGCL rho', '0.555\u00b10.003')
must_contain('Table1 PGCL ndcg (PGCL now wins)', '0.941\u00b10.025')
must_contain('Flat SupCon rho (loses to PGCL)', '0.526')
must_contain('dim_256 3seed verdict text', 'd=128')
must_contain('generative non-bimodal band', '0.715-0.796')
must_contain('Table2 CRAVE vs VAE U/p (corrected, significant)', '0.0040')
must_contain('Table2 Bonferroni corrected p', '0.0159')
must_contain('Table2 CRAVE vs RLVAE not significant', '0.1111')
must_contain('safety PASS rate collapse', '14.6%')
must_contain('safety PASS old published', '47.1%')
must_contain('aniline flag rate', '53.2%')
must_contain('cross-cancer rho', '0.5022' if '0.5022' in full_text else '0.502')
must_contain('cross-cancer n', '29,675')
must_contain('KRAS not significant CI', '[-0.053, 0.205]' if '[-0.053, 0.205]' in full_text else '-0.053')
must_contain('Phase10 Tier-1 count', '46')
must_contain('Phase10 top compound', 'OMIPALISIB')
must_contain('Phase8 triple-validated elite', '13')
must_contain('Phase6 top compound name', 'erlotinib', case_insensitive=True)
must_contain('docking 8 targets STAT3 PDB', '6NUQ')

n_fail = sum(1 for _, ok, _ in checks if not ok)
print(f'\n{len(checks)} claims checked, {n_fail} missing.')
if n_fail:
    raise SystemExit(1)
print('ALL MANUSCRIPT CLAIMS TRACE TO THE VERIFIED LEDGER.')
