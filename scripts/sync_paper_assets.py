"""Copies the generated LaTeX tables, macros and figures used by overleaf_paper/main.tex
and optionally builds an Overleaf-ready zip in dist/.

The paper directory never holds hand-edited numbers: run this after generate_report.py
and generate_figures.py.
"""
import argparse
import os
import re
import shutil
import sys
import zipfile


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync generated assets into overleaf_paper/")
    parser.add_argument("--results-dir", default="results/mvtec_ad")
    parser.add_argument("--paper-dir", default="overleaf_paper")
    parser.add_argument("--zip", action="store_true", help="Also write dist/overleaf_package.zip")
    args = parser.parse_args()

    tex_path = os.path.join(args.paper_dir, "main.tex")
    with open(tex_path, encoding="utf-8") as f:
        tex = f.read()
    tables = sorted(set(re.findall(r"\\input\{tables/([^}]+)\}", tex)))
    figures = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", tex)))
    # GPU-only assets are referenced through \optionalinput / \optionalfigure and may be absent.
    opt_tables = sorted(set(re.findall(r"\\optionalinput\{tables/([^}]+)\}", tex)))
    opt_figures = sorted(set(re.findall(r"\\optionalfigure(?:\[[^\]]*\])?\{figures/([^}]+)\}", tex)))
    for rel in opt_tables:
        name = rel if rel.endswith(".tex") else rel + ".tex"
        src, dst = os.path.join(args.results_dir, "tables", "latex", name), os.path.join(args.paper_dir, "tables", name)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        elif os.path.exists(dst):
            os.remove(dst)  # never leave a stale optional table behind
    for rel in opt_figures:
        src, dst = os.path.join(args.results_dir, "figures", rel), os.path.join(args.paper_dir, "figures", rel)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        elif os.path.exists(dst):
            os.remove(dst)

    missing = []
    for rel in tables:
        name = rel if rel.endswith(".tex") else rel + ".tex"
        src = os.path.join(args.results_dir, "tables", "latex", name)
        if not os.path.exists(src):
            missing.append(src)
            continue
        os.makedirs(os.path.join(args.paper_dir, "tables"), exist_ok=True)
        shutil.copy2(src, os.path.join(args.paper_dir, "tables", name))
    for rel in figures:
        src = os.path.join(args.results_dir, "figures", rel)
        if not os.path.exists(src):
            missing.append(src)
            continue
        os.makedirs(os.path.join(args.paper_dir, "figures"), exist_ok=True)
        shutil.copy2(src, os.path.join(args.paper_dir, "figures", rel))
    if missing:
        print("Missing generated assets referenced by main.tex:\n  " + "\n  ".join(missing))
        return 1
    print(f"Synced {len(tables)} tables and {len(figures)} figures into {args.paper_dir}/")

    if args.zip:
        os.makedirs("dist", exist_ok=True)
        out = os.path.join("dist", "overleaf_package.zip")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(args.paper_dir):
                for f in files:
                    if f.endswith((".tex", ".bib", ".png", ".pdf")) and not f.startswith("main.pdf"):
                        p = os.path.join(root, f)
                        z.write(p, os.path.relpath(p, args.paper_dir))
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
