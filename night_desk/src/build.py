"""Embed fund.json into the page so the file works on its own (opened from disk)."""
import json, pathlib
here = pathlib.Path(__file__).resolve().parent.parent
fund = json.loads((here / "fund.json").read_text())          # fail loudly on a broken registry
blob = json.dumps(fund, ensure_ascii=False, indent=1).replace("</", "<\\/")
tpl = (here / "src" / "night_desk.template.html").read_text()
assert tpl.count("/*__FUND__*/") == 1
(here / "night_desk.html").write_text(tpl.replace("/*__FUND__*/", blob))
print("built night_desk.html", (here / "night_desk.html").stat().st_size, "bytes")
