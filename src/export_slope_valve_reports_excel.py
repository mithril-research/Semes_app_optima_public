"""slope_valve_reports 마크다운 리포트를 Excel 요약표로 변환한다.

예시:
  python export_slope_valve_reports_excel.py \
      reports/rca_slope_valve/slope_valve_reports \
      --out reports/rca_slope_valve/slope_valve_summary.xlsx

입력 리포트는 rca_slope_valve.py가 생성하는 `## 판정 요약` 마크다운
표를 기준으로 파싱한다. Excel writer 의존성을 추가하지 않기 위해
표준 라이브러리 zip/xml만 사용해 .xlsx 파일을 생성한다.
"""
from __future__ import annotations

import argparse
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


OUTPUT_COLUMNS = [
    "웨이퍼명",
    "피크스텝",
    "2순위스텝",
    "의심 밸브",
    "주요센서",
    "방향",
    "1차확인",
    "h역할",
]

LABEL_TO_COLUMN = {
    "peak step": "피크스텝",
    "peakstep": "피크스텝",
    "피크 step": "피크스텝",
    "피크스텝": "피크스텝",
    "2순위 peak": "2순위스텝",
    "2순위peak": "2순위스텝",
    "2순위 피크": "2순위스텝",
    "second peak": "2순위스텝",
    "의심 밸브": "의심 밸브",
    "의심 valve": "의심 밸브",
    "suspect valve": "의심 밸브",
    "suspect valves": "의심 밸브",
    "주요 센서": "주요센서",
    "주요센서": "주요센서",
    "major sensor": "주요센서",
    "major sensors": "주요센서",
    "방향": "방향",
    "direction": "방향",
    "1차 확인": "1차확인",
    "1차확인": "1차확인",
    "primary check": "1차확인",
    "h 역할": "h역할",
    "h역할": "h역할",
    "h role": "h역할",
}

DEFAULT_REPORT_DIRS = [
    Path("reports/rca_slope_valve/slope_valve_reports"),
    Path("slope_valve_reports"),
    Path("reports/slope_valve_reports"),
]


def _display_terms(value: str) -> str:
    text = str(value or "").strip()
    replacements = (
        ("open valves", "열린 밸브"),
        ("open valve", "열린 밸브"),
        ("same valves", "동일 밸브"),
        ("same valve", "동일 밸브"),
        ("Valves", "밸브"),
        ("valves", "밸브"),
        ("Valve", "밸브"),
        ("valve", "밸브"),
    )
    for src, dst in replacements:
        text = text.replace(src, dst)
    return " ".join(text.split())


def _normal_label(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _clean_value(value: str) -> str:
    text = _display_terms(value)
    return text if text else "-"


def _extract_first_step(value: str) -> str:
    text = str(value or "")
    match = re.search(r"(?:step\s*)?(\d+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else "-"


def _extract_suspect_valves(value: str) -> str:
    first_part = str(value or "").split("|", 1)[0].strip()
    first_part = _display_terms(first_part)
    return first_part if first_part else "-"


def _split_markdown_row(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text.startswith("|"):
        return None
    parts = [part.strip() for part in text.strip("|").split("|")]
    if len(parts) < 2:
        return None
    if set("".join(parts)) <= {"-", ":", " "}:
        return None
    label = parts[0]
    if _normal_label(label) in {"항목", "item"}:
        return None
    if len(parts) >= 3:
        value = "|".join(parts[1:-1]).strip()
    else:
        value = parts[1].strip()
    return label, value


def _find_summary_rows(lines: list[str]) -> dict[str, str]:
    start = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("##") and "판정 요약" in line:
            start = i + 1
            break
    if start is None:
        return {}

    rows: dict[str, str] = {}
    in_table = False
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("##") and in_table:
            break
        parsed = _split_markdown_row(stripped)
        if parsed is None:
            if in_table and not stripped:
                break
            continue
        in_table = True
        label, value = parsed
        rows[label] = value
    return rows


def _extract_wafer_name(path: Path, lines: list[str]) -> str:
    for line in lines[:8]:
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        match = re.search(r"\breport\s*-\s*(.+)$", stripped, flags=re.IGNORECASE)
        if match:
            wafer = match.group(1).strip()
            return wafer or path.stem
    return path.stem


def parse_report(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    row = {col: "-" for col in OUTPUT_COLUMNS}
    row["웨이퍼명"] = _extract_wafer_name(path, lines)

    summary = _find_summary_rows(lines)
    for label, raw_value in summary.items():
        out_col = LABEL_TO_COLUMN.get(_normal_label(label))
        if not out_col:
            continue
        value = _clean_value(raw_value)
        if out_col in {"피크스텝", "2순위스텝"}:
            value = _extract_first_step(value)
        elif out_col == "의심 밸브":
            value = _extract_suspect_valves(value)
        row[out_col] = value
    return row


def _discover_report_dir(user_value: str | None) -> Path:
    if user_value:
        return Path(user_value)
    for path in DEFAULT_REPORT_DIRS:
        if path.is_dir():
            return path
    return DEFAULT_REPORT_DIRS[0]


def _find_reports(report_dir: Path, pattern: str) -> list[Path]:
    files = sorted(path for path in report_dir.glob(pattern) if path.is_file())
    return [path for path in files if path.suffix.lower() in {".md", ".txt"}]


def _split_items(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text or text in {"-", "없음", "확인 불가"}:
        return []
    text = text.split("|", 1)[0]
    cleaned = []
    paren_pattern = re.compile(r"([^,;/()]+)\(([^()]*)\)")

    def append_part(part: str) -> None:
        part = part.strip()
        if part and part not in {"-", "없음", "확인 불가"}:
            cleaned.append(part)

    spans = []
    for match in paren_pattern.finditer(text):
        spans.append(match.span())
        append_part(match.group(1))
        for inner in re.split(r"[,;/]+", match.group(2)):
            append_part(inner)

    plain_parts = []
    pos = 0
    for start, end in spans:
        plain_parts.append(text[pos:start])
        pos = end
    plain_parts.append(text[pos:])

    items = re.split(r"[,;/]+", ",".join(plain_parts))
    for item in items:
        part = item.strip()
        if not part or part in {"-", "없음", "확인 불가"}:
            continue
        cleaned.append(part)
    return cleaned


def _mode(values: Iterable[str]) -> str:
    ordered = []
    counts: Counter[str] = Counter()
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in {"-", "없음", "확인 불가"}:
            continue
        if value not in counts:
            ordered.append(value)
        counts[value] += 1
    if not counts:
        return "-"
    return max(ordered, key=lambda item: counts[item])


def _mode_item(values: Iterable[str]) -> str:
    items = []
    for value in values:
        items.extend(_split_items(value))
    return _mode(items)


def append_mode_row(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return rows
    mode_row = {col: "" for col in OUTPUT_COLUMNS}
    mode_row["웨이퍼명"] = "최빈값"
    mode_row["피크스텝"] = _mode(row.get("피크스텝", "") for row in rows)
    mode_row["2순위스텝"] = "-"
    mode_row["의심 밸브"] = _mode_item(row.get("의심 밸브", "") for row in rows)
    mode_row["주요센서"] = _mode_item(row.get("주요센서", "") for row in rows)
    mode_row["방향"] = _mode(row.get("방향", "") for row in rows)
    mode_row["1차확인"] = ""
    mode_row["h역할"] = ""
    return rows + [mode_row]


def _col_letter(index: int) -> str:
    index += 1
    letters = ""
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _cell_xml(row_idx: int, col_idx: int, value: str, style: int = 0) -> str:
    ref = f"{_col_letter(col_idx)}{row_idx}"
    text = escape(str(value if value is not None else ""))
    style_attr = f' s="{style}"' if style else ""
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{text}</t></is></c>'


def _sheet_xml(rows: list[list[str]]) -> str:
    col_widths = [24, 12, 14, 20, 42, 16, 34, 42]
    cols_xml = "".join(
        f'<col min="{i + 1}" max="{i + 1}" width="{width}" customWidth="1"/>'
        for i, width in enumerate(col_widths)
    )
    row_xml = []
    last_row_idx = len(rows)
    for r_idx, row in enumerate(rows, start=1):
        style = 1 if r_idx == 1 else 2 if r_idx == last_row_idx and last_row_idx > 1 else 0
        height = ' ht="24" customHeight="1"' if style else ""
        cells = "".join(_cell_xml(r_idx, c_idx, value, style=style) for c_idx, value in enumerate(row))
        row_xml.append(f'<row r="{r_idx}"{height}>{cells}</row>')
    data = "".join(row_xml)
    last_ref = f"H{max(1, len(rows))}"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
<cols>{cols_xml}</cols>
<sheetData>{data}</sheetData>
<autoFilter ref="A1:{last_ref}"/>
<pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>
</worksheet>"""


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="2">
  <font><sz val="11"/><name val="Calibri"/></font>
  <font><b/><sz val="11"/><name val="Calibri"/></font>
</fonts>
<fills count="3">
  <fill><patternFill patternType="none"/></fill>
  <fill><patternFill patternType="gray125"/></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill>
</fills>
<borders count="2">
  <border><left/><right/><top/><bottom/><diagonal/></border>
  <border>
    <left style="thin"><color rgb="FFB7B7B7"/></left>
    <right style="thin"><color rgb="FFB7B7B7"/></right>
    <top style="thin"><color rgb="FFB7B7B7"/></top>
    <bottom style="thin"><color rgb="FFB7B7B7"/></bottom>
    <diagonal/>
  </border>
</borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="3">
  <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1"/>
  <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
  <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"/>
</cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def write_xlsx(path: Path, rows: list[dict[str, str]]) -> None:
    table = [OUTPUT_COLUMNS] + [[row.get(col, "") for col in OUTPUT_COLUMNS] for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>""")
        zf.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""")
        zf.writestr("xl/workbook.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="summary" sheetId="1" r:id="rId1"/></sheets>
</workbook>""")
        zf.writestr("xl/_rels/workbook.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>""")
        zf.writestr("xl/styles.xml", _styles_xml())
        zf.writestr("xl/worksheets/sheet1.xml", _sheet_xml(table))


def build_excel(report_dir: Path, out_path: Path | None = None, pattern: str = "*.md") -> Path:
    report_dir = report_dir.resolve()
    if not report_dir.is_dir():
        raise FileNotFoundError(f"report directory not found: {report_dir}")
    reports = _find_reports(report_dir, pattern)
    if not reports:
        raise FileNotFoundError(f"no markdown reports found in: {report_dir}")

    rows = [parse_report(path) for path in reports]
    rows = append_mode_row(rows)
    if out_path is None:
        out_path = report_dir.parent / "slope_valve_report_summary.xlsx"
    write_xlsx(out_path.resolve(), rows)
    return out_path.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="slope_valve_reports 디렉터리의 RCA 마크다운 리포트를 Excel 요약표로 변환합니다."
    )
    parser.add_argument(
        "reports_dir",
        nargs="?",
        help="slope_valve_reports 디렉터리. 생략하면 기본 위치를 자동 탐색합니다.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="생성할 .xlsx 경로. 생략하면 reports_dir의 상위 디렉터리에 생성합니다.",
    )
    parser.add_argument(
        "--pattern",
        default="*.md",
        help="읽을 리포트 파일 패턴. 기본값: *.md",
    )
    args = parser.parse_args(argv)

    report_dir = _discover_report_dir(args.reports_dir)
    out_path = Path(args.out) if args.out else None
    result = build_excel(report_dir, out_path=out_path, pattern=args.pattern)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
