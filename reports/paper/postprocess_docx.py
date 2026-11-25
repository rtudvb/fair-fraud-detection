"""Give the Word output the same layout as the PDF (Quarto post-render hook, see _quarto.yml).

Pandoc's .docx has its own look; this script applies the thesis conventions that style.typ
applies to the PDF:

- 標楷體 / Times New Roman 12 pt, 1.5 line spacing, two-character first-line indent;
- chapters numbered 第一章, 第二章 and sections 1.1, 1.2 with Word's own list numbering;
- figures and tables numbered per chapter (圖 1-1, 表 2-3), also in cross-references;
- three-line tables (三線表);
- 目錄、圖目錄、表目錄 as Word fields (Word fills them in when the file is opened);
- A4 pages with page numbers.

Run by hand: python postprocess_docx.py _output/index.docx
"""

from __future__ import annotations

import os
import re
import sys
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm

FONT_LATIN = "Times New Roman"
FONT_CJK = "標楷體"
CITATION_COLOR = "C00000"  # same red as citations.lua uses for the PDF
UNNUMBERED_CHAPTERS = {"摘要", "Abstract", "參考文獻", "附錄"}
CAPTION = re.compile(r"^(表|圖)\s*(\d+)\s*[:：]\s*(.*)$", re.S)
REFERENCE = re.compile(r"^(表|圖)\s*(\d+)$")

# Child order required by the OOXML schema (Word rejects files that break it).
ORDER = {
    "pPr": "pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr "
    "suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct "
    "topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind "
    "contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment "
    "textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange",
    "rPr": "rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss "
    "imprint noProof snapToGrid vanish webHidden color spacing w kern position sz szCs "
    "highlight u effect bdr shd fitText vertAlign rtl cs em lang eastAsianLayout specVanish",
    "tblPr": "tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize "
    "tblW jc tblCellSpacing tblInd tblBorders shd tblLayout tblCellMar tblLook",
    "tcPr": "cnfStyle tcW gridSpan hMerge vMerge tcBorders shd noWrap tcMar textDirection "
    "tcFitText vAlign hideMark",
    "settings": "updateFields hdrShapeDefaults footnotePr endnotePr compat docVars rsids "
    "mathPr attachedSchema themeFontLang clrSchemeMapping doNotIncludeSubdocsInStats "
    "doNotAutoCompressPictures forceUpgrade captions readModeInkLockDown smartTagType "
    "schemaLibrary shapeDefaults doNotEmbedSmartTags decimalSymbol listSeparator",
}


def element(tag: str, **attrs: str):
    el = OxmlElement(tag)
    for key, value in attrs.items():
        el.set(qn(f"w:{key}"), value)
    return el


def local(el) -> str:
    return el.tag.rsplit("}", 1)[-1]


def put(parent, child):
    """Insert `child` into `parent` at its schema position, replacing a same-named child."""
    order = ORDER[local(parent)].split()
    name = local(child)
    for old in parent.findall(child.tag):
        parent.remove(old)
    rank = order.index(name)
    for existing in parent:
        if local(existing) in order and order.index(local(existing)) > rank:
            existing.addprevious(child)
            return child
    parent.append(child)
    return child


def get_or_add(parent, tag: str, position: int | None = None):
    child = parent.find(qn(tag))
    if child is None:
        child = element(tag)
        if position is None:
            parent.append(child)
        else:
            parent.insert(position, child)
    return child


def find_style(doc, style_id: str):
    return doc.styles.element.find(f"{qn('w:style')}[@{qn('w:styleId')}='{style_id}']")


def set_style(
    doc,
    style_id: str,
    *,
    size: float,
    bold: bool = False,
    align: str | None = None,
    line: int = 360,
    before: int = 0,
    after: int = 0,
    first_line_chars: int | None = None,
    hanging: int | None = None,
    page_break: bool = False,
    keep_next: bool = False,
    color: str = "000000",
) -> None:
    """Rewrite one paragraph style. Spacing is in twentieths of a point, line in 240ths."""
    style = find_style(doc, style_id)
    if style is None:
        style = element("w:style", type="paragraph", styleId=style_id)
        style.append(element("w:name", val=style_id))
        style.append(element("w:basedOn", val="Normal"))
        doc.styles.element.append(style)
    for old in style.findall(qn("w:pPr")) + style.findall(qn("w:rPr")):
        style.remove(old)
    ppr, rpr = element("w:pPr"), element("w:rPr")
    style.append(ppr)
    style.append(rpr)

    if keep_next:
        put(ppr, element("w:keepNext"))
    if page_break:
        put(ppr, element("w:pageBreakBefore"))
    put(
        ppr,
        element("w:spacing", before=str(before), after=str(after), line=str(line), lineRule="auto"),
    )
    if first_line_chars is not None:
        # Word reads firstLineChars; LibreOffice reads firstLine (12 pt x 2 characters).
        put(
            ppr,
            element(
                "w:ind",
                firstLineChars=str(first_line_chars),
                firstLine=str(first_line_chars * 240 // 100),
            ),
        )
    elif hanging is not None:
        put(ppr, element("w:ind", left=str(hanging), hanging=str(hanging)))
    else:
        put(ppr, element("w:ind", left="0", firstLine="0"))
    if align:
        put(ppr, element("w:jc", val=align))
    if re.fullmatch(r"Heading\d", style_id):
        put(ppr, element("w:outlineLvl", val=str(int(style_id[-1]) - 1)))

    put(
        rpr,
        element("w:rFonts", ascii=FONT_LATIN, hAnsi=FONT_LATIN, eastAsia=FONT_CJK, cs=FONT_LATIN),
    )
    put(rpr, element("w:b", val="1" if bold else "0"))
    put(rpr, element("w:bCs", val="1" if bold else "0"))
    put(rpr, element("w:i", val="0"))
    put(rpr, element("w:color", val=color))
    put(rpr, element("w:sz", val=str(int(size * 2))))
    put(rpr, element("w:szCs", val=str(int(size * 2))))
    put(rpr, element("w:lang", val="en-US", eastAsia="zh-TW"))


def apply_styles(doc) -> None:
    body = dict(size=12, align="both", first_line_chars=200)
    heading = dict(bold=True, keep_next=True)
    set_style(doc, "Normal", size=12)
    set_style(doc, "BodyText", **body)
    set_style(doc, "FirstParagraph", **body)
    set_style(doc, "Compact", size=12)
    set_style(doc, "Title", size=20, bold=True, align="center", before=2400, after=240)
    set_style(doc, "Subtitle", size=14, align="center", after=960)
    set_style(doc, "Author", size=14, align="center", after=120)
    set_style(doc, "Date", size=12, align="center")
    set_style(doc, "Heading1", size=16, align="center", after=360, page_break=True, **heading)
    set_style(doc, "Heading2", size=14, before=240, after=120, **heading)
    set_style(doc, "Heading3", size=12, before=120, after=60, **heading)
    set_style(
        doc, "FrontMatterHeading", size=16, align="center", after=360, page_break=True, **heading
    )
    set_style(doc, "TableCaption", size=11, align="center", before=240, after=60, keep_next=True)
    set_style(doc, "ImageCaption", size=11, align="center", before=60, after=240)
    set_style(doc, "Bibliography", size=12, hanging=480, align="left")
    # The repository link under the title, in red so it stands out.
    set_style(doc, "RepoLink", size=12, bold=True, align="center", before=240, color=CITATION_COLOR)
    # The simulator page after the abstract: title, screenshot and a short note.
    set_style(
        doc,
        "ShowcaseTitle",
        size=16,
        bold=True,
        align="center",
        after=360,
        page_break=True,
        keep_next=True,
    )
    set_style(doc, "ShowcaseImage", size=12, align="center", line=240, keep_next=True)
    # 「註：」 under a table: smaller, flush left, close to the table (APA-style table note).
    set_style(doc, "TableNote", size=10.5, align="both", line=300, after=120)
    set_style(doc, "ShowcaseNote", size=12, align="both", before=240, first_line_chars=200)

    # Citations (marked by citations.lua) in the drafting colour.
    citation = find_style(doc, "Citation")
    if citation is not None:
        for old in citation.findall(qn("w:rPr")):
            citation.remove(old)
        rpr = element("w:rPr")
        put(rpr, element("w:color", val=CITATION_COLOR))
        citation.append(rpr)

    # Cross-references and URLs: black, not underlined (thesis convention).
    link = find_style(doc, "Hyperlink")
    if link is not None:
        for old in link.findall(qn("w:rPr")):
            link.remove(old)
        rpr = element("w:rPr")
        put(rpr, element("w:color", val="000000"))
        link.append(rpr)


def color_repo_link(doc) -> None:
    """Keep the URL in the RepoLink paragraphs red; the Hyperlink style would make it black."""
    for p in doc.paragraphs:
        if p.style is None or p.style.style_id != "RepoLink":
            continue
        for run in p._p.iter(qn("w:r")):
            rpr = run_props(run)
            for old in rpr.findall(qn("w:color")):
                rpr.remove(old)
            put(rpr, element("w:color", val=CITATION_COLOR))


def add_heading_numbering(doc) -> None:
    """Attach a 第一章 / 1.1 / 1.1.1 multilevel list to the Heading 1-3 styles."""
    numbering = doc.part.numbering_part.element
    abstract_id = 1 + max(
        (int(n.get(qn("w:abstractNumId"))) for n in numbering.findall(qn("w:abstractNum"))),
        default=0,
    )
    num_id = 1 + max((int(n.get(qn("w:numId"))) for n in numbering.findall(qn("w:num"))), default=0)
    abstract_id, num_id = str(abstract_id), str(num_id)

    abstract = element("w:abstractNum", abstractNumId=abstract_id)
    abstract.append(element("w:multiLevelType", val="multilevel"))
    levels = [
        ("taiwaneseCountingThousand", "第%1章", "Heading1"),
        ("decimal", "%1.%2", "Heading2"),
        ("decimal", "%1.%2.%3", "Heading3"),
    ]
    for ilvl, (fmt, text, style_id) in enumerate(levels):
        lvl = element("w:lvl", ilvl=str(ilvl))
        lvl.append(element("w:start", val="1"))
        lvl.append(element("w:numFmt", val=fmt))
        lvl.append(element("w:pStyle", val=style_id))
        if ilvl:
            lvl.append(element("w:isLgl"))  # show the chapter as 1, not 一, in 1.1
        lvl.append(element("w:suff", val="space"))
        lvl.append(element("w:lvlText", val=text))
        lvl.append(element("w:lvlJc", val="left"))
        ppr = element("w:pPr")
        ppr.append(element("w:ind", left="0", firstLine="0"))
        lvl.append(ppr)
        abstract.append(lvl)
        num_pr = element("w:numPr")
        num_pr.append(element("w:ilvl", val=str(ilvl)))
        num_pr.append(element("w:numId", val=num_id))
        put(find_style(doc, style_id).find(qn("w:pPr")), num_pr)

    first_num = numbering.find(qn("w:num"))  # abstractNum elements precede num elements
    if first_num is not None:
        first_num.addprevious(abstract)
    else:
        numbering.append(abstract)
    num = element("w:num", numId=num_id)
    num.append(element("w:abstractNumId", val=abstract_id))
    numbering.append(num)


def paragraph_style(p) -> str | None:
    style = p.find(f"{qn('w:pPr')}/{qn('w:pStyle')}")
    return style.get(qn("w:val")) if style is not None else None


def paragraph_text(p) -> str:
    return "".join(t.text or "" for t in p.iter(qn("w:t"))).strip()


def paragraph_props(p):
    """Return the paragraph's one <w:pPr> (Quarto writes two into captions; keep the styled)."""
    pprs = p.findall(qn("w:pPr"))
    keep = next((x for x in pprs if x.find(qn("w:pStyle")) is not None), None)
    keep = keep if keep is not None else (pprs[0] if pprs else None)
    for x in pprs:
        if x is not keep:
            p.remove(x)
    if keep is None:
        keep = element("w:pPr")
        p.insert(0, keep)
    for jc in keep.findall(qn("w:jc"))[1:]:  # table cells: column alignment, then a stray center
        keep.remove(jc)
    return keep


def run_props(run):
    return get_or_add(run, "w:rPr", position=0)


def set_text(container, text: str) -> None:
    """Replace the text of a paragraph or hyperlink, keeping the first run's formatting."""
    runs = container.findall(qn("w:r"))
    for run in runs[1:]:
        container.remove(run)
    t = runs[0].find(qn("w:t"))
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def text_run(text: str):
    run, t = element("w:r"), element("w:t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    run.append(t)
    return run


def tab_run():
    run = element("w:r")
    run.append(element("w:tab"))
    return run


CHINESE_NUMERALS = "一二三四五六七八九十"


class References:
    """Word bookmarks on headings, captions and equation numbers, and the REF fields to them."""

    def __init__(self) -> None:
        self.targets: dict[str, tuple[str, str, str]] = {}  # anchor -> (bookmark, text, switches)
        self.next_id = 90000

    def mark(self, anchor: str, first, last, text: str, switches: str = "\\h") -> None:
        """Bookmark the runs ``first``..``last`` and register them as the target of ``anchor``."""
        self.next_id += 1
        name = f"_Ref{self.next_id}"
        first.addprevious(element("w:bookmarkStart", id=str(self.next_id), name=name))
        last.addnext(element("w:bookmarkEnd", id=str(self.next_id)))
        self.targets[anchor] = (name, text, switches)


def field_runs(instruction: str, text: str, rpr=None) -> list:
    """The runs of a complex field: begin, instruction, separate, cached result, end."""

    def run(child):
        r = element("w:r")
        if rpr is not None:
            r.append(deepcopy(rpr))
        r.append(child)
        return r

    instr = element("w:instrText")
    instr.text = f" {instruction} "
    instr.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    result = text_run(text)
    if rpr is not None:
        result.insert(0, deepcopy(rpr))
    return [
        run(element("w:fldChar", fldCharType="begin")),
        run(instr),
        run(element("w:fldChar", fldCharType="separate")),
        result,
        run(element("w:fldChar", fldCharType="end")),
    ]


def cross_references(doc, refs: References) -> None:
    """Replace links to headings, figures, tables and equations with Word REF fields.

    Word shows them as ordinary cross-references: Ctrl+click jumps to the target and F9
    refreshes the number after the document changes.
    """
    for link in list(doc.element.body.iter(qn("w:hyperlink"))):
        anchor = link.get(qn("w:anchor")) or ""
        if anchor not in refs.targets:
            continue
        name, text, switches = refs.targets[anchor]
        first = link.find(qn("w:r"))
        rpr = deepcopy(first.find(qn("w:rPr"))) if first is not None else None
        if rpr is not None:
            for style in rpr.findall(qn("w:rStyle")):
                rpr.remove(style)
        runs = field_runs(f"REF {name} {switches}", text, rpr)
        if anchor.startswith("eq-"):
            prefix = text_run("式")
            if rpr is not None:
                prefix.insert(0, deepcopy(rpr))
            runs.insert(0, prefix)
        for r in runs:
            link.addprevious(r)
        link.getparent().remove(link)


def number_equations(doc, refs: References) -> None:
    """Centre each numbered display equation and put its number, (1), at the right margin.

    Quarto writes the number inside the equation itself; it is moved into a text run after a
    right tab stop, the usual layout of equations in theses.
    """
    width = 9072  # text width in twips: A4 (21 cm) minus two 2.5 cm margins
    for p in doc.element.body.iter(qn("w:p")):
        mark = p.find(qn("w:bookmarkStart"))
        para = p.find(qn("m:oMathPara"))
        if mark is None or para is None or not mark.get(qn("w:name"), "").startswith("eq-"):
            continue
        math = para.find(qn("m:oMath"))
        runs = math.findall(qn("m:r"))
        texts = ["".join(t.text or "" for t in r.iter(qn("m:t"))) for r in runs]
        if ")" not in texts or "(" not in texts:
            continue
        start = len(texts) - 1 - texts[::-1].index("(")
        number = "".join(texts[start + 1 : -1])
        for run in runs[start:]:
            math.remove(run)
        if start > 0 and not texts[start - 1].strip():
            math.remove(runs[start - 1])
        position = list(p).index(para)
        p.remove(para)
        label = text_run(f"({number})")
        for offset, child in enumerate((tab_run(), math, tab_run(), label)):
            p.insert(position + offset, child)
        refs.mark(mark.get(qn("w:name")), label, label, f"({number})")
        ppr = paragraph_props(p)
        tabs = element("w:tabs")
        tabs.append(element("w:tab", val="center", pos=str(width // 2)))
        tabs.append(element("w:tab", val="right", pos=str(width)))
        put(ppr, tabs)
        put(ppr, element("w:spacing", before="120", after="120", line="360", lineRule="auto"))
        put(ppr, element("w:ind", left="0", firstLine="0", firstLineChars="0"))
        put(ppr, element("w:jc", val="left"))


def tidy_references(doc) -> None:
    """Drop the spaces pandoc leaves around 圖、表、式 and section references in CJK text."""
    for link in doc.element.body.iter(qn("w:hyperlink")):
        anchor = link.get(qn("w:anchor")) or ""
        if not anchor.startswith(("fig-", "tbl-", "eq-", "sec-")):
            continue
        for side in ("previous", "next"):
            run = link.getprevious() if side == "previous" else link.getnext()
            if run is None or local(run) != "r":
                continue
            t = run.find(qn("w:t"))
            if t is None or not t.text:
                continue
            neighbour = run.getprevious() if side == "previous" else run.getnext()
            beyond = paragraph_text(neighbour) if neighbour is not None else ""
            if side == "previous":
                text, other = (
                    t.text,
                    (beyond[-1:] if t.text.strip() == "" else t.text.rstrip()[-1:]),
                )
            else:
                text, other = t.text, (beyond[:1] if t.text.strip() == "" else t.text.lstrip()[:1])
            label = paragraph_text(link)
            edge = label[:1] if side == "previous" else label[-1:]
            if anchor.startswith("eq-"):
                edge = "式" if side == "previous" else ")"  # shown as 式(n)
            # Drop the space between two CJK characters (如 圖 → 如圖) or next to full-width
            # punctuation (（ 4.5 → （4.5); keep it between a number and a CJK word (圖 3-1 所示).
            if (
                other
                and not other.isascii()
                and (not edge.isascii() or edge == ")" or not other.isalnum())
            ):
                t.text = text.rstrip(" ") if side == "previous" else text.lstrip(" ")


def number_headings_and_captions(doc, refs: References) -> None:
    """Switch numbering off for unnumbered headings; number captions and references per chapter.

    Captions in 附錄 A are numbered 表 A-1, 圖 A-1, and so on.
    """
    body = doc.element.body
    chapters, chapter, unnumbered = 0, "", False
    counters: dict[tuple[str, str], int] = {}
    labels: dict[tuple[str, str], str] = {}
    sections = [0, 0]  # level-2 and level-3 counters within the chapter
    float_id = ""  # Quarto puts the fig-/tbl- bookmark just before the float and its caption
    section_id = ""  # and a heading's bookmark just before the heading
    for p in body.iter(qn("w:p"), qn("w:bookmarkStart")):
        if local(p) == "bookmarkStart":
            name = p.get(qn("w:name")) or ""
            if name.startswith(("fig-", "tbl-")):
                float_id = name
            elif name.startswith("sec-"):
                section_id = name
            continue
        style = paragraph_style(p) or ""
        if re.fullmatch(r"Heading\d", style):
            text = paragraph_text(p)
            level = int(style[-1])
            if style == "Heading1":
                unnumbered = text in UNNUMBERED_CHAPTERS
                chapters += 0 if unnumbered else 1
                chapter = "" if unnumbered else str(chapters)
                sections = [0, 0]
            elif not unnumbered and level in (2, 3):
                if level == 2:
                    sections = [sections[0] + 1, 0]
                else:
                    sections[1] += 1
            mark = p.find(qn("w:bookmarkStart"))
            anchor = mark.get(qn("w:name")) if mark is not None else section_id
            section_id = ""
            runs = p.findall(qn("w:r"))
            if not unnumbered and anchor and runs:
                number = (
                    f"第{CHINESE_NUMERALS[chapters - 1]}章"
                    if level == 1
                    else ".".join(str(n) for n in [chapters, *sections][:level])
                )
                refs.mark(anchor, runs[0], runs[-1], number, "\\w \\h")
            elif unnumbered and (appendix := re.match(r"附錄\s*([A-Z])", text)):
                chapter = appendix.group(1)
            if unnumbered:
                num_pr = element("w:numPr")
                num_pr.append(element("w:ilvl", val="0"))
                num_pr.append(element("w:numId", val="0"))
                put(paragraph_props(p), num_pr)
            continue
        match = CAPTION.match(paragraph_text(p)) if style == "ImageCaption" else None
        if not match:
            continue
        kind, number, title = match.groups()
        counters[kind, chapter] = counters.get((kind, chapter), 0) + 1
        label = f"{kind} {chapter}-{counters[kind, chapter]}" if chapter else f"{kind} {number}"
        labels[kind, number] = label
        ppr = paragraph_props(p)
        for child in list(ppr):
            ppr.remove(child)
        put(ppr, element("w:pStyle", val="TableCaption" if kind == "表" else "ImageCaption"))
        set_text(p, label)
        label_run = p.find(qn("w:r"))
        title_run = deepcopy(label_run)
        title_run.find(qn("w:t")).text = f"　{title}"
        label_run.addnext(title_run)
        if float_id:
            refs.mark(float_id, label_run, label_run, label)
            float_id = ""

    for link in body.iter(qn("w:hyperlink")):
        anchor = link.get(qn("w:anchor")) or ""
        match = REFERENCE.match("".join(t.text or "" for t in link.iter(qn("w:t"))).strip())
        if anchor.startswith(("tbl-", "fig-")) and match and match.groups() in labels:
            set_text(link, labels[match.groups()])


GROUP = "｜"  # a header cell written as 「組名｜子欄名」 sits under a spanning group title


def cell_text(cell) -> str:
    return "".join(t.text or "" for t in cell.iter(qn("w:t"))).strip()


def header_rows(rows) -> int:
    return sum(1 for r in rows if r.find(f"{qn('w:trPr')}/{qn('w:tblHeader')}") is not None)


def grid_columns(row) -> list[tuple[int, object]]:
    """(first grid column, cell) for each cell of a row, following gridSpan."""
    result, column = [], 0
    for cell in row.findall(qn("w:tc")):
        result.append((column, cell))
        span = cell.find(f"{qn('w:tcPr')}/{qn('w:gridSpan')}")
        column += int(span.get(qn("w:val"))) if span is not None else 1
    return result


def split_group_headers(doc) -> set:
    """Turn header cells 「組名｜子欄名」 into a two-row header with spanning group titles.

    The first row holds the group titles, merged across their columns, and the other
    column titles, merged down both rows; the second row holds the sub-titles. This is
    the grouped header of APA-style tables. Returns the group-title cells.
    """
    titles = set()
    for tbl in doc.element.body.iter(qn("w:tbl")):
        rows = tbl.findall(qn("w:tr"))
        if not rows or header_rows(rows) != 1:
            continue
        top = rows[0]
        cells = top.findall(qn("w:tc"))
        labels = [cell_text(c) for c in cells]
        if not any(GROUP in label for label in labels):
            continue
        sub = deepcopy(top)
        top.addnext(sub)
        for cell, label in zip(sub.findall(qn("w:tc")), labels, strict=True):
            if GROUP in label:
                set_text(cell.find(qn("w:p")), label.split(GROUP, 1)[1])
            else:
                put(get_or_add(cell, "w:tcPr", position=0), element("w:vMerge"))
                for p in cell.findall(qn("w:p")):
                    for run in p.findall(qn("w:r")):
                        p.remove(run)
        groups = [label.split(GROUP, 1)[0] if GROUP in label else None for label in labels]
        j = 0
        while j < len(cells):
            cell, group = cells[j], groups[j]
            tcpr = get_or_add(cell, "w:tcPr", position=0)
            if group is None:
                put(tcpr, element("w:vMerge", val="restart"))
                j += 1
                continue
            span = 1
            while j + span < len(cells) and groups[j + span] == group:
                top.remove(cells[j + span])
                span += 1
            if span > 1:
                put(tcpr, element("w:gridSpan", val=str(span)))
                for width in tcpr.findall(qn("w:tcW")):
                    tcpr.remove(width)
            set_text(cell.find(qn("w:p")), group)
            titles.add(cell)
            j += span
    return titles


NUMERIC = re.compile(
    r"[\d.,%+\-−–~×≥≤<>()（）、；/*⁻⁰¹²³⁴⁵⁶⁷⁸⁹\s元秒倍件筆萬億約個百分點]*\d[\d.,%+\-−–~×≥≤<>()（）、；/*⁻⁰¹²³⁴⁵⁶⁷⁸⁹\s元秒倍件筆萬億約個百分點]*"
)
SYMBOLS = re.compile(r"[✓✗—\s]*")


def column_alignment(rows, skip: int = 1) -> list[str]:
    """Numbers right, text left, check marks centred, decided per column from the body rows."""
    columns: dict[int, list[str]] = {}
    for row in rows[skip:]:
        for j, cell in enumerate(row.findall(qn("w:tc"))):
            text = cell_text(cell)
            if text and text not in {"—", "－"}:
                columns.setdefault(j, []).append(text)
    width = max((len(r.findall(qn("w:tc"))) for r in rows[skip:]), default=0)
    result = []
    for j in range(width):
        texts = columns.get(j, [])
        if texts and all(SYMBOLS.fullmatch(t) for t in texts):
            result.append("center")
        elif texts and sum(bool(NUMERIC.fullmatch(t)) for t in texts) >= 0.6 * len(texts):
            result.append("right")
        else:
            result.append("left")
    return result


def thin_rule(side: str, space: str = "0"):
    return element(f"w:{side}", val="single", sz="6", space=space, color="000000")


def format_tables(doc, group_titles=frozenset()) -> None:
    """Three-line tables (三線表) for data tables; no borders on the layout tables around floats."""
    for tbl in doc.element.body.iter(qn("w:tbl")):
        rows = tbl.findall(qn("w:tr"))
        n_header = header_rows(rows)
        is_data = n_header > 0
        borders = element("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            rule = is_data and side in ("top", "bottom")
            borders.append(
                element(
                    f"w:{side}", val="single" if rule else "nil", sz="12", space="0", color="000000"
                )
            )
        put(tbl.find(qn("w:tblPr")), borders)
        following = tbl.getnext()  # leave a gap between a float and the text after it
        at_top = tbl.getparent() is doc.element.body
        if at_top and following is not None and local(following) == "p":
            if paragraph_style(following) == "TableNote":  # a note sits close under its table
                put(paragraph_props(following), element("w:spacing", before="80"))
                following = following.getnext()
            if following is not None and local(following) == "p":
                put(paragraph_props(following), element("w:spacing", before="240"))

        for p in tbl.iter(qn("w:p")):  # keep a figure on the same page as its caption
            if p.find(f".//{qn('w:drawing')}") is not None:
                put(paragraph_props(p), element("w:keepNext"))
        if not is_data:
            continue
        margins = element("w:tblCellMar")  # narrower cell padding than Word's 0.19 cm
        for side in ("left", "right"):
            margins.append(element(f"w:{side}", w="57", type="dxa"))
        put(tbl.find(qn("w:tblPr")), margins)
        align = column_alignment(rows, skip=n_header)
        for i, row in enumerate(rows):
            header = i < n_header
            for j, cell in grid_columns(row):
                title = cell in group_titles
                tcpr = get_or_add(cell, "w:tcPr", position=0)
                if header:  # the table style's first-row border would double the group rule
                    tc_borders = element("w:tcBorders")
                    if i == n_header - 1:
                        tc_borders.append(thin_rule("bottom"))
                    else:
                        tc_borders.append(element("w:bottom", val="nil"))
                    put(tcpr, tc_borders)
                if header and n_header > 1:
                    put(tcpr, element("w:vAlign", val="bottom"))
                for p in cell.findall(qn("w:p")):
                    ppr = paragraph_props(p)
                    if title:  # a short rule under the group title, inset by the cell padding
                        rule = element("w:pBdr")
                        rule.append(thin_rule("bottom", space="1"))
                        put(ppr, rule)
                    put(
                        ppr,
                        element("w:spacing", before="40", after="40", line="276", lineRule="auto"),
                    )
                    put(ppr, element("w:ind", left="0", firstLine="0", firstLineChars="0"))
                    jc = "center" if title else (align[j] if j < len(align) else "left")
                    put(ppr, element("w:jc", val=jc))
                    for run in p.iter(qn("w:r")):
                        rpr = run_props(run)
                        if header:
                            put(rpr, element("w:b"))
                        put(rpr, element("w:sz", val="22"))
                        put(rpr, element("w:szCs", val="22"))


def page_setup(doc) -> None:
    for section in doc.sections:
        section.page_width, section.page_height = Cm(21), Cm(29.7)
        section.top_margin = section.bottom_margin = Cm(2.5)
        section.left_margin = section.right_margin = Cm(2.5)
        footer = section.footer
        footer.is_linked_to_previous = False
        p = footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p._p.append(element("w:fldSimple", instr="PAGE"))
    # Ask Word to refresh the 目錄 / 圖目錄 / 表目錄 fields when the file is opened.
    put(doc.settings.element, element("w:updateFields", val="true"))


def postprocess(path: Path) -> None:
    doc = Document(str(path))
    refs = References()
    apply_styles(doc)
    color_repo_link(doc)
    add_heading_numbering(doc)
    number_headings_and_captions(doc, refs)
    format_tables(doc, split_group_headers(doc))
    number_equations(doc, refs)
    tidy_references(doc)
    cross_references(doc, refs)
    page_setup(doc)
    doc.save(str(path))
    print(f"postprocess_docx: formatted {path}")


def main(argv: list[str]) -> None:
    files = argv or os.environ.get("QUARTO_PROJECT_OUTPUT_FILES", "").splitlines()
    for name in files:
        if name.endswith(".docx"):
            postprocess(Path(name))


if __name__ == "__main__":
    main(sys.argv[1:])
