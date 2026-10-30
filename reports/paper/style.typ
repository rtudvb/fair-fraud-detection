// Report typography, included at the start of the body (see _quarto.yml).

// Body language: Chinese (Taiwan). The document-level `lang` stays en-US so citeproc writes
// English citations ("et al.").
#set text(lang: "zh", region: "TW")

// Chapters: 第一章, 第二章, ...; sections: 1.1, 1.1.1; each chapter starts on a new page.
#set heading(numbering: (..nums) => {
  let n = nums.pos()
  if n.len() == 1 { "第" + numbering("一", n.at(0)) + "章" } else { numbering("1.1", ..n) }
})
#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  // Figures and tables are numbered per chapter (圖 1-1, 表 2-3).
  counter(figure.where(kind: "quarto-float-fig")).update(0)
  counter(figure.where(kind: "quarto-float-tbl")).update(0)
  set align(center)
  set text(size: 16pt)
  block(above: 1em, below: 1.2em, it)
}
#let per-chapter(n) = context {
  let chapter = counter(heading).get().first()
  if chapter > 0 [#chapter\-#n] else [#n]
}
#show figure.where(kind: "quarto-float-fig"): set figure(numbering: per-chapter)
#show figure.where(kind: "quarto-float-tbl"): set figure(numbering: per-chapter)

// Body text: first-line indent of two characters, comfortable line spacing.
#set par(first-line-indent: (amount: 2em, all: true), leading: 0.9em, justify: true)

// Three-line tables (三線表): thick top and bottom rules, thin rule under the header,
// no vertical lines. Header cells are bold.
#set table(stroke: none, inset: (x: 6pt, y: 5pt))
#set table.hline(stroke: 0.6pt)
#show table: it => box(stroke: (top: 1.2pt, bottom: 1.2pt), it)
#show table.cell.where(y: 0): set text(weight: "bold")
#show table.cell: set par(first-line-indent: 0em, justify: false)

// Captions: 表 above tables, 圖 below figures, slightly smaller text.
#show figure.caption: set text(size: 10pt)

// Lists of figures / tables: compute the chapter number at the figure's own location
// (the numbering function above would otherwise read the chapter at the outline).
#show outline.entry: it => {
  let el = it.element
  if el.func() != figure { return it }
  let loc = el.location()
  let chapter = counter(heading).at(loc).first()
  let n = counter(figure.where(kind: el.kind)).at(loc).first()
  let label = if chapter > 0 [#el.supplement #chapter\-#n] else [#el.supplement #n]
  link(loc, it.indented([#label], it.inner()))
}
