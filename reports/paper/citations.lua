-- Colour citations so they stand out while drafting (author's request: red).
--
-- Marks every citation: pandoc citations ([@key], @key) and hand-written Chinese ones
-- written as [（今周刊，2026）]{.cite}. Cross-references (@fig-, @tbl-, ...) stay as they are.
--   Word: character style "Citation"; postprocess_docx.py gives it the colour.
--   PDF (Typst): text fill.
-- To turn the colour off, remove this filter from _quarto.yml.

local COLOR = "#c00000"
local CROSSREF_PREFIXES = { "fig-", "tbl-", "sec-", "eq-" }

local function is_crossref(cite)
  for _, citation in ipairs(cite.citations) do
    for _, prefix in ipairs(CROSSREF_PREFIXES) do
      if citation.id:sub(1, #prefix) == prefix then
        return true
      end
    end
  end
  return false
end

local function mark(inlines)
  if FORMAT:match("docx") then
    return pandoc.Span(inlines, { ["custom-style"] = "Citation" })
  elseif FORMAT:match("typst") then
    local out = pandoc.Inlines({ pandoc.RawInline("typst", '#text(fill: rgb("' .. COLOR .. '"))[') })
    out:extend(inlines)
    out:insert(pandoc.RawInline("typst", "]"))
    return out
  end
  return inlines
end

function Cite(el)
  if is_crossref(el) then
    return nil
  end
  return mark({ el })
end

function Span(el)
  if el.classes:includes("cite") then
    return mark(el.content)
  end
end
