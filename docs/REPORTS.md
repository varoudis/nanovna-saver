# REPORT-01 — Native PDF export

The report window saves the same frozen model three ways: HTML, CSV and now
PDF. All three are drawn from one `ReportModel`, so they cannot disagree
about a value or a verdict — the PDF is not a second report implementation.

## Route

`QTextDocument` + `QPdfWriter`, both already part of the declared Qt
dependency. No web engine, no new package, nothing to vendor:

| Route | Verdict |
| --- | --- |
| `QTextDocument` + `QPdfWriter` | **chosen** — in Qt already, real text (selectable, extractable), A4 pagination |
| `QWebEnginePage.printToPdf` | rejected — adds a large optional dependency for one feature |
| reportlab / WeasyPrint | rejected — new dependency, and the layout would be a second implementation to keep in sync |

The document is built by calling `render_html(model)` and transforming it
for print, so the layout is shared with the HTML export by construction.

## What the print transform changes, and why

1. **Plots become images.** Each SVG in the Plots section is wrapped in an
   `<img>` at 2× and given its own page (`page-break-before: always`). Qt
   duplicates captions and splits images when a block straddles a page
   boundary, so one plot per page is the only arrangement that came out
   clean over an 0–79 page sweep.
2. **Wide tables become labelled paragraphs.** Above four columns Qt lays a
   table out by shrinking columns and then begins breaking *inside* cell
   text — `Antenn`/`a match`, `UNAVAI`/`LABLE` — which splits a value across
   two text runs. That destroys the round trip this export exists to
   guarantee, and CSS cannot prevent it: `white-space: nowrap`,
   `table-layout: fixed`, `width: auto`, `colgroup` and per-column widths
   were each measured, and none changed the extraction. So a table wider
   than four columns (the limit table, eight; the trace table, seven) is
   printed as one line per row:

   ```
   Limit: Antenna match Metric: S11 VSWR (ratio) Band: 10 - 11 MHz
   Result: FAIL Worst value: 3 Frequency (Hz): 11000000
   Bound at worst value: maximum 2
   ```

   Narrow tables (the channel table, four columns) keep their real table
   with a repeating `<thead>`. The **HTML export is unchanged** — it keeps
   all its tables; this is a print-only decision.
3. **A print stylesheet**: 9 pt body, point-sized headings, zero body
   margin so the printable width is not spent on a browser default, and an
   Arial-first font stack. Helvetica and Helvetica Neue substitute
   ligatures (`inﬁnite`, `oﬀset`) which a text reader then cannot match
   against the HTML; `QFont.setFeature` cannot turn this off in PySide6.

## Verified

| Check | Result |
| --- | --- |
| `pytest tests/test_pdf_report.py` | **17 passed** |
| Extractable text | `Antenna match`, `S11 VSWR (ratio)`, `10 - 11 MHz`, `no S21 channel in this trace` all come back whole from `pdftotext` |
| Page geometry | A4, 595 × 842 pt, nothing drawn past `xMax = 551.5` (a 30 pt right margin survives) |
| Pagination | multi-page reports numbered `Page 1 of M` … `Page M of M` |
| Unicode | `Ω`, `µ`, `—` and CJK round-trip |
| Long identifiers | a 122-character unbroken path wraps inside the margin rather than running off the page |
| Round trip | verdict words and values match the HTML and the CSV for the same model |
| From the GUI | `ReportWindow.save_pdf_at()` on a real `NanoVNASaver()` window writes a complete `%PDF-` … `%%EOF` document |

## Limitations

* Plots are JPEG (`/DCTDecode`) at 2× raster (~148 ppi), not vector.
* Trace-colour swatches (`.swatch`) do not paint; the trace name is still
  printed, so no information is lost, but the colour key is missing.
* No control over OpenType ligature substitution beyond the font choice.
* `pdftotext` can interleave the footer between two lines of a token that
  straddles a page break, so an identifier must fit on one page (≤ ~160
  characters at 9 pt is safe).
* Rendered and verified on macOS arm64 with offscreen Qt only. Windows and
  Linux were not rendered; the route is the same Qt API, but that is an
  expectation, not evidence.
