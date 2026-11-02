#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020 Rune B. Broberg
#  Copyright (C) 2020ff NanoVNA-Saver Authors
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Native PDF export of a frozen report (REPORT-01).

The PDF is produced from the same frozen :class:`~NanoVNASaver.Reports
.ReportModel` as the HTML and CSV exports.  The document body is the HTML
export itself, printed through ``QTextDocument`` into a ``QPdfWriter``, so
every value, verdict, provenance label and missing-channel explanation is
produced by one code path: a PDF can never disagree with the HTML it
accompanies about what was measured.  Only four things are added for print:

* a print stylesheet (point sizes, tighter table padding, A4 margins),
* plots converted from inline SVG into images the PDF writer can embed, each
  starting on its own page so a figure is never split across a page break,
* header rows promoted to ``<thead>`` so a table that spans pages repeats
  its column headings, plus a ``Page N of M`` footer,
* an empty trace-colour swatch filled so that it paints (see
  :func:`_paint_swatches`).

Pagination is explicit: the document is laid out against a body height that
excludes a footer strip, and each page is painted with a clipped context.
That keeps the footer out of the text flow and makes the page count known
before the first page is drawn.

Route comparison
================

Two routes were compared before implementing this module.

**(a) Existing Qt stack** (``QTextDocument`` + ``QPdfWriter``, chosen).
Cost: nothing new to declare or ship.  PySide6 is already a hard dependency
of the application, and the SVG rasteriser used for plots (``QtSvg``, reached
through ``QImageReader``/``QTextDocument``) is part of ``PySide6-Essentials``,
which the declared ``pyside6`` requirement already pulls in; no import of
``PySide6.QtSvg`` is needed, so an installation without it still exports
text.  Rendering fidelity, measured on the produced files:

* text is real embedded text (subset CID TrueType), searchable and
  extractable by ``pdftotext``; Unicode (``Ω``, ``µ``, ``—``, CJK) survives
  the round trip and ``pdffonts`` reports the fonts as embedded with Unicode
  mapping;
* page geometry is A4 with 15 mm margins and a repeatable body/footer split;
* plots are **rasterised**: ``QTextDocument`` can embed a ``QImage`` only as
  a JPEG (``/DCTDecode``) at the image's pixel size, so the SVG is rendered
  at twice its natural pixel size (about 148 ppi at the printed width) and
  JPEG-compressed.  Plot lines are therefore not vector paths and very fine
  detail in a plot is softened; the surrounding text stays vector.
* no control over OpenType features: Qt's PDF path has no way to disable
  ``liga``/``clig`` for a font that ligates (``QFont.setFeature`` is
  unusable from PySide6 6.8 because ``QFont.Tag`` cannot be built from a
  string).  With Arial-class fonts the extracted text is unaffected; with a
  ligating face (Helvetica, Lucida Grande) ``pdftotext`` returns the ``ﬁ``
  glyph instead of ``fi``.  The font stack below prefers non-ligating
  families for exactly that reason.

**(b) An extra PDF renderer** (reportlab, WeasyPrint, …).  Cost: a new
runtime and build dependency, which matters because this project ships
self-contained bundles: reportlab adds a pure-Python wheel plus compiled
extensions for image handling; WeasyPrint needs cairo/pango/gobject (and a
font configuration) at runtime, none of which the PySide6 wheel provides.
It would also render the report a second time in a different layout engine,
so the PDF and the HTML could diverge in table breaking, wrapping and
number formatting, and browser-print-to-PDF (the current documented
workaround) would stay a third, differently-rendering path.  Fidelity gain:
vector plot paths and lossless image compression, i.e. crisper plot lines
at high zoom, at the cost of a second layout implementation.

Chosen: (a), because the correctness requirement is that the PDF carries the
same frozen values and verdicts as HTML/CSV, and reusing the HTML keeps that
true structurally rather than by duplicated formatting code.  The known
consequences (JPEG plots, no font-feature control) are stated above and in
``docs/REPORTS.md``.

Platforms actually verified: macOS 25.6.0 (arm64) with the ``offscreen`` Qt
platform plugin, PySide6 6.8.2.1, viewing/extracting the produced files with
poppler 26.09.0 (``pdftotext``, ``pdfimages``, ``pdfinfo``).  Windows and
Linux were **not** rendered as part of this work; nothing here claims they
were.
"""

from __future__ import annotations

import base64
import re
import sys
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui

from .Reports import render_html, report_plots

if TYPE_CHECKING:
    from .Reports import ReportModel

#: Page margins, applied to every side of the A4 sheet.
PAGE_MARGIN_MM = 15.0

#: Print resolution.  Text is vector regardless; this sets the coordinate
#: precision of the page and how many device pixels a rasterised plot has.
RESOLUTION_DPI = 300

#: Height of the reserved footer strip, in PostScript points.
FOOTER_HEIGHT_PT = 18.0

#: Printed width of a plot, in points (the body is 509 pt wide at A4 with
#: 15 mm margins).
PLOT_WIDTH_PT = 505

#: Rasterisation factor: the SVG's intrinsic pixel size is multiplied by
#: this while its ``viewBox`` is left untouched, so Qt draws the plot at the
#: printed size with this much more detail (520 px -> 1040 px, about
#: 148 ppi).  Larger factors mostly add JPEG noise and file size.
PLOT_SCALE = 2

#: Font family fallbacks.  Arial and its metric-compatible replacements come
#: first because they do not substitute the ``fi``/``fl`` ligatures, which
#: would otherwise change the text a reader extracts from the PDF.
FONT_FAMILIES = ("Arial", "Liberation Sans", "DejaVu Sans")

#: Print stylesheet, injected after the HTML export's own stylesheet so it
#: wins the cascade (Qt applies the later block, as CSS source order says).
#:
#: ``body {margin: 0}`` is load-bearing.  The screen stylesheet's 24 px
#: margin is meant for a browser and shrinks the printable body by 48 px at
#: 96 dpi, which starves the eight-column limit table: Qt then wraps inside
#: words ("Freque/ncy", "UNAVAI/LABLE") and the text a reader extracts no
#: longer matches the HTML export.
#:
#: The table font and padding are sized so those eight columns fit the
#: widest content this module produces (7 pt is 30 % smaller than the 9 pt
#: body, and 1 pt of cell padding keeps the columns touching without
#: clipping).  ``-qt-table-type: root`` is deliberately *not* used: it lays
#: the table out across the body but merges adjacent cells in the extracted
#: text ("ChannelPointsStart (Hz)").
_FONT_STACK = ", ".join(f'"{name}"' for name in FONT_FAMILIES)
PRINT_CSS = f"""
body {{ margin: 0; font-family: {_FONT_STACK}, sans-serif;
       font-size: 9pt; color: #111111; line-height: 1.35; }}
h1 {{ font-size: 16pt; margin: 0 0 4pt 0; }}
h2 {{ font-size: 11.5pt; margin: 10pt 0 4pt 0; }}
h3 {{ font-size: 10pt; margin: 8pt 0 3pt 0; }}
table {{ font-size: 7pt; margin: 4pt 0 8pt 0; }}
th, td {{ padding: 1pt 2pt; }}
.meta, .limits-note, .note {{ font-size: 8pt; }}
.badge {{ font-size: 7.5pt; }}
.plotpage {{ margin: 0; page-break-before: always; page-break-inside: avoid; }}
.plotpage td {{ border: none; padding: 0; vertical-align: top; }}
.plot-title {{ font-size: 10pt; font-weight: 600; margin: 0 0 4pt 0; }}
"""

#: The plots section of the HTML export: heading plus one ``<div>`` per SVG.
_PLOTS_SECTION = re.compile(
    r'\n?<h2>Plots</h2>(?:\n<div class="plot">.*?</div>)+', re.S
)
#: A table as emitted by the HTML export (the plot tables carry a class, so
#: they are deliberately not matched here).
_PLAIN_TABLE = re.compile(r"<table>(.*?)</table>", re.S)
#: The leading header row of such a table, if any.
_HEADER_ROW = re.compile(r"<tr>(?:(?!</tr>).)*?<th\b.*?</tr>", re.S)
#: The outer ``width``/``height`` of an SVG, used to scale its rasterisation.
_SVG_SIZE = re.compile(r'width="(\d+)" height="(\d+)"')

#: The application that Qt requires before any text can be laid out.  Held
#: in a module global so it is never garbage collected while in use.
_APPLICATION: QtGui.QGuiApplication | None = None


def _application() -> QtGui.QGuiApplication:
    """Return the running Qt application, creating a GUI-only one if needed.

    ``QTextDocument`` needs an application object before it can resolve
    fonts; without one Qt aborts the process rather than raising.  A
    ``QApplication`` (used by the GUI and the test suite) is a
    ``QGuiApplication``, so it satisfies this call.  No widget is created.
    """
    global _APPLICATION  # noqa: PLW0603 - kept alive for the interpreter
    application = QtGui.QGuiApplication.instance()
    if application is None:
        arguments = [sys.argv[0]] if sys.argv else ["nanovna-saver"]
        application = QtGui.QGuiApplication(arguments)
        _APPLICATION = application
    return application


def _plot_image(svg: str) -> str:
    """Wrap an SVG in a ``<img>`` that Qt rasterises at ``PLOT_SCALE``."""
    scaled = _SVG_SIZE.sub(
        lambda match: (
            f'width="{int(match.group(1)) * PLOT_SCALE}" '
            f'height="{int(match.group(2)) * PLOT_SCALE}"'
        ),
        svg,
        count=1,
    )
    payload = base64.b64encode(scaled.encode("utf-8")).decode("ascii")
    return (
        f'<img src="data:image/svg+xml;base64,{payload}" '
        f'width="{PLOT_WIDTH_PT}">'
    )


def _plot_block(title: str, svg: str, *, first: bool) -> str:
    """One plot on its own page, captioned above the image.

    The figure is a single-cell table with ``page-break-before`` rather than
    an ordinary paragraph: Qt duplicates text and splits images when a block
    straddles a page boundary, which would print a caption twice or cut a
    plot in half.
    """
    caption = f'<div class="plot-title">{title}</div>'
    heading = "<h2>Plots</h2>" if first else ""
    return (
        '<table class="plotpage" width="100%" cellspacing="0" '
        f'cellpadding="0"><tr><td>{heading}{caption}'
        f"{_plot_image(svg)}</td></tr></table>"
    )


def _plots_html(model: ReportModel) -> str:
    """HTML for the plots section, one figure per page."""
    blocks = [
        _plot_block(title, svg, first=index == 0)
        for index, (title, svg) in enumerate(report_plots(model))
    ]
    return "".join("\n" + block for block in blocks)


def _repeat_table_headers(document: str) -> str:
    """Promote a table's first header row to ``<thead>``.

    Qt repeats a ``<thead>`` on every page a table spans; it does not repeat
    a header row that is just the first ``<tbody>`` row.
    """

    def fix(match: re.Match[str]) -> str:
        body = match.group(1)
        header = _HEADER_ROW.match(body)
        if header is None:
            return match.group(0)
        return (
            f"<table><thead>{header.group(0)}</thead>"
            f"<tbody>{body[header.end() :]}</tbody></table>"
        )

    return _PLAIN_TABLE.sub(fix, document)


#: Cell and row fragments, for turning a table into printable paragraphs.
_CELL = re.compile(r"<t([hd])\b[^>]*>(.*?)</t\1>", re.S)
_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S)
_TAG = re.compile(r"<[^>]+>")

#: Above this many columns a table cannot be printed and read back.
#:
#: Qt lays a table out by shrinking columns, and past this width it starts
#: breaking *inside* cell text ("Antenn\na match", "UNAVAI\nLABLE"), which
#: splits a value across two text runs and destroys the PDF/HTML round trip
#: this export exists to guarantee.  CSS cannot prevent it: ``white-space:
#: nowrap``, ``table-layout: fixed``, ``width: auto`` and per-column widths
#: were each measured and none changed the extraction.  Wide tables are
#: therefore printed as one labelled line per row, which is also easier to
#: read across a page break.  The HTML export keeps its tables.
_MAX_TABLE_COLUMNS = 4


def _cell_text(match: re.Match[str]) -> str:
    """The markup inside a ``<td>``/``<th>`` match."""
    return match.group(2)


def _table_to_paragraphs(match: re.Match[str]) -> str:
    """Rewrite one wide table as paragraphs, one per row."""
    original = f"<table>{match.group(1)}</table>"
    rows = _ROW.findall(match.group(1))
    if not rows:
        return original
    header = [
        _TAG.sub("", _cell_text(found)).strip()
        for found in _CELL.finditer(rows[0])
    ]
    if len(header) <= _MAX_TABLE_COLUMNS:
        return original
    blocks: list[str] = []
    for row in rows[1:]:
        matches = list(_CELL.finditer(row))
        parts: list[str] = []
        for index, cell_html in enumerate(
            _cell_text(match) for match in matches
        ):
            value = _TAG.sub("", cell_html).strip()
            if not value:
                continue
            label = header[index] if index < len(header) else ""
            parts.append(f"<b>{label}:</b> {value}" if label else value)
        if parts:
            blocks.append(f'<p class="row">{" ".join(parts)}</p>')
    if not blocks:
        return original
    return "".join(blocks)


def _printable_tables(document: str) -> str:
    """Replace tables too wide to print with one paragraph per row."""
    return _PLAIN_TABLE.sub(_table_to_paragraphs, document)


#: The trace-colour swatch of a heading: an empty ``<span>`` whose only
#: content is a background colour.
_SWATCH = re.compile(r'(<span class="swatch"[^>]*>)(</span>)')

#: Filler that gives a swatch a glyph box, and so a paintable background.
#:
#: ``QTextDocument`` paints a span's background from the box of its *text*,
#: and an empty span has no text at all: the HTML export's
#: ``<span class="swatch" ...></span>`` therefore draws nothing in the PDF
#: (measured: 0 px of the trace colour on a rasterised page, against 875 for
#: the same span with this filler).  Non-breaking spaces are used rather than
#: a letter or a table cell because they paint no glyph of their own and
#: ``pdftotext`` drops them again, so the trace name is still extracted
#: alone.
#:
#: Three of them plus a separating one: Qt sizes the box from the glyphs
#: inside the span (it ignores an inline span's ``width`` and
#: ``height``), so one space gives a 4.0 x 17 pt sliver and three give the
#: 12 x 17 pt block measured here.  The height is the heading's line box and
#: cannot be reduced from CSS, so three spaces is the closest to the square
#: the HTML export draws.  The trailing space replaces the
#: ``margin-right: 5px`` that Qt also ignores: it leaves a 3.7 pt gap before
#: the name, against 5 px (3.75 pt) in the HTML export.  Sizes measured at
#: the printed heading size.
_SWATCH_FILLER = "&nbsp;" * 3
_SWATCH_GAP = "&nbsp;"


def _paint_swatches(document: str) -> str:
    """Fill every trace-colour swatch so that Qt paints it."""
    return _SWATCH.sub(
        lambda match: (f"{match.group(1)}{_SWATCH_FILLER}</span>{_SWATCH_GAP}"),
        document,
    )


def print_html(model: ReportModel, *, include_plots: bool = True) -> str:
    """The HTML the PDF is drawn from (print stylesheet, plots as images).

    Exposed so the decisions above are inspectable and testable without
    parsing a binary; :func:`render_pdf` is the public entry point.
    """
    document = render_html(model, include_plots=include_plots)
    if include_plots:
        plots = _plots_html(model)
        if plots:
            # Play the plots in the same place as the HTML export, between
            # the traces and the closing interpretation note.
            document, replacements = _PLOTS_SECTION.subn(
                lambda _match: plots, document, count=1
            )
            if not replacements:  # pragma: no cover - defensive
                document = document.replace("</body>", f"{plots}</body>")
    document = _repeat_table_headers(document)
    document = _printable_tables(document)
    document = _paint_swatches(document)
    return document.replace("</head>", f"<style>{PRINT_CSS}</style></head>")


def _document(markup: str, width: float, height: float) -> QtGui.QTextDocument:
    """A laid-out text document for ``markup``."""
    document = QtGui.QTextDocument()
    document.setDocumentMargin(0.0)
    font = QtGui.QFont()
    font.setFamilies(list(FONT_FAMILIES))
    font.setPointSizeF(9.0)
    document.setDefaultFont(font)
    option = QtGui.QTextOption()
    # Long identifiers (a Touchstone path, a device serial) have no break
    # opportunity; wrap them rather than letting them run past the margin.
    option.setWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    document.setDefaultTextOption(option)
    document.setHtml(markup)
    document.setTextWidth(width)
    document.setPageSize(QtCore.QSizeF(width, height))
    return document


def _configure(writer: QtGui.QPdfWriter, model: ReportModel) -> None:
    writer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.PageSizeId.A4))
    writer.setResolution(RESOLUTION_DPI)
    writer.setPageMargins(
        QtCore.QMarginsF(
            PAGE_MARGIN_MM, PAGE_MARGIN_MM, PAGE_MARGIN_MM, PAGE_MARGIN_MM
        ),
        QtGui.QPageLayout.Unit.Millimeter,
    )
    writer.setTitle(model.title or "NanoVNA-Saver report")
    creator = "NanoVNA-Saver"
    if model.app_version:
        creator = f"{creator} {model.app_version}"
    writer.setCreator(creator)


def _draw_footer(
    painter: QtGui.QPainter,
    model: ReportModel,
    number: tuple[int, int],
    geometry: tuple[float, float],
) -> None:
    """Report title on the left, ``Page N of M`` on the right."""
    index, pages = number
    width, body = geometry
    font = QtGui.QFont()
    font.setFamilies(list(FONT_FAMILIES))
    font.setPointSizeF(7.5)
    painter.setFont(font)
    painter.setPen(QtGui.QColor("#555555"))
    rect = QtCore.QRectF(0.0, body + 2.0, width, FOOTER_HEIGHT_PT - 4.0)
    painter.drawText(
        rect,
        QtCore.Qt.AlignmentFlag.AlignRight
        | QtCore.Qt.AlignmentFlag.AlignVCenter,
        f"Page {index + 1} of {pages}",
    )
    if model.title:
        metrics = QtGui.QFontMetrics(font)
        title = metrics.elidedText(
            model.title,
            QtCore.Qt.TextElideMode.ElideRight,
            int(width * 0.6),
        )
        painter.drawText(
            rect,
            QtCore.Qt.AlignmentFlag.AlignLeft
            | QtCore.Qt.AlignmentFlag.AlignVCenter,
            title,
        )


def _paint(
    writer: QtGui.QPdfWriter, document: QtGui.QTextDocument, model: ReportModel
) -> int:
    """Draw every page and return the page count."""
    rect = writer.pageLayout().paintRectPoints()
    width = float(rect.width())
    body = float(rect.height()) - FOOTER_HEIGHT_PT
    document.setTextWidth(width)
    document.setPageSize(QtCore.QSizeF(width, body))
    pages = max(1, document.pageCount())
    painter = QtGui.QPainter(writer)
    try:
        # The painter works in device pixels; the document in points.
        scale = writer.resolution() / 72.0
        painter.scale(scale, scale)
        layout = document.documentLayout()
        for index in range(pages):
            context = QtGui.QAbstractTextDocumentLayout.PaintContext()
            context.clip = QtCore.QRectF(0.0, index * body, width, body - 0.5)
            painter.save()
            painter.translate(0.0, -index * body)
            layout.draw(painter, context)
            painter.restore()
            _draw_footer(painter, model, (index, pages), (width, body))
            if index < pages - 1:
                writer.newPage()
    finally:
        painter.end()
    return pages


def render_pdf(model: ReportModel, *, include_plots: bool = True) -> bytes:
    """Render the frozen report ``model`` to PDF bytes."""
    _application()
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    writer = QtGui.QPdfWriter(buffer)
    _configure(writer, model)
    rect = writer.pageLayout().paintRectPoints()
    document = _document(
        print_html(model, include_plots=include_plots),
        float(rect.width()),
        float(rect.height()) - FOOTER_HEIGHT_PT,
    )
    _paint(writer, document, model)
    # The writer writes the trailer when it is destroyed, so release the last
    # reference before reading the buffer.
    del writer
    data = bytes(buffer.data())
    buffer.close()
    return data


def write_pdf(
    model: ReportModel, filename: str, *, include_plots: bool = True
) -> None:
    """Write the frozen report ``model`` to ``filename`` as PDF."""
    with open(filename, "wb") as target:
        target.write(render_pdf(model, include_plots=include_plots))
