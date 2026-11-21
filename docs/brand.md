# Brand

transiograms uses the cratongen family look, ported from [segyrs](https://cratongen.github.io/segyrs/):
black, white and the greys between them, two typefaces, heavy horizontal rules, square flat
blocks, numbered sections and underlined links. The name is set as a plain typographic wordmark.
There is no pictorial mark.

## Wordmark and favicon

![transiograms wordmark](brand/wordmark-black.svg#only-light){ width="372" }
![transiograms wordmark](brand/wordmark-white.svg#only-dark){ width="372" }

| File | Use |
| --- | --- |
| [`brand/wordmark-black.svg`](brand/wordmark-black.svg) | The name in black ink, for a light background. The docs home uses it. |
| [`brand/wordmark-white.svg`](brand/wordmark-white.svg) | The same in white ink, for a dark background. |
| [`brand/mark-black.svg`](brand/mark-black.svg) | The letters `tg` alone, black ink. The docs header uses it, beside the printed name. |
| [`brand/mark-white.svg`](brand/mark-white.svg) | The same in white ink; the header swaps to it in dark mode. |
| [`brand/favicon.svg`](brand/favicon.svg) | White `tg` on a black square, with thicker strokes so it reads at 16 px. |

Every file is hand-written SVG with no gradient, no filter and no embedded font. The letters are
monoline paths with round bowls on a 20-unit x-height and a 3.5-unit stroke (4.5 in the favicon),
so the name renders the same on GitHub, in a browser and in a slide deck.

The name is always lowercase: `transiograms`. Set it in code font when you mean the package
(`import transiograms as tg`) and in the text face when you mean the project.

Keep clear space around the wordmark equal to half its x-height on every side. Draw it no smaller
than 120 px wide; below that, use the favicon.

## Palette

The interface uses black, white and seven greys. No hue appears in the interface: not in links,
admonitions, code highlighting, tables, search, tabs or the footer.

| Token | Hex | Role |
| --- | --- | --- |
| `--tg-black` | `#000000` | Ink in light mode, page in dark mode |
| `--tg-grey-900` | `#171717` | Code and note blocks in dark mode |
| `--tg-grey-800` | `#262626` | Hairlines in dark mode |
| `--tg-grey-600` | `#525252` | Muted text and code comments in light mode |
| `--tg-grey-500` | `#737373` | Link underlines only: it is below 4.5:1 on black, so never text |
| `--tg-grey-400` | `#a3a3a3` | Muted text and code comments in dark mode |
| `--tg-grey-200` | `#e5e5e5` | Hairlines in light mode |
| `--tg-grey-100` | `#f5f5f5` | Code and note blocks in light mode, ink in dark mode |
| `--tg-white` | `#ffffff` | Page in light mode, and the mat behind figures in dark mode |

The tokens, the fonts and the mapping onto Material for MkDocs live in one self-contained
stylesheet, `docs/stylesheets/transiograms.css`. It sets every colour variable of Material for
both schemes, so no Material hue reaches a page. Text and muted text meet 4.5:1 contrast on the
page and on code and note blocks in both modes.

## Type

| Role | Face | Fallback |
| --- | --- | --- |
| Text and headings | [Geist](https://fonts.google.com/specimen/Geist), weights 400 to 800 | `system-ui`, Helvetica, Arial |
| Code, figures, section numbers | [Geist Mono](https://fonts.google.com/specimen/Geist+Mono), weights 400 to 700 | `ui-monospace`, Menlo, Consolas |

Both are variable fonts served with the site from `docs/brand/fonts/` (Geist 1.3.1 by Vercel, SIL
Open Font License 1.1, text in `docs/brand/fonts/OFL.txt`), so the theme makes no request to
another server. Headings are heavy (800), tight and upright. Tables of numbers use tabular figures.

## Rules and blocks

A 6 px rule sits under the page title, a 3 px rule over each numbered section, and 1 px hairlines
between table rows. Blocks are square, filled and flat, with no shadow and no radius. Links are
underlined instead of coloured.

## Figures

Figures are the one place hue appears, and only where it encodes data:

- Categories (facies, lithology) keep their colours from `tg.plot`, so the same category has the
  same colour in every figure.
- Continuous values (probabilities, transition matrices) use the `Greys` ramp.
- Lines that are not categories use black and grey first; a hue comes in only when a figure needs
  more than two series.
- Text, ticks and labels are black on white paper. In dark mode the stylesheet puts each figure on
  a white mat, so the axes keep their contrast.

`examples/common.py` holds this style and `mise run examples` applies it. Change a figure by
changing its script or the style and rerunning the task, never by editing the image.

## The check

`python/tests/test_docs_theme.py` checks the theme: grey tokens only, no colour literal outside
the palette block, every Material colour variable set, admonition rules that outrank Material's
coloured ones, the fonts and their licence present, and 4.5:1 for text. `mise run docs-build` runs
it against the built site.

## Do not

- Recolour the wordmark, add a gradient, or put it on a photograph.
- Redraw the wordmark in another typeface, or set "transiograms" with a capital letter.
- Add a hue to the interface to signal state. Use weight, a rule, or a word.
- Stretch the wordmark, or add an outline or a drop shadow.
