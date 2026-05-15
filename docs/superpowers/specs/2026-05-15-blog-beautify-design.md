# Blog Beautification Design

## Overview

Beautify the personal blog (hugo-coder theme) with a warm amber-teal color palette, rounded typography, enhanced post pages, and subtle animations. All changes are implemented via Hugo template overrides and custom CSS/JS — no theme switch required.

## Design Direction

**Modern Tech x Warm Personal** — Card-based layouts, rounded corners, warm earthy tones, clean code display.

## Color Palette — Amber Teal

| Role | Light Mode | Dark Mode |
|------|-----------|-----------|
| Page BG | `#fdfaf3` | `#1e1a16` |
| Card BG | `#fff` | `#252019` |
| Heading | `#5c3d2e` | `#f0c58a` |
| Body text | `#5c4d3e` | `#c0b090` |
| Accent (amber) | `#d4874e` | `#d4874e` |
| Code accent (teal) | `#2d8a7b` | `#7dd4c4` |
| Soft BG | `#f2e1c0` | `#2d2620` |
| Meta text | `#a08070` | `#8d8070` |

## Typography

- **Body**: Nunito + PingFang SC / Microsoft YaHei, line-height 1.7
- **Code**: Fira Code (CDN), rounded code blocks
- **Headings**: Same font bold weight, `letter-spacing: -0.005em`

## Post Page Features (all 6)

1. Reading progress bar — 3px thin bar at page top
2. TOC sidebar — floating table of contents
3. Series navigation — prev/next post at bottom
4. Post meta card — tag pills, date, reading time, refined layout
5. Enhanced code blocks — filename title bar + copy button
6. Back-to-top — floating button, bottom-right

## Navigation & Footer

- **Nav**: Sticky glassmorphism (`backdrop-filter: blur`), amber border accent
- **Footer**: Minimal — copyright + tagline only

## Animations (3 items)

1. Color scheme transition — `transition: 300ms` on body
2. Link underline — left-to-right expand on hover (CSS `::after`)
3. Smooth scrolling — `scroll-behavior: smooth`

## Implementation Strategy

- Override hugo-coder templates via `layouts/` directory
- Rewrite `assets/css/custom.css` with full style overrides
- Add `assets/js/custom.js` for progress bar, TOC, copy button, back-to-top
- Load Fira Code from CDN in custom head partial
- All changes scoped to the existing theme — no theme switch, no Hugo config restructure

## Files to Create/Modify

### New files
- `assets/js/custom.js` — reading progress, TOC generation, copy button, back-to-top
- `layouts/partials/head/custom-head.html` — Fira Code CDN, custom JS loading
- `layouts/partials/posts/series-nav.html` — series navigation partial

### Modified files
- `assets/css/custom.css` — complete restyle (colors, typography, cards, nav, footer, code blocks)
- `layouts/_partials/posts/math.html` — no changes needed (keep as-is)
- `layouts/posts/single.html` — add progress bar, TOC, series nav, meta card, code enhancements

## Scope

- Homepage keeps current structure, only visual refresh
- English and Chinese content both covered by same CSS/JS
- No changes to `hugo.toml` configuration
