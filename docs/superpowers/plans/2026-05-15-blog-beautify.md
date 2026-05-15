# Blog Beautification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Beautify the hugo-coder blog with amber-teal color palette, rounded typography (Nunito + Fira Code), enhanced post pages (progress bar, TOC, series nav, code copy, back-to-top), glassmorphism nav, minimal footer, and subtle CSS animations.

**Architecture:** All changes are Hugo template overrides + custom CSS/JS injected via the theme's existing hooks (`customCSS`, `customJS`, `layouts/` overrides). No theme switch, no config restructure. The theme's SCSS loads first, then custom.css overrides via equivalent or higher specificity selectors.

**Tech Stack:** Hugo extended (SCSS via Hugo Pipes), vanilla CSS/JS, Fira Code (Google Fonts CDN), Nunito (Google Fonts CDN), Font Awesome (already bundled with theme)

---

### Task 1: Config and head extensions

**Files:**
- Modify: `hugo.toml`
- Create: `layouts/partials/head/extensions.html`

- [ ] **Step 1: Add customJS to hugo.toml**

Open `hugo.toml`. The line `customCSS = ["css/custom.css"]` is at L83. Add `customJS` below it:

```toml
  customCSS = ["css/custom.css"]
  customJS = ["js/custom.js"]
```

- [ ] **Step 2: Create head extensions partial**

Create `layouts/partials/head/extensions.html` (overrides the theme's empty partial at `themes/coder/layouts/partials/head/extensions.html`):

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Nunito:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&display=swap" rel="stylesheet">

<style>
  html { scroll-behavior: smooth; }
</style>
```

- [ ] **Step 3: Commit**

```bash
git add hugo.toml layouts/partials/head/extensions.html
git commit -m "feat: add customJS hook and head extensions (Fira Code, Nunito, smooth scroll)"
```

---

### Task 2: custom.css — complete rewrite

**Files:**
- Modify: `assets/css/custom.css`

This file replaces the existing custom.css entirely. The new content is below (write the full file).

- [ ] **Step 1: Write the new custom.css**

```css
/* ===== Design Tokens ===== */
:root {
  --bg: #fdfaf3;
  --bg-card: #fff;
  --bg-soft: #f2e1c0;
  --fg: #5c4d3e;
  --fg-heading: #5c3d2e;
  --fg-meta: #a08070;
  --accent: #d4874e;
  --accent-teal: #2d8a7b;
  --border: #e8d5b7;
  --shadow-sm: 0 1px 6px rgba(92, 61, 46, 0.06);
  --shadow-md: 0 2px 12px rgba(92, 61, 46, 0.08);
  --radius-sm: 6px;
  --radius: 10px;
  --radius-lg: 14px;
  --font: 'Nunito', 'PingFang SC', 'Microsoft YaHei UI', 'Microsoft YaHei', 'Hiragino Sans GB', 'WenQuanYi Micro Hei', sans-serif;
  --font-mono: 'Fira Code', 'Cascadia Code', 'JetBrains Mono', SFMono-Regular, Consolas, monospace;
  --transition-speed: 0.3s;
}

body.colorscheme-dark {
  --bg: #1e1a16;
  --bg-card: #252019;
  --bg-soft: #2d2620;
  --fg: #c0b090;
  --fg-heading: #f0c58a;
  --fg-meta: #8d8070;
  --accent: #d4874e;
  --accent-teal: #7dd4c4;
  --border: #3d3530;
  --shadow-sm: 0 1px 6px rgba(0, 0, 0, 0.2);
  --shadow-md: 0 2px 12px rgba(0, 0, 0, 0.3);
}

/* ===== Color scheme transition ===== */
body {
  transition: background-color var(--transition-speed) ease, color var(--transition-speed) ease;
}
body * {
  transition: background-color var(--transition-speed) ease,
              color var(--transition-speed) ease,
              border-color var(--transition-speed) ease,
              box-shadow var(--transition-speed) ease;
}

/* ===== Base typography override ===== */
body, h1, h2, h3, h4, h5, h6, p, a, li {
  font-family: var(--font);
}

body {
  color: var(--fg);
  background-color: var(--bg);
  font-size: 1.8em;
  font-weight: 400;
  line-height: 1.75;
}

h1, h2, h3, h4, h5, h6 {
  color: var(--fg-heading);
  font-weight: 700;
  letter-spacing: -0.005em;
}

h1 { font-size: 2.8rem; line-height: 1.3; }
h2 { font-size: 2.4rem; line-height: 1.35; }
h3 { font-size: 2rem; line-height: 1.4; }
h4 { font-size: 1.8rem; line-height: 1.45; }

a {
  color: var(--accent);
  font-weight: 600;
  text-decoration: none;
  position: relative;
}

/* Link underline animation (left-to-right on hover) */
a:not(.navigation-title):not(.title-link):not(.heading-link):not([rel="external"]) {
  background-image: linear-gradient(var(--accent), var(--accent));
  background-size: 0% 1.5px;
  background-repeat: no-repeat;
  background-position: 0% 100%;
  transition: background-size 0.3s ease;
}
a:not(.navigation-title):not(.title-link):not(.heading-link):not([rel="external"]):hover {
  background-size: 100% 1.5px;
  text-decoration: none;
}

.title-link {
  color: var(--fg-heading);
  font-weight: 700;
}
.title-link:hover {
  color: var(--accent);
  text-decoration: none;
}

code {
  font-family: var(--font-mono);
  font-size: 0.88em;
  padding: 0.2em 0.5em;
  border-radius: var(--radius-sm);
  background-color: var(--bg-soft);
  color: var(--fg);
}

pre {
  font-family: var(--font-mono);
  border-radius: var(--radius);
  margin: 2rem 0;
}

pre code {
  background-color: transparent;
  padding: 0;
}

/* Code block wrapper (for filename bar) */
.code-block-wrapper {
  margin: 2rem 0;
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--border);
}
.code-block-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 16px;
  background: var(--bg-soft);
  border-bottom: 1px solid var(--border);
  font-family: var(--font-mono);
  font-size: 0.78em;
  color: var(--fg-meta);
}
.code-block-header .copy-btn {
  background: none;
  border: 1px solid var(--border);
  color: var(--fg-meta);
  padding: 2px 10px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-family: var(--font);
  font-size: 0.85em;
  transition: all 0.2s ease;
}
.code-block-header .copy-btn:hover {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
.code-block-header .copy-btn.copied {
  background: var(--accent-teal);
  color: #fff;
  border-color: var(--accent-teal);
}
.code-block-wrapper pre,
.code-block-wrapper .highlight {
  margin: 0;
}
.code-block-wrapper .highlight {
  border-radius: 0;
}

/* ===== Reading progress bar ===== */
#reading-progress {
  position: fixed;
  top: 0;
  left: 0;
  width: 0%;
  height: 3px;
  background: linear-gradient(90deg, var(--accent), var(--accent-teal));
  z-index: 1000;
  transition: width 0.1s linear;
}

/* ===== Navigation (glassmorphism) ===== */
.navigation {
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(253, 250, 243, 0.85);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(212, 135, 78, 0.15);
  height: auto;
  padding: 12px 0;
}
body.colorscheme-dark .navigation {
  background: rgba(30, 26, 22, 0.85);
  border-bottom-color: rgba(212, 135, 78, 0.2);
}

.navigation a, .navigation span {
  font-family: var(--font);
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--fg-heading);
  text-shadow: none;
  letter-spacing: 0;
}
.navigation a:hover, .navigation a:focus {
  color: var(--accent);
  text-decoration: none;
}
.navigation .navigation-title {
  font-size: 1.9rem;
  font-weight: 800;
  letter-spacing: -0.01em;
  text-transform: none;
}

.navigation .container {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.navigation-list {
  display: flex;
  align-items: center;
  gap: 0;
}
.navigation-list .navigation-item a,
.navigation-list .navigation-item span {
  margin: 0 1rem;
}

@media only screen and (max-width: 768px) {
  .navigation {
    padding: 8px 0;
  }
  .navigation a, .navigation span {
    font-size: 1.5rem;
  }
  .navigation .navigation-title {
    font-size: 1.7rem;
  }
  .navigation-list {
    background-color: var(--bg);
    border-top: solid 1px var(--border);
    border-bottom: solid 1px var(--border);
  }
}

#dark-mode-toggle {
  font-size: 2rem;
  bottom: 1.5rem;
  left: 1.5rem;
}

/* ===== Footer (minimal) ===== */
.footer {
  font-size: 1.5rem;
  line-height: 1.8;
  margin-bottom: 2rem;
  color: var(--fg-meta);
  border-top: 1px solid var(--border);
  padding-top: 2rem;
}
.footer a {
  color: var(--accent);
}

/* ===== Post page ===== */
.container.post {
  max-width: 90rem;
}

/* Post meta card */
.post {
  .post-title {
    margin-bottom: 0.5em;
  }
  .post-title h1.title {
    font-size: 3rem;
    line-height: 1.3;
    font-weight: 800;
  }
  .post-meta {
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }
  .post-meta .date {
    display: flex;
    gap: 1rem;
    align-items: center;
    color: var(--fg-meta);
    font-size: 0.88em;
    margin-bottom: 0.8rem;
  }
  .post-meta .date i {
    color: var(--accent);
    width: auto;
    margin-right: 0.3rem;
  }

  /* Tag pills */
  .post-meta .tags .tag {
    display: inline-block;
    padding: 0.25em 0.7em;
    background: var(--bg-soft);
    border-radius: 20px;
    line-height: 1.5;
    margin: 0.15rem 0.3rem 0.15rem 0;
    font-size: 0.82em;
  }
  .post-meta .tags .tag a {
    color: var(--fg-meta);
  }
  .post-meta .tags .tag:hover {
    background: var(--accent);
  }
  .post-meta .tags .tag:hover a {
    color: #fff;
  }
}

/* Post content */
.post-content {
  line-height: 1.8;
}
.post-content p {
  text-align: justify;
  margin: 1.6rem 0;
}
.post-content img {
  border-radius: var(--radius);
  box-shadow: var(--shadow-sm);
}

/* TOC sidebar layout */
.post-with-toc {
  display: flex;
  gap: 3rem;
  align-items: flex-start;
}
.post-with-toc .post-body {
  flex: 1;
  min-width: 0;
}
.post-with-toc .toc-sidebar {
  width: 200px;
  flex-shrink: 0;
  position: sticky;
  top: 6rem;
  font-size: 0.82em;
  border-left: 1px solid var(--border);
  padding-left: 1.5rem;
  max-height: calc(100vh - 8rem);
  overflow-y: auto;
}
.toc-sidebar .toc-title {
  font-weight: 700;
  color: var(--fg-heading);
  margin-bottom: 0.8rem;
  font-size: 0.95em;
}
.toc-sidebar a {
  display: block;
  color: var(--fg-meta);
  padding: 0.3em 0;
  font-weight: 500;
  border-left: 2px solid transparent;
  padding-left: 0.8rem;
  margin-left: -1px;
  transition: all 0.2s ease;
}
.toc-sidebar a:hover,
.toc-sidebar a.active {
  color: var(--accent);
  border-left-color: var(--accent);
}
.toc-sidebar a.toc-h3 { padding-left: 1.5rem; }

@media only screen and (max-width: 1100px) {
  .toc-sidebar {
    display: none;
  }
}

/* Series navigation */
.series-nav {
  margin-top: 3rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border);
}
.series-nav .series-label {
  text-align: center;
  font-size: 0.85em;
  color: var(--fg-meta);
  margin-bottom: 1rem;
}
.series-nav .series-links {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
}
.series-nav .series-links a {
  font-size: 0.9em;
  color: var(--accent);
  font-weight: 600;
}

/* ===== Post list (homepage) ===== */
.list ul li {
  display: flex;
  align-items: baseline;
  padding: 1rem 0.8rem;
  margin: 0.2rem 0;
  border-radius: var(--radius);
  transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
  border-left: 3px solid transparent;
}
.list ul li:hover {
  transform: translateX(4px);
  background-color: var(--bg-card);
  box-shadow: var(--shadow-sm);
  border-left-color: var(--accent);
}
.list ul li .date {
  display: inline-block;
  flex: 0 0 11rem;
  text-align: right;
  margin-right: 2rem;
  color: var(--fg-meta);
  font-size: 0.88em;
}
.list ul li .title {
  flex: 1;
  font-size: 1.8rem;
  font-weight: 700;
  color: var(--fg-heading);
}
.list ul li .title:hover {
  color: var(--accent);
  text-decoration: none;
}
@media only screen and (max-width: 768px) {
  .list ul li {
    flex-direction: column;
    align-items: flex-start;
  }
  .list ul li .date {
    text-align: left;
    margin-right: 0;
    margin-bottom: 0.3rem;
  }
  .list ul li:hover {
    transform: none;
  }
}

/* Pagination */
.pagination li a {
  border-radius: var(--radius-sm);
}

/* ===== Back to top button ===== */
#back-to-top {
  position: fixed;
  bottom: 2rem;
  right: 2rem;
  z-index: 99;
  width: 42px;
  height: 42px;
  border-radius: 50%;
  background: var(--bg-card);
  border: 1px solid var(--border);
  box-shadow: var(--shadow-md);
  color: var(--fg-meta);
  font-size: 1.2em;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  visibility: hidden;
  transform: translateY(10px);
  transition: opacity 0.3s ease, visibility 0.3s ease, transform 0.3s ease;
}
#back-to-top.visible {
  opacity: 1;
  visibility: visible;
  transform: translateY(0);
}
#back-to-top:hover {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}

/* ===== Blockquote (preserved from original custom.css, adapted) ===== */
.post-content blockquote {
  --quote-pad: 1rem;
  margin: 1.6rem 0;
  margin-left: calc(-1 * var(--quote-pad));
  padding: 0.9rem 1rem 0.9rem var(--quote-pad);
  border-left: 3px solid var(--accent);
  background-color: var(--bg-soft);
  border-radius: var(--radius);
  font-style: normal;
}
.post-content blockquote > :first-child { margin-top: 0; }
.post-content blockquote > :last-child { margin-bottom: 0; }

/* ===== KaTeX (preserved from original) ===== */
.katex { font-size: 1.05em; }
.katex, .katex * { font-family: KaTeX_Main, KaTeX_Math, serif !important; }
.katex-display {
  overflow-x: auto;
  overflow-y: hidden;
  padding: 0.25rem 0;
}
.katex-display > .katex { max-width: 100%; }
.katex .base { line-height: 1.3; }

/* ===== Homepage centered section ===== */
.centered .about h1 {
  font-weight: 800;
  color: var(--fg-heading);
}
.centered .about h2 {
  color: var(--fg-meta);
  font-weight: 500;
  font-size: 2rem;
}
.centered .about ul li a {
  color: var(--fg-meta);
  font-size: 1.6rem;
  transition: color 0.2s ease;
}
.centered .about ul li a:hover {
  color: var(--accent);
}

/* ===== Avatar ===== */
.avatar img {
  border-radius: 50%;
  box-shadow: var(--shadow-md);
}

/* ===== Content area ===== */
.content header {
  margin-top: 3rem;
  margin-bottom: 2rem;
}
.content header h1 {
  font-size: 3.6rem;
  line-height: 1.3;
  margin: 0;
}

/* ===== Table ===== */
table td, table th {
  padding: 1.2rem;
  border-color: var(--border);
}
```

- [ ] **Step 2: Commit**

```bash
git add assets/css/custom.css
git commit -m "feat: rewrite custom.css with amber-teal palette, glass nav, post enhancements"
```

---

### Task 3: custom.js — progress bar, TOC, code copy, back-to-top

**Files:**
- Create: `assets/js/custom.js`

- [ ] **Step 1: Write custom.js**

```javascript
(function () {
  'use strict';

  // ── Reading Progress Bar ──────────────────────────────────
  const progressBar = document.createElement('div');
  progressBar.id = 'reading-progress';
  document.body.prepend(progressBar);

  function updateProgress() {
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    if (docHeight > 0) {
      progressBar.style.width = ((scrollTop / docHeight) * 100) + '%';
    }
  }
  window.addEventListener('scroll', updateProgress, { passive: true });

  // ── Back to Top Button ────────────────────────────────────
  const backBtn = document.createElement('button');
  backBtn.id = 'back-to-top';
  backBtn.innerHTML = '&#8593;';
  backBtn.setAttribute('aria-label', 'Back to top');
  document.body.appendChild(backBtn);

  backBtn.addEventListener('click', function () {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  function toggleBackBtn() {
    if (window.scrollY > 400) {
      backBtn.classList.add('visible');
    } else {
      backBtn.classList.remove('visible');
    }
  }
  window.addEventListener('scroll', toggleBackBtn, { passive: true });
  toggleBackBtn();

  // ── Table of Contents ─────────────────────────────────────
  const article = document.querySelector('.post article');
  if (article) {
    const headings = article.querySelectorAll('h2, h3');
    if (headings.length >= 2) {
      const tocNav = document.createElement('nav');
      tocNav.className = 'toc-sidebar';
      const tocTitle = document.createElement('div');
      tocTitle.className = 'toc-title';
      tocTitle.textContent = '📑 目录';
      tocNav.appendChild(tocTitle);

      headings.forEach(function (h) {
        const link = document.createElement('a');
        link.href = '#' + h.id;
        link.textContent = h.textContent;
        link.className = h.tagName === 'H3' ? 'toc-h3' : '';
        tocNav.appendChild(link);
      });

      const postContent = article.querySelector('.post-content');
      if (postContent) {
        const wrapper = document.createElement('div');
        wrapper.className = 'post-with-toc';
        const body = document.createElement('div');
        body.className = 'post-body';

        // Move post-content children into body
        while (postContent.firstChild) {
          body.appendChild(postContent.firstChild);
        }
        wrapper.appendChild(body);
        wrapper.appendChild(tocNav);
        postContent.appendChild(wrapper);

        // Highlight active TOC item on scroll
        const tocLinks = tocNav.querySelectorAll('a');
        function highlightToc() {
          let currentId = '';
          headings.forEach(function (h) {
            if (h.getBoundingClientRect().top <= 100) {
              currentId = h.id;
            }
          });
          tocLinks.forEach(function (link) {
            link.classList.toggle('active', link.getAttribute('href') === '#' + currentId);
          });
        }
        window.addEventListener('scroll', highlightToc, { passive: true });
        highlightToc();
      }
    }
  }

  // ── Code Block Copy Button ────────────────────────────────
  function wrapCodeBlocks() {
    const pres = document.querySelectorAll('.post-content pre, .highlight pre');
    pres.forEach(function (pre) {
      if (pre.closest('.code-block-wrapper')) return;

      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';

      const header = document.createElement('div');
      header.className = 'code-block-header';

      const langSpan = document.createElement('span');
      const codeEl = pre.querySelector('code');
      const langClass = codeEl ? codeEl.className : '';
      const langMatch = langClass.match(/language-(\w+)/);
      langSpan.textContent = langMatch ? langMatch[1] : 'code';

      const copyBtn = document.createElement('button');
      copyBtn.className = 'copy-btn';
      copyBtn.textContent = 'Copy';

      copyBtn.addEventListener('click', function () {
        const code = pre.textContent;
        navigator.clipboard.writeText(code).then(function () {
          copyBtn.textContent = 'Copied!';
          copyBtn.classList.add('copied');
          setTimeout(function () {
            copyBtn.textContent = 'Copy';
            copyBtn.classList.remove('copied');
          }, 2000);
        });
      });

      header.appendChild(langSpan);
      header.appendChild(copyBtn);
      wrapper.appendChild(header);

      const highlightDiv = pre.closest('.highlight');
      if (highlightDiv) {
        pre.parentNode.insertBefore(wrapper, highlightDiv);
        wrapper.appendChild(highlightDiv);
      } else {
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);
      }
    });
  }
  wrapCodeBlocks();

})();
```

- [ ] **Step 2: Commit**

```bash
git add assets/js/custom.js
git commit -m "feat: add custom.js with progress bar, TOC, code copy, back-to-top"
```

---

### Task 4: Template overrides

**Files:**
- Create: `layouts/partials/header.html`
- Create: `layouts/partials/footer.html`
- Create: `layouts/posts/single.html`
- Create: `layouts/posts/li.html`
- Create: `layouts/partials/posts/series-nav.html`

Each file overrides the corresponding theme template. Copy the theme's version first, then modify.

- [ ] **Step 1: Override header.html**

Copy `themes/coder/layouts/partials/header.html` to `layouts/partials/header.html`. The theme version already has the right structure. Only change: remove `text-transform: uppercase` behavior which we handled in CSS. No template changes needed — the CSS handles all visual changes.

The file at `layouts/partials/header.html` is identical to the theme's original (the CSS handles glassmorphism).

```bash
cp themes/coder/layouts/partials/header.html layouts/partials/header.html
```

- [ ] **Step 2: Override footer.html with minimal version**

Create `layouts/partials/footer.html`:

```html
<footer class="footer">
  <section class="container">
    &copy;
    {{ if (and .Site.Params.since (lt .Site.Params.since now.Year)) }}
      {{ .Site.Params.since }} -
    {{ end }}
    {{ now.Year }}
    {{ with .Site.Params.author }} {{ . }} {{ end }}
    {{ with .Site.Params.tagline }}
      · {{ . }}
    {{ end }}
  </section>
</footer>
```

- [ ] **Step 3: Override posts/single.html**

Create `layouts/posts/single.html`:

```html
{{ define "title" }}
  {{ .Title }} · {{ .Site.Title }}
{{ end }}
{{ define "content" }}
  {{ $hasToc := and (ge (len .TableOfContents) 80) (ne .Params.toc false) }}
  <section class="container post">
    <article>
      <header>
        <div class="post-title">
          <h1 class="title">
            <a class="title-link" href="{{ .Permalink | safeURL }}">
              {{ .Title }}
            </a>
          </h1>
        </div>
        <div class="post-meta">
          <div class="date">
            <span class="posted-on">
              <i class="fa-solid fa-calendar" aria-hidden="true"></i>
              <time datetime="{{ .Date.Format "2006-01-02T15:04:05Z07:00" }}">
                {{ .Date | time.Format (.Site.Params.dateFormat | default "January 2, 2006" ) }}
              </time>
            </span>
            <span class="reading-time">
              <i class="fa-solid fa-clock" aria-hidden="true"></i>
              {{ i18n "reading_time" .ReadingTime }}
            </span>
          </div>
          {{ with .GetTerms "categories" }}{{ partial "taxonomy/categories.html" . }}{{ end }}
          {{ with .GetTerms "tags" }}{{ partial "taxonomy/tags.html" . }}{{ end }}
          {{ with .GetTerms "authors" }}{{ partial "taxonomy/authors.html" . }}{{ end }}
        </div>
      </header>

      <div class="post-content">
        {{ if .Params.featuredImage }}
          <img src="{{ .Params.featuredImage | relURL }}" alt="Featured image"/>
        {{ end }}
        {{ .Content }}
      </div>

      <footer>
        {{ partial "posts/series.html" . }}
        {{ partial "posts/series-nav.html" . }}
        {{ partial "posts/disqus.html" . }}
        {{ partial "posts/commento.html" . }}
        {{ partial "posts/utterances.html" . }}
        {{ partial "posts/giscus.html" . }}
        {{ partial "posts/mastodon.html" . }}
        {{ partial "posts/telegram.html" . }}
        {{ partial "posts/cusdis.html" . }}
      </footer>
    </article>

    {{ partial "posts/math.html" . }}
  </section>
{{ end }}
```

- [ ] **Step 4: Override posts/li.html**

Create `layouts/posts/li.html`:

```html
<li>
  <span class="date">{{ .Date | time.Format (.Site.Params.dateFormat | default "January 2, 2006" ) }}</span>
  <a class="title" href="{{ .Params.externalLink | default .RelPermalink }}">{{ .Title | markdownify }}</a>
</li>
```

Only change from theme: added `| markdownify` to title so that markdown formatting in titles renders.

- [ ] **Step 5: Create series navigation partial**

Create `layouts/partials/posts/series-nav.html`:

```html
{{ if .Params.series }}
  {{ $currentPageUrl := .RelPermalink }}
  {{ range .Params.series }}
    {{ $name := . | anchorize }}
    {{ $series := index $.Site.Taxonomies.series $name }}
    {{ if gt (len $series.Pages) 1 }}
      <div class="series-nav">
        <div class="series-label">📂 {{ . }} 系列</div>
        <div class="series-links">
          {{ with $series.Pages.Prev . }}
            <a href="{{ .RelPermalink }}">&larr; {{ .Title }}</a>
          {{ else }}
            <span></span>
          {{ end }}
          {{ with $series.Pages.Next . }}
            <a href="{{ .RelPermalink }}">{{ .Title }} &rarr;</a>
          {{ else }}
            <span></span>
          {{ end }}
        </div>
      </div>
    {{ end }}
  {{ end }}
{{ end }}
```

- [ ] **Step 6: Commit**

```bash
git add layouts/partials/header.html layouts/partials/footer.html layouts/posts/single.html layouts/posts/li.html layouts/partials/posts/series-nav.html
git commit -m "feat: override templates for glass nav, minimal footer, enhanced post page"
```

---

### Task 5: Build, verify, and final commit

**Files:** (none new — verification only)

- [ ] **Step 1: Build the site with Hugo**

Run: `hugo --minify`
Expected: Build succeeds with no errors.

- [ ] **Step 2: Start Hugo dev server**

Run: `hugo server -D`
Expected: Server starts, site accessible at `http://localhost:1313`.

- [ ] **Step 3: Visual verification checklist**

Open in browser:
1. **Homepage**: Check post list hover effects, amber-teal colors, Nunito font, smooth scrolling
2. **Post page**: Check progress bar on scroll, TOC sidebar (on posts with enough headings), code blocks with filename + copy button, tag pills, series navigation at bottom
3. **Navigation**: Check sticky glassmorphism effect, click links
4. **Footer**: Check minimal footer with copyright only
5. **Dark mode**: Toggle dark/light, verify smooth color transition (300ms)
6. **Mobile**: Resize to 375px, verify responsive layout, hamburger menu
7. **KaTeX**: Open a post with math formulas, verify rendering is not broken

- [ ] **Step 4: Fix any issues discovered during verification**

- [ ] **Step 5: Final commit (if any fixes applied)**

```bash
git add -A
git commit -m "fix: visual verification tweaks for blog beautification"
```
