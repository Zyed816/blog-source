(function () {
  'use strict';

  // ── Reading Progress Bar ──────────────────────────────────
  var progressBar = document.createElement('div');
  progressBar.id = 'reading-progress';
  document.body.prepend(progressBar);

  function updateProgress() {
    var scrollTop = window.scrollY;
    var docHeight = document.documentElement.scrollHeight - window.innerHeight;
    if (docHeight > 0) {
      progressBar.style.width = ((scrollTop / docHeight) * 100) + '%';
    }
  }
  window.addEventListener('scroll', updateProgress, { passive: true });

  // ── Back to Top Button ────────────────────────────────────
  var backBtn = document.createElement('button');
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
  var article = document.querySelector('.post article');
  if (article) {
    var headings = article.querySelectorAll('h2, h3');
    if (headings.length >= 2) {
      var tocNav = document.createElement('nav');
      tocNav.className = 'toc-sidebar';
      var tocTitle = document.createElement('div');
      tocTitle.className = 'toc-title';
      tocTitle.textContent = '📑 目录';
      tocNav.appendChild(tocTitle);

      headings.forEach(function (h) {
        var link = document.createElement('a');
        link.href = '#' + h.id;
        link.textContent = h.textContent;
        link.className = h.tagName === 'H3' ? 'toc-h3' : '';
        tocNav.appendChild(link);
      });

      var postContent = article.querySelector('.post-content');
      if (postContent) {
        var wrapper = document.createElement('div');
        wrapper.className = 'post-with-toc';
        var body = document.createElement('div');
        body.className = 'post-body';

        // Move post-content children into body
        while (postContent.firstChild) {
          body.appendChild(postContent.firstChild);
        }
        wrapper.appendChild(body);
        wrapper.appendChild(tocNav);
        postContent.appendChild(wrapper);

        // Highlight active TOC item on scroll
        var tocLinks = tocNav.querySelectorAll('a');
        function highlightToc() {
          var currentId = '';
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
    var pres = document.querySelectorAll('.post-content pre, .highlight pre');
    pres.forEach(function (pre) {
      if (pre.closest('.code-block-wrapper')) return;

      var wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';

      var header = document.createElement('div');
      header.className = 'code-block-header';

      var langSpan = document.createElement('span');
      var codeEl = pre.querySelector('code');
      var langClass = codeEl ? codeEl.className : '';
      var langMatch = langClass.match(/language-(\w+)/);
      langSpan.textContent = langMatch ? langMatch[1] : 'code';

      var copyBtn = document.createElement('button');
      copyBtn.className = 'copy-btn';
      copyBtn.textContent = 'Copy';

      copyBtn.addEventListener('click', function () {
        var code = pre.textContent;
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

      var highlightDiv = pre.closest('.highlight');
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
