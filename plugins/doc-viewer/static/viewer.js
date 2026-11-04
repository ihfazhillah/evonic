(function () {
  var src = JSON.parse(document.getElementById('doc-src').textContent);
  var md = window.markdownit({ html: false, linkify: true, typographer: true });
  var fence = md.renderer.rules.fence, diagrams = [];
  md.renderer.rules.fence = function (t, i, o, e, s) {
    var lang = (t[i].info || '').trim().split(/\s+/)[0];
    if (lang === 'mermaid') {
      diagrams.push(t[i].content);
      return '<div class="diagram dg-' + (diagrams.length - 1) + '"></div>';
    }
    return fence(t, i, o, e, s);
  };
  var KINDS = ['note','tip','warning','danger','step','card'];
  md.block.ruler.before('fence', 'directive', function (state, start, end, silent) {
    var pos = state.bMarks[start] + state.tShift[start], max = state.eMarks[start];
    var line = state.src.slice(pos, max), m = /^:::\s*([a-z]+)\s*(.*)$/.exec(line);
    if (!m || KINDS.indexOf(m[1]) < 0) return false;
    if (silent) return true;
    var next = start + 1, depth = 1;
    for (; next < end; next++) {
      var l = state.src.slice(state.bMarks[next] + state.tShift[next], state.eMarks[next]).trim();
      if (/^:::\s*[a-z]+/.test(l)) depth++;
      else if (l === ':::' && --depth === 0) break;
    }
    var open = state.push('directive_open', 'div', 1);
    open.attrSet('class', 'dv dv-' + m[1]); open.info = m[2]; open.map = [start, next];
    if (m[2]) { var t = state.push('html_block', '', 0); t.content = '<div class="dv-title">' + md.utils.escapeHtml(m[2]) + '</div>'; }
    var old = state.parentType, oldMax = state.lineMax;
    state.parentType = 'directive'; state.lineMax = next;
    state.md.block.tokenize(state, start + 1, next);
    state.parentType = old; state.lineMax = oldMax;
    state.push('directive_close', 'div', -1);
    state.line = next + 1;
    return true;
  }, { alt: ['paragraph', 'blockquote', 'list'] });
  var root = document.getElementById('doc');
  root.innerHTML = DOMPurify.sanitize(md.render(src));
  diagrams.forEach(function (code, n) {
    var el = root.querySelector('.dg-' + n);
    if (!el) return;
    Promise.resolve().then(function () { return pele.render(code); })
      .then(function (out) {
        var svg = typeof out === 'string' ? out : (out && out.svg) || '';
        el.innerHTML = DOMPurify.sanitize(svg, { USE_PROFILES: { svg: true, svgFilters: true } });
      })
      .catch(function (err) {
        el.className = 'diagram diagram-error';
        el.textContent = 'Diagram error: ' + (err && err.message || err);
      });
  });
})();
