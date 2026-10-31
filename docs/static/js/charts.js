/* Interactive benchmark charts. No dependencies: the data is fetched as JSON and
   drawn as inline SVG, so the page stays a static GitHub Pages site. */
(function () {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';

  // plot_benchmark_sensor.py's comparison colours, so the page matches the paper's figures.
  var OURS = '#FF6060';
  var PRIOR = '#3F506D';
  var TACMAP = '#75577D';

  function el(name, attrs) {
    var node = document.createElementNS(NS, name);
    for (var k in attrs) if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
    return node;
  }

  function compact(v) {
    if (v === null || v === undefined) return '—';
    if (v >= 1e6) return (v / 1e6).toFixed(v >= 1e7 ? 0 : 1) + 'M';
    if (v >= 1e3) return (v / 1e3).toFixed(v >= 1e4 ? 0 : 1) + 'k';
    return String(Math.round(v));
  }

  function group(cls) {
    var node = document.createElement('div');
    node.className = cls;
    return node;
  }

  function legend(items) {
    var ul = document.createElement('ul');
    ul.className = 'chart__legend';
    items.forEach(function (item) {
      var li = document.createElement('li');
      var key = document.createElement('span');
      key.className = 'chart__key';
      key.style.background = item.color;
      if (item.dashed) key.classList.add('chart__key--dashed');
      if (item.hatch) key.classList.add('chart__key--' + item.hatch);
      li.appendChild(key);
      li.appendChild(document.createTextNode(item.label));
      ul.appendChild(li);
    });
    return ul;
  }

  /* ---- grouped sensor bars, styled after plot_benchmark_sensor.py ------------- */

  function hatchDefs(svg, series) {
    var defs = el('defs', {});
    var made = {};
    series.forEach(function (sel) {
      var key = sel.variant + '-' + sel.id;
      if (!sel.hatch || made[key]) return;
      made[key] = true;
      var pat = el('pattern', {
        id: 'hatch-' + key, width: 6, height: 6, patternUnits: 'userSpaceOnUse'
      });
      pat.appendChild(el('rect', { width: 6, height: 6, fill: sel.color }));
      if (sel.hatch === 'diagonal') {
        pat.appendChild(el('path', {
          d: 'M0 6 L6 0 M-1 1 L1 -1 M5 7 L7 5',
          stroke: darken(sel.color), 'stroke-width': 1.1, fill: 'none'
        }));
      } else {
        pat.appendChild(el('circle', { cx: 3, cy: 3, r: 1.1, fill: darken(sel.color) }));
      }
      defs.appendChild(pat);
    });
    svg.appendChild(defs);
  }

  function darken(hex) {
    var n = parseInt(hex.slice(1), 16);
    var f = 0.55;
    return '#' + [16, 8, 0].map(function (sh) {
      var v = Math.round(((n >> sh) & 255) * f);
      return ('0' + v.toString(16)).slice(-2);
    }).join('');
  }

  function barChartGrouped(panel, metric, variant) {
    var W = 760, H = 380, pad = { l: 78, r: 16, t: 14, b: 52 };
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'chart__svg', role: 'img' });
    svg.appendChild(el('title')).textContent = panel.title + ' \u2014 ' + metric.label;

    var series = panel.series.filter(function (s) { return s.variant === variant.id; });
    if (!series.length) return svg;
    series.forEach(function (s) { s.hatch = variant.hatch; });
    hatchDefs(svg, series);

    var xs = [];
    series.forEach(function (s) {
      s.points.forEach(function (p) { if (xs.indexOf(p.x) < 0) xs.push(p.x); });
    });
    xs.sort(function (a, b) { return a - b; });

    var hi = 0;
    series.forEach(function (s) {
      s.points.forEach(function (p) { hi = Math.max(hi, metric.get(p) + (metric.sd(p) || 0)); });
    });
    hi *= 1.1;
    var Y = function (v) { return H - pad.b - v / hi * (H - pad.t - pad.b); };
    var slot = (W - pad.l - pad.r) / xs.length;
    var width = Math.min(15, (slot - 12) / series.length);

    [0, hi / 4, hi / 2, (3 * hi) / 4, hi].forEach(function (v) {
      svg.appendChild(el('line', { x1: pad.l, x2: W - pad.r, y1: Y(v), y2: Y(v), class: 'chart-grid' }));
      var t = el('text', { x: pad.l - 10, y: Y(v) + 4, class: 'chart-ylabel' });
      t.textContent = compact(v);
      svg.appendChild(t);
    });

    xs.forEach(function (x, i) {
      var centre = pad.l + slot * (i + 0.5);
      var left = centre - (width * series.length + series.length - 1) / 2;
      series.forEach(function (s, j) {
        var point = s.points.filter(function (p) { return p.x === x; })[0];
        if (!point) return;
        var value = metric.get(point);
        var bx = left + j * (width + 1);
        var rect = el('rect', {
          x: bx, y: Y(value), width: width, height: Math.max(1, H - pad.b - Y(value)),
          fill: s.hatch ? 'url(#hatch-' + s.variant + '-' + s.id + ')' : s.color,
          stroke: darken(s.color), 'stroke-width': 0.8, class: 'chart-bar'
        });
        rect.appendChild(el('title')).textContent =
          s.label + ' (' + variant.label + ') at ' + x.toLocaleString() + ': ' +
          metric.format(value) + ' \u00b7 mean of ' + point.runs;
        svg.appendChild(rect);
        var sd = metric.sd(point);
        if (sd) {
          var cx = bx + width / 2;
          svg.appendChild(el('path', {
            d: 'M' + cx + ' ' + Y(value - sd) + 'V' + Y(value + sd) +
               'M' + (cx - 3) + ' ' + Y(value + sd) + 'h6' +
               'M' + (cx - 3) + ' ' + Y(value - sd) + 'h6',
            class: 'chart-err'
          }));
        }
      });
      var t = el('text', { x: centre, y: H - pad.b + 18, class: 'chart-xtick' });
      t.textContent = compact(x);
      svg.appendChild(t);
    });

    var xa = el('text', { x: (pad.l + W - pad.r) / 2, y: H - 8, class: 'chart-axis' });
    xa.textContent = panel.x;
    svg.appendChild(xa);
    var ya = el('text', { x: 15, y: H / 2, class: 'chart-axis', transform: 'rotate(-90 15 ' + H / 2 + ')' });
    ya.textContent = metric.label;
    svg.appendChild(ya);
    return svg;
  }

  function groupedTable(panel, metric, variant) {
    var series = panel.series.filter(function (s) { return s.variant === variant.id; });
    var xs = [];
    series.forEach(function (s) {
      s.points.forEach(function (p) { if (xs.indexOf(p.x) < 0) xs.push(p.x); });
    });
    xs.sort(function (a, b) { return a - b; });
    var head = '<tr><th scope="col">Sensor</th>' +
      xs.map(function (x) { return '<th scope="col">' + x.toLocaleString() + '</th>'; }).join('') + '</tr>';
    var body = series.map(function (s) {
      return '<tr><th scope="row">' + s.label + '</th>' + xs.map(function (x) {
        var p = s.points.filter(function (q) { return q.x === x; })[0];
        return '<td>' + (p ? metric.format(metric.get(p)) : '\u2014') + '</td>';
      }).join('') + '</tr>';
    }).join('');
    return '<table><caption>' + panel.title + ' \u2014 ' + metric.label + ', ' + variant.label +
      '</caption><thead>' + head + '</thead><tbody>' + body + '</tbody></table>';
  }

  /* ---- grouped bars: ours against one prior simulator ------------------------ */

  function barChart(spec) {
    var W = 420, H = 300, pad = { l: 66, r: 14, t: 12, b: 48 };
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'chart__svg', role: 'img' });
    svg.appendChild(el('title')).textContent = spec.title;
    var values = [];
    spec.rows.forEach(function (r) {
      spec.bars.forEach(function (b) { if (r[b.key] !== null && r[b.key] !== undefined) values.push(r[b.key]); });
    });
    var hi = (values.length ? Math.max.apply(null, values) : 1) * 1.15;
    var Y = function (v) { return H - pad.b - v / hi * (H - pad.t - pad.b); };
    var slot = (W - pad.l - pad.r) / spec.rows.length;
    var width = Math.min(26, (slot - 10) / spec.bars.length);

    [0, hi / 2, hi].forEach(function (v) {
      svg.appendChild(el('line', { x1: pad.l, x2: W - pad.r, y1: Y(v), y2: Y(v), class: 'chart-grid' }));
      var t = el('text', { x: pad.l - 10, y: Y(v) + 4, class: 'chart-ylabel' });
      t.textContent = compact(v);
      svg.appendChild(t);
    });

    spec.rows.forEach(function (row, i) {
      var centre = pad.l + slot * (i + 0.5);
      var left = centre - (width * spec.bars.length + 2 * (spec.bars.length - 1)) / 2;
      spec.bars.forEach(function (bar, j) {
        var x = left + j * (width + 2);
        var v = row[bar.key];
        if (v === null || v === undefined) {
          var oom = el('text', { x: x + width / 2, y: H - pad.b - 8, class: 'chart-oom' });
          oom.textContent = bar.missing || 'OOM';
          svg.appendChild(oom);
          return;
        }
        var rect = el('rect', {
          x: x, y: Y(v), width: width, height: Math.max(1, H - pad.b - Y(v)),
          rx: 4, class: 'chart-bar', style: 'fill:' + bar.color
        });
        rect.appendChild(el('title')).textContent =
          bar.label + ' at ' + row.envs.toLocaleString() + ' envs: ' + spec.format(v);
        svg.appendChild(rect);
      });
      var t = el('text', { x: centre, y: H - pad.b + 18, class: 'chart-xtick' });
      t.textContent = compact(row.envs);
      svg.appendChild(t);
    });

    var xa = el('text', { x: (pad.l + W - pad.r) / 2, y: H - 8, class: 'chart-axis' });
    xa.textContent = 'Parallel environments';
    svg.appendChild(xa);
    var ya = el('text', { x: 15, y: H / 2, class: 'chart-axis', transform: 'rotate(-90 15 ' + H / 2 + ')' });
    ya.textContent = spec.axis;
    svg.appendChild(ya);
    return svg;
  }

  function barTable(spec) {
    var head = '<tr><th scope="col">Environments</th>' +
      spec.bars.map(function (b) { return '<th scope="col">' + b.label + '</th>'; }).join('') + '</tr>';
    var body = spec.rows.map(function (row) {
      return '<tr><th scope="row">' + row.envs.toLocaleString() + '</th>' +
        spec.bars.map(function (b) {
          var v = row[b.key];
          return '<td>' + (v === null || v === undefined ? (b.missing || 'OOM') : spec.format(v)) + '</td>';
        }).join('') + '</tr>';
    }).join('');
    return '<table><caption>' + spec.title + '</caption><thead>' + head +
      '</thead><tbody>' + body + '</tbody></table>';
  }

  /* ---- mounts ---------------------------------------------------------------- */

  function tabs(labels, onPick) {
    var bar = group('chart__tabs');
    bar.setAttribute('role', 'tablist');
    labels.forEach(function (label, i) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'chart__tab';
      b.setAttribute('role', 'tab');
      b.setAttribute('aria-selected', String(i === 0));
      b.textContent = label;
      b.addEventListener('click', function () {
        Array.prototype.forEach.call(bar.children, function (c) {
          c.setAttribute('aria-selected', String(c === b));
        });
        onPick(i);
      });
      bar.appendChild(b);
    });
    return bar;
  }

  function withTable(host, build) {
    var figure = document.createElement('figure');
    figure.className = 'chart__figure';
    var caption = document.createElement('figcaption');
    caption.className = 'chart__caption';
    var tableBox = group('chart__table');
    tableBox.hidden = true;
    var toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'chart__toggle';
    toggle.setAttribute('aria-pressed', 'false');
    toggle.textContent = 'Show table';
    toggle.addEventListener('click', function () {
      var shown = toggle.getAttribute('aria-pressed') === 'true';
      toggle.setAttribute('aria-pressed', String(!shown));
      toggle.textContent = shown ? 'Show table' : 'Hide table';
      tableBox.hidden = shown;
    });
    host.appendChild(figure);
    figure.appendChild(caption);
    host.appendChild(toggle);
    host.appendChild(tableBox);
    return function () {
      var built = build();
      var previous = figure.querySelector('svg');
      if (previous) figure.removeChild(previous);
      figure.insertBefore(built.svg, caption);
      caption.innerHTML = '';
      caption.appendChild(built.legend);
      if (built.note) {
        var note = document.createElement('p');
        note.className = 'chart__note';
        note.textContent = built.note;
        caption.appendChild(note);
      }
      tableBox.innerHTML = built.table;
    };
  }

  function mountBenchmark(host, data) {
    var state = { panel: 0, metric: 0, variant: 0 };
    var metrics = [
      {
        label: 'Environment steps per second', short: 'Throughput',
        get: function (p) { return p.fps; }, sd: function (p) { return p.fpsSd; },
        format: function (v) { return Math.round(v).toLocaleString(); }
      },
      {
        label: 'Peak GPU memory (MB)', short: 'Memory',
        get: function (p) { return p.memMb; }, sd: function () { return 0; },
        format: function (v) { return Math.round(v).toLocaleString() + ' MB'; }
      }
    ];

    var controls = group('chart__controls');
    controls.appendChild(tabs(data.panels.map(function (p) { return p.title; }),
      function (i) { state.panel = i; state.variant = 0; rebuild(); }));
    controls.appendChild(tabs(metrics.map(function (m) { return m.short; }),
      function (i) { state.metric = i; render(); }));
    var variantBar = group('chart__tabs');
    controls.appendChild(variantBar);
    host.appendChild(controls);

    var render = withTable(host, function () {
      var panel = data.panels[state.panel];
      var available = data.variants.filter(function (v) {
        return panel.series.some(function (s) { return s.variant === v.id; });
      });
      var variant = available[Math.min(state.variant, available.length - 1)];
      var metric = metrics[state.metric];
      var shown = panel.series.filter(function (s) { return s.variant === variant.id; });
      return {
        svg: barChartGrouped(panel, metric, variant),
        legend: legend(shown.map(function (s) {
          return { label: s.label, color: s.color, hatch: variant.hatch };
        })),
        note: panel.sub + '. Measured on ' + data.gpu +
              '; bars are the mean over repeated runs, whiskers their spread.',
        table: groupedTable(panel, metric, variant)
      };
    });

    function rebuild() {
      var panel = data.panels[state.panel];
      var available = data.variants.filter(function (v) {
        return panel.series.some(function (s) { return s.variant === v.id; });
      });
      var next = tabs(available.map(function (v) { return v.label; }),
        function (i) { state.variant = i; render(); });
      controls.replaceChild(next, variantBar);
      variantBar = next;
      variantBar.hidden = available.length < 2;
      render();
    }

    rebuild();
  }

  function mountComparison(host, data) {
    var specs = [
      {
        name: 'TacSL',
        title: 'KinematicTaxel throughput against TacSL',
        axis: 'Environment steps per second',
        rows: data.tacsl['100'].filter(function (r) { return r.ours !== null && r.ours !== undefined; }),
        bars: [{ key: 'ours', label: 'Ours (10x10 taxels)', color: OURS }, { key: 'tacsl', label: 'TacSL (10x10)', color: PRIOR, missing: 'n/a' }],
        format: function (v) { return Math.round(v).toLocaleString(); },
        note: 'TacSL values are the numbers its paper reports. Ours are the committed RTX 5090 sweep.'
      },
      {
        name: 'HydroShear',
        title: 'Tactile policy rollout against HydroShear',
        axis: 'Environment steps per second',
        rows: data.hydroshear,
        bars: [{ key: 'ours', label: 'Ours (five 5x4 arrays)', color: OURS }, { key: 'hydroshear', label: 'HydroShear (one 7x9)', color: PRIOR, missing: 'n/a' }],
        format: function (v) { return Math.round(v).toLocaleString(); },
        note: 'HydroShear reports milliseconds per step; converted to environment steps per second. It does not report 2048.'
      },
      {
        name: 'TacMap',
        title: 'Depth queries against TacMap',
        axis: 'Environment steps per second',
        rows: data.tacmap,
        bars: [{ key: 'oursFps', label: 'Ours', color: OURS }, { key: 'tacmapFps', label: 'TacMap', color: TACMAP }],
        format: function (v) { return Math.round(v).toLocaleString(); },
        note: 'Five ElastomerTaxel sensors of over 1000 taxels each. TacMap runs out of memory at 2048.'
      },
      {
        name: 'TacMap memory',
        title: 'GPU memory against TacMap',
        axis: 'Peak GPU memory (MB)',
        rows: data.tacmap,
        bars: [{ key: 'oursMemMb', label: 'Ours', color: OURS }, { key: 'tacmapMemMb', label: 'TacMap', color: TACMAP }],
        format: function (v) { return Math.round(v).toLocaleString() + ' MB'; },
        note: 'Same configuration as the panel beside it. Lower is better.'
      }
    ];
    var state = { i: 0 };
    var controls = group('chart__controls');
    controls.appendChild(tabs(specs.map(function (s) { return s.name; }), function (i) { state.i = i; render(); }));
    host.appendChild(controls);
    var render = withTable(host, function () {
      var spec = specs[state.i];
      return {
        svg: barChart(spec),
        legend: legend(spec.bars.map(function (b) { return { label: b.label, color: b.color }; })),
        note: spec.note,
        table: barTable(spec)
      };
    });
    render();
  }

  function boot(selector, file, mount) {
    var host = document.querySelector(selector);
    if (!host) return;
    fetch(file)
      .then(function (r) { return r.json(); })
      .then(function (data) { mount(host, data); })
      .catch(function () {
        host.innerHTML = '<p class="chart__error">The interactive chart could not load. ' +
          'The figure below shows the same measurements.</p>';
      });
  }

  boot('#benchmark-chart', 'static/data/benchmark.json', mountBenchmark);
  boot('#comparison-chart', 'static/data/comparison.json', mountComparison);
})();

/* Hand layout picker: hand x layout, from static/data/layouts.json. */
(function () {
  'use strict';
  var host = document.querySelector('#layout-picker');
  if (!host) return;

  fetch('static/data/layouts.json').then(function (r) { return r.json(); }).then(function (hands) {
    if (!hands.length) return;
    var state = { hand: 0, layout: 1 };

    function tabRow(labels, cls, pick) {
      var bar = document.createElement('div');
      bar.className = 'chart__tabs';
      bar.setAttribute('role', 'tablist');
      labels.forEach(function (label, i) {
        var b = document.createElement('button');
        b.type = 'button';
        b.className = 'chart__tab';
        b.setAttribute('role', 'tab');
        b.textContent = label;
        b.addEventListener('click', function () { pick(i); });
        bar.appendChild(b);
      });
      bar.dataset.kind = cls;
      return bar;
    }

    var controls = document.createElement('div');
    controls.className = 'chart__controls';
    var handBar = tabRow(hands.map(function (h) { return h.label; }), 'hand', function (i) {
      state.hand = i;
      state.layout = Math.min(state.layout, hands[i].layouts.length - 1);
      rebuild();
    });
    controls.appendChild(handBar);
    var layoutBar = document.createElement('div');
    controls.appendChild(layoutBar);
    host.appendChild(controls);

    var figure = document.createElement('figure');
    figure.className = 'layout__figure';
    var img = document.createElement('img');
    img.loading = 'lazy';
    var caption = document.createElement('figcaption');
    caption.className = 'layout__caption';
    figure.appendChild(img);
    figure.appendChild(caption);
    host.appendChild(figure);

    function rebuild() {
      var hand = hands[state.hand];
      var next = tabRow(
        hand.layouts.map(function (l) { return l.label + ' \u00b7 ' + l.count; }),
        'layout',
        function (i) { state.layout = i; render(); }
      );
      controls.replaceChild(next, layoutBar);
      layoutBar = next;
      render();
    }

    function render() {
      var hand = hands[state.hand];
      var layout = hand.layouts[state.layout];
      Array.prototype.forEach.call(handBar.children, function (b, i) {
        b.setAttribute('aria-selected', String(i === state.hand));
      });
      Array.prototype.forEach.call(layoutBar.children, function (b, i) {
        b.setAttribute('aria-selected', String(i === state.layout));
      });
      img.src = 'static/images/layouts/' + layout.image;
      img.alt = hand.label + ' ' + layout.label + ' probe placement';
      var regions = Object.keys(layout.regions).map(function (k) {
        return '<li><span class="layout__region">' + k + '</span>' + layout.regions[k] + '</li>';
      }).join('');
      caption.innerHTML =
        '<p class="layout__count"><strong>' + layout.count.toLocaleString() + '</strong> probes on the ' +
        hand.label + ' hand</p>' +
        '<ul class="layout__regions">' + regions + '</ul>' +
        '<p class="layout__note">' + layout.geometry + (layout.note ? ' ' + layout.note : '') + '</p>';
    }

    rebuild();
  }).catch(function () {
    host.innerHTML = '<p class="chart__error">The layout browser could not load.</p>';
  });
})();

/* Sensor clips: fetch and loop only while on screen. */
(function () {
  'use strict';
  var clips = document.querySelectorAll('.card__video, .demo-video[loop]');
  if (!clips.length || !('IntersectionObserver' in window)) return;
  var seen = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      var video = entry.target;
      if (entry.isIntersecting) {
        if (video.preload === 'none') video.preload = 'auto';
        var playing = video.play();
        if (playing && playing.catch) playing.catch(function () {});
      } else {
        video.pause();
      }
    });
  }, { rootMargin: '200px' });
  Array.prototype.forEach.call(clips, function (v) { seen.observe(v); });
})();
