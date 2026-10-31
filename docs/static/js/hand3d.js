/* Interactive 3D view of a hand and its probe layout.

   Geometry is baked at the rest pose by scripts/export_hand_geometry.py, so the page only has
   to draw it: a merged hand mesh plus one instanced sphere per probe, coloured by region. */
(function () {
  'use strict';

  var host = document.querySelector('#hand-3d');
  if (!host) return;

  var REGION_COLORS = {
    thumb: 0x497fdb, index: 0xf3ae54, middle: 0x57bf6a,
    ring: 0xd65974, pinky: 0xa96cd1, palm: 0x767d8f, other: 0x9aa0ab
  };
  var LAYOUT_LABEL = { low: 'Synthetic low', actual: 'Device layout', high: 'Synthetic high' };
  var HAND_LABEL = { wuji: 'Wuji', allegro: 'Allegro', sharpa: 'Sharpa' };

  function fail(message) {
    host.innerHTML = '<p class="chart__error">' + message + '</p>';
  }

  function boot(THREE) {
    var state = { hand: 'wuji', layout: 'actual', cache: {} };

    var controls = document.createElement('div');
    controls.className = 'chart__controls';
    var stage = document.createElement('div');
    stage.className = 'hand3d__stage';
    var caption = document.createElement('p');
    caption.className = 'hand3d__caption caption';
    host.appendChild(controls);
    host.appendChild(stage);
    host.appendChild(caption);

    var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    stage.appendChild(renderer.domElement);
    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(38, 1, 0.01, 10);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa0ab, 2.1));
    var key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(0.4, 0.7, 0.9);
    scene.add(key);

    var group = new THREE.Group();
    scene.add(group);
    var target = new THREE.Vector3();
    var spin = { theta: 0.9, phi: 1.15, radius: 0.36, dragging: false, lx: 0, ly: 0, auto: true };

    function resize() {
      var w = stage.clientWidth || 600;
      var h = Math.max(320, Math.round(w * 0.62));
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }

    function frame() {
      camera.position.set(
        target.x + spin.radius * Math.sin(spin.phi) * Math.sin(spin.theta),
        target.y + spin.radius * Math.cos(spin.phi),
        target.z + spin.radius * Math.sin(spin.phi) * Math.cos(spin.theta)
      );
      camera.lookAt(target);
    }

    function tick() {
      if (spin.auto && !spin.dragging) spin.theta += 0.0035;
      frame();
      renderer.render(scene, camera);
      requestAnimationFrame(tick);
    }

    stage.addEventListener('pointerdown', function (e) {
      spin.dragging = true; spin.auto = false; spin.lx = e.clientX; spin.ly = e.clientY;
      stage.setPointerCapture(e.pointerId);
    });
    stage.addEventListener('pointermove', function (e) {
      if (!spin.dragging) return;
      spin.theta -= (e.clientX - spin.lx) * 0.008;
      spin.phi = Math.min(Math.PI - 0.12, Math.max(0.12, spin.phi - (e.clientY - spin.ly) * 0.008));
      spin.lx = e.clientX; spin.ly = e.clientY;
    });
    stage.addEventListener('pointerup', function (e) {
      spin.dragging = false;
      stage.releasePointerCapture(e.pointerId);
    });
    stage.addEventListener('wheel', function (e) {
      e.preventDefault();
      spin.radius = Math.min(1.2, Math.max(0.12, spin.radius * (1 + Math.sign(e.deltaY) * 0.1)));
    }, { passive: false });

    function tabs(labels, ids, current, pick) {
      var bar = document.createElement('div');
      bar.className = 'chart__tabs';
      bar.setAttribute('role', 'tablist');
      ids.forEach(function (id, i) {
        var b = document.createElement('button');
        b.type = 'button';
        b.className = 'chart__tab';
        b.setAttribute('role', 'tab');
        b.setAttribute('aria-selected', String(id === current));
        b.textContent = labels[i];
        b.addEventListener('click', function () { pick(id); });
        bar.appendChild(b);
      });
      return bar;
    }

    function rebuildControls(hand) {
      controls.innerHTML = '';
      var ids = Object.keys(HAND_LABEL);
      controls.appendChild(tabs(ids.map(function (i) { return HAND_LABEL[i]; }), ids, state.hand,
        function (id) { state.hand = id; show(); }));
      var names = Object.keys(hand.layouts);
      controls.appendChild(tabs(
        names.map(function (n) { return (LAYOUT_LABEL[n] || n) + ' · ' + hand.layouts[n].count; }),
        names, state.layout, function (id) { state.layout = id; show(); }
      ));
      var reset = document.createElement('button');
      reset.type = 'button';
      reset.className = 'chart__toggle';
      reset.textContent = 'Spin';
      reset.addEventListener('click', function () { spin.auto = !spin.auto; });
      controls.appendChild(reset);
    }

    function build(hand) {
      while (group.children.length) group.remove(group.children[0]);

      var geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(hand.mesh.vertices, 3));
      geo.setIndex(hand.mesh.indices);
      geo.computeVertexNormals();
      group.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
        color: 0xe8eaee, roughness: 0.85, metalness: 0.05, flatShading: false
      })));

      var layout = hand.layouts[state.layout] || hand.layouts[Object.keys(hand.layouts)[0]];
      var n = layout.count;
      var sphere = new THREE.SphereGeometry(1, 8, 6);
      var probes = new THREE.InstancedMesh(
        sphere,
        new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.0 }),
        n
      );
      var m = new THREE.Matrix4();
      var color = new THREE.Color();
      for (var i = 0; i < n; i++) {
        var r = Math.max(layout.radii[i], 0.0012) * 0.95;
        m.makeScale(r, r, r);
        m.setPosition(layout.positions[3 * i], layout.positions[3 * i + 1], layout.positions[3 * i + 2]);
        probes.setMatrixAt(i, m);
        color.setHex(REGION_COLORS[layout.regions[i]] || REGION_COLORS.other);
        probes.setColorAt(i, color);
      }
      probes.instanceMatrix.needsUpdate = true;
      if (probes.instanceColor) probes.instanceColor.needsUpdate = true;
      group.add(probes);

      var box = new THREE.Box3().setFromObject(group);
      box.getCenter(target);
      spin.radius = box.getSize(new THREE.Vector3()).length() * 1.25;

      var counts = {};
      layout.regions.forEach(function (region) { counts[region] = (counts[region] || 0) + 1; });
      var keys = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; });
      caption.innerHTML =
        '<strong>' + n.toLocaleString() + '</strong> probes, drawn at their sensing radius on the ' +
        'hand at rest. Drag to orbit, scroll to zoom. ' +
        '<span class="hand3d__legend">' + keys.map(function (region) {
          var hex = '#' + (REGION_COLORS[region] || REGION_COLORS.other).toString(16).padStart(6, '0');
          return '<span class="hand3d__key"><i style="background:' + hex + '"></i>' +
                 region + ' ' + counts[region] + '</span>';
        }).join('') + '</span>';
    }

    function show() {
      var cached = state.cache[state.hand];
      if (cached) {
        if (!cached.layouts[state.layout]) state.layout = Object.keys(cached.layouts)[0];
        rebuildControls(cached);
        build(cached);
        return;
      }
      caption.textContent = 'Loading ' + HAND_LABEL[state.hand] + '…';
      fetch('static/data/hands/' + state.hand + '.json')
        .then(function (r) { return r.json(); })
        .then(function (hand) { state.cache[state.hand] = hand; show(); })
        .catch(function () { fail('The 3D view could not load its geometry.'); });
    }

    window.addEventListener('resize', resize);
    resize();
    show();
    tick();
  }

  var src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.module.min.js';
  import(src).then(boot).catch(function () {
    fail('The 3D view needs WebGL and the three.js library, which did not load. ' +
         'The rendered layouts above show the same placements.');
  });
})();
