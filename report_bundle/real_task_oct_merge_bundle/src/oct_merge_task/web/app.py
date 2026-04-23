from __future__ import annotations

import json
from pathlib import Path


def write_web_app(output_dir: str | Path, payload: dict | None = None) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "index.html"
    html_path.write_text(_html_template(payload), encoding="utf-8")
    return {"html_path": str(html_path)}


def _html_template(payload: dict | None = None) -> str:
    embedded_payload = json.dumps(payload, ensure_ascii=False) if payload is not None else "null"
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OCT Merge Task Preview</title>
  <style>
    body {
      margin: 0;
      background: radial-gradient(circle at top, #10223d 0%, #050913 50%, #02040a 100%);
      color: #d9f4ff;
      font-family: "Segoe UI", sans-serif;
      overflow: hidden;
    }
    .layout {
      display: grid;
      grid-template-columns: 2fr 1fr;
      height: 100vh;
    }
    .hero {
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .panel {
      padding: 24px;
      background: rgba(6, 12, 24, 0.85);
      border-left: 1px solid rgba(120, 220, 255, 0.18);
      overflow: auto;
      backdrop-filter: blur(12px);
    }
    h1 { margin-top: 0; font-size: 28px; }
    .card {
      margin-bottom: 18px;
      padding: 14px 16px;
      border: 1px solid rgba(120, 220, 255, 0.18);
      border-radius: 14px;
      background: rgba(13, 26, 44, 0.6);
    }
    .controls {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }
    .legend-swatch {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      display: inline-block;
    }
    input[type="range"] {
      width: 160px;
    }
    .slice {
      width: 100%;
      border-radius: 10px;
      image-rendering: pixelated;
      border: 1px solid rgba(120, 220, 255, 0.2);
    }
  </style>
</head>
<body>
  <div class="layout">
    <div class="hero">
      <canvas id="scene"></canvas>
    </div>
    <div class="panel">
      <h1>OCT Merge Task Preview</h1>
      <div class="controls">
        <label>View
          <select id="viewMode">
            <option value="stitched">Stitched</option>
            <option value="volume_a">Volume A</option>
            <option value="volume_b">Volume B</option>
          </select>
        </label>
        <label>Slice
          <select id="sliceAxis">
            <option value="axial_mid">Axial</option>
          </select>
        </label>
        <label>Zoom <input id="zoom" type="range" min="40" max="240" value="100"></label>
        <label>Point Size <input id="pointSize" type="range" min="1" max="8" value="2"></label>
      </div>
      <div class="card"><strong>Algorithm Output</strong><div id="summary"></div></div>
      <div class="card"><strong>Structure Legend</strong><div id="legend"></div></div>
      <div class="card"><strong>Slice View</strong><canvas id="slice" class="slice"></canvas></div>
      <div class="card"><strong>Notes</strong><div id="notes">Drag to rotate. Mouse wheel or zoom slider to zoom. Point cloud is driven by the stitched algorithm output.</div></div>
    </div>
  </div>
  <script>
    const summary = document.getElementById('summary');
    const legend = document.getElementById('legend');
    const notes = document.getElementById('notes');
    const scene = document.getElementById('scene');
    const sceneCtx = scene.getContext('2d');
    const sliceCanvas = document.getElementById('slice');
    const sliceCtx = sliceCanvas.getContext('2d');
    const zoomInput = document.getElementById('zoom');
    const pointSizeInput = document.getElementById('pointSize');
    const viewMode = document.getElementById('viewMode');
    const sliceAxis = document.getElementById('sliceAxis');
    let payload = __EMBEDDED_PAYLOAD__;
    let rotY = 0.7;
    let rotX = -0.4;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    function resize() {
      scene.width = scene.clientWidth;
      scene.height = scene.clientHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    function applyPayload(data) {
      payload = data;
      updateLegend();
      updateNotes();
      summary.innerHTML = `
        <div>Preview shape: ${data.preview_shape.join(' x ')}</div>
        <div>Stitched shape: ${data.stitched_shape.join(' x ')}</div>
        <div>Transform tx: ${data.transform.tx}</div>
        <div>Brick count: ${data.brick_count}</div>
        <div>Estimated FPS: ${data.benchmark.estimated_fps.toFixed(2)}</div>
        <div>Point count: ${currentPointCloud().count}</div>
      `;
      drawSlice(data.slices[sliceAxis.value]);
      requestAnimationFrame(loop);
    }

    if (payload) {
      applyPayload(payload);
    } else {
      fetch('./payload.json')
        .then(r => r.json())
        .then(applyPayload)
        .catch(err => {
          summary.innerHTML = '<div>Failed to load payload. Open this page through the build script or local server.</div>';
          console.error(err);
        });
    }

    function drawSlice(slice) {
      const [h, w] = slice.shape;
      sliceCanvas.width = w;
      sliceCanvas.height = h;
      const imageData = sliceCtx.createImageData(w, h);
      for (let i = 0; i < slice.data.length; i++) {
        const value = slice.data[i];
        const idx = i * 4;
        imageData.data[idx] = value;
        imageData.data[idx + 1] = value;
        imageData.data[idx + 2] = Math.min(255, value + 40);
        imageData.data[idx + 3] = 255;
      }
      sliceCtx.putImageData(imageData, 0, 0);
    }

    function currentPointCloud() {
      return payload.point_clouds[viewMode.value];
    }

    function updateLegend() {
      if (!payload) return;
      const cloud = currentPointCloud();
      legend.innerHTML = cloud.legend.map(item => `
        <div class="legend-item">
          <span class="legend-swatch" style="background:${item.color}"></span>
          <span>${item.label}</span>
        </div>
      `).join('');
    }

    function updateNotes() {
      const mode = viewMode.value;
      if (mode === 'volume_a') {
        notes.textContent = 'Volume A: reference acquisition volume. Colors distinguish layered background, reflective blocks, tubular structures, spherical structures, and slanted structures.';
      } else if (mode === 'volume_b') {
        notes.textContent = 'Volume B: second acquisition volume. The same structural classes are shown, but after overlap cropping and distortion.';
      } else {
        notes.textContent = 'Stitched: merged output. Cyan means A-only region, orange means B-only region, and yellow means overlap/fused region.';
      }
    }

    function rotatePoint(point) {
      const [x, y, z] = point;
      const cloud = currentPointCloud();
      const cx = cloud.shape[0] / 2;
      const cy = cloud.shape[1] / 2;
      const cz = cloud.shape[2] / 2;
      let px = x - cx;
      let py = y - cy;
      let pz = z - cz;

      const cosy = Math.cos(rotY), siny = Math.sin(rotY);
      const cosx = Math.cos(rotX), sinx = Math.sin(rotX);
      let x1 = px * cosy - pz * siny;
      let z1 = px * siny + pz * cosy;
      let y1 = py * cosx - z1 * sinx;
      let z2 = py * sinx + z1 * cosx;
      return [x1, y1, z2];
    }

    function projectPoint(rotated, zoom) {
      const cloud = currentPointCloud();
      const maxDim = Math.max(cloud.shape[0], cloud.shape[1], cloud.shape[2]);
      const perspective = 1.0 / (1.0 + rotated[2] / (maxDim * 1.4));
      const sx = scene.width / 2 + rotated[0] * zoom * perspective * 6.0;
      const sy = scene.height / 2 + rotated[1] * zoom * perspective * 6.0;
      const scale = perspective;
      return [sx, sy, scale];
    }

    function loop() {
      sceneCtx.clearRect(0, 0, scene.width, scene.height);
      const cx = scene.width / 2;
      const cy = scene.height / 2;
      if (payload) {
        const zoom = Number(zoomInput.value) / 100;
        const baseSize = Number(pointSizeInput.value);
        const cloud = currentPointCloud();
        const points = cloud.points;
        const values = cloud.values;
        const labels = cloud.labels || [];
        for (let i = 0; i < points.length; i++) {
          const rotated = rotatePoint(points[i]);
          const [sx, sy, scale] = projectPoint(rotated, zoom);
          const size = Math.max(0.5, baseSize * scale * 2.0);
          const intensity = values[i];
          const alpha = Math.min(0.95, 0.15 + intensity * 0.85);
          const color = colorForLabel(labels[i], cloud.color || '#7ef7c5');
          const r = parseInt(color.slice(1,3), 16);
          const g = parseInt(color.slice(3,5), 16);
          const b = parseInt(color.slice(5,7), 16);
          sceneCtx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
          sceneCtx.beginPath();
          sceneCtx.arc(sx, sy, size, 0, Math.PI * 2);
          sceneCtx.fill();
        }
      } else {
        sceneCtx.fillStyle = 'rgba(90,220,255,0.15)';
        sceneCtx.beginPath();
        sceneCtx.arc(cx, cy, 20, 0, Math.PI * 2);
        sceneCtx.fill();
      }
      requestAnimationFrame(loop);
    }

    scene.addEventListener('mousedown', (e) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    });
    window.addEventListener('mouseup', () => dragging = false);
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      rotY += dx * 0.01;
      rotX += dy * 0.01;
      lastX = e.clientX;
      lastY = e.clientY;
    });
    scene.addEventListener('wheel', (e) => {
      e.preventDefault();
      const current = Number(zoomInput.value);
      zoomInput.value = Math.max(40, Math.min(240, current - e.deltaY * 0.05));
    }, { passive: false });
    viewMode.addEventListener('change', () => {
      updateLegend();
      updateNotes();
      if (payload) {
        summary.innerHTML = `
          <div>Preview shape: ${payload.preview_shape.join(' x ')}</div>
          <div>Stitched shape: ${payload.stitched_shape.join(' x ')}</div>
          <div>Transform tx: ${payload.transform.tx}</div>
          <div>Brick count: ${payload.brick_count}</div>
          <div>Estimated FPS: ${payload.benchmark.estimated_fps.toFixed(2)}</div>
          <div>Point count: ${currentPointCloud().count}</div>
        `;
      }
    });
    sliceAxis.addEventListener('change', () => {
      if (payload) drawSlice(payload.slices[sliceAxis.value]);
    });

    function colorForLabel(label, fallback) {
      const colors = {
        'Layered Background': '#2a6f97',
        'Reflective Block': '#51d6ff',
        'Tubular Structure': '#ff6ad5',
        'Spherical Structure': '#ffe66d',
        'Slanted Band': '#7ef7c5',
        'A-only': '#51d6ff',
        'B-only': '#ff9a4d',
        'Overlap': '#f6ff7e',
      };
      return colors[label] || fallback;
    }
  </script>
</body>
</html>
"""
    return template.replace("__EMBEDDED_PAYLOAD__", embedded_payload)
