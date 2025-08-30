/* global ort */
(() => {
  // ---------- DOM ----------
  const elUpload = document.getElementById('upload-video');
  const elProgressWrap = document.getElementById('progress-wrapper');
  const elProgress = document.getElementById('progress-bar');
  const elStatus = document.getElementById('upload-status');

  const btnStart = document.getElementById('start-processing');
  const btnPause = document.getElementById('pause-processing');
  const btnStop = document.getElementById('stop-processing');

  const videoContainer = document.getElementById('video-container');
  const video = document.getElementById('video-feed');
  const canvas = document.getElementById('detection-canvas');
  const ctx = canvas.getContext('2d');

  const infoPanel = document.getElementById('info-panel');
  const elFPS = document.getElementById('fps-counter');
  const elDet = document.getElementById('detection-counter');
  const elProc = document.getElementById('process-time');

  // Current list
  const livePanel = document.getElementById('live-panel');
  const currentListEl = document.getElementById('current-list');

  // ---------- CONFIG ----------
  const MODEL_URL = 'models/best_webgpu.onnx'; // ONNX (export ด้วย nms=False)
  const CLASSES_URL = 'models/classes.json';   // optional
  const INPUT_SIZE = 640;                      // ให้ตรงกับ export
  const SCORE_THR = 0.25;
  const IOU_THR = 0.45;
  const MAX_DETECTIONS_DRAW = 300;

  // ---------- STATE ----------
  let session = null;
  let modelBytesCache = null;
  let currentEP = null;         // 'webgpu' | 'wasm'
  let classNames = [];
  let modelInputName = null;
  let running = false;
  let paused = false;
  let haveVideo = false;
  let haveModel = false;
  let frames = 0;
  let fpsTick = performance.now();

  // ---------- ORT env ----------
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  ort.env.logLevel = 'error';

  // ---------- Helpers ----------
  function setStatus(msg, kind = '') {
    elStatus.textContent = msg;
    elStatus.classList.remove('loading', 'success', 'error');
    if (kind) elStatus.classList.add(kind);
  }
  function show(el) { el.classList.remove('hidden'); }
  function hide(el) { el.classList.add('hidden'); }

  function classNameFromId(cls) {
    return (Array.isArray(classNames) && classNames[cls] !== undefined)
      ? classNames[cls] : `cls_${cls}`;
  }

  // ดึงโมเดลพร้อม progress
  async function fetchWithProgress(url, onProgress) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
    const contentLen = +resp.headers.get('Content-Length') || 0;
    const reader = resp.body.getReader();
    let received = 0;
    const chunks = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      if (onProgress && contentLen) onProgress(Math.floor((received / contentLen) * 100));
    }
    const blob = new Blob(chunks);
    const buffer = await blob.arrayBuffer();
    return new Uint8Array(buffer);
  }

  async function loadClasses() {
    try {
      const r = await fetch(CLASSES_URL);
      if (!r.ok) return; // optional
      classNames = await r.json();
    } catch (_) {}
  }

  async function createSessionWithEP(ep) {
    if (!modelBytesCache) {
      setStatus('Loading model...', 'loading');
      show(elProgressWrap);
      modelBytesCache = await fetchWithProgress(MODEL_URL, p => { elProgress.value = p; });
      hide(elProgressWrap);
    }
    setStatus(`Initializing session (${ep.toUpperCase()})...`, 'loading');
    const s = await ort.InferenceSession.create(modelBytesCache, {
      executionProviders: [ep],
      graphOptimizationLevel: 'all'
    });
    currentEP = ep;
    return s;
  }

  async function loadModel() {
    try {
      const preferWebGPU = !!navigator.gpu;
      session = await createSessionWithEP(preferWebGPU ? 'webgpu' : 'wasm');
      modelInputName = session.inputNames[0];
      haveModel = true;
      setStatus(`Model ready (${currentEP.toUpperCase()})`, 'success');
      maybeEnableStart();
    } catch (e) {
      console.error(e);
      setStatus(`Model init failed: ${e.message}`, 'error');
      try {
        session = await createSessionWithEP('wasm');
        modelInputName = session.inputNames[0];
        haveModel = true;
        setStatus('Model ready (WASM fallback)', 'success');
        maybeEnableStart();
      } catch (e2) {
        console.error(e2);
        setStatus(`Model load error: ${e2.message}`, 'error');
      }
    }
  }

  function maybeEnableStart() {
    if (haveVideo && haveModel) {
      show(btnStart); show(btnPause); show(btnStop);
      btnStart.disabled = false;
      show(videoContainer); show(infoPanel); show(livePanel);
      ensureCanvasPixelSize();
    }
  }

  // ใช้ canvas เป็นจอแสดงผลหลัก (วาดเฟรมวิดีโอ + กรอบ)
  function useCanvasAsDisplay() {
    video.style.display = 'none';
    canvas.style.position = 'relative';
    canvas.style.display = 'block';
    canvas.style.width = '100%';
    canvas.style.height = 'auto';
    canvas.style.pointerEvents = 'none';
  }

  // ให้ขนาด pixel ของ canvas เท่ากับวิดีโอจริง
  function ensureCanvasPixelSize() {
    const vw = video.videoWidth || 0;
    const vh = video.videoHeight || 0;
    if (!vw || !vh) return;
    if (canvas.width !== vw || canvas.height !== vh) {
      canvas.width = vw;
      canvas.height = vh;
    }
  }

  // --- Offscreen สำหรับ preprocess (letterbox) ---
  const tmp = document.createElement('canvas');
  const tctx = tmp.getContext('2d');

  // เตรียมอินพุต [1,3,S,S] float32
  function preprocessToTensor() {
    const S = INPUT_SIZE;
    tmp.width = S; tmp.height = S;

    tctx.fillStyle = 'rgb(114,114,114)';
    tctx.fillRect(0, 0, S, S);

    const vw = video.videoWidth, vh = video.videoHeight;
    const scale = Math.min(S / vw, S / vh);
    const newW = Math.round(vw * scale);
    const newH = Math.round(vh * scale);
    const padX = Math.floor((S - newW) / 2);
    const padY = Math.floor((S - newH) / 2);

    tctx.drawImage(video, 0, 0, vw, vh, padX, padY, newW, newH);

    const imgData = tctx.getImageData(0, 0, S, S);
    const data = imgData.data;
    const size = S * S;
    const tensorData = new Float32Array(3 * size);

    for (let i = 0, p = 0; i < size; i++, p += 4) {
      const rv = data[p] / 255, gv = data[p + 1] / 255, bv = data[p + 2] / 255;
      tensorData[i] = rv;
      tensorData[i + size] = gv;
      tensorData[i + 2 * size] = bv;
    }
    const inputTensor = new ort.Tensor('float32', tensorData, [1, 3, S, S]);
    return { inputTensor, scale, padX, padY };
  }

  // ---------- Math & NMS ----------
  const sigmoid = x => 1 / (1 + Math.exp(-x));
  function maybeSigmoidArray(a) {
    let need = false;
    for (let k = 0; k < a.length; k++) { if (a[k] < 0 || a[k] > 1) { need = true; break; } }
    if (!need) return a;
    const out = new Float32Array(a.length);
    for (let k = 0; k < a.length; k++) out[k] = sigmoid(a[k]);
    return out;
  }
  function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
    const w = Math.max(0, x2 - x1), h = Math.max(0, y2 - y1);
    const inter = w * h;
    const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua > 0 ? inter / ua : 0;
  }
  function nmsKeep(dets, iouThr, limit = 300) {
    dets.sort((a, b) => b.score - a.score);
    const keep = [];
    const sup = new Array(dets.length).fill(false);
    for (let i = 0; i < dets.length; i++) {
      if (sup[i]) continue;
      keep.push(dets[i]);
      if (keep.length >= limit) break;
      for (let j = i + 1; j < dets.length; j++) {
        if (sup[j]) continue;
        if (dets[i].cls !== dets[j].cls) continue;
        if (iou(dets[i], dets[j]) > IOU_THR) sup[j] = true;
      }
    }
    return keep;
  }

  // ---------- Parsing (รองรับรวม/แยก) สำหรับ nms=False ----------
  function parseDetections(outputs, meta) {
    const names = session.outputNames;
    const tensors = names.map(n => outputs[n]);

    function transpose2(arr, A, B) {
      const out = new Float32Array(A * B);
      for (let i = 0; i < A; i++) for (let j = 0; j < B; j++) out[j * A + i] = arr[i * B + j];
      return out;
    }

    // แบบแยก: boxes + scores
    if (tensors.length >= 2) {
      let tBoxes = null, tScores = null;
      for (const t of tensors) {
        const d = t.dims;
        if (d.length === 3 && (d[2] === 4 || d[1] === 4)) tBoxes = t;
        else if (d.length === 3) tScores = t;
      }
      if (tBoxes && tScores) {
        let boxesArr = tBoxes.data;
        const bShape = tBoxes.dims;
        let N;
        if (bShape[0] === 1 && bShape[2] === 4) { N = bShape[1]; }
        else if (bShape[0] === 1 && bShape[1] === 4) { N = bShape[2]; boxesArr = transpose2(boxesArr, 4, N); }
        else throw new Error(`Unexpected boxes shape: ${bShape.join('x')}`);

        let scoresArr = tScores.data;
        const sShape = tScores.dims;
        let nc;
        if (sShape[0] === 1 && sShape[1] === N) nc = sShape[2];       // [1,N,nc]
        else if (sShape[0] === 1 && sShape[2] === N) {                 // [1,nc,N]
          nc = sShape[1];
          scoresArr = transpose2(scoresArr, sShape[1], sShape[2]);    // -> [N,nc]
        } else throw new Error(`Unexpected scores shape: ${sShape.join('x')}`);

        const dets = [];
        const vw = video.videoWidth, vh = video.videoHeight;
        const { scale, padX, padY } = meta;

        for (let i = 0; i < N; i++) {
          const bx = boxesArr[i * 4 + 0];
          const by = boxesArr[i * 4 + 1];
          const bw = boxesArr[i * 4 + 2];
          const bh = boxesArr[i * 4 + 3];

          let x1 = bx - bw / 2, y1 = by - bh / 2;
          let x2 = bx + bw / 2, y2 = by + bh / 2;

          x1 = (x1 - padX) / scale; y1 = (y1 - padY) / scale;
          x2 = (x2 - padX) / scale; y2 = (y2 - padY) / scale;

          const probs = maybeSigmoidArray(scoresArr.slice(i * nc, i * nc + nc));
          let best = 0, bestScore = probs[0];
          for (let c = 1; c < nc; c++) if (probs[c] > bestScore) { bestScore = probs[c]; best = c; }
          if (bestScore < SCORE_THR) continue;

          x1 = Math.max(0, Math.min(vw, x1));
          y1 = Math.max(0, Math.min(vh, y1));
          x2 = Math.max(0, Math.min(vw, x2));
          y2 = Math.max(0, Math.min(vh, y2));
          if (x2 <= x1 || y2 <= y1) continue;

          dets.push({ x1, y1, x2, y2, score: bestScore, cls: best });
        }
        return nmsKeep(dets, IOU_THR, MAX_DETECTIONS_DRAW);
      }
    }

    // แบบรวม: [1,N,no] หรือ [1,no,N] หรือ [N,no]
    const t0 = tensors[0];
    let arr = t0.data;
    const shape = t0.dims;

    let N, no;
    if (shape.length === 3 && shape[0] === 1) {
      const A = shape[1], B = shape[2];
      if (B >= 6 && B <= 4096) { N = A; no = B; }
      else { N = shape[2]; no = shape[1];
        const out = new Float32Array(N * no);
        for (let i = 0; i < no; i++) for (let j = 0; j < N; j++) out[j * no + i] = arr[i * N + j];
        arr = out;
      }
    } else if (shape.length === 2) { N = shape[0]; no = shape[1]; }
    else throw new Error(`Unexpected output shape: ${shape.join('x')}`);

    let hasObj = (no > 5);
    let nc = no - (hasObj ? 5 : 4);
    let clsStart = hasObj ? 5 : 4;
    if (classNames.length > 0) {
      if (no === 5 + classNames.length) { hasObj = true;  nc = classNames.length; clsStart = 5; }
      else if (no === 4 + classNames.length) { hasObj = false; nc = classNames.length; clsStart = 4; }
    }
    if (nc <= 0) return [];

    const dets = [];
    const vw = video.videoWidth, vh = video.videoHeight;
    const { scale, padX, padY } = meta;

    for (let i = 0; i < N; i++) {
      const base = i * no;
      const cx = arr[base + 0], cy = arr[base + 1];
      const w  = arr[base + 2], h  = arr[base + 3];

      let objConf = 1.0;
      if (hasObj) {
        objConf = arr[base + 4];
        if (objConf < 0 || objConf > 1) objConf = sigmoid(objConf);
      }

      const clsScores = maybeSigmoidArray(arr.slice(base + clsStart, base + clsStart + nc));

      let best = -1, bestScore = -1;
      for (let c = 0; c < nc; c++) {
        const sc = objConf * clsScores[c];
        if (sc > bestScore) { bestScore = sc; best = c; }
      }
      if (bestScore < SCORE_THR) continue;

      let x1 = cx - w / 2, y1 = cy - h / 2;
      let x2 = cx + w / 2, y2 = cy + h / 2;

      x1 = (x1 - padX) / scale; y1 = (y1 - padY) / scale;
      x2 = (x2 - padX) / scale; y2 = (y2 - padY) / scale;

      x1 = Math.max(0, Math.min(vw, x1));
      y1 = Math.max(0, Math.min(vh, y1));
      x2 = Math.max(0, Math.min(vw, x2));
      y2 = Math.max(0, Math.min(vh, y2));
      if (x2 <= x1 || y2 <= y1) continue;

      dets.push({ x1, y1, x2, y2, score: bestScore, cls: best });
    }

    return nmsKeep(dets, IOU_THR, MAX_DETECTIONS_DRAW);
  }

  // วาดเฟรม + กรอบ
  function drawFrameAndDetections(dets) {
    ensureCanvasPixelSize();
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.font = '14px Arial';

    for (const d of dets) {
      const { x1, y1, x2, y2, score, cls } = d;

      ctx.strokeStyle = 'red';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const label = `${classNameFromId(cls)} ${score.toFixed(2)}`;
      const tw = ctx.measureText(label).width;
      const th = 16;
      const boxY = (y1 - th < 0) ? (y1 + th) : (y1 - th);

      ctx.fillStyle = 'red';
      ctx.fillRect(x1, boxY - th, tw + 6, th);
      ctx.fillStyle = 'white';
      ctx.fillText(label, x1 + 3, boxY - 3);
    }
  }

  // อัปเดตรายการลิสต์ ณ ขณะนั้น (เรียงตาม score มาก→น้อย)
  function renderCurrentList(dets) {
    currentListEl.innerHTML = '';
    if (!dets.length) {
      const li = document.createElement('li');
      li.textContent = '— no detections —';
      currentListEl.appendChild(li);
      return;
    }
    const sorted = [...dets].sort((a, b) => b.score - a.score);
    for (const d of sorted) {
      const li = document.createElement('li');

      const left = document.createElement('div');
      left.className = 'det-left';
      const name = document.createElement('span');
      name.className = 'det-name';
      name.textContent = classNameFromId(d.cls);
      const score = document.createElement('span');
      score.className = 'det-score';
      score.textContent = `(${d.score.toFixed(2)})`;
      left.appendChild(name);
      left.appendChild(score);

      const right = document.createElement('span');
      const w = Math.max(0, d.x2 - d.x1) | 0;
      const h = Math.max(0, d.y2 - d.y1) | 0;
      right.textContent = `${w}×${h}`;

      li.appendChild(left);
      li.appendChild(right);
      currentListEl.appendChild(li);
    }
  }

  // main loop (sync กับวิดีโอเมื่อทำได้)
  async function processFrame() {
    if (!running || paused) return;

    try {
      if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
        video.requestVideoFrameCallback(async () => {
          if (!running || paused) return;
          const t0 = performance.now();

          const { inputTensor, ...meta } = preprocessToTensor();
          const outputs = await session.run({ [modelInputName]: inputTensor });
          const dets = parseDetections(outputs, meta);

          drawFrameAndDetections(dets);
          renderCurrentList(dets);

          const t1 = performance.now();
          elProc.textContent = `${Math.round(t1 - t0)} ms`;
          elDet.textContent = `${dets.length}`;

          frames++;
          const now = performance.now();
          if (now - fpsTick >= 1000) { elFPS.textContent = frames.toString(); frames = 0; fpsTick = now; }

          processFrame(); // next frame
        });
        return;
      }

      // Fallback (rAF)
      const t0 = performance.now();

      const { inputTensor, ...meta } = preprocessToTensor();
      const outputs = await session.run({ [modelInputName]: inputTensor });
      const dets = parseDetections(outputs, meta);

      drawFrameAndDetections(dets);
      renderCurrentList(dets);

      const t1 = performance.now();
      elProc.textContent = `${Math.round(t1 - t0)} ms`;
      elDet.textContent = `${dets.length}`;

      frames++;
      const now = performance.now();
      if (now - fpsTick >= 1000) { elFPS.textContent = frames.toString(); frames = 0; fpsTick = now; }

    } catch (e) {
      console.error(e);
      setStatus(`Inference error: ${e.message}`, 'error');
      running = false;
      return;
    }

    requestAnimationFrame(processFrame);
  }

  // ---------- Event wiring ----------
  elUpload.addEventListener('change', (ev) => {
    const file = ev.target.files?.[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    video.src = url;
    video.muted = true;
    video.playsInline = true;
    video.setAttribute('playsinline', '');
    video.setAttribute('webkit-playsinline', '');

    video.onloadedmetadata = () => {
      haveVideo = true;
      setStatus('Video loaded. Waiting for model...', 'success');
      useCanvasAsDisplay();
      ensureCanvasPixelSize();
      show(livePanel);
      maybeEnableStart();
    };
  });

  btnStart.addEventListener('click', async () => {
    if (!session) await loadModel();
    running = true;
    paused = false;
    setStatus(`Running detection... (${(currentEP || 'wasm').toUpperCase()})`, 'success');
    video.play();
    processFrame();
  });

  btnPause.addEventListener('click', () => {
    if (!running) return;
    paused = !paused;
    if (paused) {
      video.pause();
      btnPause.textContent = '▶ Resume';
      setStatus('Paused', 'loading');
    } else {
      video.play();
      btnPause.textContent = '⏸ Pause';
      setStatus(`Running detection... (${(currentEP || 'wasm').toUpperCase()})`, 'success');
      processFrame();
    }
  });

  btnStop.addEventListener('click', () => {
    running = false;
    paused = false;
    video.pause();
    btnPause.textContent = '⏸ Pause';
    setStatus('Stopped. Ready to start again.', 'loading');
    elFPS.textContent = '0';
    elDet.textContent = '0';
    elProc.textContent = '0ms';
    // ไม่ล้าง current list — คงผลเฟรมล่าสุดไว้
  });

  // โหลดคลาส + โมเดล (ล่วงหน้า)
  loadClasses();
  loadModel();
})();