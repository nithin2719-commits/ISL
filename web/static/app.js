// ISL Translator web client — the browser version of src/main.py.
// MediaPipe Hands runs on the raw camera frame (the model was trained on unmirrored photos),
// the preview is mirrored with CSS only, the first hand's 21 (x, y) landmarks go to /predict,
// and a sign held for 15 frames is typed.

const MP = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const HAND_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const FRAME_THRESHOLD = 15;
const CONNS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

const $ = (id) => document.getElementById(id);
const view = $('view'), ctx = view.getContext('2d'), video = $('video');
const S = { sentence: '', last: '', count: 0, inflight: false, running: false, vision: null, video: null, image: null };

// ------------------------------------------------------------------ model / UI
async function init() {
  try {
    const h = await (await fetch('/health')).json();
    $('pill-model').textContent = `${h.model} · ${h.classes.length} signs`;
    $('keys').innerHTML = h.classes.map((c) => `<span data-k="${c}">${c}</span>`).join('');
  } catch { $('pill-model').textContent = 'server unreachable'; }
  render(null);
}

async function predict(features) {
  const r = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ landmarks: features }) });
  return r.ok ? r.json() : null;
}

function render(res) {
  const label = res ? res.label : '';
  $('letter').textContent = label || '–';
  $('conf').textContent = res ? `${Math.round(res.confidence * 100)}% confidence` : 'waiting for a hand';
  $('detect').textContent = `Detecting: ${label || '—'}`;
  $('top3').innerHTML = res ? res.top.map((t) => `<li><b>${t.label}</b><span class="bar"><i style="width:${t.confidence * 100}%"></i></span><span class="p mono">${Math.round(t.confidence * 100)}%</span></li>`).join('') : '';
  const frac = Math.min(1, S.count / FRAME_THRESHOLD);
  $('ring-fill').style.strokeDashoffset = String(326.7 * (1 - frac));
  $('ring').classList.toggle('done', frac >= 1);
  document.querySelectorAll('#keys span').forEach((k) => k.classList.toggle('hit', k.dataset.k === label));
}

function renderSentence() {
  $('subtitle').innerHTML = S.sentence
    ? `${S.sentence.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/ /g, '&nbsp;')}<span class="caret"></span>`
    : '<span class="placeholder">Signed letters appear here</span>';
}

function speak(text) {
  if (text.trim() && 'speechSynthesis' in window) speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

// same stability rule as main.py: a prediction repeated FRAME_THRESHOLD times is typed once
function stabilise(label) {
  if (label && label === S.last) S.count += 1; else { S.count = 0; S.last = label; }
  if (S.count === FRAME_THRESHOLD) {
    if (!S.sentence.length || S.sentence.at(-1) !== label) { S.sentence += label; renderSentence(); speak(label); }
  }
}

// ------------------------------------------------------------------ MediaPipe
async function landmarker(mode) {
  if (!S.vision) {
    const vision = await import(`${MP}/vision_bundle.mjs`);
    S.vision = { lib: vision, files: await vision.FilesetResolver.forVisionTasks(`${MP}/wasm`) };
  }
  const key = mode === 'VIDEO' ? 'video' : 'image';
  if (!S[key]) {
    S[key] = await S.vision.lib.HandLandmarker.createFromOptions(S.vision.files, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: 'GPU' }, runningMode: mode, numHands: 2,
      minHandDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
    });
  }
  return S[key];
}

function drawHands(hands) {
  const w = view.width, h = view.height;
  ctx.lineCap = 'round';
  hands.forEach((pts, i) => {
    ctx.strokeStyle = i === 0 ? '#ff9933' : 'rgba(255,255,255,.45)';
    ctx.lineWidth = Math.max(2, w / 260);
    for (const [a, b] of CONNS) { ctx.beginPath(); ctx.moveTo(pts[a].x * w, pts[a].y * h); ctx.lineTo(pts[b].x * w, pts[b].y * h); ctx.stroke(); }
    for (const p of pts) { ctx.beginPath(); ctx.arc(p.x * w, p.y * h, Math.max(3, w / 180), 0, Math.PI * 2); ctx.fillStyle = i === 0 ? '#fff' : 'rgba(255,255,255,.6)'; ctx.fill(); }
  });
  $('pill-hand').classList.toggle('on', hands.length > 0);
  $('pill-hand').lastChild.textContent = hands.length ? `${hands.length} hand${hands.length > 1 ? 's' : ''}` : 'no hand';
}

const features = (pts) => pts.flatMap((p) => [p.x, p.y]); // first hand only, as capture.find_position does

async function startCamera() {
  const btn = $('btn-cam');
  btn.disabled = true; btn.textContent = 'Loading…';
  try {
    const lm = await landmarker('VIDEO');
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
    video.srcObject = stream; await video.play();
    view.width = video.videoWidth; view.height = video.videoHeight;
    $('empty').hidden = true; S.running = true;
    $('stage').classList.add('mirror');
    const loop = async () => {
      if (!S.running) return;
      ctx.drawImage(video, 0, 0);
      const res = lm.detectForVideo(view, performance.now());
      drawHands(res.landmarks || []);
      if (res.landmarks && res.landmarks.length && !S.inflight) {
        S.inflight = true;
        predict(features(res.landmarks[0])).then((p) => { stabilise(p ? p.label : ''); render(p); }).finally(() => { S.inflight = false; });
      } else if (!res.landmarks || !res.landmarks.length) { stabilise(''); render(null); }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  } catch (e) {
    btn.disabled = false; btn.textContent = 'Start camera';
    $('empty').querySelector('p').textContent = `Camera unavailable (${e.message || e}). Try a photo instead.`;
  }
}

async function tryPhoto(file) {
  const img = new Image();
  img.src = URL.createObjectURL(file);
  await img.decode();
  S.running = false;
  $('stage').classList.remove('mirror');
  view.width = img.naturalWidth; view.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);
  $('empty').hidden = true;
  const lm = await landmarker('IMAGE');
  const res = lm.detect(view);
  drawHands(res.landmarks || []);
  if (res.landmarks && res.landmarks.length) render(await predict(features(res.landmarks[0])));
  else { render(null); $('detect').textContent = 'No hand found in this photo'; }
}

// ------------------------------------------------------------------ controls
$('btn-cam').addEventListener('click', startCamera);
$('file').addEventListener('change', (e) => { if (e.target.files[0]) tryPhoto(e.target.files[0]); });
$('btn-space').addEventListener('click', () => { S.sentence += ' '; renderSentence(); });
$('btn-back').addEventListener('click', () => { S.sentence = S.sentence.slice(0, -1); renderSentence(); });
$('btn-clear').addEventListener('click', () => { S.sentence = ''; renderSentence(); });
$('btn-speak').addEventListener('click', () => speak(S.sentence));
addEventListener('keydown', (e) => {
  if (e.target.closest('input, textarea')) return;
  if (e.key === ' ') { e.preventDefault(); $('btn-space').click(); }
  else if (e.key === 'Backspace') { e.preventDefault(); $('btn-back').click(); }
  else if (e.key === 'Enter') $('btn-speak').click();
  else if (e.key.toLowerCase() === 'c') $('btn-clear').click();
});

init();
