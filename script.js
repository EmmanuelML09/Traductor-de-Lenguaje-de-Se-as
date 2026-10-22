/* =====================================================================
   script.js
   ---------------------------------------------------------------------
   Orquesta la interfaz completa: navegación, gestión de proyectos,
   flujo de entrenamiento por etapas (señas estáticas, dinámicas,
   secuencias y expresiones compuestas), reconocimiento en tiempo real,
   construcción de texto y evaluación.

   Se apoya en:
     - database.js    -> persistencia (IndexedDB) + modelo de "patterns"
     - camera.js       -> control de la webcam
     - recognition.js  -> detección de manos + KNN (estático) + DTW (dinámico)
   ===================================================================== */

import * as DB from "./database.js";
import { CameraController, CameraError } from "./camera.js";
import {
  initHandLandmarker,
  getModelState,
  detectHandsInVideo,
  extractFeatureVector,
  extractFrameFeatures,
  classifyStatic,
  classifyDynamic,
  buildStaticTrainingSet,
  buildDynamicTrainingSet
} from "./recognition.js";

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17]
];

const MIN_STATIC_EXAMPLES = 5;
const MIN_DYNAMIC_EXAMPLES = 5;
const MIN_SEQUENCE_UNITS = 2;
const STABLE_FRAMES_DEFAULT = 6;
const RECENT_LABELS_WINDOW = 10;

/* ---------------------------------------------------------------------
   Estado global en memoria
   --------------------------------------------------------------------- */
const state = {
  currentProject: null,
  currentPatternName: null,
  trainStep: 1,

  lastHandsTrain: null,       // último resultado de detección (etapa Entrenar)
  lastHandsTranslator: null,  // último resultado de detección (Traducir)

  isRecordingTrain: false,
  recordingTrain: null,       // { startedAt, frames:[{t,vector,rawHands}], videoW, videoH }
  pendingRecording: null,     // grabación aceptada pendiente de revisar

  isRecordingTranslate: false,
  recordingTranslate: null,

  sequenceDraftUnits: [],     // unidades elegidas en el constructor de secuencias (etapa 3)

  translatorMode: "static",   // "static" | "dynamic"
  recognitionPaused: false,
  stableLabel: null,
  stableCount: 0,
  lastConfirmedLabel: null,
  sequenceBuffer: [],         // unidades confirmadas, en orden (la "secuencia de texto")
  recentConfirmedLabels: [],  // ventana reciente para detectar secuencias/expresiones compuestas
  history: []
};

const cameras = {
  translator: new CameraController(document.getElementById("videoTranslator")),
  train: new CameraController(document.getElementById("videoTrain")),
  demo: new CameraController(document.getElementById("videoDemo"))
};
const loops = { translator: null, train: null, demo: null };

/* =====================================================================
   UTILIDADES DE INTERFAZ
   ===================================================================== */

function toast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  const el = document.createElement("div");
  el.className = `toast ${type === "error" ? "toast--error" : type === "success" ? "toast--success" : ""}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

function confirmDialog(message) {
  return new Promise((resolve) => {
    const modal = document.getElementById("confirmModal");
    document.getElementById("confirmModalText").textContent = message;
    modal.classList.remove("hidden");
    const acceptBtn = document.getElementById("confirmModalAccept");
    const cancelBtn = document.getElementById("confirmModalCancel");
    const cleanup = (result) => {
      modal.classList.add("hidden");
      acceptBtn.removeEventListener("click", onAccept);
      cancelBtn.removeEventListener("click", onCancel);
      resolve(result);
    };
    const onAccept = () => cleanup(true);
    const onCancel = () => cleanup(false);
    acceptBtn.addEventListener("click", onAccept);
    cancelBtn.addEventListener("click", onCancel);
  });
}

function escapeHTML(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function formatDate(isoString) {
  try {
    return new Date(isoString).toLocaleDateString("es-CO", { year: "numeric", month: "2-digit", day: "2-digit" });
  } catch {
    return isoString;
  }
}

function showView(viewName) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("view--active"));
  const target = document.getElementById(`view-${viewName}`);
  if (target) target.classList.add("view--active");

  document.querySelectorAll(".navlink").forEach((btn) => {
    if (btn.dataset.view === viewName) btn.setAttribute("aria-current", "page");
    else btn.removeAttribute("aria-current");
  });

  // Por privacidad y para no acumular bucles de reconocimiento activos,
  // se apaga la cámara de cualquier vista que se abandona.
  if (viewName !== "translator") stopViewCamera("translator");
  if (viewName !== "train") stopViewCamera("train");

  if (viewName === "projects") renderProjectsGrid();
  if (viewName === "signs") renderPatternsDatabase();
  if (viewName === "stats") { renderStats(); renderSettingsPanel(); }
  if (viewName === "train") renderStepUI();

  updateProjectWarnings();
  document.querySelector(".topbar__nav")?.classList.remove("open");
}

document.querySelectorAll("[data-view]").forEach((el) => {
  el.addEventListener("click", () => showView(el.dataset.view));
});
document.getElementById("burgerBtn").addEventListener("click", () => {
  document.querySelector(".topbar__nav").classList.toggle("open");
});

function updateProjectWarnings() {
  const hasProject = !!state.currentProject;
  ["translator", "train", "signs", "stats"].forEach((view) => {
    const banner = document.getElementById(`noProjectWarning-${view}`);
    if (banner) banner.classList.toggle("hidden", hasProject);
  });
  document.getElementById("trainWrapper").classList.toggle("hidden", !hasProject);
}

/* =====================================================================
   GESTIÓN DE PROYECTOS
   ===================================================================== */

function refreshProjectPill() {
  document.getElementById("currentProjectLabel").textContent =
    state.currentProject ? `Proyecto: ${state.currentProject.name}` : "Sin proyecto abierto";
}

async function renderProjectsGrid() {
  const grid = document.getElementById("projectsGrid");
  grid.innerHTML = "Cargando…";
  let projects;
  try {
    projects = await DB.getAllProjects();
  } catch (err) {
    grid.innerHTML = "";
    toast(err.message, "error");
    return;
  }

  if (projects.length === 0) {
    grid.innerHTML = `<p class="projects-empty">Todavía no has creado ningún proyecto. Escribe un nombre arriba y presiona "Crear proyecto".</p>`;
    return;
  }

  grid.innerHTML = "";
  for (const project of projects) {
    const stats = DB.computeProjectStats(project);
    const migratedNote = project._migratedFromLegacySigns
      ? `<p class="meta" style="color:var(--accent-text);">Migrado automáticamente desde una versión anterior del proyecto.</p>` : "";
    const card = document.createElement("div");
    card.className = "project-card";
    card.innerHTML = `
      <h4>${escapeHTML(project.name)}</h4>
      <p class="meta">${stats.totalPatterns} patrón(es) · ${stats.trainedCount} entrenado(s) · ${stats.totalExamples} ejemplo(s)</p>
      <p class="meta">Estática: ${stats.byType.static||0} · Dinámica: ${stats.byType.dynamic||0} · Secuencia: ${stats.byType.sequence||0} · Compuesta: ${stats.byType.composite||0}</p>
      <p class="meta">Creado: ${formatDate(project.createdAt)} · Modificado: ${formatDate(project.updatedAt)}</p>
      ${migratedNote}
      <div class="card-actions">
        <button class="btn btn--primary btn--small" data-action="open">Abrir</button>
        <button class="btn btn--ghost btn--small" data-action="rename">Renombrar</button>
        <button class="btn btn--ghost btn--small" data-action="export">Exportar</button>
        <button class="btn btn--danger btn--small" data-action="delete">Eliminar</button>
      </div>
    `;
    card.querySelector('[data-action="open"]').addEventListener("click", () => openProject(project.id));
    card.querySelector('[data-action="rename"]').addEventListener("click", () => renameProjectFlow(project));
    card.querySelector('[data-action="export"]').addEventListener("click", () => exportProjectFlow(project));
    card.querySelector('[data-action="delete"]').addEventListener("click", () => deleteProjectFlow(project));
    grid.appendChild(card);
  }
}

document.getElementById("btnCreateProject").addEventListener("click", async () => {
  const input = document.getElementById("newProjectNameInput");
  const name = input.value.trim();
  if (!name) { toast("Escribe un nombre para el proyecto.", "error"); return; }
  try {
    const project = await DB.createProject(name);
    input.value = "";
    toast(`Proyecto "${project.name}" creado.`, "success");
    await renderProjectsGrid();
    openProject(project.id);
  } catch (err) {
    toast(err.message, "error");
  }
});

async function openProject(id) {
  try {
    const project = await DB.getProject(id);
    if (!project) { toast("No se encontró el proyecto.", "error"); return; }
    state.currentProject = project;
    state.currentPatternName = null;
    state.trainStep = 1;
    state.sequenceBuffer = [];
    state.recentConfirmedLabels = [];
    state.history = [];
    refreshProjectPill();
    updateProjectWarnings();
    renderHistory();
    renderComposedText();
    if (project._migratedFromLegacySigns) {
      toast(`Proyecto "${project.name}" abierto. Se migraron automáticamente sus señas antiguas al nuevo formato.`, "success");
    } else {
      toast(`Proyecto "${project.name}" abierto.`, "success");
    }
  } catch (err) {
    toast(err.message, "error");
  }
}

async function renameProjectFlow(project) {
  const newName = prompt("Nuevo nombre del proyecto:", project.name);
  if (!newName || !newName.trim()) return;
  await DB.renameProject(project.id, newName.trim());
  if (state.currentProject && state.currentProject.id === project.id) {
    state.currentProject.name = newName.trim();
    refreshProjectPill();
  }
  renderProjectsGrid();
}

function exportProjectFlow(project) {
  DB.exportProjectToFile(project, false);
  toast("Proyecto exportado como archivo JSON (incluye los landmarks y secuencias necesarios para reconocer; sin miniaturas de imagen, para mantenerlo liviano).", "success");
}

async function deleteProjectFlow(project) {
  const ok = await confirmDialog(`¿Eliminar el proyecto "${project.name}"? Esta acción no se puede deshacer.`);
  if (!ok) return;
  await DB.deleteProject(project.id);
  if (state.currentProject && state.currentProject.id === project.id) {
    state.currentProject = null;
    refreshProjectPill();
    updateProjectWarnings();
  }
  toast("Proyecto eliminado.", "success");
  renderProjectsGrid();
}

document.getElementById("importProjectInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const project = await DB.importProjectFromFile(file);
    toast(`Proyecto "${project.name}" importado correctamente.`, "success");
    renderProjectsGrid();
  } catch (err) {
    toast(err.message, "error");
  } finally {
    e.target.value = "";
  }
});

async function saveCurrentProject() {
  if (!state.currentProject) return;
  await DB.putProject(state.currentProject);
}

/* =====================================================================
   CÁMARA (encendido/apagado/cambio) — compartido entre vistas
   ===================================================================== */

function wireCameraButtons(viewKey, prefix) {
  const btnOn = document.getElementById(`btnCamOn${prefix}`);
  const btnOff = document.getElementById(`btnCamOff${prefix}`);
  const btnSwitch = document.getElementById(`btnCamSwitch${prefix}`);
  const placeholder = document.getElementById(`cameraPlaceholder-${viewKey}`);
  const statusEl = document.getElementById(`handStatus${prefix}`);

  btnOn?.addEventListener("click", async () => {
    try {
      btnOn.disabled = true;
      await ensureModelLoaded(statusEl);
      await cameras[viewKey].start();
      placeholder?.classList.add("hidden");
      btnOff.disabled = false;
      btnSwitch.disabled = cameras[viewKey].devices.length < 2;
      startRecognitionLoop(viewKey);
      onCameraStateChanged(viewKey);
    } catch (err) {
      handleCameraError(err, statusEl);
      btnOn.disabled = false;
    }
  });

  btnOff?.addEventListener("click", () => stopViewCamera(viewKey));

  btnSwitch?.addEventListener("click", async () => {
    try {
      await cameras[viewKey].switchCamera();
      toast("Cámara cambiada.", "success");
    } catch (err) {
      toast(err.message, "error");
    }
  });
}

function stopViewCamera(viewKey) {
  cameras[viewKey].stop();
  if (loops[viewKey]) { cancelAnimationFrame(loops[viewKey]); loops[viewKey] = null; }
  const prefixMap = { translator: "Translator", train: "Train", demo: "Demo" };
  const prefix = prefixMap[viewKey];
  const btnOn = document.getElementById(`btnCamOn${prefix}`);
  const btnOff = document.getElementById(`btnCamOff${prefix}`);
  const btnSwitch = document.getElementById(`btnCamSwitch${prefix}`);
  const placeholder = document.getElementById(`cameraPlaceholder-${viewKey}`);
  if (btnOn) btnOn.disabled = false;
  if (btnOff) btnOff.disabled = true;
  if (btnSwitch) btnSwitch.disabled = true;
  placeholder?.classList.remove("hidden");
  clearOverlay(viewKey);
  onCameraStateChanged(viewKey);
}

function onCameraStateChanged(viewKey) {
  if (viewKey === "train") {
    document.getElementById("btnCaptureExample").disabled = !cameras.train.isActive || !getCurrentPattern();
    document.getElementById("btnRecordDynamic").disabled = !cameras.train.isActive || !getCurrentPattern();
  }
  if (viewKey === "translator") {
    document.getElementById("btnPauseRecognition").disabled = !cameras.translator.isActive;
    document.getElementById("btnRecordDynamicTranslate").disabled = !cameras.translator.isActive;
  }
}

async function ensureModelLoaded(statusEl) {
  if (getModelState() === "ready") return;
  if (statusEl) statusEl.textContent = "Cargando modelo de detección de manos (requiere internet la primera vez)…";
  await initHandLandmarker();
}

function handleCameraError(err, statusEl) {
  const message = err instanceof CameraError || err instanceof Error ? err.message : "Error desconocido con la cámara.";
  toast(message, "error");
  if (statusEl) statusEl.textContent = message;
}

wireCameraButtons("translator", "Translator");
wireCameraButtons("train", "Train");

/* ---------------------------------------------------------------------
   Dibujo del esqueleto de la(s) mano(s) sobre el canvas superpuesto
   --------------------------------------------------------------------- */
function getOverlayCanvas(viewKey) {
  const map = { translator: "overlayTranslator", train: "overlayTrain", demo: "overlayDemo" };
  return document.getElementById(map[viewKey]);
}
function clearOverlay(viewKey) {
  const canvas = getOverlayCanvas(viewKey);
  if (!canvas) return;
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
}
function drawHandsOnCanvas(canvas, handsLandmarks, videoW, videoH) {
  canvas.width = videoW;
  canvas.height = videoH;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!handsLandmarks) return;
  handsLandmarks.forEach((landmarks) => {
    ctx.strokeStyle = "#2fae5c";
    ctx.lineWidth = 2;
    HAND_CONNECTIONS.forEach(([a, b]) => {
      const p1 = landmarks[a], p2 = landmarks[b];
      ctx.beginPath();
      ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
      ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
      ctx.stroke();
    });
    ctx.fillStyle = "#f2c230";
    landmarks.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x * canvas.width, p.y * canvas.height, 3.4, 0, Math.PI * 2);
      ctx.fill();
    });
  });
}
function drawHandSkeleton(viewKey, handsLandmarks, video) {
  const canvas = getOverlayCanvas(viewKey);
  if (!canvas || !video.videoWidth) return;
  drawHandsOnCanvas(canvas, handsLandmarks, video.videoWidth, video.videoHeight);
}

/* =====================================================================
   BUCLE DE DETECCIÓN (compartido por Entrenar / Traducir / Demo)
   ===================================================================== */

function startRecognitionLoop(viewKey) {
  const video = cameras[viewKey].video;

  const step = () => {
    if (!cameras[viewKey].isActive) return;
    const now = performance.now();
    const result = detectHandsInVideo(video, now);
    const hands = result && result.landmarks && result.landmarks.length > 0 ? result.landmarks : null;

    drawHandSkeleton(viewKey, hands, video);

    if (viewKey === "train") {
      state.lastHandsTrain = hands;
      const statusEl = document.getElementById("handStatusTrain");
      if (statusEl) statusEl.textContent = hands ? `✅ ${hands.length} mano(s) detectada(s).` : "🖐️ No se detecta ninguna mano frente a la cámara.";
      document.getElementById("btnCaptureExample").disabled = !hands || !getCurrentPattern();
      document.getElementById("btnRecordDynamic").disabled = !getCurrentPattern();
      if (state.isRecordingTrain) collectRecordingFrame("train", hands, video, now);
      if (state.isRecordingVerify) collectRecordingFrame("verify", hands, video, now);
    }

    if (viewKey === "translator") {
      state.lastHandsTranslator = hands;
      const statusEl = document.getElementById("handStatusTranslator");
      if (statusEl) statusEl.textContent = hands ? `✅ ${hands.length} mano(s) detectada(s).` : "🖐️ No se detecta ninguna mano frente a la cámara.";
      if (state.isRecordingTranslate) {
        collectRecordingFrame("translator", hands, video, now);
      } else if (state.translatorMode === "static" && !state.recognitionPaused) {
        runStaticClassification(hands, "translator");
      }
    }

    if (viewKey === "demo") {
      runStaticClassification(hands, "demo");
    }

    loops[viewKey] = requestAnimationFrame(step);
  };
  loops[viewKey] = requestAnimationFrame(step);
}

/* =====================================================================
   PATRONES: creación, selección y utilidades
   ===================================================================== */

function getCurrentPattern() {
  if (!state.currentProject || !state.currentPatternName) return null;
  return DB.findPattern(state.currentProject, state.currentPatternName);
}

/* =====================================================================
   ETAPA 1 — Crear / continuar categoría
   ===================================================================== */

document.getElementById("btnStartPattern").addEventListener("click", () => {
  if (!state.currentProject) { toast("Abre o crea un proyecto primero.", "error"); return; }
  const name = document.getElementById("patternNameInput").value.trim();
  const type = document.getElementById("patternTypeSelect").value;
  const description = document.getElementById("patternDescInput").value;
  const source = document.getElementById("patternSourceInput").value;
  if (!name) { toast("Escribe el nombre o etiqueta de la unidad.", "error"); return; }

  const pattern = DB.createOrGetPattern(state.currentProject, { name, type, description, source });
  state.currentPatternName = pattern.name;
  state.sequenceDraftUnits = pattern.sequenceUnits ? pattern.sequenceUnits.slice() : [];
  saveCurrentProject();
  toast(`Categoría "${pattern.name}" (${DB.PATTERN_TYPES[pattern.type].label}) lista.`, "success");

  // Las secuencias/expresiones compuestas no necesitan cámara: se saltan directo a la etapa 3.
  state.trainStep = pattern.type === "sequence" || pattern.type === "composite" ? 3 : 2;
  renderStepUI();
});

/* =====================================================================
   STEPPER: navegación entre etapas
   ===================================================================== */

document.querySelectorAll(".stepper__dot").forEach((dot) => {
  dot.addEventListener("click", () => {
    let n = Number(dot.dataset.step);
    if (n > 1 && !state.currentPatternName) { toast("Primero crea o selecciona una categoría (Etapa 1).", "error"); return; }
    const pattern = getCurrentPattern();
    if (n === 2 && pattern && (pattern.type === "sequence" || pattern.type === "composite")) n = 3;
    state.trainStep = n;
    renderStepUI();
  });
});

document.getElementById("btnStepBack").addEventListener("click", () => {
  let n = state.trainStep - 1;
  const pattern = getCurrentPattern();
  if (n === 2 && pattern && (pattern.type === "sequence" || pattern.type === "composite")) n = 1;
  state.trainStep = Math.max(1, n);
  renderStepUI();
});

document.getElementById("btnStepNext").addEventListener("click", () => {
  const pattern = getCurrentPattern();
  const step = state.trainStep;

  if (step === 1) {
    if (!pattern) { toast("Primero crea o selecciona una categoría.", "error"); return; }
  }
  if (step === 2) {
    if (!cameras.train.isActive) { toast("Enciende la cámara antes de continuar (o vuelve a esta etapa luego).", "error"); return; }
  }
  if (step === 3) {
    if (!pattern) return;
    if (pattern.type === "static" || pattern.type === "dynamic") {
      if (pattern.examples.length === 0) { toast("Captura al menos un ejemplo antes de continuar.", "error"); return; }
    } else {
      if (pattern.sequenceUnits.length < MIN_SEQUENCE_UNITS) { toast(`Agrega al menos ${MIN_SEQUENCE_UNITS} unidades a la secuencia.`, "error"); return; }
    }
  }
  if (step === 5) {
    if (!pattern || !pattern.trained) { toast("Entrena el patrón antes de continuar.", "error"); return; }
  }

  let n = step + 1;
  if (n === 2 && pattern && (pattern.type === "sequence" || pattern.type === "composite")) n = 3;
  state.trainStep = Math.min(7, n);
  renderStepUI();
});

document.getElementById("btnFinishTraining").addEventListener("click", () => {
  showView("signs");
});

/* ---------------------------------------------------------------------
   Render del contenido de cada etapa
   --------------------------------------------------------------------- */

function renderStepUI() {
  const pattern = getCurrentPattern();
  const step = state.trainStep;

  document.querySelectorAll(".stepper__dot").forEach((dot) => {
    const n = Number(dot.dataset.step);
    dot.classList.toggle("stepper__dot--active", n === step);
    dot.classList.toggle("stepper__dot--done", n < step);
  });
  document.querySelectorAll(".step-panel").forEach((panel) => {
    panel.classList.toggle("step-panel--active", Number(panel.dataset.stepPanel) === step);
  });
  document.getElementById("btnStepBack").disabled = step === 1;
  document.getElementById("btnStepNext").classList.toggle("hidden", step === 7);

  if (pattern) {
    document.getElementById("currentPatternLabel").textContent = pattern.name;
    document.getElementById("currentPatternTypeTag").textContent = DB.PATTERN_TYPES[pattern.type].label;
  } else {
    document.getElementById("currentPatternLabel").textContent = "ninguna";
    document.getElementById("currentPatternTypeTag").textContent = "";
  }

  if (step === 3) renderCaptureStage(pattern);
  if (step === 4) renderReviewStage(pattern);
  if (step === 5) renderTrainStage(pattern);
  if (step === 6) renderVerifyStage(pattern);
  if (step === 7) renderSaveStage(pattern);
}

function renderCaptureStage(pattern) {
  const isStatic = pattern && pattern.type === "static";
  const isDynamic = pattern && pattern.type === "dynamic";
  const isSequenceLike = pattern && (pattern.type === "sequence" || pattern.type === "composite");

  document.getElementById("staticCaptureBlock").classList.toggle("hidden", !isStatic);
  document.getElementById("dynamicCaptureBlock").classList.toggle("hidden", !isDynamic);
  document.getElementById("sequenceBuilderBlock").classList.toggle("hidden", !isSequenceLike);
  document.getElementById("examplesReviewInline").classList.toggle("hidden", isSequenceLike);

  if (isStatic || isDynamic) {
    document.getElementById("btnCaptureExample").disabled = !cameras.train.isActive || !state.lastHandsTrain;
    document.getElementById("btnRecordDynamic").disabled = !cameras.train.isActive || state.isRecordingTrain;
    renderExamplesGrid(pattern);
  }
  if (isSequenceLike) {
    renderUnitPicker(pattern);
    renderSequenceBuilderList();
  }
}

function renderExamplesGrid(pattern) {
  const grid = document.getElementById("examplesGrid");
  const countLabel = document.getElementById("exampleCountLabel");
  if (!pattern) { grid.innerHTML = ""; countLabel.textContent = "Ejemplos registrados: 0"; return; }

  countLabel.textContent = `Ejemplos registrados: ${pattern.examples.length}`;
  if (pattern.examples.length === 0) {
    grid.innerHTML = `<p class="examples-empty">Todavía no hay ejemplos capturados.</p>`;
    return;
  }
  grid.innerHTML = "";
  pattern.examples.forEach((ex) => {
    const div = document.createElement("div");
    div.className = "example-thumb";
    const tag = pattern.type === "dynamic" ? `${ex.frameCount || (ex.frames ? ex.frames.length : 0)} fot. · ${((ex.durationMs||0)/1000).toFixed(1)}s` : "";
    div.innerHTML = `
      ${ex.thumbnail ? `<img src="${ex.thumbnail}" alt="Ejemplo de ${pattern.name}">` : `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-muted);font-size:0.7rem;">Sin imagen</div>`}
      ${tag ? `<span class="example-thumb__tag">${tag}</span>` : ""}
      <button class="example-thumb__del" title="Eliminar ejemplo" aria-label="Eliminar ejemplo">✕</button>
    `;
    div.querySelector(".example-thumb__del").addEventListener("click", () => {
      DB.removeExample(pattern, ex.id);
      saveCurrentProject();
      renderCaptureStage(pattern);
    });
    grid.appendChild(div);
  });
}

function renderUnitPicker(pattern) {
  const container = document.getElementById("unitPickerList");
  const candidates = state.currentProject.patterns.filter(
    (p) => p.name !== pattern.name && (p.type === "static" || p.type === "dynamic") && p.trained
  );
  if (candidates.length === 0) {
    container.innerHTML = `<p class="examples-empty">Todavía no hay señas estáticas o dinámicas entrenadas para componer una secuencia. Entrena al menos dos primero.</p>`;
    return;
  }
  container.innerHTML = candidates.map((p) => `
    <label><input type="checkbox" value="${escapeHTML(p.name)}"> ${escapeHTML(p.name)} <span class="type-tag">${DB.PATTERN_TYPES[p.type].label}</span></label>
  `).join("");
}

document.getElementById("btnAddUnitToSequence").addEventListener("click", () => {
  const checked = Array.from(document.querySelectorAll("#unitPickerList input:checked")).map((i) => i.value);
  if (checked.length === 0) { toast("Selecciona al menos una unidad.", "error"); return; }
  state.sequenceDraftUnits.push(...checked);
  document.querySelectorAll("#unitPickerList input:checked").forEach((i) => (i.checked = false));
  renderSequenceBuilderList();
});

function renderSequenceBuilderList() {
  const list = document.getElementById("sequenceBuilderList");
  const pattern = getCurrentPattern();
  if (state.sequenceDraftUnits.length === 0) {
    list.innerHTML = `<li style="list-style:none; color:var(--text-muted); font-family:var(--font-body);">Aún no has agregado unidades.</li>`;
  } else {
    list.innerHTML = state.sequenceDraftUnits.map((name, idx) => `
      <li>
        ${escapeHTML(name)}
        <button data-action="up" data-idx="${idx}" title="Mover arriba">↑</button>
        <button data-action="down" data-idx="${idx}" title="Mover abajo">↓</button>
        <button data-action="remove" data-idx="${idx}" title="Quitar">✕</button>
      </li>
    `).join("");
    list.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number(btn.dataset.idx);
        const action = btn.dataset.action;
        if (action === "remove") state.sequenceDraftUnits.splice(idx, 1);
        if (action === "up" && idx > 0) {
          [state.sequenceDraftUnits[idx - 1], state.sequenceDraftUnits[idx]] = [state.sequenceDraftUnits[idx], state.sequenceDraftUnits[idx - 1]];
        }
        if (action === "down" && idx < state.sequenceDraftUnits.length - 1) {
          [state.sequenceDraftUnits[idx + 1], state.sequenceDraftUnits[idx]] = [state.sequenceDraftUnits[idx], state.sequenceDraftUnits[idx + 1]];
        }
        renderSequenceBuilderList();
      });
    });
  }
  if (pattern && (pattern.type === "sequence" || pattern.type === "composite")) {
    DB.setSequenceUnits(pattern, state.sequenceDraftUnits);
    saveCurrentProject();
  }
}

/* ---------------------------------------------------------------------
   Captura ESTÁTICA
   --------------------------------------------------------------------- */
document.getElementById("btnCaptureExample").addEventListener("click", () => {
  const pattern = getCurrentPattern();
  if (!pattern) return;
  const hands = state.lastHandsTrain;
  if (!hands) { toast("No se detecta ninguna mano frente a la cámara en este momento.", "error"); return; }
  const vector = extractFeatureVector(hands[0]);
  const thumbnail = cameras.train.captureThumbnail();
  DB.addStaticExample(pattern, vector, thumbnail);
  saveCurrentProject();
  renderCaptureStage(pattern);
  toast(`Ejemplo capturado (${pattern.examples.length} en total).`, "success");
});

/* ---------------------------------------------------------------------
   Captura DINÁMICA (grabación temporal)
   --------------------------------------------------------------------- */
const recordDurationInput = document.getElementById("recordDurationInput");
recordDurationInput.addEventListener("input", () => {
  document.getElementById("recordDurationLabel").textContent = `${recordDurationInput.value} s`;
});

function recordingStateFor(target) {
  if (target === "train") return { flag: "isRecordingTrain", rec: "recordingTrain" };
  if (target === "verify") return { flag: "isRecordingVerify", rec: "recordingVerify" };
  return { flag: "isRecordingTranslate", rec: "recordingTranslate" };
}

function collectRecordingFrame(target, hands, video, now) {
  const { rec: recKey } = recordingStateFor(target);
  const rec = state[recKey];
  if (!rec) return;
  if (!hands) return; // no se registra el fotograma si no hay mano visible en ese instante
  rec.frames.push({ t: now - rec.startedAt, vector: extractFrameFeatures(hands), rawHands: hands });
  rec.videoW = video.videoWidth;
  rec.videoH = video.videoHeight;

  const elapsed = (now - rec.startedAt) / 1000;
  const duration = rec.duration;
  const countdownId = target === "train" ? "recordingCountdown" : target === "verify" ? "recordingCountdownVerify" : "recordingCountdownTranslator";
  const countdownEl = document.getElementById(countdownId);
  if (countdownEl) countdownEl.textContent = `Grabando… ${Math.max(0, duration - elapsed).toFixed(1)}s restantes`;
  if (elapsed >= duration) stopRecording(target);
}

function startRecording(target, duration) {
  const now = performance.now();
  const { flag, rec: recKey } = recordingStateFor(target);
  state[flag] = true;
  state[recKey] = { startedAt: now, frames: [], duration, videoW: 0, videoH: 0 };

  if (target === "train") {
    document.getElementById("recordingIndicator").classList.remove("hidden");
    document.getElementById("recordingPreview").classList.add("hidden");
    document.getElementById("btnRecordDynamic").disabled = true;
  } else if (target === "verify") {
    document.getElementById("recordingIndicatorVerify").classList.remove("hidden");
    document.getElementById("btnVerifyRecordDynamic").disabled = true;
  } else {
    document.getElementById("recordingIndicatorTranslator").classList.remove("hidden");
    document.getElementById("btnRecordDynamicTranslate").disabled = true;
  }
}

function stopRecording(target) {
  const { flag, rec: recKey } = recordingStateFor(target);
  state[flag] = false;
  const rec = state[recKey];

  if (target === "train") {
    document.getElementById("recordingIndicator").classList.add("hidden");
    document.getElementById("btnRecordDynamic").disabled = false;
    if (!rec || rec.frames.length < 3) {
      toast("No se detectó suficiente movimiento de la mano durante la grabación. Inténtalo de nuevo.", "error");
      return;
    }
    state.pendingRecording = rec;
    document.getElementById("recordingFrameCount").textContent = rec.frames.length;
    document.getElementById("recordingPreview").classList.remove("hidden");
  } else if (target === "verify") {
    document.getElementById("recordingIndicatorVerify").classList.add("hidden");
    document.getElementById("btnVerifyRecordDynamic").disabled = !cameras.train.isActive;
    if (!rec || rec.frames.length < 3) {
      toast("No se detectó suficiente movimiento durante la grabación de prueba.", "error");
      return;
    }
    classifyAndLogDynamicVerification(rec);
  } else {
    document.getElementById("recordingIndicatorTranslator").classList.add("hidden");
    document.getElementById("btnRecordDynamicTranslate").disabled = !cameras.translator.isActive;
    if (!rec || rec.frames.length < 3) {
      toast("No se detectó suficiente movimiento durante la grabación.", "error");
      return;
    }
    classifyAndConfirmDynamic(rec);
  }
}

document.getElementById("btnRecordDynamic").addEventListener("click", () => {
  const pattern = getCurrentPattern();
  if (!pattern) return;
  if (!cameras.train.isActive) { toast("Enciende la cámara primero.", "error"); return; }
  startRecording("train", Number(recordDurationInput.value));
});

document.getElementById("btnVerifyRecordDynamic").addEventListener("click", () => {
  const pattern = getCurrentPattern();
  if (!pattern) return;
  if (!cameras.train.isActive) { toast("Enciende la cámara primero (vuelve a la Etapa 2 si es necesario).", "error"); return; }
  startRecording("verify", state.currentProject?.settings?.recordSeconds || 2.5);
});

/** Clasifica una grabación de prueba (dinámica) y registra el resultado en el historial de evaluación, SIN agregarla al entrenamiento. */
function classifyAndLogDynamicVerification(rec) {
  const pattern = getCurrentPattern();
  if (!pattern || !state.currentProject) return;
  const trainingSet = buildDynamicTrainingSet(state.currentProject);
  const sequence = rec.frames.map((f) => f.vector);
  const result = trainingSet.length > 0 ? classifyDynamic(sequence, trainingSet, state.currentProject.settings.dynamicK) : null;
  const threshold = state.currentProject.settings.similarityThreshold;
  const predicted = result && result.similarity >= threshold ? result.label : null;

  showVerifyResult(predicted || "SEÑA NO RECONOCIDA", result ? `Puntuación de similitud: ${result.similarity}%` : "Sin patrones dinámicos entrenados todavía.");
  DB.addTestResult(state.currentProject, { expectedLabel: pattern.name, predictedLabel: predicted, mode: "dynamic", score: result ? result.similarity : null });
  saveCurrentProject();
}

document.getElementById("btnDiscardRecording").addEventListener("click", () => {
  state.pendingRecording = null;
  document.getElementById("recordingPreview").classList.add("hidden");
});

document.getElementById("btnAcceptRecording").addEventListener("click", () => {
  const pattern = getCurrentPattern();
  const rec = state.pendingRecording;
  if (!pattern || !rec) return;
  const frames = rec.frames.map((f) => ({ t: f.t, vector: f.vector })); // se descartan los landmarks crudos: solo se guardan características
  const lastHands = rec.frames[rec.frames.length - 1].rawHands;
  const thumbnail = cameras.train.captureThumbnail();
  DB.addDynamicExample(pattern, frames, thumbnail);
  saveCurrentProject();
  state.pendingRecording = null;
  document.getElementById("recordingPreview").classList.add("hidden");
  renderCaptureStage(pattern);
  toast(`Secuencia agregada (${pattern.examples.length} en total).`, "success");
});

document.getElementById("btnReplayRecording").addEventListener("click", () => {
  const rec = state.pendingRecording;
  if (!rec || rec.frames.length === 0) return;
  const canvas = getOverlayCanvas("train");
  let i = 0;
  function playFrame() {
    if (i >= rec.frames.length) return;
    drawHandsOnCanvas(canvas, rec.frames[i].rawHands, rec.videoW, rec.videoH);
    const next = rec.frames[i + 1];
    const delay = next ? Math.max(16, next.t - rec.frames[i].t) : 0;
    i++;
    if (next) setTimeout(playFrame, delay);
  }
  playFrame();
});

/* =====================================================================
   ETAPA 4 — Revisar
   ===================================================================== */
function renderReviewStage(pattern) {
  const box = document.getElementById("reviewSummary");
  if (!pattern) { box.innerHTML = ""; return; }
  if (pattern.type === "static" || pattern.type === "dynamic") {
    box.innerHTML = `
      <div class="review-summary__row"><span>Nombre</span><strong>${escapeHTML(pattern.name)}</strong></div>
      <div class="review-summary__row"><span>Tipo</span><strong>${DB.PATTERN_TYPES[pattern.type].label}</strong></div>
      <div class="review-summary__row"><span>Ejemplos capturados</span><strong>${pattern.examples.length}</strong></div>
      <div class="review-summary__row"><span>Descripción</span><strong>${escapeHTML(pattern.description || "—")}</strong></div>
      <div class="review-summary__row"><span>Fuente de verificación</span><strong>${escapeHTML(pattern.source || "—")}</strong></div>
    `;
  } else {
    box.innerHTML = `
      <div class="review-summary__row"><span>Nombre</span><strong>${escapeHTML(pattern.name)}</strong></div>
      <div class="review-summary__row"><span>Tipo</span><strong>${DB.PATTERN_TYPES[pattern.type].label}</strong></div>
      <div class="review-summary__row"><span>Unidades en orden</span><strong>${pattern.sequenceUnits.map(escapeHTML).join(" → ") || "—"}</strong></div>
      <div class="review-summary__row"><span>Fuente de verificación</span><strong>${escapeHTML(pattern.source || "—")}</strong></div>
    `;
  }
}

/* =====================================================================
   ETAPA 5 — Entrenar
   ===================================================================== */
function renderTrainStage(pattern) {
  const hint = document.getElementById("trainRequirementHint");
  const btn = document.getElementById("btnTrainPattern");
  if (!pattern) { hint.textContent = ""; btn.disabled = true; return; }

  if (pattern.type === "static") {
    hint.textContent = `Se necesitan al menos ${MIN_STATIC_EXAMPLES} ejemplos (tienes ${pattern.examples.length}).`;
    btn.disabled = pattern.examples.length < MIN_STATIC_EXAMPLES;
  } else if (pattern.type === "dynamic") {
    hint.textContent = `Se necesitan al menos ${MIN_DYNAMIC_EXAMPLES} secuencias grabadas (tienes ${pattern.examples.length}).`;
    btn.disabled = pattern.examples.length < MIN_DYNAMIC_EXAMPLES;
  } else {
    hint.textContent = `Se necesitan al menos ${MIN_SEQUENCE_UNITS} unidades en la secuencia (tienes ${pattern.sequenceUnits.length}).`;
    btn.disabled = pattern.sequenceUnits.length < MIN_SEQUENCE_UNITS;
  }
  btn.textContent = pattern.trained ? "Volver a entrenar" : "Entrenar patrón";
}

document.getElementById("btnTrainPattern").addEventListener("click", async () => {
  const pattern = getCurrentPattern();
  if (!pattern) return;
  DB.markPatternTrained(pattern);
  await saveCurrentProject();
  toast(`Patrón "${pattern.name}" entrenado correctamente.`, "success");
  renderStepUI();
});

/* =====================================================================
   ETAPA 6 — Verificar
   ===================================================================== */
let lastVerification = null;

function renderVerifyStage(pattern) {
  const isSeqLike = pattern && (pattern.type === "sequence" || pattern.type === "composite");
  const isStatic = pattern && pattern.type === "static";
  const isDynamic = pattern && pattern.type === "dynamic";
  document.getElementById("verifyStaticDynamicBlock").classList.toggle("hidden", !pattern || isSeqLike);
  document.getElementById("verifySequenceBlock").classList.toggle("hidden", !isSeqLike);
  document.getElementById("verifyResultBox").classList.add("hidden");
  document.getElementById("btnVerifyPattern").classList.toggle("hidden", !isStatic);
  document.getElementById("btnVerifyRecordDynamic").classList.toggle("hidden", !isDynamic);
  document.getElementById("btnVerifyPattern").disabled = !cameras.train.isActive;
  document.getElementById("btnVerifyRecordDynamic").disabled = !cameras.train.isActive;
}

document.getElementById("btnVerifyPattern").addEventListener("click", async () => {
  const pattern = getCurrentPattern();
  if (!pattern || pattern.type !== "static") return;

  const hands = state.lastHandsTrain;
  if (!hands) { toast("No se detecta ninguna mano en este momento.", "error"); return; }
  const vector = extractFeatureVector(hands[0]);
  const trainingSet = buildStaticTrainingSet(state.currentProject);
  const result = classifyStatic(vector, trainingSet, state.currentProject.settings.k);
  const threshold = state.currentProject.settings.confidenceThreshold;
  const predicted = result && result.confidence >= threshold ? result.label : null;
  lastVerification = { mode: "static", expectedLabel: pattern.name, predictedLabel: predicted, score: result ? result.confidence : null };
  showVerifyResult(predicted || "SEÑA NO RECONOCIDA", result ? `Confianza: ${result.confidence}%` : "");
  DB.addTestResult(state.currentProject, lastVerification);
  await saveCurrentProject();
});

function showVerifyResult(label, scoreText) {
  document.getElementById("verifyResultBox").classList.remove("hidden");
  document.getElementById("verifyResultText").textContent = label;
  document.getElementById("verifyResultScore").textContent = scoreText;
}

document.getElementById("btnVerifyCorrect").addEventListener("click", async () => {
  if (!state.currentProject || state.currentProject.testHistory.length === 0) return;
  state.currentProject.testHistory[state.currentProject.testHistory.length - 1].correct = true;
  state.currentProject.testHistory[state.currentProject.testHistory.length - 1].correctedByUser = true;
  await saveCurrentProject();
  toast("Resultado marcado como correcto.", "success");
});
document.getElementById("btnVerifyIncorrect").addEventListener("click", async () => {
  if (!state.currentProject || state.currentProject.testHistory.length === 0) return;
  state.currentProject.testHistory[state.currentProject.testHistory.length - 1].correct = false;
  state.currentProject.testHistory[state.currentProject.testHistory.length - 1].correctedByUser = true;
  await saveCurrentProject();
  toast("Resultado marcado como incorrecto.", "success");
});

/* =====================================================================
   ETAPA 7 — Guardar
   ===================================================================== */
function renderSaveStage(pattern) {
  // El guardado ya ocurre automáticamente en cada cambio; esta etapa solo confirma visualmente.
}

/* =====================================================================
   TRADUCIR — reconocimiento en tiempo real y construcción de texto
   ===================================================================== */

document.getElementById("btnModeStatic").addEventListener("click", () => setTranslatorMode("static"));
document.getElementById("btnModeDynamic").addEventListener("click", () => setTranslatorMode("dynamic"));

function setTranslatorMode(mode) {
  state.translatorMode = mode;
  document.getElementById("btnModeStatic").classList.toggle("mode-toggle__btn--active", mode === "static");
  document.getElementById("btnModeDynamic").classList.toggle("mode-toggle__btn--active", mode === "dynamic");
  document.getElementById("dynamicRecordBlockTranslator").classList.toggle("hidden", mode !== "dynamic");
}

document.getElementById("btnPauseRecognition").addEventListener("click", (e) => {
  state.recognitionPaused = !state.recognitionPaused;
  e.target.textContent = state.recognitionPaused ? "▶ Reanudar reconocimiento" : "⏸ Pausar reconocimiento";
});

document.getElementById("btnRecordDynamicTranslate").addEventListener("click", () => {
  if (!cameras.translator.isActive) return;
  const duration = state.currentProject?.settings?.recordSeconds || 2.5;
  startRecording("translator", duration);
});

function classifyAndConfirmDynamic(rec) {
  if (!state.currentProject) return;
  const trainingSet = buildDynamicTrainingSet(state.currentProject);
  if (trainingSet.length === 0) {
    toast("No existen patrones dinámicos entrenados en este proyecto.", "error");
    return;
  }
  const sequence = rec.frames.map((f) => f.vector);
  const result = classifyDynamic(sequence, trainingSet, state.currentProject.settings.dynamicK);
  const threshold = state.currentProject.settings.similarityThreshold;

  if (!result || result.similarity < threshold) {
    setResultDisplay("DINÁMICA", "SEÑA NO RECONOCIDA", result ? `Puntuación de similitud: ${result.similarity}%` : "Puntuación de similitud: —", result ? result.similarity : 0);
    return;
  }
  setResultDisplay("DINÁMICA", result.label, `Puntuación de similitud: ${result.similarity}%`, result.similarity);
  if (state.currentProject.settings.autoAddEnabled) {
    confirmUnit(result.label, result.similarity, "dynamic");
  } else {
    state.lastPendingUnit = { label: result.label, score: result.similarity, type: "dynamic" };
    toast("Seña reconocida. Presiona \"Confirmar unidad\" para agregarla al texto.", "info");
  }
}

function runStaticClassification(hands, target) {
  if (!state.currentProject) {
    setResultDisplay("", "—", "Abre un proyecto para traducir", 0);
    return;
  }
  const trainingSet = buildStaticTrainingSet(state.currentProject);
  if (trainingSet.length === 0) {
    setResultDisplay("", "—", "No existen patrones estáticos entrenados en este proyecto", 0);
    return;
  }
  if (!hands) {
    setResultDisplay("", "—", "Confianza: —", 0);
    state.stableLabel = null; state.stableCount = 0;
    return;
  }

  const vector = extractFeatureVector(hands[0]);
  const k = state.currentProject.settings.k;
  const classification = classifyStatic(vector, trainingSet, k);
  const threshold = state.currentProject.settings.confidenceThreshold;

  if (!classification || classification.confidence < threshold) {
    setResultDisplay("ESTÁTICA", "SEÑA NO RECONOCIDA", classification ? `Confianza: ${classification.confidence}%` : "Confianza: —", classification ? classification.confidence : 0);
    state.stableLabel = null; state.stableCount = 0;
    return;
  }

  setResultDisplay("ESTÁTICA", classification.label, `Confianza: ${classification.confidence}%`, classification.confidence);
  state.lastPendingUnit = { label: classification.label, score: classification.confidence, type: "static" };

  const stableFrames = state.currentProject.settings.stableFrames || STABLE_FRAMES_DEFAULT;
  if (state.stableLabel === classification.label) state.stableCount++;
  else { state.stableLabel = classification.label; state.stableCount = 1; }

  if (state.stableCount === stableFrames && state.lastConfirmedLabel !== classification.label) {
    if (target === "translator" && state.currentProject.settings.autoAddEnabled) {
      confirmUnit(classification.label, classification.confidence, "static");
    } else if (target === "demo") {
      confirmUnit(classification.label, classification.confidence, "static", true);
    }
    state.lastConfirmedLabel = classification.label;
  }
}

function setResultDisplay(typeLabel, label, scoreText, scoreValue) {
  const isDemo = document.getElementById("demoOverlay") && !document.getElementById("demoOverlay").classList.contains("hidden");
  if (isDemo) {
    document.getElementById("demoResultText").textContent = label;
    document.getElementById("demoConfidenceText").textContent = scoreText;
    return;
  }
  document.getElementById("resultTypeTag").textContent = typeLabel;
  document.getElementById("resultText").textContent = label;
  document.getElementById("confidenceText").textContent = scoreText;
  document.getElementById("confidenceFill").style.width = `${Math.max(0, Math.min(100, scoreValue))}%`;
}

/* ---------------------------------------------------------------------
   Construcción de la secuencia de texto
   --------------------------------------------------------------------- */

function confirmUnit(label, score, type, skipStats) {
  state.sequenceBuffer.push(label);
  state.history.unshift({ label, score, type, at: new Date() });
  state.history = state.history.slice(0, 30);

  state.recentConfirmedLabels.push(label);
  if (state.recentConfirmedLabels.length > RECENT_LABELS_WINDOW) state.recentConfirmedLabels.shift();

  renderHistory();
  renderComposedText();
  checkSequenceMatches();

  if (!skipStats && state.currentProject) {
    state.currentProject.stats.recognitions = (state.currentProject.stats.recognitions || 0) + 1;
    state.currentProject.stats.confidenceSum = (state.currentProject.stats.confidenceSum || 0) + score;
    saveCurrentProject();
  }
}

document.getElementById("btnConfirmUnit").addEventListener("click", () => {
  if (!state.lastPendingUnit) { toast("No hay ninguna unidad detectada para confirmar todavía.", "error"); return; }
  confirmUnit(state.lastPendingUnit.label, state.lastPendingUnit.score, state.lastPendingUnit.type);
  state.lastPendingUnit = null;
});

document.getElementById("btnUndoUnit").addEventListener("click", () => {
  if (state.sequenceBuffer.length === 0) return;
  state.sequenceBuffer.pop();
  renderComposedText();
  toast("Se deshizo la última unidad.", "success");
});

document.getElementById("btnClearText").addEventListener("click", () => {
  state.sequenceBuffer = [];
  state.recentConfirmedLabels = [];
  state.lastConfirmedLabel = null;
  renderComposedText();
  document.getElementById("compositeMatchBanner").classList.add("hidden");
});

document.getElementById("btnCopyText").addEventListener("click", async () => {
  const text = state.sequenceBuffer.join(" ");
  if (!text) { toast("No hay texto para copiar todavía.", "error"); return; }
  try {
    await navigator.clipboard.writeText(text);
    toast("Texto copiado al portapapeles.", "success");
  } catch {
    toast("No se pudo copiar automáticamente. Selecciona el texto manualmente.", "error");
  }
});

document.getElementById("toggleAutoAdd").addEventListener("change", (e) => {
  if (state.currentProject) {
    state.currentProject.settings.autoAddEnabled = e.target.checked;
    saveCurrentProject();
  }
});

function renderComposedText() {
  const el = document.getElementById("composedTextDisplay");
  el.textContent = state.sequenceBuffer.length > 0 ? state.sequenceBuffer.join(" ") : "—";
}

function renderHistory() {
  const list = document.getElementById("historyList");
  if (state.history.length === 0) {
    list.innerHTML = `<li class="history-empty">Aún no hay reconocimientos.</li>`;
    return;
  }
  list.innerHTML = state.history.map((h) => {
    const scoreLabel = h.type === "dynamic" ? `similitud ${h.score}%` : `confianza ${h.score}%`;
    return `<li>${escapeHTML(h.label)} <span style="color:var(--text-muted); font-family:var(--font-body);">— ${scoreLabel} · ${DB.PATTERN_TYPES[h.type]?.label || h.type}</span></li>`;
  }).join("");
}

document.getElementById("btnClearHistory").addEventListener("click", () => {
  state.history = [];
  renderHistory();
});

/**
 * Comprueba si el final de la ventana reciente de unidades confirmadas
 * coincide EXACTAMENTE, en el mismo orden, con las unidades de alguna
 * secuencia o expresión compuesta entrenada. Si hay coincidencia, se
 * muestra un aviso adicional; las unidades individuales ya mostradas en
 * el texto NO se reemplazan, para no ocultar lo que realmente se
 * reconoció (secuencia visual vs. etiqueta compuesta).
 */
function checkSequenceMatches() {
  if (!state.currentProject) return;
  const banner = document.getElementById("compositeMatchBanner");
  const candidates = state.currentProject.patterns.filter(
    (p) => (p.type === "sequence" || p.type === "composite") && p.trained && p.sequenceUnits.length > 0
  );
  for (const pattern of candidates) {
    const n = pattern.sequenceUnits.length;
    const tail = state.recentConfirmedLabels.slice(-n);
    if (tail.length === n && tail.every((label, i) => label === pattern.sequenceUnits[i])) {
      banner.classList.remove("hidden");
      banner.textContent = `🔗 Secuencia reconocida: "${pattern.name}" (compuesta por: ${pattern.sequenceUnits.join(" → ")}). Esto es una concatenación de etiquetas entrenadas, no una traducción lingüística validada.`;
      return;
    }
  }
}

/* =====================================================================
   PATRONES ENTRENADOS (vista "Patrones")
   ===================================================================== */

function renderPatternsDatabase() {
  const container = document.getElementById("signsList");
  if (!state.currentProject) { container.innerHTML = ""; return; }
  const patterns = state.currentProject.patterns;
  if (patterns.length === 0) {
    container.innerHTML = `<p class="projects-empty">Este proyecto todavía no tiene patrones. Ve a "Entrenar" para registrar el primero.</p>`;
    return;
  }
  container.innerHTML = "";
  patterns.forEach((pattern) => {
    const row = document.createElement("div");
    row.className = "sign-row";
    const metaText = pattern.type === "static" || pattern.type === "dynamic"
      ? `${pattern.examples.length} ejemplo(s)`
      : `${pattern.sequenceUnits.length} unidad(es): ${pattern.sequenceUnits.join(" → ") || "—"}`;
    row.innerHTML = `
      <div>
        <p class="sign-row__name">${escapeHTML(pattern.name)} <span class="type-tag">${DB.PATTERN_TYPES[pattern.type].label}</span></p>
        <p class="sign-row__meta">${metaText} ·
          <span class="${pattern.trained ? "sign-row__status--trained" : "sign-row__status--pending"}">${pattern.trained ? "Entrenada ✓" : "Sin entrenar"}</span>
        </p>
        ${pattern.description ? `<p class="sign-row__meta">${escapeHTML(pattern.description)}</p>` : ""}
        ${pattern.source ? `<p class="sign-row__meta">Fuente: ${escapeHTML(pattern.source)}</p>` : ""}
      </div>
      <div class="sign-row__actions">
        <button class="btn btn--ghost btn--small" data-action="edit">Abrir en Entrenar</button>
        <button class="btn btn--danger btn--small" data-action="delete">Eliminar</button>
      </div>
    `;
    row.querySelector('[data-action="edit"]').addEventListener("click", () => {
      state.currentPatternName = pattern.name;
      state.sequenceDraftUnits = pattern.sequenceUnits ? pattern.sequenceUnits.slice() : [];
      state.trainStep = 3;
      showView("train");
    });
    row.querySelector('[data-action="delete"]').addEventListener("click", async () => {
      const ok = await confirmDialog(`¿Eliminar el patrón "${pattern.name}"? Esta acción no se puede deshacer.`);
      if (!ok) return;
      DB.removePattern(state.currentProject, pattern.name);
      await saveCurrentProject();
      renderPatternsDatabase();
      toast("Patrón eliminado.", "success");
    });
    container.appendChild(row);
  });
}

/* =====================================================================
   ESTADÍSTICAS
   ===================================================================== */

function renderStats() {
  const setVal = (id, value) => (document.getElementById(id).textContent = value ?? "N/D");
  if (!state.currentProject) {
    ["statStatic","statDynamic","statSequences","statTrained","statExamples","statRecognitions","statAccuracy","statEvalAccuracy"].forEach((id) => setVal(id, "N/D"));
    return;
  }
  const stats = DB.computeProjectStats(state.currentProject);
  setVal("statStatic", stats.byType.static || 0);
  setVal("statDynamic", stats.byType.dynamic || 0);
  setVal("statSequences", (stats.byType.sequence || 0) + (stats.byType.composite || 0));
  setVal("statTrained", stats.trainedCount);
  setVal("statExamples", stats.totalExamples);
  setVal("statRecognitions", stats.recognitions);
  setVal("statAccuracy", stats.avgConfidence !== null ? `${stats.avgConfidence}%` : "N/D");
  setVal("statEvalAccuracy", stats.evaluation ? `${stats.evaluation.accuracy}% (${stats.evaluation.total} pruebas)` : "N/D");
}

/* =====================================================================
   CONFIGURACIÓN DEL RECONOCEDOR (umbrales ajustables)
   ===================================================================== */

const settingsControls = [
  { id: "settingConfidenceThreshold", key: "confidenceThreshold", suffix: "%" },
  { id: "settingSimilarityThreshold", key: "similarityThreshold", suffix: "%" },
  { id: "settingK", key: "k", suffix: "" },
  { id: "settingStableFrames", key: "stableFrames", suffix: "" }
];

function renderSettingsPanel() {
  const hasProject = !!state.currentProject;
  document.getElementById("noProjectWarning-settings").classList.toggle("hidden", hasProject);
  document.getElementById("settingsPanel").classList.toggle("hidden", !hasProject);
  if (!hasProject) return;

  settingsControls.forEach(({ id, key, suffix }) => {
    const input = document.getElementById(id);
    const label = document.getElementById(`${id}Label`);
    const value = state.currentProject.settings[key];
    input.value = value;
    label.textContent = `${value}${suffix}`;
  });
}

settingsControls.forEach(({ id, key, suffix }) => {
  document.getElementById(id).addEventListener("input", (e) => {
    document.getElementById(`${id}Label`).textContent = `${e.target.value}${suffix}`;
  });
  document.getElementById(id).addEventListener("change", (e) => {
    if (!state.currentProject) return;
    state.currentProject.settings[key] = Number(e.target.value);
    saveCurrentProject();
    toast("Configuración del reconocedor actualizada para este proyecto.", "success");
  });
});

/* =====================================================================
   MODO DEMOSTRACIÓN
   ===================================================================== */

document.getElementById("btnDemoMode").addEventListener("click", async () => {
  if (!state.currentProject) { toast("Abre un proyecto con señas entrenadas antes de usar el modo demostración.", "error"); return; }
  document.getElementById("demoOverlay").classList.remove("hidden");
  try {
    await ensureModelLoaded(null);
    await cameras.demo.start();
    startRecognitionLoop("demo");
  } catch (err) {
    toast(err.message, "error");
  }
});
document.getElementById("btnExitDemo").addEventListener("click", () => {
  stopViewCamera("demo");
  document.getElementById("demoOverlay").classList.add("hidden");
});

/* =====================================================================
   INICIALIZACIÓN
   ===================================================================== */

(async function init() {
  refreshProjectPill();
  updateProjectWarnings();
  renderHistory();
  renderComposedText();
  setTranslatorMode("static");
  initHandLandmarker().catch((err) => console.warn(err));
})();
