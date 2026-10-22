/* =====================================================================
   recognition.js
   ---------------------------------------------------------------------
   Módulo responsable de:
   1. Cargar el detector de manos (MediaPipe Tasks Vision - HandLandmarker).
   2. Convertir los 21 landmarks de una mano en un vector numérico
      normalizado (igual que en la versión anterior del prototipo).
   3. Clasificar SEÑAS ESTÁTICAS con K-Vecinos-más-Cercanos (KNN),
      comparando un único vector de características.
   4. (NUEVO) Clasificar SEÑAS DINÁMICAS comparando SECUENCIAS completas
      de vectores (una por cada fotograma capturado durante la grabación),
      usando Dynamic Time Warping (DTW) para medir qué tan parecidos son
      dos movimientos en el tiempo, aunque se hayan ejecutado a distinta
      velocidad.

   ¿Por qué DTW para las señas dinámicas?
   - Comparar únicamente la foto inicial o final de un movimiento no
     distingue señas que dependen de CÓMO se mueve la mano (trayectoria,
     dirección, orden de las posiciones).
   - DTW compara dos secuencias completas y encuentra la mejor manera de
     "alinearlas" en el tiempo, por lo que tolera que una persona haga el
     mismo movimiento un poco más rápido o más lento que en el ejemplo
     grabado durante el entrenamiento. Es una técnica clásica, liviana
     (no requiere descargar un modelo adicional) y perfectamente viable
     en JavaScript puro para el tamaño de datos de un prototipo escolar.
   - Es una primera versión funcional de reconocimiento temporal: no es
     un modelo de aprendizaje profundo entrenado específicamente para
     secuencias, pero SÍ utiliza realmente la trayectoria capturada
     (no solo una fotografía) para decidir la seña más parecida.

   IMPORTANTE SOBRE LA CONFIANZA EN SEÑAS DINÁMICAS Y SECUENCIAS:
   DTW no produce una probabilidad calibrada (como sí puede hacerlo un
   clasificador probabilístico entrenado estadísticamente). Por eso, en
   vez de llamarlo "confianza", el prototipo lo llama explícitamente
   "puntuación de similitud" y lo calcula a partir de la distancia DTW
   normalizada, dejando claro que es una estimación relativa, no una
   medición certificada.
   ===================================================================== */

import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

let handLandmarker = null;
let modelState = "not-loaded"; // "not-loaded" | "loading" | "ready" | "error"
let modelError = null;

export function getModelState() {
  return modelState;
}
export function getModelError() {
  return modelError;
}

/** Descarga e inicializa el modelo de detección de manos. Se debe llamar una sola vez al iniciar la app. */
export async function initHandLandmarker() {
  if (modelState === "ready") return handLandmarker;
  modelState = "loading";
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2 // Se detectan hasta 2 manos: necesario para señas que usan ambas manos.
    });
    modelState = "ready";
    return handLandmarker;
  } catch (err) {
    modelState = "error";
    modelError =
      "No se pudo cargar el modelo de detección de manos. Verifica tu conexión a internet la primera vez que usas la aplicación. Detalle: " +
      (err && err.message ? err.message : err);
    throw new Error(modelError);
  }
}

/**
 * Detecta manos en el fotograma actual de un elemento <video>.
 * Devuelve el resultado bruto de MediaPipe o null si el modelo aún no
 * está listo o el video no tiene datos todavía.
 * result.landmarks es un arreglo con una entrada por cada mano detectada
 * (cada una con 21 puntos {x,y,z}); result.handedness indica si cada una
 * es "Left" o "Right" según el modelo.
 */
export function detectHandsInVideo(videoElement, timestampMs) {
  if (modelState !== "ready" || !handLandmarker) return null;
  if (!videoElement || videoElement.readyState < 2) return null;
  return handLandmarker.detectForVideo(videoElement, timestampMs);
}

/* ---------------------------------------------------------------------
   Extracción de características de UNA mano en UN instante
   --------------------------------------------------------------------- */

function distance3D(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}

/**
 * Convierte los 21 landmarks de una mano en un vector de 63 números,
 * normalizado respecto a la muñeca (posición) y a la distancia
 * muñeca-dedo medio (escala), para que el reconocimiento no dependa de
 * en qué parte de la imagen esté la mano ni de qué tan cerca esté de la
 * cámara.
 */
export function extractFeatureVector(landmarks) {
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const scale = distance3D(wrist, middleMcp) || 1e-6;

  const vector = [];
  for (const point of landmarks) {
    vector.push((point.x - wrist.x) / scale);
    vector.push((point.y - wrist.y) / scale);
    vector.push((point.z - wrist.z) / scale);
  }
  return vector;
}

/**
 * Extrae un "fotograma de características" combinando hasta dos manos:
 * [mano_1 (63 valores), mano_2 (63 valores)]. Si solo hay una mano, la
 * segunda se rellena con ceros relativos (no se inventa una posición).
 * Esto permite representar señas de una o de dos manos con el mismo
 * formato de vector, y conservar su relación espacial si corresponde.
 */
export function extractFrameFeatures(handsLandmarks) {
  const hand1 = handsLandmarks[0] ? extractFeatureVector(handsLandmarks[0]) : new Array(63).fill(0);
  const hand2 = handsLandmarks[1] ? extractFeatureVector(handsLandmarks[1]) : new Array(63).fill(0);
  return hand1.concat(hand2);
}

/* ---------------------------------------------------------------------
   Clasificador K-Vecinos-más-Cercanos (KNN) — señas ESTÁTICAS
   --------------------------------------------------------------------- */

function euclideanDistance(v1, v2) {
  let sum = 0;
  const len = Math.min(v1.length, v2.length);
  for (let i = 0; i < len; i++) {
    const d = v1[i] - v2[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/**
 * Clasifica un vector de características estático contra un conjunto de
 * ejemplos ya etiquetados, usando KNN con voto ponderado por distancia
 * inversa (los vecinos más cercanos pesan más).
 */
export function classifyStatic(featureVector, trainingSet, k = 5) {
  if (!trainingSet || trainingSet.length === 0) return null;

  const distances = trainingSet.map((item) => ({
    label: item.label,
    dist: euclideanDistance(featureVector, item.vector)
  }));
  distances.sort((a, b) => a.dist - b.dist);

  const neighbors = distances.slice(0, Math.min(k, distances.length));

  const weightByLabel = {};
  let totalWeight = 0;
  for (const n of neighbors) {
    const weight = 1 / (n.dist + 1e-6);
    weightByLabel[n.label] = (weightByLabel[n.label] || 0) + weight;
    totalWeight += weight;
  }

  let bestLabel = null;
  let bestWeight = -1;
  for (const label in weightByLabel) {
    if (weightByLabel[label] > bestWeight) {
      bestWeight = weightByLabel[label];
      bestLabel = label;
    }
  }

  const confidence = totalWeight > 0 ? Math.round((bestWeight / totalWeight) * 100) : 0;
  return { label: bestLabel, confidence, nearestDistance: neighbors[0].dist };
}

/* ---------------------------------------------------------------------
   Reconocimiento temporal (DTW) — señas DINÁMICAS
   --------------------------------------------------------------------- */

const DTW_RESAMPLE_LENGTH = 16; // número fijo de "paradas" usadas para comparar dos secuencias

/**
 * Reduce o amplía una secuencia de vectores a una longitud fija mediante
 * interpolación lineal, para poder comparar secuencias grabadas con
 * distinta cantidad de fotogramas (distinta duración o framerate).
 */
export function resampleSequence(sequence, targetLength = DTW_RESAMPLE_LENGTH) {
  if (!sequence || sequence.length === 0) return [];
  if (sequence.length === 1) return new Array(targetLength).fill(sequence[0]);
  if (sequence.length === targetLength) return sequence;

  const result = [];
  for (let i = 0; i < targetLength; i++) {
    const t = i / (targetLength - 1);
    const srcIndex = t * (sequence.length - 1);
    const lower = Math.floor(srcIndex);
    const upper = Math.min(Math.ceil(srcIndex), sequence.length - 1);
    const frac = srcIndex - lower;
    if (lower === upper) {
      result.push(sequence[lower]);
    } else {
      const a = sequence[lower], b = sequence[upper];
      result.push(a.map((val, idx) => val + (b[idx] - val) * frac));
    }
  }
  return result;
}

/**
 * Distancia Dynamic Time Warping (DTW) entre dos secuencias de vectores.
 * Encuentra el camino de alineación de menor costo acumulado entre
 * ambas secuencias y lo normaliza por su longitud total, para que
 * secuencias más largas no obtengan automáticamente distancias mayores.
 */
export function dtwDistance(seqA, seqB) {
  const n = seqA.length, m = seqB.length;
  if (n === 0 || m === 0) return Infinity;

  const cost = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(Infinity));
  cost[0][0] = 0;

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const d = euclideanDistance(seqA[i - 1], seqB[j - 1]);
      cost[i][j] = d + Math.min(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1]);
    }
  }
  return cost[n][m] / (n + m);
}

/**
 * Clasifica una secuencia dinámica (vector de fotogramas) contra los
 * ejemplos dinámicos entrenados, usando 1-vecino-más-cercano por DTW
 * (con desempate ponderado si hay varios ejemplos muy cercanos).
 * Devuelve una "puntuación de similitud" (0-100), NO una probabilidad.
 */
export function classifyDynamic(sequenceVectors, trainingSet, k = 3) {
  if (!trainingSet || trainingSet.length === 0) return null;
  if (!sequenceVectors || sequenceVectors.length < 2) return null;

  const query = resampleSequence(sequenceVectors);
  const distances = trainingSet.map((item) => ({
    label: item.label,
    dist: dtwDistance(query, resampleSequence(item.sequence))
  }));
  distances.sort((a, b) => a.dist - b.dist);

  const neighbors = distances.slice(0, Math.min(k, distances.length));
  const weightByLabel = {};
  let totalWeight = 0;
  for (const n of neighbors) {
    const weight = 1 / (n.dist + 1e-3);
    weightByLabel[n.label] = (weightByLabel[n.label] || 0) + weight;
    totalWeight += weight;
  }
  let bestLabel = null, bestWeight = -1;
  for (const label in weightByLabel) {
    if (weightByLabel[label] > bestWeight) {
      bestWeight = weightByLabel[label];
      bestLabel = label;
    }
  }

  // La distancia DTW no tiene un rango fijo conocido de antemano, así que
  // se convierte en una puntuación de similitud 0-100 solo para mostrarla
  // de forma intuitiva en la interfaz. NO debe interpretarse como una
  // probabilidad estadística calibrada.
  const nearestDistance = neighbors[0].dist;
  const similarity = Math.max(0, Math.min(100, Math.round(100 - nearestDistance * 100)));

  return { label: bestLabel, similarity, nearestDistance, voteShare: Math.round((bestWeight / totalWeight) * 100) };
}

/* ---------------------------------------------------------------------
   Construcción de conjuntos de entrenamiento a partir de un proyecto
   --------------------------------------------------------------------- */

/** Conjunto plano {label, vector} con todos los ejemplos de patrones ESTÁTICOS entrenados. */
export function buildStaticTrainingSet(project) {
  const trainingSet = [];
  for (const pattern of project.patterns) {
    if (pattern.type !== "static" || !pattern.trained) continue;
    for (const example of pattern.examples) {
      if (Array.isArray(example.landmarks)) {
        trainingSet.push({ label: pattern.name, vector: example.landmarks });
      }
    }
  }
  return trainingSet;
}

/** Conjunto plano {label, sequence} con todas las secuencias de patrones DINÁMICOS entrenados. */
export function buildDynamicTrainingSet(project) {
  const trainingSet = [];
  for (const pattern of project.patterns) {
    if (pattern.type !== "dynamic" || !pattern.trained) continue;
    for (const example of pattern.examples) {
      if (Array.isArray(example.frames) && example.frames.length > 1) {
        trainingSet.push({ label: pattern.name, sequence: example.frames.map((f) => f.vector) });
      }
    }
  }
  return trainingSet;
}
