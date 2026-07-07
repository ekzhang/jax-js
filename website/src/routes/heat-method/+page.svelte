<script lang="ts">
  import { resolve } from "$app/paths";

  import { defaultDevice, init, jit, lax, numpy as np } from "@jax-js/jax";
  import {
    Crosshair,
    ExternalLink,
    Gauge,
    MousePointer2,
    RefreshCw,
    RotateCcw,
    Shuffle,
  } from "@lucide/svelte";
  import { onMount } from "svelte";
  import { OrthographicCamera, Vector3 } from "three";
  import type { BufferGeometry, Object3D } from "three";
  import { OrbitControls } from "three/addons/controls/OrbitControls.js";
  import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

  const CANVAS_WIDTH = 1120;
  const CANVAS_HEIGHT = 760;
  const CANVAS_PAD = 52;
  const POISSON_REGULARIZER = 1e-5;
  const EPS = 1e-8;

  type Vec3 = [number, number, number];

  type RawMesh = {
    title: string;
    positions: Float32Array;
    faces: Int32Array;
  };

  type MeshSource = {
    id: string;
    title: string;
    description: string;
    heatScale: number;
  } & (
    | {
        kind: "obj";
        url: string;
      }
    | {
        kind: "bunny";
        url: string;
      }
    | {
        kind: "procedural";
        build: () => RawMesh;
      }
  );

  type BunnyModule = {
    default?: {
      positions?: number[][];
      cells?: number[][];
    };
    positions?: number[][];
    cells?: number[][];
  };

  type MeshData = {
    title: string;
    description: string;
    vertexCount: number;
    faceCount: number;
    maxDegree: number;
    maxIncident: number;
    positions: Float32Array;
    heights: Float32Array;
    projected: Float32Array;
    faces: Int32Array;
    neighbors: Int32Array;
    weights: Float32Array;
    mass: Float32Array;
    faceIndices: Int32Array;
    gradBasis: Float32Array;
    incidentFaces: Int32Array;
    incidentAreas: Float32Array;
    incidentGrad: Float32Array;
    meanEdgeLength: number;
  };

  type SolverMode = "cg" | "cholesky";

  type SolveResult = {
    values: Float32Array;
    factorMs?: number;
  };

  type HeatSolver = {
    mode: SolverMode;
    setupMs: number;
    solve: (
      source: number,
      heatScale: number,
      iterations: number,
    ) => Promise<SolveResult>;
    dispose: () => void;
  };

  type BinaryArrayKernel = {
    (x: np.Array, y: np.Array): np.Array;
    dispose: () => void;
  };

  let canvas: HTMLCanvasElement;
  let mesh = $state<MeshData | null>(null);
  let solver: HeatSolver | null = null;
  let distance = $state<Float32Array | null>(null);
  let sourceIndex = $state(0);
  let solving = $state(false);
  let initialized = $state(false);
  let deviceName = $state("starting");
  let solveMs = $state(0);
  let distanceSpan = $state(0);
  let heatScale = $state(1.2);
  let cgIterations = $state(48);
  let solverMode = $state<SolverMode>("cg");
  let factorMs = $state(0);
  let selectedMeshId = $state("bunny");
  let loadingMesh = $state(false);
  let loadError = $state("");
  let isDraggingView = $state(false);
  let pendingTimer: number | undefined;
  let solveSerial = 0;
  let meshSerial = 0;
  let viewDepth = new Float32Array();
  let faceDepth = new Float32Array();
  let faceOrder: number[] = [];
  let orbitCamera: OrthographicCamera | null = null;
  let orbitControls: OrbitControls | null = null;
  let clickPointerId: number | null = null;
  let clickStartClientX = 0;
  let clickStartClientY = 0;
  let suppressNextClick = false;

  const meshSources: MeshSource[] = [
    {
      id: "bunny",
      title: "Stanford Bunny",
      description: "Classic Stanford bunny from the jsDelivr npm package.",
      kind: "bunny",
      url: "https://cdn.jsdelivr.net/npm/bunny@1.0.1/+esm",
      heatScale: 1.05,
    },
    {
      id: "spot",
      title: "Spot",
      description: "Keenan Crane's Spot model from common test meshes.",
      kind: "obj",
      url: "https://cdn.jsdelivr.net/gh/alecjacobson/common-3d-test-models@master/data/spot.obj",
      heatScale: 0.95,
    },
    {
      id: "horse",
      title: "Horse",
      description:
        "Quad horse from libigl tutorial data, triangulated by OBJLoader.",
      kind: "obj",
      url: "https://cdn.jsdelivr.net/gh/libigl/libigl-tutorial-data@master/horse_quad.obj",
      heatScale: 1.25,
    },
    {
      id: "cow",
      title: "Cow",
      description: "Classic cow test mesh from common-3d-test-models.",
      kind: "obj",
      url: "https://cdn.jsdelivr.net/gh/alecjacobson/common-3d-test-models@master/data/cow.obj",
      heatScale: 1.15,
    },
    {
      id: "suzanne",
      title: "Suzanne",
      description: "Blender's compact Suzanne monkey mesh.",
      kind: "obj",
      url: "https://cdn.jsdelivr.net/gh/alecjacobson/common-3d-test-models@master/data/suzanne.obj",
      heatScale: 1,
    },
    {
      id: "plane",
      title: "Plane",
      description: "Procedural triangulated plane for Euclidean circles.",
      kind: "procedural",
      build: () => makePlaneMesh(46),
      heatScale: 1.25,
    },
    {
      id: "sphere",
      title: "Sphere",
      description: "Procedural sphere with closed topology.",
      kind: "procedural",
      build: () => makeSphereMesh(36, 72),
      heatScale: 1,
    },
    {
      id: "torus",
      title: "Torus",
      description: "Procedural torus with wraparound geodesics.",
      kind: "procedural",
      build: () => makeTorusMesh(72, 24),
      heatScale: 1,
    },
  ];

  function sub(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  }

  function dot(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  function cross(a: Vec3, b: Vec3): Vec3 {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }

  function norm(a: Vec3): number {
    return Math.sqrt(dot(a, a));
  }

  function scale(a: Vec3, s: number): Vec3 {
    return [a[0] * s, a[1] * s, a[2] * s];
  }

  function addMapValue(map: Map<number, number>, key: number, value: number) {
    map.set(key, (map.get(key) ?? 0) + value);
  }

  function vertexAt(positions: Float32Array, index: number): Vec3 {
    return [
      positions[index * 3],
      positions[index * 3 + 1],
      positions[index * 3 + 2],
    ];
  }

  function selectedSource(): MeshSource {
    return (
      meshSources.find((source) => source.id === selectedMeshId) ??
      meshSources[0]
    );
  }

  function makeRawMesh(
    title: string,
    positions: number[],
    faces: number[],
  ): RawMesh {
    return {
      title,
      positions: new Float32Array(positions),
      faces: new Int32Array(faces),
    };
  }

  function makePlaneMesh(size: number): RawMesh {
    const positions: number[] = [];
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const u = x / (size - 1);
        const v = y / (size - 1);
        positions.push((u - 0.5) * 2.4, (v - 0.5) * 2.05, 0);
      }
    }

    const faces: number[] = [];
    for (let y = 0; y < size - 1; y++) {
      for (let x = 0; x < size - 1; x++) {
        const v00 = y * size + x;
        const v10 = y * size + x + 1;
        const v01 = (y + 1) * size + x;
        const v11 = (y + 1) * size + x + 1;
        if ((x + y) % 2 === 0) {
          faces.push(v00, v10, v11, v00, v11, v01);
        } else {
          faces.push(v00, v10, v01, v10, v11, v01);
        }
      }
    }

    return makeRawMesh("Plane", positions, faces);
  }

  function makeSphereMesh(rings: number, segments: number): RawMesh {
    const positions: number[] = [0, 0, 1];
    for (let ring = 1; ring < rings; ring++) {
      const phi = (Math.PI * ring) / rings;
      const radius = Math.sin(phi);
      const z = Math.cos(phi);
      for (let segment = 0; segment < segments; segment++) {
        const theta = (Math.PI * 2 * segment) / segments;
        positions.push(radius * Math.cos(theta), radius * Math.sin(theta), z);
      }
    }
    const bottom = positions.length / 3;
    positions.push(0, 0, -1);

    const ringIndex = (ring: number, segment: number) =>
      1 + (ring - 1) * segments + ((segment + segments) % segments);

    const faces: number[] = [];
    for (let segment = 0; segment < segments; segment++) {
      faces.push(0, ringIndex(1, segment + 1), ringIndex(1, segment));
    }
    for (let ring = 1; ring < rings - 1; ring++) {
      for (let segment = 0; segment < segments; segment++) {
        const a = ringIndex(ring, segment);
        const b = ringIndex(ring, segment + 1);
        const c = ringIndex(ring + 1, segment);
        const d = ringIndex(ring + 1, segment + 1);
        faces.push(a, c, d, a, d, b);
      }
    }
    for (let segment = 0; segment < segments; segment++) {
      faces.push(
        ringIndex(rings - 1, segment),
        ringIndex(rings - 1, segment + 1),
        bottom,
      );
    }

    return makeRawMesh("Sphere", positions, faces);
  }

  function makeTorusMesh(
    majorSegments: number,
    minorSegments: number,
  ): RawMesh {
    const positions: number[] = [];
    const majorRadius = 0.76;
    const minorRadius = 0.28;
    for (let major = 0; major < majorSegments; major++) {
      const theta = (Math.PI * 2 * major) / majorSegments;
      const cosTheta = Math.cos(theta);
      const sinTheta = Math.sin(theta);
      for (let minor = 0; minor < minorSegments; minor++) {
        const phi = (Math.PI * 2 * minor) / minorSegments;
        const radial = majorRadius + minorRadius * Math.cos(phi);
        positions.push(
          radial * cosTheta,
          radial * sinTheta,
          minorRadius * Math.sin(phi),
        );
      }
    }

    const vertex = (major: number, minor: number) =>
      ((major + majorSegments) % majorSegments) * minorSegments +
      ((minor + minorSegments) % minorSegments);
    const faces: number[] = [];
    for (let major = 0; major < majorSegments; major++) {
      for (let minor = 0; minor < minorSegments; minor++) {
        const a = vertex(major, minor);
        const b = vertex(major + 1, minor);
        const c = vertex(major + 1, minor + 1);
        const d = vertex(major, minor + 1);
        faces.push(a, b, c, a, c, d);
      }
    }

    return makeRawMesh("Torus", positions, faces);
  }

  function normalizeRawMesh(raw: RawMesh): RawMesh {
    const positions = raw.positions;
    const vertexCount = positions.length / 3;
    if (!vertexCount || raw.faces.length < 3) {
      throw new Error(`${raw.title} did not contain usable triangles.`);
    }

    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (let v = 0; v < vertexCount; v++) {
      const x = positions[v * 3];
      const y = positions[v * 3 + 1];
      const z = positions[v * 3 + 2];
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      minZ = Math.min(minZ, z);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      maxZ = Math.max(maxZ, z);
    }

    const centerX = (minX + maxX) * 0.5;
    const centerY = (minY + maxY) * 0.5;
    const centerZ = (minZ + maxZ) * 0.5;
    const fitScale =
      2.35 / Math.max(maxX - minX, maxY - minY, maxZ - minZ, EPS);
    const normalized = new Float32Array(positions.length);
    for (let v = 0; v < vertexCount; v++) {
      normalized[v * 3] = (positions[v * 3] - centerX) * fitScale;
      normalized[v * 3 + 1] = (positions[v * 3 + 1] - centerY) * fitScale;
      normalized[v * 3 + 2] = (positions[v * 3 + 2] - centerZ) * fitScale;
    }

    return {
      title: raw.title,
      positions: normalized,
      faces: raw.faces,
    };
  }

  async function loadBunnyModule(url: string): Promise<RawMesh> {
    const module = (await import(/* @vite-ignore */ url)) as BunnyModule;
    const positions = module.positions ?? module.default?.positions;
    const cells = module.cells ?? module.default?.cells;
    if (!positions || !cells) {
      throw new Error("The bunny module did not expose positions and cells.");
    }

    const rawPositions = new Float32Array(positions.length * 3);
    for (let i = 0; i < positions.length; i++) {
      const point = positions[i];
      rawPositions[i * 3] = point[0];
      rawPositions[i * 3 + 1] = point[2];
      rawPositions[i * 3 + 2] = point[1];
    }

    const faces: number[] = [];
    for (const cell of cells) {
      for (let i = 1; i < cell.length - 1; i++) {
        faces.push(cell[0], cell[i], cell[i + 1]);
      }
    }

    return {
      title: "Stanford Bunny",
      positions: rawPositions,
      faces: new Int32Array(faces),
    };
  }

  function parseObjMesh(title: string, text: string): RawMesh {
    const object = new OBJLoader().parse(text);
    object.updateMatrixWorld(true);

    const positions: number[] = [];
    const faces: number[] = [];
    const welded = new Map<string, number>();
    const temp = new Vector3();

    object.traverse((child: Object3D) => {
      const geometry = (child as Object3D & { geometry?: BufferGeometry })
        .geometry;
      const position = geometry?.getAttribute("position");
      if (!geometry || !position) return;

      const index = geometry.getIndex();
      const count = index ? index.count : position.count;
      const addVertex = (vertexIndex: number) => {
        temp.fromBufferAttribute(position, vertexIndex);
        temp.applyMatrix4(child.matrixWorld);
        const x = temp.x;
        const y = temp.z;
        const z = temp.y;
        const key = `${x.toFixed(5)},${y.toFixed(5)},${z.toFixed(5)}`;
        const existing = welded.get(key);
        if (existing !== undefined) return existing;

        const next = positions.length / 3;
        positions.push(x, y, z);
        welded.set(key, next);
        return next;
      };

      for (let i = 0; i <= count - 3; i += 3) {
        const a = addVertex(index ? index.getX(i) : i);
        const b = addVertex(index ? index.getX(i + 1) : i + 1);
        const c = addVertex(index ? index.getX(i + 2) : i + 2);
        faces.push(a, b, c);
      }
    });

    return makeRawMesh(title, positions, faces);
  }

  async function loadRawMesh(source: MeshSource): Promise<RawMesh> {
    if (source.kind === "procedural") return source.build();
    if (source.kind === "bunny") return loadBunnyModule(source.url);

    const response = await fetch(source.url);
    if (!response.ok) {
      throw new Error(`Could not load ${source.title}: ${response.status}`);
    }

    return parseObjMesh(source.title, await response.text());
  }

  function buildMesh(rawMesh: RawMesh, source: MeshSource): MeshData {
    const raw = normalizeRawMesh(rawMesh);
    const vertexCount = raw.positions.length / 3;
    const positions = raw.positions;
    const heights = new Float32Array(vertexCount);
    for (let v = 0; v < vertexCount; v++) {
      heights[v] = positions[v * 3 + 2];
    }

    const faceList: number[] = [];
    for (let i = 0; i < raw.faces.length; i += 3) {
      const i0 = raw.faces[i];
      const i1 = raw.faces[i + 1];
      const i2 = raw.faces[i + 2];
      if (
        i0 < 0 ||
        i1 < 0 ||
        i2 < 0 ||
        i0 >= vertexCount ||
        i1 >= vertexCount ||
        i2 >= vertexCount ||
        i0 === i1 ||
        i1 === i2 ||
        i2 === i0
      ) {
        continue;
      }

      const normal = cross(
        sub(vertexAt(positions, i1), vertexAt(positions, i0)),
        sub(vertexAt(positions, i2), vertexAt(positions, i0)),
      );
      if (norm(normal) > EPS) faceList.push(i0, i1, i2);
    }

    if (!faceList.length) {
      throw new Error(`${raw.title} did not contain usable triangles.`);
    }

    const faces = new Int32Array(faceList);
    const faceCount = faces.length / 3;
    const faceIndices = new Int32Array(faces);
    const gradBasis = new Float32Array(faceCount * 9);
    const mass = new Float32Array(vertexCount);
    const adjacency = Array.from(
      { length: vertexCount },
      () => new Map<number, number>(),
    );
    const incidents = Array.from(
      { length: vertexCount },
      () => [] as { face: number; area: number; grad: Vec3 }[],
    );

    let totalEdgeLength = 0;
    let edgeSamples = 0;

    for (let f = 0; f < faceCount; f++) {
      const i0 = faces[f * 3];
      const i1 = faces[f * 3 + 1];
      const i2 = faces[f * 3 + 2];
      const p0 = vertexAt(positions, i0);
      const p1 = vertexAt(positions, i1);
      const p2 = vertexAt(positions, i2);

      const e01 = sub(p1, p0);
      const e02 = sub(p2, p0);
      const normal = cross(e01, e02);
      const area2 = Math.max(norm(normal), EPS);
      const area = 0.5 * area2;
      const invNormalSq = 1 / (area2 * area2);

      const grads = [
        scale(cross(normal, sub(p2, p1)), invNormalSq),
        scale(cross(normal, sub(p0, p2)), invNormalSq),
        scale(cross(normal, sub(p1, p0)), invNormalSq),
      ] satisfies Vec3[];

      for (let local = 0; local < 3; local++) {
        gradBasis[f * 9 + local * 3] = grads[local][0];
        gradBasis[f * 9 + local * 3 + 1] = grads[local][1];
        gradBasis[f * 9 + local * 3 + 2] = grads[local][2];
      }

      for (const [local, vertex] of [i0, i1, i2].entries()) {
        mass[vertex] += area / 3;
        incidents[vertex].push({ face: f, area, grad: grads[local] });
      }

      const cot0 = dot(sub(p1, p0), sub(p2, p0)) / area2;
      const cot1 = dot(sub(p0, p1), sub(p2, p1)) / area2;
      const cot2 = dot(sub(p0, p2), sub(p1, p2)) / area2;

      for (const [a, b, w] of [
        [i1, i2, cot0 * 0.5],
        [i0, i2, cot1 * 0.5],
        [i0, i1, cot2 * 0.5],
      ] as const) {
        const weight = Number.isFinite(w) ? Math.max(0, w) : 0;
        addMapValue(adjacency[a], b, weight);
        addMapValue(adjacency[b], a, weight);
        totalEdgeLength += norm(
          sub(vertexAt(positions, a), vertexAt(positions, b)),
        );
        edgeSamples++;
      }
    }

    const maxDegree = Math.max(1, ...adjacency.map((row) => row.size));
    const neighbors = new Int32Array(vertexCount * maxDegree);
    const weights = new Float32Array(vertexCount * maxDegree);
    for (let v = 0; v < vertexCount; v++) {
      neighbors.fill(v, v * maxDegree, (v + 1) * maxDegree);
      let k = 0;
      for (const [neighbor, weight] of adjacency[v]) {
        neighbors[v * maxDegree + k] = neighbor;
        weights[v * maxDegree + k] = weight;
        k++;
      }
    }

    const maxIncident = Math.max(1, ...incidents.map((row) => row.length));
    const incidentFaces = new Int32Array(vertexCount * maxIncident);
    const incidentAreas = new Float32Array(vertexCount * maxIncident);
    const incidentGrad = new Float32Array(vertexCount * maxIncident * 3);
    for (let v = 0; v < vertexCount; v++) {
      for (let k = 0; k < incidents[v].length; k++) {
        const incident = incidents[v][k];
        incidentFaces[v * maxIncident + k] = incident.face;
        incidentAreas[v * maxIncident + k] = incident.area;
        incidentGrad[(v * maxIncident + k) * 3] = incident.grad[0];
        incidentGrad[(v * maxIncident + k) * 3 + 1] = incident.grad[1];
        incidentGrad[(v * maxIncident + k) * 3 + 2] = incident.grad[2];
      }
    }

    const projected = new Float32Array(vertexCount * 2);
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let v = 0; v < vertexCount; v++) {
      const x = positions[v * 3];
      const y = positions[v * 3 + 1];
      const z = positions[v * 3 + 2];
      const sx = x + 0.32 * y;
      const sy = 0.78 * y - 0.78 * z;
      projected[v * 2] = sx;
      projected[v * 2 + 1] = sy;
      minX = Math.min(minX, sx);
      minY = Math.min(minY, sy);
      maxX = Math.max(maxX, sx);
      maxY = Math.max(maxY, sy);
    }

    const projectedWidth = Math.max(maxX - minX, EPS);
    const projectedHeight = Math.max(maxY - minY, EPS);
    const scaleToFit = Math.min(
      (CANVAS_WIDTH - CANVAS_PAD * 2) / projectedWidth,
      (CANVAS_HEIGHT - CANVAS_PAD * 2) / projectedHeight,
    );
    const offsetX = (CANVAS_WIDTH - projectedWidth * scaleToFit) * 0.5;
    const offsetY = (CANVAS_HEIGHT - projectedHeight * scaleToFit) * 0.5;
    for (let v = 0; v < vertexCount; v++) {
      projected[v * 2] = offsetX + (projected[v * 2] - minX) * scaleToFit;
      projected[v * 2 + 1] =
        offsetY + (projected[v * 2 + 1] - minY) * scaleToFit;
    }

    return {
      title: raw.title,
      description: source.description,
      vertexCount,
      faceCount,
      maxDegree,
      maxIncident,
      positions,
      heights,
      projected,
      faces,
      neighbors,
      weights,
      mass,
      faceIndices,
      gradBasis,
      incidentFaces,
      incidentAreas,
      incidentGrad,
      meanEdgeLength: edgeSamples ? totalEdgeLength / edgeSamples : 1,
    };
  }

  function makeCgSolver(data: MeshData): HeatSolver {
    const vertexCount = data.vertexCount;
    const maxDegree = data.maxDegree;
    const maxIncident = data.maxIncident;
    const neighbors = np.array(data.neighbors as Int32Array<ArrayBuffer>, {
      shape: [vertexCount, maxDegree],
      dtype: np.int32,
    });
    const weights = np.array(data.weights as Float32Array<ArrayBuffer>, {
      shape: [vertexCount, maxDegree],
      dtype: np.float32,
    });
    const mass = np.array(data.mass as Float32Array<ArrayBuffer>, {
      shape: [vertexCount],
      dtype: np.float32,
    });
    const faceIndices = np.array(data.faceIndices as Int32Array<ArrayBuffer>, {
      shape: [data.faceCount, 3],
      dtype: np.int32,
    });
    const gradBasis = np.array(data.gradBasis as Float32Array<ArrayBuffer>, {
      shape: [data.faceCount, 3, 3],
      dtype: np.float32,
    });
    const incidentFaces = np.array(
      data.incidentFaces as Int32Array<ArrayBuffer>,
      {
        shape: [vertexCount, maxIncident],
        dtype: np.int32,
      },
    );
    const incidentAreas = np.array(
      data.incidentAreas as Float32Array<ArrayBuffer>,
      {
        shape: [vertexCount, maxIncident],
        dtype: np.float32,
      },
    );
    const incidentGrad = np.array(
      data.incidentGrad as Float32Array<ArrayBuffer>,
      {
        shape: [vertexCount, maxIncident, 3],
        dtype: np.float32,
      },
    );
    const vertexIds = np.arange(vertexCount, undefined, undefined, {
      dtype: np.int32,
    });

    function laplace(x: np.Array): np.Array {
      const neighborValues = np.take(x.ref, neighbors.ref);
      const centerValues = x.reshape([vertexCount, 1]);
      return weights.ref.mul(centerValues.sub(neighborValues)).sum(1);
    }

    function heatMatvec(x: np.Array, time: np.Array): np.Array {
      const diagonal = mass.ref.mul(x.ref);
      return diagonal.add(laplace(x).mul(time));
    }

    function poissonMatvec(x: np.Array, regularizer: np.Array): np.Array {
      return laplace(x.ref).add(mass.ref.mul(x).mul(regularizer));
    }

    function heatCgStep(
      x: np.Array,
      r: np.Array,
      p: np.Array,
      rsold: np.Array,
      time: np.Array,
    ) {
      const ap = heatMatvec(p.ref, time);
      const denom = np.dot(p.ref, ap.ref).add(1e-12);
      const alpha = rsold.ref.div(denom);
      const nextX = x.add(p.ref.mul(alpha.ref));
      const nextR = r.sub(ap.mul(alpha));
      const rsnew = np.dot(nextR.ref, nextR.ref);
      const beta = rsnew.ref.div(rsold);
      const nextP = nextR.ref.add(p.mul(beta));
      return [nextX, nextR, nextP, rsnew];
    }

    function poissonCgStep(
      x: np.Array,
      r: np.Array,
      p: np.Array,
      rsold: np.Array,
      regularizer: np.Array,
    ) {
      const ap = poissonMatvec(p.ref, regularizer);
      const denom = np.dot(p.ref, ap.ref).add(1e-12);
      const alpha = rsold.ref.div(denom);
      const nextX = x.add(p.ref.mul(alpha.ref));
      const nextR = r.sub(ap.mul(alpha));
      const rsnew = np.dot(nextR.ref, nextR.ref);
      const beta = rsnew.ref.div(rsold);
      const nextP = nextR.ref.add(p.mul(beta));
      return [nextX, nextR, nextP, rsnew];
    }

    const computeDivergence = jit(function computeDivergence(u: np.Array) {
      const faceU = np.take(u, faceIndices.ref);
      const gradU = faceU.slice([], [], null).mul(gradBasis.ref).sum(1);
      const length = np
        .sqrt(gradU.ref.mul(gradU.ref).sum(1, { keepdims: true }).add(EPS))
        .add(EPS);
      const field = gradU.mul(-1).div(length);
      const incidentField = np.take(field, incidentFaces.ref, 0);
      return incidentField
        .mul(incidentGrad.ref)
        .sum(2)
        .mul(incidentAreas.ref)
        .sum(1)
        .mul(-1);
    });

    const centerVector = jit(function centerVector(x: np.Array) {
      return x.sub(x.ref.mean());
    });

    const sourceVector = jit(function sourceVector(source: np.Array) {
      return vertexIds.ref.equal(source).astype(np.float32);
    });

    const heatRunners = new Map<number, BinaryArrayKernel>();
    const poissonRunners = new Map<number, BinaryArrayKernel>();

    function buildCgRunner(
      name: string,
      iterations: number,
      step: typeof heatCgStep,
    ): BinaryArrayKernel {
      const runner = jit(function runCachedCg(
        b: np.Array,
        param: np.Array,
      ): np.Array {
        let x = np.zeros([vertexCount], { dtype: np.float32 });
        let r = b;
        let p = r.ref;
        let rs = np.dot(r.ref, r.ref);
        for (let i = 0; i < iterations; i++) {
          [x, r, p, rs] = step(x, r, p, rs, param.ref) as [
            np.Array,
            np.Array,
            np.Array,
            np.Array,
          ];
        }
        return x;
      }) as BinaryArrayKernel;
      Object.defineProperty(runner, "name", { value: name });
      return runner;
    }

    function cachedRunner(
      cache: Map<number, BinaryArrayKernel>,
      name: string,
      iterations: number,
      step: typeof heatCgStep,
    ): BinaryArrayKernel {
      let runner = cache.get(iterations);
      if (!runner) {
        runner = buildCgRunner(name, iterations, step);
        cache.set(iterations, runner);
      }
      return runner;
    }

    return {
      mode: "cg",
      setupMs: 0,
      async solve(source: number, heatScaleValue: number, iterations: number) {
        const heatTime = np.array(data.meanEdgeLength ** 2 * heatScaleValue, {
          dtype: np.float32,
        });
        const regularizer = np.array(POISSON_REGULARIZER, {
          dtype: np.float32,
        });
        const sourceIndex = np.array(source, { dtype: np.int32 });
        const heat = cachedRunner(
          heatRunners,
          "runHeatCg",
          iterations,
          heatCgStep,
        )(sourceVector(sourceIndex), heatTime);

        const divergence = centerVector(computeDivergence(heat));
        const phi = cachedRunner(
          poissonRunners,
          "runPoissonCg",
          iterations,
          poissonCgStep,
        )(divergence, regularizer);

        return { values: (await phi.data()) as Float32Array };
      },
      dispose() {
        neighbors.dispose();
        weights.dispose();
        mass.dispose();
        faceIndices.dispose();
        gradBasis.dispose();
        incidentFaces.dispose();
        incidentAreas.dispose();
        incidentGrad.dispose();
        vertexIds.dispose();
        sourceVector.dispose();
        for (const runner of heatRunners.values()) runner.dispose();
        for (const runner of poissonRunners.values()) runner.dispose();
        computeDivergence.dispose();
        centerVector.dispose();
      },
    };
  }

  function buildDenseOperator(
    data: MeshData,
    laplaceScale: number,
    massScale: number,
  ): Float32Array {
    const n = data.vertexCount;
    const matrix = new Float32Array(n * n);
    for (let v = 0; v < n; v++) {
      let diagonal = data.mass[v] * massScale;
      const row = v * n;
      const neighborBase = v * data.maxDegree;
      for (let k = 0; k < data.maxDegree; k++) {
        const weight = data.weights[neighborBase + k] * laplaceScale;
        if (weight === 0) continue;
        const neighbor = data.neighbors[neighborBase + k];
        matrix[row + neighbor] -= weight;
        diagonal += weight;
      }
      matrix[row + v] += diagonal;
    }
    return matrix;
  }

  async function factorDenseOperator(
    matrix: Float32Array,
    vertexCount: number,
  ): Promise<np.Array> {
    const array = np.array(matrix as Float32Array<ArrayBuffer>, {
      shape: [vertexCount, vertexCount],
      dtype: np.float32,
    });
    let factor: np.Array | null = null;
    try {
      factor = np.linalg.cholesky(array, { symmetrizeInput: false });
      await factor.blockUntilReady();
      return factor;
    } catch (error) {
      factor?.dispose();
      throw error;
    }
  }

  async function makeDenseCholeskySolver(
    data: MeshData,
    initialHeatScale: number,
  ): Promise<HeatSolver> {
    const vertexCount = data.vertexCount;
    const faceIndices = np.array(data.faceIndices as Int32Array<ArrayBuffer>, {
      shape: [data.faceCount, 3],
      dtype: np.int32,
    });
    const gradBasis = np.array(data.gradBasis as Float32Array<ArrayBuffer>, {
      shape: [data.faceCount, 3, 3],
      dtype: np.float32,
    });
    const incidentFaces = np.array(
      data.incidentFaces as Int32Array<ArrayBuffer>,
      {
        shape: [vertexCount, data.maxIncident],
        dtype: np.int32,
      },
    );
    const incidentAreas = np.array(
      data.incidentAreas as Float32Array<ArrayBuffer>,
      {
        shape: [vertexCount, data.maxIncident],
        dtype: np.float32,
      },
    );
    const incidentGrad = np.array(
      data.incidentGrad as Float32Array<ArrayBuffer>,
      {
        shape: [vertexCount, data.maxIncident, 3],
        dtype: np.float32,
      },
    );
    const vertexIds = np.arange(vertexCount, undefined, undefined, {
      dtype: np.int32,
    });

    const computeDivergence = jit(function computeDivergence(u: np.Array) {
      const faceU = np.take(u, faceIndices.ref);
      const gradU = faceU.slice([], [], null).mul(gradBasis.ref).sum(1);
      const length = np
        .sqrt(gradU.ref.mul(gradU.ref).sum(1, { keepdims: true }).add(EPS))
        .add(EPS);
      const field = gradU.mul(-1).div(length);
      const incidentField = np.take(field, incidentFaces.ref, 0);
      return incidentField
        .mul(incidentGrad.ref)
        .sum(2)
        .mul(incidentAreas.ref)
        .sum(1)
        .mul(-1);
    });

    const centerVector = jit(function centerVector(x: np.Array) {
      return x.sub(x.ref.mean());
    });

    const sourceVector = jit(function sourceVector(source: np.Array) {
      return vertexIds.ref.equal(source).astype(np.float32);
    });

    let heatFactor: np.Array | null = null;
    let heatFactorTime = Number.NaN;
    let lastFactorMs = 0;

    async function refactorHeat(heatScaleValue: number) {
      const heatTime = data.meanEdgeLength ** 2 * heatScaleValue;
      if (heatFactor && Math.abs(heatTime - heatFactorTime) < 1e-12) return;

      const start = performance.now();
      const nextFactor = await factorDenseOperator(
        buildDenseOperator(data, heatTime, 1),
        vertexCount,
      );
      heatFactor?.dispose();
      heatFactor = nextFactor;
      heatFactorTime = heatTime;
      lastFactorMs = performance.now() - start;
    }

    function solveWithFactor(factor: np.Array, b: np.Array): np.Array {
      const rhs = b.reshape([vertexCount, 1]);
      const y = lax.linalg.triangularSolve(factor.ref, rhs, {
        leftSide: true,
        lower: true,
      });
      return lax.linalg
        .triangularSolve(factor.ref, y, {
          leftSide: true,
          lower: true,
          transposeA: true,
        })
        .reshape([vertexCount]);
    }

    const setupStart = performance.now();
    const poissonFactor = await factorDenseOperator(
      buildDenseOperator(data, 1, POISSON_REGULARIZER),
      vertexCount,
    );
    await refactorHeat(initialHeatScale);
    const setupMs = performance.now() - setupStart;
    lastFactorMs = setupMs;

    return {
      mode: "cholesky",
      setupMs,
      async solve(source: number, heatScaleValue: number) {
        await refactorHeat(heatScaleValue);
        if (!heatFactor) {
          throw new Error("Dense Cholesky heat factor was not initialized.");
        }

        const sourceIndex = np.array(source, { dtype: np.int32 });
        const heat = solveWithFactor(heatFactor, sourceVector(sourceIndex));
        const divergence = centerVector(computeDivergence(heat));
        const phi = solveWithFactor(poissonFactor, divergence);
        return {
          values: (await phi.data()) as Float32Array,
          factorMs: lastFactorMs,
        };
      },
      dispose() {
        heatFactor?.dispose();
        poissonFactor.dispose();
        faceIndices.dispose();
        gradBasis.dispose();
        incidentFaces.dispose();
        incidentAreas.dispose();
        incidentGrad.dispose();
        vertexIds.dispose();
        sourceVector.dispose();
        computeDivergence.dispose();
        centerVector.dispose();
      },
    };
  }

  async function makeSolver(
    data: MeshData,
    mode: SolverMode,
    initialHeatScale: number,
  ): Promise<HeatSolver> {
    if (mode === "cholesky") {
      return makeDenseCholeskySolver(data, initialHeatScale);
    }
    return makeCgSolver(data);
  }

  function normalizeDistance(
    values: Float32Array,
    source: number,
  ): Float32Array {
    let mean = 0;
    for (const value of values) mean += value;
    mean /= values.length;

    const sign = values[source] > mean ? -1 : 1;
    let min = Infinity;
    let max = -Infinity;
    for (const value of values) {
      const signed = sign * value;
      min = Math.min(min, signed);
      max = Math.max(max, signed);
    }
    const span = Math.max(max - min, EPS);
    distanceSpan = span;

    const normalized = new Float32Array(values.length);
    for (let i = 0; i < values.length; i++) {
      normalized[i] = (sign * values[i] - min) / span;
    }
    return normalized;
  }

  function palette(t: number): [number, number, number] {
    const stops: [number, number, number][] = [
      [37, 69, 112],
      [35, 132, 132],
      [108, 170, 91],
      [236, 188, 84],
      [218, 91, 80],
    ];
    const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    const f = x - i;
    return [
      Math.round(stops[i][0] * (1 - f) + stops[i + 1][0] * f),
      Math.round(stops[i][1] * (1 - f) + stops[i + 1][1] * f),
      Math.round(stops[i][2] * (1 - f) + stops[i + 1][2] * f),
    ];
  }

  function setDefaultCameraView() {
    if (!orbitCamera) return;
    orbitCamera.position.set(-2.8, 3.8, 2.2);
    orbitCamera.zoom = 1;
    orbitCamera.lookAt(0, 0, 0);
    orbitCamera.updateProjectionMatrix();
    orbitCamera.updateMatrixWorld(true);
    if (orbitControls) {
      orbitControls.target.set(0, 0, 0);
      orbitControls.update();
    }
  }

  function setupOrbitControls() {
    if (!canvas || orbitControls) return;
    const aspect = CANVAS_WIDTH / CANVAS_HEIGHT;
    const viewHeight = 3.25;
    orbitCamera = new OrthographicCamera(
      (-viewHeight * aspect) / 2,
      (viewHeight * aspect) / 2,
      viewHeight / 2,
      -viewHeight / 2,
      0.1,
      100,
    );
    orbitCamera.up.set(0, 0, 1);
    orbitControls = new OrbitControls(orbitCamera, canvas);
    orbitControls.enableDamping = false;
    orbitControls.enablePan = false;
    orbitControls.screenSpacePanning = false;
    orbitControls.minPolarAngle = 0.08;
    orbitControls.maxPolarAngle = Math.PI - 0.08;
    orbitControls.minZoom = 0.72;
    orbitControls.maxZoom = 2.8;
    orbitControls.rotateSpeed = 0.62;
    orbitControls.zoomSpeed = 0.7;
    orbitControls.addEventListener("start", () => {
      isDraggingView = true;
    });
    orbitControls.addEventListener("change", render);
    orbitControls.addEventListener("end", () => {
      isDraggingView = false;
      render();
    });
    setDefaultCameraView();
    orbitControls.saveState();
  }

  function updateProjection(data: MeshData) {
    const camera = orbitCamera;
    if (!camera) return;
    if (viewDepth.length !== data.vertexCount) {
      viewDepth = new Float32Array(data.vertexCount);
    }
    if (faceDepth.length !== data.faceCount) {
      faceDepth = new Float32Array(data.faceCount);
      faceOrder = Array.from({ length: data.faceCount }, (_, index) => index);
    }

    const aspect = CANVAS_WIDTH / CANVAS_HEIGHT;
    const viewHeight = 3.25;
    camera.left = (-viewHeight * aspect) / 2;
    camera.right = (viewHeight * aspect) / 2;
    camera.top = viewHeight / 2;
    camera.bottom = -viewHeight / 2;
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    const projected = new Vector3();
    const viewSpace = new Vector3();

    for (let v = 0; v < data.vertexCount; v++) {
      projected.set(
        data.positions[v * 3],
        data.positions[v * 3 + 1],
        data.positions[v * 3 + 2],
      );
      viewSpace.copy(projected).applyMatrix4(camera.matrixWorldInverse);
      viewDepth[v] = viewSpace.z;
      projected.project(camera);
      data.projected[v * 2] = ((projected.x + 1) * CANVAS_WIDTH) / 2;
      data.projected[v * 2 + 1] = ((1 - projected.y) * CANVAS_HEIGHT) / 2;
    }

    for (let f = 0; f < data.faceCount; f++) {
      const a = data.faces[f * 3];
      const b = data.faces[f * 3 + 1];
      const c = data.faces[f * 3 + 2];
      faceDepth[f] = (viewDepth[a] + viewDepth[b] + viewDepth[c]) / 3;
      faceOrder[f] = f;
    }
    faceOrder.sort((a, b) => faceDepth[a] - faceDepth[b]);
  }

  function fillTriangle(
    ctx: CanvasRenderingContext2D,
    a: number,
    b: number,
    c: number,
    color: string,
  ) {
    if (!mesh) return;
    const p = mesh.projected;
    ctx.beginPath();
    ctx.moveTo(p[a * 2], p[a * 2 + 1]);
    ctx.lineTo(p[b * 2], p[b * 2 + 1]);
    ctx.lineTo(p[c * 2], p[c * 2 + 1]);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }

  function strokeTriangleEdges(
    ctx: CanvasRenderingContext2D,
    a: number,
    b: number,
    c: number,
  ) {
    if (!mesh) return;
    const p = mesh.projected;
    ctx.beginPath();
    ctx.moveTo(p[a * 2], p[a * 2 + 1]);
    ctx.lineTo(p[b * 2], p[b * 2 + 1]);
    ctx.lineTo(p[c * 2], p[c * 2 + 1]);
    ctx.closePath();
    ctx.lineWidth = 0.45;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.22)";
    ctx.stroke();
  }

  function contourPoint(
    a: number,
    b: number,
    va: number,
    vb: number,
    level: number,
  ): [number, number] | null {
    if (
      !mesh ||
      (level < va && level < vb) ||
      (level > va && level > vb) ||
      va === vb
    ) {
      return null;
    }
    const t = (level - va) / (vb - va);
    const p = mesh.projected;
    return [
      p[a * 2] * (1 - t) + p[b * 2] * t,
      p[a * 2 + 1] * (1 - t) + p[b * 2 + 1] * t,
    ];
  }

  function render() {
    if (!canvas || !mesh) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    updateProjection(mesh);

    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    ctx.fillStyle = "#f6f5ef";
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    const values = distance;
    const heights = mesh.heights;
    let minH = Infinity;
    let maxH = -Infinity;
    for (const h of heights) {
      minH = Math.min(minH, h);
      maxH = Math.max(maxH, h);
    }
    const hSpan = Math.max(maxH - minH, EPS);

    for (const f of faceOrder) {
      const a = mesh.faces[f * 3];
      const b = mesh.faces[f * 3 + 1];
      const c = mesh.faces[f * 3 + 2];
      const value = values ? (values[a] + values[b] + values[c]) / 3 : 0.5;
      const height =
        ((heights[a] + heights[b] + heights[c]) / 3 - minH) / hSpan;
      const [r0, g0, b0] = palette(values ? 1 - value : 0.5);
      const light = 0.78 + 0.28 * height;
      fillTriangle(
        ctx,
        a,
        b,
        c,
        `rgb(${Math.round(r0 * light)}, ${Math.round(g0 * light)}, ${Math.round(b0 * light)})`,
      );
      strokeTriangleEdges(ctx, a, b, c);

      if (values) {
        ctx.lineWidth = 1.05;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.72)";
        const ids = [
          mesh.faces[f * 3],
          mesh.faces[f * 3 + 1],
          mesh.faces[f * 3 + 2],
        ];
        const vals = [values[ids[0]], values[ids[1]], values[ids[2]]];
        for (let level = 0.08; level < 1; level += 0.08) {
          const points = [
            contourPoint(ids[0], ids[1], vals[0], vals[1], level),
            contourPoint(ids[1], ids[2], vals[1], vals[2], level),
            contourPoint(ids[2], ids[0], vals[2], vals[0], level),
          ].filter((p): p is [number, number] => p !== null);
          if (points.length === 2) {
            ctx.beginPath();
            ctx.moveTo(points[0][0], points[0][1]);
            ctx.lineTo(points[1][0], points[1][1]);
            ctx.stroke();
          }
        }
      }
    }

    const sx = mesh.projected[sourceIndex * 2];
    const sy = mesh.projected[sourceIndex * 2 + 1];
    if (isVertexVisible(mesh, sourceIndex)) {
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#1f2937";
      ctx.fillStyle = "#fff8e8";
      ctx.beginPath();
      ctx.arc(sx, sy, 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx - 15, sy);
      ctx.lineTo(sx + 15, sy);
      ctx.moveTo(sx, sy - 15);
      ctx.lineTo(sx, sy + 15);
      ctx.stroke();
    }
  }

  function pointInTriangle(
    px: number,
    py: number,
    ax: number,
    ay: number,
    bx: number,
    by: number,
    cx: number,
    cy: number,
  ) {
    const v0x = cx - ax;
    const v0y = cy - ay;
    const v1x = bx - ax;
    const v1y = by - ay;
    const v2x = px - ax;
    const v2y = py - ay;
    const dot00 = v0x * v0x + v0y * v0y;
    const dot01 = v0x * v1x + v0y * v1y;
    const dot02 = v0x * v2x + v0y * v2y;
    const dot11 = v1x * v1x + v1y * v1y;
    const dot12 = v1x * v2x + v1y * v2y;
    const denom = dot00 * dot11 - dot01 * dot01;
    if (Math.abs(denom) < EPS) return false;
    const invDenom = 1 / denom;
    const u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    const v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    return u >= -0.002 && v >= -0.002 && u + v <= 1.002;
  }

  function faceAtPoint(data: MeshData, x: number, y: number): number | null {
    const p = data.projected;
    for (let i = faceOrder.length - 1; i >= 0; i--) {
      const face = faceOrder[i];
      const a = data.faces[face * 3];
      const b = data.faces[face * 3 + 1];
      const c = data.faces[face * 3 + 2];
      if (
        pointInTriangle(
          x,
          y,
          p[a * 2],
          p[a * 2 + 1],
          p[b * 2],
          p[b * 2 + 1],
          p[c * 2],
          p[c * 2 + 1],
        )
      ) {
        return face;
      }
    }
    return null;
  }

  function nearestVertexOnFace(
    data: MeshData,
    face: number,
    x: number,
    y: number,
  ) {
    const p = data.projected;
    let best = data.faces[face * 3];
    let bestDist = Infinity;
    for (let local = 0; local < 3; local++) {
      const vertex = data.faces[face * 3 + local];
      const dx = p[vertex * 2] - x;
      const dy = p[vertex * 2 + 1] - y;
      const d = dx * dx + dy * dy;
      if (d < bestDist) {
        best = vertex;
        bestDist = d;
      }
    }
    return best;
  }

  function pickVisibleVertex(data: MeshData, x: number, y: number): number {
    const face = faceAtPoint(data, x, y);
    if (face !== null) return nearestVertexOnFace(data, face, x, y);
    return nearestVertexInData(data, x, y);
  }

  function faceIncludesVertex(data: MeshData, face: number, vertex: number) {
    return (
      data.faces[face * 3] === vertex ||
      data.faces[face * 3 + 1] === vertex ||
      data.faces[face * 3 + 2] === vertex
    );
  }

  function isVertexVisible(data: MeshData, vertex: number) {
    const x = data.projected[vertex * 2];
    const y = data.projected[vertex * 2 + 1];
    const face = faceAtPoint(data, x, y);
    if (face === null || faceIncludesVertex(data, face, vertex)) return true;
    return faceDepth[face] <= viewDepth[vertex] + 0.025;
  }

  function nearestVertexInData(data: MeshData, x: number, y: number): number {
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < data.vertexCount; i++) {
      const dx = data.projected[i * 2] - x;
      const dy = data.projected[i * 2 + 1] - y;
      const d = dx * dx + dy * dy;
      if (d < bestDist) {
        best = i;
        bestDist = d;
      }
    }
    return best;
  }

  function closestVertexToCamera(data: MeshData): number {
    let best = 0;
    let bestDepth = -Infinity;
    for (let v = 0; v < data.vertexCount; v++) {
      const x = data.projected[v * 2];
      const y = data.projected[v * 2 + 1];
      const visibleOnCanvas =
        x >= -CANVAS_PAD &&
        x <= CANVAS_WIDTH + CANVAS_PAD &&
        y >= -CANVAS_PAD &&
        y <= CANVAS_HEIGHT + CANVAS_PAD;
      if (visibleOnCanvas && viewDepth[v] > bestDepth) {
        best = v;
        bestDepth = viewDepth[v];
      }
    }
    return best;
  }

  function canvasPoint(event: MouseEvent | PointerEvent): [number, number] {
    const rect = canvas.getBoundingClientRect();
    return [
      ((event.clientX - rect.left) / rect.width) * CANVAS_WIDTH,
      ((event.clientY - rect.top) / rect.height) * CANVAS_HEIGHT,
    ];
  }

  function resetView() {
    setDefaultCameraView();
    render();
  }

  function handleCanvasPointerDown(event: PointerEvent) {
    if (!canvas || !mesh || loadingMesh) return;
    clickPointerId = event.pointerId;
    clickStartClientX = event.clientX;
    clickStartClientY = event.clientY;
    suppressNextClick = false;
  }

  function handleCanvasPointerMove(event: PointerEvent) {
    if (clickPointerId !== event.pointerId || !mesh) return;
    const totalDx = event.clientX - clickStartClientX;
    const totalDy = event.clientY - clickStartClientY;
    if (Math.hypot(totalDx, totalDy) > 4) {
      suppressNextClick = true;
      isDraggingView = true;
    }
  }

  function finishCanvasPointer(event: PointerEvent) {
    if (clickPointerId !== event.pointerId) return;
    clickPointerId = null;
    isDraggingView = false;
    render();
  }

  function handleCanvasPointerCancel(event: PointerEvent) {
    if (clickPointerId !== event.pointerId) return;
    clickPointerId = null;
    isDraggingView = false;
    render();
  }

  function handleCanvasClick(event: MouseEvent) {
    if (suppressNextClick) {
      suppressNextClick = false;
      return;
    }
    if (!canvas || !mesh || solving || loadingMesh) return;
    const [x, y] = canvasPoint(event);
    sourceIndex = pickVisibleVertex(mesh, x, y);
    void solveCurrent();
  }

  function randomSource() {
    if (!mesh || solving || loadingMesh) return;
    sourceIndex = Math.floor(Math.random() * mesh.vertexCount);
    void solveCurrent();
  }

  function queueSolve() {
    if (!initialized || loadingMesh) return;
    if (pendingTimer !== undefined) window.clearTimeout(pendingTimer);
    pendingTimer = window.setTimeout(() => {
      pendingTimer = undefined;
      void solveCurrent();
    }, 120);
  }

  async function rebuildSolver() {
    if (!mesh || loadingMesh) return;
    const built = mesh;
    const serial = ++meshSerial;
    solveSerial++;
    loadingMesh = true;
    solving = false;
    initialized = false;
    loadError = "";
    distance = null;
    distanceSpan = 0;
    solveMs = 0;
    factorMs = 0;

    const oldSolver = solver;
    solver = null;
    oldSolver?.dispose();

    try {
      const nextSolver = await makeSolver(built, solverMode, heatScale);
      if (serial !== meshSerial) {
        nextSolver.dispose();
        return;
      }
      solver = nextSolver;
      factorMs = nextSolver.setupMs;
      initialized = true;
      loadingMesh = false;
      render();
      await solveCurrent();
    } catch (error) {
      if (serial === meshSerial) {
        loadError = error instanceof Error ? error.message : String(error);
      }
    } finally {
      if (serial === meshSerial) loadingMesh = false;
    }
  }

  async function solveCurrent() {
    if (!solver || !mesh || loadingMesh) return;
    const serial = ++solveSerial;
    solving = true;
    const start = performance.now();
    try {
      const result = await solver.solve(sourceIndex, heatScale, cgIterations);
      if (serial !== solveSerial) return;
      if (result.factorMs !== undefined) factorMs = result.factorMs;
      distance = normalizeDistance(result.values, sourceIndex);
      solveMs = performance.now() - start;
      render();
    } catch (error) {
      if (serial === solveSerial) {
        loadError = error instanceof Error ? error.message : String(error);
      }
    } finally {
      if (serial === solveSerial) solving = false;
    }
  }

  async function loadMesh(sourceId = selectedMeshId) {
    const source =
      meshSources.find((item) => item.id === sourceId) ?? meshSources[0];
    selectedMeshId = source.id;
    const serial = ++meshSerial;
    solveSerial++;
    if (pendingTimer !== undefined) {
      window.clearTimeout(pendingTimer);
      pendingTimer = undefined;
    }

    loadingMesh = true;
    solving = false;
    initialized = false;
    loadError = "";
    distance = null;
    distanceSpan = 0;
    solveMs = 0;
    factorMs = 0;

    const oldSolver = solver;
    solver = null;
    oldSolver?.dispose();
    let assignedMesh = false;

    try {
      const raw = await loadRawMesh(source);
      if (serial !== meshSerial) return;

      const built = buildMesh(raw, source);
      if (serial !== meshSerial) return;

      mesh = built;
      assignedMesh = true;
      heatScale = source.heatScale;
      updateProjection(built);
      sourceIndex = closestVertexToCamera(built);
      solver = await makeSolver(built, solverMode, heatScale);
      if (serial !== meshSerial) {
        solver.dispose();
        return;
      }
      factorMs = solver.setupMs;
      initialized = true;
      loadingMesh = false;
      render();
      await solveCurrent();
    } catch (error) {
      if (serial === meshSerial) {
        if (!assignedMesh) mesh = null;
        loadError = error instanceof Error ? error.message : String(error);
      }
    } finally {
      if (serial === meshSerial) loadingMesh = false;
    }
  }

  onMount(() => {
    let disposed = false;
    setupOrbitControls();

    (async () => {
      const available = await init("webgpu");
      if (available.includes("webgpu")) {
        defaultDevice("webgpu");
      }
      deviceName = defaultDevice();

      if (!disposed) await loadMesh(selectedMeshId);
    })();

    return () => {
      disposed = true;
      meshSerial++;
      solveSerial++;
      if (pendingTimer !== undefined) window.clearTimeout(pendingTimer);
      solver?.dispose();
      orbitControls?.dispose();
      orbitControls = null;
      orbitCamera = null;
    };
  });
</script>

<svelte:head>
  <title>Heat Method Geodesics - jax-js</title>
</svelte:head>

<main class="min-h-screen bg-[#f6f5ef] text-slate-900 font-tiktok">
  <header
    class="max-w-screen-2xl mx-auto px-5 sm:px-8 py-4 flex items-center justify-between gap-4"
  >
    <a href={resolve("/")} class="text-sm font-medium hover:text-primary"
      >jax-js</a
    >
    <div class="flex items-center gap-2 text-xs text-slate-600">
      <Gauge size={16} />
      <span>{deviceName}</span>
    </div>
  </header>

  <section
    class="max-w-screen-2xl mx-auto px-5 sm:px-8 pb-8 grid xl:grid-cols-[minmax(0,1fr)_320px] gap-5"
  >
    <div class="min-w-0">
      <div class="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 class="text-2xl sm:text-3xl font-medium tracking-normal">
            Heat Method Geodesics
          </h1>
          <p class="text-sm text-slate-600 mt-1">
            Surface distance on a triangulated mesh, after
            <a
              class="paper-link"
              href="https://arxiv.org/pdf/1204.6216"
              target="_blank"
              rel="noreferrer"
            >
              Geodesics in Heat
              <ExternalLink size={13} />
            </a>
            .
          </p>
        </div>
        <div class="flex items-center gap-2">
          <button
            class="icon-button"
            aria-label="Reset view"
            title="Reset view"
            onclick={resetView}
            disabled={!mesh}
          >
            <RotateCcw size={18} />
          </button>
          <button
            class="icon-button"
            aria-label="Random source"
            title="Random source"
            onclick={randomSource}
            disabled={!initialized || solving || loadingMesh}
          >
            <Shuffle size={18} />
          </button>
          <button
            class="command-button"
            onclick={() => void solveCurrent()}
            disabled={!initialized || solving || loadingMesh}
          >
            <RefreshCw
              size={17}
              class={solving || loadingMesh ? "animate-spin" : ""}
            />
            <span
              >{loadingMesh
                ? solverMode === "cholesky"
                  ? "Factoring"
                  : "Loading"
                : solving
                  ? "Solving"
                  : "Recompute"}</span
            >
          </button>
        </div>
      </div>

      <div class="canvas-shell">
        <canvas
          bind:this={canvas}
          width={CANVAS_WIDTH}
          height={CANVAS_HEIGHT}
          class:dragging={isDraggingView}
          onpointerdown={handleCanvasPointerDown}
          onpointermove={handleCanvasPointerMove}
          onpointerup={finishCanvasPointer}
          onpointercancel={handleCanvasPointerCancel}
          onclick={handleCanvasClick}
          aria-label="Heat method geodesic distance field"
        ></canvas>
        {#if loadingMesh}
          <div class="solve-badge">
            {solverMode === "cholesky" ? "factorizing" : "loading mesh"}
          </div>
        {:else if solving}
          <div class="solve-badge">
            {solverMode === "cholesky" ? "triangular solves" : "running CG"}
          </div>
        {/if}
      </div>
    </div>

    <aside class="control-panel">
      <label class="control">
        <span>solver</span>
        <select
          bind:value={solverMode}
          onchange={() => void rebuildSolver()}
          disabled={loadingMesh || solving || !mesh}
        >
          <option value="cholesky">Dense Cholesky</option>
          <option value="cg">Iterative CG</option>
        </select>
        <output>
          {solverMode === "cholesky"
            ? "Precompute dense factors for fast source changes."
            : "Iterative matvec solves."}
        </output>
      </label>

      <label class="control">
        <span>mesh</span>
        <select
          bind:value={selectedMeshId}
          onchange={() => void loadMesh(selectedMeshId)}
          disabled={loadingMesh || solving}
        >
          {#each meshSources as source}
            <option value={source.id}>{source.title}</option>
          {/each}
        </select>
        <output>{mesh?.description ?? selectedSource().description}</output>
      </label>

      {#if loadError}
        <div class="error-message">{loadError}</div>
      {/if}

      <div class="metric-row">
        <div>
          <div class="metric-label">source</div>
          <div class="metric-value">{sourceIndex}</div>
        </div>
        <Crosshair size={20} class="text-slate-500" />
      </div>

      <div class="metric-row">
        <div>
          <div class="metric-label">solve</div>
          <div class="metric-value">
            {solveMs ? `${solveMs.toFixed(1)} ms` : "-"}
          </div>
        </div>
        <MousePointer2 size={20} class="text-slate-500" />
      </div>

      {#if solverMode === "cholesky"}
        <div class="metric-row">
          <div>
            <div class="metric-label">factor</div>
            <div class="metric-value">
              {factorMs ? `${factorMs.toFixed(1)} ms` : "-"}
            </div>
          </div>
          <Gauge size={20} class="text-slate-500" />
        </div>
      {/if}

      <label class="control">
        <span>heat time</span>
        <input
          type="range"
          min="0.35"
          max="2.8"
          step="0.05"
          bind:value={heatScale}
          oninput={queueSolve}
          disabled={!initialized || loadingMesh || solving}
        />
        <output>{heatScale.toFixed(2)} h^2</output>
      </label>

      <label class="control">
        <span>CG iterations</span>
        <input
          type="range"
          min="24"
          max="120"
          step="2"
          bind:value={cgIterations}
          oninput={queueSolve}
          disabled={!initialized ||
            loadingMesh ||
            solving ||
            solverMode === "cholesky"}
        />
        <output>{cgIterations}</output>
      </label>

      <div class="stats">
        <div>
          <span>vertices</span>
          <strong>{mesh?.vertexCount ?? "-"}</strong>
        </div>
        <div>
          <span>faces</span>
          <strong>{mesh?.faceCount ?? "-"}</strong>
        </div>
        <div>
          <span>neighbor slots</span>
          <strong>{mesh?.maxDegree ?? "-"}</strong>
        </div>
        <div>
          <span>distance span</span>
          <strong>{distanceSpan ? distanceSpan.toFixed(3) : "-"}</strong>
        </div>
      </div>

      <div class="legend" aria-hidden="true">
        <div></div>
        <div class="flex justify-between text-xs text-slate-500 mt-2">
          <span>near</span>
          <span>far</span>
        </div>
      </div>
    </aside>
  </section>
</main>

<style lang="postcss">
  @reference "$app.css";

  .canvas-shell {
    @apply relative overflow-hidden border border-slate-300 bg-[#f6f5ef] shadow-sm;
    border-radius: 8px;
  }

  canvas {
    @apply block w-full h-auto touch-none cursor-grab;
  }

  canvas.dragging {
    @apply cursor-grabbing;
  }

  .solve-badge {
    @apply absolute right-4 top-4 bg-slate-950/80 text-white text-xs px-3 py-1.5;
    border-radius: 999px;
  }

  .control-panel {
    @apply border border-slate-300 bg-white/70 p-4 h-fit shadow-sm space-y-4;
    border-radius: 8px;
  }

  .metric-row {
    @apply flex items-center justify-between border-b border-slate-200 pb-3;
  }

  .metric-label {
    @apply text-xs uppercase tracking-wide text-slate-500;
  }

  .metric-value {
    @apply text-xl font-medium tabular-nums;
  }

  .control {
    @apply grid gap-2 text-sm;
  }

  .control span {
    @apply text-slate-600;
  }

  .control select {
    @apply min-h-10 w-full border border-slate-300 bg-white px-3 text-sm text-slate-900;
    border-radius: 6px;
  }

  .control input {
    @apply w-full accent-primary;
  }

  .control output {
    @apply text-xs tabular-nums text-slate-500;
  }

  .paper-link {
    @apply inline-flex items-center gap-1 text-slate-900 underline decoration-slate-300 underline-offset-2 hover:text-primary hover:decoration-primary;
  }

  .error-message {
    @apply border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800;
    border-radius: 6px;
  }

  .stats {
    @apply grid grid-cols-2 gap-3 border-t border-slate-200 pt-4;
  }

  .stats div {
    @apply grid gap-0.5;
  }

  .stats span {
    @apply text-xs text-slate-500;
  }

  .stats strong {
    @apply text-sm font-medium tabular-nums;
  }

  .legend > div:first-child {
    @apply h-3;
    border-radius: 999px;
    background: linear-gradient(
      90deg,
      rgb(218, 91, 80),
      rgb(236, 188, 84),
      rgb(108, 170, 91),
      rgb(35, 132, 132),
      rgb(37, 69, 112)
    );
  }

  .icon-button,
  .command-button {
    @apply inline-flex items-center justify-center border border-slate-300 bg-white/80 hover:bg-white disabled:opacity-50 disabled:hover:bg-white/80 active:scale-[0.98];
    border-radius: 6px;
    min-height: 38px;
  }

  .icon-button {
    @apply w-10;
  }

  .command-button {
    @apply gap-2 px-3 text-sm font-medium;
  }
</style>
