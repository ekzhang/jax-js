export type DecodedAudio = {
  samples: Float32Array;
  duration: number;
  playbackUrl: string;
};

export type WhisperFeatures = {
  data: Float32Array;
  frames: number;
  mels: number;
};

const SAMPLE_RATE = 16_000;
const N_FFT = 400;
const HOP_LENGTH = 160;
const N_MELS = 80;
const N_FRAMES = 3000;
const N_SAMPLES = N_FRAMES * HOP_LENGTH;
const DFT_FACTOR = 20;

type MelFilters = {
  starts: Int16Array;
  weights: Float32Array[];
};

let melFiltersCache: MelFilters | null = null;
let stftPlanCache: {
  window: Float32Array;
  cos20: Float32Array;
  sin20: Float32Array;
  twiddleRe: Float32Array;
  twiddleIm: Float32Array;
} | null = null;

export async function decodeAudioFromUrl(url: string): Promise<DecodedAudio> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`audio fetch failed: ${response.status}`);

  const bytes = await response.arrayBuffer();
  let playbackUrl: string | null = URL.createObjectURL(
    new Blob([bytes], {
      type: response.headers.get("content-type") ?? "audio/wav",
    }),
  );
  const AudioContextCtor =
    window.AudioContext ??
    (window as typeof window & { webkitAudioContext?: typeof AudioContext })
      .webkitAudioContext;
  if (!AudioContextCtor) throw new Error("AudioContext is not available");

  const context = new AudioContextCtor();
  try {
    const buffer = await context.decodeAudioData(bytes.slice(0));
    const url = playbackUrl;
    playbackUrl = null;
    return {
      samples: resampleMono(buffer, SAMPLE_RATE),
      duration: buffer.duration,
      playbackUrl: url,
    };
  } finally {
    if (playbackUrl) URL.revokeObjectURL(playbackUrl);
    await context.close();
  }
}

export function whisperLogMel(samples: Float32Array): WhisperFeatures {
  const inputLength = Math.min(samples.length, N_SAMPLES);
  const padded = new Float32Array(N_SAMPLES + N_FFT);
  padded.set(samples.subarray(0, inputLength), N_FFT / 2);

  const filters = melFilters();
  const plan = stftPlan();
  const stageRe = new Float32Array(N_FFT);
  const stageIm = new Float32Array(N_FFT);
  const power = new Float32Array(N_FFT / 2 + 1);
  const out = new Float32Array(N_MELS * N_FRAMES);
  const activeFrames = Math.min(
    N_FRAMES,
    Math.max(1, Math.ceil((inputLength + N_FFT / 2) / HOP_LENGTH)),
  );
  let maxLog = -Infinity;

  for (let frame = 0; frame < activeFrames; frame++) {
    dft400Power(padded, frame * HOP_LENGTH, plan, stageRe, stageIm, power);

    for (let mel = 0; mel < N_MELS; mel++) {
      let energy = 0;
      const start = filters.starts[mel];
      const weights = filters.weights[mel];
      for (let i = 0; i < weights.length; i++) {
        energy += weights[i] * power[start + i];
      }
      const logEnergy = Math.log10(Math.max(energy, 1e-10));
      out[mel * N_FRAMES + frame] = logEnergy;
      maxLog = Math.max(maxLog, logEnergy);
    }
  }

  if (activeFrames < N_FRAMES) {
    for (let mel = 0; mel < N_MELS; mel++) {
      out.fill(-10, mel * N_FRAMES + activeFrames, (mel + 1) * N_FRAMES);
    }
  }

  const floor = maxLog - 8;
  for (let i = 0; i < out.length; i++) {
    out[i] = (Math.max(out[i], floor) + 4) / 4;
  }

  return { data: out, frames: N_FRAMES, mels: N_MELS };
}

function resampleMono(buffer: AudioBuffer, targetRate: number): Float32Array {
  const outputLength = Math.max(1, Math.round(buffer.duration * targetRate));
  const output = new Float32Array(outputLength);
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, i) =>
    buffer.getChannelData(i),
  );
  const scale = buffer.sampleRate / targetRate;

  for (let i = 0; i < output.length; i++) {
    const source = i * scale;
    const left = Math.floor(source);
    const right = Math.min(left + 1, buffer.length - 1);
    const mix = source - left;
    let value = 0;
    for (const channel of channels) {
      value += channel[left] * (1 - mix) + channel[right] * mix;
    }
    output[i] = value / channels.length;
  }
  return output;
}

function stftPlan() {
  if (stftPlanCache) return stftPlanCache;

  const window = new Float32Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) {
    window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N_FFT);
  }

  const cos20 = new Float32Array(DFT_FACTOR * DFT_FACTOR);
  const sin20 = new Float32Array(DFT_FACTOR * DFT_FACTOR);
  for (let k = 0; k < DFT_FACTOR; k++) {
    for (let n = 0; n < DFT_FACTOR; n++) {
      const angle = (2 * Math.PI * k * n) / DFT_FACTOR;
      cos20[k * DFT_FACTOR + n] = Math.cos(angle);
      sin20[k * DFT_FACTOR + n] = Math.sin(angle);
    }
  }

  const twiddleRe = new Float32Array(N_FFT);
  const twiddleIm = new Float32Array(N_FFT);
  for (let n2 = 0; n2 < DFT_FACTOR; n2++) {
    for (let k1 = 0; k1 < DFT_FACTOR; k1++) {
      const angle = (2 * Math.PI * n2 * k1) / N_FFT;
      twiddleRe[n2 * DFT_FACTOR + k1] = Math.cos(angle);
      twiddleIm[n2 * DFT_FACTOR + k1] = Math.sin(angle);
    }
  }

  stftPlanCache = { window, cos20, sin20, twiddleRe, twiddleIm };
  return stftPlanCache;
}

function dft400Power(
  padded: Float32Array,
  start: number,
  plan: NonNullable<typeof stftPlanCache>,
  stageRe: Float32Array,
  stageIm: Float32Array,
  power: Float32Array,
) {
  const { window, cos20, sin20, twiddleRe, twiddleIm } = plan;

  for (let n2 = 0; n2 < DFT_FACTOR; n2++) {
    for (let k1 = 0; k1 < DFT_FACTOR; k1++) {
      let re = 0;
      let im = 0;
      const trigOffset = k1 * DFT_FACTOR;
      for (let n1 = 0; n1 < DFT_FACTOR; n1++) {
        const sampleOffset = n2 + DFT_FACTOR * n1;
        const value = padded[start + sampleOffset] * window[sampleOffset];
        re += value * cos20[trigOffset + n1];
        im -= value * sin20[trigOffset + n1];
      }

      const index = n2 * DFT_FACTOR + k1;
      const c = twiddleRe[index];
      const s = twiddleIm[index];
      stageRe[index] = re * c + im * s;
      stageIm[index] = im * c - re * s;
    }
  }

  for (let k2 = 0; k2 < DFT_FACTOR; k2++) {
    for (let k1 = 0; k1 < DFT_FACTOR; k1++) {
      const bin = k1 + DFT_FACTOR * k2;
      if (bin >= power.length) return;

      let re = 0;
      let im = 0;
      const trigOffset = k2 * DFT_FACTOR;
      for (let n2 = 0; n2 < DFT_FACTOR; n2++) {
        const index = n2 * DFT_FACTOR + k1;
        const a = stageRe[index];
        const b = stageIm[index];
        const c = cos20[trigOffset + n2];
        const s = sin20[trigOffset + n2];
        re += a * c + b * s;
        im += b * c - a * s;
      }
      power[bin] = re * re + im * im;
    }
  }
}

function melFilters(): MelFilters {
  if (melFiltersCache) return melFiltersCache;

  const fftFreqs = Array.from(
    { length: N_FFT / 2 + 1 },
    (_, i) => (SAMPLE_RATE / 2) * (i / (N_FFT / 2)),
  );
  const melMin = hzToMel(0);
  const melMax = hzToMel(SAMPLE_RATE / 2);
  const melPoints = Array.from({ length: N_MELS + 2 }, (_, i) =>
    melToHz(melMin + ((melMax - melMin) * i) / (N_MELS + 1)),
  );

  const starts = new Int16Array(N_MELS);
  const weights: Float32Array[] = [];
  for (let mel = 0; mel < N_MELS; mel++) {
    const left = melPoints[mel];
    const center = melPoints[mel + 1];
    const right = melPoints[mel + 2];
    const enorm = 2 / (right - left);
    let start = -1;
    const values: number[] = [];

    for (let bin = 0; bin < fftFreqs.length; bin++) {
      const freq = fftFreqs[bin];
      const lower = (freq - left) / (center - left);
      const upper = (right - freq) / (right - center);
      const weight = Math.max(0, Math.min(lower, upper)) * enorm;
      if (weight > 0) {
        if (start < 0) start = bin;
        values.push(weight);
      } else if (start >= 0) {
        break;
      }
    }

    starts[mel] = Math.max(0, start);
    weights.push(new Float32Array(values));
  }

  melFiltersCache = { starts, weights };
  return melFiltersCache;
}

function hzToMel(hz: number): number {
  const minLogHz = 1000;
  const minLogMel = 15;
  const logStep = Math.log(6.4) / 27;
  return hz < minLogHz
    ? (3 * hz) / 200
    : minLogMel + Math.log(hz / minLogHz) / logStep;
}

function melToHz(mel: number): number {
  const minLogHz = 1000;
  const minLogMel = 15;
  const logStep = Math.log(6.4) / 27;
  return mel < minLogMel
    ? (200 * mel) / 3
    : minLogHz * Math.exp(logStep * (mel - minLogMel));
}
