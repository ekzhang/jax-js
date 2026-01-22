// Audio playback utilities for TTS output.

const SAMPLE_RATE = 24000; // 24kHz sample rate for Mimi codec

export interface AudioPlayer {
  /** Play a chunk of PCM samples (Float32Array in range [-1, 1]). */
  playChunk(samples: Float32Array): void;

  /** Resume audio context if suspended (required after user interaction). */
  resume(): Promise<void>;

  /** Wait for all queued audio to finish, then close the audio context. */
  close(): Promise<void>;

  /** Get all played audio as a WAV blob. */
  toWav(): Blob;

  /** Get the underlying AudioContext. */
  readonly context: AudioContext;
}

/**
 * Creates a streaming audio player for playing PCM chunks as they're generated.
 * Each chunk is scheduled to play immediately after the previous one.
 */
/**
 * Converts PCM samples (Float32Array in range [-1, 1]) to a WAV file Blob.
 */
export function samplesToWav(
  samples: Float32Array,
  sampleRate = SAMPLE_RATE,
): Blob {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = samples.length * (bitsPerSample / 8);
  const headerSize = 44;

  const buffer = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");

  // fmt subchunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // subchunk size
  view.setUint16(20, 1, true); // audio format (PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data subchunk
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  // Write PCM samples as 16-bit integers
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    view.setInt16(offset, int16, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

export function createStreamingPlayer(): AudioPlayer {
  const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  let nextStartTime = audioCtx.currentTime;
  let lastEndedPromise: Promise<void> = Promise.resolve();
  const chunks: Float32Array[] = [];

  return {
    playChunk(samples: Float32Array) {
      chunks.push(samples.slice());

      const buffer = audioCtx.createBuffer(1, samples.length, SAMPLE_RATE);
      buffer.getChannelData(0).set(samples);

      const source = audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(audioCtx.destination);

      // Schedule this chunk right after the previous one
      const startTime = Math.max(nextStartTime, audioCtx.currentTime);
      source.start(startTime);
      nextStartTime = startTime + buffer.duration;

      // Track when this source finishes playing
      lastEndedPromise = new Promise((resolve) => {
        source.onended = () => resolve();
      });
    },

    async resume() {
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }
    },

    async close() {
      await lastEndedPromise;
      await audioCtx.close();
    },

    toWav() {
      const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
      const combined = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }
      return samplesToWav(combined);
    },

    get context() {
      return audioCtx;
    },
  };
}
