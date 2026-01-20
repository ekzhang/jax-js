// Audio playback utilities for TTS output.

const SAMPLE_RATE = 24000; // 24kHz sample rate for Mimi codec

/**
 * Creates a streaming audio player for playing PCM chunks as they're generated.
 * Each chunk is scheduled to play immediately after the previous one.
 */
export function createStreamingPlayer() {
  const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  let nextStartTime = audioCtx.currentTime;
  let lastEndedPromise: Promise<void> = Promise.resolve();

  return {
    /** Play a chunk of PCM samples (Float32Array in range [-1, 1]). */
    playChunk(samples: Float32Array) {
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

    /** Resume audio context if suspended (required after user interaction). */
    async resume() {
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }
    },

    /** Wait for all queued audio to finish, then close the audio context. */
    async close() {
      await lastEndedPromise;
      await audioCtx.close();
    },

    /** Get the underlying AudioContext. */
    get context() {
      return audioCtx;
    },
  };
}
