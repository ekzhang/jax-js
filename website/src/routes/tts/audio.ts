/** Audio playback utilities for TTS output. */

const SAMPLE_RATE = 24000; // 24kHz sample rate for Mimi codec

/** Play a single buffer of PCM samples (Float32Array in range [-1, 1]). */
export async function playPcm(samples: Float32Array): Promise<void> {
  const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });

  const buffer = audioCtx.createBuffer(1, samples.length, SAMPLE_RATE);
  buffer.getChannelData(0).set(samples);

  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);

  return new Promise((resolve) => {
    source.onended = () => {
      audioCtx.close();
      resolve();
    };
    source.start();
  });
}
