<script lang="ts">
  import { cachedFetch, safetensors, tokenizers } from "@jax-js/loaders";

  async function main() {
    const tokenizer = await tokenizers.get("clip");
    console.log(tokenizer.encode("hello world"));

    const obj = safetensors.parse(
      await cachedFetch(
        "https://huggingface.co/ekzhang/jax-js-models/resolve/main/mobileclip2-s0.safetensors",
      ),
    );
    console.log(obj);
  }
</script>

<button onclick={main}>go</button>
