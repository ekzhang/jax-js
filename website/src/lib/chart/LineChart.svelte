<script lang="ts">
  import * as Plot from "@observablehq/plot";
  import { max, min } from "d3-array";
  import { schemeTableau10 } from "d3-scale-chromatic";

  const colors = schemeTableau10;

  interface Props {
    title?: string;
    data: any[];
    x: string;
    y: string | string[];
  }

  let { title, data, x, y }: Props = $props();

  let div: HTMLElement | undefined = $state();
  let divWidth: number = $state(0);
  let divHeight: number = $state(0);

  let ys = $derived(Array.isArray(y) ? y : [y]);
  let disabled: Record<string, boolean> = $state({});
  let ysEnabled = $derived(ys.filter((yField) => !disabled[yField]));

  $effect(() => {
    div?.firstChild?.remove();

    const minX = min(data, (d: any) => d[x] as number) || 0;
    const maxX = Math.max(max(data, (d: any) => d[x] as number) || 0, 10);

    const chart = Plot.plot({
      width: divWidth,
      height: divHeight,
      marginTop: 12,
      marks: [
        Plot.ruleY([0]),
        Plot.gridY({
          strokeDasharray: "1,2",
          strokeOpacity: 0.25,
        }),
        ys.flatMap((yField, i) => {
          if (disabled[yField]) return [];
          return [
            Plot.lineY(data, {
              x,
              y: yField,
              stroke: colors[i % colors.length],
              tip: true,
            }),
            // Only show dots for small datasets to avoid clutter.
            data.length <= 50
              ? Plot.dotY(data, {
                  x,
                  y: yField,
                  fill: colors[i % colors.length],
                  r: 2.5,
                })
              : null,
          ];
        }),
      ],
      x: {
        domain: [minX, maxX],
        inset: 10,
        ticks: 8,
        tickFormat: "d",
      },
      y: {
        insetTop: 10,
      },
    });

    div?.append(chart);
  });
</script>

<div class="w-full h-full flex flex-col p-1">
  <p class="text-sm shrink-0 truncate text-center -mb-2">{title}</p>
  <div
    class="w-full grow-1"
    bind:this={div}
    bind:clientWidth={divWidth}
    bind:clientHeight={divHeight}
    role="img"
  ></div>
  <!-- Legend with toggle-able series. -->
  <div class="shrink-0 flex gap-1 px-4 -mt-2">
    {#each ys as yField, i}
      <button
        class="flex items-center gap-1 px-1 py-0.5 rounded enabled:hover:bg-gray-100"
        onclick={() => (disabled[yField] = !disabled[yField])}
        disabled={ysEnabled.length <= 1 && !disabled[yField]}
      >
        <div
          class="w-3 h-3 rounded-sm"
          style:background-color={disabled[yField]
            ? "transparent"
            : colors[i % colors.length]}
          style:border="1px solid {colors[i % colors.length]}"
        ></div>
        <span class="text-xs text-gray-700">{yField}</span>
      </button>
    {/each}
  </div>
</div>
