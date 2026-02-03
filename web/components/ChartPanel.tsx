import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface ChartPanelProps {
  chart?: Record<string, any> | null;
  title?: string;
}

export default function ChartPanel({ chart, title }: ChartPanelProps) {
  const hasChart = Boolean(chart && Array.isArray(chart.data) && chart.layout);
  const ink = "#0f172a";
  const grid = "rgba(15, 23, 42, 0.12)";
  const zero = "rgba(15, 23, 42, 0.2)";

  return (
    <div className="p-4 rounded-2xl bg-white/50 border border-white/80 backdrop-blur-sm w-full h-full overflow-hidden">
      {hasChart ? (
        <Plot
          data={chart?.data}
          layout={{
            ...chart?.layout,
            autosize: true,
            paper_bgcolor: "rgba(255,255,255,0.5)",
            plot_bgcolor: "rgba(255,255,255,0)",
            font: { family: "var(--font-body), sans-serif", color: ink, size: 12 },
            xaxis: {
              ...(chart?.layout?.xaxis || {}),
              gridcolor: grid,
              zerolinecolor: zero,
              automargin: true,
            },
            yaxis: {
              ...(chart?.layout?.yaxis || {}),
              gridcolor: grid,
              zerolinecolor: zero,
              automargin: true,
            },
            margin: { l: 80, r: 80, t: 60, b: 80 },
          }}
          config={{ displayModeBar: false, responsive: true, staticPlot: false }}
          style={{ width: "100%", height: "100%", minHeight: 360 }}
          useResizeHandler
        />
      ) : null}
    </div>
  );
}
