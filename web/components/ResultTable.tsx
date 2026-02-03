interface ResultTableProps {
  rows: Record<string, any>[];
  variant?: "card" | "flat";
  showTitle?: boolean;
}

export default function ResultTable({ rows, variant = "card", showTitle = true }: ResultTableProps) {
  const isCard = variant === "card";

  if (!rows || rows.length === 0) {
    return (
      <div className="rounded-xl border border-white/50 bg-white/30 p-4">
        <p className="text-sm text-slate-600">No data available</p>
      </div>
    );
  }

  const columns = Object.keys(rows[0] || {});

  return (
    <div className={`rounded-xl border border-white/50 ${isCard ? "bg-white/50" : "bg-white/30"} overflow-hidden`}>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/50 bg-white/40">
              {columns.map((col) => (
                <th key={col} className="text-left px-4 py-3 text-xs font-semibold text-slate-700">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr key={idx} className="border-b border-white/30 hover:bg-white/40 transition-colors">
                {columns.map((col) => {
                  let cellValue = row[col];
                  
                  if (typeof cellValue === "number") {
                    // Check if it's an integer value
                    if (Number.isInteger(cellValue)) {
                      cellValue = cellValue.toString();
                    } else {
                      // For decimals, show up to 2 decimal places, removing trailing zeros
                      cellValue = parseFloat(cellValue.toFixed(2)).toString();
                    }
                  } else {
                    cellValue = String(cellValue ?? "");
                  }
                  
                  return (
                    <td key={col} className="px-4 py-2 text-slate-700">
                      {cellValue}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
