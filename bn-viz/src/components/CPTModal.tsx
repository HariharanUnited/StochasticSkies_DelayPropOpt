import React, { useState } from "react";

interface CPTModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  // Data can be a single table OR a map of "Tab Name" -> Table
  data: any;
  type: "node" | "edge";
  nodeId?: string;
}

const CPTModal: React.FC<CPTModalProps> = ({
  isOpen,
  onClose,
  title,
  data,
  type,
  nodeId,
}) => {
  const [activeTab, setActiveTab] = useState(type === "node" ? "time" : "main");

  React.useEffect(() => {
    if (isOpen && type === "node" && data) {
      // Default to "time" or "t" if available, else first key
      if (data["time"]) setActiveTab("time");
      else if (data["t"]) setActiveTab("t");
      else {
        const keys = Object.keys(data);
        if (keys.length > 0) setActiveTab(keys[0]);
      }
    } else {
      setActiveTab("main");
    }
  }, [isOpen, type, data]);

  if (!isOpen || !data) return null;

  let content = null;
  let tabs = null;

  if (type === "node") {
    const tabKeys = Object.keys(data);
    if (tabKeys.length > 1) {
      tabs = (
        <div className="mb-4 flex border-b border-slate-200/70">
          {tabKeys.map((k) => (
            <div
              key={k}
              onClick={() => setActiveTab(k)}
              className={`cursor-pointer px-4 py-2 text-sm font-bold transition-colors ${
                activeTab === k
                  ? "border-b-2 border-sky-400 bg-slate-100/80 text-sky-600"
                  : "text-slate-500 hover:bg-slate-100/80 hover:text-slate-700"
              }`}
            >
              {k === "t" || k === "time"
                ? `Time Model (${nodeId})`
                : k === "g" || k === "prior"
                  ? `Global Prior (${nodeId})`
                  : k}
            </div>
          ))}
        </div>
      );
    }
    content = renderTable(data[activeTab]);
  } else {
    // Edge
    content = renderTable(data);
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="glass-panel flex max-h-[80vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl shadow-2xl shadow-black/50"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-200/70 bg-slate-100/80 p-4">
          <h2 className="flex items-center gap-2 text-lg font-bold text-slate-900">
            {type === "node" ? "📊" : "🔗"} {title}
          </h2>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-600 transition-colors hover:bg-slate-200/80 hover:text-rose-600"
          >
            ✕
          </button>
        </div>

        <div className="custom-scrollbar flex-1 overflow-y-auto p-4">
          {tabs}
          <div className="cpt-content">{content}</div>
        </div>
      </div>
    </div>
  );
};

// Helper: Render Prob Table
function renderTable(tableData: any) {
  if (!tableData)
    return (
      <div className="p-4 text-center text-slate-500 italic">
        No Data Available
      </div>
    );

  const rows = Object.keys(tableData).sort();
  if (rows.length === 0)
    return (
      <div className="p-4 text-center text-slate-500 italic">Empty Table</div>
    );

  const firstVal = tableData[rows[0]];
  const isFlat = typeof firstVal === "number";

  if (isFlat) {
    // Flat Table (e.g. G Prior)
    return (
      <div className="overflow-x-auto rounded-xl border border-slate-200/70">
        <table className="w-full text-left text-sm text-slate-700">
          <thead className="bg-slate-100/80 text-xs font-bold text-slate-600 uppercase">
            <tr>
              <th className="border-b border-slate-200/70 px-4 py-3">State</th>
              <th className="border-b border-slate-200/70 px-4 py-3 text-center">
                Probability
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200/60">
            {rows.map((rKey) => {
              const val = tableData[rKey];
              const isHigh = val > 0.5;
              const isMed = val > 0.1;

              return (
                <tr key={rKey} className="transition-colors hover:bg-slate-100/80">
                  <td className="border-r border-slate-200/60 px-4 py-2 font-mono">
                    {rKey}
                  </td>
                  <td className="relative px-4 py-2 text-center">
                    {isMed && (
                      <div
                        className={`absolute inset-0 opacity-20 ${isHigh ? "bg-emerald-500" : "bg-emerald-500"}`}
                        style={{ width: `${val * 100}%` }}
                      />
                    )}
                    <span
                      className={`relative z-10 font-bold ${isHigh ? "text-emerald-600" : isMed ? "text-emerald-500" : "text-slate-600"}`}
                    >
                      {(val * 100).toFixed(1)}%
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }

  // 2D Table
  const binKeys = Object.keys(firstVal)
    .map(Number)
    .sort((a, b) => a - b);

  return (
    <div className="overflow-x-auto rounded-xl border border-slate-200/70">
      <table className="w-full text-left text-sm text-slate-700">
        <thead className="bg-slate-100/80 text-xs font-bold text-slate-600 uppercase">
          <tr>
            <th className="border-b border-slate-200/70 px-4 py-3">
              Parents State
            </th>
            {binKeys.map((b) => (
              <th
                key={b}
                className="border-b border-slate-200/70 px-4 py-3 text-center text-sky-600/80"
              >
                Bin {b}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-200/60">
          {rows.map((rKey) => (
            <tr key={rKey} className="transition-colors hover:bg-slate-100/80">
              <td className="border-r border-slate-200/60 px-4 py-2 font-mono whitespace-nowrap text-slate-600">
                {rKey.replace(/\|/g, ", ")}
              </td>
              {binKeys.map((b) => {
                const val = tableData[rKey][b];
                const isHigh = val > 0.5;
                const isMed = val > 0.1;

                return (
                  <td key={b} className="relative px-4 py-2 text-center">
                    {isMed && (
                      <div
                        className={`absolute inset-0 opacity-20 ${isHigh ? "bg-sky-500" : "bg-sky-500"}`}
                      />
                    )}
                    <span
                      className={`relative z-10 font-bold ${isHigh ? "text-sky-600" : isMed ? "text-sky-500" : "text-slate-500"}`}
                    >
                      {(val * 100).toFixed(1)}%
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default CPTModal;

