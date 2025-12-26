import { useState, useEffect } from "react";

interface ScenarioModalProps {
  isOpen: boolean;
  onClose: () => void;
  type: "node" | "edge";
  data: {
    id: string; // Node ID or Edge ID
    source?: string;
    target?: string;
    edgeType?: string; // 'ac', 'pilot', 'cabin', 'pax'
    tooltip?: string; // Payload from server
  };

  // Current State
  currentEvidence: Record<string, number | string>;
  currentTargets: Set<string>;

  // Result (Tool 1 Output)
  inferenceResult?: {
    prob_delay: number;
    expected_delay: number;
    metrics?: any;
    cause_breakdown?: Record<string, number>;
  };

  // Callbacks
  onSave: (changes: {
    evidence?: Record<string, number | string>;
    targets?: Set<string>;
  }) => void;
}

export default function ScenarioModal({
  isOpen,
  onClose,
  type,
  data,
  currentEvidence,
  currentTargets,
  inferenceResult,
  onSave,
}: ScenarioModalProps) {
  const [activeTab, setActiveTab] = useState<"evidence" | "target" | "result">(
    "evidence"
  );

  // Local State for Form
  const [valT, setValT] = useState<string>("");
  const [valG, setValG] = useState<string>("");
  const [valEdge, setValEdge] = useState<string>(""); // For Edges
  const [isTarget, setIsTarget] = useState(false);

  // Initialize from props when opening
  useEffect(() => {
    if (!isOpen) return;

    // Auto-switch to Results tab if results exist and no evidence/target interaction
    if (inferenceResult) {
      setActiveTab("result");
    } else {
      setActiveTab("evidence");
    }

    if (type === "node") {
      const tVar = `${data.id}_t`;
      const gVar = `${data.id}_g`;

      // Check if target
      setIsTarget(currentTargets.has(tVar));

      // Check active evidence
      // Init independent states
      const tVal =
        currentEvidence[tVar] !== undefined
          ? currentEvidence[tVar].toString()
          : "";
      const gVal =
        currentEvidence[gVar] !== undefined
          ? currentEvidence[gVar].toString()
          : "";
      setValT(tVal);
      setValG(gVal);
    } else {
      // Edge
      // Resolve variable
      const variable = getEdgeVariable(data);
      if (currentEvidence[variable] !== undefined) {
        setValEdge(currentEvidence[variable].toString());
      } else {
        setValEdge("");
      }
    }
  }, [isOpen, data, type, currentEvidence, currentTargets]);

  const getEdgeVariable = (d: any) => {
    if (d.edgeType === "ac") return `${d.target}_k`;
    if (d.edgeType === "pilot") return `${d.target}_q`;
    if (d.edgeType === "cabin") return `${d.target}_c`;
    if (d.edgeType === "pax") return `${d.target}_p_${d.source}`;
    return "";
  };

  if (!isOpen) return null;

  const handleSave = () => {
    const changes: any = {};

    if (type === "node") {
      const tVar = `${data.id}_t`;
      const gVar = `${data.id}_g`;

      // Handle Target
      const nextTargets = new Set(currentTargets);
      if (isTarget) nextTargets.add(tVar);
      else nextTargets.delete(tVar);
      changes.targets = nextTargets;

      // Handle Evidence
      const nextEvidence = { ...currentEvidence };
      if (activeTab === "evidence") {
        // Handle T
        if (valT !== "") {
          nextEvidence[tVar] = parseInt(valT);
        } else {
          delete nextEvidence[tVar];
        }
        // Handle G
        if (valG !== "") {
          nextEvidence[gVar] = valG;
        } else {
          delete nextEvidence[gVar];
        }
      }
      changes.evidence = nextEvidence;
    } else {
      // Edge
      const v = getEdgeVariable(data);
      const nextEvidence = { ...currentEvidence };
      if (valEdge !== "") {
        nextEvidence[v] = parseInt(valEdge);
      } else {
        delete nextEvidence[v];
      }
      changes.evidence = nextEvidence;
    }

    onSave(changes);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm">
      <div className="glass-panel flex max-h-[90vh] w-full max-w-md flex-col overflow-hidden rounded-2xl border border-slate-200/70 shadow-2xl shadow-black/50">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-200/70 bg-slate-100/80 p-4">
          <h3 className="pr-8 text-lg font-bold text-slate-900">
            {type === "node"
              ? `Configure Node ${data.id}`
              : `Configure Connection`}
          </h3>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-600 transition-colors hover:bg-slate-200/80 hover:text-rose-600"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="custom-scrollbar flex-1 overflow-y-auto p-4">
          {type === "node" && (
            <div className="mb-6 flex space-x-1 rounded-lg border border-slate-200/60 bg-white/70 p-1">
              <button
                className={`flex-1 rounded px-3 py-1.5 text-xs font-bold tracking-wider uppercase transition-all ${activeTab === "evidence" ? "bg-sky-500/20 text-sky-600 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}
                onClick={() => setActiveTab("evidence")}
              >
                Evidence
              </button>
              <button
                className={`flex-1 rounded px-3 py-1.5 text-xs font-bold tracking-wider uppercase transition-all ${activeTab === "target" ? "bg-indigo-500/20 text-indigo-600 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}
                onClick={() => setActiveTab("target")}
              >
                Target
              </button>
              {inferenceResult && (
                <button
                  className={`flex-1 rounded px-3 py-1.5 text-xs font-bold tracking-wider uppercase transition-all ${activeTab === "result" ? "bg-emerald-500/20 text-emerald-600 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}
                  onClick={() => setActiveTab("result")}
                >
                  Results
                </button>
              )}
            </div>
          )}

          {type === "edge" && (
            <div className="mb-6">
              <div className="mb-2 flex items-center gap-2 border-b border-slate-200/70 pb-2 text-sm font-bold text-slate-800">
                <span className="rounded bg-slate-200 p-1 text-slate-600">
                  🔗
                </span>{" "}
                Edge Description
              </div>
              <div className="mb-3 rounded bg-slate-100/80 p-2 font-mono text-sm text-slate-600">
                {data.source} <span className="text-slate-600">→</span>{" "}
                {data.target}
              </div>

              {/* Rich Info Panel */}
              <div className="glass-card rounded-lg p-3 font-mono text-xs whitespace-pre-wrap text-slate-600">
                {data.tooltip || "No specific metrics available for this link."}
              </div>

              <div className="mt-2 text-[10px] text-slate-500 italic">
                * Impact Delta shows contribution of this specific connection.
              </div>
            </div>
          )}

          {/* Evidence Form */}
          {(activeTab === "evidence" || type === "edge") && (
            <div className="space-y-4">
              {type === "node" && (
                <div className="space-y-4">
                  {/* Section T */}
                  <div className="glass-card rounded-xl border border-sky-500/10 p-4">
                    <label className="mb-2 block text-xs font-bold text-sky-600 uppercase">
                      Time / Delay (t)
                    </label>
                    <select
                      className="glass-input w-full rounded-lg p-2.5 text-sm text-slate-800"
                      value={valT}
                      onChange={(e) => setValT(e.target.value)}
                    >
                      <option value="" className="bg-white">
                        (No Info)
                      </option>
                      <option value="0" className="bg-white">
                        Bin 0 (On Time / Early)
                      </option>
                      <option value="1" className="bg-white">
                        Bin 1 (0-15m Late)
                      </option>
                      <option value="2" className="bg-white">
                        Bin 2 (15-30m Late)
                      </option>
                      <option value="3" className="bg-white">
                        Bin 3 (30-60m Late)
                      </option>
                      <option value="4" className="bg-white">
                        Bin 4 (60m+ Late)
                      </option>
                    </select>
                  </div>

                  {/* Section G */}
                  <div className="glass-card rounded-xl border border-emerald-500/10 p-4">
                    <label className="mb-2 block text-xs font-bold text-emerald-600 uppercase">
                      Turnaround Cause (g)
                    </label>
                    <select
                      className="glass-input w-full rounded-lg p-2.5 text-sm text-slate-800"
                      value={valG}
                      onChange={(e) => setValG(e.target.value)}
                    >
                      <option value="" className="bg-white">
                        (No Info)
                      </option>
                      <option value="None" className="bg-white">
                        None (Normal)
                      </option>
                      <option value="Reactionary" className="bg-white">
                        Reactionary
                      </option>
                      <option value="Pax" className="bg-white">
                        Pax Connection
                      </option>
                      <option value="Ops" className="bg-white">
                        Operations/Tech
                      </option>
                      <option value="Weather" className="bg-white">
                        Weather
                      </option>
                      <option value="ATC" className="bg-white">
                        ATC
                      </option>
                      <option value="Other" className="bg-white">
                        Other
                      </option>
                    </select>
                  </div>
                </div>
              )}

              {type === "edge" && (
                <div>
                  <label className="mb-2 block text-xs font-bold text-slate-600">
                    State Value (Bin)
                  </label>
                  <select
                    className="glass-input w-full rounded-lg p-2.5 text-sm text-slate-800"
                    value={valEdge}
                    onChange={(e) => setValEdge(e.target.value)}
                  >
                    <option value="" className="bg-white">
                      -- No Evidence --
                    </option>
                    <option value="0" className="bg-white">
                      Bin 0 (Available/On-Time)
                    </option>
                    <option value="1" className="bg-white">
                      Bin 1 (Minor Delay)
                    </option>
                    <option value="2" className="bg-white">
                      Bin 2 (Moderate)
                    </option>
                    <option value="3" className="bg-white">
                      Bin 3 (Significant)
                    </option>
                    <option value="4" className="bg-white">
                      Bin 4 (Severe)
                    </option>
                  </select>
                </div>
              )}
            </div>
          )}

          {/* Target Form */}
          {activeTab === "target" && type === "node" && (
            <div className="space-y-4 py-4">
              <label className="glass-card flex cursor-pointer items-start space-x-3 rounded-xl border border-transparent p-4 transition-colors hover:border-indigo-500/30 hover:bg-slate-100/80">
                <input
                  type="checkbox"
                  className="mt-1 h-5 w-5 rounded border-slate-300 bg-white/70 text-indigo-500 focus:ring-indigo-500"
                  checked={isTarget}
                  onChange={(e) => setIsTarget(e.target.checked)}
                />
                <div>
                  <span className="block text-sm font-bold text-indigo-600">
                    Set as Prediction Target
                  </span>
                  <span className="text-xs text-slate-500">
                    Observe how delays propagate to this flight node during
                    inference.
                  </span>
                </div>
              </label>
            </div>
          )}

          {/* Results Display */}
          {activeTab === "result" && inferenceResult && (
            <div className="space-y-4 pt-2">
              <div className="glass-card rounded-xl border border-emerald-500/10 bg-gradient-to-br from-emerald-900/10 to-transparent p-4">
                <div className="mb-3 text-[10px] font-bold tracking-wider text-emerald-600 uppercase">
                  Prediction Result
                </div>
                <div className="flex items-end justify-between">
                  <div>
                    <div className="text-4xl font-black text-slate-900">
                      {inferenceResult.expected_delay.toFixed(1)}
                      <span className="ml-1 text-lg font-normal text-slate-500">
                        min
                      </span>
                    </div>
                    <div className="mt-1 text-xs font-bold text-slate-500 uppercase">
                      Expected Delay
                    </div>
                  </div>
                  <div className="text-right">
                    <div
                      className={`text-2xl font-black ${inferenceResult.prob_delay > 0.5 ? "text-rose-600" : "text-emerald-600"}`}
                    >
                      {Math.round(inferenceResult.prob_delay * 100)}%
                    </div>
                    <div className="mt-1 text-xs font-bold text-slate-500 uppercase">
                      Prob &gt; 30m
                    </div>
                  </div>
                </div>
                {/* New Section: Most Likely Bin */}
                {/* Scenario B: Impact Display */}
                {inferenceResult.metrics &&
                  Math.abs(inferenceResult.metrics.DMx) > 0.1 && (
                    <div className="mt-4 flex items-center justify-between rounded-lg border border-rose-500/20 bg-rose-500/10 p-2">
                      <span className="text-xs font-bold text-rose-600 uppercase">
                        Impact (Delta)
                      </span>
                      <span className="font-mono text-sm font-bold text-rose-600">
                        {inferenceResult.metrics.DMx > 0 ? "+" : ""}
                        {inferenceResult.metrics.DMx.toFixed(1)}m
                      </span>
                    </div>
                  )}

                {/* Confidence Section */}
                <div className="mt-4 flex items-center justify-between border-t border-slate-200/60 pt-3">
                  <div className="text-xs font-bold text-slate-500 uppercase">
                    Most Likely State
                  </div>
                  <div className="text-sm">
                    <span className="font-bold text-emerald-600">
                      Bin {(inferenceResult as any).most_likely_bin}
                    </span>
                    <span className="ml-1 text-slate-500">
                      (
                      {Math.round(
                        ((inferenceResult as any).most_likely_bin_prob || 0) *
                          100
                      )}
                      %)
                    </span>
                  </div>
                </div>
              </div>

              {inferenceResult.cause_breakdown && (
                <div className="glass-card rounded-xl p-4">
                  <div className="mb-3 text-[10px] font-bold text-slate-500 uppercase">
                    Breakdown
                  </div>
                  <div className="space-y-2">
                    {Object.entries(inferenceResult.cause_breakdown).map(
                      ([k, v]) => (
                        <div
                          key={k}
                          className="flex items-center justify-between text-xs"
                        >
                          <span className="text-slate-600">{k}</span>
                          <div className="flex items-center gap-2">
                            <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-200">
                              <div
                                className="h-full rounded-full bg-slate-400"
                                style={{
                                  width: `${Math.min(v as number, 100)}%`,
                                }}
                              ></div>
                            </div>
                            <span className="w-6 text-right font-mono font-bold text-slate-700">
                              {v}
                            </span>
                          </div>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 border-t border-slate-200/70 bg-white/80 p-4">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-bold tracking-wider text-slate-600 uppercase transition-colors hover:text-slate-900"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="glass-btn rounded-lg px-6 py-2 text-xs font-bold tracking-wider text-slate-900 uppercase shadow-lg shadow-sky-900/20"
          >
            Save Configuration
          </button>
        </div>
      </div>
    </div>
  );
}

