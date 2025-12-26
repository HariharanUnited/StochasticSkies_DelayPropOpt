import { useCallback, useEffect, useState, memo } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  MarkerType,
  Position,
  useEdgesState,
  useNodesState,
  getBezierPath,
  type EdgeProps,
  type Edge,
  type Node,
} from "reactflow";
import "reactflow/dist/style.css";
import "../App.css";
import CPTModal from "../components/CPTModal";
import ScenarioModal from "../components/ScenarioModal";
import ToolsPanel, { type ToolMode } from "../components/ToolsPanel";
import SwapPage from "../swap_page/index";
import {
  Plane,
  Search,
  RotateCcw,
  Settings,
  Network,
  AlertTriangle,
  Info,
  Zap,
  Activity,
} from "lucide-react";
import { motion } from "framer-motion";

// --- Types ---
type Flight = {
  flight_id: string;
  tail_id: string;
  departure_airport: string;
  arrival_airport: string;
  scheduled_departure: number;
  scheduled_arrival: number;
  inbound_aircraft_flight?: string | null;
  inbound_pilot_flight?: string | null;
  inbound_cabin_flight?: string | null;
  inbound_passenger_flights?: string[];
};

type NetworkJson = { flights: Flight[] };

// Define CPT Data structure
interface CPTData {
  cpts: Record<string, any>;
  g_priors: Record<string, any>;
  resource_cpds: Record<string, any>;
}

interface TopKItem {
  id: string;
  val: number;
}

interface Scorecard {
  robustness_score: number;
  top_k_dmx: TopKItem[];
  top_k_dme: TopKItem[];
  top_k_dmc: TopKItem[];
  risky_hubs: string[];
  fragile_tails: string[];
}

// --- Constants ---
const laneGap = 150;
const timeScale = 3;
const SERVER_URL = "http://localhost:8000";
const timebankOptions = [
  { label: "00:00-03:00", start: 0, end: 180 },
  { label: "03:00-06:00", start: 180, end: 360 },
  { label: "06:00-09:00", start: 360, end: 540 },
  { label: "09:00-12:00", start: 540, end: 720 },
  { label: "12:00-15:00", start: 720, end: 900 },
  { label: "15:00-18:00", start: 900, end: 1080 },
  { label: "18:00-21:00", start: 1080, end: 1260 },
  { label: "21:00-24:00", start: 1260, end: 1440 },
];

// --- Custom Edge ---
const OffsetBezier = memo(
  ({
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    data,
    label,
    markerEnd,
    style,
  }: EdgeProps) => {
    const vy = (data as any)?.vy || 0;
    const color = (data as any)?.color || style?.stroke;
    const labelColor = color || "#1e293b";
    const [path, labelX, labelY] = getBezierPath({
      sourceX,
      sourceY: sourceY + vy,
      targetX,
      targetY: targetY + vy,
    });
    return (
      <>
        <path
          id={id}
          d={path}
          style={{ ...style, stroke: color, fill: "none" }}
          markerEnd={markerEnd}
        />
        {label && (
          <g>
            <text
              x={labelX}
              y={labelY}
              fill={labelColor}
              fontSize={11}
              fontWeight={700}
              dy={-4}
              textAnchor="middle"
            >
              {label}
            </text>
            {/* Info Button for CPT */}
            <g className="cpt-info-btn-edge" style={{ cursor: "pointer" }}>
              <circle cx={labelX + 15} cy={labelY - 8} r={5} fill={color} />
              <text
                x={labelX + 15}
                y={labelY - 6}
                fill="white"
                fontSize={8}
                fontWeight="bold"
                textAnchor="middle"
              >
                i
              </text>
            </g>
            {/* Scenario Button (Tool 1) - Gear Icon */}
            <g className="scenario-btn" style={{ cursor: "pointer" }}>
              <rect
                x={labelX - 25}
                cy={labelY - 13}
                width={12}
                height={12}
                rx={2}
                fill="#ef4444"
                y={labelY - 14}
              />
              <text
                x={labelX - 19}
                y={labelY - 5}
                fill="white"
                fontSize={8}
                fontWeight="bold"
                textAnchor="middle"
              >
                ⚙
              </text>
            </g>
          </g>
        )}
      </>
    );
  }
);

const edgeTypes = {
  offsetBezier: OffsetBezier,
};

// Stable Props to avoid re-renders/warnings
const fitoptions = { padding: 0.2 };
const EXTENT: [[number, number], [number, number]] = [
  [-1000, -1000],
  [50000, 5000],
];
const proOpts = { hideAttribution: true };

// --- Helpers ---
const formatTime = (minutes: number) => {
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}`;
};

function buildGraph(net: NetworkJson) {
  const tailY = new Map<string, number>();
  let lane = 0;
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  for (const f of net.flights) {
    if (!tailY.has(f.tail_id)) {
      tailY.set(f.tail_id, lane * laneGap);
      lane += 1;
    }
    const y = tailY.get(f.tail_id)!;
    const x = f.scheduled_departure * timeScale;

    nodes.push({
      id: f.flight_id,
      position: { x, y },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      className:
        "glass-card !p-0 flex items-center justify-center text-center text-[10px] font-bold text-slate-800 bg-amber-500/10 border-amber-500/30 shadow-lg shadow-black/20 hover:border-amber-400 hover:bg-amber-500/20 hover:shadow-[0_0_20px_rgba(251,191,36,0.3)] transition-all",
      style: {
        width: 100,
        height: 70,
      },
      data: {
        label: (
          <div className="relative flex h-full w-full flex-col items-center justify-center leading-tight">
            <div className="text-[10px] text-amber-600">{f.flight_id}</div>
            <div className="text-[9px] opacity-70">
              {f.departure_airport}-{f.arrival_airport}
            </div>
            <div className="font-mono text-[8px] opacity-50">
              {formatTime(f.scheduled_departure)}-
              {formatTime(f.scheduled_arrival)}
            </div>

            {/* Info Button */}
            <div className="cpt-info-btn-node absolute -top-1.5 -right-1.5 z-10 flex h-4 w-4 cursor-pointer items-center justify-center rounded-full bg-sky-500 text-[10px] font-bold text-white shadow-md shadow-sky-900/50 transition-transform hover:scale-110">
              i
            </div>
            {/* Scenario Button (Tool 1) */}
            <div
              className="scenario-btn absolute -top-1.5 -left-1.5 z-10 flex h-4 w-4 cursor-pointer items-center justify-center rounded-md bg-rose-500 text-[10px] font-bold text-white shadow-md shadow-rose-900/50 transition-transform hover:scale-110"
              style={{ display: "none" }}
            >
              ⚙
            </div>
          </div>
        ),
      },
    });
  }

  const addEdge = (
    src?: string | null,
    tgt?: string,
    type?: string,
    color?: string
  ) => {
    if (src && tgt) {
      const order: Record<string, number> = {
        ac: -20,
        pilot: -7,
        cabin: 7,
        pax: 20,
      };
      const vy = order[type || ""] ?? 0;
      edges.push({
        id: `${src}->${tgt}-${type}`,
        source: src,
        target: tgt,
        type: "offsetBezier",
        label: type,
        data: { type, vy, color },
        animated: false,
        interactionWidth: 40,
        labelStyle: { fontSize: 11, fill: color || "#1e293b", fontWeight: 700 },
        style: { stroke: color || "#64748b", strokeWidth: 2, opacity: 0.8 }, // Slate-500 default
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: color || "#64748b",
          width: 10,
          height: 10,
        },
      });
    }
  };

  for (const f of net.flights) {
    addEdge(f.inbound_aircraft_flight, f.flight_id, "ac", "#4a90e2");
    addEdge(f.inbound_pilot_flight, f.flight_id, "pilot", "#3eb489");
    addEdge(f.inbound_cabin_flight, f.flight_id, "cabin", "#e26a6a");
    if (f.inbound_passenger_flights) {
      for (const pf of f.inbound_passenger_flights)
        addEdge(pf, f.flight_id, "pax", "#b57edc");
    }
  }

  return { nodes, edges };
}

// --- Main App ---
export default function App() {
  const params = new URLSearchParams(window.location.search);
  const isSwapPage = params.get("page") === "swap";
  if (isSwapPage) {
    return <SwapPage />;
  }
  const [rfNodes, setRfNodes, onNodesChange] = useNodesState([]);
  const [rfEdges, setRfEdges, onEdgesChange] = useEdgesState([]);

  // State for Models
  const [initialNetworkData, setInitialNetworkData] =
    useState<NetworkJson | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>(""); // Default empty, wait for list
  const [availableHubs, setAvailableHubs] = useState<string[]>([]);
  const [selectedHub, setSelectedHub] = useState<string>("");
  const [selectedTimebank, setSelectedTimebank] = useState<string>(
    timebankOptions[0].label
  );
  const [error, setError] = useState<string | null>(null);
  const [hubTurnPct, setHubTurnPct] = useState<number>(20);
  const [hubTurnRunning, setHubTurnRunning] = useState(false);
  const [hubTurnStatus, setHubTurnStatus] = useState<string | null>(null);

  // CPT & Modal State
  const [cptData, setCptData] = useState<CPTData | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [modalTitle, setModalTitle] = useState("");
  const [modalData, setModalData] = useState<any>(null);
  const [modalType, setModalType] = useState<"node" | "edge">("node");
  const [modalNodeId, setModalNodeId] = useState<string>("");

  // Hover State
  const [hoverInfo, setHoverInfo] = useState<{
    x: number;
    y: number;
    content: React.ReactNode;
  } | null>(null);

  const [activeTool, setActiveTool] = useState<ToolMode>("explore");
  const [inferenceResults, setInferenceResults] = useState<Record<string, any>>(
    {}
  );
  const [scorecard, setScorecard] = useState<Scorecard | null>(null);
  const [dmMode, setDmMode] = useState<"dmc" | "dme" | "dmx">("dmx");
  const [activeTopKModal, setActiveTopKModal] = useState<
    "dmc" | "dme" | "dmx" | null
  >(null);
  const [isScorecardCollapsed, setIsScorecardCollapsed] = useState(false);
  const [connectionTop, setConnectionTop] = useState<{ [key: string]: any[] }>({
    ac: [],
    pilot: [],
    cabin: [],
    pax: [],
  });
  const [isConnectionCollapsed, setIsConnectionCollapsed] = useState(false);

  const [_selectedNode, setSelectedNode] = useState<string | null>(null); // Restored State

  // --- Tool 3 Handler ---
  const startDMAnalysis = async () => {
    if (!selectedModel) return;
    setIsProcessing(true);
    try {
      const resp = await fetch(`${SERVER_URL}/tools/dm_analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selectedModel,
          evidence: {},
          target: [],
          shock_mode: "do",
          hop_limit: 5,
        }),
      });
      const data = await resp.json();
      setInferenceResults(data.network_state);
      setScorecard(data.scorecard || null);
    } catch (e) {
      console.error(e);
      alert("DM Analysis Failed");
    } finally {
      setIsProcessing(false);
    }
  };

  // Hub Turnaround
  const handleRunHubTurnaround = async () => {
    if (!selectedModel) return;
    if (!selectedHub) {
      alert("Select a hub first");
      return;
    }

    setHubTurnRunning(true);
    setHubTurnStatus(null);

    try {
      const resp = await fetch(`${SERVER_URL}/tools/hub_turnaround`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selectedModel,
          hub: selectedHub,
          pct: hubTurnPct,
        }),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      const newModel = data.new_model;
      if (!newModel) throw new Error("Backend did not return new_model");

      // Add to sidebar list immediately
      setAvailableModels((prev) =>
        prev.includes(newModel) ? prev : [newModel, ...prev]
      );

      // Switch to it -> your existing model-loading useEffect will reload network & CPTs
      setSelectedModel(newModel);

      setHubTurnStatus(
        `Created: ${newModel} (affected: ${data.affected?.length ?? 0} flights)`
      );
    } catch (e: any) {
      console.error(e);
      setHubTurnStatus(`Failed: ${e?.message ?? String(e)}`);
      alert("Hub turnaround failed. Check console.");
    } finally {
      setHubTurnRunning(false);
    }
  };

  // Recolor Effect for Tool 3
  useEffect(() => {
    if (activeTool !== "tool3" || Object.keys(inferenceResults).length === 0)
      return;

    setRfNodes((nds) =>
      nds.map((n) => {
        const state = inferenceResults[n.id + "_t"];

        if (state && state.dm_metrics) {
          const val = state.dm_metrics[dmMode];
          let color = "#ffffff"; // Default White

          if (dmMode === "dmx") {
            // DMx: Difference Ratio (Sensitivity) - Scale 1.0 to ~1.5
            // Very sensitive
            if (val >= 1.25)
              color = "#ef4444"; // Red
            else if (val >= 1.15)
              color = "#ea580c"; // Dark Orange
            else if (val >= 1.1)
              color = "#f97316"; // Orange
            else if (val >= 1.05)
              color = "#facc15"; // Yellow
            else if (val >= 1.01)
              color = "#bef264"; // Lime
            else color = "#dcfce7"; // Pale Green
          } else {
            // DMc / DMe: Total Ratio - Scale 1.0 to ~5.0
            if (val >= 3.0)
              color = "#b91c1c"; // Dark Red
            else if (val >= 2.0)
              color = "#ef4444"; // Red
            else if (val >= 1.6)
              color = "#ea580c"; // Dark Orange
            else if (val >= 1.3)
              color = "#f97316"; // Orange
            else if (val >= 1.1)
              color = "#facc15"; // Yellow
            else if (val > 1.0) color = "#bef264"; // Lime
          }

          return {
            ...n,
            style: { ...n.style, background: color, border: "2px solid #555" },
            data: {
              ...n.data,
              tooltip: `DM Mode: ${dmMode.toUpperCase()}\nVal: ${val.toFixed(
                2
              )}`,
            },
          };
        }
        return { ...n, style: { ...n.style, background: "#ffffff" } };
      })
    );
  }, [activeTool, dmMode, inferenceResults, setRfNodes]);

  // Scenario Modal State
  const [scenarioModalOpen, setScenarioModalOpen] = useState(false);
  const [scenarioModalType, setScenarioModalType] = useState<"node" | "edge">(
    "node"
  );
  const [scenarioModalData, setScenarioModalData] = useState<any>(null);

  // const [selectedEdge, setSelectedEdge] = useState<any | null>(null); // Removed
  const [evidence, setEvidence] = useState<Record<string, number | string>>({});
  const [targets, setTargets] = useState<Set<string>>(new Set());
  const [allTargets, setAllTargets] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  // 1. Fetch Available Models from Backend
  useEffect(() => {
    // Cache bust to ensure fresh list
    fetch(`${SERVER_URL}/visualizations?t=${Date.now()}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.visualizations) {
          setAvailableModels(data.visualizations);

          if (data.visualizations.length > 0) {
            if (!data.visualizations.includes(selectedModel)) {
              // If previous model deleted, select first available
              setSelectedModel(data.visualizations[0]);
            }
          } else {
            // No models at all
            setSelectedModel("");
            setRfNodes([]);
            setRfEdges([]);
          }
        }
      })
      .catch((err) => console.error("Failed to fetch models", err));
  }, []);

  // 2. Fetch Network Data when Model Changes
  useEffect(() => {
    if (!selectedModel) return;

    // Use Backend Endpoint to ensure we get the file regardless of where it is stored
    const url = `${SERVER_URL}/visualizations/${selectedModel}/network?t=${Date.now()}`;
    const urlCpt = `${SERVER_URL}/visualizations/${selectedModel}/cpts?t=${Date.now()}`;

    console.log("Loading model:", url);
    setError(null);

    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((net: NetworkJson) => {
        const { nodes, edges } = buildGraph(net);
        setRfNodes(nodes);
        setRfEdges(edges);
        // Store Golden Copy for Reset
        setInitialNetworkData(net);
        const hubs = Array.from(
          new Set(net.flights.map((f) => f.departure_airport))
        ).sort();
        setAvailableHubs(hubs);
        setSelectedHub((prev) =>
          prev && hubs.includes(prev) ? prev : hubs[0] || ""
        );
        setSelectedTimebank(timebankOptions[0].label);
      })
      .catch((e) => {
        console.error(e);
        setError(`Failed to load ${url}: ${e.message}`);
      });

    // Load CPT Data
    fetch(urlCpt)
      .then((r) => {
        if (r.ok) return r.json();
        console.warn("CPT data missing/error for", selectedModel);
        return null;
      })
      .then((data) => {
        if (data) setCptData(data);
        else setCptData(null);
      })
      .catch((e) => console.error("CPT Fetch Error", e));
  }, [selectedModel, setRfNodes, setRfEdges]);

  // --- State for Search & Highlight ---
  // --- State for Search & Highlight ---
  const [searchTerm, setSearchTerm] = useState("");
  const [highlightedId, setHighlightedId] = useState<string | null>(null);

  // Highlighting Logic (Centralized)
  const highlightGraph = useCallback(
    (centerNodeId: string | null) => {
      setHighlightedId(centerNodeId);

      if (!centerNodeId) {
        setRfEdges((eds) =>
          eds.map((e) => ({
            ...e,
            style: { ...e.style, opacity: 0.9, strokeWidth: 4 },
            animated: false,
          }))
        );
        setRfNodes((nds) =>
          nds.map((n) => ({
            ...n,
            style: { ...n.style, opacity: 1, border: "2px solid #d8dff0" },
          }))
        );
        return;
      }

      // --- Path Traversal Logic ---
      const connectedEdges = new Set<string>();
      const connectedNodes = new Set<string>([centerNodeId]);

      // Helper: Traverse chain for a specific type
      const traceChain = (type: string) => {
        const queue = [centerNodeId];
        const visited = new Set([centerNodeId]);

        let head = 0;
        while (head < queue.length) {
          const curr = queue[head++];
          rfEdges.forEach((e) => {
            if ((e as any).data?.type === type) {
              const isSource = e.source === curr;
              const isTarget = e.target === curr;

              if (isSource || isTarget) {
                if (!connectedEdges.has(e.id)) {
                  connectedEdges.add(e.id);
                  const other = isSource ? e.target : e.source;
                  if (!visited.has(other)) {
                    visited.add(other);
                    connectedNodes.add(other);
                    queue.push(other);
                  }
                }
              }
            }
          });
        }
      };

      // 1. Trace Resource Chains
      ["ac", "pilot", "cabin"].forEach((t) => traceChain(t));

      // 2. Add Immediate Pax (One-Hop)
      rfEdges.forEach((e) => {
        if (
          (e as any).data?.type === "pax" &&
          (e.source === centerNodeId || e.target === centerNodeId)
        ) {
          connectedEdges.add(e.id);
          connectedEdges.add(e.id);
          connectedNodes.add(e.source);
          connectedNodes.add(e.target);
        }
      });

      // Apply Styles
      setRfEdges((eds) =>
        eds.map((e) =>
          connectedEdges.has(e.id)
            ? {
                ...e,
                style: { ...e.style, opacity: 1, strokeWidth: 5 },
                animated: true,
              }
            : {
                ...e,
                style: { ...e.style, opacity: 0.1, strokeWidth: 2 },
                animated: false,
              }
        )
      );
      setRfNodes((nds) =>
        nds.map((n) => {
          const isCenter = n.id === centerNodeId;
          const isConnected = connectedNodes.has(n.id);
          return {
            ...n,
            style: {
              ...n.style,
              opacity: isConnected ? 1 : 0.2, // Path nodes visible
              border: isCenter ? "4px solid #3b82f6" : "2px solid #d8dff0",
            },
          };
        })
      );
    },
    [rfEdges, setRfNodes, setRfEdges]
  );

  // Search Effect
  useEffect(() => {
    if (searchTerm && searchTerm.length > 2) {
      const match = rfNodes.find(
        (n) => n.id.toLowerCase() === searchTerm.toLowerCase()
      );
      if (match) {
        // Gate Check: Only calling highlight if different
        if (match.id !== highlightedId) {
          highlightGraph(match.id);
        }
      }
    } else if (searchTerm === "") {
      if (highlightedId !== null) {
        highlightGraph(null);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchTerm, highlightGraph, highlightedId]); // Removed rfNodes

  const getTimebankRange = (label: string) => {
    const match = timebankOptions.find((o) => o.label === label);
    if (match) return { start: match.start, end: match.end };
    const parts = label.split("-");
    const toMinutes = (s: string) => {
      const [h, m] = s.split(":").map((x) => parseInt(x, 10));
      return (isNaN(h) ? 0 : h) * 60 + (isNaN(m) ? 0 : m);
    };
    if (parts.length === 2)
      return { start: toMinutes(parts[0]), end: toMinutes(parts[1]) };
    return { start: 0, end: 180 };
  };

  const renderPieChart = (breakdown: Record<string, number>) => {
    const entries: [string, number, string][] = [
      ["Aircraft (k)", breakdown["Aircraft (k)"] || 0, "#ef4444"],
      ["Pilot (q)", breakdown["Pilot (q)"] || 0, "#3b82f6"],
      ["Crew (c)", breakdown["Crew (c)"] || 0, "#f59e0b"],
      ["Pax (p)", breakdown["Pax (p)"] || 0, "#8b5cf6"],
    ];
    const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
    let startAngle = 0;
    const radius = 35;
    const cx = 40,
      cy = 40;
    const slices = entries.map(([label, val, color]) => {
      const angle = (val / total) * Math.PI * 2;
      const endAngle = startAngle + angle;
      const x1 = cx + radius * Math.cos(startAngle);
      const y1 = cy + radius * Math.sin(startAngle);
      const x2 = cx + radius * Math.cos(endAngle);
      const y2 = cy + radius * Math.sin(endAngle);
      const largeArc = angle > Math.PI ? 1 : 0;
      const d = `M ${cx} ${cy} L ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2} Z`;
      startAngle = endAngle;
      return <path key={label} d={d} fill={color} />;
    });

    return (
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <svg width={80} height={80} viewBox="0 0 80 80">
          {slices}
        </svg>
        <div style={{ fontSize: "0.75rem", lineHeight: 1.3 }}>
          {entries.map(([label, val, color]) => (
            <div
              key={label}
              style={{ display: "flex", alignItems: "center", gap: 6 }}
            >
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: "50%",
                  background: color,
                  display: "inline-block",
                }}
              />
              <span>{label}</span>
              <span style={{ fontFamily: "monospace" }}>
                {Math.round((val / (total || 1)) * 100)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // --- Inference Logic ---
  const handleRunInference = async () => {
    if (!selectedModel) return;
    setIsProcessing(true);
    try {
      let endpoint = "/infer";
      let payload: any = {
        model: selectedModel,
        evidence: evidence,
        target: allTargets ? ["ALL"] : Array.from(targets),
        shock_mode: "do",
      };

      if (activeTool === "tool2") {
        endpoint = "/infer_diagnostics";
      } else if (activeTool === "hub") {
        if (!selectedHub) throw new Error("Select a hub for Hub Disruptor");
        const { start, end } = getTimebankRange(selectedTimebank);
        endpoint = "/tools/hub_disruptor";
        payload = {
          model: selectedModel,
          evidence: evidence,
          target: [],
          shock_mode: "do",
          hub: selectedHub,
          hub_code: selectedHub,
          timebank_start: start,
          timebank_end: end,
          timebank_label: selectedTimebank,
        };
      } else if (activeTool === "late") {
        endpoint = "/tools/lateness_contributor";
        payload = {
          model: selectedModel,
          evidence: evidence,
          target: targets.size > 0 ? Array.from(targets) : [],
        };
      } else if (activeTool === "connection") {
        endpoint = "/tools/connection_risk";
        payload = {
          model: selectedModel,
          evidence: {},
          target: [],
        };
      }
      console.log(
        "Sending Inference Payload:",
        JSON.stringify(payload, null, 2)
      );
      console.log(`Hitting Endpoint: ${endpoint}`);

      const response = await fetch(`${SERVER_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error("Inference Request Failed");

      const data = await response.json();

      // 1. Store Results
      if (data.network_state) {
        setInferenceResults(data.network_state);
      }
      if (activeTool !== "tool3") {
        setScorecard(null);
      }
      if (activeTool === "connection") {
        const groups: any = { ac: [], pilot: [], cabin: [], pax: [] };
        if (data.visuals && data.visuals.edges) {
          data.visuals.edges.forEach((ve: any) => {
            const t = ve.type || "other";
            if (!groups[t]) return;
            let score = 0;
            let effect = 0;
            if (ve.tooltip) {
              const m = ve.tooltip.match(/Score:\s*([\d\.]+)/i);
              if (m) score = parseFloat(m[1]);
              const efx = ve.tooltip.match(/Effect:\s*([\d\.]+)/i);
              if (efx) effect = parseFloat(efx[1]);
            }
            groups[t].push({ src: ve.from, dst: ve.to, score, effect });
          });
          Object.keys(groups).forEach((k) => {
            groups[k].sort((a: any, b: any) => (b.score || 0) - (a.score || 0));
            groups[k] = groups[k].slice(0, 10);
          });
        }
        setConnectionTop(groups);
      } else {
        setConnectionTop({ ac: [], pilot: [], cabin: [], pax: [] });
      }

      // 2. Update Visuals (Colors)
      if (data.visuals && data.visuals.nodes) {
        const vizNodes = data.visuals.nodes;
        setRfNodes((nds) =>
          nds.map((n) => {
            const viz = vizNodes[n.id];
            if (viz) {
              return {
                ...n,
                style: {
                  ...n.style,
                  background: viz.color,
                  border: viz.border || n.style?.border,
                },
                data: { ...n.data, tooltip: viz.tooltip },
              };
            }
            return n;
          })
        );
      }

      // 3. Update Visuals (Edges) - THIS WAS MISSING
      if (data.visuals && data.visuals.edges) {
        const vizEdges = data.visuals.edges; // List of {from, to, color, thickness}
        // Create lookup
        const edgeMap = new Map<string, any>();
        vizEdges.forEach((ve: any) => {
          // Key by Type to allow multiple edges between same nodes (e.g. AC vs Pilot)
          const key = ve.type
            ? `${ve.from}->${ve.to}-${ve.type}`
            : `${ve.from}->${ve.to}`;
          edgeMap.set(key, ve);
        });

        setRfEdges((eds) =>
          eds.map((e) => {
            const type = (e as any).data?.type;
            const key = type
              ? `${e.source}->${e.target}-${type}`
              : `${e.source}->${e.target}`;
            const viz = edgeMap.get(key);

            if (viz) {
              // Extract label from tooltip (Impact or Type)
              let label = undefined;
              if (viz.tooltip) {
                // Try to match Impact/Delta
                const impact = viz.tooltip.match(/Impact.*: ([\+\-\d\.]+m)/);
                if (impact) label = impact[1];
                else if (viz.tooltip.includes("Evidence")) label = "Evidence";
              }

              return {
                ...e,
                label: label, // Show label on edge
                labelStyle: { fill: viz.color, fontWeight: 700, fontSize: 11 },
                labelBgStyle: {
                  fill: "#ffffff",
                  fillOpacity: 0.9,
                  stroke: viz.color,
                  strokeWidth: 1,
                },

                style: {
                  ...e.style,
                  stroke: viz.color,
                  strokeWidth: viz.thickness,
                  opacity: 1,
                },
                animated: true,
                data: { ...e.data, color: viz.color, tooltip: viz.tooltip }, // Custom tooltip support if added later
              };
            }
            // Non-active edges: Fade them out
            return {
              ...e,
              style: { ...e.style, stroke: "#e5e7eb", opacity: 0.3 },
              label: undefined,
            };
          })
        );
      }
    } catch (e) {
      console.error(e);
      alert("Inference Failed. Check console.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-white font-sans text-slate-900">
      {/* Top Bar */}
      <motion.div
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="glass-panel relative z-50 flex h-16 items-center justify-between px-6"
      >
        <div className="flex items-center gap-4">
          <div className="rounded-lg border border-sky-500/30 bg-sky-500/20 p-2">
            <Plane className="h-6 w-6 text-sky-600" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-slate-900">
              Causal Delay Handler
            </h1>
            <div className="flex items-center gap-2 text-xs text-slate-600">
              <div
                className={`h-2 w-2 rounded-full ${cptData ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "animate-pulse bg-amber-500"}`}
              />
              <span>
                {cptData ? "CPT Net Active" : "Loading Model Loop..."}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="group relative">
            <Search className="absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-slate-500 transition-colors group-focus-within:text-sky-600" />
            <input
              type="text"
              placeholder="Search Flight..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="glass-input w-64 rounded-full py-2 pr-4 pl-10 text-sm transition-all focus:w-80"
            />
          </div>
          <button
            onClick={() => {
              setSearchTerm("");
              highlightGraph(null);
            }}
            className="rounded-full p-2 text-slate-600 transition-colors hover:bg-slate-200/80 hover:text-rose-600"
            title="Reset View"
          >
            <RotateCcw className="h-5 w-5" />
          </button>
        </div>
      </motion.div>

      <div className="relative flex flex-1 overflow-hidden">
        {/* Left Sidebar: Network Switcher */}
        {/* Left Sidebar: Network Switcher */}
        <div className="group z-40 flex w-14 flex-col border-t-0 border-r border-b-0 border-l-0 border-slate-200/70 bg-white/70 backdrop-blur-md transition-[width] duration-300 hover:w-64">
          <div className="flex h-14 items-center border-b border-slate-200/70 px-4">
            <Network className="h-5 w-5 shrink-0 text-slate-600" />
            <span className="ml-3 whitespace-nowrap text-xs font-bold tracking-wider text-slate-500 uppercase opacity-0 transition-opacity duration-300 group-hover:opacity-100">
              Available Networks
            </span>
          </div>
          <div className="custom-scrollbar flex-1 space-y-1 overflow-y-auto overflow-x-hidden p-2">
            {availableModels.map((m) => (
              <button
                key={m}
                onClick={() => setSelectedModel(m)}
                className={`flex w-full items-center gap-3 rounded-lg p-2 text-left text-sm transition-all ${
                  selectedModel === m
                    ? "bg-sky-500/20 text-sky-600 ring-1 ring-sky-500/30"
                    : "text-slate-600 hover:bg-slate-100/80 hover:text-slate-800"
                }`}
              >
                <Activity
                  className={`h-4 w-4 shrink-0 ${selectedModel === m ? "text-sky-600" : "text-slate-600"}`}
                />
                <span className="truncate opacity-0 transition-opacity duration-300 group-hover:opacity-100">
                  {m}
                </span>
              </button>
            ))}
            {availableModels.length === 0 && (
              <div className="p-4 text-center text-sm text-slate-600 italic opacity-0 group-hover:opacity-100">
                No models found...
              </div>
            )}
          </div>
        </div>

        {/* Center: Canvas */}
        <div className="relative flex-1 bg-slate-50">
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="absolute top-4 left-4 z-50 flex items-center gap-3 rounded-lg border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-rose-600 shadow-lg backdrop-blur-xl"
            >
              <AlertTriangle className="h-5 w-5" />
              <span className="text-sm font-medium">{error}</span>
            </motion.div>
          )}
          <ReactFlow
            nodes={rfNodes}
            edges={rfEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            // Removed duplicate handlers
            fitView
            fitViewOptions={fitoptions}
            minZoom={0.5}
            maxZoom={4}
            translateExtent={EXTENT}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={true}
            zoomOnDoubleClick={true}
            edgeTypes={edgeTypes}
            proOptions={proOpts}
            onNodeMouseEnter={(e, n) => {
              // Lookup fresh node state to ensure tooltip data is current
              const node = rfNodes.find((rn) => rn.id === n.id) || n;
              const breakdown =
                activeTool === "late"
                  ? inferenceResults[`${node.id}_t`] &&
                    inferenceResults[`${node.id}_t`].cause_breakdown
                  : null;
              setHoverInfo({
                x: e.clientX,
                y: e.clientY,
                content: (
                  <div>
                    <strong>Flight {node.id}</strong>
                    <br />
                    {activeTool === "late" && breakdown ? (
                      <div style={{ marginTop: 4 }}>
                        {renderPieChart(breakdown)}
                      </div>
                    ) : node.data?.tooltip ? (
                      <div
                        style={{
                          fontSize: "0.8rem",
                          whiteSpace: "pre-line",
                          marginTop: 4,
                        }}
                      >
                        {node.data.tooltip}
                      </div>
                    ) : (
                      <span style={{ fontSize: "0.8rem" }}>
                        Click (i) for Details
                      </span>
                    )}
                  </div>
                ),
              });
            }}
            onNodeMouseMove={(e) => {
              setHoverInfo((prev) =>
                prev ? { ...prev, x: e.clientX, y: e.clientY } : null
              );
            }}
            onNodeMouseLeave={() => setHoverInfo(null)}
            onEdgeMouseEnter={(e, edge) => {
              setHoverInfo({
                x: e.clientX,
                y: e.clientY,
                content: (
                  <div>
                    <strong>Connection</strong>
                    <br />
                    <span style={{ fontSize: "0.8rem" }}>
                      {edge.source} → {edge.target}
                    </span>
                    <br />
                    <span style={{ fontSize: "0.75rem", color: "#888" }}>
                      Type: {(edge as any).data?.type}
                    </span>
                  </div>
                ),
              });
            }}
            onEdgeMouseMove={(e) => {
              setHoverInfo((prev) =>
                prev ? { ...prev, x: e.clientX, y: e.clientY } : null
              );
            }}
            onEdgeMouseLeave={() => setHoverInfo(null)}
            onNodeClick={(e, node) => {
              const target = e.target as HTMLElement;

              // 1. Info Button
              if (target.closest(".cpt-info-btn-node")) {
                if (!cptData || !cptData.cpts) {
                  alert(
                    `No CPT Data loaded for ${node.id}. (Try selecting a model v3)`
                  );
                  return;
                }
                setModalTitle(`Time Model: ${node.id}`);
                setModalData({
                  time: cptData.cpts?.[node.id],
                  prior: cptData.g_priors?.[node.id],
                });
                setModalType("node");
                setModalNodeId(node.id);
                setModalOpen(true);
                return;
              }

              // 2. Scenario Configuration (Tool 1 or Tool 2 Active)
              // Open modal on ANY click if tool is active (Direct Manipulation)
              if (activeTool === "tool1" || activeTool === "tool2") {
                setScenarioModalType("node");
                setScenarioModalData({ id: node.id });
                setScenarioModalOpen(true);
                return;
              }

              // Legacy Scenario Button check (Fallback)
              if (target.closest(".scenario-btn") || target.innerText === "⚙") {
                setScenarioModalType("node");
                setScenarioModalData({ id: node.id });
                setScenarioModalOpen(true);
                return;
              }

              // Default: Search/Highlight
              setSearchTerm(node.id);
              setSelectedNode(node.id);
            }}
            onPaneClick={() => {
              // Clear Search = Clear Graph
              setSearchTerm("");
            }}
            onEdgeClick={(e, edge) => {
              const target = e.target as HTMLElement;

              // 1. Info Button
              let isInfoClick = false;
              if (
                ["circle", "text", "g"].includes(target.tagName) &&
                target.closest(".cpt-info-btn-edge")
              )
                isInfoClick = true;

              if (isInfoClick) {
                if (!cptData || !cptData.resource_cpds) {
                  alert("CPT/Resource Data not loaded.");
                  return;
                }
                const type = (edge as any).data?.type;
                const src = (edge as any).source;
                const tgt = (edge as any).target;
                const keyMap: Record<string, string> = {
                  ac: "k",
                  pilot: "q",
                  cabin: "c",
                  pax: "pax",
                };
                const key = keyMap[type];
                const label = (edge as any).label;

                if (key) {
                  const table = cptData.resource_cpds?.[tgt]?.[key]?.[src];
                  if (table) {
                    setModalTitle(
                      `Connection Model: ${src} -> ${tgt} (${label})`
                    );
                    setModalData(table);
                    setModalType("edge");
                    setModalOpen(true);
                  } else {
                    alert(`No data found.`);
                  }
                }
              }

              // Legacy Scenario Button check
              let isScenarioClick = false;
              if (
                target.closest(".scenario-btn") ||
                (target.tagName === "text" && target.innerHTML === "⚙")
              )
                isScenarioClick = true;

              if (isScenarioClick) {
                setScenarioModalType("edge");
                setScenarioModalData({
                  id: edge.id,
                  source: edge.source,
                  target: edge.target,
                  edgeType: (edge as any).data?.type,
                  tooltip: (edge as any).data?.tooltip,
                });
                setScenarioModalOpen(true);
                return;
              }

              // Default: Highlight Route (Legacy behavior)
              const etype = (edge as any).data?.type;
              if (!etype) return;

              const groupEdgeIds = new Set<string>();
              const startSrc = (edge as any).source;
              const startTgt = (edge as any).target;

              if (etype === "pax") {
                groupEdgeIds.add(edge.id);
              } else {
                const visitedNodes = new Set<string>([startSrc, startTgt]);
                groupEdgeIds.add(edge.id);
                let frontier = [startSrc, startTgt];
                const sameTypeEdges = rfEdges.filter(
                  (e) => (e as any).data?.type === etype
                );

                while (frontier.length) {
                  const next: string[] = [];
                  for (const nId of frontier) {
                    for (const e of sameTypeEdges) {
                      if (groupEdgeIds.has(e.id)) continue;
                      if (e.source === nId || e.target === nId) {
                        groupEdgeIds.add(e.id);
                        const other = e.source === nId ? e.target : e.source;
                        if (!visitedNodes.has(other)) {
                          visitedNodes.add(other);
                          next.push(other);
                        }
                      }
                    }
                  }
                  frontier = next;
                }
              }

              // Apply Edge Highlight (visual only)
              setRfEdges((eds) =>
                eds.map((e) => {
                  if (groupEdgeIds.has(e.id)) {
                    return {
                      ...e,
                      style: { ...e.style, opacity: 1, strokeWidth: 4 },
                    };
                  } else {
                    return {
                      ...e,
                      style: { ...e.style, opacity: 0.1, strokeWidth: 2 },
                    };
                  }
                })
              );
              setRfNodes((nds) =>
                nds.map((n) => ({ ...n, style: { ...n.style, opacity: 1 } }))
              );
            }}
          >
            <Background gap={16} color="#e4ebff" />
            <Controls />
            <MiniMap nodeColor={() => "#ffcc00"} maskColor="#f7f9ff" />
          </ReactFlow>

          {activeTool === "hubturn" && (
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              className="glass-panel absolute top-5 left-5 z-50 w-72 rounded-xl p-4 shadow-2xl"
            >
              <h3 className="mb-3 flex items-center gap-2 text-sm font-bold text-slate-800">
                <RotateCcw className="h-4 w-4 text-amber-600" /> Hub Turnaround
              </h3>

              <div className="space-y-3">
                <div>
                  <label className="text-xs font-semibold text-slate-500 uppercase">
                    Hub
                  </label>
                  <select
                    value={selectedHub}
                    onChange={(e) => setSelectedHub(e.target.value)}
                    className="glass-input mt-1 w-full rounded-lg p-2 text-slate-800"
                  >
                    {availableHubs.map((h) => (
                      <option key={h} value={h} className="bg-white">
                        {h}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-xs font-semibold text-slate-500 uppercase">
                    Improvement ({hubTurnPct}%)
                  </label>
                  <div className="mt-1 flex items-center gap-2">
                    <input
                      type="range"
                      min={0}
                      max={50}
                      step={5}
                      value={hubTurnPct}
                      onChange={(e) =>
                        setHubTurnPct(parseInt(e.target.value, 10))
                      }
                      className="h-1 flex-1 cursor-pointer appearance-none rounded-lg bg-slate-200 accent-sky-500"
                    />
                    <span className="w-8 text-right font-mono text-xs font-bold text-sky-600">
                      {hubTurnPct}%
                    </span>
                  </div>
                </div>

                <button
                  onClick={handleRunHubTurnaround}
                  disabled={hubTurnRunning}
                  className={`flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-bold tracking-wider uppercase transition-all ${
                    hubTurnRunning
                      ? "cursor-not-allowed bg-slate-200 text-slate-500"
                      : "bg-gradient-to-r from-sky-600 to-blue-600 text-white shadow-lg shadow-sky-900/50 hover:from-sky-500 hover:to-blue-500"
                  }`}
                >
                  {hubTurnRunning ? (
                    <>
                      <RotateCcw className="h-3 w-3 animate-spin" /> Applying...
                    </>
                  ) : (
                    <>
                      <Zap className="h-3 w-3" /> Apply & Create Helper
                    </>
                  )}
                </button>

                {hubTurnStatus && (
                  <div
                    className={`rounded border p-2 text-xs ${
                      hubTurnStatus.startsWith("Failed")
                        ? "border-rose-500/20 bg-rose-500/10 text-rose-600"
                        : "border-emerald-500/20 bg-emerald-500/10 text-emerald-600"
                    }`}
                  >
                    {hubTurnStatus}
                  </div>
                )}

                <div className="text-[10px] leading-tight text-slate-500">
                  Writes new CPTs and forks the model.
                </div>
              </div>
            </motion.div>
          )}

          {/* Floating Action Button (Run Tools) */}
          {(activeTool === "tool1" ||
            activeTool === "tool2" ||
            activeTool === "hub" ||
            activeTool === "late" ||
            activeTool === "connection") && (
            <motion.div
              initial={{ y: 50, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              className="pointer-events-none absolute bottom-10 left-1/2 z-50 flex w-full max-w-md -translate-x-1/2 flex-col items-center gap-3"
            >
              {/* Tool-specific inputs - Enable pointer events for children */}
              <div className="pointer-events-auto contents">
                {(activeTool === "tool1" || activeTool === "tool2") && (
                  <label className="glass-panel flex cursor-pointer items-center gap-2 rounded-full border-sky-500/30 px-4 py-2 text-sm font-medium text-slate-800 transition-colors hover:bg-sky-500/10">
                    <input
                      type="checkbox"
                      checked={allTargets}
                      onChange={(e) => setAllTargets(e.target.checked)}
                      className="h-4 w-4 accent-sky-500"
                    />
                    Infer All Targets
                  </label>
                )}

                {activeTool === "hub" && (
                  <div className="glass-panel flex min-w-[280px] flex-col gap-3 rounded-xl p-4">
                    <div className="flex items-center gap-2 text-sm font-bold tracking-wider text-rose-600 uppercase">
                      <AlertTriangle className="h-4 w-4" /> Hub Disruptor
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-[10px] font-bold text-slate-500 uppercase">
                          Hub
                        </label>
                        <select
                          value={selectedHub}
                          onChange={(e) => setSelectedHub(e.target.value)}
                          className="glass-input mt-1 w-full rounded-lg p-2 text-slate-800"
                        >
                          {availableHubs.map((h) => (
                            <option key={h} value={h} className="bg-white">
                              {h}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="text-[10px] font-bold text-slate-500 uppercase">
                          Time
                        </label>
                        <select
                          value={selectedTimebank}
                          onChange={(e) => setSelectedTimebank(e.target.value)}
                          className="glass-input mt-1 w-full rounded-lg p-2 text-slate-800"
                        >
                          {timebankOptions.map((tb) => (
                            <option
                              key={tb.label}
                              value={tb.label}
                              className="bg-white"
                            >
                              {tb.label}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                    <div className="text-[10px] text-slate-500">
                      Forces all departures at hub/timebank to Bin 4 (Severe
                      Delay).
                    </div>
                  </div>
                )}

                {activeTool === "late" && (
                  <div className="glass-panel flex items-center gap-2 rounded-full px-5 py-3 text-sm text-slate-700">
                    <Info className="h-4 w-4 text-sky-600" />
                    <span>
                      Run to view lateness contributors (Hover flights for pie
                      charts).
                    </span>
                  </div>
                )}

                {activeTool === "connection" && (
                  <div className="glass-panel flex items-center gap-2 rounded-full px-5 py-3 text-sm text-slate-700">
                    <Activity className="h-4 w-4 text-amber-600" />
                    <span>
                      Ranking top-10 risky connections via do-calculus.
                    </span>
                  </div>
                )}

                <button
                  onClick={handleRunInference}
                  disabled={isProcessing}
                  className={`flex items-center gap-2 rounded-full px-8 py-3 text-sm font-bold tracking-wider uppercase shadow-lg transition-all ${
                    isProcessing
                      ? "cursor-wait bg-slate-200 text-slate-500"
                      : "bg-gradient-to-r from-sky-500 to-blue-600 text-white shadow-sky-900/40 hover:scale-105 hover:from-sky-400 hover:to-blue-500 hover:shadow-sky-500/20 active:scale-95"
                  }`}
                >
                  {isProcessing ? (
                    <>
                      <RotateCcw className="h-4 w-4 animate-spin" />{" "}
                      Processing...
                    </>
                  ) : (
                    <>
                      <Zap className="h-4 w-4 fill-white" /> Run Inference
                    </>
                  )}
                </button>
              </div>
            </motion.div>
          )}

          {/* Tool 3 Controls & Scorecard */}
          {activeTool === "tool3" && (
            <>
              {/* Control Panel Top-Left */}
              <motion.div
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                className="glass-panel absolute top-5 left-16 z-50 w-64 rounded-xl p-4 shadow-2xl"
              >
                <h3 className="mb-4 flex items-center gap-2 text-sm font-bold text-slate-800">
                  <Settings className="h-4 w-4 text-sky-600" /> Multipliers
                </h3>

                <button
                  onClick={startDMAnalysis}
                  disabled={isProcessing}
                  className={`flex w-full items-center justify-center gap-2 rounded-lg py-2.5 text-xs font-bold tracking-wider uppercase transition-all ${
                    isProcessing
                      ? "cursor-wait bg-slate-200 text-slate-500"
                      : "bg-gradient-to-r from-emerald-500 to-teal-500 text-white shadow-lg shadow-emerald-900/40 hover:from-emerald-400 hover:to-teal-400 hover:shadow-emerald-500/20"
                  }`}
                >
                  {isProcessing ? (
                    "Calculating..."
                  ) : (
                    <>
                      <Zap className="h-3 w-3" /> Run Calculations
                    </>
                  )}
                </button>

                <div className="mt-4">
                  <div className="mb-2 text-xs font-semibold tracking-wide text-slate-600 uppercase">
                    View Mode
                  </div>
                  <div className="space-y-1">
                    {["dmc", "dme", "dmx"].map((m) => (
                      <label
                        key={m}
                        className={`flex cursor-pointer items-center rounded-lg p-2 transition-colors ${
                          dmMode === m
                            ? "border border-sky-500/30 bg-sky-500/20"
                            : "border border-transparent hover:bg-slate-100/80"
                        }`}
                      >
                        <input
                          type="radio"
                          name="dmMode"
                          checked={dmMode === m}
                          onChange={() => setDmMode(m as any)}
                          className="mr-2 h-4 w-4 accent-sky-500"
                        />
                        <span
                          className={`font-mono text-sm font-medium ${dmMode === m ? "text-sky-600" : "text-slate-600"}`}
                        >
                          {m.toUpperCase()}
                        </span>
                      </label>
                    ))}
                  </div>
                </div>
              </motion.div>

              {/* Scorecard Top-Right (Below Tools Panel) */}
              {scorecard && (
                <motion.div
                  initial={{ x: 20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  className={`glass-panel absolute top-24 right-5 z-40 overflow-hidden rounded-xl shadow-2xl transition-all duration-300 ${isScorecardCollapsed ? "w-48" : "max-h-[80vh] w-80"}`}
                >
                  <div className="flex items-center justify-between border-b border-slate-200/70 bg-slate-100/80 p-4">
                    <h3 className="flex items-center gap-2 text-sm font-bold tracking-widest text-slate-800 uppercase">
                      <Activity className="h-4 w-4 text-emerald-600" />{" "}
                      Robustness
                    </h3>
                    <button
                      onClick={() =>
                        setIsScorecardCollapsed(!isScorecardCollapsed)
                      }
                      className="text-slate-600 hover:text-slate-900"
                    >
                      {isScorecardCollapsed ? "▼" : "▲"}
                    </button>
                  </div>

                  <div className="bg-white/70 p-6 text-center">
                    <div
                      className={`text-5xl font-black ${
                        scorecard.robustness_score > 80
                          ? "text-emerald-600 drop-shadow-[0_0_10px_rgba(52,211,153,0.5)]"
                          : scorecard.robustness_score > 50
                            ? "text-amber-600"
                            : "text-rose-500"
                      }`}
                    >
                      {scorecard.robustness_score}
                    </div>
                    <div className="mt-2 text-[10px] font-bold tracking-[0.2em] text-slate-500 uppercase">
                      Score
                    </div>
                  </div>

                  {!isScorecardCollapsed && (
                    <div className="custom-scrollbar max-h-[60vh] space-y-6 overflow-y-auto p-4">
                      <div>
                        <h4 className="mb-3 text-xs font-bold tracking-wide text-slate-600 uppercase">
                          Top K Analysis
                        </h4>
                        <div className="grid grid-cols-3 gap-2">
                          {["dmx", "dme", "dmc"].map((k) => (
                            <button
                              key={k}
                              onClick={() => setActiveTopKModal(k as any)}
                              className="glass-btn rounded px-1 py-2 text-[10px] font-bold uppercase"
                            >
                              {k}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div>
                        <h4 className="mb-3 flex items-center gap-2 text-xs font-bold tracking-wide text-slate-600 uppercase">
                          <AlertTriangle className="h-3 w-3 text-amber-500" />{" "}
                          Risky Hubs
                        </h4>
                        {scorecard.risky_hubs.length > 0 ? (
                          <ul className="space-y-1">
                            {scorecard.risky_hubs.map((h, i) => (
                              <li
                                key={i}
                                className="rounded border border-amber-500/20 bg-amber-500/10 px-2 py-1 text-sm text-amber-200"
                              >
                                {h}
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <div className="text-xs text-slate-600 italic">
                            No risks detected
                          </div>
                        )}
                      </div>

                      <div>
                        <h4 className="mb-3 flex items-center gap-2 text-xs font-bold tracking-wide text-slate-600 uppercase">
                          <Info className="h-3 w-3 text-rose-500" /> Fragile
                          Tails
                        </h4>
                        <ul className="space-y-1">
                          {scorecard.fragile_tails.map((t, i) => (
                            <li
                              key={i}
                              className="rounded border border-rose-500/20 bg-rose-500/10 px-2 py-1 font-mono text-sm text-rose-600"
                            >
                              {t}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
                </motion.div>
              )}

              {/* Modal for Top K */}
              {activeTopKModal && scorecard && (
                <div
                  className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 p-6 backdrop-blur-sm"
                  onClick={() => setActiveTopKModal(null)}
                >
                  <motion.div
                    initial={{ scale: 0.95, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="w-full max-w-lg overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <div className="flex items-center justify-between border-b border-slate-200/60 bg-slate-100/80 p-6">
                      <h3 className="text-lg font-bold text-slate-900">
                        Top 10 Flights - {activeTopKModal.toUpperCase()}
                      </h3>
                      <button
                        onClick={() => setActiveTopKModal(null)}
                        className="text-slate-600 hover:text-slate-900"
                      >
                        <RotateCcw className="h-5 w-5 opacity-0" />X
                      </button>
                    </div>

                    <div className="p-6">
                      <table className="w-full text-left text-sm">
                        <thead className="bg-slate-100/80 text-xs text-slate-500 uppercase">
                          <tr>
                            <th className="rounded-l-lg px-4 py-3">Flight</th>
                            <th className="rounded-r-lg px-4 py-3">Value</th>
                          </tr>
                        </thead>
                        <tbody className="text-slate-700">
                          {scorecard[`top_k_${activeTopKModal}`].map(
                            (item: any, i: number) => (
                              <tr
                                key={i}
                                className="border-b border-slate-200/60 transition-colors hover:bg-slate-100/80"
                              >
                                <td className="px-4 py-3 font-mono text-sky-600">
                                  {item.id}
                                </td>
                                <td className="px-4 py-3 font-bold text-emerald-600">
                                  {item.val.toFixed(2)}
                                </td>
                              </tr>
                            )
                          )}
                        </tbody>
                      </table>
                    </div>

                    <div className="flex justify-end border-t border-slate-200/60 p-4">
                      <button
                        onClick={() => setActiveTopKModal(null)}
                        className="rounded-lg bg-slate-100 px-4 py-2 text-sm font-medium transition-colors hover:bg-slate-200"
                      >
                        Close
                      </button>
                    </div>
                  </motion.div>
                </div>
              )}
            </>
          )}

          {activeTool === "connection" && (
            <motion.div
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              className="glass-panel absolute top-24 right-5 z-40 w-72 rounded-xl shadow-2xl"
            >
              <div className="flex items-center justify-between border-b border-slate-200/70 bg-slate-100/80 p-3">
                <h3 className="text-sm font-bold text-slate-800">
                  Connection Risk (Top 10)
                </h3>
                <button
                  onClick={() =>
                    setIsConnectionCollapsed(!isConnectionCollapsed)
                  }
                  className="text-slate-600 hover:text-slate-900"
                >
                  {isConnectionCollapsed ? "Expand" : "Collapse"}
                </button>
              </div>
              {!isConnectionCollapsed && (
                <div className="max-h-[60vh] space-y-2 overflow-y-auto p-3">
                  {[
                    { key: "ac", label: "Aircraft", color: "bg-blue-500" },
                    { key: "pilot", label: "Pilot", color: "bg-sky-500" },
                    { key: "cabin", label: "Cabin Crew", color: "bg-rose-500" },
                    { key: "pax", label: "Passengers", color: "bg-violet-500" },
                  ].map((cfg) => (
                    <div
                      key={cfg.key}
                      className="rounded-lg border border-slate-200/70 bg-white/70 p-3"
                    >
                      <div className="mb-2 flex items-center gap-2">
                        <span className={`h-2 w-2 rounded-full ${cfg.color}`} />
                        <strong className="text-xs tracking-wider text-slate-700 uppercase">
                          {cfg.label}
                        </strong>
                      </div>
                      <ol className="list-decimal space-y-1 pl-4 text-xs text-slate-600">
                        {(connectionTop[cfg.key] || []).map(
                          (e: any, idx: number) => (
                            <li key={`${cfg.key}-${idx}`}>
                              <span className="text-slate-800">
                                {e.src} → {e.dst}
                              </span>
                              <span className="block text-[10px] text-slate-500">
                                Score: {e.score?.toFixed(2) ?? "0.00"}
                              </span>
                            </li>
                          )
                        )}
                        {(connectionTop[cfg.key] || []).length === 0 && (
                          <li className="italic opacity-50">No data yet</li>
                        )}
                      </ol>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </div>

        {/* Tools Sidebar (Right Pane) */}
        <ToolsPanel
          activeTool={activeTool}
          setActiveTool={(t) => {
            // User Request: Re-clicking tool or switching resets everything
            if (t === "tool1" && initialNetworkData) {
              // Pre-clean if entering Tool 1? Or just allow?
              // Let's reset purely on clear.
              // Logic check: if we switch, we might want to keep current state?
              // User request: "Switching back to default view" -> usually implies resetting.
              const { nodes, edges } = buildGraph(initialNetworkData);
              setRfNodes(nodes);
              setRfEdges(edges);
            }
            setActiveTool(t);
          }}
          onClearAll={() => {
            setEvidence({});
            setTargets(new Set());
            setAllTargets(false);
            setInferenceResults({});
            setScorecard(null);
            setSelectedHub(availableHubs[0] || "");
            setSelectedTimebank(timebankOptions[0].label);
            // Robust Reset: Restore from Golden Copy
            if (initialNetworkData) {
              const { nodes, edges } = buildGraph(initialNetworkData);
              setRfNodes(nodes);
              setRfEdges(edges);
            }
          }}
          onOpenSwap={() => {
            window.open(
              "http://localhost:5173/swap",
              "_blank",
              "noopener,noreferrer"
            );
          }}
        />

        {/* Scenario Configuration Modal */}
        <ScenarioModal
          isOpen={scenarioModalOpen}
          onClose={() => setScenarioModalOpen(false)}
          type={scenarioModalType}
          data={scenarioModalData || { id: "" }}
          currentEvidence={evidence}
          currentTargets={targets}
          inferenceResult={
            scenarioModalType === "node" && scenarioModalData
              ? inferenceResults[scenarioModalData.id]
              : undefined
          }
          onSave={(changes) => {
            if (changes.evidence) setEvidence(changes.evidence);
            if (changes.targets) setTargets(changes.targets);

            // Immediate Visual Feedback (Border Update)
            const newEv = changes.evidence || evidence;
            const newTgt = changes.targets || targets;

            setRfNodes((nds) =>
              nds.map((n) => {
                const tVar = `${n.id}_t`;
                const gVar = `${n.id}_g`;
                const isEv =
                  newEv[tVar] !== undefined || newEv[gVar] !== undefined;
                const isTgt = newTgt.has(tVar);

                let border = "2px solid #d8dff0"; // Default
                if (isEv)
                  border = "4px solid #ef4444"; // Evidence (Red)
                else if (isTgt) border = "4px solid #3b82f6"; // Target (Blue)

                return { ...n, style: { ...n.style, border } };
              })
            );
          }}
        />
      </div>
      <CPTModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        title={modalTitle}
        data={modalData}
        type={modalType}
        nodeId={modalNodeId}
      />

      {/* Tooltip Overlay */}
      {hoverInfo && (
        <div
          className="pointer-events-none fixed z-[9999] rounded-xl border border-slate-200/70 bg-white/95 p-3 text-sm text-slate-900 shadow-xl backdrop-blur-md"
          style={{
            top: hoverInfo.y + 10,
            left: hoverInfo.x + 10,
          }}
        >
          {hoverInfo.content}
        </div>
      )}
    </div>
  );
}


