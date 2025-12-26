import { memo, useCallback, useEffect, useState } from "react";

import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  MarkerType,
  Position,
  getBezierPath,
  useEdgesState,
  useNodesState,
  type Edge,
  type EdgeProps,
  type Node,
} from "reactflow";
import "reactflow/dist/style.css";
import "../App.css";
import CPTModal from "../components/CPTModal";
import {
  Search,
  RotateCcw,
  Network,
  Activity,
  Play,
  ArrowLeftRight,
  ListFilter,
  AlertTriangle,
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

interface CPTData {
  cpts: Record<string, any>;
  g_priors: Record<string, any>;
  resource_cpds: Record<string, any>;
}

type SwapRole = "aircraft" | "pilot" | "crew";

type SwapOption = {
  pred1: string;
  succ1: string;
  pred2: string;
  succ2: string;
  chain1: string;
  chain2: string;
  role: SwapRole;
  benefit: number;
};

type SwapRunResult = {
  key: string;
  variant_id: string;
  benefit: number;
  paths?: { network: string; cpts: string; model: string };
};

// --- Constants ---
const laneGap = 150;
const timeScale = 3;

// Swap server
const SWAP_SERVER_URL = "http://localhost:8001";

// Stable props
const fitoptions = { padding: 0.2 };
const EXTENT: [[number, number], [number, number]] = [
  [-1000, -1000],
  [50000, 5000],
];
const proOpts = { hideAttribution: true };

// --- Custom Edge (info button only) ---
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
          </g>
        )}
      </>
    );
  }
);

const edgeTypes = { offsetBezier: OffsetBezier };

// --- Helpers ---
const formatTime = (minutes: number) => {
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}`;
};

const roleToEdgeType: Record<SwapRole, "ac" | "pilot" | "cabin"> = {
  aircraft: "ac",
  pilot: "pilot",
  crew: "cabin",
};

const vyForType = (type: string) => {
  const order: Record<string, number> = {
    ac: -20,
    pilot: -7,
    cabin: 7,
    pax: 20,
  };
  return order[type] ?? 0;
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
          </div>
        ),
      },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    });
  }

  const addEdge = (
    src?: string | null,
    tgt?: string,
    type?: string,
    color?: string
  ) => {
    if (!src || !tgt) return;
    const vy = vyForType(type || "");

    edges.push({
      id: `${src}->${tgt}-${type}`,
      source: src,
      target: tgt,
      type: "offsetBezier",
      label: type,
      data: { type, vy, color },
      animated: false,
      interactionWidth: 40,
      labelStyle: { fontSize: 10, fill: color || "#1e293b", fontWeight: 700 },
      style: { stroke: color || "#64748b", strokeWidth: 2, opacity: 0.8 }, // Slate-500
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: color || "#64748b",
        width: 10,
        height: 10,
      },
    });
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

const swapKey = (s: SwapOption) =>
  `${s.role}|${s.chain1}|${s.pred1}->${s.succ1}||${s.chain2}|${s.pred2}->${s.succ2}`;

function findEdgeIdByTriple(
  edges: Edge[],
  source: string,
  target: string,
  type: string
) {
  // Prefer canonical ID
  const canonical = `${source}->${target}-${type}`;
  const direct = edges.find((e) => e.id === canonical);
  if (direct) return direct.id;

  // Fallback: match by endpoints + type
  const hit = edges.find(
    (e) =>
      e.source === source &&
      e.target === target &&
      (e as any).data?.type === type
  );
  return hit?.id || null;
}

function applyEnabledSwaps(
  baselineEdges: Edge[],
  swaps: SwapOption[],
  enabledKeys: Set<string>
): Edge[] {
  // Start from baseline each time (so disable = clean restore)
  const edgesArr = baselineEdges.map((e) => ({
    ...e,
    data: { ...(e as any).data },
    style: { ...(e.style || {}) },
  })) as Edge[];

  const edgeById = new Map(edgesArr.map((e) => [e.id, e]));

  const uniqueId = (base: string) => {
    if (!edgeById.has(base)) return base;
    let k = 1;
    while (edgeById.has(`${base}-swap${k}`)) k++;
    return `${base}-swap${k}`;
  };

  for (const s of swaps) {
    const key = swapKey(s);
    if (!enabledKeys.has(key)) continue;

    const etype = roleToEdgeType[s.role];
    const old1Id = findEdgeIdByTriple(
      Array.from(edgeById.values()),
      s.pred1,
      s.succ1,
      etype
    );
    const old2Id = findEdgeIdByTriple(
      Array.from(edgeById.values()),
      s.pred2,
      s.succ2,
      etype
    );

    const old1 = old1Id ? edgeById.get(old1Id) : undefined;
    const old2 = old2Id ? edgeById.get(old2Id) : undefined;

    // Remove old arcs (if present)
    if (old1Id) edgeById.delete(old1Id);
    if (old2Id) edgeById.delete(old2Id);

    // Add new swapped arcs
    const swappedStroke = "#4b0082";
    const mk = (src: string, tgt: string, template?: Edge) => {
      const baseId = `${src}->${tgt}-${etype}`;
      const id = uniqueId(baseId);
      const vy = vyForType(etype);

      const baseStyle = template?.style || {
        stroke: swappedStroke,
        strokeWidth: 5,
        opacity: 1,
      };
      const style = {
        ...baseStyle,
        stroke: swappedStroke,
        strokeWidth: Math.max(Number((baseStyle as any).strokeWidth || 4), 5),
        opacity: 1,
        strokeDasharray: "6 4",
      } as any;

      return {
        id,
        source: src,
        target: tgt,
        type: "offsetBezier",
        label: etype,
        data: {
          ...(template as any)?.data,
          type: etype,
          vy,
          color: swappedStroke,
          swapped: true,
          swap_key: key,
        },
        animated: true,
        interactionWidth: 40,
        labelStyle: { fontSize: 10, fill: "#1d2a4a", fontWeight: 800 },
        style,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: swappedStroke,
          width: 10,
          height: 10,
        },
      } as Edge;
    };

    const new1 = mk(s.pred1, s.succ2, old1);
    const new2 = mk(s.pred2, s.succ1, old2);

    edgeById.set(new1.id, new1);
    edgeById.set(new2.id, new2);
  }

  return Array.from(edgeById.values());
}

export default function SwapPage() {
  const [rfNodes, setRfNodes, onNodesChange] = useNodesState([]);
  const [rfEdges, setRfEdges, onEdgesChange] = useEdgesState([]);

  const [initialNetworkData, setInitialNetworkData] =
    useState<NetworkJson | null>(null);
  const [baselineEdges, setBaselineEdges] = useState<Edge[]>([]);
  const [previewVariantId, setPreviewVariantId] = useState<string>("");

  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");

  const [error, setError] = useState<string | null>(null);

  // CPT Modal State
  const [cptData, setCptData] = useState<CPTData | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [modalTitle, setModalTitle] = useState("");
  const [modalData, setModalData] = useState<any>(null);
  const [modalType, setModalType] = useState<"node" | "edge">("node");
  const [modalNodeId, setModalNodeId] = useState<string>("");

  // Hover tooltip
  const [hoverInfo, setHoverInfo] = useState<{
    x: number;
    y: number;
    content: any;
  } | null>(null);

  // Search/highlight
  const [searchTerm, setSearchTerm] = useState("");
  const [highlightedId, setHighlightedId] = useState<string | null>(null);

  // --- Swap Options Panel state ---
  const [swapRole, setSwapRole] = useState<SwapRole>("aircraft");
  const [minCt, setMinCt] = useState<number>(25);
  const [horizon, setHorizon] = useState<number>(180);
  const [limit, setLimit] = useState<number>(50);

  const [swapsLoading, setSwapsLoading] = useState(false);
  const [swapsError, setSwapsError] = useState<string | null>(null);
  const [swaps, setSwaps] = useState<SwapOption[]>([]);
  const [enabledSwapKeys, setEnabledSwapKeys] = useState<Set<string>>(
    new Set()
  );
  const [runSwapsLoading, setRunSwapsLoading] = useState(false);
  const [swapRunResults, setSwapRunResults] = useState<
    Record<string, SwapRunResult>
  >({});
  const [finalizeSelection, setFinalizeSelection] = useState<string>("");
  const [finalizeBusy, setFinalizeBusy] = useState(false);

  const fetchModels = useCallback(() => {
    fetch(`${SWAP_SERVER_URL}/visualizations?t=${Date.now()}`)
      .then((r) => r.json())
      .then((data) => {
        const list = (data?.visualizations || []) as string[];
        setAvailableModels(list);

        const params = new URLSearchParams(window.location.search);
        const qModel = params.get("model") || "";
        const initial =
          qModel && list.includes(qModel) ? qModel : list[0] || "";

        setSelectedModel((prev) =>
          prev && list.includes(prev) ? prev : initial
        );
      })
      .catch((err) => console.error("Failed to fetch models", err));
  }, []);

  // Fetch model list (from swap server)
  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // Load network + CPT on model change (from swap server)
  useEffect(() => {
    if (!selectedModel) return;

    const url = `${SWAP_SERVER_URL}/visualizations/${selectedModel}/network?t=${Date.now()}`;
    const urlCpt = `${SWAP_SERVER_URL}/visualizations/${selectedModel}/cpts?t=${Date.now()}`;

    setError(null);
    setPreviewVariantId("");

    // Clear swaps and toggles when switching models
    setSwaps([]);
    setEnabledSwapKeys(new Set());
    setSwapsError(null);
    setSwapRunResults({});
    setRunSwapsLoading(false);
    setFinalizeSelection("");

    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((net: NetworkJson) => {
        const { nodes, edges } = buildGraph(net);
        setRfNodes(nodes);
        setRfEdges(edges);
        setBaselineEdges(edges);
        setInitialNetworkData(net);
      })
      .catch((e) => {
        console.error(e);
        setError(`Failed to load ${url}: ${e.message}`);
      });

    fetch(urlCpt)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => setCptData(data || null))
      .catch((e) => console.error("CPT Fetch Error", e));
  }, [selectedModel, setRfNodes, setRfEdges]);

  // Re-apply swaps whenever toggles change (or new swap list arrives)
  useEffect(() => {
    if (!baselineEdges.length) return;
    const next = applyEnabledSwaps(baselineEdges, swaps, enabledSwapKeys);
    setRfEdges(next);
  }, [baselineEdges, swaps, enabledSwapKeys, setRfEdges]);

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

      const connectedEdges = new Set<string>();
      const connectedNodes = new Set<string>([centerNodeId]);

      const traceChain = (type: string) => {
        const queue = [centerNodeId];
        const visited = new Set([centerNodeId]);
        let head = 0;

        while (head < queue.length) {
          const curr = queue[head++];
          rfEdges.forEach((e) => {
            if ((e as any).data?.type !== type) return;

            const isSource = e.source === curr;
            const isTarget = e.target === curr;
            if (!isSource && !isTarget) return;

            if (!connectedEdges.has(e.id)) {
              connectedEdges.add(e.id);
              const other = isSource ? e.target : e.source;
              if (!visited.has(other)) {
                visited.add(other);
                connectedNodes.add(other);
                queue.push(other);
              }
            }
          });
        }
      };

      ["ac", "pilot", "cabin"].forEach((t) => traceChain(t));

      rfEdges.forEach((e) => {
        if (
          (e as any).data?.type === "pax" &&
          (e.source === centerNodeId || e.target === centerNodeId)
        ) {
          connectedEdges.add(e.id);
          connectedNodes.add(e.source);
          connectedNodes.add(e.target);
        }
      });

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
              opacity: isConnected ? 1 : 0.2,
              border: isCenter ? "4px solid #3b82f6" : "2px solid #d8dff0",
            },
          };
        })
      );
    },
    [rfEdges, setRfEdges, setRfNodes]
  );

  // Search effect
  useEffect(() => {
    if (searchTerm && searchTerm.length > 2) {
      const match = rfNodes.find(
        (n) => n.id.toLowerCase() === searchTerm.toLowerCase()
      );
      if (match && match.id !== highlightedId) highlightGraph(match.id);
    } else if (searchTerm === "" && highlightedId !== null) {
      highlightGraph(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchTerm, highlightedId, highlightGraph]);

  const resetView = useCallback(() => {
    setSearchTerm("");
    setHoverInfo(null);
    setHighlightedId(null);
    setPreviewVariantId("");

    // restore edges to "baseline + currently enabled swaps"
    if (initialNetworkData) {
      const { nodes, edges } = buildGraph(initialNetworkData);
      setRfNodes(nodes);
      setBaselineEdges(edges); // triggers reapply via effect
    } else {
      highlightGraph(null);
    }
  }, [initialNetworkData, setRfNodes, setBaselineEdges, highlightGraph]);

  const listSwaps = useCallback(async () => {
    if (!selectedModel) return;

    setSwapsLoading(true);
    setSwapsError(null);

    try {
      const params = new URLSearchParams();
      params.set("role", swapRole);
      params.set("min_ct", String(minCt));
      params.set("horizon", String(horizon));
      params.set("limit", String(limit));

      const url = `${SWAP_SERVER_URL}/swap_options/${encodeURIComponent(
        selectedModel
      )}?${params.toString()}`;
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const data = await resp.json();
      const payload = (data?.swaps || []) as SwapOption[];

      // New list => clear enabled swaps and restore baseline edges
      setEnabledSwapKeys(new Set());
      setSwaps(payload);
    } catch (e: any) {
      console.error(e);
      setSwapsError(e?.message || "Failed to list swaps.");
      setSwaps([]);
      setEnabledSwapKeys(new Set());
    } finally {
      setSwapsLoading(false);
    }
  }, [selectedModel, swapRole, minCt, horizon, limit]);

  const runAllSwaps = useCallback(async () => {
    if (!selectedModel || swaps.length === 0) return;

    setRunSwapsLoading(true);
    setSwapsError(null);

    try {
      const url = `${SWAP_SERVER_URL}/swap_runs/${encodeURIComponent(
        selectedModel
      )}`;
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role: swapRole, swaps }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const data = await resp.json();
      const resultMap: Record<string, SwapRunResult> = {};
      (data?.results || []).forEach((r: SwapRunResult) => {
        if (r?.key) resultMap[r.key] = r;
      });

      setSwapRunResults(resultMap);
      setSwaps((prev) =>
        prev.map((s) => {
          const hit = resultMap[swapKey(s)];
          return hit ? { ...s, benefit: hit.benefit } : s;
        })
      );
      setFinalizeSelection("");
    } catch (e: any) {
      console.error(e);
      setSwapsError(e?.message || "Failed to run swaps.");
    } finally {
      setRunSwapsLoading(false);
    }
  }, [selectedModel, swapRole, swaps]);

  const finalizeVariant = useCallback(async () => {
    if (!finalizeSelection) return;
    const swap = swaps.find((s) => swapKey(s) === finalizeSelection);
    if (!swap) {
      alert("Select a swap to finalize.");
      return;
    }
    setFinalizeBusy(true);
    try {
      const url = `${SWAP_SERVER_URL}/swap_runs/${encodeURIComponent(
        selectedModel
      )}/finalize`;
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ swap, role: swap.role }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      await fetchModels();
      if (data?.finalized_as) {
        setSelectedModel(data.finalized_as);
      }
      alert(`Finalized as ${data?.finalized_as || ""}`);
    } catch (e: any) {
      console.error(e);
      alert(e?.message || "Failed to finalize swap.");
    } finally {
      setFinalizeBusy(false);
    }
  }, [finalizeSelection, swaps, selectedModel, fetchModels]);

  const toggleSwap = useCallback((s: SwapOption) => {
    const key = swapKey(s);
    setEnabledSwapKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-slate-50 font-sans text-slate-900">
      {/* Top Bar */}
      <motion.div
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="glass-panel relative z-50 flex h-16 items-center justify-between px-6"
      >
        <div className="flex items-center gap-4">
          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/20 p-2">
            <ArrowLeftRight className="h-6 w-6 text-emerald-600" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-slate-900">
              Swap Implementer
            </h1>
            <div className="flex items-center gap-2 text-xs text-slate-600">
              <div
                className={`h-2 w-2 rounded-full ${cptData ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "animate-pulse bg-amber-500"}`}
              />
              <span>{cptData ? "CPT Net Active" : "Loading..."}</span>
              {previewVariantId && (
                <span className="ml-2 rounded bg-amber-500 px-1.5 text-[10px] font-bold text-slate-900">
                  Preview: {previewVariantId}
                </span>
              )}
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
            onClick={resetView}
            className="rounded-full p-2 text-slate-600 transition-colors hover:bg-slate-200/80 hover:text-rose-600"
            title="Reset View"
          >
            <RotateCcw className="h-5 w-5" />
          </button>
        </div>
      </motion.div>

      <div className="relative flex flex-1 overflow-hidden">
        {/* Left: Network Switcher */}
        <motion.div
          initial={{ x: -100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="glass-panel z-40 flex w-64 flex-col border-t-0 border-r border-b-0 border-l-0"
        >
          <div className="border-b border-slate-200/70 p-4">
            <h2 className="mb-2 flex items-center gap-2 text-xs font-bold tracking-wider text-slate-500 uppercase">
              <Network className="h-3 w-3" /> Available Networks
            </h2>
          </div>
          <div className="custom-scrollbar flex-1 space-y-1 overflow-y-auto p-2">
            {availableModels.map((m) => (
              <button
                key={m}
                onClick={() => setSelectedModel(m)}
                className={`group flex w-full items-center gap-3 rounded-lg px-4 py-3 text-left text-sm transition-all ${
                  selectedModel === m
                    ? "border border-emerald-500/30 bg-emerald-500/20 text-emerald-600 shadow-[0_0_15px_rgba(16,185,129,0.15)]"
                    : "border border-transparent text-slate-600 hover:bg-slate-100/80 hover:text-slate-800"
                }`}
              >
                <Activity
                  className={`h-4 w-4 ${selectedModel === m ? "text-emerald-600" : "text-slate-600 group-hover:text-slate-600"}`}
                />
                <span className="truncate">{m}</span>
              </button>
            ))}
            {availableModels.length === 0 && (
              <div className="p-4 text-center text-sm text-slate-600 italic">
                No models found...
              </div>
            )}
          </div>
        </motion.div>

        {/* Center: Canvas */}
        <div className="relative flex-1 bg-slate-50">
          {error && (
            <div className="absolute top-5 left-5 z-50 flex items-center gap-2 rounded-lg border border-rose-500/20 bg-rose-500/10 px-4 py-2 text-sm text-rose-600">
              <AlertTriangle className="h-4 w-4" /> {error}
            </div>
          )}

          <ReactFlow
            nodes={rfNodes}
            edges={rfEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            fitView
            fitViewOptions={fitoptions}
            minZoom={0.5}
            maxZoom={4}
            translateExtent={EXTENT}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable
            zoomOnDoubleClick
            edgeTypes={edgeTypes}
            proOptions={proOpts}
            onNodeMouseEnter={(e, n) => {
              const node = rfNodes.find((rn) => rn.id === n.id) || n;
              setHoverInfo({
                x: e.clientX,
                y: e.clientY,
                content: (
                  <div>
                    <strong className="text-sky-600">Flight {node.id}</strong>
                    <br />
                    {node.data?.tooltip ? (
                      <div className="mt-1 text-xs whitespace-pre-line text-slate-600">
                        {node.data.tooltip}
                      </div>
                    ) : (
                      <span className="text-[10px] text-slate-500 italic">
                        Click (i) for Details
                      </span>
                    )}
                  </div>
                ),
              });
            }}
            onNodeMouseMove={(e) =>
              setHoverInfo((prev) =>
                prev ? { ...prev, x: e.clientX, y: e.clientY } : null
              )
            }
            onNodeMouseLeave={() => setHoverInfo(null)}
            onEdgeMouseEnter={(e, edge) => {
              setHoverInfo({
                x: e.clientX,
                y: e.clientY,
                content: (
                  <div>
                    <strong className="text-emerald-600">Connection</strong>
                    <br />
                    <span className="text-xs text-slate-700">
                      {edge.source} → {edge.target}
                    </span>
                    <br />
                    <span className="text-[10px] text-slate-500">
                      Type: {(edge as any).data?.type}
                    </span>
                    {(edge as any).data?.swapped && (
                      <div className="mt-1 text-[10px] font-bold text-emerald-600">
                        Swapped (preview)
                      </div>
                    )}
                  </div>
                ),
              });
            }}
            onEdgeMouseMove={(e) =>
              setHoverInfo((prev) =>
                prev ? { ...prev, x: e.clientX, y: e.clientY } : null
              )
            }
            onEdgeMouseLeave={() => setHoverInfo(null)}
            onPaneClick={() => setSearchTerm("")}
            onNodeClick={(e, node) => {
              const target = e.target as HTMLElement;

              if (target.closest(".cpt-info-btn-node")) {
                if (!cptData || !cptData.cpts) {
                  alert(`No CPT Data loaded for ${node.id}.`);
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

              setSearchTerm(node.id);
            }}
            onEdgeClick={(e, edge) => {
              const target = e.target as HTMLElement;

              const isInfoClick =
                ["circle", "text", "g"].includes(
                  target.tagName.toLowerCase()
                ) && !!target.closest(".cpt-info-btn-edge");

              if (isInfoClick) {
                if (!cptData || !cptData.resource_cpds) {
                  alert("CPT/Resource Data not loaded.");
                  return;
                }
                const type = (edge as any).data?.type;
                const src = edge.source;
                const tgt = edge.target;
                const keyMap: Record<string, string> = {
                  ac: "k",
                  pilot: "q",
                  cabin: "c",
                  pax: "pax",
                };
                const key = keyMap[type];

                if (key) {
                  const table = cptData.resource_cpds?.[tgt]?.[key]?.[src];
                  if (table) {
                    setModalTitle(
                      `Connection Model: ${src} -> ${tgt} (${type})`
                    );
                    setModalData(table);
                    setModalType("edge");
                    setModalOpen(true);
                  } else {
                    alert("No data found.");
                  }
                }
                return;
              }
            }}
          >
            <Background gap={16} color="#334155" />
            <Controls className="!border-slate-200 !bg-slate-100 !fill-slate-400" />
            <MiniMap
              className="!border-slate-200 !bg-white"
              nodeColor={() => "#0ea5e9"}
              maskColor="rgba(15, 23, 42, 0.7)"
            />
          </ReactFlow>

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

        {/* Right: Swap Options Panel */}
        <div className="glass-panel z-40 flex w-[480px] flex-col gap-4 overflow-hidden border-t-0 border-r-0 border-b-0 border-l bg-white/70 p-4">
          <div className="glass-card flex flex-col gap-4 rounded-xl p-4">
            <div className="flex items-center justify-between border-b border-slate-200/70 pb-3">
              <div className="flex items-center gap-2 text-sm font-bold text-slate-900">
                <ListFilter className="h-4 w-4 text-emerald-600" /> Swap Options
              </div>
              <span className="rounded-full bg-slate-100/80 px-2 py-1 text-[10px] font-bold tracking-wider text-slate-600 uppercase">
                Read-Only Model
              </span>
            </div>

            <div className="grid grid-cols-1 gap-3">
              <div className="flex items-center justify-between rounded-lg border border-slate-200/60 bg-slate-100/80 p-2">
                <span className="text-xs font-bold text-slate-600">
                  Target Model
                </span>
                <span className="font-mono text-xs text-emerald-600">
                  {selectedModel || "—"}
                </span>
              </div>

              <div>
                <label className="mb-1 block text-[10px] font-bold text-slate-500 uppercase">
                  Role
                </label>
                <select
                  value={swapRole}
                  onChange={(e) => setSwapRole(e.target.value as SwapRole)}
                  className="glass-input w-full rounded-lg p-2.5 text-slate-800"
                >
                  <option value="aircraft" className="bg-white">
                    aircraft
                  </option>
                  <option value="pilot" className="bg-white">
                    pilot
                  </option>
                  <option value="crew" className="bg-white">
                    crew
                  </option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-1 block text-[10px] font-bold text-slate-500 uppercase">
                    Min CT (min)
                  </label>
                  <input
                    type="number"
                    min={0}
                    value={minCt}
                    onChange={(e) => setMinCt(Number(e.target.value))}
                    className="glass-input w-full rounded-lg p-2"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-[10px] font-bold text-slate-500 uppercase">
                    Horizon (min)
                  </label>
                  <input
                    type="number"
                    min={0}
                    value={horizon}
                    onChange={(e) => setHorizon(Number(e.target.value))}
                    className="glass-input w-full rounded-lg p-2"
                  />
                </div>
              </div>

              <div>
                <label className="mb-1 block text-[10px] font-bold text-slate-500 uppercase">
                  Limit
                </label>
                <input
                  type="number"
                  min={1}
                  value={limit}
                  onChange={(e) => setLimit(Number(e.target.value))}
                  className="glass-input w-full rounded-lg p-2"
                />
              </div>

              <div className="flex gap-2 pt-2">
                <button
                  onClick={listSwaps}
                  disabled={swapsLoading || !selectedModel}
                  className={`flex flex-1 items-center justify-center gap-2 rounded-lg py-2.5 text-xs font-bold tracking-wider uppercase transition-all ${
                    swapsLoading || !selectedModel
                      ? "cursor-not-allowed bg-slate-200 text-slate-500"
                      : "bg-gradient-to-r from-sky-600 to-blue-600 text-white shadow-lg shadow-sky-900/30 hover:from-sky-500 hover:to-blue-500"
                  }`}
                >
                  {swapsLoading ? (
                    <RotateCcw className="h-3 w-3 animate-spin" />
                  ) : (
                    <Search className="h-3 w-3" />
                  )}{" "}
                  List
                </button>
                <button
                  onClick={runAllSwaps}
                  disabled={
                    runSwapsLoading || swaps.length === 0 || !selectedModel
                  }
                  className={`flex flex-1 items-center justify-center gap-2 rounded-lg py-2 text-xs font-bold tracking-wider uppercase transition-all ${
                    runSwapsLoading || swaps.length === 0 || !selectedModel
                      ? "cursor-not-allowed bg-slate-200 text-slate-500"
                      : "bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-lg shadow-emerald-900/30 hover:from-emerald-500 hover:to-teal-500"
                  }`}
                >
                  {runSwapsLoading ? (
                    <RotateCcw className="h-3 w-3 animate-spin" />
                  ) : (
                    <Play className="h-3 w-3" />
                  )}{" "}
                  Run
                </button>
              </div>

              {swapsError && (
                <div className="rounded border border-rose-500/20 bg-rose-500/10 p-2 text-xs text-rose-600">
                  {swapsError}
                </div>
              )}
            </div>
          </div>

          <div className="glass-card flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl">
            <div className="flex items-center justify-between border-b border-slate-200/70 bg-slate-100/80 p-3">
              <div className="text-sm font-bold text-slate-800">Results</div>
              <div className="font-mono text-xs text-slate-500">
                {swapsLoading
                  ? "..."
                  : `${swaps.length} swap${swaps.length === 1 ? "" : "s"}`}
              </div>
            </div>

            <div className="custom-scrollbar flex-1 overflow-auto p-2">
              {swaps.length === 0 ? (
                <div className="mt-10 text-center text-xs text-slate-500 italic">
                  Click <strong>List Swaps</strong> to fetch candidates.
                </div>
              ) : (
                <table className="w-full border-collapse text-left">
                  <thead className="sticky top-0 bg-slate-100/80 text-[10px] font-bold text-slate-500 uppercase backdrop-blur-md">
                    <tr>
                      <th className="p-2">Use</th>
                      <th className="p-2 text-center">Final</th>
                      <th className="p-2">Chain 1</th>
                      <th className="p-2">Chain 2</th>
                      <th className="p-2 text-right">Benefit</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 text-xs text-slate-700">
                    {swaps.map((s, idx) => {
                      const key = swapKey(s);
                      const checked = enabledSwapKeys.has(key);
                      const runMeta = swapRunResults[key];
                      const isSelected = finalizeSelection === key;
                      return (
                        <tr
                          key={`${key}-${idx}`}
                          className={`transition-colors hover:bg-slate-100/80 ${isSelected ? "bg-emerald-500/10" : ""}`}
                        >
                          <td className="p-2">
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => toggleSwap(s)}
                              className="h-4 w-4 cursor-pointer rounded accent-sky-500"
                              title="Preview"
                            />
                          </td>
                          <td className="p-2 text-center">
                            <input
                              type="radio"
                              name="finalize"
                              checked={isSelected}
                              onChange={() => setFinalizeSelection(key)}
                              className="h-4 w-4 cursor-pointer accent-emerald-500"
                            />
                          </td>
                          <td className="p-2 font-mono">
                            <div className="text-sky-600">{s.chain1}</div>
                            <div className="text-[10px] text-slate-500">
                              {s.pred1} → {s.succ1}
                            </div>
                          </td>
                          <td className="p-2 font-mono">
                            <div className="text-amber-600">{s.chain2}</div>
                            <div className="text-[10px] text-slate-500">
                              {s.pred2} → {s.succ2}
                            </div>
                          </td>
                          <td className="p-2 text-right font-mono font-bold text-slate-900">
                            {Number(s.benefit || runMeta?.benefit || 0).toFixed(
                              2
                            )}
                            {runMeta?.variant_id && (
                              <div className="mt-0.5 text-[9px] text-emerald-600">
                                {runMeta.variant_id}
                              </div>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>

            <div className="flex items-center justify-between border-t border-slate-200/70 bg-white/70 p-3">
              <div className="text-[10px] text-slate-500">
                Checked ={" "}
                <span className="font-bold text-emerald-600">Preview</span>
              </div>
              <button
                onClick={finalizeVariant}
                disabled={!finalizeSelection || finalizeBusy}
                className={`rounded-lg px-3 py-1.5 text-xs font-bold tracking-wider uppercase transition-all ${
                  !finalizeSelection || finalizeBusy
                    ? "cursor-not-allowed bg-slate-200 text-slate-500"
                    : "bg-emerald-500 text-white shadow-lg shadow-emerald-900/40 hover:bg-emerald-400"
                }`}
              >
                {finalizeBusy ? "Finalizing..." : "Finalize"}
              </button>
            </div>
          </div>
        </div>
        <CPTModal
          isOpen={modalOpen}
          onClose={() => setModalOpen(false)}
          title={modalTitle}
          data={modalData}
          type={modalType}
          nodeId={modalNodeId}
        />
      </div>
    </div>
  );
}


