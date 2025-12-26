import React from "react";
import {
  Network,
  Activity,
  RotateCcw,
  Zap,
  Search,
  AlertTriangle,
  Info,
  ArrowLeftRight,
} from "lucide-react";

export type ToolMode =
  | "explore"
  | "tool1"
  | "tool2"
  | "tool3"
  | "hub"
  | "late"
  | "connection"
  | "hubturn";

interface ToolsPanelProps {
  activeTool: ToolMode;
  setActiveTool: (t: ToolMode) => void;
  onClearAll: () => void;
  onOpenSwap?: () => void;
}

export default function ToolsPanel({
  activeTool,
  setActiveTool,
  onClearAll,
  onOpenSwap,
}: ToolsPanelProps) {
  const tools: {
    id: ToolMode;
    label: string;
    desc: string;
    icon: React.ElementType;
    color: string;
  }[] = [
    {
      id: "explore",
      label: "Explore",
      desc: "View network without editing",
      icon: Search,
      color: "text-slate-600",
    },
    {
      id: "tool1",
      label: "Propagator",
      desc: "Predict downstream impact (Forward)",
      icon: ArrowLeftRight,
      color: "text-blue-600",
    },
    {
      id: "tool2",
      label: "Diagnostics",
      desc: "Trace root causes (Reverse)",
      icon: Activity,
      color: "text-teal-600",
    },
    {
      id: "tool3",
      label: "Multipliers",
      desc: "Delay Multipliers & Robustness",
      icon: Zap,
      color: "text-amber-600",
    },
    {
      id: "hub",
      label: "Hub Disruptor",
      desc: "Pick hub + timebank, force Bin 4",
      icon: AlertTriangle,
      color: "text-rose-600",
    },
    {
      id: "hubturn",
      label: "Hub Turnaround",
      desc: "Improve hub priors (Scenario A)",
      icon: RotateCcw,
      color: "text-sky-600",
    },
    {
      id: "late",
      label: "Lateness Contributor",
      desc: "Hover pies for k/q/c/p shares",
      icon: Info,
      color: "text-indigo-600",
    },
    {
      id: "connection",
      label: "Connection Risk",
      desc: "Top-10 risky connections",
      icon: Network,
      color: "text-violet-600",
    },
  ];

  return (
    <div className="group fixed top-24 right-5 bottom-8 z-50 flex w-16 flex-col border border-slate-200/70 bg-white/70 shadow-2xl backdrop-blur-md transition-[width] duration-300 hover:w-72 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="flex h-16 shrink-0 items-center border-b border-slate-200/70 px-4">
        <div className="flex items-center gap-3 overflow-hidden">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-sky-500/20 text-sky-600 ring-1 ring-sky-500/50">
            <Zap className="h-5 w-5" />
          </div>
          <div className="min-w-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100">
            <h2 className="whitespace-nowrap text-sm font-bold text-slate-900 uppercase tracking-wider">
              Toolbox
            </h2>
            <p className="whitespace-nowrap text-[10px] text-slate-600">
              Select Analysis Mode
            </p>
          </div>
        </div>
      </div>

      {/* Tools List */}
      <div className="custom-scrollbar flex-1 space-y-1 overflow-y-auto overflow-x-hidden p-2">
        {tools.map((tool) => {
          const Icon = tool.icon;
          const isActive = activeTool === tool.id;
          return (
            <button
              key={tool.id}
              onClick={() => setActiveTool(tool.id)}
              className={`flex w-full items-center gap-3 rounded-xl border border-transparent p-2 text-left transition-all hover:bg-slate-100/80 ${
                isActive ? "bg-slate-100/80" : ""
              }`}
            >
              <div
                className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-slate-100/80 transition-colors ${
                  isActive ? tool.color : "text-slate-600 group-hover:text-slate-600"
                }`}
              >
                <Icon className="h-5 w-5" />
              </div>
              <div className="min-w-0 flex-1 opacity-0 transition-opacity duration-300 group-hover:opacity-100">
                <div
                  className={`truncate text-sm font-bold ${isActive ? "text-slate-800" : "text-slate-600 group-hover:text-slate-700"}`}
                >
                  {tool.label}
                </div>
                <div className="truncate text-[10px] text-slate-600">
                  {tool.desc}
                </div>
              </div>
              {isActive && (
                <div className={`h-2 w-2 shrink-0 rounded-full shadow-[0_0_8px_rgba(255,255,255,0.5)] opacity-0 transition-opacity duration-300 group-hover:opacity-100 ${tool.color.replace('text-', 'bg-')}`} />
              )}
            </button>
          );
        })}

        {/* Swap Implementer */}
        <button
          onClick={() => onOpenSwap && onOpenSwap()}
          className="mt-4 flex w-full items-center gap-3 rounded-xl border border-transparent p-2 text-left transition-all hover:border-slate-200/60 hover:bg-slate-100/80"
        >
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-500">
            <ArrowLeftRight className="h-4 w-4" />
          </div>
          <div className="min-w-0 flex-1 opacity-0 transition-opacity duration-300 group-hover:opacity-100">
            <div className="truncate text-sm font-bold text-slate-700 group-hover:text-slate-900">
              Swap Implementer
            </div>
            <div className="truncate text-[10px] text-slate-600">
              Open swap dashboard
            </div>
          </div>
        </button>
      </div>

      {/* Footer Actions */}
      <div className="border-t border-slate-200/70 p-2">
        <button
          onClick={onClearAll}
          className="flex w-full items-center gap-3 rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-rose-600 transition-colors hover:bg-rose-500/20 hover:text-rose-700"
        >
          <RotateCcw className="h-4 w-4 shrink-0" />
          <span className="whitespace-nowrap text-xs font-bold uppercase tracking-wider opacity-0 transition-opacity duration-300 group-hover:opacity-100">
            Reset View
          </span>
        </button>
      </div>
    </div>
  );
}

