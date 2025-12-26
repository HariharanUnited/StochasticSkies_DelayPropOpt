import { Link } from "react-router-dom";
import { LayoutDashboard, ArrowLeftRight, Plane } from "lucide-react";

const Home = () => {
  return (
    <div className="relative flex min-h-screen w-full flex-col items-center justify-center overflow-hidden bg-slate-50 p-8">
      {/* Background Ambience */}
      <div className="pointer-events-none absolute top-0 left-0 h-full w-full overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] h-[40%] w-[40%] rounded-full bg-blue-500/20 blur-[120px]" />
        <div className="absolute right-[-10%] bottom-[-10%] h-[40%] w-[40%] rounded-full bg-emerald-500/20 blur-[120px]" />
      </div>

      <div className="z-10 mb-12 text-center">
        <div className="mb-6 inline-flex items-center justify-center rounded-2xl bg-white/80 p-3 shadow-2xl ring-1 shadow-sky-200/40 ring-slate-200/70">
          <Plane className="h-12 w-12 text-sky-600" />
        </div>
        <h1 className="mb-4 text-5xl font-black tracking-tight text-slate-900 drop-shadow-xl">
          Aero<span className="text-sky-600">Viz</span>
        </h1>
        <p className="mx-auto max-w-md text-lg text-slate-600">
          Advanced Causal Network Visualization & Optimization for Flight
          Operations
        </p>
      </div>

      <div className="z-10 grid w-full max-w-2xl grid-cols-1 gap-6 md:grid-cols-2">
        <Link
          to="/dashboard"
          className="group glass-panel relative overflow-hidden rounded-2xl border border-slate-200/70 p-6 transition-all hover:border-sky-400/60 hover:shadow-[0_0_30px_rgba(14,165,233,0.2)]"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-sky-500/10 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
          <div className="relative z-10 flex flex-col items-center text-center">
            <LayoutDashboard className="mb-4 h-10 w-10 text-sky-600 transition-transform group-hover:scale-110" />
            <h2 className="mb-2 text-xl font-bold text-slate-900">Dashboard</h2>
            <p className="text-sm text-slate-600">
              Visualize flight networks, analyze delays, and explore causal
              relationships.
            </p>
          </div>
        </Link>

        <Link
          to="/swap"
          className="group glass-panel relative overflow-hidden rounded-2xl border border-slate-200/70 p-6 transition-all hover:border-emerald-400/60 hover:shadow-[0_0_30px_rgba(16,185,129,0.2)]"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/10 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
          <div className="relative z-10 flex flex-col items-center text-center">
            <ArrowLeftRight className="mb-4 h-10 w-10 text-emerald-600 transition-transform group-hover:scale-110" />
            <h2 className="mb-2 text-xl font-bold text-slate-900">
              Swap Optimizer
            </h2>
            <p className="text-sm text-slate-600">
              Manage resource swaps, run simulations, and optimize network
              recovery.
            </p>
          </div>
        </Link>
      </div>

      {/* Feature List */}
      <div className="mt-12 grid w-full max-w-4xl grid-cols-1 gap-8 md:grid-cols-2">
        <div className="glass-panel rounded-xl border border-slate-200/70 bg-white/80 p-6">
          <h3 className="mb-4 text-sm font-bold tracking-widest text-sky-600 uppercase">
            Dashboard Capabilities
          </h3>
          <ul className="space-y-3 text-sm text-slate-700">
            <li className="flex gap-2">
              <span className="text-blue-500">●</span>
              <span>
                <strong className="text-slate-900">Propagator:</strong> Predict
                downstream impact (Forward Analysis)
              </span>
            </li>
            <li className="flex gap-2">
              <span className="text-teal-500">●</span>
              <span>
                <strong className="text-slate-900">Diagnostics:</strong> Trace root
                causes (Backward Inference)
              </span>
            </li>
            <li className="flex gap-2">
              <span className="text-amber-500">●</span>
              <span>
                <strong className="text-slate-900">Multipliers:</strong> Analyze
                delay multipliers & robustness scores
              </span>
            </li>
            <li className="flex gap-2">
              <span className="text-rose-500">●</span>
              <span>
                <strong className="text-slate-900">Hub Disruptor:</strong> Simulate
                major hub disruptions
              </span>
            </li>
          </ul>
        </div>
        <div className="glass-panel rounded-xl border border-slate-200/70 bg-white/80 p-6">
          <h3 className="mb-4 text-sm font-bold tracking-widest text-emerald-600 uppercase">
            Additional Tools
          </h3>
          <ul className="space-y-3 text-sm text-slate-700">
            <li className="flex gap-2">
              <span className="text-sky-600">●</span>
              <span>
                <strong className="text-slate-900">Hub Turnaround:</strong> Optimizing
                ground times and improvement priors
              </span>
            </li>
            <li className="flex gap-2">
              <span className="text-indigo-500">●</span>
              <span>
                <strong className="text-slate-900">Lateness Contributor:</strong>{" "}
                Detailed factor breakdown (k/q/c/p)
              </span>
            </li>
            <li className="flex gap-2">
              <span className="text-violet-500">●</span>
              <span>
                <strong className="text-slate-900">Connection Risk:</strong> Monitor
                top risky connections
              </span>
            </li>
            <li className="flex gap-2">
              <span className="text-emerald-600">●</span>
              <span>
                <strong className="text-slate-900">Swap Implementer:</strong> Execute resource swaps and
                compare benefits
              </span>
            </li>
          </ul>
        </div>
      </div>

      <div className="mt-12 font-mono text-xs text-slate-600">
        v2.0 • Aero Glass Design System
      </div>
    </div>
  );
};

export default Home;

