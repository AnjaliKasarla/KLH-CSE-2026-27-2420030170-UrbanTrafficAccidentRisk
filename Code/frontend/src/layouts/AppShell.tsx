import { motion } from "framer-motion";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  ChevronRight,
  Gauge,
  Map,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";

interface AppShellProps {
  children: ReactNode;
}

const navigation = [
  {
    label: "Overview",
    icon: Gauge,
    path: "/",
  },
  {
    label: "Risk Assessment",
    icon: Activity,
    path: "/risk-assessment",
  },
  {
    label: "Risk Intelligence",
    icon: Map,
    path: "/risk-intelligence",
  },
  {
    label: "Analytics",
    icon: BarChart3,
    path: "/analytics",
  },
  {
    label: "AI Explainability",
    icon: BrainCircuit,
    path: "/explainability",
  },
];

const platformNavigation = [
  {
    label: "Safety Intelligence",
    icon: Sparkles,
    path: "/safety-intelligence",
  },
];

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="min-h-screen overflow-x-hidden bg-[#05050a] text-slate-100">
      {/* Ambient background */}
      <div className="ambient-grid" />
      <div className="ambient-glow ambient-glow-purple" />
      <div className="ambient-glow ambient-glow-blue" />
      <div className="ambient-glow ambient-glow-indigo" />

      <div className="relative flex min-h-screen">
        {/* ========================================================= */}
        {/* SIDEBAR */}
        {/* ========================================================= */}

        <aside className="glass-strong fixed inset-y-0 left-0 z-40 hidden w-[260px] border-y-0 border-l-0 lg:flex lg:flex-col">
          {/* Brand */}
          <div className="flex h-20 items-center gap-3 border-b border-white/[0.06] px-6">
            <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-cyan-400 shadow-[0_0_30px_rgba(124,58,237,0.25)]">
              <ShieldCheck className="h-5 w-5 text-white" />

              <div className="absolute inset-0 rounded-xl bg-white/10" />
            </div>

            <div>
              <div className="text-sm font-semibold tracking-tight">
                Urban<span className="text-violet-400">Risk</span>
              </div>

              <div className="text-[10px] uppercase tracking-[0.18em] text-slate-500">
                Intelligence
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-3 py-6">
            {/* Intelligence */}
            <p className="mb-3 px-3 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-600">
              Intelligence
            </p>

            <div className="space-y-1">
              {navigation.map((item) => {
                const Icon = item.icon;

                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    end={item.path === "/"}
                    className={({ isActive }) =>
                      `group relative flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm transition-all duration-300 ${
                        isActive
                          ? "bg-white/[0.07] text-white"
                          : "text-slate-500 hover:bg-white/[0.04] hover:text-slate-200"
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        {/* Active indicator */}
                        {isActive && (
                          <motion.div
                            layoutId="active-navigation"
                            className="absolute inset-y-2 left-0 w-[2px] rounded-full bg-gradient-to-b from-violet-400 to-cyan-400"
                            transition={{
                              type: "spring",
                              stiffness: 400,
                              damping: 30,
                            }}
                          />
                        )}

                        {/* Icon */}
                        <Icon
                          className={`h-[18px] w-[18px] transition-colors ${
                            isActive
                              ? "text-violet-300"
                              : "text-slate-600 group-hover:text-slate-300"
                          }`}
                        />

                        {/* Label */}
                        <span>{item.label}</span>

                        {/* Arrow */}
                        {isActive && (
                          <ChevronRight className="ml-auto h-4 w-4 text-slate-600" />
                        )}
                      </>
                    )}
                  </NavLink>
                );
              })}
            </div>

            {/* Divider */}
            <div className="my-7 h-px bg-white/[0.05]" />

            {/* Platform */}
            <p className="mb-3 px-3 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-600">
              Platform
            </p>

            <div className="space-y-1">
              {platformNavigation.map((item) => {
                const Icon = item.icon;

                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    className={({ isActive }) =>
                      `group relative flex w-full items-center gap-3 rounded-xl px-3 py-3 text-sm transition-all duration-300 ${
                        isActive
                          ? "bg-white/[0.07] text-white"
                          : "text-slate-500 hover:bg-white/[0.04] hover:text-slate-200"
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        {isActive && (
                          <motion.div
                            layoutId="active-platform"
                            className="absolute inset-y-2 left-0 w-[2px] rounded-full bg-gradient-to-b from-violet-400 to-cyan-400"
                          />
                        )}

                        <Icon
                          className={`h-[18px] w-[18px] transition-colors ${
                            isActive
                              ? "text-cyan-300"
                              : "text-slate-600 group-hover:text-cyan-300"
                          }`}
                        />

                        <span>{item.label}</span>

                        {isActive && (
                          <ChevronRight className="ml-auto h-4 w-4 text-slate-600" />
                        )}
                      </>
                    )}
                  </NavLink>
                );
              })}
            </div>
          </nav>

          {/* ======================================================= */}
          {/* SYSTEM STATUS */}
          {/* ======================================================= */}

          <div className="m-4 rounded-2xl border border-white/[0.06] bg-white/[0.025] p-4">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-xs font-medium text-slate-400">
                System Status
              </span>

              <span className="flex items-center gap-1.5 text-[10px] text-emerald-400">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.8)]" />
                Online
              </span>
            </div>

            <div className="text-[11px] leading-relaxed text-slate-600">
              XGBoost · SHAP · RAG · LLM
            </div>
          </div>
        </aside>

        {/* ========================================================= */}
        {/* MAIN */}
        {/* ========================================================= */}

        <main className="min-w-0 flex-1 lg:ml-[260px]">
          {/* Top bar */}
          <header className="glass-strong sticky top-0 z-30 flex h-20 items-center justify-between border-x-0 border-t-0 px-5 md:px-8">
            <div>
              <p className="text-xs font-medium text-slate-500">
                Urban Traffic Risk Intelligence
              </p>

              <h1 className="mt-0.5 text-sm font-semibold text-slate-200">
                AI-powered safety analytics
              </h1>
            </div>

            <div className="flex items-center gap-3">
              {/* Engine status */}
              <div className="hidden items-center gap-2 rounded-full border border-white/[0.07] bg-white/[0.025] px-3 py-1.5 sm:flex">
                <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.8)]" />

                <span className="text-[11px] text-slate-500">
                  Prediction engine ready
                </span>
              </div>

              {/* AI avatar */}
              <div className="flex h-9 w-9 items-center justify-center rounded-full border border-violet-400/20 bg-gradient-to-br from-violet-500/20 to-cyan-400/10 text-xs font-semibold text-violet-200">
                AI
              </div>
            </div>
          </header>

          {/* Page content */}
          <div className="p-5 md:p-8">{children}</div>
        </main>
      </div>
    </div>
  );
}