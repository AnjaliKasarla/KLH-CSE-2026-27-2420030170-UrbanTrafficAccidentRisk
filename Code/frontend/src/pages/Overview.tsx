import {
  Activity,
  ArrowUpRight,
  BrainCircuit,
  Database,
  Gauge,
  GitBranch,
  ShieldCheck,
  Sparkles,
  Target,
  TrendingUp,
  TriangleAlert,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const riskDistribution = [
  { name: "Slight", value: 263280 },
  { name: "Serious", value: 40740 },
  { name: "Fatal", value: 3953 },
];

const trendData = [
  { month: "Jan", slight: 21400, serious: 3500, fatal: 330 },
  { month: "Feb", slight: 20800, serious: 3350, fatal: 310 },
  { month: "Mar", slight: 22600, serious: 3600, fatal: 340 },
  { month: "Apr", slight: 21800, serious: 3450, fatal: 325 },
  { month: "May", slight: 23100, serious: 3650, fatal: 350 },
  { month: "Jun", slight: 22400, serious: 3400, fatal: 332 },
];

const pipeline = [
  {
    icon: Database,
    title: "Data",
    description: "307,972 processed records",
    status: "Ready",
  },
  {
    icon: BrainCircuit,
    title: "ML Model",
    description: "XGBoost multiclass classifier",
    status: "Active",
  },
  {
    icon: Target,
    title: "Explainability",
    description: "SHAP + LIME analysis",
    status: "Active",
  },
  {
    icon: Sparkles,
    title: "AI Layer",
    description: "RAG + grounded LLM",
    status: "Active",
  },
];

const modelMetrics = [
  { label: "Accuracy", value: "59.35%" },
  { label: "Macro F1", value: "34.74%" },
  { label: "Macro Recall", value: "47.49%" },
  { label: "ROC-AUC", value: "65.00%" },
];

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function StatCard({
  icon: Icon,
  label,
  value,
  description,
  accent = "violet",
}: {
  icon: typeof Activity;
  label: string;
  value: string;
  description: string;
  accent?: "violet" | "cyan" | "emerald" | "amber";
}) {
  const accentClasses = {
    violet: "bg-violet-500/10 text-violet-300 ring-violet-400/20",
    cyan: "bg-cyan-500/10 text-cyan-300 ring-cyan-400/20",
    emerald: "bg-emerald-500/10 text-emerald-300 ring-emerald-400/20",
    amber: "bg-amber-500/10 text-amber-300 ring-amber-400/20",
  };

  return (
    <div className="group relative overflow-hidden rounded-2xl border border-white/8 bg-white/[0.035] p-5 backdrop-blur-xl transition-all duration-300 hover:-translate-y-1 hover:border-white/15 hover:bg-white/[0.055]">
      <div className="absolute -right-10 -top-10 h-28 w-28 rounded-full bg-violet-500/10 blur-3xl transition-all duration-500 group-hover:bg-violet-500/20" />

      <div className="relative flex items-start justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-[0.16em] text-slate-500">
            {label}
          </p>
          <p className="mt-3 text-2xl font-semibold tracking-tight text-white">
            {value}
          </p>
          <p className="mt-1 text-xs text-slate-500">{description}</p>
        </div>

        <div
          className={`rounded-xl p-2.5 ring-1 ${accentClasses[accent]}`}
        >
          <Icon size={18} />
        </div>
      </div>
    </div>
  );
}

export default function Overview() {
  const total = riskDistribution.reduce((sum, item) => sum + item.value, 0);

  return (
    <div className="min-h-full space-y-6 pb-10">
      {/* Hero */}
      <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-violet-500/[0.12] via-white/[0.035] to-cyan-500/[0.08] p-7 shadow-2xl shadow-violet-950/20">
        <div className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-violet-500/15 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-28 left-1/3 h-72 w-72 rounded-full bg-cyan-500/10 blur-3xl" />

        <div className="relative flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
          <div className="max-w-3xl">
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-violet-400/20 bg-violet-400/10 px-3 py-1.5 text-xs font-medium text-violet-200">
              <Sparkles size={13} />
              AI-powered traffic risk intelligence
            </div>

            <h1 className="text-3xl font-semibold tracking-tight text-white md:text-4xl">
              Urban Traffic Risk
              <span className="bg-gradient-to-r from-violet-300 via-fuchsia-300 to-cyan-300 bg-clip-text text-transparent">
                {" "}
                Overview
              </span>
            </h1>

            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400">
              A unified view of accident-risk analytics, predictive modeling,
              explainability, and grounded safety intelligence.
            </p>
          </div>

          <div className="flex items-center gap-2 rounded-2xl border border-emerald-400/15 bg-emerald-400/[0.06] px-4 py-3">
            <div className="flex h-2.5 w-2.5 items-center justify-center rounded-full bg-emerald-400 shadow-lg shadow-emerald-400/50">
              <div className="h-1 w-1 rounded-full bg-white" />
            </div>

            <div>
              <p className="text-xs font-medium text-emerald-300">
                System operational
              </p>
              <p className="text-[11px] text-slate-500">
                XGBoost · SHAP · RAG · LLM
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* KPI row */}
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          icon={Database}
          label="Accident Records"
          value={formatNumber(total)}
          description="Processed dataset records"
          accent="violet"
        />

        <StatCard
          icon={Gauge}
          label="Model Accuracy"
          value="59.35%"
          description="Final XGBoost test accuracy"
          accent="cyan"
        />

        <StatCard
          icon={TrendingUp}
          label="ROC-AUC"
          value="65.00%"
          description="Multiclass test evaluation"
          accent="emerald"
        />

        <StatCard
          icon={ShieldCheck}
          label="Risk Classes"
          value="3"
          description="Fatal · Serious · Slight"
          accent="amber"
        />
      </section>

      {/* Main analytics */}
      <section className="grid gap-6 xl:grid-cols-[1.45fr_0.9fr]">
        {/* Trend */}
        <div className="rounded-2xl border border-white/8 bg-white/[0.035] p-5 backdrop-blur-xl">
          <div className="mb-6 flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <Activity size={17} className="text-violet-300" />
                <h2 className="text-sm font-semibold text-white">
                  Risk Activity
                </h2>
              </div>

              <p className="mt-1 text-xs text-slate-500">
                Project-level analytical view
              </p>
            </div>

            <div className="rounded-xl border border-white/8 bg-white/[0.03] px-3 py-2 text-right">
              <p className="text-[10px] uppercase tracking-wider text-slate-600">
                Dataset
              </p>
              <p className="mt-0.5 text-xs font-medium text-slate-300">
                307.9K records
              </p>
            </div>
          </div>

          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={trendData}
                margin={{ top: 10, right: 5, left: -20, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="slightFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopOpacity={0.3} />
                    <stop offset="100%" stopOpacity={0} />
                  </linearGradient>

                  <linearGradient id="seriousFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopOpacity={0.22} />
                    <stop offset="100%" stopOpacity={0} />
                  </linearGradient>

                  <linearGradient id="fatalFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopOpacity={0.18} />
                    <stop offset="100%" stopOpacity={0} />
                  </linearGradient>
                </defs>

                <CartesianGrid
                  vertical={false}
                  stroke="rgba(255,255,255,0.06)"
                />

                <XAxis
                  dataKey="month"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#64748b", fontSize: 11 }}
                />

                <YAxis
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#64748b", fontSize: 10 }}
                />

                <Tooltip
                  contentStyle={{
                    background: "rgba(15, 18, 30, 0.94)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: "14px",
                    color: "#fff",
                    fontSize: "12px",
                  }}
                />

                <Area
                  type="monotone"
                  dataKey="slight"
                  stroke="#a78bfa"
                  strokeWidth={2}
                  fill="url(#slightFill)"
                  name="Slight"
                />

                <Area
                  type="monotone"
                  dataKey="serious"
                  stroke="#22d3ee"
                  strokeWidth={2}
                  fill="url(#seriousFill)"
                  name="Serious"
                />

                <Area
                  type="monotone"
                  dataKey="fatal"
                  stroke="#fb7185"
                  strokeWidth={2}
                  fill="url(#fatalFill)"
                  name="Fatal"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-4 flex flex-wrap gap-5 text-xs">
            <span className="flex items-center gap-2 text-slate-400">
              <span className="h-2 w-2 rounded-full bg-violet-400" />
              Slight
            </span>

            <span className="flex items-center gap-2 text-slate-400">
              <span className="h-2 w-2 rounded-full bg-cyan-400" />
              Serious
            </span>

            <span className="flex items-center gap-2 text-slate-400">
              <span className="h-2 w-2 rounded-full bg-rose-400" />
              Fatal
            </span>
          </div>
        </div>

        {/* Distribution */}
        <div className="rounded-2xl border border-white/8 bg-white/[0.035] p-5 backdrop-blur-xl">
          <div className="mb-2">
            <div className="flex items-center gap-2">
              <Target size={17} className="text-cyan-300" />
              <h2 className="text-sm font-semibold text-white">
                Risk Distribution
              </h2>
            </div>

            <p className="mt-1 text-xs text-slate-500">
              Observed target-class distribution
            </p>
          </div>

          <div className="relative h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskDistribution}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={72}
                  outerRadius={102}
                  paddingAngle={3}
                  stroke="none"
                >
                  {riskDistribution.map((entry) => (
                    <Cell
                      key={entry.name}
                      fill={
                        entry.name === "Slight"
                          ? "#8b5cf6"
                          : entry.name === "Serious"
                            ? "#06b6d4"
                            : "#fb7185"
                      }
                    />
                  ))}
                </Pie>

                <Tooltip
                  contentStyle={{
                    background: "rgba(15, 18, 30, 0.94)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: "14px",
                    color: "#fff",
                    fontSize: "12px",
                  }}
                  formatter={(value) => [
                    formatNumber(Number(value)),
                    "Records",
                  ]}
                />
              </PieChart>
            </ResponsiveContainer>

            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-2xl font-semibold text-white">
                307.9K
              </span>
              <span className="text-[10px] uppercase tracking-[0.15em] text-slate-600">
                Total
              </span>
            </div>
          </div>

          <div className="space-y-2">
            {riskDistribution.map((item) => {
              const percentage = ((item.value / total) * 100).toFixed(2);

              return (
                <div
                  key={item.name}
                  className="flex items-center justify-between rounded-xl border border-white/6 bg-white/[0.025] px-3 py-2.5"
                >
                  <div className="flex items-center gap-2.5">
                    <span
                      className={`h-2 w-2 rounded-full ${
                        item.name === "Slight"
                          ? "bg-violet-400"
                          : item.name === "Serious"
                            ? "bg-cyan-400"
                            : "bg-rose-400"
                      }`}
                    />

                    <span className="text-xs text-slate-300">
                      {item.name}
                    </span>
                  </div>

                  <div className="text-right">
                    <span className="text-xs font-medium text-white">
                      {formatNumber(item.value)}
                    </span>
                    <span className="ml-2 text-[10px] text-slate-600">
                      {percentage}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Pipeline */}
      <section className="rounded-2xl border border-white/8 bg-white/[0.035] p-5 backdrop-blur-xl">
        <div className="mb-5 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <GitBranch size={17} className="text-violet-300" />
              <h2 className="text-sm font-semibold text-white">
                Intelligence Pipeline
              </h2>
            </div>

            <p className="mt-1 text-xs text-slate-500">
              From accident data to explainable safety intelligence
            </p>
          </div>

          <div className="hidden items-center gap-1.5 text-[10px] text-emerald-300 sm:flex">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            All core services operational
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {pipeline.map((item, index) => {
            const Icon = item.icon;

            return (
              <div
                key={item.title}
                className="group relative rounded-2xl border border-white/7 bg-black/10 p-4 transition-all duration-300 hover:border-violet-400/20 hover:bg-white/[0.035]"
              >
                {index < pipeline.length - 1 && (
                  <div className="absolute -right-3 top-1/2 z-10 hidden h-px w-3 bg-gradient-to-r from-violet-400/30 to-transparent xl:block" />
                )}

                <div className="flex items-start justify-between">
                  <div className="rounded-xl border border-white/8 bg-white/[0.04] p-2.5 text-violet-300">
                    <Icon size={17} />
                  </div>

                  <span className="rounded-full border border-emerald-400/15 bg-emerald-400/5 px-2 py-1 text-[9px] font-medium uppercase tracking-wider text-emerald-300">
                    {item.status}
                  </span>
                </div>

                <h3 className="mt-4 text-sm font-medium text-white">
                  {item.title}
                </h3>

                <p className="mt-1 text-xs leading-5 text-slate-500">
                  {item.description}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Model performance + system intelligence */}
      <section className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-2xl border border-white/8 bg-white/[0.035] p-5 backdrop-blur-xl">
          <div className="mb-5 flex items-center gap-2">
            <Gauge size={17} className="text-cyan-300" />
            <div>
              <h2 className="text-sm font-semibold text-white">
                Final Model Performance
              </h2>
              <p className="mt-1 text-xs text-slate-500">
                XGBoost test-set evaluation
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            {modelMetrics.map((metric) => (
              <div
                key={metric.label}
                className="rounded-xl border border-white/7 bg-black/10 p-4"
              >
                <p className="text-[10px] uppercase tracking-wider text-slate-600">
                  {metric.label}
                </p>
                <p className="mt-2 text-xl font-semibold text-white">
                  {metric.value}
                </p>
              </div>
            ))}
          </div>

          <div className="mt-4 rounded-xl border border-amber-400/10 bg-amber-400/[0.035] p-3">
            <div className="flex gap-2.5">
              <TriangleAlert
                size={15}
                className="mt-0.5 shrink-0 text-amber-300"
              />
              <p className="text-[11px] leading-5 text-slate-500">
                The dataset is strongly imbalanced toward the Slight class.
                Macro metrics therefore provide important context beyond
                overall accuracy.
              </p>
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-white/8 bg-white/[0.035] p-5 backdrop-blur-xl">
          <div className="mb-5 flex items-center gap-2">
            <Zap size={17} className="text-violet-300" />
            <div>
              <h2 className="text-sm font-semibold text-white">
                Intelligence Stack
              </h2>
              <p className="mt-1 text-xs text-slate-500">
                How the platform turns predictions into insights
              </p>
            </div>
          </div>

          <div className="space-y-3">
            <div className="rounded-xl border border-violet-400/10 bg-violet-400/[0.035] p-4">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-violet-400/10 p-2 text-violet-300">
                  <BrainCircuit size={16} />
                </div>
                <div>
                  <p className="text-xs font-medium text-white">
                    XGBoost Prediction
                  </p>
                  <p className="mt-0.5 text-[11px] text-slate-500">
                    Produces the accident-risk classification.
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-cyan-400/10 bg-cyan-400/[0.035] p-4">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-cyan-400/10 p-2 text-cyan-300">
                  <Activity size={16} />
                </div>
                <div>
                  <p className="text-xs font-medium text-white">
                    SHAP + LIME
                  </p>
                  <p className="mt-0.5 text-[11px] text-slate-500">
                    Explain model behavior at global and local levels.
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-emerald-400/10 bg-emerald-400/[0.035] p-4">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-emerald-400/10 p-2 text-emerald-300">
                  <Sparkles size={16} />
                </div>
                <div>
                  <p className="text-xs font-medium text-white">
                    RAG + LLM
                  </p>
                  <p className="mt-0.5 text-[11px] text-slate-500">
                    Grounds safety guidance and generates explanations.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-5 flex items-center justify-between border-t border-white/7 pt-4">
            <span className="text-[10px] uppercase tracking-wider text-slate-600">
              Prediction → Explanation → Guidance
            </span>

            <ArrowUpRight size={15} className="text-slate-600" />
          </div>
        </div>
      </section>
    </div>
  );
}