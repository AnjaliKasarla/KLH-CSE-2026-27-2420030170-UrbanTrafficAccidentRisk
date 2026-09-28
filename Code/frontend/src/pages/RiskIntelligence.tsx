import type { ReactNode } from "react";

import {
  Activity,
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  Database,
  Gauge,
  Layers3,
  ShieldAlert,
  Sparkles,
  Target,
  TrendingUp,
} from "lucide-react";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

/* ================================================================
   DATASET / MODEL METRICS
================================================================ */

const overviewMetrics = [
  {
    label: "Accident Records",
    value: "307,972",
    description: "Validated records",
    icon: Database,
  },
  {
    label: "Risk Classes",
    value: "3",
    description: "Fatal · Serious · Slight",
    icon: ShieldAlert,
  },
  {
    label: "Engineered Features",
    value: "37",
    description: "Temporal + contextual",
    icon: Layers3,
  },
  {
    label: "Active Model",
    value: "XGBoost",
    description: "Final production model",
    icon: BrainCircuit,
  },
];

const performanceMetrics = [
  {
    label: "Accuracy",
    value: "59.35%",
    numeric: 59.35,
    icon: Target,
  },
  {
    label: "Macro F1",
    value: "34.74%",
    numeric: 34.74,
    icon: Gauge,
  },
  {
    label: "Macro Recall",
    value: "47.49%",
    numeric: 47.49,
    icon: Activity,
  },
  {
    label: "ROC-AUC",
    value: "65.00%",
    numeric: 65,
    icon: TrendingUp,
  },
];

const classDistribution = [
  {
    name: "Slight",
    value: 263280,
    percentage: 85.49,
  },
  {
    name: "Serious",
    value: 40740,
    percentage: 13.23,
  },
  {
    name: "Fatal",
    value: 3953,
    percentage: 1.28,
  },
];

const modelComparison = [
  {
    model: "Logistic Regression",
    accuracy: 41.71,
    macroF1: 27.48,
    macroRecall: 43.86,
    rocAuc: 59.39,
  },
  {
    model: "Decision Tree",
    accuracy: 59.42,
    macroF1: 34.12,
    macroRecall: 42.0,
    rocAuc: 52.93,
  },
  {
    model: "Random Forest",
    accuracy: 61.44,
    macroF1: 34.87,
    macroRecall: 47.26,
    rocAuc: 63.77,
  },
  {
    model: "XGBoost",
    accuracy: 60.18,
    macroF1: 35.12,
    macroRecall: 47.16,
    rocAuc: 65.33,
  },
];

const shapDrivers = [
  {
    feature: "Speed Limit",
    importance: 0.140857,
  },
  {
    feature: "Junction Detail",
    importance: 0.054012,
  },
  {
    feature: "Urban / Rural",
    importance: 0.045771,
  },
  {
    feature: "Accident Hour",
    importance: 0.042588,
  },
  {
    feature: "Hour Cos",
    importance: 0.042144,
  },
  {
    feature: "Accident Minute",
    importance: 0.041142,
  },
  {
    feature: "Junction Detail · Roundabout",
    importance: 0.039943,
  },
  {
    feature: "Latitude",
    importance: 0.036346,
  },
  {
    feature: "Accident Day of Year",
    importance: 0.031472,
  },
  {
    feature: "Speed Limit Category",
    importance: 0.029773,
  },
];

const confusionMatrix = [
  {
    actual: "Fatal",
    fatal: 378,
    serious: 196,
    slight: 217,
  },
  {
    actual: "Serious",
    fatal: 1823,
    serious: 2505,
    slight: 3820,
  },
  {
    actual: "Slight",
    fatal: 7653,
    serious: 11329,
    slight: 33674,
  },
];

const pieColors = [
  "url(#slightGradient)",
  "url(#seriousGradient)",
  "url(#fatalGradient)",
];

const tooltipStyle = {
  backgroundColor: "#0b0b12",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: "12px",
  color: "#e2e8f0",
};

/* ================================================================
   PAGE
================================================================ */

export default function RiskIntelligence() {
  return (
    <div className="space-y-8 pb-10">
      {/* ============================================================
          HEADER
      ============================================================ */}

      <section className="relative overflow-hidden rounded-3xl border border-white/[0.06] bg-white/[0.015] p-7 md:p-8">
        <div className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-violet-500/[0.08] blur-3xl" />

        <div className="pointer-events-none absolute -bottom-32 left-1/3 h-64 w-64 rounded-full bg-cyan-400/[0.05] blur-3xl" />

        <div className="relative">
          <div className="mb-4 flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-violet-400/20 bg-violet-500/10">
              <Sparkles className="h-4 w-4 text-violet-300" />
            </div>

            <span className="text-[10px] font-semibold uppercase tracking-[0.24em] text-violet-300">
              Model Intelligence
            </span>

            <span className="ml-2 flex items-center gap-1.5 rounded-full border border-emerald-400/10 bg-emerald-400/5 px-2.5 py-1 text-[9px] uppercase tracking-wider text-emerald-300">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
              Live Model
            </span>
          </div>

          <h1 className="max-w-3xl text-3xl font-semibold tracking-tight text-white md:text-4xl">
            Risk Intelligence
          </h1>

          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
            Dataset-driven intelligence from the trained accident
            severity pipeline. Explore class distribution, model
            performance, confusion patterns, and global SHAP drivers.
          </p>

          <div className="mt-6 flex flex-wrap gap-2">
            <StatusChip
              label="307,972 records"
              icon={<Database className="h-3 w-3" />}
            />

            <StatusChip
              label="37 engineered features"
              icon={<Layers3 className="h-3 w-3" />}
            />

            <StatusChip
              label="XGBoost"
              icon={<BrainCircuit className="h-3 w-3" />}
            />

            <StatusChip
              label="SHAP enabled"
              icon={<Sparkles className="h-3 w-3" />}
            />
          </div>
        </div>
      </section>

      {/* ============================================================
          OVERVIEW METRICS
      ============================================================ */}

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {overviewMetrics.map((metric) => {
          const Icon = metric.icon;

          return (
            <MetricCard
              key={metric.label}
              label={metric.label}
              value={metric.value}
              description={metric.description}
              icon={<Icon className="h-4 w-4" />}
            />
          );
        })}
      </section>

      {/* ============================================================
          PERFORMANCE
      ============================================================ */}

      <section>
        <SectionHeading
          eyebrow="FINAL EVALUATION"
          title="Model Performance"
          description="Final held-out test-set performance of the deployed XGBoost classifier."
          icon={<BarChart3 className="h-4 w-4" />}
        />

        <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {performanceMetrics.map((metric) => {
            const Icon = metric.icon;

            return (
              <PerformanceCard
                key={metric.label}
                label={metric.label}
                value={metric.value}
                numeric={metric.numeric}
                icon={<Icon className="h-4 w-4" />}
              />
            );
          })}
        </div>
      </section>

      {/* ============================================================
          DISTRIBUTION + MODEL SNAPSHOT
      ============================================================ */}

      <section className="grid gap-6 xl:grid-cols-[1fr_1.25fr]">
        {/* Class distribution */}
        <div className="glass rounded-2xl border border-white/[0.06] p-6">
          <SectionHeader
            icon={<ShieldAlert className="h-4 w-4" />}
            title="Risk Class Distribution"
            subtitle="Target distribution in the validated dataset"
          />

          <div className="mt-5 h-[270px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <defs>
                  <linearGradient
                    id="slightGradient"
                    x1="0"
                    y1="0"
                    x2="1"
                    y2="1"
                  >
                    <stop
                      offset="0%"
                      stopColor="#22c55e"
                    />
                    <stop
                      offset="100%"
                      stopColor="#06b6d4"
                    />
                  </linearGradient>

                  <linearGradient
                    id="seriousGradient"
                    x1="0"
                    y1="0"
                    x2="1"
                    y2="1"
                  >
                    <stop
                      offset="0%"
                      stopColor="#f59e0b"
                    />
                    <stop
                      offset="100%"
                      stopColor="#f97316"
                    />
                  </linearGradient>

                  <linearGradient
                    id="fatalGradient"
                    x1="0"
                    y1="0"
                    x2="1"
                    y2="1"
                  >
                    <stop
                      offset="0%"
                      stopColor="#ef4444"
                    />
                    <stop
                      offset="100%"
                      stopColor="#fb7185"
                    />
                  </linearGradient>
                </defs>

                <Pie
                  data={classDistribution}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={72}
                  outerRadius={102}
                  paddingAngle={3}
                  stroke="none"
                >
                  {classDistribution.map((_, index) => (
                    <Cell
                      key={`class-${index}`}
                      fill={pieColors[index]}
                    />
                  ))}
                </Pie>

                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(value) => {
                    if (typeof value === "number") {
                      return value.toLocaleString();
                    }

                    return String(value ?? "");
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-3 gap-3">
            {classDistribution.map((item, index) => (
              <div
                key={item.name}
                className="rounded-xl border border-white/[0.05] bg-white/[0.02] p-3"
              >
                <div className="flex items-center gap-2">
                  <span
                    className={`h-2 w-2 rounded-full ${
                      index === 0
                        ? "bg-cyan-400"
                        : index === 1
                          ? "bg-orange-400"
                          : "bg-rose-400"
                    }`}
                  />

                  <span className="text-[10px] text-slate-500">
                    {item.name}
                  </span>
                </div>

                <p className="mt-2 font-mono text-sm font-semibold text-white">
                  {item.percentage.toFixed(2)}%
                </p>

                <p className="mt-1 text-[9px] text-slate-600">
                  {item.value.toLocaleString()} records
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Final model snapshot */}
        <div className="glass rounded-2xl border border-white/[0.06] p-6">
          <SectionHeader
            icon={<BrainCircuit className="h-4 w-4" />}
            title="Production Model Snapshot"
            subtitle="Final XGBoost test-set evaluation"
          />

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <SnapshotItem
              label="Accuracy"
              value="59.35%"
              description="Overall classification accuracy"
            />

            <SnapshotItem
              label="Macro Precision"
              value="37.00%"
              description="Class-balanced precision"
            />

            <SnapshotItem
              label="Macro Recall"
              value="47.49%"
              description="Class-balanced recall"
            />

            <SnapshotItem
              label="Macro F1"
              value="34.74%"
              description="Balanced F1 across classes"
            />

            <SnapshotItem
              label="Weighted F1"
              value="66.79%"
              description="Support-weighted F1"
            />

            <SnapshotItem
              label="ROC-AUC"
              value="65.00%"
              description="One-vs-rest multiclass AUC"
            />
          </div>

          <div className="mt-5 rounded-xl border border-amber-400/10 bg-amber-400/[0.03] p-4">
            <div className="flex gap-3">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" />

              <p className="text-[11px] leading-5 text-slate-500">
                The target distribution is highly imbalanced,
                with Slight severity representing the majority
                of observations. Macro metrics therefore provide
                important class-balanced context alongside
                overall accuracy.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================
          MODEL COMPARISON
      ============================================================ */}

      <section className="glass rounded-2xl border border-white/[0.06] p-6">
        <SectionHeader
          icon={<BarChart3 className="h-4 w-4" />}
          title="Model Comparison"
          subtitle="Validation performance across the evaluated classifiers"
        />

        <div className="mt-6 h-[340px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={modelComparison}
              margin={{
                top: 10,
                right: 10,
                left: -15,
                bottom: 10,
              }}
            >
              <CartesianGrid
                stroke="rgba(255,255,255,0.04)"
                vertical={false}
              />

              <XAxis
                dataKey="model"
                tick={{
                  fill: "#64748b",
                  fontSize: 10,
                }}
                axisLine={false}
                tickLine={false}
              />

              <YAxis
                domain={[0, 70]}
                tick={{
                  fill: "#64748b",
                  fontSize: 10,
                }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(value) =>
                  `${value}%`
                }
              />

              <Tooltip
                contentStyle={tooltipStyle}
                formatter={(value) => {
                  if (typeof value === "number") {
                    return `${value.toFixed(2)}%`;
                  }

                  return String(value ?? "");
                }}
              />

              <Legend
                wrapperStyle={{
                  color: "#64748b",
                  fontSize: "10px",
                  paddingTop: "15px",
                }}
              />

              <Bar
                dataKey="accuracy"
                name="Accuracy"
                fill="#8b5cf6"
                radius={[4, 4, 0, 0]}
              />

              <Bar
                dataKey="macroF1"
                name="Macro F1"
                fill="#22d3ee"
                radius={[4, 4, 0, 0]}
              />

              <Bar
                dataKey="rocAuc"
                name="ROC-AUC"
                fill="#34d399"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-5 overflow-x-auto">
          <table className="w-full min-w-[680px] border-collapse">
            <thead>
              <tr className="border-b border-white/[0.05] text-left">
                <th className="pb-3 text-[9px] font-semibold uppercase tracking-wider text-slate-600">
                  Model
                </th>

                <th className="pb-3 text-right text-[9px] font-semibold uppercase tracking-wider text-slate-600">
                  Accuracy
                </th>

                <th className="pb-3 text-right text-[9px] font-semibold uppercase tracking-wider text-slate-600">
                  Macro F1
                </th>

                <th className="pb-3 text-right text-[9px] font-semibold uppercase tracking-wider text-slate-600">
                  Macro Recall
                </th>

                <th className="pb-3 text-right text-[9px] font-semibold uppercase tracking-wider text-slate-600">
                  ROC-AUC
                </th>
              </tr>
            </thead>

            <tbody>
              {modelComparison.map((model) => (
                <tr
                  key={model.model}
                  className="border-b border-white/[0.035] last:border-0"
                >
                  <td className="py-3 text-xs text-slate-300">
                    <div className="flex items-center gap-2">
                      {model.model === "XGBoost" && (
                        <span className="h-1.5 w-1.5 rounded-full bg-violet-400 shadow-[0_0_8px_rgba(167,139,250,0.8)]" />
                      )}

                      {model.model}
                    </div>
                  </td>

                  <td className="py-3 text-right font-mono text-xs text-slate-400">
                    {model.accuracy.toFixed(2)}%
                  </td>

                  <td className="py-3 text-right font-mono text-xs text-slate-400">
                    {model.macroF1.toFixed(2)}%
                  </td>

                  <td className="py-3 text-right font-mono text-xs text-slate-400">
                    {model.macroRecall.toFixed(2)}%
                  </td>

                  <td className="py-3 text-right font-mono text-xs text-slate-400">
                    {model.rocAuc.toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ============================================================
          SHAP GLOBAL IMPORTANCE
      ============================================================ */}

      <section className="glass rounded-2xl border border-white/[0.06] p-6">
        <SectionHeader
          icon={<Sparkles className="h-4 w-4" />}
          title="Global Model Drivers"
          subtitle="Mean absolute SHAP contribution across the explainability sample"
        />

        <div className="mt-6 grid gap-8 xl:grid-cols-[1fr_1.3fr]">
          <div className="space-y-3">
            {shapDrivers.map((item, index) => {
              const width =
                (item.importance /
                  shapDrivers[0].importance) *
                100;

              return (
                <div key={item.feature}>
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <div className="flex min-w-0 items-center gap-2">
                      <span className="w-5 shrink-0 font-mono text-[9px] text-slate-700">
                        {String(index + 1).padStart(2, "0")}
                      </span>

                      <span className="truncate text-xs text-slate-300">
                        {item.feature}
                      </span>
                    </div>

                    <span className="shrink-0 font-mono text-[10px] text-violet-300">
                      {item.importance.toFixed(4)}
                    </span>
                  </div>

                  <div className="ml-7 h-1.5 overflow-hidden rounded-full bg-white/[0.04]">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-violet-500 via-indigo-400 to-cyan-400 transition-all duration-700"
                      style={{
                        width: `${width}%`,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="rounded-2xl border border-violet-400/[0.08] bg-violet-500/[0.025] p-5">
            <div className="flex items-start gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/10">
                <BrainCircuit className="h-4 w-4 text-violet-300" />
              </div>

              <div>
                <h4 className="text-sm font-semibold text-white">
                  Explainability Signal
                </h4>

                <p className="mt-1 text-[11px] leading-5 text-slate-500">
                  SHAP identifies which features have the
                  strongest influence on the trained model's
                  outputs across the explanation sample.
                </p>
              </div>
            </div>

            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              <InsightCard
                label="Top Driver"
                value="Speed Limit"
                description="Mean |SHAP| = 0.1409"
              />

              <InsightCard
                label="Temporal Signal"
                value="Accident Hour"
                description="Mean |SHAP| = 0.0426"
              />

              <InsightCard
                label="Spatial Signal"
                value="Latitude"
                description="Mean |SHAP| = 0.0363"
              />

              <InsightCard
                label="Road Context"
                value="Junction Detail"
                description="Mean |SHAP| = 0.0540"
              />
            </div>

            <div className="mt-5 border-t border-white/[0.05] pt-4">
              <p className="text-[10px] leading-5 text-slate-600">
                These values represent model behavior and
                feature contribution, not causal relationships.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================
          CONFUSION MATRIX
      ============================================================ */}

      <section className="glass rounded-2xl border border-white/[0.06] p-6">
        <SectionHeader
          icon={<Target className="h-4 w-4" />}
          title="Confusion Matrix"
          subtitle="Final XGBoost test-set predictions"
        />

        <div className="mt-6 grid gap-6 lg:grid-cols-[1fr_0.8fr]">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[520px] border-separate border-spacing-2">
              <thead>
                <tr>
                  <th className="p-3 text-left text-[9px] uppercase tracking-wider text-slate-700">
                    Actual \ Predicted
                  </th>

                  <th className="rounded-lg bg-white/[0.02] p-3 text-center text-[9px] uppercase tracking-wider text-slate-500">
                    Fatal
                  </th>

                  <th className="rounded-lg bg-white/[0.02] p-3 text-center text-[9px] uppercase tracking-wider text-slate-500">
                    Serious
                  </th>

                  <th className="rounded-lg bg-white/[0.02] p-3 text-center text-[9px] uppercase tracking-wider text-slate-500">
                    Slight
                  </th>
                </tr>
              </thead>

              <tbody>
                {confusionMatrix.map((row) => (
                  <tr key={row.actual}>
                    <td className="rounded-lg border border-white/[0.04] bg-white/[0.015] p-3 text-xs font-medium text-slate-400">
                      {row.actual}
                    </td>

                    <MatrixCell
                      value={row.fatal}
                      strongest={row.actual === "Fatal"}
                    />

                    <MatrixCell
                      value={row.serious}
                      strongest={
                        row.actual === "Serious"
                      }
                    />

                    <MatrixCell
                      value={row.slight}
                      strongest={
                        row.actual === "Slight"
                      }
                    />
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="rounded-2xl border border-white/[0.05] bg-white/[0.015] p-5">
            <h4 className="text-sm font-semibold text-white">
              Reading the Matrix
            </h4>

            <div className="mt-4 space-y-3">
              <LegendRow
                label="Correct Fatal"
                value="378"
                color="bg-rose-400"
              />

              <LegendRow
                label="Correct Serious"
                value="2,505"
                color="bg-orange-400"
              />

              <LegendRow
                label="Correct Slight"
                value="33,674"
                color="bg-cyan-400"
              />
            </div>

            <div className="mt-5 rounded-xl border border-amber-400/10 bg-amber-400/[0.03] p-4">
              <div className="flex gap-3">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" />

                <p className="text-[10px] leading-5 text-slate-500">
                  The confusion matrix shows substantial
                  cross-class confusion, particularly between
                  Serious and Slight cases. This reflects the
                  difficulty of separating minority severity
                  classes in the imbalanced dataset.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================
          ARCHITECTURE SIGNAL
      ============================================================ */}

      <section className="rounded-2xl border border-violet-400/[0.08] bg-gradient-to-r from-violet-500/[0.04] via-transparent to-cyan-400/[0.03] p-6">
        <div className="flex flex-col gap-5 md:flex-row md:items-center md:justify-between">
          <div className="flex items-start gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-violet-400/10 bg-violet-500/10">
              <Activity className="h-5 w-5 text-violet-300" />
            </div>

            <div>
              <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-violet-300">
                Intelligence Stack
              </p>

              <h3 className="mt-1 text-base font-semibold text-white">
                Predict → Explain → Ground → Assist
              </h3>

              <p className="mt-1 max-w-2xl text-xs leading-5 text-slate-500">
                XGBoost performs the risk classification.
                SHAP explains model behavior. RAG retrieves
                relevant safety knowledge. The LLM converts
                these signals into grounded user-facing
                explanations.
              </p>
            </div>
          </div>

          <div className="flex shrink-0 flex-wrap gap-2">
            <StackBadge label="XGBoost" />
            <StackBadge label="SHAP" />
            <StackBadge label="RAG" />
            <StackBadge label="LLM" />
          </div>
        </div>
      </section>
    </div>
  );
}

/* ================================================================
   COMPONENTS
================================================================ */

function SectionHeading({
  eyebrow,
  title,
  description,
  icon,
}: {
  eyebrow: string;
  title: string;
  description: string;
  icon: ReactNode;
}) {
  return (
    <div className="flex items-end justify-between gap-5">
      <div>
        <div className="mb-2 flex items-center gap-2">
          <span className="text-violet-300">{icon}</span>

          <span className="text-[9px] font-semibold uppercase tracking-[0.2em] text-slate-600">
            {eyebrow}
          </span>
        </div>

        <h2 className="text-xl font-semibold text-white">
          {title}
        </h2>

        <p className="mt-1 text-xs text-slate-500">
          {description}
        </p>
      </div>
    </div>
  );
}

function SectionHeader({
  icon,
  title,
  subtitle,
}: {
  icon: ReactNode;
  title: string;
  subtitle: string;
}) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-500/10">
        <span className="text-violet-300">{icon}</span>
      </div>

      <div>
        <h3 className="text-sm font-semibold text-white">
          {title}
        </h3>

        <p className="mt-0.5 text-[11px] text-slate-500">
          {subtitle}
        </p>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  description,
  icon,
}: {
  label: string;
  value: string;
  description: string;
  icon: ReactNode;
}) {
  return (
    <div className="glass group rounded-2xl border border-white/[0.06] p-5 transition-all duration-300 hover:border-violet-400/[0.12] hover:bg-white/[0.025]">
      <div className="flex items-start justify-between">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-500/10 text-violet-300">
          {icon}
        </div>

        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.7)]" />
      </div>

      <p className="mt-5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
        {label}
      </p>

      <p className="mt-1 text-2xl font-semibold tracking-tight text-white">
        {value}
      </p>

      <p className="mt-1 text-[10px] text-slate-500">
        {description}
      </p>
    </div>
  );
}

function PerformanceCard({
  label,
  value,
  numeric,
  icon,
}: {
  label: string;
  value: string;
  numeric: number;
  icon: ReactNode;
}) {
  return (
    <div className="glass rounded-2xl border border-white/[0.06] p-5">
      <div className="flex items-center justify-between">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-400/10 text-cyan-300">
          {icon}
        </div>

        <span className="font-mono text-[9px] text-slate-600">
          TEST
        </span>
      </div>

      <p className="mt-4 text-[10px] uppercase tracking-[0.15em] text-slate-600">
        {label}
      </p>

      <div className="mt-1 flex items-end justify-between">
        <span className="text-2xl font-semibold text-white">
          {value}
        </span>

        <span className="mb-1 text-[9px] text-slate-600">
          {numeric.toFixed(2)}
        </span>
      </div>

      <div className="mt-4 h-1 overflow-hidden rounded-full bg-white/[0.04]">
        <div
          className="h-full rounded-full bg-gradient-to-r from-violet-500 to-cyan-400"
          style={{
            width: `${Math.min(numeric, 100)}%`,
          }}
        />
      </div>
    </div>
  );
}

function SnapshotItem({
  label,
  value,
  description,
}: {
  label: string;
  value: string;
  description: string;
}) {
  return (
    <div className="rounded-xl border border-white/[0.05] bg-white/[0.02] p-4">
      <p className="text-[9px] uppercase tracking-[0.14em] text-slate-600">
        {label}
      </p>

      <p className="mt-1 font-mono text-lg font-semibold text-white">
        {value}
      </p>

      <p className="mt-1 text-[9px] leading-4 text-slate-600">
        {description}
      </p>
    </div>
  );
}

function InsightCard({
  label,
  value,
  description,
}: {
  label: string;
  value: string;
  description: string;
}) {
  return (
    <div className="rounded-xl border border-white/[0.05] bg-black/10 p-3">
      <p className="text-[9px] uppercase tracking-wider text-slate-600">
        {label}
      </p>

      <p className="mt-1 text-xs font-semibold text-slate-200">
        {value}
      </p>

      <p className="mt-1 text-[9px] text-slate-600">
        {description}
      </p>
    </div>
  );
}

function MatrixCell({
  value,
  strongest,
}: {
  value: number;
  strongest: boolean;
}) {
  return (
    <td
      className={`rounded-lg border p-4 text-center ${
        strongest
          ? "border-violet-400/10 bg-violet-500/[0.10]"
          : "border-white/[0.04] bg-white/[0.015]"
      }`}
    >
      <span
        className={`font-mono text-sm font-semibold ${
          strongest
            ? "text-violet-200"
            : "text-slate-500"
        }`}
      >
        {value.toLocaleString()}
      </span>
    </td>
  );
}

function LegendRow({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-white/[0.04] bg-white/[0.015] px-3 py-2.5">
      <div className="flex items-center gap-2">
        <span
          className={`h-2 w-2 rounded-full ${color}`}
        />

        <span className="text-[10px] text-slate-500">
          {label}
        </span>
      </div>

      <span className="font-mono text-[10px] text-slate-300">
        {value}
      </span>
    </div>
  );
}

function StatusChip({
  label,
  icon,
}: {
  label: string;
  icon: ReactNode;
}) {
  return (
    <div className="flex items-center gap-2 rounded-full border border-white/[0.06] bg-white/[0.025] px-3 py-1.5 text-[9px] text-slate-500">
      <span className="text-violet-300">{icon}</span>
      {label}
    </div>
  );
}

function StackBadge({ label }: { label: string }) {
  return (
    <span className="rounded-full border border-violet-400/10 bg-violet-500/[0.05] px-3 py-1.5 text-[9px] font-medium uppercase tracking-wider text-violet-300">
      {label}
    </span>
  );
}