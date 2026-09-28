import type { ReactNode } from "react";

import {
  Activity,
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  Database,
  Gauge,
  Layers3,
  Target,
  TrendingUp,
  Zap,
} from "lucide-react";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

/* ================================================================
   REAL PROJECT RESULTS
   These values come from the trained/evaluated project artifacts.
================================================================ */

const riskDistribution = [
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
    macroPrecision: 28.0,
    macroRecall: 43.86,
    macroF1: 27.48,
    rocAuc: 59.39,
  },
  {
    model: "Decision Tree",
    accuracy: 59.42,
    macroPrecision: 36.0,
    macroRecall: 42.0,
    macroF1: 34.12,
    rocAuc: 52.93,
  },
  {
    model: "Random Forest",
    accuracy: 61.44,
    macroPrecision: 37.0,
    macroRecall: 47.26,
    macroF1: 34.87,
    rocAuc: 63.77,
  },
  {
    model: "XGBoost",
    accuracy: 60.18,
    macroPrecision: 37.0,
    macroRecall: 47.16,
    macroF1: 35.12,
    rocAuc: 65.33,
  },
];

const finalMetrics = [
  {
    label: "Accuracy",
    value: 59.35,
    display: "59.35%",
    icon: Target,
  },
  {
    label: "Macro Precision",
    value: 37.0,
    display: "37.00%",
    icon: Gauge,
  },
  {
    label: "Macro Recall",
    value: 47.49,
    display: "47.49%",
    icon: Activity,
  },
  {
    label: "Macro F1",
    value: 34.74,
    display: "34.74%",
    icon: Zap,
  },
  {
    label: "Weighted F1",
    value: 66.79,
    display: "66.79%",
    icon: TrendingUp,
  },
  {
    label: "ROC-AUC",
    value: 65.0,
    display: "65.00%",
    icon: BrainCircuit,
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

const classMetrics = [
  {
    className: "Fatal",
    precision: 3.84,
    recall: 47.79,
    f1: 7.1,
    color: "bg-rose-400",
    text: "text-rose-300",
  },
  {
    className: "Serious",
    precision: 17.86,
    recall: 30.74,
    f1: 22.59,
    color: "bg-orange-400",
    text: "text-orange-300",
  },
  {
    className: "Slight",
    precision: 89.3,
    recall: 63.95,
    f1: 74.53,
    color: "bg-cyan-400",
    text: "text-cyan-300",
  },
];

/*
 * Mean absolute SHAP importance values from the project's
 * global explainability analysis.
 */
const shapImportance = [
  {
    feature: "Speed Limit",
    importance: 0.140857,
  },
  {
    feature: "Junction Detail · Not at junction",
    importance: 0.054012,
  },
  {
    feature: "Urban / Rural · Rural",
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
    feature: "Speed Limit Category · Low",
    importance: 0.029773,
  },
  {
    feature: "Weather Category · Clear",
    importance: 0.02659,
  },
  {
    feature: "Accident Week",
    importance: 0.020426,
  },
];

const rocComparison = [
  {
    model: "Logistic Regression",
    rocAuc: 59.39,
    macroF1: 27.48,
  },
  {
    model: "Decision Tree",
    rocAuc: 52.93,
    macroF1: 34.12,
  },
  {
    model: "Random Forest",
    rocAuc: 63.77,
    macroF1: 34.87,
  },
  {
    model: "XGBoost",
    rocAuc: 65.33,
    macroF1: 35.12,
  },
];

const chartTooltipStyle = {
  backgroundColor: "#090910",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: "12px",
  color: "#e2e8f0",
};

const pieColors = [
  "#22d3ee",
  "#f97316",
  "#fb7185",
];

/* ================================================================
   PAGE
================================================================ */

export default function Analytics() {
  return (
    <div className="space-y-8 pb-12">
      {/* ============================================================
          HERO
      ============================================================ */}

      <section className="relative overflow-hidden rounded-3xl border border-white/[0.06] bg-white/[0.015] p-7 md:p-8">
        <div className="pointer-events-none absolute -right-32 -top-32 h-80 w-80 rounded-full bg-cyan-500/[0.07] blur-3xl" />

        <div className="pointer-events-none absolute -bottom-32 left-1/3 h-72 w-72 rounded-full bg-violet-500/[0.07] blur-3xl" />

        <div className="relative">
          <div className="mb-4 flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-cyan-400/20 bg-cyan-400/10">
              <BarChart3 className="h-4 w-4 text-cyan-300" />
            </div>

            <span className="text-[10px] font-semibold uppercase tracking-[0.24em] text-cyan-300">
              Analytics Engine
            </span>

            <span className="ml-2 flex items-center gap-1.5 rounded-full border border-emerald-400/10 bg-emerald-400/5 px-2.5 py-1 text-[9px] uppercase tracking-wider text-emerald-300">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
              Dataset Verified
            </span>
          </div>

          <h1 className="text-3xl font-semibold tracking-tight text-white md:text-4xl">
            Analytics
          </h1>

          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
            A consolidated analytical view of accident severity,
            model performance, classification behavior, and global
            feature importance across the trained risk engine.
          </p>

          <div className="mt-6 flex flex-wrap gap-2">
            <AnalyticsChip
              icon={<Database className="h-3 w-3" />}
              label="307,972 records"
            />

            <AnalyticsChip
              icon={<Layers3 className="h-3 w-3" />}
              label="37 features"
            />

            <AnalyticsChip
              icon={<BrainCircuit className="h-3 w-3" />}
              label="XGBoost"
            />

            <AnalyticsChip
              icon={<Activity className="h-3 w-3" />}
              label="SHAP"
            />
          </div>
        </div>
      </section>

      {/* ============================================================
          TOP KPIs
      ============================================================ */}

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KpiCard
          label="Total Records"
          value="307,972"
          description="Final validated accident records"
          icon={<Database className="h-4 w-4" />}
        />

        <KpiCard
          label="Risk Classes"
          value="3"
          description="Fatal · Serious · Slight"
          icon={<Target className="h-4 w-4" />}
        />

        <KpiCard
          label="Final Accuracy"
          value="59.35%"
          description="Held-out test set"
          icon={<Gauge className="h-4 w-4" />}
        />

        <KpiCard
          label="ROC-AUC"
          value="65.00%"
          description="Multiclass one-vs-rest"
          icon={<TrendingUp className="h-4 w-4" />}
        />
      </section>

      {/* ============================================================
          RISK DISTRIBUTION
      ============================================================ */}

      <section className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
        <Panel
          title="Risk Distribution"
          subtitle="Observed target distribution in the validated dataset"
          icon={<Target className="h-4 w-4" />}
        >
          <div className="mt-4 h-[310px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskDistribution}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={78}
                  outerRadius={112}
                  paddingAngle={3}
                  stroke="none"
                >
                  {riskDistribution.map((_, index) => (
                    <Cell
                      key={`risk-${index}`}
                      fill={pieColors[index]}
                    />
                  ))}
                </Pie>

                <Tooltip
                  contentStyle={chartTooltipStyle}
                  formatter={(value) =>
                    typeof value === "number"
                      ? value.toLocaleString()
                      : String(value ?? "")
                  }
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-3 gap-3">
            {riskDistribution.map((item, index) => (
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

                <p className="mt-2 text-lg font-semibold text-white">
                  {item.percentage.toFixed(2)}%
                </p>

                <p className="mt-1 text-[9px] text-slate-600">
                  {item.value.toLocaleString()} records
                </p>
              </div>
            ))}
          </div>
        </Panel>

        <Panel
          title="Class Balance Signal"
          subtitle="Severity distribution and implications for evaluation"
          icon={<AlertTriangle className="h-4 w-4" />}
        >
          <div className="mt-5 space-y-5">
            <DistributionRow
              label="Slight"
              value={85.49}
              count="263,280"
              color="from-cyan-500 to-emerald-400"
            />

            <DistributionRow
              label="Serious"
              value={13.23}
              count="40,740"
              color="from-amber-500 to-orange-400"
            />

            <DistributionRow
              label="Fatal"
              value={1.28}
              count="3,953"
              color="from-red-500 to-rose-400"
            />
          </div>

          <div className="mt-7 rounded-xl border border-amber-400/10 bg-amber-400/[0.03] p-4">
            <div className="flex gap-3">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" />

              <div>
                <p className="text-xs font-medium text-amber-200">
                  Imbalanced target
                </p>

                <p className="mt-1 text-[10px] leading-5 text-slate-500">
                  Slight severity represents the majority of
                  observations. Macro-level metrics therefore
                  provide additional class-balanced evaluation
                  context beyond overall accuracy.
                </p>
              </div>
            </div>
          </div>
        </Panel>
      </section>

      {/* ============================================================
          MODEL COMPARISON
      ============================================================ */}

      <Panel
        title="Model Comparison"
        subtitle="Validation metrics across the evaluated classifiers"
        icon={<BrainCircuit className="h-4 w-4" />}
      >
        <div className="mt-6 h-[370px]">
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
                contentStyle={chartTooltipStyle}
                formatter={(value) =>
                  typeof value === "number"
                    ? `${value.toFixed(2)}%`
                    : String(value ?? "")
                }
              />

              <Legend
                wrapperStyle={{
                  fontSize: "10px",
                  color: "#64748b",
                  paddingTop: "15px",
                }}
              />

              <Bar
                dataKey="accuracy"
                name="Accuracy"
                fill="#8b5cf6"
                radius={[5, 5, 0, 0]}
              />

              <Bar
                dataKey="macroF1"
                name="Macro F1"
                fill="#22d3ee"
                radius={[5, 5, 0, 0]}
              />

              <Bar
                dataKey="rocAuc"
                name="ROC-AUC"
                fill="#34d399"
                radius={[5, 5, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-5 overflow-x-auto">
          <table className="w-full min-w-[760px] border-collapse">
            <thead>
              <tr className="border-b border-white/[0.05]">
                <th className="pb-3 text-left text-[9px] uppercase tracking-wider text-slate-600">
                  Model
                </th>

                <th className="pb-3 text-right text-[9px] uppercase tracking-wider text-slate-600">
                  Accuracy
                </th>

                <th className="pb-3 text-right text-[9px] uppercase tracking-wider text-slate-600">
                  Macro Precision
                </th>

                <th className="pb-3 text-right text-[9px] uppercase tracking-wider text-slate-600">
                  Macro Recall
                </th>

                <th className="pb-3 text-right text-[9px] uppercase tracking-wider text-slate-600">
                  Macro F1
                </th>

                <th className="pb-3 text-right text-[9px] uppercase tracking-wider text-slate-600">
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
                    {model.macroPrecision.toFixed(2)}%
                  </td>

                  <td className="py-3 text-right font-mono text-xs text-slate-400">
                    {model.macroRecall.toFixed(2)}%
                  </td>

                  <td className="py-3 text-right font-mono text-xs text-slate-400">
                    {model.macroF1.toFixed(2)}%
                  </td>

                  <td className="py-3 text-right font-mono text-xs text-slate-400">
                    {model.rocAuc.toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* ============================================================
          ROC-AUC / MACRO F1
      ============================================================ */}

      <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <Panel
          title="ROC-AUC vs Macro-F1"
          subtitle="Comparison of discrimination and class-balanced performance"
          icon={<TrendingUp className="h-4 w-4" />}
        >
          <div className="mt-6 h-[330px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={rocComparison}
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
                    fontSize: 9,
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
                  contentStyle={chartTooltipStyle}
                  formatter={(value) =>
                    typeof value === "number"
                      ? `${value.toFixed(2)}%`
                      : String(value ?? "")
                  }
                />

                <Legend
                  wrapperStyle={{
                    fontSize: "10px",
                    paddingTop: "15px",
                  }}
                />

                <Bar
                  dataKey="rocAuc"
                  name="ROC-AUC"
                  fill="#8b5cf6"
                  radius={[5, 5, 0, 0]}
                />

                <Line
                  type="monotone"
                  dataKey="macroF1"
                  name="Macro F1"
                  stroke="#22d3ee"
                  strokeWidth={2}
                  dot={{
                    r: 4,
                    fill: "#22d3ee",
                    strokeWidth: 0,
                  }}
                />

                <ReferenceLine
                  y={50}
                  stroke="rgba(255,255,255,0.08)"
                  strokeDasharray="4 4"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel
          title="Final Test Metrics"
          subtitle="XGBoost held-out test evaluation"
          icon={<Gauge className="h-4 w-4" />}
        >
          <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
            {finalMetrics.map((metric) => {
              const Icon = metric.icon;

              return (
                <div
                  key={metric.label}
                  className="flex items-center gap-3 rounded-xl border border-white/[0.05] bg-white/[0.015] p-3"
                >
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-500/10 text-violet-300">
                    <Icon className="h-4 w-4" />
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-[10px] text-slate-500">
                        {metric.label}
                      </span>

                      <span className="font-mono text-xs font-semibold text-white">
                        {metric.display}
                      </span>
                    </div>

                    <div className="mt-2 h-1 overflow-hidden rounded-full bg-white/[0.04]">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-violet-500 to-cyan-400"
                        style={{
                          width: `${metric.value}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </Panel>
      </section>

      {/* ============================================================
          CLASS-WISE METRICS
      ============================================================ */}

      <Panel
        title="Class-wise Performance"
        subtitle="Precision, recall and F1 for each severity class"
        icon={<Layers3 className="h-4 w-4" />}
      >
        <div className="mt-6 grid gap-4 md:grid-cols-3">
          {classMetrics.map((item) => (
            <div
              key={item.className}
              className="rounded-2xl border border-white/[0.05] bg-white/[0.015] p-5"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span
                    className={`h-2.5 w-2.5 rounded-full ${item.color}`}
                  />

                  <span className="text-sm font-semibold text-white">
                    {item.className}
                  </span>
                </div>

                <span
                  className={`text-[9px] uppercase tracking-wider ${item.text}`}
                >
                  Test
                </span>
              </div>

              <div className="mt-5 space-y-4">
                <MetricProgress
                  label="Precision"
                  value={item.precision}
                />

                <MetricProgress
                  label="Recall"
                  value={item.recall}
                />

                <MetricProgress
                  label="F1 Score"
                  value={item.f1}
                />
              </div>
            </div>
          ))}
        </div>
      </Panel>

      {/* ============================================================
          CONFUSION MATRIX
      ============================================================ */}

      <Panel
        title="Confusion Matrix"
        subtitle="Final XGBoost test-set classification behavior"
        icon={<Target className="h-4 w-4" />}
      >
        <div className="mt-6 grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] border-separate border-spacing-2">
              <thead>
                <tr>
                  <th className="p-3 text-left text-[9px] uppercase tracking-wider text-slate-700">
                    Actual / Predicted
                  </th>

                  <th className="rounded-lg bg-white/[0.025] p-3 text-center text-[9px] uppercase tracking-wider text-rose-300/70">
                    Fatal
                  </th>

                  <th className="rounded-lg bg-white/[0.025] p-3 text-center text-[9px] uppercase tracking-wider text-orange-300/70">
                    Serious
                  </th>

                  <th className="rounded-lg bg-white/[0.025] p-3 text-center text-[9px] uppercase tracking-wider text-cyan-300/70">
                    Slight
                  </th>
                </tr>
              </thead>

              <tbody>
                {confusionMatrix.map((row) => (
                  <tr key={row.actual}>
                    <td className="rounded-lg border border-white/[0.04] bg-white/[0.015] p-4 text-xs font-medium text-slate-400">
                      {row.actual}
                    </td>

                    <MatrixCell
                      value={row.fatal}
                      highlighted={row.actual === "Fatal"}
                    />

                    <MatrixCell
                      value={row.serious}
                      highlighted={
                        row.actual === "Serious"
                      }
                    />

                    <MatrixCell
                      value={row.slight}
                      highlighted={
                        row.actual === "Slight"
                      }
                    />
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="rounded-2xl border border-white/[0.05] bg-white/[0.015] p-5">
            <h3 className="text-sm font-semibold text-white">
              Classification Signal
            </h3>

            <p className="mt-2 text-[10px] leading-5 text-slate-500">
              The diagonal cells represent correctly classified
              observations. Off-diagonal cells represent
              misclassification between severity classes.
            </p>

            <div className="mt-5 space-y-3">
              <MatrixSummary
                label="Correct Fatal"
                value="378"
                color="bg-rose-400"
              />

              <MatrixSummary
                label="Correct Serious"
                value="2,505"
                color="bg-orange-400"
              />

              <MatrixSummary
                label="Correct Slight"
                value="33,674"
                color="bg-cyan-400"
              />
            </div>

            <div className="mt-5 rounded-xl border border-amber-400/10 bg-amber-400/[0.03] p-4">
              <div className="flex gap-3">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" />

                <p className="text-[10px] leading-5 text-slate-500">
                  Serious and Slight observations show substantial
                  cross-class confusion, consistent with the
                  imbalanced target distribution.
                </p>
              </div>
            </div>
          </div>
        </div>
      </Panel>

      {/* ============================================================
          SHAP GLOBAL IMPORTANCE
      ============================================================ */}

      <Panel
        title="SHAP Global Importance"
        subtitle="Mean absolute SHAP contribution across the explainability sample"
        icon={<BrainCircuit className="h-4 w-4" />}
      >
        <div className="mt-6 grid gap-8 xl:grid-cols-[1.2fr_0.8fr]">
          <div className="space-y-3">
            {shapImportance.map((item, index) => {
              const width =
                (item.importance /
                  shapImportance[0].importance) *
                100;

              return (
                <div key={item.feature}>
                  <div className="mb-2 flex items-center justify-between gap-4">
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
                      className="h-full rounded-full bg-gradient-to-r from-violet-500 via-indigo-400 to-cyan-400"
                      style={{
                        width: `${Math.max(
                          3,
                          width,
                        )}%`,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="rounded-2xl border border-violet-400/[0.08] bg-violet-500/[0.025] p-5">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-violet-500/10">
                <BrainCircuit className="h-5 w-5 text-violet-300" />
              </div>

              <div>
                <p className="text-[9px] font-semibold uppercase tracking-[0.18em] text-violet-300">
                  Explainability
                </p>

                <h3 className="mt-1 text-sm font-semibold text-white">
                  Global model drivers
                </h3>

                <p className="mt-2 text-[10px] leading-5 text-slate-500">
                  SHAP measures how features contribute to the
                  trained model's predictions. Larger mean absolute
                  values indicate stronger influence on model output.
                </p>
              </div>
            </div>

            <div className="mt-6 grid gap-3">
              <DriverCard
                rank="01"
                feature="Speed Limit"
                value="0.140857"
              />

              <DriverCard
                rank="02"
                feature="Junction Detail"
                value="0.054012"
              />

              <DriverCard
                rank="03"
                feature="Urban / Rural"
                value="0.045771"
              />

              <DriverCard
                rank="04"
                feature="Accident Hour"
                value="0.042588"
              />
            </div>

            <div className="mt-5 border-t border-white/[0.05] pt-4">
              <p className="text-[9px] leading-5 text-slate-600">
                SHAP values describe model behavior and feature
                contribution. They should not be interpreted as
                causal effects.
              </p>
            </div>
          </div>
        </div>
      </Panel>

      {/* ============================================================
          ANALYTICS SUMMARY
      ============================================================ */}

      <section className="rounded-2xl border border-violet-400/[0.08] bg-gradient-to-r from-violet-500/[0.04] via-transparent to-cyan-400/[0.03] p-6">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-start gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-violet-400/10 bg-violet-500/10">
              <Activity className="h-5 w-5 text-violet-300" />
            </div>

            <div>
              <p className="text-[9px] font-semibold uppercase tracking-[0.2em] text-violet-300">
                Analytics Summary
              </p>

              <h3 className="mt-1 text-base font-semibold text-white">
                Model performance + explainability
              </h3>

              <p className="mt-1 max-w-3xl text-xs leading-5 text-slate-500">
                The analytics layer combines dataset composition,
                classifier evaluation, class-wise diagnostics,
                confusion analysis, and global SHAP importance into
                a single evaluation surface.
              </p>
            </div>
          </div>

          <div className="flex shrink-0 flex-wrap gap-2">
            <StackBadge label="DATASET" />
            <StackBadge label="ML" />
            <StackBadge label="METRICS" />
            <StackBadge label="SHAP" />
          </div>
        </div>
      </section>
    </div>
  );
}

/* ================================================================
   REUSABLE COMPONENTS
================================================================ */

function Panel({
  title,
  subtitle,
  icon,
  children,
}: {
  title: string;
  subtitle: string;
  icon: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="glass rounded-2xl border border-white/[0.06] p-6">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-500/10">
          <span className="text-violet-300">{icon}</span>
        </div>

        <div>
          <h2 className="text-sm font-semibold text-white">
            {title}
          </h2>

          <p className="mt-0.5 text-[11px] text-slate-500">
            {subtitle}
          </p>
        </div>
      </div>

      {children}
    </section>
  );
}

function AnalyticsChip({
  icon,
  label,
}: {
  icon: ReactNode;
  label: string;
}) {
  return (
    <div className="flex items-center gap-2 rounded-full border border-white/[0.06] bg-white/[0.025] px-3 py-1.5 text-[9px] text-slate-500">
      <span className="text-violet-300">{icon}</span>
      {label}
    </div>
  );
}

function KpiCard({
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
    <div className="glass group rounded-2xl border border-white/[0.06] p-5 transition-all duration-300 hover:border-violet-400/[0.12]">
      <div className="flex items-start justify-between">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-500/10 text-violet-300">
          {icon}
        </div>

        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
      </div>

      <p className="mt-5 text-[9px] font-semibold uppercase tracking-[0.16em] text-slate-600">
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

function DistributionRow({
  label,
  value,
  count,
  color,
}: {
  label: string;
  value: number;
  count: string;
  color: string;
}) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-300">
            {label}
          </span>

          <span className="text-[9px] text-slate-600">
            {count}
          </span>
        </div>

        <span className="font-mono text-xs text-slate-400">
          {value.toFixed(2)}%
        </span>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-white/[0.04]">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${color}`}
          style={{
            width: `${value}%`,
          }}
        />
      </div>
    </div>
  );
}

function MetricProgress({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <span className="text-[10px] text-slate-500">
          {label}
        </span>

        <span className="font-mono text-[10px] text-slate-300">
          {value.toFixed(2)}%
        </span>
      </div>

      <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.04]">
        <div
          className="h-full rounded-full bg-gradient-to-r from-violet-500 to-cyan-400"
          style={{
            width: `${Math.min(value, 100)}%`,
          }}
        />
      </div>
    </div>
  );
}

function MatrixCell({
  value,
  highlighted,
}: {
  value: number;
  highlighted: boolean;
}) {
  return (
    <td
      className={`rounded-lg border p-4 text-center ${
        highlighted
          ? "border-violet-400/10 bg-violet-500/[0.10]"
          : "border-white/[0.04] bg-white/[0.015]"
      }`}
    >
      <span
        className={`font-mono text-sm font-semibold ${
          highlighted
            ? "text-violet-200"
            : "text-slate-500"
        }`}
      >
        {value.toLocaleString()}
      </span>
    </td>
  );
}

function MatrixSummary({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="flex items-center justify-between rounded-xl border border-white/[0.04] bg-white/[0.015] px-3 py-3">
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

function DriverCard({
  rank,
  feature,
  value,
}: {
  rank: string;
  feature: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-white/[0.05] bg-black/10 p-3">
      <span className="w-5 font-mono text-[9px] text-slate-700">
        {rank}
      </span>

      <div className="min-w-0 flex-1">
        <p className="truncate text-[11px] text-slate-300">
          {feature}
        </p>
      </div>

      <span className="font-mono text-[10px] text-violet-300">
        {value}
      </span>
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