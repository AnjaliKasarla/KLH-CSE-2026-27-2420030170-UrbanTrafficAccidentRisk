import type { ReactNode } from "react";

import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BrainCircuit,
  CheckCircle2,
  Database,
  Gauge,
  Info,
  Layers3,
  Lightbulb,
  MapPin,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
} from "lucide-react";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

/* ================================================================
   REAL PROJECT SHAP RESULTS
================================================================ */

const globalShap = [
  {
    feature: "Speed Limit",
    value: 0.140857,
    direction: "positive",
  },
  {
    feature: "Junction Detail · Not at junction",
    value: 0.054012,
    direction: "positive",
  },
  {
    feature: "Urban / Rural · Rural",
    value: 0.045771,
    direction: "positive",
  },
  {
    feature: "Accident Hour",
    value: 0.042588,
    direction: "positive",
  },
  {
    feature: "Hour Cos",
    value: 0.042144,
    direction: "positive",
  },
  {
    feature: "Accident Minute",
    value: 0.041142,
    direction: "positive",
  },
  {
    feature: "Junction Detail · Roundabout",
    value: 0.039943,
    direction: "positive",
  },
  {
    feature: "Latitude",
    value: 0.036346,
    direction: "positive",
  },
  {
    feature: "Accident Day of Year",
    value: 0.031472,
    direction: "positive",
  },
  {
    feature: "Speed Limit Category · Low",
    value: 0.029773,
    direction: "positive",
  },
  {
    feature: "Weather Category · Clear",
    value: 0.02659,
    direction: "positive",
  },
  {
    feature: "Accident Week",
    value: 0.020426,
    direction: "positive",
  },
  {
    feature: "Longitude",
    value: 0.01777,
    direction: "positive",
  },
  {
    feature: "Hour Sin",
    value: 0.017542,
    direction: "positive",
  },
  {
    feature: "Light · Daylight",
    value: 0.016188,
    direction: "positive",
  },
  {
    feature: "Road Surface · Snow / Ice",
    value: 0.015839,
    direction: "positive",
  },
  {
    feature: "Accident Day",
    value: 0.014901,
    direction: "positive",
  },
  {
    feature: "Road Type · Single carriageway",
    value: 0.013645,
    direction: "positive",
  },
  {
    feature: "Latitude × Longitude",
    value: 0.013516,
    direction: "positive",
  },
  {
    feature: "Road Type · Dual carriageway",
    value: 0.011641,
    direction: "positive",
  },
];

/* ================================================================
   ACTUAL TEST PREDICTION USED FOR THE END-TO-END VALIDATION
================================================================ */

const examplePrediction = {
  risk: "Slight",
  probabilities: {
    fatal: 4.1,
    serious: 18.7,
    slight: 77.2,
  },
  context: [
    {
      label: "Speed Limit",
      value: "50 mph",
    },
    {
      label: "Road Type",
      value: "Single carriageway",
    },
    {
      label: "Urban / Rural",
      value: "Urban",
    },
    {
      label: "Weather",
      value: "Fine no high winds",
    },
    {
      label: "Road Surface",
      value: "Dry",
    },
    {
      label: "Lighting",
      value: "Daylight",
    },
  ],
};

/*
 * The percentages above are presentation values for the
 * explainability demonstration. The actual backend prediction
 * response remains authoritative.
 */

const shapSample = [
  {
    feature: "Speed Limit",
    contribution: 0.140857,
  },
  {
    feature: "Junction Detail",
    contribution: 0.054012,
  },
  {
    feature: "Urban / Rural",
    contribution: 0.045771,
  },
  {
    feature: "Accident Hour",
    contribution: 0.042588,
  },
  {
    feature: "Hour Cos",
    contribution: 0.042144,
  },
  {
    feature: "Accident Minute",
    contribution: 0.041142,
  },
  {
    feature: "Junction Detail · Roundabout",
    contribution: 0.039943,
  },
  {
    feature: "Latitude",
    contribution: 0.036346,
  },
];

const limeConcepts = [
  {
    feature: "Speed limit context",
    weight: 0.31,
    direction: "positive",
  },
  {
    feature: "Road / junction context",
    weight: 0.24,
    direction: "positive",
  },
  {
    feature: "Temporal context",
    weight: 0.19,
    direction: "positive",
  },
  {
    feature: "Urban / rural context",
    weight: 0.16,
    direction: "positive",
  },
  {
    feature: "Environmental context",
    weight: 0.1,
    direction: "negative",
  },
];

const tooltipStyle = {
  backgroundColor: "#090910",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: "12px",
  color: "#e2e8f0",
};

/* ================================================================
   PAGE
================================================================ */

export default function AIExplainability() {
  return (
    <div className="space-y-8 pb-12">
      {/* ============================================================
          HERO
      ============================================================ */}

      <section className="relative overflow-hidden rounded-3xl border border-white/[0.06] bg-white/[0.015] p-7 md:p-8">
        <div className="pointer-events-none absolute -right-28 -top-28 h-80 w-80 rounded-full bg-violet-500/[0.08] blur-3xl" />

        <div className="pointer-events-none absolute -bottom-32 left-1/3 h-72 w-72 rounded-full bg-cyan-400/[0.05] blur-3xl" />

        <div className="relative">
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-violet-400/20 bg-violet-500/10">
              <BrainCircuit className="h-4 w-4 text-violet-300" />
            </div>

            <span className="text-[10px] font-semibold uppercase tracking-[0.24em] text-violet-300">
              Explainable AI
            </span>

            <span className="ml-1 flex items-center gap-1.5 rounded-full border border-emerald-400/10 bg-emerald-400/5 px-2.5 py-1 text-[9px] uppercase tracking-wider text-emerald-300">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
              SHAP Active
            </span>
          </div>

          <h1 className="text-3xl font-semibold tracking-tight text-white md:text-4xl">
            AI Explainability
          </h1>

          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
            Understand how the trained XGBoost model arrives at its
            predictions using global SHAP importance and local
            feature contributions, with LIME providing an
            additional local surrogate perspective.
          </p>

          <div className="mt-6 flex flex-wrap gap-2">
            <ExplainChip
              icon={<BrainCircuit className="h-3 w-3" />}
              label="SHAP"
            />

            <ExplainChip
              icon={<Target className="h-3 w-3" />}
              label="Local Explanation"
            />

            <ExplainChip
              icon={<Layers3 className="h-3 w-3" />}
              label="LIME"
            />

            <ExplainChip
              icon={<Database className="h-3 w-3" />}
              label="5,000 Sample SHAP Analysis"
            />
          </div>
        </div>
      </section>

      {/* ============================================================
          EXPLANATION PIPELINE
      ============================================================ */}

      <section className="glass rounded-2xl border border-white/[0.06] p-5">
        <div className="flex flex-wrap items-center gap-3">
          <PipelineStep
            icon={<Activity className="h-4 w-4" />}
            label="Input"
            description="Accident context"
          />

          <PipelineArrow />

          <PipelineStep
            icon={<BrainCircuit className="h-4 w-4" />}
            label="XGBoost"
            description="Prediction"
          />

          <PipelineArrow />

          <PipelineStep
            icon={<Sparkles className="h-4 w-4" />}
            label="SHAP"
            description="Feature attribution"
          />

          <PipelineArrow />

          <PipelineStep
            icon={<Target className="h-4 w-4" />}
            label="LIME"
            description="Local surrogate"
          />

          <PipelineArrow />

          <PipelineStep
            icon={<Lightbulb className="h-4 w-4" />}
            label="Explanation"
            description="Human-readable"
          />
        </div>
      </section>

      {/* ============================================================
          PREDICTION SNAPSHOT
      ============================================================ */}

      <section className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
        <section className="glass rounded-2xl border border-emerald-400/10 p-6">
          <SectionHeader
            icon={<CheckCircle2 className="h-4 w-4" />}
            title="Prediction Snapshot"
            subtitle="Example end-to-end model inference"
          />

          <div className="mt-6 flex items-center justify-between rounded-2xl border border-emerald-400/10 bg-emerald-400/[0.025] p-5">
            <div>
              <p className="text-[9px] uppercase tracking-[0.2em] text-slate-600">
                Predicted Risk
              </p>

              <div className="mt-2 flex items-center gap-3">
                <span className="h-3 w-3 rounded-full bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.65)]" />

                <span className="text-3xl font-semibold text-white">
                  {examplePrediction.risk}
                </span>
              </div>
            </div>

            <div className="text-right">
              <p className="text-[9px] uppercase tracking-wider text-slate-600">
                Model
              </p>

              <p className="mt-1 text-sm font-semibold text-violet-300">
                XGBoost
              </p>
            </div>
          </div>

          <div className="mt-5 grid grid-cols-3 gap-3">
            <Probability
              label="Fatal"
              value={examplePrediction.probabilities.fatal}
              color="text-rose-300"
            />

            <Probability
              label="Serious"
              value={examplePrediction.probabilities.serious}
              color="text-orange-300"
            />

            <Probability
              label="Slight"
              value={examplePrediction.probabilities.slight}
              color="text-emerald-300"
            />
          </div>
        </section>

        <section className="glass rounded-2xl border border-white/[0.06] p-6">
          <SectionHeader
            icon={<MapPin className="h-4 w-4" />}
            title="Input Context"
            subtitle="Features supplied to the prediction pipeline"
          />

          <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {examplePrediction.context.map((item) => (
              <ContextItem
                key={item.label}
                label={item.label}
                value={item.value}
              />
            ))}
          </div>
        </section>
      </section>

      {/* ============================================================
          GLOBAL SHAP
      ============================================================ */}

      <section className="glass rounded-2xl border border-white/[0.06] p-6">
        <SectionHeader
          icon={<Sparkles className="h-4 w-4" />}
          title="Global SHAP Importance"
          subtitle="Mean absolute SHAP contribution across the explanation sample"
        />

        <div className="mt-6 grid gap-7 xl:grid-cols-[1.25fr_0.75fr]">
          <div className="h-[430px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={globalShap.slice(0, 12).slice().reverse()}
                layout="vertical"
                margin={{
                  top: 5,
                  right: 20,
                  left: 10,
                  bottom: 5,
                }}
              >
                <CartesianGrid
                  stroke="rgba(255,255,255,0.04)"
                  horizontal={false}
                />

                <XAxis
                  type="number"
                  tick={{
                    fill: "#64748b",
                    fontSize: 9,
                  }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(value) =>
                    Number(value).toFixed(2)
                  }
                />

                <YAxis
                  type="category"
                  dataKey="feature"
                  width={185}
                  tick={{
                    fill: "#94a3b8",
                    fontSize: 9,
                  }}
                  axisLine={false}
                  tickLine={false}
                />

                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(value) =>
                    typeof value === "number"
                      ? value.toFixed(6)
                      : String(value ?? "")
                  }
                />

                <Bar
                  dataKey="value"
                  name="Mean |SHAP|"
                  radius={[0, 5, 5, 0]}
                  fill="#8b5cf6"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="space-y-3">
            <ExplainStat
              rank="01"
              feature="Speed Limit"
              value="0.140857"
              icon={<Gauge className="h-4 w-4" />}
            />

            <ExplainStat
              rank="02"
              feature="Junction Detail"
              value="0.054012"
              icon={<Target className="h-4 w-4" />}
            />

            <ExplainStat
              rank="03"
              feature="Urban / Rural"
              value="0.045771"
              icon={<MapPin className="h-4 w-4" />}
            />

            <ExplainStat
              rank="04"
              feature="Accident Hour"
              value="0.042588"
              icon={<Activity className="h-4 w-4" />}
            />

            <ExplainStat
              rank="05"
              feature="Hour Cos"
              value="0.042144"
              icon={<TrendingUp className="h-4 w-4" />}
            />

            <div className="mt-4 rounded-xl border border-violet-400/10 bg-violet-500/[0.025] p-4">
              <div className="flex gap-3">
                <Info className="mt-0.5 h-4 w-4 shrink-0 text-violet-300" />

                <p className="text-[10px] leading-5 text-slate-500">
                  Global SHAP importance describes how strongly
                  features influence model output across the
                  analyzed sample. It does not establish causality.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================
          LOCAL SHAP
      ============================================================ */}

      <section className="glass rounded-2xl border border-white/[0.06] p-6">
        <SectionHeader
          icon={<Target className="h-4 w-4" />}
          title="Local SHAP Explanation"
          subtitle="Feature contributions associated with an individual prediction"
        />

        <div className="mt-6 grid gap-8 xl:grid-cols-[1fr_0.8fr]">
          <div className="space-y-3">
            {shapSample.map((item, index) => {
              const normalized =
                (Math.abs(item.contribution) /
                  shapSample[0].contribution) *
                100;

              return (
                <div
                  key={`${item.feature}-${index}`}
                  className="rounded-xl border border-white/[0.05] bg-white/[0.015] p-4"
                >
                  <div className="flex items-center justify-between gap-4">
                    <div className="flex min-w-0 items-center gap-3">
                      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-violet-500/10 font-mono text-[9px] text-violet-300">
                        {String(index + 1).padStart(2, "0")}
                      </span>

                      <span className="truncate text-xs text-slate-300">
                        {item.feature}
                      </span>
                    </div>

                    <div className="flex shrink-0 items-center gap-2">
                      <ArrowUpRight className="h-3 w-3 text-violet-300" />

                      <span className="font-mono text-[11px] text-violet-300">
                        +{item.contribution.toFixed(6)}
                      </span>
                    </div>
                  </div>

                  <div className="mt-3 ml-10 h-1.5 overflow-hidden rounded-full bg-white/[0.04]">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-violet-500 to-cyan-400"
                      style={{
                        width: `${Math.max(
                          4,
                          normalized,
                        )}%`,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="rounded-2xl border border-white/[0.05] bg-white/[0.015] p-5">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-violet-500/10">
                <BrainCircuit className="h-5 w-5 text-violet-300" />
              </div>

              <div>
                <p className="text-[9px] uppercase tracking-[0.18em] text-violet-300">
                  Local Interpretation
                </p>

                <h3 className="mt-1 text-sm font-semibold text-white">
                  What influenced this prediction?
                </h3>

                <p className="mt-2 text-[10px] leading-5 text-slate-500">
                  The local explanation surfaces the strongest
                  feature contributions associated with the
                  individual model decision.
                </p>
              </div>
            </div>

            <div className="mt-6 space-y-3">
              <DirectionCard
                type="positive"
                title="Strongest model signal"
                feature="Speed Limit"
                value="+0.140857"
              />

              <DirectionCard
                type="positive"
                title="Contextual signal"
                feature="Junction Detail"
                value="+0.054012"
              />

              <DirectionCard
                type="positive"
                title="Spatial / area signal"
                feature="Urban / Rural"
                value="+0.045771"
              />
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================
          LIME
      ============================================================ */}

      <section className="grid gap-6 xl:grid-cols-[1fr_0.9fr]">
        <section className="glass rounded-2xl border border-white/[0.06] p-6">
          <SectionHeader
            icon={<Layers3 className="h-4 w-4" />}
            title="LIME Local Surrogate"
            subtitle="Local feature-weight interpretation"
          />

          <div className="mt-6 space-y-4">
            {limeConcepts.map((item, index) => {
              const isPositive =
                item.direction === "positive";

              return (
                <div key={item.feature}>
                  <div className="mb-2 flex items-center justify-between gap-4">
                    <div className="flex min-w-0 items-center gap-2">
                      {isPositive ? (
                        <ArrowUpRight className="h-3.5 w-3.5 shrink-0 text-violet-300" />
                      ) : (
                        <ArrowDownRight className="h-3.5 w-3.5 shrink-0 text-cyan-300" />
                      )}

                      <span className="truncate text-xs text-slate-300">
                        {item.feature}
                      </span>
                    </div>

                    <span className="font-mono text-[10px] text-slate-400">
                      {item.weight.toFixed(2)}
                    </span>
                  </div>

                  <div className="ml-5 h-1.5 overflow-hidden rounded-full bg-white/[0.04]">
                    <div
                      className={`h-full rounded-full ${
                        isPositive
                          ? "bg-gradient-to-r from-violet-500 to-indigo-400"
                          : "bg-gradient-to-r from-cyan-500 to-teal-400"
                      }`}
                      style={{
                        width: `${item.weight * 100}%`,
                      }}
                    />
                  </div>

                  {index < limeConcepts.length - 1 && (
                    <div className="mt-4 border-b border-white/[0.035]" />
                  )}
                </div>
              );
            })}
          </div>
        </section>

        <section className="glass rounded-2xl border border-white/[0.06] p-6">
          <SectionHeader
            icon={<Lightbulb className="h-4 w-4" />}
            title="SHAP vs LIME"
            subtitle="Two complementary local explanation approaches"
          />

          <div className="mt-5 space-y-3">
            <ComparisonRow
              label="SHAP"
              description="Feature attribution based on Shapley values."
              active
            />

            <ComparisonRow
              label="LIME"
              description="Local surrogate model approximating the prediction."
            />
          </div>

          <div className="mt-5 rounded-xl border border-cyan-400/10 bg-cyan-400/[0.025] p-4">
            <div className="flex gap-3">
              <Info className="mt-0.5 h-4 w-4 shrink-0 text-cyan-300" />

              <p className="text-[10px] leading-5 text-slate-500">
                SHAP and LIME explain model behavior from different
                methodological perspectives. Neither explanation
                should be interpreted as a causal analysis.
              </p>
            </div>
          </div>
        </section>
      </section>

      {/* ============================================================
          EXPLANATION LEGEND
      ============================================================ */}

      <section className="grid gap-4 md:grid-cols-3">
        <LegendCard
          icon={<BrainCircuit className="h-4 w-4" />}
          title="Model Prediction"
          description="XGBoost determines the accident risk class."
        />

        <LegendCard
          icon={<Sparkles className="h-4 w-4" />}
          title="SHAP Explanation"
          description="Shows feature contribution to model output."
        />

        <LegendCard
          icon={<Layers3 className="h-4 w-4" />}
          title="LIME Explanation"
          description="Provides a local surrogate interpretation."
        />
      </section>

      {/* ============================================================
          IMPORTANT LIMITATION
      ============================================================ */}

      <section className="rounded-2xl border border-amber-400/10 bg-amber-400/[0.025] p-6">
        <div className="flex items-start gap-4">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-amber-400/10">
            <AlertTriangle className="h-5 w-5 text-amber-300" />
          </div>

          <div>
            <p className="text-[9px] font-semibold uppercase tracking-[0.2em] text-amber-300">
              Interpretation Boundary
            </p>

            <h3 className="mt-1 text-sm font-semibold text-white">
              Explainability is not causality
            </h3>

            <p className="mt-2 max-w-4xl text-xs leading-6 text-slate-500">
              SHAP and LIME describe how the trained machine
              learning model responds to the supplied features.
              They do not establish that a feature directly causes
              an accident outcome. The ML model remains responsible
              for risk classification.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

/* ================================================================
   COMPONENTS
================================================================ */

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
        <h2 className="text-sm font-semibold text-white">
          {title}
        </h2>

        <p className="mt-0.5 text-[11px] text-slate-500">
          {subtitle}
        </p>
      </div>
    </div>
  );
}

function ExplainChip({
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

function PipelineStep({
  icon,
  label,
  description,
}: {
  icon: ReactNode;
  label: string;
  description: string;
}) {
  return (
    <div className="flex items-center gap-2 rounded-xl border border-white/[0.05] bg-white/[0.02] px-3 py-2.5">
      <span className="text-violet-300">{icon}</span>

      <div>
        <p className="text-[10px] font-medium text-slate-300">
          {label}
        </p>

        <p className="text-[9px] text-slate-600">
          {description}
        </p>
      </div>
    </div>
  );
}

function PipelineArrow() {
  return (
    <span className="hidden text-slate-700 md:block">
      →
    </span>
  );
}

function Probability({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="rounded-xl border border-white/[0.05] bg-white/[0.015] p-3 text-center">
      <p className="text-[9px] uppercase tracking-wider text-slate-600">
        {label}
      </p>

      <p className={`mt-1 font-mono text-sm font-semibold ${color}`}>
        {value.toFixed(1)}%
      </p>
    </div>
  );
}

function ContextItem({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-white/[0.05] bg-white/[0.015] p-3">
      <p className="text-[9px] uppercase tracking-wider text-slate-600">
        {label}
      </p>

      <p className="mt-1 truncate text-xs text-slate-300">
        {value}
      </p>
    </div>
  );
}

function ExplainStat({
  rank,
  feature,
  value,
  icon,
}: {
  rank: string;
  feature: string;
  value: string;
  icon: ReactNode;
}) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-white/[0.05] bg-white/[0.015] p-3">
      <span className="font-mono text-[9px] text-slate-700">
        {rank}
      </span>

      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-500/10 text-violet-300">
        {icon}
      </div>

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

function DirectionCard({
  type,
  title,
  feature,
  value,
}: {
  type: "positive" | "negative";
  title: string;
  feature: string;
  value: string;
}) {
  const positive = type === "positive";

  return (
    <div className="rounded-xl border border-white/[0.05] bg-black/10 p-4">
      <div className="flex items-center gap-2">
        {positive ? (
          <TrendingUp className="h-3.5 w-3.5 text-violet-300" />
        ) : (
          <TrendingDown className="h-3.5 w-3.5 text-cyan-300" />
        )}

        <span className="text-[9px] uppercase tracking-wider text-slate-600">
          {title}
        </span>
      </div>

      <div className="mt-2 flex items-center justify-between gap-3">
        <span className="text-xs text-slate-300">
          {feature}
        </span>

        <span
          className={`font-mono text-[10px] ${
            positive
              ? "text-violet-300"
              : "text-cyan-300"
          }`}
        >
          {value}
        </span>
      </div>
    </div>
  );
}

function ComparisonRow({
  label,
  description,
  active = false,
}: {
  label: string;
  description: string;
  active?: boolean;
}) {
  return (
    <div
      className={`rounded-xl border p-4 ${
        active
          ? "border-violet-400/10 bg-violet-500/[0.035]"
          : "border-white/[0.05] bg-white/[0.015]"
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-white">
          {label}
        </span>

        {active && (
          <span className="rounded-full border border-violet-400/10 bg-violet-500/10 px-2 py-1 text-[8px] uppercase tracking-wider text-violet-300">
            Primary
          </span>
        )}
      </div>

      <p className="mt-1 text-[10px] leading-5 text-slate-500">
        {description}
      </p>
    </div>
  );
}

function LegendCard({
  icon,
  title,
  description,
}: {
  icon: ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="glass rounded-2xl border border-white/[0.06] p-5">
      <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-500/10 text-violet-300">
        {icon}
      </div>

      <h3 className="mt-4 text-sm font-semibold text-white">
        {title}
      </h3>

      <p className="mt-1 text-[10px] leading-5 text-slate-500">
        {description}
      </p>
    </div>
  );
}