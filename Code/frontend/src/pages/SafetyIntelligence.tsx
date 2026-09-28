import {
  AlertTriangle,
  ArrowRight,
  BrainCircuit,
  Car,
  CheckCircle2,
  CloudRain,
  Eye,
  Gauge,
  Info,
  Lightbulb,
  MapPin,
  Shield,
  Sparkles,
  Sun,
  TrafficCone,
  Waves,
  Wind,
} from "lucide-react";

type GuidanceCard = {
  title: string;
  description: string;
  icon: typeof Gauge;
  accent: string;
  points: string[];
};

const guidanceCards: GuidanceCard[] = [
  {
    title: "Speed & Speed Limits",
    description:
      "Speed is an important contextual feature used by the accident-risk model.",
    icon: Gauge,
    accent: "violet",
    points: [
      "Respect posted speed limits and adapt speed to road conditions.",
      "Reduce speed when visibility, weather, or surface conditions deteriorate.",
      "Avoid interpreting model feature importance as a causal relationship.",
    ],
  },
  {
    title: "Junction Safety",
    description:
      "Junction configuration and control are represented in the prediction pipeline.",
    icon: TrafficCone,
    accent: "cyan",
    points: [
      "Approach junctions with increased attention and appropriate speed.",
      "Follow traffic-control signs and signals.",
      "Take additional care at complex junction layouts and roundabouts.",
    ],
  },
  {
    title: "Weather Conditions",
    description:
      "Weather context is incorporated into the accident feature set and retrieval layer.",
    icon: CloudRain,
    accent: "blue",
    points: [
      "Adjust driving behavior for rain, wind, fog, or reduced visibility.",
      "Increase following distance when conditions reduce stopping ability.",
      "Use appropriate lighting and maintain visibility in adverse weather.",
    ],
  },
  {
    title: "Road Surface",
    description:
      "Road-surface conditions form part of the structured accident context.",
    icon: Waves,
    accent: "emerald",
    points: [
      "Allow additional stopping distance on wet, icy, or slippery surfaces.",
      "Avoid abrupt braking or steering when traction is reduced.",
      "Drive according to the conditions rather than normal dry-road behavior.",
    ],
  },
  {
    title: "Lighting & Visibility",
    description:
      "Lighting conditions are included in the model's contextual features.",
    icon: Eye,
    accent: "amber",
    points: [
      "Use appropriate vehicle lighting in low-visibility conditions.",
      "Remain alert to pedestrians, cyclists, and roadside hazards.",
      "Reduce speed when visibility is insufficient for safe stopping.",
    ],
  },
  {
    title: "Urban & Rural Roads",
    description:
      "The model explicitly represents urban/rural road context.",
    icon: MapPin,
    accent: "rose",
    points: [
      "Expect different traffic patterns and hazards across road environments.",
      "Maintain awareness of pedestrians and intersections in urban areas.",
      "Remain alert to speed, visibility, and road-layout changes on rural roads.",
    ],
  },
  {
    title: "Driver Awareness",
    description:
      "Human attention remains an important part of practical road safety.",
    icon: BrainCircuit,
    accent: "fuchsia",
    points: [
      "Stay focused on the road and surrounding traffic.",
      "Anticipate the behavior of nearby road users.",
      "Avoid distractions that reduce situational awareness.",
    ],
  },
  {
    title: "Vehicle Safety",
    description:
      "Vehicle context is represented within the accident feature pipeline.",
    icon: Car,
    accent: "indigo",
    points: [
      "Maintain the vehicle in safe operating condition.",
      "Use appropriate safety equipment and restraints.",
      "Check visibility, lighting, and tires before travel.",
    ],
  },
];

const pipelineSteps = [
  {
    number: "01",
    title: "Risk Prediction",
    description: "XGBoost classifies the accident context.",
    icon: BrainCircuit,
  },
  {
    number: "02",
    title: "Model Explanation",
    description: "SHAP and LIME provide interpretable model insights.",
    icon: Sparkles,
  },
  {
    number: "03",
    title: "Knowledge Retrieval",
    description: "RAG retrieves relevant road-safety guidance.",
    icon: Shield,
  },
  {
    number: "04",
    title: "AI Guidance",
    description: "The LLM converts grounded context into readable guidance.",
    icon: Lightbulb,
  },
];

const accentMap: Record<
  string,
  {
    icon: string;
    glow: string;
    border: string;
    badge: string;
  }
> = {
  violet: {
    icon: "text-violet-300",
    glow: "bg-violet-500/10",
    border: "border-violet-400/15",
    badge: "bg-violet-400/10 text-violet-300",
  },
  cyan: {
    icon: "text-cyan-300",
    glow: "bg-cyan-500/10",
    border: "border-cyan-400/15",
    badge: "bg-cyan-400/10 text-cyan-300",
  },
  blue: {
    icon: "text-blue-300",
    glow: "bg-blue-500/10",
    border: "border-blue-400/15",
    badge: "bg-blue-400/10 text-blue-300",
  },
  emerald: {
    icon: "text-emerald-300",
    glow: "bg-emerald-500/10",
    border: "border-emerald-400/15",
    badge: "bg-emerald-400/10 text-emerald-300",
  },
  amber: {
    icon: "text-amber-300",
    glow: "bg-amber-500/10",
    border: "border-amber-400/15",
    badge: "bg-amber-400/10 text-amber-300",
  },
  rose: {
    icon: "text-rose-300",
    glow: "bg-rose-500/10",
    border: "border-rose-400/15",
    badge: "bg-rose-400/10 text-rose-300",
  },
  fuchsia: {
    icon: "text-fuchsia-300",
    glow: "bg-fuchsia-500/10",
    border: "border-fuchsia-400/15",
    badge: "bg-fuchsia-400/10 text-fuchsia-300",
  },
  indigo: {
    icon: "text-indigo-300",
    glow: "bg-indigo-500/10",
    border: "border-indigo-400/15",
    badge: "bg-indigo-400/10 text-indigo-300",
  },
};

export default function SafetyIntelligence() {
  return (
    <div className="min-h-full space-y-6 pb-10">
      {/* Hero */}
      <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-emerald-500/[0.10] via-white/[0.035] to-cyan-500/[0.08] p-7 shadow-2xl shadow-emerald-950/20">
        <div className="pointer-events-none absolute -right-28 -top-28 h-80 w-80 rounded-full bg-emerald-500/10 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-32 left-1/3 h-80 w-80 rounded-full bg-cyan-500/10 blur-3xl" />

        <div className="relative flex flex-col justify-between gap-7 lg:flex-row lg:items-end">
          <div className="max-w-3xl">
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5 text-xs font-medium text-emerald-200">
              <ShieldCheckIcon />
              RAG-powered safety intelligence
            </div>

            <h1 className="text-3xl font-semibold tracking-tight text-white md:text-4xl">
              Safety{" "}
              <span className="bg-gradient-to-r from-emerald-300 via-cyan-300 to-violet-300 bg-clip-text text-transparent">
                Intelligence
              </span>
            </h1>

            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400">
              Context-aware road-safety guidance grounded in the project's
              knowledge base and connected to the explainable accident-risk
              pipeline.
            </p>
          </div>

          <div className="flex shrink-0 items-center gap-3 rounded-2xl border border-emerald-400/15 bg-emerald-400/[0.05] px-4 py-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-400/10 text-emerald-300">
              <Shield size={18} />
            </div>

            <div>
              <p className="text-xs font-medium text-emerald-300">
                Knowledge layer active
              </p>
              <p className="mt-0.5 text-[11px] text-slate-500">
                RAG · 384-dim embeddings · grounded LLM
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Safety status */}
      <section className="grid gap-4 md:grid-cols-3">
        <StatusCard
          icon={Shield}
          title="Knowledge Base"
          value="Active"
          description="Road-safety guidance available for retrieval."
          tone="emerald"
        />

        <StatusCard
          icon={Sparkles}
          title="Retrieval"
          value="12 chunks"
          description="Indexed knowledge chunks in the local RAG store."
          tone="cyan"
        />

        <StatusCard
          icon={Lightbulb}
          title="AI Guidance"
          value="Grounded"
          description="LLM explanations use retrieved safety context."
          tone="violet"
        />
      </section>

      {/* Important distinction */}
      <section className="relative overflow-hidden rounded-2xl border border-amber-400/15 bg-amber-400/[0.035] p-5">
        <div className="pointer-events-none absolute -right-20 -top-20 h-44 w-44 rounded-full bg-amber-400/5 blur-3xl" />

        <div className="relative flex gap-4">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-amber-400/15 bg-amber-400/10 text-amber-300">
            <AlertTriangle size={18} />
          </div>

          <div>
            <h2 className="text-sm font-semibold text-white">
              Prediction and guidance are separate layers
            </h2>

            <p className="mt-1.5 max-w-4xl text-xs leading-5 text-slate-400">
              The machine-learning model remains responsible for the accident
              risk classification. SHAP and LIME explain model behavior, while
              RAG retrieves safety knowledge and the LLM converts that context
              into user-facing guidance. Safety retrieval does not determine
              the predicted risk class.
            </p>
          </div>
        </div>
      </section>

      {/* Guidance library */}
      <section>
        <div className="mb-5 flex items-end justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Shield size={17} className="text-emerald-300" />
              <h2 className="text-sm font-semibold text-white">
                Safety Guidance Library
              </h2>
            </div>

            <p className="mt-1 text-xs text-slate-500">
              Core domains represented in the project's road-safety knowledge
              base.
            </p>
          </div>

          <div className="hidden rounded-full border border-white/8 bg-white/[0.025] px-3 py-1.5 text-[10px] text-slate-500 sm:block">
            8 guidance domains
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {guidanceCards.map((card) => {
            const Icon = card.icon;
            const styles = accentMap[card.accent];

            return (
              <article
                key={card.title}
                className={`group relative overflow-hidden rounded-2xl border ${styles.border} bg-white/[0.035] p-5 backdrop-blur-xl transition-all duration-300 hover:-translate-y-1 hover:bg-white/[0.05]`}
              >
                <div
                  className={`pointer-events-none absolute -right-10 -top-10 h-28 w-28 rounded-full ${styles.glow} blur-3xl transition-all duration-500 group-hover:scale-150`}
                />

                <div className="relative">
                  <div className="flex items-start justify-between">
                    <div
                      className={`rounded-xl border border-white/8 bg-white/[0.04] p-2.5 ${styles.icon}`}
                    >
                      <Icon size={18} />
                    </div>

                    <span
                      className={`rounded-full px-2 py-1 text-[9px] font-medium uppercase tracking-wider ${styles.badge}`}
                    >
                      Guidance
                    </span>
                  </div>

                  <h3 className="mt-4 text-sm font-semibold text-white">
                    {card.title}
                  </h3>

                  <p className="mt-1.5 min-h-[42px] text-xs leading-5 text-slate-500">
                    {card.description}
                  </p>

                  <div className="mt-4 space-y-2.5 border-t border-white/7 pt-4">
                    {card.points.map((point) => (
                      <div key={point} className="flex gap-2.5">
                        <CheckCircle2
                          size={13}
                          className={`mt-0.5 shrink-0 ${styles.icon}`}
                        />
                        <p className="text-[11px] leading-5 text-slate-400">
                          {point}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      </section>

      {/* Context signals */}
      <section className="grid gap-6 lg:grid-cols-[1fr_1.35fr]">
        <div className="rounded-2xl border border-white/8 bg-white/[0.035] p-5 backdrop-blur-xl">
          <div className="mb-5">
            <div className="flex items-center gap-2">
              <Wind size={17} className="text-cyan-300" />
              <h2 className="text-sm font-semibold text-white">
                Safety Context Signals
              </h2>
            </div>

            <p className="mt-1 text-xs text-slate-500">
              Context represented across the prediction and knowledge layers.
            </p>
          </div>

          <div className="space-y-3">
            <ContextSignal
              icon={Gauge}
              label="Speed"
              value="Speed limit"
              tone="violet"
            />

            <ContextSignal
              icon={TrafficCone}
              label="Junction"
              value="Control + layout"
              tone="cyan"
            />

            <ContextSignal
              icon={CloudRain}
              label="Weather"
              value="Weather conditions"
              tone="blue"
            />

            <ContextSignal
              icon={Waves}
              label="Surface"
              value="Road surface"
              tone="emerald"
            />

            <ContextSignal
              icon={Sun}
              label="Lighting"
              value="Light conditions"
              tone="amber"
            />

            <ContextSignal
              icon={MapPin}
              label="Environment"
              value="Urban / rural"
              tone="rose"
            />
          </div>
        </div>

        {/* Retrieval explanation */}
        <div className="rounded-2xl border border-white/8 bg-white/[0.035] p-5 backdrop-blur-xl">
          <div className="mb-5">
            <div className="flex items-center gap-2">
              <Sparkles size={17} className="text-violet-300" />
              <h2 className="text-sm font-semibold text-white">
                How Safety Intelligence Works
              </h2>
            </div>

            <p className="mt-1 text-xs text-slate-500">
              The complete explainable AI path from prediction to guidance.
            </p>
          </div>

          <div className="space-y-3">
            {pipelineSteps.map((step, index) => {
              const Icon = step.icon;

              return (
                <div key={step.number} className="relative">
                  {index < pipelineSteps.length - 1 && (
                    <div className="absolute left-[19px] top-[44px] h-5 w-px bg-gradient-to-b from-violet-400/30 to-transparent" />
                  )}

                  <div className="flex gap-3 rounded-2xl border border-white/7 bg-black/10 p-3.5 transition-colors hover:bg-white/[0.025]">
                    <div className="relative flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-violet-400/15 bg-violet-400/[0.07] text-violet-300">
                      <Icon size={16} />
                    </div>

                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-[9px] text-violet-400/70">
                          {step.number}
                        </span>

                        <h3 className="text-xs font-semibold text-white">
                          {step.title}
                        </h3>
                      </div>

                      <p className="mt-1 text-[11px] leading-5 text-slate-500">
                        {step.description}
                      </p>
                    </div>

                    {index < pipelineSteps.length - 1 && (
                      <ArrowRight
                        size={14}
                        className="ml-auto mt-1 shrink-0 text-slate-700"
                      />
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Practical recommendations */}
      <section className="rounded-2xl border border-violet-400/15 bg-gradient-to-br from-violet-500/[0.07] via-white/[0.025] to-cyan-500/[0.05] p-6">
        <div className="flex flex-col gap-5 md:flex-row md:items-center md:justify-between">
          <div className="flex gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-violet-400/15 bg-violet-400/10 text-violet-300">
              <Lightbulb size={19} />
            </div>

            <div>
              <h2 className="text-sm font-semibold text-white">
                Context-aware recommendations
              </h2>

              <p className="mt-1 max-w-2xl text-xs leading-5 text-slate-500">
                When a prediction is generated, the system can combine the
                model's explanation with relevant retrieved safety guidance to
                provide a more understandable risk narrative.
              </p>
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-2 rounded-xl border border-emerald-400/15 bg-emerald-400/[0.05] px-3 py-2">
            <CheckCircle2 size={14} className="text-emerald-300" />
            <span className="text-[10px] font-medium text-emerald-300">
              Grounded guidance
            </span>
          </div>
        </div>
      </section>

      {/* Footer note */}
      <div className="flex items-start gap-2 px-1">
        <Info size={13} className="mt-0.5 shrink-0 text-slate-600" />

        <p className="max-w-4xl text-[10px] leading-5 text-slate-600">
          Safety guidance is informational and should not be interpreted as a
          replacement for applicable traffic laws, official road signs, or
          professional safety guidance. Model explanations describe learned
          model behavior and should not be interpreted as causal evidence.
        </p>
      </div>
    </div>
  );
}

function ShieldCheckIcon() {
  return <Shield size={13} />;
}

function StatusCard({
  icon: Icon,
  title,
  value,
  description,
  tone,
}: {
  icon: typeof Shield;
  title: string;
  value: string;
  description: string;
  tone: "emerald" | "cyan" | "violet";
}) {
  const styles = {
    emerald: {
      icon: "text-emerald-300",
      bg: "bg-emerald-400/10",
      border: "border-emerald-400/15",
    },
    cyan: {
      icon: "text-cyan-300",
      bg: "bg-cyan-400/10",
      border: "border-cyan-400/15",
    },
    violet: {
      icon: "text-violet-300",
      bg: "bg-violet-400/10",
      border: "border-violet-400/15",
    },
  }[tone];

  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.035] p-5 backdrop-blur-xl">
      <div className="flex items-start justify-between">
        <div
          className={`rounded-xl border ${styles.border} ${styles.bg} p-2.5 ${styles.icon}`}
        >
          <Icon size={18} />
        </div>

        <span className="flex items-center gap-1.5 rounded-full border border-emerald-400/10 bg-emerald-400/[0.04] px-2 py-1 text-[9px] font-medium uppercase tracking-wider text-emerald-300">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
          Ready
        </span>
      </div>

      <p className="mt-4 text-xs text-slate-500">{title}</p>

      <p className="mt-1 text-lg font-semibold text-white">{value}</p>

      <p className="mt-1 text-[11px] leading-5 text-slate-600">
        {description}
      </p>
    </div>
  );
}

function ContextSignal({
  icon: Icon,
  label,
  value,
  tone,
}: {
  icon: typeof Gauge;
  label: string;
  value: string;
  tone: keyof typeof accentMap;
}) {
  const styles = accentMap[tone];

  return (
    <div className="flex items-center gap-3 rounded-xl border border-white/7 bg-black/10 px-3.5 py-3">
      <div
        className={`rounded-lg border border-white/7 bg-white/[0.03] p-2 ${styles.icon}`}
      >
        <Icon size={15} />
      </div>

      <div className="min-w-0 flex-1">
        <p className="text-[10px] uppercase tracking-wider text-slate-600">
          {label}
        </p>

        <p className="mt-0.5 text-xs font-medium text-slate-300">{value}</p>
      </div>

      <span className={`h-1.5 w-1.5 rounded-full ${styles.glow}`} />
    </div>
  );
}