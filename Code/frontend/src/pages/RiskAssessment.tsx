import { useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  CloudRain,
  Gauge,
  Loader2,
  MapPin,
  ShieldAlert,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
} from "lucide-react";

interface PredictionResponse {
  predicted_risk: "Fatal" | "Serious" | "Slight";

  probabilities: {
    fatal: number;
    serious: number;
    slight: number;
  };

  shap_contributions: Array<{
    feature: string;
    contribution: number;
  }>;

  retrieved_knowledge: Array<{
    content?: string;
    text?: string;
    score?: number;
    source?: string;
  }>;

  explanation: string;
}

interface AccidentForm {
  accident_date: string;
  time: string;
  junction_control: string;
  junction_detail: string;
  light_conditions: string;
  local_authority_district: string;
  carriageway_hazards: string;
  police_force: string;
  road_surface_conditions: string;
  road_type: string;
  speed_limit: number;
  urban_or_rural_area: string;
  weather_conditions: string;
  vehicle_type: string;
  latitude: number;
  longitude: number;
}

interface ExplanationSection {
  title: string;
  content: string[];
}

const initialForm: AccidentForm = {
  accident_date: "2026-09-25",
  time: "18:30",
  junction_control: "Give way or uncontrolled",
  junction_detail: "Not at junction",
  light_conditions: "Daylight",
  local_authority_district: "Unknown",
  carriageway_hazards: "None",
  police_force: "Unknown",
  road_surface_conditions: "Dry",
  road_type: "Single carriageway",
  speed_limit: 50,
  urban_or_rural_area: "Urban",
  weather_conditions: "Fine no high winds",
  vehicle_type: "Car",
  latitude: 17.385,
  longitude: 78.4867,
};

/* ================================================================
   Markdown / AI Output Helpers
================================================================ */

function cleanInlineMarkdown(value: string) {
  return value
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/__(.*?)__/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/_(.*?)_/g, "$1")
    .replace(/`(.*?)`/g, "$1")
    .replace(/^>\s*/, "")
    .trim();
}

function parseExplanation(explanation: string): ExplanationSection[] {
  if (!explanation?.trim()) {
    return [];
  }

  const lines = explanation
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const sections: ExplanationSection[] = [];
  let current: ExplanationSection | null = null;

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (/^#{1,6}\s+/.test(line)) {
      const title = cleanInlineMarkdown(
        line.replace(/^#{1,6}\s+/, ""),
      );

      if (current) {
        sections.push(current);
      }

      current = {
        title,
        content: [],
      };

      continue;
    }

    if (/^---+$/.test(line)) {
      continue;
    }

    if (!current) {
      current = {
        title: "Risk Assessment",
        content: [],
      };
    }

    const cleaned = cleanInlineMarkdown(
      line.replace(/^[-•]\s*/, ""),
    );

    if (cleaned) {
      current.content.push(cleaned);
    }
  }

  if (current) {
    sections.push(current);
  }

  return sections;
}

function cleanRagContent(content: string) {
  const normalized = content
    .replace(/\r\n/g, "\n")
    .replace(/---+/g, "")
    .replace(/#{1,6}\s*/g, "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\[(.*?)\]\(.*?\)/g, "$1")
    .trim();

  return normalized;
}

function extractRagTitle(content: string, index: number) {
  const match = content.match(
    /#{1,6}\s*(?:\d+\.\s*)?([^#\n]+?)(?=\s+(?:Safety guidance|Safety Guidance|Vehicle condition|Vehicle Condition|High-speed context|Adverse-weather context|Poor-visibility context|Junction context)|$)/i,
  );

  if (match?.[1]) {
    return cleanInlineMarkdown(match[1].trim());
  }

  const numbered = content.match(
    /#{1,6}\s*\d+\.\s*([^\n]+)/,
  );

  if (numbered?.[1]) {
    return cleanInlineMarkdown(numbered[1]);
  }

  return `Safety Guidance ${index + 1}`;
}

function extractRagBody(content: string) {
  let body = content;

  body = body.replace(
    /#{1,6}\s*(?:\d+\.\s*)?[^#\n]+?(?=\s+(?:Safety guidance|Safety Guidance)|\n|$)/i,
    "",
  );

  body = body
    .replace(/#{1,6}\s*/g, "")
    .replace(/---+/g, "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\[(.*?)\]\(.*?\)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();

  return body || cleanRagContent(content);
}

function formatFeatureName(feature: string) {
  return feature
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .trim();
}

/* ================================================================
   Main Page
================================================================ */

export default function RiskAssessment() {
  const [form, setForm] = useState<AccidentForm>(initialForm);
  const [result, setResult] =
    useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const updateField = (
    field: keyof AccidentForm,
    value: string | number,
  ) => {
    setForm((previous) => ({
      ...previous,
      [field]: value,
    }));
  };

  const predictRisk = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        "http://localhost:8000/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(form),
        },
      );

      if (!response.ok) {
        const message = await response.text();

        throw new Error(
          message || "Prediction request failed.",
        );
      }

      const data: PredictionResponse =
        await response.json();

      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the prediction API.",
      );
    } finally {
      setLoading(false);
    }
  };

  const riskClass =
    result?.predicted_risk === "Fatal"
      ? "risk-fatal"
      : result?.predicted_risk === "Serious"
        ? "risk-serious"
        : "risk-slight";

  const explanationSections = result
    ? parseExplanation(result.explanation)
    : [];

  const positiveDrivers =
    result?.shap_contributions
      .filter((item) => item.contribution > 0)
      .slice(0, 4) ?? [];

  const negativeDrivers =
    result?.shap_contributions
      .filter((item) => item.contribution < 0)
      .slice(0, 4) ?? [];

  return (
    <div className="space-y-8">
      {/* Header */}
      <section>
        <div className="mb-3 flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-violet-400/20 bg-violet-500/10">
            <Target className="h-4 w-4 text-violet-300" />
          </div>

          <span className="text-[11px] font-semibold uppercase tracking-[0.24em] text-violet-300">
            Risk Engine
          </span>
        </div>

        <h1 className="text-3xl font-semibold tracking-tight text-white md:text-4xl">
          Accident Risk Assessment
        </h1>

        <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
          Submit real accident context to the trained XGBoost
          model. The prediction is then explained with SHAP,
          grounded with road-safety knowledge, and converted
          into actionable guidance through the LLM.
        </p>
      </section>

      {/* Pipeline */}
      <section className="glass rounded-2xl border border-white/[0.06] p-4">
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <PipelineItem
            icon={<Activity size={14} />}
            label="XGBoost"
            description="Predict"
          />

          <ArrowRight
            className="hidden text-slate-700 sm:block"
            size={15}
          />

          <PipelineItem
            icon={<BrainCircuit size={14} />}
            label="SHAP"
            description="Explain"
          />

          <ArrowRight
            className="hidden text-slate-700 sm:block"
            size={15}
          />

          <PipelineItem
            icon={<Sparkles size={14} />}
            label="RAG"
            description="Ground"
          />

          <ArrowRight
            className="hidden text-slate-700 sm:block"
            size={15}
          />

          <PipelineItem
            icon={<ShieldAlert size={14} />}
            label="LLM"
            description="Assist"
          />
        </div>
      </section>

      {/* Main grid */}
      <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        {/* FORM */}
        <section className="glass rounded-2xl border border-white/[0.06] p-6">
          <div className="mb-6 flex items-start justify-between">
            <div>
              <h2 className="text-lg font-semibold text-white">
                Accident Context
              </h2>

              <p className="mt-1 text-xs text-slate-500">
                Provide the conditions surrounding the accident.
              </p>
            </div>

            <div className="rounded-lg border border-cyan-400/10 bg-cyan-400/5 px-3 py-1.5 text-[10px] uppercase tracking-wider text-cyan-300">
              Live Model
            </div>
          </div>

          <div className="space-y-6">
            {/* Time & location */}
            <FormSection
              icon={<Clock3 size={15} />}
              title="Time & Location"
            >
              <div className="grid gap-4 sm:grid-cols-2">
                <Field label="Accident Date">
                  <input
                    type="date"
                    value={form.accident_date}
                    onChange={(e) =>
                      updateField(
                        "accident_date",
                        e.target.value,
                      )
                    }
                    className="input-field"
                  />
                </Field>

                <Field label="Time">
                  <input
                    type="time"
                    value={form.time}
                    onChange={(e) =>
                      updateField(
                        "time",
                        e.target.value,
                      )
                    }
                    className="input-field"
                  />
                </Field>

                <Field label="Latitude">
                  <input
                    type="number"
                    step="0.0001"
                    value={form.latitude}
                    onChange={(e) =>
                      updateField(
                        "latitude",
                        Number(e.target.value),
                      )
                    }
                    className="input-field"
                  />
                </Field>

                <Field label="Longitude">
                  <input
                    type="number"
                    step="0.0001"
                    value={form.longitude}
                    onChange={(e) =>
                      updateField(
                        "longitude",
                        Number(e.target.value),
                      )
                    }
                    className="input-field"
                  />
                </Field>

                <Field label="Local Authority District">
                  <input
                    value={
                      form.local_authority_district
                    }
                    onChange={(e) =>
                      updateField(
                        "local_authority_district",
                        e.target.value,
                      )
                    }
                    className="input-field"
                    placeholder="District"
                  />
                </Field>

                <Field label="Police Force">
                  <input
                    value={form.police_force}
                    onChange={(e) =>
                      updateField(
                        "police_force",
                        e.target.value,
                      )
                    }
                    className="input-field"
                    placeholder="Police force"
                  />
                </Field>
              </div>
            </FormSection>

            {/* Road */}
            <FormSection
              icon={<MapPin size={15} />}
              title="Road Context"
            >
              <div className="grid gap-4 sm:grid-cols-2">
                <Field label="Road Type">
                  <select
                    value={form.road_type}
                    onChange={(e) =>
                      updateField(
                        "road_type",
                        e.target.value,
                      )
                    }
                    className="input-field"
                  >
                    <option>
                      Single carriageway
                    </option>
                    <option>
                      Dual carriageway
                    </option>
                    <option>Roundabout</option>
                    <option>One way street</option>
                    <option>Slip road</option>
                  </select>
                </Field>

                <Field label="Speed Limit">
                  <select
                    value={form.speed_limit}
                    onChange={(e) =>
                      updateField(
                        "speed_limit",
                        Number(e.target.value),
                      )
                    }
                    className="input-field"
                  >
                    <option value={20}>20 mph</option>
                    <option value={30}>30 mph</option>
                    <option value={40}>40 mph</option>
                    <option value={50}>50 mph</option>
                    <option value={60}>60 mph</option>
                    <option value={70}>70 mph</option>
                  </select>
                </Field>

                <Field label="Junction Control">
                  <select
                    value={form.junction_control}
                    onChange={(e) =>
                      updateField(
                        "junction_control",
                        e.target.value,
                      )
                    }
                    className="input-field"
                  >
                    <option>
                      Give way or uncontrolled
                    </option>
                    <option>
                      Auto traffic signal
                    </option>
                    <option>Stop sign</option>
                    <option>
                      Authorised person
                    </option>
                    <option>
                      Not at junction or within 20 metres
                    </option>
                  </select>
                </Field>

                <Field label="Junction Detail">
                  <select
                    value={form.junction_detail}
                    onChange={(e) =>
                      updateField(
                        "junction_detail",
                        e.target.value,
                      )
                    }
                    className="input-field"
                  >
                    <option>
                      Not at junction
                    </option>
                    <option>Crossroads</option>
                    <option>T junction</option>
                    <option>Roundabout</option>
                    <option>
                      Private drive or entrance
                    </option>
                    <option>Other junction</option>
                  </select>
                </Field>

                <Field label="Urban / Rural">
                  <select
                    value={
                      form.urban_or_rural_area
                    }
                    onChange={(e) =>
                      updateField(
                        "urban_or_rural_area",
                        e.target.value,
                      )
                    }
                    className="input-field"
                  >
                    <option>Urban</option>
                    <option>Rural</option>
                  </select>
                </Field>

                <Field label="Carriageway Hazards">
                  <select
                    value={
                      form.carriageway_hazards
                    }
                    onChange={(e) =>
                      updateField(
                        "carriageway_hazards",
                        e.target.value,
                      )
                    }
                    className="input-field"
                  >
                    <option>None</option>
                    <option>
                      Other object on road
                    </option>
                    <option>
                      Involvement with previous accident
                    </option>
                    <option>
                      Animal or pedestrian in carriageway
                    </option>
                    <option>Any animal</option>
                  </select>
                </Field>
              </div>
            </FormSection>

            {/* Environment */}
            <FormSection
              icon={<CloudRain size={15} />}
              title="Environmental Conditions"
            >
              <div className="grid gap-4 sm:grid-cols-2">
                <Field label="Weather">
                  <select
                    value={
                      form.weather_conditions
                    }
                    onChange={(e) =>
                      updateField(
                        "weather_conditions",
                        e.target.value,
                      )
                    }
                    className="input-field"
                  >
                    <option>
                      Fine no high winds
                    </option>
                    <option>
                      Raining no high winds
                    </option>
                    <option>
                      Raining + high winds
                    </option>
                    <option>
                      Snowing no high winds
                    </option>
                    <option>
                      Snowing + high winds
                    </option>
                    <option>Fog or mist</option>
                    <option>Other</option>
                  </select>
                </Field>

                <Field label="Road Surface">
                  <select
                    value={
                      form.road_surface_conditions
                    }
                    onChange={(e) =>
                      updateField(
                        "road_surface_conditions",
                        e.target.value,
                      )
                    }
                    className="input-field"
                  >
                    <option>Dry</option>
                    <option>Wet or damp</option>
                    <option>Snow</option>
                    <option>Frost or ice</option>
                    <option>
                      Flood over 3cm deep
                    </option>
                  </select>
                </Field>

                <Field label="Light Conditions">
                  <select
                    value={form.light_conditions}
                    onChange={(e) =>
                      updateField(
                        "light_conditions",
                        e.target.value,
                      )
                    }
                    className="input-field"
                  >
                    <option>Daylight</option>
                    <option>
                      Darkness - lights lit
                    </option>
                    <option>
                      Darkness - lights unlit
                    </option>
                    <option>
                      Darkness - no lighting
                    </option>
                    <option>
                      Darkness - lighting unknown
                    </option>
                  </select>
                </Field>

                <Field label="Vehicle Type">
                  <select
                    value={form.vehicle_type}
                    onChange={(e) =>
                      updateField(
                        "vehicle_type",
                        e.target.value,
                      )
                    }
                    className="input-field"
                  >
                    <option>Car</option>
                    <option>
                      Van / Goods 3.5 tonnes mgw or under
                    </option>
                    <option>Bus or coach</option>
                    <option>Motorcycle</option>
                    <option>Goods vehicle</option>
                    <option>Other vehicle</option>
                  </select>
                </Field>
              </div>
            </FormSection>

            {/* Error */}
            {error && (
              <div className="flex gap-3 rounded-xl border border-red-400/20 bg-red-500/[0.06] p-4">
                <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-red-400" />

                <div>
                  <p className="text-sm font-medium text-red-300">
                    Prediction request failed
                  </p>

                  <p className="mt-1 break-words text-xs leading-5 text-red-300/60">
                    {error}
                  </p>
                </div>
              </div>
            )}

            {/* Predict */}
            <button
              type="button"
              onClick={predictRisk}
              disabled={loading}
              className="group flex w-full items-center justify-center gap-3 rounded-xl bg-gradient-to-r from-violet-500 to-indigo-500 px-5 py-4 text-sm font-semibold text-white shadow-[0_0_35px_rgba(124,58,237,0.2)] transition-all duration-300 hover:scale-[1.01] hover:shadow-[0_0_45px_rgba(124,58,237,0.35)] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Running XGBoost + SHAP + RAG...
                </>
              ) : (
                <>
                  <Gauge className="h-5 w-5" />
                  Assess Accident Risk
                  <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
                </>
              )}
            </button>
          </div>
        </section>

        {/* RESULTS */}
        <section className="space-y-6">
          {!result && !loading && (
            <div className="glass flex min-h-[520px] flex-col items-center justify-center rounded-2xl border border-white/[0.06] p-8 text-center">
              <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl border border-violet-400/20 bg-violet-500/10 shadow-[0_0_40px_rgba(124,58,237,0.12)]">
                <Target className="h-7 w-7 text-violet-300" />
              </div>

              <h3 className="text-lg font-semibold text-white">
                Awaiting Assessment
              </h3>

              <p className="mt-2 max-w-sm text-sm leading-6 text-slate-500">
                Submit the accident context to generate a
                real XGBoost risk prediction and its
                supporting intelligence.
              </p>

              <div className="mt-7 grid grid-cols-2 gap-3 text-left">
                <MiniCapability
                  icon={<Activity />}
                  label="Risk prediction"
                />

                <MiniCapability
                  icon={<BrainCircuit />}
                  label="SHAP insights"
                />

                <MiniCapability
                  icon={<Sparkles />}
                  label="RAG guidance"
                />

                <MiniCapability
                  icon={<ShieldAlert />}
                  label="LLM explanation"
                />
              </div>
            </div>
          )}

          {loading && (
            <div className="glass flex min-h-[520px] flex-col items-center justify-center rounded-2xl border border-white/[0.06] p-8 text-center">
              <div className="relative mb-7">
                <div className="h-20 w-20 animate-spin rounded-full border border-violet-400/20 border-t-violet-400" />

                <div className="absolute inset-0 flex items-center justify-center">
                  <BrainCircuit className="h-7 w-7 text-violet-300" />
                </div>
              </div>

              <h3 className="text-lg font-semibold text-white">
                Intelligence Pipeline Running
              </h3>

              <p className="mt-2 text-sm text-slate-500">
                XGBoost → SHAP → RAG → LLM
              </p>
            </div>
          )}

          {result && (
            <>
              {/* Risk result */}
              <div
                className={`glass rounded-2xl border p-6 ${
                  result.predicted_risk === "Fatal"
                    ? "border-red-400/20"
                    : result.predicted_risk === "Serious"
                      ? "border-amber-400/20"
                      : "border-emerald-400/20"
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">
                      Predicted Risk
                    </p>

                    <div className="mt-2 flex items-center gap-3">
                      <span
                        className={`h-3 w-3 rounded-full ${riskClass}`}
                      />

                      <h2 className="text-3xl font-semibold text-white">
                        {result.predicted_risk}
                      </h2>
                    </div>
                  </div>

                  <CheckCircle2 className="h-6 w-6 text-cyan-400" />
                </div>

                {/* Probability bars */}
                <div className="mt-7 space-y-4">
                  <ProbabilityBar
                    label="Fatal"
                    value={result.probabilities.fatal}
                    type="fatal"
                  />

                  <ProbabilityBar
                    label="Serious"
                    value={result.probabilities.serious}
                    type="serious"
                  />

                  <ProbabilityBar
                    label="Slight"
                    value={result.probabilities.slight}
                    type="slight"
                  />
                </div>
              </div>

              {/* AI Explanation */}
              <div className="glass rounded-2xl border border-white/[0.06] p-6">
                <div className="mb-6 flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-violet-400/10 bg-violet-500/10 shadow-[0_0_25px_rgba(124,58,237,0.1)]">
                    <BrainCircuit className="h-5 w-5 text-violet-300" />
                  </div>

                  <div>
                    <h3 className="text-sm font-semibold text-white">
                      AI Risk Explanation
                    </h3>

                    <p className="mt-0.5 text-[11px] text-slate-500">
                      Grounded model interpretation
                    </p>
                  </div>
                </div>

                {explanationSections.length === 0 ? (
                  <div className="rounded-xl border border-white/[0.05] bg-white/[0.02] p-4 text-sm text-slate-500">
                    No explanation was returned for this
                    prediction.
                  </div>
                ) : (
                  <div className="space-y-5">
                    {explanationSections.map(
                      (section, index) => (
                        <div
                          key={`${section.title}-${index}`}
                          className="rounded-xl border border-white/[0.05] bg-white/[0.02] p-4"
                        >
                          <div className="mb-3 flex items-center gap-2">
                            <div className="h-1.5 w-1.5 rounded-full bg-violet-400 shadow-[0_0_10px_rgba(167,139,250,0.7)]" />

                            <h4 className="text-xs font-semibold uppercase tracking-[0.12em] text-violet-200">
                              {section.title}
                            </h4>
                          </div>

                          <div className="space-y-2">
                            {section.content.map(
                              (line, lineIndex) => (
                                <div
                                  key={lineIndex}
                                  className="flex gap-3 text-sm leading-6 text-slate-400"
                                >
                                  {section.content.length >
                                    1 && (
                                    <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-slate-600" />
                                  )}

                                  <p>{line}</p>
                                </div>
                              ),
                            )}
                          </div>
                        </div>
                      ),
                    )}
                  </div>
                )}
              </div>

              {/* SHAP */}
              <div className="glass rounded-2xl border border-white/[0.06] p-6">
                <div className="mb-5 flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-semibold text-white">
                      Model Drivers
                    </h3>

                    <p className="mt-1 text-[11px] text-slate-500">
                      SHAP feature contributions
                    </p>
                  </div>

                  <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-500/10">
                    <BrainCircuit className="h-4 w-4 text-violet-300" />
                  </div>
                </div>

                {result.shap_contributions.length === 0 ? (
                  <div className="rounded-xl border border-white/[0.05] bg-white/[0.02] p-4 text-xs text-slate-500">
                    SHAP explanation is temporarily
                    unavailable for this prediction.
                  </div>
                ) : (
                  <div className="space-y-5">
                    {/* Positive */}
                    {positiveDrivers.length > 0 && (
                      <div>
                        <div className="mb-3 flex items-center gap-2">
                          <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />

                          <span className="text-[10px] font-semibold uppercase tracking-[0.15em] text-emerald-300">
                            Positive Contribution
                          </span>
                        </div>

                        <div className="space-y-2">
                          {positiveDrivers.map(
                            (item, index) => (
                              <ShapRow
                                key={`positive-${item.feature}-${index}`}
                                item={item}
                                positive
                              />
                            ),
                          )}
                        </div>
                      </div>
                    )}

                    {/* Negative */}
                    {negativeDrivers.length > 0 && (
                      <div>
                        <div className="mb-3 flex items-center gap-2">
                          <TrendingDown className="h-3.5 w-3.5 text-cyan-400" />

                          <span className="text-[10px] font-semibold uppercase tracking-[0.15em] text-cyan-300">
                            Negative Contribution
                          </span>
                        </div>

                        <div className="space-y-2">
                          {negativeDrivers.map(
                            (item, index) => (
                              <ShapRow
                                key={`negative-${item.feature}-${index}`}
                                item={item}
                                positive={false}
                              />
                            ),
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                <p className="mt-5 border-t border-white/[0.05] pt-4 text-[10px] leading-5 text-slate-600">
                  SHAP values describe how the trained model
                  used each feature for this prediction. They
                  indicate model behavior and should not be
                  interpreted as causal effects.
                </p>
              </div>

              {/* RAG */}
              <div className="glass rounded-2xl border border-white/[0.06] p-6">
                <div className="mb-5 flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-400/10 bg-cyan-400/10 shadow-[0_0_25px_rgba(34,211,238,0.08)]">
                    <Sparkles className="h-5 w-5 text-cyan-300" />
                  </div>

                  <div>
                    <h3 className="text-sm font-semibold text-white">
                      Safety Intelligence
                    </h3>

                    <p className="mt-0.5 text-[11px] text-slate-500">
                      Retrieved RAG knowledge
                    </p>
                  </div>
                </div>

                {result.retrieved_knowledge.length ===
                0 ? (
                  <div className="rounded-xl border border-white/[0.05] bg-white/[0.02] p-4 text-xs text-slate-500">
                    No safety guidance was retrieved.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {result.retrieved_knowledge.map(
                      (item, index) => {
                        const rawContent =
                          item.content ||
                          item.text ||
                          "Safety guidance retrieved.";

                        const title =
                          extractRagTitle(
                            rawContent,
                            index,
                          );

                        const body =
                          extractRagBody(rawContent);

                        return (
                          <div
                            key={index}
                            className="group rounded-xl border border-cyan-400/[0.08] bg-cyan-400/[0.025] p-4 transition-all duration-300 hover:border-cyan-400/[0.18] hover:bg-cyan-400/[0.04]"
                          >
                            <div className="flex items-start justify-between gap-4">
                              <div className="flex items-center gap-2">
                                <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-cyan-400/10">
                                  <ShieldAlert className="h-3.5 w-3.5 text-cyan-300" />
                                </div>

                                <h4 className="text-xs font-semibold text-cyan-100">
                                  {title}
                                </h4>
                              </div>

                              {typeof item.score ===
                                "number" && (
                                <span className="shrink-0 rounded-full border border-cyan-400/10 bg-cyan-400/5 px-2 py-1 font-mono text-[9px] text-cyan-300/70">
                                  {(
                                    item.score * 100
                                  ).toFixed(0)}
                                  % match
                                </span>
                              )}
                            </div>

                            <p className="mt-3 text-xs leading-6 text-slate-400">
                              {body}
                            </p>

                            {item.source && (
                              <div className="mt-3 flex items-center gap-2 border-t border-white/[0.04] pt-3">
                                <span className="h-1 w-1 rounded-full bg-cyan-400" />

                                <span className="text-[9px] uppercase tracking-wider text-cyan-400/60">
                                  {item.source}
                                </span>
                              </div>
                            )}
                          </div>
                        );
                      },
                    )}
                  </div>
                )}
              </div>

              {/* Pipeline footer */}
              <div className="rounded-2xl border border-violet-400/[0.08] bg-violet-500/[0.025] p-4">
                <div className="flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-wider">
                  <span className="text-slate-600">
                    Intelligence pipeline
                  </span>

                  <span className="text-violet-300">
                    XGBoost
                  </span>

                  <ArrowRight className="h-3 w-3 text-slate-700" />

                  <span className="text-violet-300">
                    SHAP
                  </span>

                  <ArrowRight className="h-3 w-3 text-slate-700" />

                  <span className="text-cyan-300">
                    RAG
                  </span>

                  <ArrowRight className="h-3 w-3 text-slate-700" />

                  <span className="text-cyan-300">
                    LLM
                  </span>

                  <span className="ml-auto flex items-center gap-1.5 text-emerald-400">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
                    Live
                  </span>
                </div>
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}

/* ================================================================
   Components
================================================================ */

function PipelineItem({
  icon,
  label,
  description,
}: {
  icon: ReactNode;
  label: string;
  description: string;
}) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-white/[0.05] bg-white/[0.02] px-3 py-2">
      <span className="text-violet-300">{icon}</span>

      <span className="font-medium text-slate-300">
        {label}
      </span>

      <span className="text-slate-600">·</span>

      <span className="text-slate-500">
        {description}
      </span>
    </div>
  );
}

function FormSection({
  icon,
  title,
  children,
}: {
  icon: ReactNode;
  title: string;
  children: ReactNode;
}) {
  return (
    <div>
      <div className="mb-4 flex items-center gap-2">
        <span className="text-violet-300">{icon}</span>

        <h3 className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-400">
          {title}
        </h3>
      </div>

      {children}
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-[11px] font-medium text-slate-500">
        {label}
      </span>

      {children}
    </label>
  );
}

function ProbabilityBar({
  label,
  value,
  type,
}: {
  label: string;
  value: number;
  type: "fatal" | "serious" | "slight";
}) {
  const percentage = Math.max(
    0,
    Math.min(100, value * 100),
  );

  const bar =
    type === "fatal"
      ? "from-red-500 to-rose-400"
      : type === "serious"
        ? "from-amber-500 to-orange-400"
        : "from-emerald-500 to-cyan-400";

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs text-slate-400">
          {label}
        </span>

        <span className="font-mono text-xs text-slate-300">
          {percentage.toFixed(2)}%
        </span>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-white/[0.05]">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${bar} transition-all duration-700`}
          style={{
            width: `${percentage}%`,
          }}
        />
      </div>
    </div>
  );
}

function ShapRow({
  item,
  positive,
}: {
  item: {
    feature: string;
    contribution: number;
  };
  positive: boolean;
}) {
  const magnitude = Math.min(
    Math.abs(item.contribution) * 500,
    100,
  );

  return (
    <div className="rounded-xl border border-white/[0.05] bg-white/[0.02] p-3 transition-all duration-300 hover:border-white/[0.09] hover:bg-white/[0.03]">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          {positive ? (
            <TrendingUp className="h-3.5 w-3.5 shrink-0 text-emerald-400" />
          ) : (
            <TrendingDown className="h-3.5 w-3.5 shrink-0 text-cyan-400" />
          )}

          <span className="truncate text-xs text-slate-300">
            {formatFeatureName(item.feature)}
          </span>
        </div>

        <span
          className={`shrink-0 font-mono text-[11px] ${
            positive
              ? "text-emerald-300"
              : "text-cyan-300"
          }`}
        >
          {item.contribution > 0 ? "+" : ""}
          {item.contribution.toFixed(4)}
        </span>
      </div>

      <div className="h-1 overflow-hidden rounded-full bg-white/[0.05]">
        <div
          className={`h-full rounded-full transition-all duration-700 ${
            positive
              ? "bg-gradient-to-r from-emerald-500 to-cyan-400"
              : "bg-gradient-to-r from-cyan-500 to-violet-400"
          }`}
          style={{
            width: `${magnitude}%`,
          }}
        />
      </div>
    </div>
  );
}

function MiniCapability({
  icon,
  label,
}: {
  icon: ReactNode;
  label: string;
}) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-white/[0.05] bg-white/[0.02] px-3 py-2">
      <span className="text-violet-300">{icon}</span>

      <span className="text-[11px] text-slate-500">
        {label}
      </span>
    </div>
  );
}