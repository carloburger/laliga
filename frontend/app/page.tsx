"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const DEFAULT_MATCHDAYS = Array.from({ length: 38 }, (_, i) => i + 1);

// ==========================================
// TYPES
// ==========================================

interface Fixture {
  id: number;
  matchday: number;
  date: string;
  home: string;
  away: string;
  // Optional until the backend returns them (Football-Data: status + score.fullTime)
  status?: string;
  home_score?: number | null;
  away_score?: number | null;
}

interface Sims {
  home_win_prob: number;
  draw_prob: number;
  away_win_prob: number;
  over_2_5_prob: number;
  btts_prob: number;
}

interface Prediction {
  lambda_home: number;
  lambda_away: number;
  sims: Sims;
}

type Outcome = "H" | "D" | "A";

// ==========================================
// HELPERS
// ==========================================

const STATUS_BADGES: Record<string, { label: string; className: string }> = {
  FINISHED: { label: "FT", className: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30" },
  IN_PLAY: { label: "Live", className: "bg-red-500/15 text-red-300 border-red-500/30 animate-pulse" },
  PAUSED: { label: "HT", className: "bg-red-500/15 text-red-300 border-red-500/30" },
  POSTPONED: { label: "Postponed", className: "bg-slate-700/40 text-slate-300 border-slate-600" },
  SUSPENDED: { label: "Suspended", className: "bg-slate-700/40 text-slate-300 border-slate-600" },
  CANCELLED: { label: "Cancelled", className: "bg-slate-700/40 text-slate-300 border-slate-600" },
};

function hasScore(f: Fixture): boolean {
  return (
    (f.status === "FINISHED" || f.status === "IN_PLAY" || f.status === "PAUSED") &&
    f.home_score != null &&
    f.away_score != null
  );
}

function actualOutcome(f: Fixture): Outcome | null {
  if (f.status !== "FINISHED" || f.home_score == null || f.away_score == null) return null;
  if (f.home_score > f.away_score) return "H";
  if (f.home_score < f.away_score) return "A";
  return "D";
}

function predictedOutcome(s: Sims): Outcome {
  const { home_win_prob: h, draw_prob: d, away_win_prob: a } = s;
  if (h >= d && h >= a) return "H";
  return a >= d ? "A" : "D";
}

function outcomeLabel(outcome: Outcome, f: Fixture): string {
  if (outcome === "H") return `${f.home} win`;
  if (outcome === "A") return `${f.away} win`;
  return "Draw";
}

function outcomeProb(outcome: Outcome, s: Sims): number {
  if (outcome === "H") return s.home_win_prob;
  if (outcome === "A") return s.away_win_prob;
  return s.draw_prob;
}

// ==========================================
// SMALL COMPONENTS
// ==========================================

function StatusBadge({ status }: { status?: string }) {
  const badge = status ? STATUS_BADGES[status] : undefined;
  if (!badge) return null;
  return (
    <span className={`px-2 py-0.5 rounded border text-[10px] font-semibold ${badge.className}`}>
      {badge.label}
    </span>
  );
}

function ProbabilityBar({ label, prob, highlight }: { label: string; prob: number; highlight: boolean }) {
  const pct = prob * 100;
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className={highlight ? "text-amber-400 font-semibold" : "text-slate-400"}>{label}</span>
        <span className="font-mono font-bold text-white">{pct.toFixed(1)}%</span>
      </div>
      <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
        <div
          className={`h-full rounded-full ${highlight ? "bg-amber-400" : "bg-slate-600"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ==========================================
// MAIN COMPONENT
// ==========================================

export default function Home() {
  const [matchdays, setMatchdays] = useState<number[]>([]);
  const [matchday, setMatchday] = useState<number | null>(null);
  const [fixtures, setFixtures] = useState<Fixture[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [loadingFixtures, setLoadingFixtures] = useState(false);
  const [fixturesError, setFixturesError] = useState<string | null>(null);

  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  const selectedFixture = fixtures.find((f) => f.id === selectedId) ?? null;
  const home = selectedFixture?.home;
  const away = selectedFixture?.away;

  // 1. Load the matchday list and start on the current matchday
  useEffect(() => {
    let cancelled = false;

    async function loadMatchdays() {
      let list = DEFAULT_MATCHDAYS;
      let current = 1;
      try {
        const res = await fetch(`${API_BASE}/api/matchdays`);
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data.matchdays) && data.matchdays.length > 0) list = data.matchdays;
          if (typeof data.current_matchday === "number") current = data.current_matchday;
        }
      } catch {
        // Backend unreachable: fall back to defaults; the fixtures request will report the error
      }
      if (cancelled) return;
      setMatchdays(list);
      setMatchday(current);
    }

    loadMatchdays();
    return () => {
      cancelled = true;
    };
  }, []);

  // 2. Load fixtures whenever the matchday changes
  useEffect(() => {
    if (matchday === null) return;
    const day = matchday;
    let cancelled = false;

    async function loadFixtures() {
      setLoadingFixtures(true);
      setFixturesError(null);
      setPrediction(null);
      try {
        const res = await fetch(`${API_BASE}/api/fixtures?matchday=${day}`);
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `Request failed (${res.status})`);
        }
        const data = await res.json();
        const list: Fixture[] = Array.isArray(data) ? data : data.fixtures ?? [];
        if (cancelled) return;
        setFixtures(list);
        setSelectedId(list[0]?.id ?? null);
      } catch (err) {
        if (cancelled) return;
        setFixtures([]);
        setSelectedId(null);
        setFixturesError(err instanceof Error ? err.message : "Could not reach the backend.");
      } finally {
        if (!cancelled) setLoadingFixtures(false);
      }
    }

    loadFixtures();
    return () => {
      cancelled = true;
    };
  }, [matchday]);

  // 3. Predict the selected match automatically
  useEffect(() => {
    if (!home || !away) {
      setPrediction(null);
      setLoadingPrediction(false);
      return;
    }
    let cancelled = false;

    async function predict(homeTeam: string, awayTeam: string) {
      setLoadingPrediction(true);
      setPredictionError(null);
      setPrediction(null);
      try {
        const res = await fetch(`${API_BASE}/api/simulate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam }),
        });
        if (!res.ok) throw new Error(`Prediction failed (${res.status})`);
        const data: Prediction = await res.json();
        if (!cancelled) setPrediction(data);
      } catch (err) {
        if (!cancelled) {
          setPredictionError(err instanceof Error ? err.message : "Prediction failed.");
        }
      } finally {
        if (!cancelled) setLoadingPrediction(false);
      }
    }

    predict(home, away);
    return () => {
      cancelled = true;
    };
  }, [home, away]);

  const pick = prediction ? predictedOutcome(prediction.sims) : null;
  const actual = selectedFixture ? actualOutcome(selectedFixture) : null;

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-10 font-sans">
      {/* Header */}
      <header className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white">
            La Liga Match Predictor <span className="text-amber-400">2026/27</span>
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Win, draw and loss probabilities from a Dixon-Coles Poisson model with LightGBM
          </p>
        </div>

        <div className="flex items-center gap-3 bg-slate-900 border border-slate-800 p-2.5 rounded-xl shadow-md">
          <label htmlFor="matchday-select" className="text-xs font-semibold text-slate-400">
            Matchday
          </label>
          <select
            id="matchday-select"
            value={matchday ?? ""}
            disabled={matchday === null}
            onChange={(e) => setMatchday(Number(e.target.value))}
            className="bg-slate-800 text-amber-400 font-bold px-3 py-1.5 rounded-lg border border-slate-700 focus:outline-none focus:ring-2 focus:ring-amber-400 cursor-pointer disabled:opacity-50"
          >
            {matchdays.map((m) => (
              <option key={m} value={m}>
                Matchday {m}
              </option>
            ))}
          </select>
        </div>
      </header>

      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-12 gap-8">
        {/* Left: fixtures */}
        <section className="md:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-bold text-white">Matchday {matchday ?? "-"} fixtures</h2>
            <span className="text-xs font-mono bg-slate-800 text-slate-400 px-2.5 py-1 rounded-md border border-slate-700">
              {fixtures.length} matches
            </span>
          </div>

          <div className="space-y-3 max-h-[660px] overflow-y-auto pr-1">
            {loadingFixtures || matchday === null ? (
              <div className="text-center py-12 text-slate-500 font-mono text-sm">Loading fixtures...</div>
            ) : fixturesError ? (
              <div className="text-center py-12 px-4 text-sm text-red-300">
                <p className="font-semibold mb-1">Could not load fixtures</p>
                <p className="text-slate-400">{fixturesError}</p>
              </div>
            ) : fixtures.length === 0 ? (
              <div className="text-center py-12 text-slate-500 text-sm">No fixtures for this matchday yet.</div>
            ) : (
              fixtures.map((f) => {
                const isSelected = selectedId === f.id;
                return (
                  <button
                    key={f.id}
                    type="button"
                    onClick={() => setSelectedId(f.id)}
                    className={`w-full text-left p-4 rounded-xl border transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-amber-400 ${
                      isSelected
                        ? "bg-slate-800 border-amber-400 shadow-lg shadow-amber-500/10"
                        : "bg-slate-950/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900"
                    }`}
                  >
                    <div className="flex justify-between items-center text-xs mb-2">
                      <span className="text-amber-400 font-semibold">{f.date}</span>
                      <StatusBadge status={f.status} />
                    </div>

                    <div className="flex justify-between items-center text-sm font-semibold text-slate-200">
                      <span className="truncate pr-1">{f.home}</span>
                      {hasScore(f) ? (
                        <span className="font-mono font-bold text-white px-2 whitespace-nowrap">
                          {f.home_score} - {f.away_score}
                        </span>
                      ) : (
                        <span className="text-xs text-slate-500 font-normal px-2">vs</span>
                      )}
                      <span className="truncate pl-1 text-right">{f.away}</span>
                    </div>
                  </button>
                );
              })
            )}
          </div>
        </section>

        {/* Right: prediction */}
        <section className="md:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
          <h2 className="text-xl font-bold text-white mb-4">Match prediction</h2>

          {!selectedFixture ? (
            <div className="text-center py-20 text-slate-500">Select a match to see the model&apos;s prediction.</div>
          ) : (
            <div className="space-y-6">
              {/* Match header */}
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 flex justify-between items-center gap-4">
                <div className="min-w-0">
                  <p className="text-xs text-slate-400">Match</p>
                  <p className="text-base font-bold text-amber-400 truncate">
                    {selectedFixture.home} vs {selectedFixture.away}
                  </p>
                </div>
                <div className="text-right shrink-0">
                  <div className="flex items-center justify-end gap-2 mb-0.5">
                    <p className="text-xs text-slate-400">Kickoff</p>
                    <StatusBadge status={selectedFixture.status} />
                  </div>
                  <p className="text-sm font-mono text-slate-200">{selectedFixture.date}</p>
                </div>
              </div>

              {/* Score (finished or live) */}
              {hasScore(selectedFixture) && (
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 text-center">
                  <p className="text-xs text-slate-400 mb-1">
                    {selectedFixture.status === "FINISHED" ? "Final score" : "Current score"}
                  </p>
                  <p className="text-3xl font-black font-mono text-white">
                    {selectedFixture.home_score} - {selectedFixture.away_score}
                  </p>
                </div>
              )}

              {loadingPrediction && (
                <div className="text-center py-10 text-slate-500 font-mono text-sm">Running the model...</div>
              )}

              {predictionError && !loadingPrediction && (
                <div className="p-4 rounded-xl border border-red-500/30 bg-red-500/10 text-sm text-red-300">
                  {predictionError}. Check that the backend is running and that /api/simulate returns the prediction.
                </div>
              )}

              {prediction && pick && !loadingPrediction && (
                <>
                  {/* Model pick */}
                  <div className="bg-slate-950 p-4 rounded-xl border border-amber-400/40 flex justify-between items-center gap-4">
                    <div>
                      <p className="text-xs text-slate-400">Model pick</p>
                      <p className="text-lg font-bold text-amber-400">{outcomeLabel(pick, selectedFixture)}</p>
                      <p className="text-xs text-slate-400 font-mono mt-0.5">
                        {(outcomeProb(pick, prediction.sims) * 100).toFixed(1)}% probability
                      </p>
                    </div>
                    {actual && (
                      <div className="text-right">
                        <p
                          className={`text-sm font-bold ${
                            actual === pick ? "text-emerald-400" : "text-red-400"
                          }`}
                        >
                          {actual === pick ? "Correct" : "Missed"}
                        </p>
                        <p className="text-xs text-slate-400">Result: {outcomeLabel(actual, selectedFixture)}</p>
                      </div>
                    )}
                  </div>

                  {/* Probabilities */}
                  <div className="space-y-3">
                    <ProbabilityBar
                      label={`${selectedFixture.home} win`}
                      prob={prediction.sims.home_win_prob}
                      highlight={pick === "H"}
                    />
                    <ProbabilityBar label="Draw" prob={prediction.sims.draw_prob} highlight={pick === "D"} />
                    <ProbabilityBar
                      label={`${selectedFixture.away} win`}
                      prob={prediction.sims.away_win_prob}
                      highlight={pick === "A"}
                    />
                  </div>

                  {/* Expected goals */}
                  <div className="grid grid-cols-2 gap-4 text-center">
                    <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                      <p className="text-xs text-slate-400">{selectedFixture.home} expected goals</p>
                      <p className="text-2xl font-black text-emerald-400">{prediction.lambda_home}</p>
                    </div>
                    <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                      <p className="text-xs text-slate-400">{selectedFixture.away} expected goals</p>
                      <p className="text-2xl font-black text-emerald-400">{prediction.lambda_away}</p>
                    </div>
                  </div>

                  {/* Goals outlook */}
                  <div className="grid grid-cols-2 gap-4 text-center">
                    <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                      <p className="text-xs text-slate-400">Over 2.5 goals</p>
                      <p className="text-lg font-bold text-white">
                        {(prediction.sims.over_2_5_prob * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                      <p className="text-xs text-slate-400">Both teams score</p>
                      <p className="text-lg font-bold text-white">
                        {(prediction.sims.btts_prob * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}