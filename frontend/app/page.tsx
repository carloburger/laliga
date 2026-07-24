"use client";

import { useEffect, useState } from "react";

// ==========================================
// TYPES & INTERFACES
// ==========================================

interface Fixture {
  id: number;
  matchday: number;
  date: string;
  home: string;
  away: string;
  odds_h_american?: string;
  odds_d_american?: string;
  odds_a_american?: string;
  odds_h: number | null;
  odds_d: number | null;
  odds_a: number | null;
}

interface SimulationResult {
  lambda_home: number;
  lambda_away: number;
  sims: {
    home_win_prob: number;
    draw_prob: number;
    away_win_prob: number;
    over_2_5_prob: number;
    btts_prob: number;
  };
  fair_odds: {
    odds_h_american: string;
    odds_d_american: string;
    odds_a_american: string;
    odds_h: number;
    odds_d: number;
    odds_a: number;
  };
  suggestions: string[];
}

interface OddsState {
  home: string;
  draw: string;
  away: string;
}

// ==========================================
// HELPER UTILITIES
// ==========================================

/**
 * Converts American Odds string (+110, -125) to Implied Probability Percentage (%)
 */
function parseAmericanToImplied(americanStr: string): number {
  if (!americanStr) return 0;
  const clean = americanStr.replace("+", "").trim();
  const num = parseFloat(clean);
  if (isNaN(num) || num === 0) return 0;

  if (num > 0) {
    return (100 / (num + 100)) * 100;
  } else {
    const pos = Math.abs(num);
    return (pos / (pos + 100)) * 100;
  }
}

// ==========================================
// MAIN COMPONENT
// ==========================================

export default function Home() {
  // Navigation & Data States
  const [matchday, setMatchday] = useState<number>(1);
  const [availableMatchdays, setAvailableMatchdays] = useState<number[]>([]);
  const [fixtures, setFixtures] = useState<Fixture[]>([]);
  const [selectedFixture, setSelectedFixture] = useState<Fixture | null>(null);

  // Editable Market Odds State (American Odds)
  const [odds, setOdds] = useState<OddsState>({
    home: "+110",
    draw: "+240",
    away: "+220",
  });

  // Simulation & Async Loading States
  const [simulation, setSimulation] = useState<SimulationResult | null>(null);
  const [loadingFixtures, setLoadingFixtures] = useState<boolean>(false);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);

  // 1. Fetch available matchday dropdown list
  useEffect(() => {
    async function fetchMatchdays() {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/matchdays");
        const data = await res.json();
        setAvailableMatchdays(data.matchdays || Array.from({ length: 38 }, (_, i) => i + 1));
      } catch (err) {
        setAvailableMatchdays(Array.from({ length: 38 }, (_, i) => i + 1));
      }
    }
    fetchMatchdays();
  }, []);

  // 2. Fetch fixtures whenever selected matchday changes
  useEffect(() => {
    async function fetchFixtures() {
      setLoadingFixtures(true);
      setSimulation(null);
      try {
        const res = await fetch(`http://127.0.0.1:8000/api/fixtures?matchday=${matchday}`);
        const data = await res.json();
        const list: Fixture[] = Array.isArray(data) ? data : data.fixtures || [];
        setFixtures(list);

        if (list.length > 0) {
          handleSelectFixture(list[0]);
        }
      } catch (err) {
        console.error("Failed to load fixtures:", err);
        setFixtures([]);
      } finally {
        setLoadingFixtures(false);
      }
    }
    fetchFixtures();
  }, [matchday]);

  // Handle Match Selection & Sync Editable Inputs
  const handleSelectFixture = (fixture: Fixture) => {
    setSelectedFixture(fixture);
    setOdds({
      home: fixture.odds_h_american || "+110",
      draw: fixture.odds_d_american || "+240",
      away: fixture.odds_a_american || "+220",
    });
    setSimulation(null);
  };

  // Run Monte Carlo Simulation
  const handleRunSimulation = async () => {
    if (!selectedFixture) return;
    setIsSimulating(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          home_team: selectedFixture.home,
          away_team: selectedFixture.away,
          odds_h: odds.home,
          odds_d: odds.draw,
          odds_a: odds.away,
        }),
      });
      const data = await res.json();
      setSimulation(data);
    } catch (err) {
      console.error("Simulation error:", err);
    } finally {
      setIsSimulating(false);
    }
  };

  // Compute Live Implied Probabilities and Margin
  const impliedH = parseAmericanToImplied(odds.home);
  const impliedD = parseAmericanToImplied(odds.draw);
  const impliedA = parseAmericanToImplied(odds.away);
  const marketMargin = impliedH + impliedD + impliedA;

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-10 font-sans">
      {/* Header Bar */}
      <header className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white">
            La Liga Intelligence Platform <span className="text-amber-400">2026/27</span>
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Monte Carlo Simulations • Dixon-Coles Poisson • American Market Lines
          </p>
        </div>

        {/* Matchday Selector */}
        <div className="flex items-center gap-3 bg-slate-900 border border-slate-800 p-2.5 rounded-xl shadow-md">
          <label htmlFor="matchday-select" className="text-xs font-semibold uppercase text-slate-400">
            Matchday:
          </label>
          <select
            id="matchday-select"
            value={matchday}
            onChange={(e) => setMatchday(Number(e.target.value))}
            className="bg-slate-800 text-amber-400 font-bold px-3 py-1.5 rounded-lg border border-slate-700 focus:outline-none focus:ring-2 focus:ring-amber-400 cursor-pointer"
          >
            {availableMatchdays.map((m) => (
              <option key={m} value={m}>
                Matchday {m}
              </option>
            ))}
          </select>
        </div>
      </header>

      {/* Main Grid Layout */}
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-12 gap-8">
        
        {/* Left Panel: Fixtures List */}
        <section className="md:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-bold text-white">Matchday {matchday} Schedule</h2>
            <span className="text-xs font-mono bg-slate-800 text-slate-400 px-2.5 py-1 rounded-md border border-slate-700">
              {fixtures.length} Fixtures
            </span>
          </div>

          <div className="space-y-3 max-h-[660px] overflow-y-auto pr-1 custom-scrollbar">
            {loadingFixtures ? (
              <div className="text-center py-12 text-slate-500 font-mono text-sm">
                Fetching fixtures from pipeline...
              </div>
            ) : fixtures.length === 0 ? (
              <div className="text-center py-12 text-slate-500 text-sm">
                No fixtures found for this matchday.
              </div>
            ) : (
              fixtures.map((f) => {
                const isSelected = selectedFixture?.id === f.id;
                return (
                  <div
                    key={f.id}
                    onClick={() => handleSelectFixture(f)}
                    className={`p-4 rounded-xl border transition-all cursor-pointer ${
                      isSelected
                        ? "bg-slate-800 border-amber-400 shadow-lg shadow-amber-500/10"
                        : "bg-slate-950/60 border-slate-800 hover:border-slate-700 hover:bg-slate-850"
                    }`}
                  >
                    <div className="flex justify-between items-center text-xs text-slate-400 font-medium mb-2">
                      <span className="text-amber-400 font-semibold">{f.date}</span>
                      <span className="bg-slate-900 border border-slate-800 px-2 py-0.5 rounded text-[10px] font-mono">
                        EST
                      </span>
                    </div>

                    <div className="flex justify-between items-center text-sm font-semibold text-slate-200">
                      <span className="truncate pr-1">{f.home}</span>
                      <span className="text-xs text-slate-500 font-normal px-2">vs</span>
                      <span className="truncate pl-1 text-right">{f.away}</span>
                    </div>

                    <div className="grid grid-cols-3 gap-2 mt-3 text-center text-xs font-mono">
                      <div className="bg-slate-900/80 p-1.5 rounded border border-slate-800">
                        <span className="text-slate-500 text-[10px] block">1</span>
                        <span className="text-slate-300 font-bold">{f.odds_h_american || "-"}</span>
                      </div>
                      <div className="bg-slate-900/80 p-1.5 rounded border border-slate-800">
                        <span className="text-slate-500 text-[10px] block">X</span>
                        <span className="text-slate-300 font-bold">{f.odds_d_american || "-"}</span>
                      </div>
                      <div className="bg-slate-900/80 p-1.5 rounded border border-slate-800">
                        <span className="text-slate-500 text-[10px] block">2</span>
                        <span className="text-slate-300 font-bold">{f.odds_a_american || "-"}</span>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </section>

        {/* Right Panel: Odds Controls & Simulation Engine */}
        <section className="md:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
          <div>
            <h2 className="text-xl font-bold text-white mb-4">
              Match Intelligence & Predictive Engine
            </h2>

            {selectedFixture ? (
              <>
                {/* Active Fixture Header */}
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 mb-6 flex justify-between items-center">
                  <div>
                    <p className="text-xs text-slate-400">Selected Match</p>
                    <p className="text-base font-bold text-amber-400">
                      {selectedFixture.home} vs {selectedFixture.away}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-slate-400">Kickoff</p>
                    <p className="text-sm font-mono text-slate-200">{selectedFixture.date}</p>
                  </div>
                </div>

                {/* Editable Odds Controls */}
                <div className="bg-slate-950/80 border border-slate-800 p-4 rounded-xl mb-6">
                  <div className="flex justify-between items-center mb-3">
                    <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                      Market Lines (American Odds) & Overround
                    </h3>
                    <span className="text-[11px] font-mono text-slate-400">
                      Vig / Margin: <strong className="text-amber-400">{marketMargin.toFixed(1)}%</strong>
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-3 mb-4">
                    <div>
                      <label className="text-[11px] text-slate-400 block mb-1">Home (1)</label>
                      <input
                        type="text"
                        value={odds.home}
                        onChange={(e) => setOdds({ ...odds, home: e.target.value })}
                        className="w-full bg-slate-900 border border-slate-700 text-white font-mono font-bold px-3 py-1.5 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-400"
                        placeholder="+110"
                      />
                      <p className="text-[11px] text-slate-400 mt-1 font-mono">
                        Implied: <strong className="text-slate-200">{impliedH.toFixed(1)}%</strong>
                      </p>
                    </div>

                    <div>
                      <label className="text-[11px] text-slate-400 block mb-1">Draw (X)</label>
                      <input
                        type="text"
                        value={odds.draw}
                        onChange={(e) => setOdds({ ...odds, draw: e.target.value })}
                        className="w-full bg-slate-900 border border-slate-700 text-white font-mono font-bold px-3 py-1.5 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-400"
                        placeholder="+240"
                      />
                      <p className="text-[11px] text-slate-400 mt-1 font-mono">
                        Implied: <strong className="text-slate-200">{impliedD.toFixed(1)}%</strong>
                      </p>
                    </div>

                    <div>
                      <label className="text-[11px] text-slate-400 block mb-1">Away (2)</label>
                      <input
                        type="text"
                        value={odds.away}
                        onChange={(e) => setOdds({ ...odds, away: e.target.value })}
                        className="w-full bg-slate-900 border border-slate-700 text-white font-mono font-bold px-3 py-1.5 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-400"
                        placeholder="+220"
                      />
                      <p className="text-[11px] text-slate-400 mt-1 font-mono">
                        Implied: <strong className="text-slate-200">{impliedA.toFixed(1)}%</strong>
                      </p>
                    </div>
                  </div>

                  <button
                    onClick={handleRunSimulation}
                    disabled={isSimulating}
                    className="w-full bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold py-2.5 px-4 rounded-lg transition-colors shadow-lg shadow-amber-500/10 cursor-pointer disabled:opacity-50"
                  >
                    {isSimulating ? "Executing Monte Carlo (10,000 Sims)..." : "Run Match Simulation"}
                  </button>
                </div>

                {/* Simulation Output Card */}
                {simulation && (
                  <div className="space-y-6">
                    {/* Expected Goal Expectations (Clean λ rendering) */}
                    <div className="grid grid-cols-2 gap-4 text-center">
                      <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                        <p className="text-xs text-slate-400">Home xG (λh)</p>
                        <p className="text-2xl font-black text-emerald-400">{simulation.lambda_home}</p>
                      </div>
                      <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                        <p className="text-xs text-slate-400">Away xG (λa)</p>
                        <p className="text-2xl font-black text-emerald-400">{simulation.lambda_away}</p>
                      </div>
                    </div>

                    {/* Model Probability vs Market Implied Comparison */}
                    <div>
                      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                        Model Probability vs Market Implied
                      </h3>
                      <div className="grid grid-cols-3 gap-3 text-center">
                        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                          <p className="text-xs text-slate-400">Home Win</p>
                          <p className="text-lg font-bold text-white">
                            {(simulation.sims.home_win_prob * 100).toFixed(1)}%
                          </p>
                          <p className="text-[10px] text-slate-500 font-mono mt-0.5">
                            Mkt: {impliedH.toFixed(1)}%
                          </p>
                        </div>

                        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                          <p className="text-xs text-slate-400">Draw</p>
                          <p className="text-lg font-bold text-white">
                            {(simulation.sims.draw_prob * 100).toFixed(1)}%
                          </p>
                          <p className="text-[10px] text-slate-500 font-mono mt-0.5">
                            Mkt: {impliedD.toFixed(1)}%
                          </p>
                        </div>

                        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                          <p className="text-xs text-slate-400">Away Win</p>
                          <p className="text-lg font-bold text-white">
                            {(simulation.sims.away_win_prob * 100).toFixed(1)}%
                          </p>
                          <p className="text-[10px] text-slate-500 font-mono mt-0.5">
                            Mkt: {impliedA.toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Value Insights & Suggestions */}
                    <div className="space-y-2">
                      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                        Value Edge & Insights
                      </h3>
                      {simulation.suggestions.map((insight, idx) => (
                        <div key={idx} className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg text-xs text-amber-300 font-medium">
                          {insight}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center py-20 text-slate-500">
                Select a match from the schedule panel to enter custom lines and simulate.
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}