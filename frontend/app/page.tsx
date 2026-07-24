'use client';

import { useState, useEffect } from 'react';

function decimalToAmerican(decimal: number): string {
  if (!decimal || decimal <= 1) return '+100';
  if (decimal >= 2.0) {
    const american = Math.round((decimal - 1) * 100);
    return `+${american}`;
  } else {
    const american = Math.round(-100 / (decimal - 1));
    return `${american}`;
  }
}

function americanToDecimal(americanStr: string | number): number {
  const cleanStr = String(americanStr).trim().replace('+', '');
  const american = parseInt(cleanStr, 10);
  if (isNaN(american) || american === 0) return 2.0;
  if (american > 0) {
    return parseFloat((american / 100 + 1).toFixed(3));
  } else {
    return parseFloat((100 / Math.abs(american) + 1).toFixed(3));
  }
}

interface Fixture {
  id: number;
  date: string;
  home: string;
  away: string;
  odds_h: number;
  odds_d: number;
  odds_a: number;
}

interface SimResult {
  lambda_home: number;
  lambda_away: number;
  sims: {
    home_win_prob: number;
    draw_prob: number;
    away_win_prob: number;
    over_2_5_prob: number;
    btts_prob: number;
    avg_home_goals: number;
    avg_away_goals: number;
    top_scorelines: Record<string, number>;
  };
  suggestions: string[];
}

export default function Home() {
  const [fixtures, setFixtures] = useState<Fixture[]>([]);
  const [teams, setTeams] = useState<string[]>([]);
  
  const [homeTeam, setHomeTeam] = useState('Real Madrid');
  const [awayTeam, setAwayTeam] = useState('Real Sociedad');
  
  const [americanOddsH, setAmericanOddsH] = useState('-154');
  const [americanOddsD, setAmericanOddsD] = useState('+290');
  const [americanOddsA, setAmericanOddsA] = useState('+425');

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SimResult | null>(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/fixtures')
      .then((res) => res.json())
      .then((data) => setFixtures(data))
      .catch((err) => console.error('Failed to load fixtures:', err));

    fetch('http://localhost:8000/api/teams')
      .then((res) => res.json())
      .then((data) => {
        if (data.teams && data.teams.length > 0) {
          setTeams(data.teams);
        } else {
          setTeams(['FC Barcelona', 'Real Madrid', 'Real Sociedad', 'Athletic Club', 'Atletico Madrid', 'Sevilla FC']);
        }
      })
      .catch(() => {
        setTeams(['FC Barcelona', 'Real Madrid', 'Real Sociedad', 'Athletic Club', 'Atletico Madrid', 'Sevilla FC']);
      });
  }, []);

const loadFixtureIntoSim = (f: Fixture) => {
    // 1. Reset simulation results so old metrics disappear immediately
    setResult(null);

    // 2. Ensure team options exist in state
    setTeams((prev) => {
      const updated = [...prev];
      if (!updated.includes(f.home)) updated.push(f.home);
      if (!updated.includes(f.away)) updated.push(f.away);
      return updated;
    });

    // 3. Set teams directly
    setHomeTeam(f.home);
    setAwayTeam(f.away);

    // 4. Set American odds strings
    setAmericanOddsH(decimalToAmerican(f.odds_h));
    setAmericanOddsD(decimalToAmerican(f.odds_d));
    setAmericanOddsA(decimalToAmerican(f.odds_a));
  };

  const handleSimulate = async () => {
    setLoading(true);
    const decH = americanToDecimal(americanOddsH);
    const decD = americanToDecimal(americanOddsD);
    const decA = americanToDecimal(americanOddsA);

    try {
      const res = await fetch('http://localhost:8000/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          home_team: homeTeam,
          away_team: awayTeam,
          odds_h: decH,
          odds_d: decD,
          odds_a: decA,
        }),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error('Simulation error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans">
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="h-8 w-1.5 bg-emerald-500 rounded-full"></div>
            <div>
              <h1 className="text-xl font-bold text-white flex items-center gap-2">
                LA LIGA <span className="text-emerald-400 font-extrabold">SIMULATION ENGINE</span>
              </h1>
              <p className="text-xs text-slate-400">Monte Carlo 10k Poisson Model</p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* FIXTURE SIDEBAR */}
        <section className="lg:col-span-4 space-y-3">
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Matchday Fixtures & Lines
          </h2>
          <div className="max-h-[640px] overflow-y-auto pr-2 space-y-2.5">
            {fixtures.map((f) => (
              <div
                key={f.id}
                className="p-3.5 bg-slate-900/60 border border-slate-800 rounded-xl flex flex-col gap-2.5 hover:border-slate-700 transition"
              >
                <div className="flex justify-between items-center text-xs text-slate-400">
                  <span className="font-mono">{f.date}</span>
                  <div className="flex gap-1.5 font-mono text-[11px]">
                    <span className="bg-slate-950 px-1.5 py-0.5 rounded text-emerald-400">
                      {decimalToAmerican(f.odds_h)}
                    </span>
                    <span className="bg-slate-950 px-1.5 py-0.5 rounded text-amber-400">
                      {decimalToAmerican(f.odds_d)}
                    </span>
                    <span className="bg-slate-950 px-1.5 py-0.5 rounded text-sky-400">
                      {decimalToAmerican(f.odds_a)}
                    </span>
                  </div>
                </div>

                <div className="flex justify-between items-center">
                  <p className="text-sm font-semibold text-slate-200">
                    {f.home} <span className="text-slate-500 font-normal">vs</span> {f.away}
                  </p>
                  <button
                    onClick={() => loadFixtureIntoSim(f)}
                    className="text-xs bg-slate-800 hover:bg-emerald-600 hover:text-slate-950 text-slate-300 font-medium px-2.5 py-1 rounded-md border border-slate-700 transition cursor-pointer"
                  >
                    Load Match
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* MAIN WORKSPACE */}
        <section className="lg:col-span-8 space-y-6">
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h2 className="text-lg font-bold text-white mb-6">Match Configuration</h2>

            {/* TEAM DROPDOWNS */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-5">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5 uppercase">
                  Home Team
                </label>
                <select
                  value={homeTeam}
                  onChange={(e) => setHomeTeam(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm text-white focus:ring-2 focus:ring-emerald-500"
                >
                  {teams.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5 uppercase">
                  Away Team
                </label>
                <select
                  value={awayTeam}
                  onChange={(e) => setAwayTeam(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm text-white focus:ring-2 focus:ring-emerald-500"
                >
                  {teams.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* AMERICAN ODDS INPUTS */}
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">
                  Home Win (American)
                </label>
                <input
                  type="text"
                  value={americanOddsH}
                  onChange={(e) => setAmericanOddsH(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-sm font-mono text-emerald-400 focus:ring-2 focus:ring-emerald-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">
                  Draw (American)
                </label>
                <input
                  type="text"
                  value={americanOddsD}
                  onChange={(e) => setAmericanOddsD(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-sm font-mono text-amber-400 focus:ring-2 focus:ring-emerald-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">
                  Away Win (American)
                </label>
                <input
                  type="text"
                  value={americanOddsA}
                  onChange={(e) => setAmericanOddsA(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-sm font-mono text-sky-400 focus:ring-2 focus:ring-emerald-500"
                />
              </div>
            </div>

            <button
              onClick={handleSimulate}
              disabled={loading}
              className="w-full py-3.5 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold rounded-xl transition cursor-pointer"
            >
              {loading ? 'Running 10,000 Monte Carlo Iterations...' : 'Run Simulation Engine'}
            </button>
          </div>

          {/* RESULTS DISPLAY */}
          {result && (
            <div className="space-y-6">
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex justify-between items-center">
                <div>
                  <h3 className="text-2xl font-black text-white">
                    {homeTeam} <span className="text-slate-500 font-normal">vs</span> {awayTeam}
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">
                    LightGBM Projected xG: <span className="text-emerald-400 font-mono">{result.lambda_home.toFixed(2)}</span> – <span className="text-emerald-400 font-mono">{result.lambda_away.toFixed(2)}</span>
                  </p>
                </div>
                <div className="text-right">
                  <span className="text-xs text-slate-400 block">Avg Simulated Goals</span>
                  <span className="text-xl font-mono font-bold text-slate-100">
                    {result.sims.avg_home_goals.toFixed(2)} - {result.sims.avg_away_goals.toFixed(2)}
                  </span>
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                <h4 className="text-xs font-semibold text-slate-400 uppercase mb-4">
                  Model Win Probabilities vs Sportsbook Implied
                </h4>
                <div className="grid grid-cols-3 gap-4">
                  <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl text-center">
                    <p className="text-xs text-slate-400 mb-1">{homeTeam}</p>
                    <p className="text-2xl font-black text-emerald-400">
                      {(result.sims.home_win_prob * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs font-mono text-slate-500 mt-1">
                      Implied: {((1 / americanToDecimal(americanOddsH)) * 100).toFixed(1)}%
                    </p>
                  </div>

                  <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl text-center">
                    <p className="text-xs text-slate-400 mb-1">Draw</p>
                    <p className="text-2xl font-black text-amber-400">
                      {(result.sims.draw_prob * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs font-mono text-slate-500 mt-1">
                      Implied: {((1 / americanToDecimal(americanOddsD)) * 100).toFixed(1)}%
                    </p>
                  </div>

                  <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl text-center">
                    <p className="text-xs text-slate-400 mb-1">{awayTeam}</p>
                    <p className="text-2xl font-black text-sky-400">
                      {(result.sims.away_win_prob * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs font-mono text-slate-500 mt-1">
                      Implied: {((1 / americanToDecimal(americanOddsA)) * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                <h4 className="text-xs font-bold uppercase text-emerald-400 mb-3">Model Insights</h4>
                <div className="space-y-2 text-sm text-slate-200">
                  {result.suggestions.map((sug, i) => (
                    <div key={i} className="flex items-start gap-2 bg-slate-950/60 p-3 rounded-xl border border-slate-800">
                      <span className="text-emerald-400 font-bold">›</span>
                      <p>{sug}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}