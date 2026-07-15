"use client";

import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  CheckCircle2,
  Compass,
  FileText,
  Flame,
  HelpCircle,
  Info,
  Layers3,
  Leaf,
  RefreshCcw,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Thermometer,
  Trash2,
  Trees,
  Waves,
  Zap
} from "lucide-react";

type CellType =
  | "dense-forest"
  | "sparse-forest"
  | "grassland"
  | "water"
  | "disturbed-soil"
  | "residential"
  | "industrial"
  | "green-highway";

type ScenarioKey = "wilderness" | "sprawl" | "sustainable" | "ecovillage" | null;

interface Cell {
  id: number;
  row: number;
  col: number;
  type: CellType;
  initialType: CellType;
}

interface CellTypeMeta {
  type: CellType;
  label: string;
  color: string;
  bgClass: string;
  canopy: number;
  builtUp: number;
  bareSoil: number;
  lstDelta: number;
  description: string;
}

const CELL_META: Record<CellType, CellTypeMeta> = {
  "dense-forest": {
    type: "dense-forest",
    label: "Dense Forest",
    color: "#163a28",
    bgClass: "cell-dense-forest",
    canopy: 100,
    builtUp: 0,
    bareSoil: 0,
    lstDelta: -2.0,
    description: "Highly protected forest core with closed canopy."
  },
  "sparse-forest": {
    type: "sparse-forest",
    label: "Sparse Forest",
    color: "#54a36f",
    bgClass: "cell-sparse-forest",
    canopy: 50,
    builtUp: 0,
    bareSoil: 10,
    lstDelta: -0.8,
    description: "Fragmented or regenerating woodland with open canopy."
  },
  grassland: {
    type: "grassland",
    label: "Grassland/Shrub",
    color: "#a4bfa7",
    bgClass: "cell-grassland",
    canopy: 15,
    builtUp: 0,
    bareSoil: 15,
    lstDelta: 0.0,
    description: "Open fields, natural shrublands, or grazing pastures."
  },
  water: {
    type: "water",
    label: "Water Body",
    color: "#27739b",
    bgClass: "cell-water",
    canopy: 0,
    builtUp: 0,
    bareSoil: 0,
    lstDelta: -1.5,
    description: "Natural streams, lakes, or drainage basins."
  },
  "disturbed-soil": {
    type: "disturbed-soil",
    label: "Disturbed Soil",
    color: "#b08765",
    bgClass: "cell-disturbed-soil",
    canopy: 0,
    builtUp: 0,
    bareSoil: 80,
    lstDelta: 1.0,
    description: "Cleared ground, quarries, or exposed topsoil."
  },
  residential: {
    type: "residential",
    label: "Residential Zone",
    color: "#e59d3d",
    bgClass: "cell-residential",
    canopy: 10,
    builtUp: 70,
    bareSoil: 20,
    lstDelta: 2.5,
    description: "Housing settlements, public squares, and gardens."
  },
  industrial: {
    type: "industrial",
    label: "Industrial/Mining",
    color: "#a53e32",
    bgClass: "cell-industrial",
    canopy: 0,
    builtUp: 95,
    bareSoil: 5,
    lstDelta: 4.5,
    description: "Concrete plants, mining pits, or logistics blocks."
  },
  "green-highway": {
    type: "green-highway",
    label: "Green Infrastructure",
    color: "#5c7d8c",
    bgClass: "cell-green-highway",
    canopy: 25,
    builtUp: 55,
    bareSoil: 20,
    lstDelta: 0.8,
    description: "Roadways built with wildlife underpasses and tree verges."
  }
};

const GRID_SIZE = 10;

interface MigrationRoute {
  id: string;
  name: string;
  coords: Array<{ r: number; c: number }>;
}

const MIGRATION_ROUTES: MigrationRoute[] = [
  {
    id: "elephant-corridor",
    name: "Elephant Migration Route (North-South)",
    coords: [
      { r: 0, c: 3 },
      { r: 1, c: 3 },
      { r: 2, c: 3 },
      { r: 3, c: 4 },
      { r: 4, c: 4 },
      { r: 5, c: 4 },
      { r: 6, c: 5 },
      { r: 7, c: 5 },
      { r: 8, c: 5 },
      { r: 9, c: 5 }
    ]
  },
  {
    id: "riparian-corridor",
    name: "Riparian Waterway Corridor (Diagonal)",
    coords: [
      { r: 9, c: 0 },
      { r: 8, c: 1 },
      { r: 7, c: 2 },
      { r: 6, c: 3 },
      { r: 5, c: 4 },
      { r: 4, c: 5 },
      { r: 3, c: 6 },
      { r: 2, c: 7 },
      { r: 1, c: 8 },
      { r: 0, c: 9 }
    ]
  },
  {
    id: "canopy-corridor",
    name: "Gibbon Canopy Corridor (West-East)",
    coords: [
      { r: 3, c: 0 },
      { r: 3, c: 1 },
      { r: 3, c: 2 },
      { r: 4, c: 3 },
      { r: 4, c: 4 },
      { r: 4, c: 5 },
      { r: 5, c: 6 },
      { r: 5, c: 7 },
      { r: 5, c: 8 },
      { r: 5, c: 9 }
    ]
  }
];

function createNaturalGrid(): Cell[] {
  const grid: Cell[] = [];
  for (let r = 0; r < GRID_SIZE; r += 1) {
    for (let c = 0; c < GRID_SIZE; c += 1) {
      const id = r * GRID_SIZE + c;
      let type: CellType = "grassland";

      const isWater = r + c === 9 || (r === 8 && c === 0) || (r === 0 && c === 8);
      const isCore = r >= 2 && r <= 6 && c >= 2 && c <= 6;
      const isBuffer = r >= 1 && r <= 7 && c >= 1 && c <= 7;

      if (isWater) {
        type = "water";
      } else if (isCore) {
        type = "dense-forest";
      } else if (isBuffer) {
        type = "sparse-forest";
      }

      grid.push({ id, row: r, col: c, type, initialType: type });
    }
  }
  return grid;
}

function getSvgPathString(coords: Array<{ r: number; c: number }>) {
  return coords
    .map((pt, idx) => `${idx === 0 ? "M" : "L"} ${pt.c * 40 + 20} ${pt.r * 40 + 20}`)
    .join(" ");
}

const SCENARIO_META: Record<Exclude<ScenarioKey, null>, { label: string; icon: React.ReactNode; btnClass: string; desc: string }> = {
  wilderness: {
    label: "Natural State",
    icon: <Trees size={13} />,
    btnClass: "",
    desc: "Pristine wilderness baseline"
  },
  sprawl: {
    label: "Unplanned Sprawl",
    icon: <Flame size={13} />,
    btnClass: "warning-btn",
    desc: "Max urban encroachment"
  },
  sustainable: {
    label: "Smart Growth",
    icon: <Sparkles size={13} />,
    btnClass: "success-btn",
    desc: "Eco-optimised development"
  },
  ecovillage: {
    label: "Eco-Village",
    icon: <Leaf size={13} />,
    btnClass: "info-btn",
    desc: "Low-footprint settlement"
  }
};

export default function PlanningSandbox() {
  const [grid, setGrid] = useState<Cell[]>(() => createNaturalGrid());
  const [selectedTool, setSelectedTool] = useState<CellType | "clear">("residential");
  const [isMouseDown, setIsMouseDown] = useState(false);
  const [showWildlifePaths, setShowWildlifePaths] = useState(true);
  const [activeScenario, setActiveScenario] = useState<ScenarioKey>("wilderness");

  // Policy Toggles
  const [policyWildlife, setPolicyWildlife] = useState(false);
  const [policyGreenRoofs, setPolicyGreenRoofs] = useState(false);
  const [policyDrainage, setPolicyDrainage] = useState(false);

  const initialStats = useMemo(() => {
    let totalCanopy = 0;
    let totalBareSoil = 0;
    let totalBuiltUp = 0;

    for (const cell of grid) {
      const meta = CELL_META[cell.initialType];
      totalCanopy += meta.canopy;
      totalBareSoil += meta.bareSoil;
      totalBuiltUp += meta.builtUp;
    }

    return {
      canopy: totalCanopy / GRID_SIZE ** 2,
      bareSoil: totalBareSoil / GRID_SIZE ** 2,
      builtUp: totalBuiltUp / GRID_SIZE ** 2
    };
  }, [grid]);

  const simulation = useMemo(() => {
    let currentCanopy = 0;
    let currentBareSoil = 0;
    let currentBuiltUp = 0;
    let baseLst = 0;
    let forestCount = 0;
    let residentialCount = 0;
    let industrialCount = 0;
    let roadCount = 0;

    for (const cell of grid) {
      const meta = CELL_META[cell.type];
      currentCanopy += meta.canopy;
      currentBareSoil += meta.bareSoil;
      currentBuiltUp += meta.builtUp;

      let cellLst = meta.lstDelta;
      if (policyGreenRoofs && (cell.type === "residential" || cell.type === "industrial")) {
        cellLst -= 1.2;
      }
      baseLst += cellLst;

      if (cell.type === "dense-forest" || cell.type === "sparse-forest") forestCount += 1;
      if (cell.type === "residential") residentialCount += 1;
      if (cell.type === "industrial") industrialCount += 1;
      if (cell.type === "green-highway") roadCount += 1;
    }

    const canopyPct = currentCanopy / GRID_SIZE ** 2;
    const bareSoilPct = currentBareSoil / GRID_SIZE ** 2;
    const builtUpPct = currentBuiltUp / GRID_SIZE ** 2;
    const lstDelta = baseLst / GRID_SIZE ** 2;

    const canopyLoss = initialStats.canopy > 0
      ? Math.max(0, (initialStats.canopy - canopyPct) / initialStats.canopy) * 100
      : 0;

    const visited = new Set<number>();
    let forestPatches = 0;

    for (let r = 0; r < GRID_SIZE; r += 1) {
      for (let c = 0; c < GRID_SIZE; c += 1) {
        const idx = r * GRID_SIZE + c;
        const cell = grid[idx];
        const isForest = cell.type === "dense-forest" || cell.type === "sparse-forest";

        if (isForest && !visited.has(idx)) {
          forestPatches += 1;
          const queue = [idx];
          visited.add(idx);
          while (queue.length > 0) {
            const curr = queue.shift()!;
            const cr = Math.floor(curr / GRID_SIZE);
            const cc = curr % GRID_SIZE;
            const neighbors = [
              { r: cr - 1, c: cc },
              { r: cr + 1, c: cc },
              { r: cr, c: cc - 1 },
              { r: cr, c: cc + 1 }
            ];
            for (const n of neighbors) {
              if (n.r >= 0 && n.r < GRID_SIZE && n.c >= 0 && n.c < GRID_SIZE) {
                const nIdx = n.r * GRID_SIZE + n.c;
                const nCell = grid[nIdx];
                const nForest = nCell.type === "dense-forest" || nCell.type === "sparse-forest";
                if (nForest && !visited.has(nIdx)) {
                  visited.add(nIdx);
                  queue.push(nIdx);
                }
              }
            }
          }
        }
      }
    }

    let fragmentationIndex = 0;
    if (forestCount === 0) {
      fragmentationIndex = 100;
    } else if (forestPatches > 1) {
      fragmentationIndex = Math.min(100, 10 + (forestPatches - 1) * 20);
    } else {
      fragmentationIndex = 5;
    }

    if (policyWildlife && fragmentationIndex > 5) {
      fragmentationIndex = Math.max(5, fragmentationIndex - 35);
    }

    let soilRiskScore = bareSoilPct * 1.5;
    if (policyDrainage) {
      soilRiskScore = Math.max(0, soilRiskScore - 12);
    }
    soilRiskScore = Math.min(100, Math.max(0, soilRiskScore));

    const routesStatus = MIGRATION_ROUTES.map((route) => {
      let isBlocked = false;
      const blockedCells: Array<{ r: number; c: number }> = [];

      for (const pt of route.coords) {
        const cell = grid[pt.r * GRID_SIZE + pt.c];
        const isUrbanBarrier = cell.type === "residential" || cell.type === "industrial";
        const isRoadBarrier = cell.type === "green-highway" && !policyWildlife;

        if (isUrbanBarrier || isRoadBarrier) {
          isBlocked = true;
          blockedCells.push(pt);
        }
      }

      return { ...route, isBlocked, blockedCells };
    });

    const activePathsCount = routesStatus.filter((r) => !r.isBlocked).length;
    const connectivityPct = Math.round((activePathsCount / MIGRATION_ROUTES.length) * 100);

    const deforestationPenalty = canopyLoss * 0.8;
    const fragmentationPenalty = Math.max(0, fragmentationIndex - 5) * 0.35;
    const heatPenalty = Math.max(0, lstDelta + 0.8) * 8.0;
    const soilPenalty = soilRiskScore * 0.25;
    const connectivityPenalty = (100 - connectivityPct) * 0.35;

    const rawScore = 100 - (deforestationPenalty + fragmentationPenalty + heatPenalty + soilPenalty + connectivityPenalty);
    const score = Math.max(0, Math.min(100, Math.round(rawScore)));

    let grade = "F";
    let gradeColor = "risk";
    let gradeText = "Ecological Collapse";

    if (score >= 85) { grade = "A"; gradeColor = "good"; gradeText = "Regenerative Urban Design"; }
    else if (score >= 70) { grade = "B"; gradeColor = "good-subtle"; gradeText = "Sustainable Siting"; }
    else if (score >= 55) { grade = "C"; gradeColor = "watch"; gradeText = "Ecological Modification"; }
    else if (score >= 40) { grade = "D"; gradeColor = "watch-severe"; gradeText = "High Disruption Risk"; }

    return {
      canopyPct, bareSoilPct, builtUpPct, lstDelta, canopyLoss,
      fragmentationIndex, soilRiskScore, score, grade, gradeColor, gradeText,
      residentialCount, industrialCount, roadCount, forestCount, forestPatches,
      routesStatus, connectivityPct
    };
  }, [grid, policyGreenRoofs, policyWildlife, policyDrainage, initialStats]);

  function paintCell(id: number) {
    setActiveScenario(null);
    setGrid((prev) =>
      prev.map((cell) => {
        if (cell.id !== id) return cell;
        return { ...cell, type: selectedTool === "clear" ? cell.initialType : selectedTool };
      })
    );
  }

  function handleCellMouseDown(id: number) {
    setIsMouseDown(true);
    paintCell(id);
  }

  function handleCellMouseEnter(id: number) {
    if (isMouseDown) paintCell(id);
  }

  function handleMouseUp() { setIsMouseDown(false); }

  function loadScenario(scenarioName: ScenarioKey) {
    if (!scenarioName) return;
    setActiveScenario(scenarioName);
    setGrid((prev) => {
      const initialGrid = createNaturalGrid();

      if (scenarioName === "wilderness") {
        setPolicyWildlife(false);
        setPolicyGreenRoofs(false);
        setPolicyDrainage(false);
        return initialGrid;
      }

      if (scenarioName === "sprawl") {
        setPolicyWildlife(false);
        setPolicyGreenRoofs(false);
        setPolicyDrainage(false);
        return initialGrid.map((cell) => {
          if (cell.row === 4) return { ...cell, type: "green-highway" };
          if (cell.row >= 3 && cell.row <= 5 && cell.col >= 3 && cell.col <= 6) return { ...cell, type: "industrial" };
          if ((cell.row === 2 || cell.row === 6) && cell.col >= 2 && cell.col <= 7) return { ...cell, type: "residential" };
          return cell;
        });
      }

      if (scenarioName === "sustainable") {
        setPolicyWildlife(true);
        setPolicyGreenRoofs(true);
        setPolicyDrainage(true);
        return initialGrid.map((cell) => {
          if (cell.row === 9 && cell.col !== 0 && cell.col !== 9) return { ...cell, type: "green-highway" };
          if (cell.row === 8 && cell.col >= 2 && cell.col <= 7) return { ...cell, type: "residential" };
          if (cell.row === 8 && cell.col === 1) return { ...cell, type: "industrial" };
          return cell;
        });
      }

      if (scenarioName === "ecovillage") {
        setPolicyWildlife(true);
        setPolicyGreenRoofs(false);
        setPolicyDrainage(true);
        return initialGrid.map((cell) => {
          const isCornerGrass1 = cell.row <= 1 && cell.col <= 2 && cell.type === "grassland";
          const isCornerGrass2 = cell.row >= 8 && cell.col >= 7 && cell.col <= 9 && cell.type === "grassland";
          if (isCornerGrass1 || isCornerGrass2) return { ...cell, type: "residential" };
          return cell;
        });
      }

      return prev;
    });
  }

  function resetAll() {
    setGrid(createNaturalGrid());
    setActiveScenario("wilderness");
    setPolicyWildlife(false);
    setPolicyGreenRoofs(false);
    setPolicyDrainage(false);
  }

  return (
    <main className="page sandbox-page" onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>

      {/* ── Hero ── */}
      <section className="sandbox-hero">
        <div>
          <span className="badge">Zoning Sandbox</span>
          <h1>Simulate Eco-Planning & Urban Siting footprints.</h1>
          <p>
            Paint development blocks on the satellite grid. Our real-time simulator computes canopy loss,
            forest fragmentation (connected corridors), heat island LST delta, and runoff risk to grade your plan.
          </p>
        </div>

        {/* ── Improved Simulation Presets ── */}
        <div className="sandbox-presets-card">
          <div className="sandbox-presets-header">
            <Zap size={14} aria-hidden />
            <span>Simulation Presets</span>
            {activeScenario ? (
              <span className={`sandbox-active-badge variant-${activeScenario}`}>
                {SCENARIO_META[activeScenario].icon}
                <span>{SCENARIO_META[activeScenario].label} active</span>
              </span>
            ) : (
              <span className="sandbox-presets-hint">
                <HelpCircle size={11} />
                Hover & click a preset
              </span>
            )}
          </div>
          <div className="sandbox-presets-grid">
            {(Object.entries(SCENARIO_META) as [Exclude<ScenarioKey, null>, typeof SCENARIO_META[keyof typeof SCENARIO_META]][]).map(([key, meta]) => (
              <button
                key={key}
                type="button"
                id={`sandbox-preset-${key}`}
                className={`sandbox-preset-tile preset-${key} ${activeScenario === key ? "active" : ""}`}
                onClick={() => loadScenario(key)}
              >
                <span className="sandbox-preset-tip">Click to load {meta.label}</span>
                <div className="sandbox-preset-icon">{meta.icon}</div>
                <div className="sandbox-preset-copy">
                  <strong>{meta.label}</strong>
                  <span>{meta.desc}</span>
                </div>
                {activeScenario === key && (
                  <span className="sandbox-preset-check" aria-label="Active">
                    <CheckCircle2 size={13} />
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ── Main Grid Layout: Canvas left, Score+Indicators right ── */}
      <div className="sandbox-grid-layout">

        {/* ── Left: Canvas + Policies ── */}
        <div className="sandbox-canvas-col">
          <section className="surface sandbox-map-panel">
            <div className="surface-head">
              <div>
                <h2>Interactive Siting Canvas</h2>
                <p className="muted">Click and drag to paint zones. Clear to restore natural landscape.</p>
              </div>
              <div className="canvas-header-actions">
                <button
                  type="button"
                  className={`btn btn-toggle-paths ${showWildlifePaths ? "active" : ""}`}
                  onClick={() => setShowWildlifePaths(!showWildlifePaths)}
                >
                  <Compass size={14} /> {showWildlifePaths ? "Hide Pathways" : "Show Pathways"}
                </button>
                <button type="button" className="btn btn-reset" onClick={resetAll} aria-label="Reset canvas">
                  <RefreshCcw size={14} /> Reset
                </button>
              </div>
            </div>

            <div className="paint-palette">
              {Object.values(CELL_META).map((meta) => {
                const active = selectedTool === meta.type;
                return (
                  <button
                    type="button"
                    key={meta.type}
                    className={`palette-item ${active ? "active" : ""}`}
                    onClick={() => setSelectedTool(meta.type)}
                    style={{ "--border-color": meta.color } as React.CSSProperties}
                    title={meta.description}
                  >
                    <span className="palette-color" style={{ backgroundColor: meta.color }} />
                    <span className="palette-label">{meta.label}</span>
                  </button>
                );
              })}
              <button
                type="button"
                className={`palette-item erase-tool ${selectedTool === "clear" ? "active" : ""}`}
                onClick={() => setSelectedTool("clear")}
                title="Revert cells back to their initial natural vegetation state."
              >
                <Trash2 size={14} />
                <span className="palette-label">Erase / Clear</span>
              </button>
            </div>

            <div className="map-grid-container">
              <div className="grid-labels-row">
                {Array.from({ length: GRID_SIZE }).map((_, idx) => (
                  <span key={idx} className="grid-label">{String.fromCharCode(65 + idx)}</span>
                ))}
              </div>
              <div className="grid-main-row">
                <div className="grid-labels-col">
                  {Array.from({ length: GRID_SIZE }).map((_, idx) => (
                    <span key={idx} className="grid-label">{idx + 1}</span>
                  ))}
                </div>
                <div className="grid-svg-wrapper">
                  <svg viewBox="0 0 400 400" width="100%" height="100%" role="grid" aria-label="Planning Sandbox Grid">
                    {grid.map((cell) => {
                      const meta = CELL_META[cell.type];
                      const x = cell.col * 40;
                      const y = cell.row * 40;
                      let cellSymbol = null;
                      if (cell.type === "dense-forest") {
                        cellSymbol = <circle cx={x + 20} cy={y + 20} r="3" fill="rgba(255,255,255,0.25)" />;
                      } else if (cell.type === "water") {
                        cellSymbol = <line x1={x + 10} y1={y + 20} x2={x + 30} y2={y + 20} stroke="rgba(255,255,255,0.22)" strokeWidth="1.5" />;
                      } else if (cell.type === "residential") {
                        cellSymbol = <path d={`M ${x+15} ${y+25} L ${x+20} ${y+16} L ${x+25} ${y+25} Z`} fill="rgba(255,255,255,0.5)" />;
                      } else if (cell.type === "industrial") {
                        cellSymbol = <rect x={x+15} y={y+16} width="10" height="10" fill="rgba(255,255,255,0.4)" />;
                      } else if (cell.type === "green-highway") {
                        cellSymbol = <line x1={x+5} y1={y+20} x2={x+35} y2={y+20} stroke="#ffffff" strokeWidth="2.5" strokeDasharray="3 3" />;
                      }
                      return (
                        <g key={cell.id}>
                          <rect
                            x={x} y={y} width="38.5" height="38.5" rx="4"
                            fill={meta.color}
                            className={`sandbox-grid-rect ${meta.bgClass}`}
                            onMouseDown={() => handleCellMouseDown(cell.id)}
                            onMouseEnter={() => handleCellMouseEnter(cell.id)}
                          />
                          {cellSymbol}
                        </g>
                      );
                    })}
                    {showWildlifePaths && simulation.routesStatus.map((route) => {
                      const pathString = getSvgPathString(route.coords);
                      return (
                        <g key={route.id} className="wildlife-path-group">
                          <path d={pathString} fill="none" stroke={route.isBlocked ? "rgba(165, 62, 50, 0.32)" : "rgba(31, 122, 76, 0.35)"} strokeWidth="8" strokeLinecap="round" className={route.isBlocked ? "path-glow-blocked" : "path-glow-active"} />
                          <path d={pathString} fill="none" stroke={route.isBlocked ? "#a53e32" : "#54a36f"} strokeWidth="3.5" strokeLinecap="round" strokeDasharray={route.isBlocked ? "4 8" : "10 12"} className={`wildlife-path-core ${route.isBlocked ? "blocked" : "active"}`} />
                        </g>
                      );
                    })}
                  </svg>
                </div>
              </div>
            </div>

            <div className="grid-legend">
              <span><i className="legend-wilderness" /> Forest Core</span>
              <span><i className="legend-water" /> Waterway</span>
              <span><i className="legend-developed" /> Developed (Urban/Industrial)</span>
              <span><i className="legend-corridor" /> Green Infrastructure</span>
            </div>
          </section>

          {/* ── Mitigation Policies (moved under canvas) ── */}
          <section className="surface policy-toggles-under">
            <div className="surface-head">
              <div>
                <h2>Mitigation Policies</h2>
                <p className="muted">Enact regulations to buffer development impacts.</p>
              </div>
              <SlidersHorizontal size={16} className="text-muted" />
            </div>
            <div className="policy-toggles-grid">
              <label className="toggle-label">
                <input type="checkbox" checked={policyWildlife} onChange={(e) => setPolicyWildlife(e.target.checked)} />
                <div className="toggle-copy">
                  <strong>Wildlife Corridors / Underpasses</strong>
                  <span>Mitigates fragmentation score and clears highways for animal migration.</span>
                </div>
              </label>
              <label className="toggle-label">
                <input type="checkbox" checked={policyGreenRoofs} onChange={(e) => setPolicyGreenRoofs(e.target.checked)} />
                <div className="toggle-copy">
                  <strong>Urban Canopy / Green Roof Mandate</strong>
                  <span>Cools residential and industrial thermal LST delta.</span>
                </div>
              </label>
              <label className="toggle-label">
                <input type="checkbox" checked={policyDrainage} onChange={(e) => setPolicyDrainage(e.target.checked)} />
                <div className="toggle-copy">
                  <strong>Eco-Drainage & Bioswales</strong>
                  <span>Lowers erosion risks associated with soil disturbance.</span>
                </div>
              </label>
            </div>
          </section>
        </div>

        {/* ── Right Sidebar: Score + Indicators + Inventory ── */}
        <aside className="sandbox-sidebar flex-column">
          <section className="surface eco-scorecard flex-column align-center text-center">
            <span className="eyebrow">Eco-Planning Health</span>
            <div className={`score-badge-ring ${simulation.gradeColor}`}>
              <div className="score-ring-inner">
                <strong>{simulation.score}</strong>
                <span>Grade {simulation.grade}</span>
              </div>
            </div>
            <h3 className="score-grade-text">{simulation.gradeText}</h3>
            <p className="muted text-small">
              Derived from forest loss, habitat fragmentation, heat index, and wildlife connectivity.
            </p>
          </section>

          <section className="surface indicators-list flex-column gap-12">
            <h2>Simulation Indicators</h2>

            <article className="indicator-strip">
              <div className="indicator-summary">
                <div className="indicator-label-row"><Compass size={16} /><span>Wildlife Connectivity</span></div>
                <strong className={simulation.connectivityPct > 60 ? "text-success" : simulation.connectivityPct > 30 ? "text-warning" : "text-danger"}>{simulation.connectivityPct}%</strong>
              </div>
              <div className="progress-bar"><i className={`bar-fill ${simulation.connectivityPct > 60 ? "fill-success" : simulation.connectivityPct > 30 ? "fill-warning" : "fill-danger"}`} style={{ width: `${simulation.connectivityPct}%` }} /></div>
              <div className="routes-status-list">
                {simulation.routesStatus.map((route) => (
                  <div key={route.id} className="route-status-item">
                    <span className={`route-bullet ${route.isBlocked ? "blocked" : "active"}`} />
                    <span className="route-name text-small">{route.name.split(" (")[0]}</span>
                    <span className={`route-status-label ${route.isBlocked ? "text-danger" : "text-success"}`}>{route.isBlocked ? "Blocked" : "Clear"}</span>
                  </div>
                ))}
              </div>
            </article>

            <article className="indicator-strip">
              <div className="indicator-summary">
                <div className="indicator-label-row"><Trees size={16} /><span>Deforestation (Canopy Loss)</span></div>
                <strong className={simulation.canopyLoss > 20 ? "text-danger" : simulation.canopyLoss > 5 ? "text-warning" : "text-success"}>{simulation.canopyLoss.toFixed(1)}%</strong>
              </div>
              <div className="progress-bar"><i className={`bar-fill ${simulation.canopyLoss > 20 ? "fill-danger" : simulation.canopyLoss > 5 ? "fill-warning" : "fill-success"}`} style={{ width: `${simulation.canopyLoss}%` }} /></div>
              <p className="indicator-detail">Natural canopy dropped from {initialStats.canopy.toFixed(0)}% to {simulation.canopyPct.toFixed(0)}% total coverage.</p>
            </article>

            <article className="indicator-strip">
              <div className="indicator-summary">
                <div className="indicator-label-row"><Activity size={16} /><span>Forest Fragmentation</span></div>
                <strong className={simulation.fragmentationIndex > 45 ? "text-danger" : simulation.fragmentationIndex > 15 ? "text-warning" : "text-success"}>{simulation.fragmentationIndex}%</strong>
              </div>
              <div className="progress-bar"><i className={`bar-fill ${simulation.fragmentationIndex > 45 ? "fill-danger" : simulation.fragmentationIndex > 15 ? "fill-warning" : "fill-success"}`} style={{ width: `${simulation.fragmentationIndex}%` }} /></div>
              <p className="indicator-detail">Forest is split into <strong>{simulation.forestPatches} separate clusters</strong>. Corridors are {simulation.forestPatches > 1 ? "severed" : "connected"}.</p>
            </article>

            <article className="indicator-strip">
              <div className="indicator-summary">
                <div className="indicator-label-row"><Thermometer size={16} /><span>Heat Island (LST Delta)</span></div>
                <strong className={simulation.lstDelta > 1.2 ? "text-danger" : simulation.lstDelta > 0.1 ? "text-warning" : "text-success"}>{simulation.lstDelta >= 0 ? "+" : ""}{simulation.lstDelta.toFixed(2)}°C</strong>
              </div>
              <div className="progress-bar"><i className={`bar-fill ${simulation.lstDelta > 1.2 ? "fill-danger" : simulation.lstDelta > 0.1 ? "fill-warning" : "fill-success"}`} style={{ width: `${Math.max(0, Math.min(100, (simulation.lstDelta + 2) * 20))}%` }} /></div>
              <p className="indicator-detail">Average thermal delta relative to baseline natural landscape temperature.</p>
            </article>

            <article className="indicator-strip">
              <div className="indicator-summary">
                <div className="indicator-label-row"><Waves size={16} /><span>Erosion Risk (Exposed Soil)</span></div>
                <strong className={simulation.soilRiskScore > 40 ? "text-danger" : simulation.soilRiskScore > 15 ? "text-warning" : "text-success"}>{simulation.soilRiskScore.toFixed(0)}/100</strong>
              </div>
              <div className="progress-bar"><i className={`bar-fill ${simulation.soilRiskScore > 40 ? "fill-danger" : simulation.soilRiskScore > 15 ? "fill-warning" : "fill-success"}`} style={{ width: `${simulation.soilRiskScore}%` }} /></div>
              <p className="indicator-detail">Calculated runoff danger from bare ground ratio ({simulation.bareSoilPct.toFixed(0)}% exposed soil).</p>
            </article>
          </section>

          <section className="surface footprint-inventory">
            <h2>Placement Inventory</h2>
            <div className="inventory-grid">
              <div><span>Homes</span><strong>{simulation.residentialCount} blocks</strong></div>
              <div><span>Industry/Mining</span><strong>{simulation.industrialCount} blocks</strong></div>
              <div><span>Infrastructure</span><strong>{simulation.roadCount} blocks</strong></div>
              <div><span>Forest Sectors</span><strong>{simulation.forestCount} blocks</strong></div>
            </div>
          </section>
        </aside>
      </div>

      {/* ── Guidelines ── */}
      <section className="surface sandbox-guidelines">
        <div className="surface-head">
          <div>
            <h2>Sustainable Siting Guidelines</h2>
            <p className="muted">Use remote sensing insights to build high-scoring eco-plans.</p>
          </div>
          <Info size={20} className="text-muted" />
        </div>
        <div className="guidelines-grid">
          <article>
            <CheckCircle2 className="text-success" size={18} />
            <strong>A-Grade Siting Policy</strong>
            <p>Restrict built-up spaces to the outer edges. Slicing through the central woodland with highways increases forest fragmentation scores and drops the final health score.</p>
          </article>
          <article>
            <AlertTriangle className="text-warning" size={18} />
            <strong>Thermal Heat Abatement</strong>
            <p>Residential and industrial spaces increase local temperature up to +4.5°C. Enact the Green Roof Mandate and cluster residential zones near forest borders to cool the heat footprint.</p>
          </article>
          <article>
            <ShieldAlert className="text-danger" size={18} />
            <strong>Soil Preservation</strong>
            <p>Excess industrial excavation exposes bare soil. Enable bioswales and restrict mining footprints to avoid high erosion and heavy sediment runoff penalties.</p>
          </article>
        </div>
      </section>
    </main>
  );
}
