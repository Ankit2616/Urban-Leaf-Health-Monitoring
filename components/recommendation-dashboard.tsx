"use client";

import { useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, Compass, MapPinned, ShieldAlert, Trees } from "lucide-react";
import type { RecommendationData, RecommendationZone } from "@/lib/recommendation";

function zoneFill(category: RecommendationZone["category"]) {
  if (category === "Preferred zone") return "rgba(31, 122, 76, 0.78)";
  if (category === "Conditional") return "rgba(211, 145, 50, 0.78)";
  return "rgba(165, 62, 50, 0.82)";
}

function zoneStroke(category: RecommendationZone["category"]) {
  if (category === "Preferred zone") return "#1f7a4c";
  if (category === "Conditional") return "#b77722";
  return "#a53e32";
}

function iconFor(category: RecommendationZone["category"]) {
  if (category === "Preferred zone") return CheckCircle2;
  if (category === "Conditional") return AlertTriangle;
  return ShieldAlert;
}

function project(bounds: [number, number, number, number], full: [number, number, number, number]) {
  const [minLon, minLat, maxLon, maxLat] = bounds;
  const [fullMinLon, fullMinLat, fullMaxLon, fullMaxLat] = full;
  const width = fullMaxLon - fullMinLon;
  const height = fullMaxLat - fullMinLat;
  return {
    x: ((minLon - fullMinLon) / width) * 100,
    y: ((fullMaxLat - maxLat) / height) * 100,
    width: ((maxLon - minLon) / width) * 100,
    height: ((maxLat - minLat) / height) * 100
  };
}

export default function RecommendationDashboard({ data }: { data: RecommendationData }) {
  const [selectedId, setSelectedId] = useState(data.recommendation.primaryZoneId);
  const selected = data.zones.find((zone) => zone.id === selectedId) ?? data.zones[0];
  const fullBounds = data.studyArea.bounds;
  
  const plottedZones = useMemo(
    () => data.zones.map((zone) => ({ zone, rect: project(zone.bounds, fullBounds) })),
    [data.zones, fullBounds]
  );

  return (
    <main className="page recommendation-page">
      <section className="recommendation-hero">
        <div className="hero-content">
          <span className="badge glass-badge">Decision Support</span>
          <h1>Optimized Spatial Planning & Ecological Sensitivity Matrix</h1>
          <p>
            This operational layer ranks peripheral Hasdeo zones by environmental fragility, 
            core proximity, and evidence confidence to dictate where development pressure carries the lowest ecological tax.
          </p>
        </div>
        
        {/* Restored Original Right-Side Block */}
        <div className="recommendation-headline">
          <div className="recommendation-headline-img" />
          <div className="recommendation-headline-body">
            <span>Strategic Directive</span>
            <strong>{data.recommendation.headline}</strong>
            <p>{data.recommendation.summary}</p>
          </div>
        </div>
      </section>

      <section className="recommendation-kpis">
        <article className="stat stat-card">
          <div className="stat-icon-wrapper forest"><Trees size={16} aria-hidden /></div>
          <div className="stat-content">
            <span>Disturbance Signal</span>
            <strong>{Math.abs(data.context.ndviDropPct).toFixed(1)}%</strong>
            <small>Canopy collapse in the March-April 2022 window.</small>
          </div>
        </article>
        <article className="stat stat-card">
          <div className="stat-icon-wrapper amber"><ShieldAlert size={16} aria-hidden /></div>
          <div className="stat-content">
            <span>Surface Exposure</span>
            <strong>+{data.context.bsiRisePct.toFixed(1)}%</strong>
            <small>BSI surge against the pre-event baseline.</small>
          </div>
        </article>
        <article className="stat stat-card">
          <div className="stat-icon-wrapper danger"><Compass size={16} aria-hidden /></div>
          <div className="stat-content">
            <span>Thermal Pressure</span>
            <strong>+{data.context.heatRisePct.toFixed(1)}%</strong>
            <small>Radiometric heat rise mapped to the disruption sector.</small>
          </div>
        </article>
      </section>

      <div className="recommendation-grid">
        <section className="surface recommendation-map-surface">
          <div className="surface-head">
            <div>
              <h2>Hasdeo Topology Map</h2>
              <p className="muted">Select a perimeter zone to audit its scoring telemetry.</p>
            </div>
            <span className="badge outline-badge">{selected.label}</span>
          </div>

          <div className="zone-map">
            <svg viewBox="0 0 100 100" role="img" aria-label="Hasdeo recommendation map">
              <rect x="0" y="0" width="100" height="100" rx="5" fill="#f4f7f4" stroke="#d3ded5" />
              {plottedZones.map(({ zone, rect }) => (
                <g 
                  key={zone.id} 
                  onClick={() => setSelectedId(zone.id)} 
                  className={`map-zone-group ${selected.id === zone.id ? 'active' : ''}`}
                >
                  <rect
                    x={rect.x}
                    y={rect.y}
                    width={rect.width}
                    height={rect.height}
                    rx="2"
                    fill={zoneFill(zone.category)}
                    stroke={selected.id === zone.id ? "#07130f" : zoneStroke(zone.category)}
                    strokeWidth={selected.id === zone.id ? 1.5 : 0.7}
                    className="zone-rect"
                  />
                  <text x={rect.x + rect.width / 2} y={rect.y + rect.height / 2} textAnchor="middle" dominantBaseline="middle">
                    {zone.label.replace("Hasdeo ", "")}
                  </text>
                </g>
              ))}
            </svg>
          </div>

          <div className="zone-legend">
            <span className="legend-item"><i className="good" /> Preferred Zone</span>
            <span className="legend-item"><i className="watch" /> Conditional</span>
            <span className="legend-item"><i className="risk" /> Avoid</span>
          </div>
        </section>

        <aside className="surface recommendation-detail">
          <div className="surface-head">
            <div>
              <h2>{selected.label}</h2>
              <p className="muted">{selected.rationale}</p>
            </div>
          </div>

          <div className={`status-badge ${selected.category.toLowerCase()}`}>
            {selected.category}
          </div>
          
          <div className="recommendation-metric-grid">
            <div className="metric-box">
              <span>Suitability</span>
              <strong>{selected.suitabilityScore.toFixed(0)}</strong>
            </div>
            <div className="metric-box">
              <span>Risk</span>
              <strong>{selected.environmentalRisk.toFixed(0)}</strong>
            </div>
            <div className="metric-box">
              <span>Core Overlap</span>
              <strong>{selected.coreOverlapPct.toFixed(1)}%</strong>
            </div>
            <div className="metric-box">
              <span>Periphery</span>
              <strong>{selected.peripheralAccessPct.toFixed(1)}%</strong>
            </div>
          </div>

          <div className="score-breakdown">
            <h3>Component Weighting</h3>
            <div className="score-row">
              <span>Core Sensitivity</span>
              <div className="progress-track"><i style={{ width: `${selected.coreOverlapPct}%` }} /></div>
            </div>
            <div className="score-row">
              <span>Centrality</span>
              <div className="progress-track"><i style={{ width: `${selected.centralityPct}%` }} /></div>
            </div>
            <div className="score-row">
              <span>Periphery Advantage</span>
              <div className="progress-track"><i style={{ width: `${selected.peripheralAccessPct}%` }} /></div>
            </div>
            <div className="score-row">
              <span>Evidence Confidence</span>
              <div className="progress-track"><i style={{ width: `${selected.evidenceScore}%` }} /></div>
            </div>
          </div>

          <ul className="evidence-list audit-list">
            {selected.facts.map((fact) => (
              <li key={fact}>{fact}</li>
            ))}
          </ul>
        </aside>
      </div>

      <section className="surface rank-section">
        <div className="surface-head">
          <div>
            <h2>Sector Ranking Matrix</h2>
            <p className="muted">Higher baseline scores correlate to reduced ecological disruption under the current planning heuristic.</p>
          </div>
        </div>
        <div className="recommendation-rank-grid">
          {data.zones.map((zone, index) => {
            const Icon = iconFor(zone.category);
            return (
              <button
                type="button"
                key={zone.id}
                className={`rank-card ${selected.id === zone.id ? "active" : ""}`}
                onClick={() => setSelectedId(zone.id)}
              >
                <div className="rank-header">
                  <span className="rank-number">#{index + 1}</span>
                  <div className="rank-icon"><Icon size={18} aria-hidden /></div>
                </div>
                <strong>{zone.label}</strong>
                <p>{zone.category}</p>
                <em>{zone.suitabilityScore.toFixed(0)}/100 Suitability Score</em>
              </button>
            );
          })}
        </div>
      </section>

      <section className="surface">
        <div className="surface-head">
          <div>
            <h2>Methodology & Cautions</h2>
            <p className="muted">This layer is engineered for preliminary spatial filtering, not as a replacement for formal ground-truth clearance.</p>
          </div>
          <div className="icon-wrapper">
            <MapPinned size={20} aria-hidden />
          </div>
        </div>
        <div className="recommendation-method-grid">
          {data.methodology.map((item) => (
            <article key={item.label} className="method-card">
              <span className="weight-badge">{Math.round(item.weight * 100)}% Weight</span>
              <strong>{item.label}</strong>
              <p>{item.description}</p>
            </article>
          ))}
        </div>
        <div className="recommendation-note">
          <ShieldAlert size={16} />
          <span><strong>Current Evidence Note:</strong> {data.context.disturbanceWindow} {data.context.recoverySignal}</span>
        </div>
      </section>
    </main>
  );
}