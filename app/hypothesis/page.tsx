import { ArrowDownRight, Flame, Layers3, MapPinned, Trees } from "lucide-react";

const evidence = [
  {
    icon: MapPinned,
    title: "Study Area",
    body: "Hasdeo Arand and adjacent control regions form the primary monitoring perimeter for deforestation and mining impact analysis."
  },
  {
    icon: Layers3,
    title: "Observation Stack",
    body: "Fusing RGB inspection imagery, multispectral Sentinel bands, monthly feature tables, and QGIS-labeled training sectors."
  },
  {
    icon: ArrowDownRight,
    title: "Expected Signal",
    body: "Vegetation indices rapidly decline during clearance windows, inversely mirroring spikes in bare-soil and built-up indicators."
  },
  {
    icon: Flame,
    title: "Event Focus",
    body: "The March–April 2022 disruption window serves as the baseline disturbance period for our comparative workflow."
  }
];

export default function HypothesisPage() {
  return (
    <main className="page hypothesis">
      <div className="hypothesis-kicker">
        <span className="badge">Phase 1 Hypothesis</span>
        <p className="muted">Methodological framework for monitoring, segmentation, and spatial planning.</p>
      </div>

      <section className="hypothesis-hero">
        <h1>Forest degradation is quantifiable through synchronized declines in vegetation indices and spikes in exposed-land signals.</h1>
      </section>

      <section className="hypothesis-grid">
        <article className="surface">
          <div className="surface-head">
            <div>
              <h2>Research Framing</h2>
              <p className="muted">Transforming the core hypothesis into an interactively auditable evidence trail.</p>
            </div>
            <div className="icon-wrapper">
              <Trees size={24} aria-hidden />
            </div>
          </div>
          <div className="surface-body">
            <p>
              The working hypothesis posits that major land-cover disruptions in Hasdeo manifest as measurable shifts
              across satellite-derived telemetry. Intact canopy ecosystems reliably sustain higher NDVI, EVI, SAVI, LAI, and GCI values.
              Conversely, disturbed or excavated sectors exhibit pronounced BSI, NDBI, and thermal signatures.
            </p>
            <p>
              This interactive dashboard tests that hypothesis in real-time: isolate a coordinate region, define a chronological before/after window,
              toggle feature indicators, and cross-reference nearest-neighbor curated imagery against the numeric timeline.
            </p>
          </div>
        </article>

        <aside className="surface">
          <div className="surface-head">
            <h2>Decision Matrix</h2>
          </div>
          <p className="muted" style={{ marginBottom: '16px' }}>
            A degradation event is validated when canopy features collapse concurrently with surges in surface exposure metrics.
          </p>
          <ul className="rule-list">
            <li>
              <span className="indicator drop"></span>
              <strong>Canopy Loss:</strong> Drops in NDVI, EVI, SAVI, LAI, or GCI.
            </li>
            <li>
              <span className="indicator spike"></span>
              <strong>Surface Exposure:</strong> Spikes in BSI or NDBI.
            </li>
            <li>
              <span className="indicator neutral"></span>
              <strong>Context Verification:</strong> Curated visual assets corroborate the telemetry.
            </li>
          </ul>
        </aside>
      </section>

      <section className="surface evidence-section">
        <div className="surface-head">
          <div>
            <h2>Evidence Architecture</h2>
            <p className="muted">The underlying components exposed to the application layer to guarantee auditable results.</p>
          </div>
        </div>
        <ul className="evidence-list">
          {evidence.map((item) => {
            const Icon = item.icon;
            return (
              <li key={item.title} className="evidence-card">
                <div className="status-line">
                  <div className="icon-box">
                    <Icon size={18} strokeWidth={2.5} aria-hidden />
                  </div>
                  <strong>{item.title}</strong>
                </div>
                <p className="muted">{item.body}</p>
              </li>
            );
          })}
        </ul>
      </section>
    </main>
  );
}