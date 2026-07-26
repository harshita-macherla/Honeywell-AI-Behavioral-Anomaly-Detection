import { riskColorVar } from "../../utils/format";

/**
 * RiskGauge
 * =========
 * Risk-score visualization (0-100) rendered as a semicircular instrument
 * dial rather than a generic donut/progress-bar -- a deliberate nod to
 * Honeywell's own heritage in physical gauges and industrial controls.
 * Zone ticks mark the exact thresholds risk_scoring_engine_v2.py's
 * risk_level_for() uses (35 / 60 / 80), so the dial's zones are literally
 * the backend's own decision boundaries, not a decorative approximation.
 */

const SIZE = 220;
const STROKE = 16;
const CX = SIZE / 2;
const CY = SIZE / 2 + 10;
const R = SIZE / 2 - STROKE;
const START_ANGLE = 180; // degrees, left
const END_ANGLE = 0; // degrees, right
const ZONES = [
  { from: 0, to: 35, color: "var(--risk-low)" },
  { from: 35, to: 60, color: "var(--risk-medium)" },
  { from: 60, to: 80, color: "var(--risk-high)" },
  { from: 80, to: 100, color: "var(--risk-critical)" },
];

function polarToCartesian(cx, cy, r, angleDeg) {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy - r * Math.sin(rad) };
}

function describeArc(cx, cy, r, startScore, endScore) {
  const startAngle = START_ANGLE - (startScore / 100) * (START_ANGLE - END_ANGLE);
  const endAngle = START_ANGLE - (endScore / 100) * (START_ANGLE - END_ANGLE);
  const start = polarToCartesian(cx, cy, r, startAngle);
  const end = polarToCartesian(cx, cy, r, endAngle);
  const largeArcFlag = Math.abs(startAngle - endAngle) <= 180 ? 0 : 1;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`;
}

export default function RiskGauge({ score, level, size = "md" }) {
  const clamped = Math.max(0, Math.min(100, score ?? 0));
  const scale = size === "sm" ? 0.62 : 1;
  const color = riskColorVar(level);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "var(--sp-2)" }}>
      <svg
        width={SIZE * scale}
        height={(CY + STROKE) * scale}
        viewBox={`0 0 ${SIZE} ${CY + STROKE}`}
        role="img"
        aria-label={`Risk score ${clamped} out of 100, ${level}`}
      >
        {/* Track: faint zone bands */}
        {ZONES.map((z) => (
          <path
            key={z.color}
            d={describeArc(CX, CY, R, z.from, z.to)}
            fill="none"
            stroke={z.color}
            strokeOpacity="0.18"
            strokeWidth={STROKE}
            strokeLinecap="butt"
          />
        ))}
        {/* Threshold ticks at 35 / 60 / 80 */}
        {[35, 60, 80].map((t) => {
          const angle = START_ANGLE - (t / 100) * 180;
          const outer = polarToCartesian(CX, CY, R + STROKE / 2 + 3, angle);
          const inner = polarToCartesian(CX, CY, R - STROKE / 2 - 3, angle);
          return (
            <line
              key={t}
              x1={inner.x}
              y1={inner.y}
              x2={outer.x}
              y2={outer.y}
              stroke="var(--bg-app)"
              strokeWidth="2"
            />
          );
        })}
        {/* Active fill up to the current score */}
        <path
          d={describeArc(CX, CY, R, 0, clamped)}
          fill="none"
          stroke={color}
          strokeWidth={STROKE}
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 6px ${color})` }}
        />
        <text
          x={CX}
          y={CY - 14}
          textAnchor="middle"
          className="mono"
          fontSize="34"
          fontWeight="600"
          fill="var(--text-primary)"
        >
          {clamped.toFixed(0)}
        </text>
        <text x={CX} y={CY + 8} textAnchor="middle" fontSize="11" letterSpacing="0.06em" fill="var(--text-tertiary)">
          RISK SCORE / 100
        </text>
      </svg>
      {size !== "sm" && (
        <span className="eyebrow" style={{ color }}>
          {level} tier
        </span>
      )}
    </div>
  );
}
