import { Link } from "react-router-dom";
import { Radar } from "lucide-react";

export default function NotFoundPage() {
  return (
    <div className="state-block" style={{ minHeight: "60vh" }}>
      <Radar size={32} className="state-block__icon" />
      <div>
        <p style={{ fontWeight: 600, fontSize: "var(--fs-lg)", marginBottom: 4 }}>Nothing on the scope</p>
        <p style={{ color: "var(--text-secondary)" }}>This page doesn&rsquo;t exist, or the entity/alert ID wasn&rsquo;t found.</p>
      </div>
      <Link to="/" className="btn btn--primary">
        Back to overview
      </Link>
    </div>
  );
}
