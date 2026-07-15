"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart3, FlaskConical, Layers3, Leaf, MapPinned, TreePine } from "lucide-react";

const LINKS = [
  { href: "/", label: "Timeline", icon: BarChart3 },
  { href: "/segmentation-lab", label: "Segmentation Lab", icon: Layers3 },
  { href: "/sandbox", label: "Planning Sandbox", icon: TreePine },
  { href: "/recommendation", label: "Recommendation", icon: MapPinned },
  { href: "/hypothesis", label: "Hypothesis", icon: FlaskConical }
];

export default function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="topbar">
      <div className="topbar-shell">
        <Link className="brand" href="/">
          <span className="brand-mark">
            <Leaf size={18} aria-hidden />
          </span>
          <span className="brand-copy">
            <strong>Urban Leaf Monitoring</strong>
            <small>Remote sensing planning workspace</small>
          </span>
        </Link>

        <nav className="nav">
          {LINKS.map((item) => {
            const Icon = item.icon;
            const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <Link key={item.href} href={item.href} className={active ? "active" : undefined}>
                <Icon size={16} aria-hidden />
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
