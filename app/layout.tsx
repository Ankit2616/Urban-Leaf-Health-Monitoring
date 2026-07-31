import type { Metadata } from "next";
import { Manrope } from "next/font/google";
import SiteHeader from "@/components/site-header";
import "./globals.css";

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-sans"
});

export const metadata: Metadata = {
  title: "Urban Leaf Monitoring",
  description: "Timeline-based vegetation health and land-cover change monitoring."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={manrope.variable}>
        <div className="desktop-only-blocker" role="alert" aria-live="assertive">
          <div className="desktop-only-blocker-card">
            <strong>Desktop View Required</strong>
            <p>
              Please open this website on a desktop or larger screen. Some features are only available in desktop view,
              and mobile screens are not interactive enough for this experience.
            </p>
          </div>
        </div>
        <SiteHeader />
        {children}
      </body>
    </html>
  );
}
