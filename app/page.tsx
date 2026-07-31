import TimelineDashboard from "@/components/timeline-dashboard";
import { FEATURE_KEYS, getRegions } from "@/lib/data";

export default async function HomePage() {
  const regions = await getRegions();
  return <TimelineDashboard regions={regions} featureKeys={FEATURE_KEYS} />;
}
