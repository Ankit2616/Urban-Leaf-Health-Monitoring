import RecommendationDashboard from "@/components/recommendation-dashboard";
import { getHasdeoRecommendationData } from "@/lib/recommendation";

export default async function RecommendationPage() {
  const data = await getHasdeoRecommendationData();
  return <RecommendationDashboard data={data} />;
}
