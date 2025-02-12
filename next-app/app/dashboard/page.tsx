"use client";

import { Card } from "@/components/ui/card";
import { MetricsCard } from "@/components/ui/metrics-card";
import { StatsChart } from "@/components/ui/stats-chart";
import { Users, Clock } from "lucide-react";
import { useEffect, useState } from "react";

const videoData = {
  message: "Video lol1.mp4 has been processed.",
  download_url: "/download/lol1.mp4",
  file_size: 4864585,
  statistics: {
    timestamp_counts: {
      "0.0": 50,
      "0.04": 36,
      "0.08": 51,
      "0.12": 54,
      "0.16": 50,
      "0.2": 64,
      "0.24": 56,
      "0.28": 47,
      "0.32": 43,
      "0.36": 48,
      "0.4": 54,
      "0.44": 51,
      "0.48": 48,
      "0.52": 48,
      "0.56": 65,
      "0.6": 62,
      "0.64": 70,
      "0.68": 54,
      "0.72": 45,
      "0.76": 42,
      "0.8": 44,
      "0.84": 52,
      "0.88": 48,
      "0.92": 49,
      "0.96": 48,
      "1.0": 57,
      "1.04": 54,
      "1.08": 41,
      "1.12": 49,
      "1.16": 48,
      "1.2": 45,
      "1.24": 51,
      "1.28": 40,
      "1.32": 47,
      "1.36": 51,
      "1.4": 56,
      "1.44": 62,
      "1.48": 66,
      "1.52": 59,
      "1.56": 57,
      "1.6": 67,
      "1.64": 59,
      "1.68": 52,
      "1.72": 51,
      "1.76": 46,
      "1.8": 43,
      "1.84": 53,
      "1.88": 52,
      "1.92": 60,
      "1.96": 60,
      "2.0": 58,
      "2.04": 8,
      "2.08": 7,
      "2.12": 6,
      "2.16": 7,
      "2.2": 5,
      "2.24": 9,
      "2.28": 8,
      "2.32": 7,
      "2.36": 7,
      "2.4": 9,
      "2.44": 8,
      "2.48": 9,
      "2.52": 4,
      "2.56": 7,
      "2.6": 6,
      "2.64": 4,
      "2.68": 4,
      "2.72": 3,
      "2.76": 5,
      "2.8": 8,
      "2.84": 6,
      "2.88": 6,
      "2.92": 1,
      "2.96": 2,
      "3.0": 1,
      "3.04": 1,
      "3.08": 4,
      "3.12": 4,
      "3.16": 3,
      "3.2": 5,
      "3.24": 3,
      "3.28": 4,
      "3.32": 5,
      "3.36": 7,
      "3.4": 5,
      "3.44": 3,
      "3.48": 4,
      "3.52": 5,
      "3.56": 5,
      "3.6": 3,
      "3.64": 5,
      "3.68": 3,
      "3.72": 2,
      "3.76": 1,
      "3.8": 4,
      "3.84": 4,
      "3.88": 2,
      "3.92": 4,
      "3.96": 2,
      "4.0": 4,
      "4.04": 5,
      "4.08": 5,
      "4.12": 5,
      "4.16": 4,
      "4.2": 1,
      "4.28": 2,
      "4.32": 4,
      "4.36": 3,
      "4.4": 2,
      "4.44": 1,
      "4.48": 3,
      "4.52": 5,
      "4.56": 5,
      "4.6": 3,
      "4.64": 3,
      "4.68": 4,
      "4.72": 4,
      "4.76": 6,
      "4.8": 6,
      "4.84": 4,
      "4.88": 5,
      "4.92": 5,
      "4.96": 7,
      "5.0": 6,
      "5.04": 9,
      "5.08": 10,
      "5.12": 9,
      "5.16": 7,
      "5.2": 6,
      "5.24": 5,
      "5.28": 9,
      "5.32": 8,
      "5.36": 5,
      "5.4": 5,
      "5.44": 7,
      "5.48": 7,
      "5.52": 8,
      "5.56": 3,
      "5.6": 4,
      "5.64": 5,
      "5.68": 3,
      "5.72": 3,
      "5.76": 3,
      "5.8": 2,
      "5.84": 2,
    },
    total_unique_people_estimate: 32,
    average_people_per_frame: 21.37,
    max_people_in_frame: 70,
    model_confidence: { average: 0.396, min: 0.25, max: 0.894 },
    video_duration: 5.88,
    frames_processed: 147,
  },
};

const chartData = Object.entries(videoData.statistics.timestamp_counts).map(
  ([timestamp, count]) => ({
    timestamp,
    count,
  })
);

export default function Page() {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      <main className="flex-1 bg-background relative">
        {isMounted && (
          <div
            className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:14px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]"
            aria-hidden="true"
          />
        )}

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-foreground">
              Video Analysis Dashboard
            </h1>
            <p className="text-muted-foreground mt-1">
              Analysis for video:{" "}
              {process.env.NEXT_PUBLIC_DEFAULT_VIDEO_FILENAME}
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-4">
            <MetricsCard
              title="Total Unique People"
              value={videoData.statistics.total_unique_people_estimate}
              icon={<Users className="h-4 w-4" />}
            />
            <MetricsCard
              title="Average People per Frame"
              value={videoData.statistics.average_people_per_frame.toFixed(2)}
              icon={<Users className="h-4 w-4" />}
            />
            <MetricsCard
              title="Max People in Frame"
              value={videoData.statistics.max_people_in_frame}
              icon={<Users className="h-4 w-4" />}
            />
            <MetricsCard
              title="Video Duration"
              value={`${videoData.statistics.video_duration.toFixed(2)}s`}
              icon={<Clock className="h-4 w-4" />}
            />
          </div>

          <Card className="mt-8 p-6">
            <div className="mb-4">
              <h2 className="text-xl font-semibold text-foreground">
                People Count Over Time
              </h2>
            </div>
            <StatsChart data={chartData} />
            <p className="text-center text-sm text-muted-foreground mt-4">
              Number of people detected at each timestamp in the video
            </p>
          </Card>

          <div className="grid gap-6 md:grid-cols-3 mt-8">
            <MetricsCard
              title="Model Confidence"
              value={videoData.statistics.model_confidence.average.toFixed(3)}
              subtitle={`Min: ${videoData.statistics.model_confidence.min.toFixed(
                3
              )}, Max: ${videoData.statistics.model_confidence.max.toFixed(3)}`}
            />
            <MetricsCard
              title="Frames Processed"
              value={videoData.statistics.frames_processed}
            />
            <MetricsCard
              title="File Size"
              value={`${(videoData.file_size / (1024 * 1024)).toFixed(2)} MB`}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
