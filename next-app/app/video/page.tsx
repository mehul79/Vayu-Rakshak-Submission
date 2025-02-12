"use client";

import { useState, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";

export default function VideoDetectionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      "video/mp4": [".mp4"],
      "video/quicktime": [".mov"],
      "video/x-msvideo": [".avi"],
      "video/webm": [".webm"],
    },
    multiple: false,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles && acceptedFiles.length > 0) {
        const uploadedFile = acceptedFiles[0];
        console.log("File uploaded:", uploadedFile);
        setFile(uploadedFile);
        const objectUrl = URL.createObjectURL(uploadedFile);
        setPreviewUrl(objectUrl);
        setError(null);
      }
    },
    noClick: !!file,
  });

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleSubmit = async () => {
    if (!file) {
      setError("Please upload a video first.");
      return;
    }

    console.log("[Frontend] Starting video processing...");
    setIsLoading(true);
    setError(null);
    setProcessedVideoUrl(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      console.log("[Frontend] Sending request to /api/predict...");
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Video processing failed: ${response.statusText}`);
      }

      const data = await response.json();
      console.log("[Frontend] Processing response:", data);

      if (data.videoUrl) {
        console.log("[Frontend] Processed video URL:", data.videoUrl);
        setProcessedVideoUrl(data.videoUrl);
      } else {
        throw new Error("No download URL received in response");
      }
    } catch (err) {
      console.error("[Frontend] Error:", err);
      setError("An error occurred while processing the video.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="relative">
        {/* Gradient Background */}
        <div className="absolute inset-0 bg-gradient-to-b from-background via-background/90 to-background">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:14px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />
        </div>

        {/* Main Content */}
        <div className="container mx-auto px-4 py-8 relative">
          <h1 className="text-3xl font-bold mb-8 text-center text-primary">
            Video Detection Platform
          </h1>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Left Column - Upload and Preview */}
            <div className="space-y-4">
              {previewUrl && (
                <div className="mb-4">
                  <video
                    src={previewUrl}
                    controls
                    className="w-full h-auto rounded-lg"
                  />
                </div>
              )}
              {file ? (
                <p className="text-muted-foreground mb-4">
                  Selected file: {file.name}
                </p>
              ) : (
                <div
                  {...getRootProps()}
                  className="border-2 border-dashed border-primary rounded-lg p-8 text-center transition-colors cursor-pointer hover:bg-primary/10"
                >
                  <input {...getInputProps()} />
                  {isDragActive ? (
                    <p>Drop the video here ...</p>
                  ) : (
                    <p>Drag and drop a video here, or click to select a file</p>
                  )}
                </div>
              )}
              <Button
                onClick={handleSubmit}
                disabled={!file || isLoading}
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  "Process Video"
                )}
              </Button>
              {error && <p className="text-destructive text-sm">{error}</p>}
            </div>

            {/* Right Column - Processed Video Preview */}
            <div className="relative">
              {/* Gradient Overlay */}
              <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-primary/10 rounded-2xl blur-3xl" />
              {/* Processed Video Container */}
              <div className="relative bg-card rounded-2xl border p-6 shadow-2xl min-h-[300px] flex items-center justify-center">
                {processedVideoUrl ? (
                  <video
                    controls
                    className="w-full h-auto"
                    onError={(e) => {
                      console.error("Video loading error:", e);
                      setError("Error loading processed video");
                    }}
                  >
                    <source src={processedVideoUrl} type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>
                ) : (
                  <p className="text-muted-foreground">
                    Processed video will appear here
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
