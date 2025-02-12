import { NextResponse } from "next/server";
import axios from "axios";

const ML_SERVICE_URL = process.env.NEXT_PUBLIC_ML_SERVICE_URL;

export async function POST(request: Request) {
  try {
    console.log("[Backend] Received upload request");
    const formData = await request.formData();
    const file = formData.get("file");

    if (!file || !(file instanceof Blob)) {
      console.error("[Backend] No file provided");
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    // Create FormData for ML service
    const formDataForML = new FormData();
    formDataForML.append("file", file);

    // Forward to ML service
    console.log("[Backend] Forwarding request to ML service");
    const response = await axios.post(
      `${ML_SERVICE_URL}/predict`,
      formDataForML,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );

    const mlResponse = response.data;
    console.log("[Backend] ML service response:", mlResponse);

    // Get the original filename since ML service uses the same name
    const originalFile = file as File;
    const filename = originalFile.name;
    const videoUrl = `/api/video/${encodeURIComponent(filename)}`;
    console.log("[Backend] Generated video URL:", videoUrl);

    return NextResponse.json({
      ...mlResponse,
      videoUrl: videoUrl,
    });
  } catch (error) {
    console.error("[Backend] Error:", error);
    if (axios.isAxiosError(error)) {
      console.error("[Backend] Axios error details:", error.response?.data);
    }
    return NextResponse.json(
      { error: "Failed to process video" },
      { status: 500 }
    );
  }
}
