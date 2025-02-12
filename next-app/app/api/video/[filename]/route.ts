import { NextResponse } from "next/server";
import { createReadStream, statSync } from "fs";
import path from "path";

export async function GET(
  request: Request,
  { params }: { params: { filename: string } }
) {
  try {
    const filename = decodeURIComponent(params.filename);
    console.log("[Video API] Requested filename:", filename);

    const videoPath = path.join(
      process.env.NEXT_PUBLIC_VIDEO_STORAGE_PATH,
      filename
    );
    console.log("[Video API] Full video path:", videoPath);

    // Verify file exists and is readable
    try {
      const stat = statSync(videoPath);
      console.log("[Video API] File stats:", {
        size: stat.size,
        isFile: stat.isFile(),
        path: videoPath,
      });
    } catch (err) {
      console.error("[Video API] File stat error:", err);
      return NextResponse.json(
        { error: "Video file not found" },
        { status: 404 }
      );
    }

    const stat = statSync(videoPath);
    const fileSize = stat.size;

    // Get range from request header
    const range = request.headers.get("range");
    console.log("[Video API] Range header:", range);

    if (range) {
      const parts = range.replace(/bytes=/, "").split("-");
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
      const chunksize = end - start + 1;

      console.log("[Video API] Streaming range:", {
        start,
        end,
        chunksize,
        fileSize,
      });

      const stream = createReadStream(videoPath, { start, end });
      const headers = {
        "Content-Range": `bytes ${start}-${end}/${fileSize}`,
        "Accept-Ranges": "bytes",
        "Content-Length": chunksize.toString(),
        "Content-Type": "video/mp4",
      };

      return new NextResponse(stream as any, {
        status: 206,
        headers,
      });
    } else {
      console.log("[Video API] Streaming full file");
      const stream = createReadStream(videoPath);
      const headers = {
        "Content-Length": fileSize.toString(),
        "Content-Type": "video/mp4",
        "Accept-Ranges": "bytes",
      };

      return new NextResponse(stream as any, {
        status: 200,
        headers,
      });
    }
  } catch (error) {
    console.error("[Video API] Error:", error);
    return NextResponse.json(
      { error: "Failed to serve video" },
      { status: 500 }
    );
  }
}
