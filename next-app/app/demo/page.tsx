"use client";
import { useEffect, useRef, useState } from "react";
import styles from "./demo.module.css";

export default function DemoPage() {
  const [position, setPosition] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  const handleMove = (event: MouseEvent | TouchEvent) => {
    if (!isDragging.current || !containerRef.current) return;

    const containerRect = containerRef.current.getBoundingClientRect();
    const clientX =
      "touches" in event ? event.touches[0].clientX : event.clientX;

    const x = clientX - containerRect.left;
    const newPosition = (x / containerRect.width) * 100;

    setPosition(Math.min(100, Math.max(0, newPosition)));
  };

  const startDragging = () => {
    isDragging.current = true;
  };

  const stopDragging = () => {
    isDragging.current = false;
  };

  useEffect(() => {
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", stopDragging);
    window.addEventListener("touchmove", handleMove);
    window.addEventListener("touchend", stopDragging);

    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", stopDragging);
      window.removeEventListener("touchmove", handleMove);
      window.removeEventListener("touchend", stopDragging);
    };
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:14px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />

      <h1 className="text-3xl font-bold text-center mb-8 relative">
        People Detection Demo
      </h1>

      <div
        ref={containerRef}
        className="relative w-[800px] aspect-video rounded-lg mx-auto overflow-hidden"
      >
        {/* Before Video */}
        <div
          className="absolute top-0 left-0 w-full h-full"
          style={{
            clipPath: `inset(0 ${100 - position}% 0 0)`,
          }}
        >
          <video
            className="absolute top-0 left-0 w-full h-full object-cover"
            autoPlay
            loop
            muted
            playsInline
          >
            <source
              src={process.env.NEXT_PUBLIC_DEMO_BEFORE_VIDEO}
              type="video/mp4"
            />
            Your browser does not support the video tag.
          </video>
        </div>

        {/* After Video */}
        <div
          className="absolute top-0 left-0 w-full h-full"
          style={{
            clipPath: `inset(0 0 0 ${position}%)`,
          }}
        >
          <video
            className="absolute top-0 left-0 w-full h-full object-cover"
            autoPlay
            loop
            muted
            playsInline
          >
            <source
              src={process.env.NEXT_PUBLIC_DEMO_AFTER_VIDEO}
              type="video/mp4"
            />
            Your browser does not support the video tag.
          </video>
        </div>

        {/* Divider Line */}
        <div
          className="absolute top-0 bottom-0 w-1 bg-white cursor-ew-resize"
          style={{
            left: `${position}%`,
            transform: "translateX(-50%)",
          }}
          onMouseDown={startDragging}
          onTouchStart={startDragging}
        >
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full flex items-center justify-center shadow-lg">
            <svg
              className="w-4 h-4 text-gray-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 7h8M8 12h8M8 17h8"
              />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}
