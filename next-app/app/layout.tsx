import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/navbar";

export const metadata: Metadata = {
  title: process.env.NEXT_PUBLIC_APP_NAME,
  description:
    "This project is an AI-powered real-time video processing system designed for disaster response. It analyzes drone footage to detect people in earthquake, forest fire, and building collapse zones, providing real-time alerts with coordinates to authorities, enabling faster rescue operations and saving lives.",
  icons: {
    icon: "/flaticon.png",
    shortcut: "/flaticon.png",
    apple: "/flaticon.png",
    other: {
      rel: "icon",
      url: "/flaticon.png",
    },
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <div className="relative flex min-h-screen flex-col">
          <Navbar />
          {children}
        </div>
      </body>
    </html>
  );
}
