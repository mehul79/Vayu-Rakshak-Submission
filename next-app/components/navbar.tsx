"use client"
import Link from "next/link";
import { Expand } from "lucide-react";
import { Button } from "@/components/ui/button";
import { redirect } from "next/navigation";

export function Navbar() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        <Link href="/" className="flex items-center space-x-2">
          <Expand className="h-6 w-6 text-primary" />
          <span className="text-xl font-bold">Vayu Rakshak</span>
        </Link>
        <nav className="hidden md:flex items-center space-x-6 text-sm font-medium">
          <Link
            href="/#features"
            className="transition-colors hover:text-primary"
          >
            Features
          </Link>
          <Link href="/video" className="transition-colors hover:text-primary">
            Vayu Drishti
          </Link>
          <Link href="/demo" className="transition-colors hover:text-primary">
            Demo
          </Link>
          <Link
            href="/dashboard"
            className="transition-colors hover:text-primary"
          >
            Dashboard
          </Link>
          <Link href="/#presentation" className="transition-colors hover:text-primary">
            Presentation
          </Link>
        </nav>
        {/* <Button asChild>    
          <Link href="/demo">Vayu Drishti</Link>
        </Button> */}
      </div>
    </header>
  );
}
