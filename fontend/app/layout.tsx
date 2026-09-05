import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Forex Model Logs",
  description: "Dashboard for forex model run logs and forecast outputs"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
