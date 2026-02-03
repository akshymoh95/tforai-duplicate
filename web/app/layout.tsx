import "./globals.css";
import { Bricolage_Grotesque, Manrope } from "next/font/google";
import Providers from "./providers";

const display = Bricolage_Grotesque({
  subsets: ["latin"],
  variable: "--font-display",
});

const body = Manrope({
  subsets: ["latin"],
  variable: "--font-body",
});

export const metadata = {
  title: "AI Insights Studio",
  description: "Narrative analytics for RM performance, mandate profitability, and sentiment.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable}`}>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
