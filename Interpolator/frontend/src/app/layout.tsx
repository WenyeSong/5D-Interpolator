import "./styles.css";

export const metadata = {
    title: "5D Interpolator",
    description: "Frontend for interpolator system",
  };
  
  export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
      <html lang="en">
        <body>{children}</body>
      </html>
    );
  }
  