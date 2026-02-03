/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        "ink": "#101315",
        "paper": "#f7f2e9",
        "amber": "#f1b24a",
        "teal": "#2f9e9d",
        "rose": "#d9624f",
        "night": "#0a0e12"
      },
      fontFamily: {
        display: ["'Bricolage Grotesque'", "sans-serif"],
        body: ["'Manrope'", "sans-serif"]
      },
      boxShadow: {
        glow: "0 10px 40px rgba(241, 178, 74, 0.25)",
        card: "0 20px 60px rgba(16, 19, 21, 0.12)"
      },
      keyframes: {
        floaty: {
          "0%,100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-12px)" }
        },
        sweep: {
          "0%": { backgroundPosition: "0% 50%" },
          "100%": { backgroundPosition: "100% 50%" }
        }
      },
      animation: {
        floaty: "floaty 8s ease-in-out infinite",
        sweep: "sweep 10s linear infinite"
      }
    }
  },
  plugins: []
};
