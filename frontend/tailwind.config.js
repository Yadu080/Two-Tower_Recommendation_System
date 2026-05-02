/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        nf: {
          red:    '#E50914',
          redhov: '#B81D24',
          bg:     '#141414',
          card:   '#2F2F2F',
          gray:   '#808080',
        },
      },
      fontFamily: {
        sans: ['"Netflix Sans"', 'Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
