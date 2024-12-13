import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    cssCodeSplit: true,
  },
  plugins: [
    // Add this if you're not already using it
    {
      name: 'css-inline',
      transform(code, id) {
        if (id.endsWith('?inline')) {
          return {
            code: `export default ${JSON.stringify(code)}`,
            map: null
          }
        }
      }
    }
  ]
}) 