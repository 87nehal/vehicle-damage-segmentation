import path from "node:path"
import { fileURLToPath } from "node:url"
import type { NextConfig } from "next"

const dir = path.dirname(fileURLToPath(import.meta.url))

const nextConfig: NextConfig = {
  turbopack: {
    root: dir,
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8001/api/:path*",
      },
    ]
  },
}

export default nextConfig
