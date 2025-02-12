/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: ['localhost'],
  },
  async rewrites() {
    return [
      {
        source: '/ml-api/:path*',
        destination: 'http://localhost:8080/:path*',
      },
    ]
  }
}

module.exports = nextConfig