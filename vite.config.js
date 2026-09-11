import { defineConfig } from 'vite';
import path from 'path';
import fs from 'fs';

// Helper to find the latest renders directory in data/
function getLatestRendersDirectory() {
  const dataDir = path.resolve(__dirname, 'data');
  if (!fs.existsSync(dataDir)) return null;

  // Check if data/renders exists
  const standardRenders = path.resolve(dataDir, 'renders');
  if (fs.existsSync(standardRenders) && fs.statSync(standardRenders).isDirectory()) {
    return standardRenders;
  }

  // Otherwise find latest timestamped renders_YYYYMMDD_HHMMSS
  const entries = fs.readdirSync(dataDir)
    .filter(name => name.startsWith('renders_') && fs.statSync(path.join(dataDir, name)).isDirectory())
    .sort()
    .reverse();

  return entries.length > 0 ? path.join(dataDir, entries[0]) : null;
}

// Custom plugin to serve the /data directory with range request support for smooth video streaming & seeking
function serveDataDirectory() {
  return {
    name: 'serve-data-directory',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const decodedUrl = decodeURIComponent(req.url.split('?')[0]);
        if (decodedUrl.startsWith('/data/')) {
          let relativePath = decodedUrl.slice(1); // remove leading slash
          let filePath = path.resolve(__dirname, relativePath);

          // If requested /data/renders/... but data/renders doesn't have the file, check latest renders_*
          if (!fs.existsSync(filePath) && decodedUrl.startsWith('/data/renders/')) {
            const fileName = path.basename(decodedUrl);
            const latestDir = getLatestRendersDirectory();
            if (latestDir) {
              const fallbackPath = path.join(latestDir, fileName);
              if (fs.existsSync(fallbackPath)) {
                filePath = fallbackPath;
              }
            }
          }

          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            const stat = fs.statSync(filePath);
            const fileSize = stat.size;
            const range = req.headers.range;

            // Set content type
            const ext = path.extname(filePath).toLowerCase();
            const mimeTypes = {
              '.mp4': 'video/mp4',
              '.webm': 'video/webm',
              '.png': 'image/png',
              '.jpg': 'image/jpeg',
              '.jpeg': 'image/jpeg',
              '.json': 'application/json',
              '.csv': 'text/csv'
            };
            const contentType = mimeTypes[ext] || 'application/octet-stream';

            if (range) {
              const parts = range.replace(/bytes=/, '').split('-');
              const start = parseInt(parts[0], 10);
              const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
              const chunksize = (end - start) + 1;
              const file = fs.createReadStream(filePath, { start, end });
              const head = {
                'Content-Range': `bytes ${start}-${end}/${fileSize}`,
                'Accept-Ranges': 'bytes',
                'Content-Length': chunksize,
                'Content-Type': contentType,
              };
              res.writeHead(206, head);
              file.pipe(res);
            } else {
              const head = {
                'Content-Length': fileSize,
                'Content-Type': contentType,
                'Accept-Ranges': 'bytes',
              };
              res.writeHead(200, head);
              fs.createReadStream(filePath).pipe(res);
            }
            return;
          }
        }
        next();
      });
    }
  };
}

export default defineConfig({
  plugins: [serveDataDirectory()],
  server: {
    port: 5173,
    open: false,
    fs: {
      allow: ['..']
    }
  }
});
