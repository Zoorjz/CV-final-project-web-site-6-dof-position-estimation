import { defineConfig } from 'vite';
import path from 'path';
import fs from 'fs';

/**
 * Scans data/renders and data/ for the latest timestamped renders directory (e.g. renders_YYYYMMDD_HHMMSS).
 * Priority:
 * 1. data/renders/renders_YYYYMMDD_HHMMSS (newest by name)
 * 2. data/renders (if it directly contains .mp4 files)
 * 3. data/renders_YYYYMMDD_HHMMSS (root data folder fallback)
 */
function resolveLatestRendersDirectory() {
  const dataDir = path.resolve(__dirname, 'data');
  const rendersDir = path.join(dataDir, 'renders');

  let candidates = [];

  // Check subdirectories in data/renders/
  if (fs.existsSync(rendersDir) && fs.statSync(rendersDir).isDirectory()) {
    const subEntries = fs.readdirSync(rendersDir)
      .filter(name => name.startsWith('renders_') && fs.statSync(path.join(rendersDir, name)).isDirectory())
      .map(name => ({ name, fullPath: path.join(rendersDir, name) }));
    candidates.push(...subEntries);

    // If data/renders itself has .mp4 files directly and no subdirs
    const directMp4s = fs.readdirSync(rendersDir).filter(f => f.endsWith('.mp4'));
    if (directMp4s.length > 0 && subEntries.length === 0) {
      candidates.push({ name: 'renders_direct', fullPath: rendersDir });
    }
  }

  // Check top-level data/renders_YYYYMMDD_HHMMSS
  if (fs.existsSync(dataDir)) {
    const topEntries = fs.readdirSync(dataDir)
      .filter(name => name.startsWith('renders_') && fs.statSync(path.join(dataDir, name)).isDirectory())
      .map(name => ({ name, fullPath: path.join(dataDir, name) }));
    candidates.push(...topEntries);
  }

  if (candidates.length === 0) return null;

  // Sort descending to get the newest timestamp
  candidates.sort((a, b) => b.name.localeCompare(a.name));
  return candidates[0].fullPath;
}

/**
 * Extracts metadata (frame count, duration, timestamp) from README.md in the render folder if available
 */
function getDatasetMetadata(dirPath) {
  if (!dirPath || !fs.existsSync(dirPath)) return null;
  const dirName = path.basename(dirPath);
  const readmePath = path.join(dirPath, 'README.md');

  let metadata = {
    activeDirectory: dirName,
    fullPath: dirPath,
    timestamp: dirName.replace(/^renders_/, ''),
    frameCount: 500,
    duration: 13.89,
    startFrame: 2200,
    fps: 35.997,
    files: fs.readdirSync(dirPath).filter(f => f.endsWith('.mp4'))
  };

  if (fs.existsSync(readmePath)) {
    try {
      const content = fs.readFileSync(readmePath, 'utf8');
      const startMatch = content.match(/\* \*\*Start Frame\*\*:\s*`?(\d+)`?/i);
      const countMatch = content.match(/\* \*\*Frame Count\*\*:\s*`?(\d+)`?/i);
      const tsMatch = content.match(/\*?Generation Timestamp\*?:\s*`?([0-9_]+)`?/i);
      const durationMatch = content.match(/Duration\s*\|\s*Frame Count.*?\|\s*([\d\.]+)\s*s/i);

      if (startMatch) metadata.startFrame = parseInt(startMatch[1], 10);
      if (countMatch) metadata.frameCount = parseInt(countMatch[1], 10);
      if (tsMatch) metadata.timestamp = tsMatch[1];
      if (durationMatch) metadata.duration = parseFloat(durationMatch[1]);
      if (metadata.frameCount && metadata.duration) {
        metadata.fps = metadata.frameCount / metadata.duration;
      }
    } catch (err) {
      console.warn('Error parsing dataset README.md:', err);
    }
  }

  return metadata;
}

// Custom plugin to serve dynamic renders directory with byte-range video streaming support
function serveDataDirectory() {
  return {
    name: 'serve-data-directory',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const decodedUrl = decodeURIComponent(req.url.split('?')[0]);

        // API Endpoint for dynamic dataset info
        if (decodedUrl === '/api/dataset-info') {
          const latestDir = resolveLatestRendersDirectory();
          const meta = getDatasetMetadata(latestDir);
          res.writeHead(200, {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache'
          });
          res.end(JSON.stringify(meta || { activeDirectory: 'none', files: [] }));
          return;
        }

        // Static video / data serving with latest timestamp resolution
        if (decodedUrl.startsWith('/data/')) {
          let relativePath = decodedUrl.slice(1); // remove leading slash
          let filePath = path.resolve(__dirname, relativePath);

          // If path is /data/renders/... resolve dynamically to the latest timestamped folder
          if (decodedUrl.startsWith('/data/renders/')) {
            const fileName = path.basename(decodedUrl);
            const latestDir = resolveLatestRendersDirectory();
            if (latestDir) {
              const candidatePath = path.join(latestDir, fileName);
              if (fs.existsSync(candidatePath)) {
                filePath = candidatePath;
              }
            }
          }

          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            const stat = fs.statSync(filePath);
            const fileSize = stat.size;
            const range = req.headers.range;

            const ext = path.extname(filePath).toLowerCase();
            const mimeTypes = {
              '.mp4': 'video/mp4',
              '.webm': 'video/webm',
              '.png': 'image/png',
              '.jpg': 'image/jpeg',
              '.jpeg': 'image/jpeg',
              '.json': 'application/json',
              '.csv': 'text/csv',
              '.md': 'text/markdown'
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
