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
  const geomDir = path.join(rendersDir, 'geometry-based');
  const dataDrivenDir = path.join(rendersDir, 'data-driven');

  let candidates = [];

  // Check subdirectories in data/renders/geometry-based/
  if (fs.existsSync(geomDir) && fs.statSync(geomDir).isDirectory()) {
    const geomEntries = fs.readdirSync(geomDir)
      .filter(name => name.startsWith('renders_') && fs.statSync(path.join(geomDir, name)).isDirectory())
      .map(name => ({ name, fullPath: path.join(geomDir, name) }));
    candidates.push(...geomEntries);
  }

  // Check subdirectories in data/renders/data-driven/
  if (fs.existsSync(dataDrivenDir) && fs.statSync(dataDrivenDir).isDirectory()) {
    const ddEntries = fs.readdirSync(dataDrivenDir)
      .filter(name => name.startsWith('renders_') && fs.statSync(path.join(dataDrivenDir, name)).isDirectory())
      .map(name => ({ name, fullPath: path.join(dataDrivenDir, name) }));
    candidates.push(...ddEntries);
  }

  // Check subdirectories in data/renders/ (legacy)
  if (fs.existsSync(rendersDir) && fs.statSync(rendersDir).isDirectory()) {
    const subEntries = fs.readdirSync(rendersDir)
      .filter(name => name.startsWith('renders_') && fs.statSync(path.join(rendersDir, name)).isDirectory())
      .map(name => ({ name, fullPath: path.join(rendersDir, name) }));
    candidates.push(...subEntries);
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

function findRenderFileInCandidates(fileName) {
  const dataDir = path.resolve(__dirname, 'data');
  const rendersDir = path.join(dataDir, 'renders');
  const geomDir = path.join(rendersDir, 'geometry-based');
  const dataDrivenDir = path.join(rendersDir, 'data-driven');

  const searchDirs = [];
  if (fs.existsSync(geomDir)) {
    const gSubs = fs.readdirSync(geomDir)
      .filter(n => n.startsWith('renders_') && fs.statSync(path.join(geomDir, n)).isDirectory())
      .map(n => path.join(geomDir, n));
    searchDirs.push(...gSubs);
  }
  if (fs.existsSync(dataDrivenDir)) {
    const ddSubs = fs.readdirSync(dataDrivenDir)
      .filter(n => n.startsWith('renders_') && fs.statSync(path.join(dataDrivenDir, n)).isDirectory())
      .map(n => path.join(dataDrivenDir, n));
    searchDirs.push(...ddSubs);
  }
  if (fs.existsSync(rendersDir)) {
    const subs = fs.readdirSync(rendersDir)
      .filter(n => n.startsWith('renders_') && fs.statSync(path.join(rendersDir, n)).isDirectory())
      .map(n => path.join(rendersDir, n));
    searchDirs.push(...subs);
  }

  searchDirs.sort((a, b) => path.basename(b).localeCompare(path.basename(a)));

  for (const dir of searchDirs) {
    const p = path.join(dir, fileName);
    if (fs.existsSync(p) && fs.statSync(p).isFile()) {
      return p;
    }
  }
  return null;
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
            const foundPath = findRenderFileInCandidates(fileName);
            if (foundPath) {
              filePath = foundPath;
            } else {
              const latestDir = resolveLatestRendersDirectory();
              if (latestDir) {
                const candidatePath = path.join(latestDir, fileName);
                if (fs.existsSync(candidatePath)) {
                  filePath = candidatePath;
                }
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
              '.gif': 'image/gif',
              '.json': 'application/json',
              '.csv': 'text/csv',
              '.md': 'text/markdown'
            };
            const contentType = mimeTypes[ext] || 'application/octet-stream';

            if (range) {
              const parts = range.replace(/bytes=/, '').split('-');
              let start = parseInt(parts[0], 10);
              let end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;

              if (isNaN(start) || start < 0) start = 0;
              if (isNaN(end) || end >= fileSize) end = fileSize - 1;

              if (start >= fileSize || start > end) {
                res.writeHead(416, {
                  'Content-Range': `bytes */${fileSize}`,
                  'Content-Type': contentType,
                });
                res.end();
                return;
              }

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
