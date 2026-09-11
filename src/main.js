/**
 * Main Application Bootstrap
 * 
 * Mounts the data-driven presentation report blocks and initializes
 * the dual-pipeline synchronized video comparison system.
 */

import { reportBlocks } from "./config/reportContent.js";
import { PIPELINE_CONFIG, getAllPipelineVideoUrls } from "./config/videoPipelineConfig.js";
import { PresentationRenderer } from "./components/PresentationRenderer.js";
import { VideoPlayer } from "./components/VideoPlayer.js";
import { VideoSyncController } from "./components/VideoSyncController.js";
import { videoPreloader } from "./utils/videoPreloader.js";

document.addEventListener("DOMContentLoaded", () => {
  // 1. Render Presentation & Report Section
  const presentationRenderer = new PresentationRenderer("presentation-root", reportBlocks);
  presentationRenderer.render();

  // 2. Initialize Left Video Player (3D Pipeline / PnP)
  const playerLeft = new VideoPlayer({
    containerId: "player-pnp-container",
    pipelineId: "pnp3d",
    config: PIPELINE_CONFIG.pnp3d,
    isDataDriven: false,
    showGT: false
  });

  // 3. Initialize Right Video Player (Data Driven Pipeline: CNN / ML)
  const playerRight = new VideoPlayer({
    containerId: "player-data-container",
    pipelineId: "dataDriven",
    config: PIPELINE_CONFIG.dataDriven,
    isDataDriven: true,
    defaultMode: PIPELINE_CONFIG.dataDriven.defaultMode || "CNN",
    showGT: false
  });

  // 4. Initialize Global Synchronization Controller
  const syncController = new VideoSyncController({
    controlBarId: "sync-control-bar",
    playerLeft: playerLeft,
    playerRight: playerRight
  });

  // 5. Preload all pipeline video tracks in the background for snappy responsiveness
  const allVideoUrls = getAllPipelineVideoUrls();
  videoPreloader.preloadAll(allVideoUrls);

  console.log("6-DoF Tracking Benchmark App initialized successfully.");
});
