/**
 * Video Player Component
 * 
 * Represents a single pipeline player card with:
 * - Header (Title, Subtitle, or Mode Dropdown)
 * - Native HTML5 Video Element
 * - Discrete Stage Slider with clickable ticks
 * - Dynamic Stage Title & Detailed Description
 * - Placeholder Badges & Metadata
 */

import { resolveVideoUrl } from "../config/videoPipelineConfig.js";

export class VideoPlayer {
  constructor(options) {
    this.containerId = options.containerId;
    this.container = document.getElementById(this.containerId);
    this.pipelineId = options.pipelineId; // "pnp3d" or "dataDriven"
    this.config = options.config;
    this.isDataDriven = options.isDataDriven || false;
    this.currentMode = options.defaultMode || "CNN";
    const initialStages = this.getCurrentStages();
    this.currentStageIndex = options.defaultStageIndex !== undefined 
      ? options.defaultStageIndex 
      : Math.max(0, initialStages.length - 2);
    this.showGT = options.showGT || false;
    this.onVideoChange = options.onVideoChange || (() => {});
    this.onUserSeek = options.onUserSeek || (() => {});
    this.onUserPlayPause = options.onUserPlayPause || (() => {});

    this.videoElement = null;
    this.sliderElement = null;
    this.ticksContainer = null;
    this.stageInfoContainer = null;

    this.render();
  }

  getCurrentStages() {
    if (this.isDataDriven) {
      return this.config.modes[this.currentMode].stages;
    }
    return this.config.stages;
  }

  getCurrentStage() {
    const stages = this.getCurrentStages();
    if (this.currentStageIndex >= stages.length) {
      this.currentStageIndex = stages.length - 1;
    }
    return stages[this.currentStageIndex];
  }

  render() {
    if (!this.container) return;

    const stages = this.getCurrentStages();
    const stage = this.getCurrentStage();
    const videoUrl = resolveVideoUrl(stage, this.showGT);

    let headerHtml = "";
    if (this.isDataDriven) {
      headerHtml = `
        <div class="player-header">
          <div class="player-title-group">
            <h3 class="player-title">${this.config.title}</h3>
            <div class="mode-select-wrapper">
              <label for="${this.containerId}-mode" class="mode-label">Model Architecture:</label>
              <select id="${this.containerId}-mode" class="player-dropdown">
                <option value="CNN" ${this.currentMode === "CNN" ? "selected" : ""}>CNN (Deep Neural Pose)</option>
                <option value="ML" ${this.currentMode === "ML" ? "selected" : ""}>ML (Feature Regressor)</option>
              </select>
            </div>
          </div>
          <span class="pipeline-badge data-badge">${this.config.modes[this.currentMode].badge}</span>
        </div>
      `;
    } else {
      headerHtml = `
        <div class="player-header">
          <div class="player-title-group">
            <h3 class="player-title">${this.config.title}</h3>
            <span class="player-subtitle">${this.config.subtitle}</span>
          </div>
          <span class="pipeline-badge pnp-badge">${this.config.badge}</span>
        </div>
      `;
    }

    this.container.innerHTML = `
      <div class="player-card">
        ${headerHtml}
        
        <div class="video-frame-container">
          <div class="video-overlay-loader" id="${this.containerId}-loader">
            <div class="loader-spinner"></div>
          </div>
          <video 
            id="${this.containerId}-video"
            class="pipeline-video"
            src="${videoUrl}"
            preload="auto"
            muted
            playsinline
            tabindex="0"
          ></video>
          <div class="video-meta-overlay">
            <span class="overlay-shutter">${stage.shutter}</span>
            ${stage.isPlaceholder ? `<span class="overlay-placeholder">Demo Fallback</span>` : ""}
          </div>
        </div>

        <div class="player-controls-area">
          <div class="stage-slider-block">
            <div class="slider-header">
              <span class="slider-label">Pipeline Stage Timeline</span>
              <span class="stage-step-indicator" id="${this.containerId}-step-indicator">
                Step ${this.currentStageIndex + 1} of ${stages.length}
              </span>
            </div>

            <!-- Custom Discrete Slider with Ticks -->
            <div class="custom-discrete-slider">
              <input 
                type="range" 
                id="${this.containerId}-slider" 
                class="stage-range-slider"
                min="0" 
                max="${stages.length - 1}" 
                value="${this.currentStageIndex}" 
                step="1"
              />
              <div class="slider-ticks" id="${this.containerId}-ticks">
                ${this.renderTicks(stages, this.currentStageIndex)}
              </div>
            </div>
          </div>

          <!-- Dynamic Stage Info Block -->
          <div class="stage-info-card" id="${this.containerId}-info">
            ${this.renderStageInfo(stage, this.currentStageIndex, stages.length)}
          </div>
        </div>
      </div>
    `;

    this.bindElements();
  }

  renderTicks(stages, activeIndex) {
    return stages.map((stg, i) => `
      <div class="tick-mark ${i === activeIndex ? "active" : ""} ${i < activeIndex ? "passed" : ""}" data-index="${i}" title="${stg.name}">
        <span class="tick-label">${stg.name}</span>
      </div>
    `).join("");
  }

  renderStageInfo(stage, index, total) {
    return `
      <div class="stage-info-header">
        <div class="stage-name-wrapper">
          <span class="stage-number">0${index + 1}</span>
          <h4 class="stage-name">${stage.name}</h4>
        </div>
        <span class="stage-shutter-pill">${stage.shutter} Exposure</span>
      </div>
      <p class="stage-description">${stage.description}</p>
      ${stage.isPlaceholder ? `
        <div class="placeholder-notice">
          <svg class="info-icon" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
          </svg>
          <span><strong>Notice:</strong> ${stage.placeholderNote || "Specialized render stage in progress — displaying synchronized reference stream."}</span>
        </div>
      ` : ""}
    `;
  }

  bindElements() {
    this.videoElement = document.getElementById(`${this.containerId}-video`);
    this.sliderElement = document.getElementById(`${this.containerId}-slider`);
    this.ticksContainer = document.getElementById(`${this.containerId}-ticks`);
    this.stageInfoContainer = document.getElementById(`${this.containerId}-info`);
    const loader = document.getElementById(`${this.containerId}-loader`);

    if (this.videoElement) {
      this.videoElement.addEventListener("waiting", () => {
        if (loader) loader.classList.add("visible");
      });
      this.videoElement.addEventListener("canplay", () => {
        if (loader) loader.classList.remove("visible");
      });
      this.videoElement.addEventListener("playing", () => {
        if (loader) loader.classList.remove("visible");
      });
    }

    // Dropdown change for Data-Driven mode
    if (this.isDataDriven) {
      const dropdown = document.getElementById(`${this.containerId}-mode`);
      if (dropdown) {
        dropdown.addEventListener("change", (e) => {
          this.setMode(e.target.value);
        });
      }
    }

    // Slider input change
    if (this.sliderElement) {
      this.sliderElement.addEventListener("input", (e) => {
        this.setStageIndex(parseInt(e.target.value, 10));
      });
    }

    // Click on tick mark directly
    if (this.ticksContainer) {
      this.ticksContainer.addEventListener("click", (e) => {
        const tick = e.target.closest(".tick-mark");
        if (tick && tick.dataset.index !== undefined) {
          const newIdx = parseInt(tick.dataset.index, 10);
          this.setStageIndex(newIdx);
        }
      });
    }
  }

  setMode(mode) {
    if (this.currentMode === mode) return;
    this.currentMode = mode;
    const stages = this.getCurrentStages();
    this.currentStageIndex = Math.max(0, stages.length - 2);
    this.render();
    this.onVideoChange();
  }

  setStageIndex(newIndex) {
    const stages = this.getCurrentStages();
    if (newIndex < 0 || newIndex >= stages.length) return;
    if (this.currentStageIndex === newIndex) return;

    this.currentStageIndex = newIndex;
    this.updateStageUI();
    this.onVideoChange();
  }

  setShowGT(showGT) {
    if (this.showGT === showGT) return;
    this.showGT = showGT;
    this.updateVideoSourceOnly();
  }

  updateStageUI() {
    const stages = this.getCurrentStages();
    const stage = this.getCurrentStage();

    if (this.sliderElement) {
      this.sliderElement.max = stages.length - 1;
      this.sliderElement.value = this.currentStageIndex;
    }

    const stepIndicator = document.getElementById(`${this.containerId}-step-indicator`);
    if (stepIndicator) {
      stepIndicator.textContent = `Step ${this.currentStageIndex + 1} of ${stages.length}`;
    }

    if (this.ticksContainer) {
      this.ticksContainer.innerHTML = this.renderTicks(stages, this.currentStageIndex);
    }

    if (this.stageInfoContainer) {
      this.stageInfoContainer.innerHTML = this.renderStageInfo(stage, this.currentStageIndex, stages.length);
    }

    this.updateVideoSourceOnly();
  }

  updateVideoSourceOnly() {
    if (!this.videoElement) return;
    const stage = this.getCurrentStage();
    const newUrl = resolveVideoUrl(stage, this.showGT);

    // Capture current time & state before swapping src
    const currentTime = this.videoElement.currentTime;
    const isPaused = this.videoElement.paused;

    if (this.videoElement.getAttribute("src") !== newUrl) {
      this.videoElement.src = newUrl;
      this.videoElement.load();
      
      const handleCanPlay = () => {
        this.videoElement.removeEventListener("canplay", handleCanPlay);
        if (Number.isFinite(currentTime) && currentTime < (this.videoElement.duration || Infinity)) {
          this.videoElement.currentTime = currentTime;
        }
        if (!isPaused) {
          this.videoElement.play().catch(() => {});
        }
      };

      this.videoElement.addEventListener("canplay", handleCanPlay);
    }

    // Update overlay metadata
    const metaContainer = this.container.querySelector(".video-meta-overlay");
    if (metaContainer) {
      metaContainer.innerHTML = `
        <span class="overlay-shutter">${stage.shutter}</span>
        ${stage.isPlaceholder ? `<span class="overlay-placeholder">Demo Fallback</span>` : ""}
      `;
    }
  }
}
