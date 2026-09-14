/**
 * Video Sync Controller
 * 
 * Manages dual-video synchronization across Left (3D Pipeline) and Right (Data-Driven Pipeline) players.
 * Features:
 * - Master Play/Pause coordinator
 * - Global Timeline Scrubber & Frame-accurate stepper
 * - Time Alignment & Drift Correction Loop
 * - Ground Truth overlay toggle for both players
 * - Playback rate control & Keyboard shortcuts
 */

export class VideoSyncController {
  constructor(options) {
    this.controlBarId = options.controlBarId;
    this.controlBar = document.getElementById(this.controlBarId);
    this.playerLeft = options.playerLeft;
    this.playerRight = options.playerRight;

    this.isPlaying = false;
    this.isScrubbing = false;
    this.showGT = false;
    this.isSimulated = false;
    this.playbackRate = 1.0;
    this.isLooping = true;
    this.fps = 36.0;
    this.totalFrames = 500;
    this.startFrame = 2200;
    this.activeDirectory = "renders_20260911_233706";

    this.syncInterval = null;
    this.isSyncingInternally = false;

    this.initControls();
    this.bindVideoEvents();
    this.startDriftSyncLoop();
    this.fetchDatasetMetadata();
  }

  async fetchDatasetMetadata() {
    const rawBase = import.meta.env.BASE_URL || "./";
    const basePrefix = rawBase.endsWith("/") ? rawBase : `${rawBase}/`;
    const staticJsonUrl = `${basePrefix}api/dataset-info.json`;

    try {
      let res = await fetch(staticJsonUrl);
      if (!res.ok) {
        res = await fetch("/api/dataset-info");
      }
      if (res.ok) {
        const data = await res.json();
        if (data.frameCount) this.totalFrames = data.frameCount;
        if (data.startFrame) this.startFrame = data.startFrame;
        if (data.fps) this.fps = data.fps;
        if (data.activeDirectory) this.activeDirectory = data.activeDirectory;

        // Update footer reference if element exists
        const dataRefEl = document.querySelector("#data-reference .block-body p");
        if (dataRefEl) {
          const dur = (data.duration || (this.totalFrames / this.fps)).toFixed(2);
          const offset = (data.timeSyncOffset !== undefined ? data.timeSyncOffset : 18.97).toFixed(2);
          dataRefEl.innerHTML = `Current synchronized renders loaded from <code>public/data/renders/</code>. Sequence contains ${this.totalFrames} frames (${dur}s) starting at frame ${this.startFrame}, evaluated with time synchronization offset &Delta;t = +${offset}s against Meta Quest VR 6-DoF ground truth.`;
        }

        this.updateTimeUI();
      }
    } catch (err) {
      console.log("Using static dataset configuration:", err);
    }
  }

  get v1() {
    return this.playerLeft?.videoElement;
  }

  get v2() {
    return this.playerRight?.videoElement;
  }

  get masterDuration() {
    const d1 = this.v1?.duration || 0;
    const d2 = this.v2?.duration || 0;
    return Math.max(d1, d2) || 5.56;
  }

  get masterCurrentTime() {
    return this.v1?.currentTime || this.v2?.currentTime || 0;
  }

  initControls() {
    if (!this.controlBar) return;

    this.controlBar.innerHTML = `
      <div class="sync-control-panel">
        <div class="sync-panel-top">
          <!-- Left: Play/Pause, Frame Stepping, Speed -->
          <div class="control-group-left">
            <button id="master-play-btn" class="btn-primary-play" title="Play/Pause (Spacebar)">
              <svg id="play-icon" class="btn-icon" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z"/>
              </svg>
              <svg id="pause-icon" class="btn-icon hidden" viewBox="0 0 24 24" fill="currentColor">
                <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
              </svg>
              <span id="play-btn-text">Play</span>
            </button>

            <div class="frame-step-group">
              <button id="step-back-btn" class="btn-secondary-step" title="Step Back 1 Frame (Left Arrow)">
                <svg viewBox="0 0 20 20" fill="currentColor" class="step-icon">
                  <path fill-rule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clip-rule="evenodd"/>
                </svg>
                <span>-1 Fr</span>
              </button>
              <button id="step-forward-btn" class="btn-secondary-step" title="Step Forward 1 Frame (Right Arrow)">
                <span>+1 Fr</span>
                <svg viewBox="0 0 20 20" fill="currentColor" class="step-icon">
                  <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd"/>
                </svg>
              </button>
            </div>

            <!-- Playback Speed -->
            <div class="speed-control-group">
              <label for="playback-speed-select" class="control-label">Speed:</label>
              <select id="playback-speed-select" class="speed-select">
                <option value="0.25">0.25x</option>
                <option value="0.5">0.5x</option>
                <option value="1" selected>1.0x (Normal)</option>
                <option value="1.5">1.5x</option>
                <option value="2">2.0x</option>
              </select>
            </div>
          </div>

          <!-- Right: Raspberry Pi 4B Toggle, Ground Truth Toggle & Sync Status -->
          <div class="control-group-right">
            <label class="sim-toggle-card" for="global-sim-checkbox" title="Simulate Raspberry Pi 4B real-time frame rates (CNN: 5 FPS, ML: 25 FPS, PnP: 75 FPS)">
              <input type="checkbox" id="global-sim-checkbox" class="sim-checkbox" />
              <div class="sim-toggle-switch"></div>
              <div class="sim-label-group">
                <span class="sim-title">Simulate Raspberry Pi 4B</span>
                <span class="sim-subtitle">Hardware FPS (5/25/75 FPS)</span>
              </div>
            </label>

            <label class="gt-toggle-card" for="global-gt-checkbox">
              <input type="checkbox" id="global-gt-checkbox" class="gt-checkbox" />
              <div class="gt-toggle-switch"></div>
              <div class="gt-label-group">
                <span class="gt-title">Show Ground Truth</span>
                <span class="gt-subtitle">VR 6-DoF Overlay</span>
              </div>
            </label>

            <div class="sync-status-badge" id="sync-status">
              <span class="status-dot"></span>
              <span class="status-text">Synced</span>
            </div>
          </div>
        </div>

        <!-- Global Progress / Scrubber Bar -->
        <div class="timeline-scrubber-wrapper">
          <div class="time-display" id="time-display">00:00.00 / 00:05.56 (Frame 001/200)</div>
          <div class="scrubber-bar-container">
            <input 
              type="range" 
              id="global-timeline-slider" 
              class="global-scrubber-input" 
              min="0" 
              max="5.56" 
              step="0.01" 
              value="0" 
            />
            <div class="scrubber-progress-fill" id="scrubber-fill"></div>
          </div>
        </div>
      </div>
    `;

    this.bindControlElements();
  }

  bindControlElements() {
    const playBtn = document.getElementById("master-play-btn");
    const stepBackBtn = document.getElementById("step-back-btn");
    const stepFwdBtn = document.getElementById("step-forward-btn");
    const speedSelect = document.getElementById("playback-speed-select");
    const gtCheckbox = document.getElementById("global-gt-checkbox");
    const simCheckbox = document.getElementById("global-sim-checkbox");
    const scrubber = document.getElementById("global-timeline-slider");

    if (playBtn) {
      playBtn.addEventListener("click", () => this.togglePlayPause());
    }

    if (stepBackBtn) {
      stepBackBtn.addEventListener("click", () => this.stepFrames(-1));
    }

    if (stepFwdBtn) {
      stepFwdBtn.addEventListener("click", () => this.stepFrames(1));
    }

    if (speedSelect) {
      speedSelect.addEventListener("change", (e) => {
        this.setPlaybackRate(parseFloat(e.target.value));
      });
    }

    if (gtCheckbox) {
      gtCheckbox.addEventListener("change", (e) => {
        this.setGroundTruth(e.target.checked);
      });
    }

    if (simCheckbox) {
      simCheckbox.addEventListener("change", (e) => {
        this.setSimulated(e.target.checked);
      });
    }

    if (scrubber) {
      scrubber.addEventListener("input", (e) => {
        this.isScrubbing = true;
        const targetTime = parseFloat(e.target.value);
        this.seekTo(targetTime);
      });

      scrubber.addEventListener("change", (e) => {
        this.isScrubbing = false;
        const targetTime = parseFloat(e.target.value);
        this.seekTo(targetTime);
      });
    }

    // Keyboard navigation shortcuts
    window.addEventListener("keydown", (e) => {
      // Don't intercept when user is typing in an input
      if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;

      if (e.code === "Space") {
        e.preventDefault();
        this.togglePlayPause();
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        this.stepFrames(-1);
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        this.stepFrames(1);
      }
    });
  }

  bindVideoEvents() {
    [this.playerLeft, this.playerRight].forEach((player) => {
      if (!player) return;

      player.onVideoChange = () => {
        this.handlePlayerVideoSourceChange();
      };
    });

    this.attachVideoListeners(this.v1);
    this.attachVideoListeners(this.v2);
  }

  attachVideoListeners(video) {
    if (!video) return;

    video.addEventListener("play", () => {
      if (this.isSyncingInternally) return;
      this.handleVideoPlayEvent(video);
    });

    video.addEventListener("pause", () => {
      if (this.isSyncingInternally) return;
      this.handleVideoPauseEvent(video);
    });

    video.addEventListener("timeupdate", () => {
      if (!this.isScrubbing) {
        this.updateTimeUI();
      }
    });

    video.addEventListener("ended", () => {
      this.handleVideoEnded();
    });
  }

  handlePlayerVideoSourceChange() {
    // Reattach listeners to new video elements if re-rendered
    setTimeout(() => {
      this.attachVideoListeners(this.v1);
      this.attachVideoListeners(this.v2);
      this.syncPlaybackRate();
    }, 50);
  }

  handleVideoPlayEvent(sourceVideo) {
    this.isPlaying = true;
    this.updatePlayBtnUI();
    const otherVideo = sourceVideo === this.v1 ? this.v2 : this.v1;

    if (otherVideo && otherVideo.paused) {
      this.isSyncingInternally = true;
      otherVideo.currentTime = sourceVideo.currentTime;
      otherVideo.play().catch(() => {}).finally(() => {
        this.isSyncingInternally = false;
      });
    }
  }

  handleVideoPauseEvent(sourceVideo) {
    if (this.isSyncingInternally) return;
    this.isPlaying = false;
    this.updatePlayBtnUI();
    const otherVideo = sourceVideo === this.v1 ? this.v2 : this.v1;

    if (otherVideo && !otherVideo.paused) {
      this.isSyncingInternally = true;
      otherVideo.pause();
      otherVideo.currentTime = sourceVideo.currentTime;
      this.isSyncingInternally = false;
    }
  }

  handleVideoEnded() {
    if (this.isLooping) {
      this.seekTo(0);
      this.play();
    } else {
      this.pause();
    }
  }

  togglePlayPause() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  play() {
    this.isPlaying = true;
    this.updatePlayBtnUI();
    const targetTime = this.masterCurrentTime;

    if (this.v1) {
      this.v1.currentTime = targetTime;
      this.v1.play().catch(() => {});
    }
    if (this.v2) {
      this.v2.currentTime = targetTime;
      this.v2.play().catch(() => {});
    }
  }

  pause() {
    this.isPlaying = false;
    this.updatePlayBtnUI();
    if (this.v1) this.v1.pause();
    if (this.v2) this.v2.pause();
  }

  seekTo(timeInSeconds) {
    const duration = this.masterDuration;
    const clampedTime = Math.max(0, Math.min(timeInSeconds, duration));

    this.isSyncingInternally = true;
    if (this.v1) this.v1.currentTime = clampedTime;
    if (this.v2) this.v2.currentTime = clampedTime;
    this.isSyncingInternally = false;

    this.updateTimeUI(clampedTime);
  }

  stepFrames(frameCount) {
    this.pause();
    const frameDuration = this.masterDuration / Math.max(1, this.totalFrames);
    const newTime = this.masterCurrentTime + frameCount * frameDuration;
    this.seekTo(newTime);
  }

  setPlaybackRate(rate) {
    this.playbackRate = rate;
    this.syncPlaybackRate();
  }

  syncPlaybackRate() {
    if (this.v1) this.v1.playbackRate = this.playbackRate;
    if (this.v2) this.v2.playbackRate = this.playbackRate;
  }

  setGroundTruth(showGT) {
    this.showGT = showGT;
    this.playerLeft?.setShowGT(showGT);
    this.playerRight?.setShowGT(showGT);
  }

  setSimulated(isSimulated) {
    this.isSimulated = isSimulated;
    this.playerLeft?.setSimulated(isSimulated);
    this.playerRight?.setSimulated(isSimulated);
  }

  startDriftSyncLoop() {
    if (this.syncInterval) clearInterval(this.syncInterval);

    this.syncInterval = setInterval(() => {
      if (!this.isPlaying || this.isScrubbing || !this.v1 || !this.v2) return;

      const t1 = this.v1.currentTime;
      const t2 = this.v2.currentTime;
      const drift = Math.abs(t1 - t2);

      const statusEl = document.getElementById("sync-status");

      if (drift > 0.05) {
        if (statusEl) {
          statusEl.classList.add("re-aligning");
          statusEl.querySelector(".status-text").textContent = "Aligning...";
        }
        // Snap the lagging or leading video to match
        this.v2.currentTime = t1;
      } else {
        if (statusEl) {
          statusEl.classList.remove("re-aligning");
          statusEl.querySelector(".status-text").textContent = "Synced";
        }
      }
    }, 150);
  }

  updatePlayBtnUI() {
    const playIcon = document.getElementById("play-icon");
    const pauseIcon = document.getElementById("pause-icon");
    const playText = document.getElementById("play-btn-text");

    if (this.isPlaying) {
      playIcon?.classList.add("hidden");
      pauseIcon?.classList.remove("hidden");
      if (playText) playText.textContent = "Pause";
    } else {
      playIcon?.classList.remove("hidden");
      pauseIcon?.classList.add("hidden");
      if (playText) playText.textContent = "Play";
    }
  }

  updateTimeUI(forcedTime = null) {
    const currentTime = forcedTime !== null ? forcedTime : this.masterCurrentTime;
    const duration = this.masterDuration;

    const frameProgress = duration > 0 ? (currentTime / duration) : 0;
    const frameOffset = Math.floor(frameProgress * this.totalFrames);
    const currentFrame = Math.min(this.totalFrames, Math.max(1, frameOffset + 1));
    const absoluteFrame = this.startFrame + frameOffset;

    const format = (sec) => {
      const m = Math.floor(sec / 60).toString().padStart(2, "0");
      const s = (sec % 60).toFixed(2).padStart(5, "0");
      return `${m}:${s}`;
    };

    const timeDisplay = document.getElementById("time-display");
    if (timeDisplay) {
      timeDisplay.textContent = `${format(currentTime)} / ${format(duration)} (Frame ${currentFrame.toString().padStart(3, "0")}/${this.totalFrames} | #${absoluteFrame})`;
    }

    const scrubber = document.getElementById("global-timeline-slider");
    if (scrubber && !this.isScrubbing) {
      scrubber.max = duration;
      scrubber.value = currentTime;
    }

    const fill = document.getElementById("scrubber-fill");
    if (fill && duration > 0) {
      const percentage = (currentTime / duration) * 100;
      fill.style.width = `${percentage}%`;
    }
  }
}
