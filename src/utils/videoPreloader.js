/**
 * Video Preloader Utility
 * 
 * Preloads video files in memory so changing pipeline stages or toggling GT feels instantaneous.
 */

class VideoPreloader {
  constructor() {
    this.preloaded = new Set();
    this.videoElements = new Map();
  }

  /**
   * Preload a single video URL
   */
  preload(url) {
    if (!url || this.preloaded.has(url)) return;
    this.preloaded.add(url);

    // Using hidden video element to buffer initial segment
    const video = document.createElement("video");
    video.preload = "auto";
    video.muted = true;
    video.playsInline = true;
    video.src = url;
    video.load();

    this.videoElements.set(url, video);
  }

  /**
   * Preload a list of URLs in the background
   */
  preloadAll(urls) {
    if (!Array.isArray(urls)) return;
    urls.forEach(url => this.preload(url));
  }
}

export const videoPreloader = new VideoPreloader();
