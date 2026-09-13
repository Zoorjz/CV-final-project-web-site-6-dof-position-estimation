/**
 * Presentation Section Renderer
 * 
 * Generates clean, accessible, academic presentation cards dynamically
 * from the reportContent.js data configuration with large-font slide layouts.
 */

export class PresentationRenderer {
  constructor(containerId, data) {
    this.container = document.getElementById(containerId);
    this.data = data;
  }

  render() {
    if (!this.container || !this.data) return;
    this.container.innerHTML = "";

    this.data.forEach((block, index) => {
      const blockEl = document.createElement("section");
      blockEl.className = `presentation-block block-${block.layout} ${block.id ? `block-${block.id}` : ""}`;
      blockEl.id = `block-${block.id || index}`;
      blockEl.setAttribute("data-layout", block.layout);

      switch (block.layout) {
        case "hero":
          blockEl.innerHTML = this.renderHeroBlock(block);
          break;
        case "image-right":
          blockEl.innerHTML = this.renderImageRightBlock(block);
          break;
        case "image-left":
          blockEl.innerHTML = this.renderImageLeftBlock(block);
          break;
        case "dual-image-right":
          blockEl.innerHTML = this.renderDualImageBlock(block);
          break;
        case "image-bottom":
          blockEl.innerHTML = this.renderImageBottomBlock(block);
          break;
        default:
          blockEl.innerHTML = this.renderStandardBlock(block);
          break;
      }

      this.container.appendChild(blockEl);
    });
  }

  renderHeroBlock(block) {
    const statsHtml = block.stats ? `
      <div class="hero-stats-grid">
        ${block.stats.map(stat => `
          <div class="stat-card">
            <span class="stat-value">${stat.value}</span>
            <span class="stat-label">${stat.label}</span>
            ${stat.subtext ? `<span class="stat-subtext">${stat.subtext}</span>` : ""}
          </div>
        `).join("")}
      </div>
    ` : "";

    return `
      <div class="hero-content">
        ${block.badge ? `<div class="hero-badge"><span class="badge-dot"></span>${block.badge}</div>` : ""}
        <h1 class="hero-title">${block.title}</h1>
        <p class="hero-subtitle">${block.subtitle}</p>
        ${statsHtml}
      </div>
    `;
  }

  renderHeaderTag(block) {
    if (!block.tag) return "";
    return `
      <div class="block-tag-group">
        <span class="block-tag">${block.tag}</span>
      </div>
    `;
  }

  renderSubsections(subsections) {
    if (!subsections || !subsections.length) return "";
    return `
      <div class="block-subsections">
        ${subsections.map(sub => `
          <div class="subsection-group">
            ${sub.title ? `<h3 class="subsection-title">${sub.title}</h3>` : ""}
            ${sub.items && sub.items.length ? `
              <ul class="subsection-list">
                ${sub.items.map(item => `
                  <li class="subsection-item">
                    <span class="bullet-point"></span>
                    <span class="bullet-text">${item}</span>
                  </li>
                `).join("")}
              </ul>
            ` : ""}
          </div>
        `).join("")}
      </div>
    `;
  }

  renderBodyContent(block) {
    let bodyHtml = "";
    if (block.content && block.content.length) {
      bodyHtml += `<div class="block-body">${block.content.map(p => `<p>${p}</p>`).join("")}</div>`;
    }
    if (block.subsections) {
      bodyHtml += this.renderSubsections(block.subsections);
    }
    return bodyHtml;
  }

  renderImageRightBlock(block) {
    return `
      <div class="block-inner split-grid">
        <div class="block-text-column">
          ${this.renderHeaderTag(block)}
          <h2 class="block-title">${block.title}</h2>
          ${this.renderBodyContent(block)}
          ${this.renderHighlights(block.highlights)}
        </div>
        <div class="block-media-column">
          ${this.renderMedia(block.image)}
        </div>
      </div>
    `;
  }

  renderImageLeftBlock(block) {
    return `
      <div class="block-inner split-grid reverse-split">
        <div class="block-media-column">
          ${this.renderMedia(block.image)}
        </div>
        <div class="block-text-column">
          ${this.renderHeaderTag(block)}
          <h2 class="block-title">${block.title}</h2>
          ${this.renderBodyContent(block)}
          ${this.renderHighlights(block.highlights)}
        </div>
      </div>
    `;
  }

  renderDualImageBlock(block) {
    const images = block.images || (block.image ? [block.image] : []);
    return `
      <div class="block-inner split-grid dual-image-layout">
        <div class="block-text-column">
          ${this.renderHeaderTag(block)}
          <h2 class="block-title">${block.title}</h2>
          ${this.renderBodyContent(block)}
          ${this.renderHighlights(block.highlights)}
        </div>
        <div class="block-media-column dual-media-column">
          <div class="dual-images-grid">
            ${images.map(img => this.renderMedia(img, false, "dual-img-card")).join("")}
          </div>
        </div>
      </div>
    `;
  }

  renderImageBottomBlock(block) {
    return `
      <div class="block-inner full-width-card">
        <div class="card-header">
          ${this.renderHeaderTag(block)}
          <h2 class="block-title">${block.title}</h2>
          ${this.renderBodyContent(block)}
        </div>
        <div class="card-media-large">
          ${this.renderMedia(block.image, true)}
        </div>
        ${this.renderHighlights(block.highlights)}
      </div>
    `;
  }

  renderStandardBlock(block) {
    return `
      <div class="block-inner">
        ${this.renderHeaderTag(block)}
        <h2 class="block-title">${block.title}</h2>
        ${this.renderBodyContent(block)}
      </div>
    `;
  }

  renderHighlights(highlights) {
    if (!highlights || !highlights.length) return "";
    return `
      <ul class="block-highlights">
        ${highlights.map(item => `
          <li>
            <svg class="check-icon" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
            </svg>
            <span>${item}</span>
          </li>
        `).join("")}
      </ul>
    `;
  }

  resolveAssetUrl(src) {
    if (!src) return "";
    if (src.startsWith("http://") || src.startsWith("https://") || src.startsWith("data:")) {
      return src;
    }
    const rawBase = import.meta.env.BASE_URL || "./";
    const basePrefix = rawBase.endsWith("/") ? rawBase : `${rawBase}/`;
    const cleanSrc = src.startsWith("/") ? src.slice(1) : src;
    return `${basePrefix}${cleanSrc}`;
  }

  renderMedia(image, isLarge = false, extraClass = "") {
    if (!image) return "";
    const cropClass = image.isEnlargedCrop ? "figure-crop-zoom" : "";
    const resolvedSrc = this.resolveAssetUrl(image.src);
    return `
      <figure class="block-figure ${isLarge ? "figure-large" : ""} ${extraClass} ${cropClass}">
        <div class="figure-img-wrapper">
          <img src="${resolvedSrc}" alt="${image.alt || ""}" loading="lazy" class="figure-image" />
        </div>
        ${image.caption ? `<figcaption class="figure-caption">${image.caption}</figcaption>` : ""}
      </figure>
    `;
  }
}
