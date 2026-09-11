/**
 * Presentation Section Renderer
 * 
 * Generates clean, accessible, academic presentation cards dynamically
 * from the reportContent.js data configuration.
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
      blockEl.className = `presentation-block block-${block.layout}`;
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

  renderImageRightBlock(block) {
    return `
      <div class="block-inner split-grid">
        <div class="block-text-column">
          ${block.tag ? `<span class="block-tag">${block.tag}</span>` : ""}
          <h2 class="block-title">${block.title}</h2>
          <div class="block-body">
            ${block.content.map(p => `<p>${p}</p>`).join("")}
          </div>
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
          ${block.tag ? `<span class="block-tag">${block.tag}</span>` : ""}
          <h2 class="block-title">${block.title}</h2>
          <div class="block-body">
            ${block.content.map(p => `<p>${p}</p>`).join("")}
          </div>
          ${this.renderHighlights(block.highlights)}
        </div>
      </div>
    `;
  }

  renderImageBottomBlock(block) {
    return `
      <div class="block-inner full-width-card">
        <div class="card-header">
          ${block.tag ? `<span class="block-tag">${block.tag}</span>` : ""}
          <h2 class="block-title">${block.title}</h2>
          <div class="block-body">
            ${block.content.map(p => `<p>${p}</p>`).join("")}
          </div>
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
        ${block.tag ? `<span class="block-tag">${block.tag}</span>` : ""}
        <h2 class="block-title">${block.title}</h2>
        <div class="block-body">
          ${block.content ? block.content.map(p => `<p>${p}</p>`).join("") : ""}
        </div>
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

  renderMedia(image, isLarge = false) {
    if (!image) return "";
    return `
      <figure class="block-figure ${isLarge ? "figure-large" : ""}">
        <div class="figure-img-wrapper">
          <img src="${image.src}" alt="${image.alt || ""}" loading="lazy" class="figure-image" />
        </div>
        ${image.caption ? `<figcaption class="figure-caption">${image.caption}</figcaption>` : ""}
      </figure>
    `;
  }
}
