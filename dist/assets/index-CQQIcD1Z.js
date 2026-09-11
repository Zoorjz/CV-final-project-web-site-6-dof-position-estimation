(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))i(a);new MutationObserver(a=>{for(const s of a)if(s.type==="childList")for(const l of s.addedNodes)l.tagName==="LINK"&&l.rel==="modulepreload"&&i(l)}).observe(document,{childList:!0,subtree:!0});function t(a){const s={};return a.integrity&&(s.integrity=a.integrity),a.referrerPolicy&&(s.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?s.credentials="include":a.crossOrigin==="anonymous"?s.credentials="omit":s.credentials="same-origin",s}function i(a){if(a.ep)return;a.ep=!0;const s=t(a);fetch(a.href,s)}})();const y=[{id:"hero",layout:"hero",badge:"Final Research Report & Interactive Demo",title:"Optical 6-DoF Rigid Body Tracking",subtitle:"High-precision optical pose estimation benchmarked against millimeter-accurate Meta Quest VR Ground Truth across classical vision and deep learning pipelines.",stats:[{label:"Median Position Error",value:"18.84 mm",subtext:"Rigid body translation accuracy"},{label:"Median Angular Error",value:"3.55°",subtext:"3D Euler orientation accuracy"},{label:"Shutter Speeds",value:"1ms / 10ms",subtext:"Dark IR vs. Ambient Visual"},{label:"Temporal Alignment",value:"Δt = +18.97 s",subtext:"Affine cross-correlation sync"}]},{id:"motivation-hardware",layout:"image-right",tag:"Hardware & Acquisition",title:"Dual-Exposure Capture & Marker Array",content:["Precise 6-DoF tracking in uncontrolled indoor environments faces extreme challenges from ambient illumination, motion blur, and visual occlusions. To solve this, our acquisition system employs an interleaved dual-exposure Raspberry Pi HQ camera rig.","The tracking target consists of a custom rigid bar equipped with calibrated Infrared (IR) LED optical markers. By capturing alternating high-speed short-exposure frames (**1,000 µs / Dark IR**) and regular exposure frames (**10,000 µs / Bright Visual**), we achieve crisp marker isolation without sacrificing scene context."],highlights:["1,000 µs IR dark stream isolates active markers at near-zero background noise.","10,000 µs ambient stream preserves visual scene features for data-driven modeling.","Rigid body geometry defined with sub-millimeter calibrated 3D marker coordinates."],image:{src:"/assets/sample_1000us.png",alt:"Short shutter 1000 µs IR capture showing isolated LED markers",caption:"Figure 1: Isolated IR optical marker centroids under 1,000 µs short-shutter exposure."}},{id:"ground-truth-calibration",layout:"image-left",tag:"Sensor Calibration",title:"VR Ground Truth & Coordinate Alignment",content:["To rigorously evaluate pose estimation accuracy, a Meta Quest 6-DoF VR controller was mechanically coupled to the optical marker bar. The VR tracking system provides high-frequency ground-truth trajectories with millimeter accuracy.","A dual-stage optimization pipeline resolves the coordinate frame transformations: the camera-to-world transform ($T_{\\text{cam} \\to \\text{vr}}$) and controller-to-bar offset ($T_{\\text{ctrl} \\to \\text{bar}}$), alongside an affine temporal cross-correlation ($t_{\\text{GT}} = t_{\\text{video}} + 18.9700\\text{ s}$)."],highlights:["Spatial Extrinsics: $T_{\\text{cam}\\to\\text{vr}}$ estimated via non-linear least squares.","Temporal offset: $\\Delta t = +18.9700\\text{ s}$ verified with motion velocity peaks.","Controller offset: $T_{\\text{ctrl}\\to\\text{bar}} = [23.24, -7.10, -1.51]\\text{ mm}$."],image:{src:"/assets/sample_10000us.png",alt:"Regular exposure 10000 µs visual frame showing ambient tracking environment",caption:"Figure 2: 10,000 µs ambient visual frame displaying controller rig & tracking bar."}},{id:"spatial-temporal-results",layout:"image-bottom",tag:"Benchmark & Trajectory",title:"Spatial-Temporal Alignment & Error Evaluation",content:["The estimated optical trajectory is quantitatively compared against the transformed VR ground truth over a continuous 200-frame motion sequence. Trajectories exhibit consistent spatial fidelity across complex 3D helical sweeps and sharp rotational maneuvers.","The alignment yields a **median 3D translation error of 18.84 mm** and a **median angular orientation error of 3.55°**, establishing a robust baseline for evaluating both classical geometry-based and learned neural estimators."],image:{src:"/assets/alignment_plots.png",alt:"Alignment curves and 3D trajectory comparison between PnP and VR Ground Truth",caption:"Figure 3: Synchronized 3D trajectory curves, Euler angles, and residual error distributions over time."}},{id:"classical-pipeline",layout:"image-right",tag:"Classical Vision",title:"3D Geometric Pipeline: PnP & EKF",content:["The classical 3D pipeline leverages rigorous projective geometry. High-speed dark frames are processed through adaptive thresholding and 2D sub-pixel blob centroiding.","The resulting 2D-3D point correspondences are solved using the Perspective-n-Point (PnP) algorithm with RANSAC outlier rejection, followed by an Extended Kalman Filter (EKF) constant-velocity motion model that suppresses high-frequency jitter."],highlights:["Sub-pixel ellipse fitting achieves $<0.2\\text{ px}$ centroid precision.","Robust PnP solver estimates 6-DoF transformation matrix $[R | t]$.","EKF smoothing guarantees temporal continuity and velocity estimation."],image:{src:"/assets/sample_rendered_frame.png",alt:"Overlaid 3D Coordinate axes and trajectory trail on optical frame",caption:"Figure 4: Rendered optical 3D pose (RGB axes) and trajectory trail aligned with VR Ground Truth."}},{id:"data-driven-pipeline",layout:"image-left",tag:"Data-Driven Approaches",title:"Deep Learning & Regressor Pipelines",content:["In parallel with classical geometric methods, two data-driven paradigms are investigated: an end-to-end **Convolutional Neural Network (CNN)** predicting 6-DoF pose directly from intensity images, and a hybrid **Machine Learning (ML)** regressor mapping extracted 2D blob features to 3D poses.","These models offer superior resilience under severe optical marker occlusions and ambient reflections where classical 2D-3D correspondence matching may fail."],highlights:["CNN Pipeline: Direct end-to-end regression from raw frame to $[q, t]$.","ML Pipeline: Geometric feature embedding + Random Forest / MLP regressor.","Comparative benchmark highlights tradeoffs between computational cost and accuracy."],image:{src:"/assets/sample_rendered_frame.png",alt:"Visual representation of data-driven tracking pipeline",caption:"Figure 5: 6-DoF pose estimation under data-driven neural and machine learning estimators."}}],d="/data/renders",c={pnp3d:{id:"pnp3d",title:"3D Pipeline",subtitle:"PnP",badge:"Classical Vision",stages:[{id:"regular_video",name:"Regular Video",shutter:"10,000 µs",description:"10,000 µs ambient visual stream showing the full room environment and tracking rig.",files:{raw:"11_bright_raw.mp4",gt:"12_bright_gt.mp4"},isPlaceholder:!1},{id:"short_shutter_video",name:"Short Shutter Video",shutter:"1,000 µs",description:"1,000 µs high-speed dark IR frame eliminating ambient illumination and motion blur.",files:{raw:"01_dark_raw.mp4",gt:"02_dark_gt.mp4"},isPlaceholder:!1},{id:"filtration",name:"Filtration",shutter:"1,000 µs",description:"Multi-stage optical pre-filtration (3x3 Gaussian blur, dynamic peak threshold T>=180, morphological opening) isolating genuine LEDs with zero background noise.",files:{raw:"03_dark_filtration.mp4",gt:"04_dark_filtration_gt.mp4"},isPlaceholder:!1},{id:"blob_detection",name:"Blob Detection",shutter:"1,000 µs",description:"Filtered stream with sub-pixel 2D centroid moments: Cyan candidate rings & Top-4 Red tracking markers.",files:{raw:"05_dark_blobs.mp4",gt:"06_dark_blobs_gt.mp4"},isPlaceholder:!1},{id:"pnp",name:"PnP",shutter:"1,000 µs",description:"SQPnP 6-DoF rigid body pose estimation ([R | t]) with RGB 3D axes, active red markers, and cyan motion trail.",files:{raw:"07_dark_pnp.mp4",gt:"08_dark_pnp_gt.mp4"},isPlaceholder:!1},{id:"ekf",name:"EKF",shutter:"1,000 µs",description:"13-State Extended Kalman Filter constant-velocity state smoothing with RGB 3D axes and bright aqua trajectory trail.",files:{raw:"09_dark_ekf.mp4",gt:"10_dark_ekf_gt.mp4"},isPlaceholder:!1}]},dataDriven:{id:"dataDriven",title:"Data Driven Pipeline",defaultMode:"CNN",modes:{CNN:{id:"CNN",name:"CNN",fullName:"Convolutional Neural Network",badge:"Deep Learning (Direct Pose)",stages:[{id:"regular_video",name:"Regular Video",shutter:"10,000 µs",description:"Ambient RGB visual input provided to the neural convolutional feature backbone.",files:{raw:"11_bright_raw.mp4",gt:"12_bright_gt.mp4"},isPlaceholder:!1},{id:"dark_video",name:"Dark Video",shutter:"1,000 µs",description:"1,000 µs high-contrast input tensor passed to convolutional layers for direct marker localization.",files:{raw:"01_dark_raw.mp4",gt:"02_dark_gt.mp4"},isPlaceholder:!1},{id:"cnn_output",name:"CNN Output",shutter:"1,000 µs",description:"Direct 6-DoF pose prediction [q, t] regressed by deep convolutional neural network.",files:{raw:"09_dark_ekf.mp4",gt:"10_dark_ekf_gt.mp4"},isPlaceholder:!0,placeholderNote:"CNN checkpoint inference render in progress"}]},ML:{id:"ML",name:"ML",fullName:"Machine Learning (Feature Regressor)",badge:"Learned 2D-to-3D Regressor",stages:[{id:"regular_video",name:"Regular Video",shutter:"10,000 µs",description:"Ambient visual frame capturing scene lighting and background spatial context.",files:{raw:"11_bright_raw.mp4",gt:"12_bright_gt.mp4"},isPlaceholder:!1},{id:"dark_video",name:"Dark Video",shutter:"1,000 µs",description:"1,000 µs short-shutter frame isolating active IR LED markers.",files:{raw:"01_dark_raw.mp4",gt:"02_dark_gt.mp4"},isPlaceholder:!1},{id:"filtration",name:"Filtration",shutter:"1,000 µs",description:"Optical pre-filtration isolating active optical IR LED emissions.",files:{raw:"03_dark_filtration.mp4",gt:"04_dark_filtration_gt.mp4"},isPlaceholder:!1},{id:"blob_detection",name:"Blob Detection",shutter:"1,000 µs",description:"Extracted 2D centroid coordinates and spatial moments compiled into tabular feature vectors.",files:{raw:"05_dark_blobs.mp4",gt:"06_dark_blobs_gt.mp4"},isPlaceholder:!1},{id:"ml_output",name:"ML Algorithm Output",shutter:"1,000 µs",description:"Predicted 6-DoF pose output from trained Random Forest / MLP feature regression model.",files:{raw:"09_dark_ekf.mp4",gt:"10_dark_ekf_gt.mp4"},isPlaceholder:!0,placeholderNote:"ML regressor output render in progress"}]}}}};function g(r,e){if(!r||!r.files)return"";const t=e&&r.files.gt||r.files.raw;return`${d}/${t}`}function b(){const r=new Set;return c.pnp3d.stages.forEach(e=>{e.files.raw&&r.add(`${d}/${e.files.raw}`),e.files.gt&&r.add(`${d}/${e.files.gt}`)}),Object.values(c.dataDriven.modes).forEach(e=>{e.stages.forEach(t=>{t.files.raw&&r.add(`${d}/${t.files.raw}`),t.files.gt&&r.add(`${d}/${t.files.gt}`)})}),Array.from(r)}class k{constructor(e,t){this.container=document.getElementById(e),this.data=t}render(){!this.container||!this.data||(this.container.innerHTML="",this.data.forEach((e,t)=>{const i=document.createElement("section");switch(i.className=`presentation-block block-${e.layout}`,i.id=`block-${e.id||t}`,i.setAttribute("data-layout",e.layout),e.layout){case"hero":i.innerHTML=this.renderHeroBlock(e);break;case"image-right":i.innerHTML=this.renderImageRightBlock(e);break;case"image-left":i.innerHTML=this.renderImageLeftBlock(e);break;case"image-bottom":i.innerHTML=this.renderImageBottomBlock(e);break;default:i.innerHTML=this.renderStandardBlock(e);break}this.container.appendChild(i)}))}renderHeroBlock(e){const t=e.stats?`
      <div class="hero-stats-grid">
        ${e.stats.map(i=>`
          <div class="stat-card">
            <span class="stat-value">${i.value}</span>
            <span class="stat-label">${i.label}</span>
            ${i.subtext?`<span class="stat-subtext">${i.subtext}</span>`:""}
          </div>
        `).join("")}
      </div>
    `:"";return`
      <div class="hero-content">
        ${e.badge?`<div class="hero-badge"><span class="badge-dot"></span>${e.badge}</div>`:""}
        <h1 class="hero-title">${e.title}</h1>
        <p class="hero-subtitle">${e.subtitle}</p>
        ${t}
      </div>
    `}renderImageRightBlock(e){return`
      <div class="block-inner split-grid">
        <div class="block-text-column">
          ${e.tag?`<span class="block-tag">${e.tag}</span>`:""}
          <h2 class="block-title">${e.title}</h2>
          <div class="block-body">
            ${e.content.map(t=>`<p>${t}</p>`).join("")}
          </div>
          ${this.renderHighlights(e.highlights)}
        </div>
        <div class="block-media-column">
          ${this.renderMedia(e.image)}
        </div>
      </div>
    `}renderImageLeftBlock(e){return`
      <div class="block-inner split-grid reverse-split">
        <div class="block-media-column">
          ${this.renderMedia(e.image)}
        </div>
        <div class="block-text-column">
          ${e.tag?`<span class="block-tag">${e.tag}</span>`:""}
          <h2 class="block-title">${e.title}</h2>
          <div class="block-body">
            ${e.content.map(t=>`<p>${t}</p>`).join("")}
          </div>
          ${this.renderHighlights(e.highlights)}
        </div>
      </div>
    `}renderImageBottomBlock(e){return`
      <div class="block-inner full-width-card">
        <div class="card-header">
          ${e.tag?`<span class="block-tag">${e.tag}</span>`:""}
          <h2 class="block-title">${e.title}</h2>
          <div class="block-body">
            ${e.content.map(t=>`<p>${t}</p>`).join("")}
          </div>
        </div>
        <div class="card-media-large">
          ${this.renderMedia(e.image,!0)}
        </div>
        ${this.renderHighlights(e.highlights)}
      </div>
    `}renderStandardBlock(e){return`
      <div class="block-inner">
        ${e.tag?`<span class="block-tag">${e.tag}</span>`:""}
        <h2 class="block-title">${e.title}</h2>
        <div class="block-body">
          ${e.content?e.content.map(t=>`<p>${t}</p>`).join(""):""}
        </div>
      </div>
    `}renderHighlights(e){return!e||!e.length?"":`
      <ul class="block-highlights">
        ${e.map(t=>`
          <li>
            <svg class="check-icon" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
            </svg>
            <span>${t}</span>
          </li>
        `).join("")}
      </ul>
    `}renderMedia(e,t=!1){return e?`
      <figure class="block-figure ${t?"figure-large":""}">
        <div class="figure-img-wrapper">
          <img src="${e.src}" alt="${e.alt||""}" loading="lazy" class="figure-image" />
        </div>
        ${e.caption?`<figcaption class="figure-caption">${e.caption}</figcaption>`:""}
      </figure>
    `:""}}class m{constructor(e){this.containerId=e.containerId,this.container=document.getElementById(this.containerId),this.pipelineId=e.pipelineId,this.config=e.config,this.isDataDriven=e.isDataDriven||!1,this.currentMode=e.defaultMode||"CNN",this.currentStageIndex=0,this.showGT=e.showGT||!1,this.onVideoChange=e.onVideoChange||(()=>{}),this.onUserSeek=e.onUserSeek||(()=>{}),this.onUserPlayPause=e.onUserPlayPause||(()=>{}),this.videoElement=null,this.sliderElement=null,this.ticksContainer=null,this.stageInfoContainer=null,this.render()}getCurrentStages(){return this.isDataDriven?this.config.modes[this.currentMode].stages:this.config.stages}getCurrentStage(){const e=this.getCurrentStages();return this.currentStageIndex>=e.length&&(this.currentStageIndex=e.length-1),e[this.currentStageIndex]}render(){if(!this.container)return;const e=this.getCurrentStages(),t=this.getCurrentStage(),i=g(t,this.showGT);let a="";this.isDataDriven?a=`
        <div class="player-header">
          <div class="player-title-group">
            <h3 class="player-title">${this.config.title}</h3>
            <div class="mode-select-wrapper">
              <label for="${this.containerId}-mode" class="mode-label">Model Architecture:</label>
              <select id="${this.containerId}-mode" class="player-dropdown">
                <option value="CNN" ${this.currentMode==="CNN"?"selected":""}>CNN (Deep Neural Pose)</option>
                <option value="ML" ${this.currentMode==="ML"?"selected":""}>ML (Feature Regressor)</option>
              </select>
            </div>
          </div>
          <span class="pipeline-badge data-badge">${this.config.modes[this.currentMode].badge}</span>
        </div>
      `:a=`
        <div class="player-header">
          <div class="player-title-group">
            <h3 class="player-title">${this.config.title}</h3>
            <span class="player-subtitle">${this.config.subtitle}</span>
          </div>
          <span class="pipeline-badge pnp-badge">${this.config.badge}</span>
        </div>
      `,this.container.innerHTML=`
      <div class="player-card">
        ${a}
        
        <div class="video-frame-container">
          <div class="video-overlay-loader" id="${this.containerId}-loader">
            <div class="loader-spinner"></div>
          </div>
          <video 
            id="${this.containerId}-video"
            class="pipeline-video"
            src="${i}"
            preload="auto"
            muted
            playsinline
            tabindex="0"
          ></video>
          <div class="video-meta-overlay">
            <span class="overlay-shutter">${t.shutter}</span>
            ${t.isPlaceholder?'<span class="overlay-placeholder">Demo Fallback</span>':'<span class="overlay-live">Active Render</span>'}
          </div>
        </div>

        <div class="player-controls-area">
          <div class="stage-slider-block">
            <div class="slider-header">
              <span class="slider-label">Pipeline Stage Timeline</span>
              <span class="stage-step-indicator" id="${this.containerId}-step-indicator">
                Step ${this.currentStageIndex+1} of ${e.length}
              </span>
            </div>

            <!-- Custom Discrete Slider with Ticks -->
            <div class="custom-discrete-slider">
              <input 
                type="range" 
                id="${this.containerId}-slider" 
                class="stage-range-slider"
                min="0" 
                max="${e.length-1}" 
                value="${this.currentStageIndex}" 
                step="1"
              />
              <div class="slider-ticks" id="${this.containerId}-ticks">
                ${this.renderTicks(e,this.currentStageIndex)}
              </div>
            </div>
          </div>

          <!-- Dynamic Stage Info Block -->
          <div class="stage-info-card" id="${this.containerId}-info">
            ${this.renderStageInfo(t,this.currentStageIndex,e.length)}
          </div>
        </div>
      </div>
    `,this.bindElements()}renderTicks(e,t){return e.map((i,a)=>`
      <div class="tick-mark ${a===t?"active":""} ${a<t?"passed":""}" data-index="${a}" title="${i.name}">
        <div class="tick-dot"></div>
        <span class="tick-label">${i.name}</span>
      </div>
    `).join("")}renderStageInfo(e,t,i){return`
      <div class="stage-info-header">
        <div class="stage-name-wrapper">
          <span class="stage-number">0${t+1}</span>
          <h4 class="stage-name">${e.name}</h4>
        </div>
        <span class="stage-shutter-pill">${e.shutter} Exposure</span>
      </div>
      <p class="stage-description">${e.description}</p>
      ${e.isPlaceholder?`
        <div class="placeholder-notice">
          <svg class="info-icon" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
          </svg>
          <span><strong>Notice:</strong> ${e.placeholderNote||"Specialized render stage in progress — displaying synchronized reference stream."}</span>
        </div>
      `:""}
    `}bindElements(){this.videoElement=document.getElementById(`${this.containerId}-video`),this.sliderElement=document.getElementById(`${this.containerId}-slider`),this.ticksContainer=document.getElementById(`${this.containerId}-ticks`),this.stageInfoContainer=document.getElementById(`${this.containerId}-info`);const e=document.getElementById(`${this.containerId}-loader`);if(this.videoElement&&(this.videoElement.addEventListener("waiting",()=>{e&&e.classList.add("visible")}),this.videoElement.addEventListener("canplay",()=>{e&&e.classList.remove("visible")}),this.videoElement.addEventListener("playing",()=>{e&&e.classList.remove("visible")})),this.isDataDriven){const t=document.getElementById(`${this.containerId}-mode`);t&&t.addEventListener("change",i=>{this.setMode(i.target.value)})}this.sliderElement&&this.sliderElement.addEventListener("input",t=>{this.setStageIndex(parseInt(t.target.value,10))}),this.ticksContainer&&this.ticksContainer.addEventListener("click",t=>{const i=t.target.closest(".tick-mark");if(i&&i.dataset.index!==void 0){const a=parseInt(i.dataset.index,10);this.setStageIndex(a)}})}setMode(e){this.currentMode!==e&&(this.currentMode=e,this.currentStageIndex=0,this.render(),this.onVideoChange())}setStageIndex(e){const t=this.getCurrentStages();e<0||e>=t.length||this.currentStageIndex!==e&&(this.currentStageIndex=e,this.updateStageUI(),this.onVideoChange())}setShowGT(e){this.showGT!==e&&(this.showGT=e,this.updateVideoSourceOnly())}updateStageUI(){const e=this.getCurrentStages(),t=this.getCurrentStage();this.sliderElement&&(this.sliderElement.max=e.length-1,this.sliderElement.value=this.currentStageIndex);const i=document.getElementById(`${this.containerId}-step-indicator`);i&&(i.textContent=`Step ${this.currentStageIndex+1} of ${e.length}`),this.ticksContainer&&(this.ticksContainer.innerHTML=this.renderTicks(e,this.currentStageIndex)),this.stageInfoContainer&&(this.stageInfoContainer.innerHTML=this.renderStageInfo(t,this.currentStageIndex,e.length)),this.updateVideoSourceOnly()}updateVideoSourceOnly(){if(!this.videoElement)return;const e=this.getCurrentStage(),t=g(e,this.showGT),i=this.videoElement.currentTime,a=this.videoElement.paused;if(this.videoElement.getAttribute("src")!==t){this.videoElement.src=t,this.videoElement.load();const l=()=>{this.videoElement.removeEventListener("canplay",l),Number.isFinite(i)&&i<(this.videoElement.duration||1/0)&&(this.videoElement.currentTime=i),a||this.videoElement.play().catch(()=>{})};this.videoElement.addEventListener("canplay",l)}const s=this.container.querySelector(".video-meta-overlay");s&&(s.innerHTML=`
        <span class="overlay-shutter">${e.shutter}</span>
        ${e.isPlaceholder?'<span class="overlay-placeholder">Demo Fallback</span>':'<span class="overlay-live">Active Render</span>'}
      `)}}class ${constructor(e){this.controlBarId=e.controlBarId,this.controlBar=document.getElementById(this.controlBarId),this.playerLeft=e.playerLeft,this.playerRight=e.playerRight,this.isPlaying=!1,this.isScrubbing=!1,this.showGT=!1,this.playbackRate=1,this.isLooping=!0,this.fps=36,this.totalFrames=500,this.startFrame=2200,this.activeDirectory="renders_20260911_233706",this.syncInterval=null,this.isSyncingInternally=!1,this.initControls(),this.bindVideoEvents(),this.startDriftSyncLoop(),this.fetchDatasetMetadata()}async fetchDatasetMetadata(){try{const e=await fetch("/api/dataset-info");if(e.ok){const t=await e.json();t.frameCount&&(this.totalFrames=t.frameCount),t.startFrame&&(this.startFrame=t.startFrame),t.fps&&(this.fps=t.fps),t.activeDirectory&&(this.activeDirectory=t.activeDirectory);const i=document.querySelector("#data-reference .block-body p");i&&(i.innerHTML=`Current synchronized renders loaded from <code>data/renders/${this.activeDirectory}</code>. Sequence contains ${this.totalFrames} frames (${(t.duration||13.89).toFixed(2)}s) starting at frame ${this.startFrame}, evaluated with time synchronization offset &Delta;t = +18.97s against Meta Quest VR 6-DoF ground truth.`),this.updateTimeUI()}}catch(e){console.log("Using static dataset configuration:",e)}}get v1(){var e;return(e=this.playerLeft)==null?void 0:e.videoElement}get v2(){var e;return(e=this.playerRight)==null?void 0:e.videoElement}get masterDuration(){var i,a;const e=((i=this.v1)==null?void 0:i.duration)||0,t=((a=this.v2)==null?void 0:a.duration)||0;return Math.max(e,t)||5.56}get masterCurrentTime(){var e,t;return((e=this.v1)==null?void 0:e.currentTime)||((t=this.v2)==null?void 0:t.currentTime)||0}initControls(){this.controlBar&&(this.controlBar.innerHTML=`
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

          <!-- Right: Ground Truth Toggle & Sync Status -->
          <div class="control-group-right">
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
    `,this.bindControlElements())}bindControlElements(){const e=document.getElementById("master-play-btn"),t=document.getElementById("step-back-btn"),i=document.getElementById("step-forward-btn"),a=document.getElementById("playback-speed-select"),s=document.getElementById("global-gt-checkbox"),l=document.getElementById("global-timeline-slider");e&&e.addEventListener("click",()=>this.togglePlayPause()),t&&t.addEventListener("click",()=>this.stepFrames(-1)),i&&i.addEventListener("click",()=>this.stepFrames(1)),a&&a.addEventListener("change",n=>{this.setPlaybackRate(parseFloat(n.target.value))}),s&&s.addEventListener("change",n=>{this.setGroundTruth(n.target.checked)}),l&&(l.addEventListener("input",n=>{this.isScrubbing=!0;const o=parseFloat(n.target.value);this.seekTo(o)}),l.addEventListener("change",n=>{this.isScrubbing=!1;const o=parseFloat(n.target.value);this.seekTo(o)})),window.addEventListener("keydown",n=>{var o;["INPUT","SELECT","TEXTAREA"].includes((o=document.activeElement)==null?void 0:o.tagName)||(n.code==="Space"?(n.preventDefault(),this.togglePlayPause()):n.code==="ArrowLeft"?(n.preventDefault(),this.stepFrames(-1)):n.code==="ArrowRight"&&(n.preventDefault(),this.stepFrames(1)))})}bindVideoEvents(){[this.playerLeft,this.playerRight].forEach(e=>{e&&(e.onVideoChange=()=>{this.handlePlayerVideoSourceChange()})}),this.attachVideoListeners(this.v1),this.attachVideoListeners(this.v2)}attachVideoListeners(e){e&&(e.addEventListener("play",()=>{this.isSyncingInternally||this.handleVideoPlayEvent(e)}),e.addEventListener("pause",()=>{this.isSyncingInternally||this.handleVideoPauseEvent(e)}),e.addEventListener("timeupdate",()=>{this.isScrubbing||this.updateTimeUI()}),e.addEventListener("ended",()=>{this.handleVideoEnded()}))}handlePlayerVideoSourceChange(){setTimeout(()=>{this.attachVideoListeners(this.v1),this.attachVideoListeners(this.v2),this.syncPlaybackRate()},50)}handleVideoPlayEvent(e){this.isPlaying=!0,this.updatePlayBtnUI();const t=e===this.v1?this.v2:this.v1;t&&t.paused&&(this.isSyncingInternally=!0,t.currentTime=e.currentTime,t.play().catch(()=>{}).finally(()=>{this.isSyncingInternally=!1}))}handleVideoPauseEvent(e){if(this.isSyncingInternally)return;this.isPlaying=!1,this.updatePlayBtnUI();const t=e===this.v1?this.v2:this.v1;t&&!t.paused&&(this.isSyncingInternally=!0,t.pause(),t.currentTime=e.currentTime,this.isSyncingInternally=!1)}handleVideoEnded(){this.isLooping?(this.seekTo(0),this.play()):this.pause()}togglePlayPause(){this.isPlaying?this.pause():this.play()}play(){this.isPlaying=!0,this.updatePlayBtnUI();const e=this.masterCurrentTime;this.v1&&(this.v1.currentTime=e,this.v1.play().catch(()=>{})),this.v2&&(this.v2.currentTime=e,this.v2.play().catch(()=>{}))}pause(){this.isPlaying=!1,this.updatePlayBtnUI(),this.v1&&this.v1.pause(),this.v2&&this.v2.pause()}seekTo(e){const t=this.masterDuration,i=Math.max(0,Math.min(e,t));this.isSyncingInternally=!0,this.v1&&(this.v1.currentTime=i),this.v2&&(this.v2.currentTime=i),this.isSyncingInternally=!1,this.updateTimeUI(i)}stepFrames(e){this.pause();const t=1/this.fps,i=this.masterCurrentTime+e*t;this.seekTo(i)}setPlaybackRate(e){this.playbackRate=e,this.syncPlaybackRate()}syncPlaybackRate(){this.v1&&(this.v1.playbackRate=this.playbackRate),this.v2&&(this.v2.playbackRate=this.playbackRate)}setGroundTruth(e){var t,i;this.showGT=e,(t=this.playerLeft)==null||t.setShowGT(e),(i=this.playerRight)==null||i.setShowGT(e)}startDriftSyncLoop(){this.syncInterval&&clearInterval(this.syncInterval),this.syncInterval=setInterval(()=>{if(!this.isPlaying||this.isScrubbing||!this.v1||!this.v2)return;const e=this.v1.currentTime,t=this.v2.currentTime,i=Math.abs(e-t),a=document.getElementById("sync-status");i>.05?(a&&(a.classList.add("re-aligning"),a.querySelector(".status-text").textContent="Aligning..."),this.v2.currentTime=e):a&&(a.classList.remove("re-aligning"),a.querySelector(".status-text").textContent="Synced")},150)}updatePlayBtnUI(){const e=document.getElementById("play-icon"),t=document.getElementById("pause-icon"),i=document.getElementById("play-btn-text");this.isPlaying?(e==null||e.classList.add("hidden"),t==null||t.classList.remove("hidden"),i&&(i.textContent="Pause")):(e==null||e.classList.remove("hidden"),t==null||t.classList.add("hidden"),i&&(i.textContent="Play"))}updateTimeUI(e=null){const t=e!==null?e:this.masterCurrentTime,i=this.masterDuration,a=Math.floor(t*this.fps),s=Math.min(this.totalFrames,a+1),l=this.startFrame+a,n=h=>{const v=Math.floor(h/60).toString().padStart(2,"0"),f=(h%60).toFixed(2).padStart(5,"0");return`${v}:${f}`},o=document.getElementById("time-display");o&&(o.textContent=`${n(t)} / ${n(i)} (Frame ${s.toString().padStart(3,"0")}/${this.totalFrames} | #${l})`);const u=document.getElementById("global-timeline-slider");u&&!this.isScrubbing&&(u.max=i,u.value=t);const p=document.getElementById("scrubber-fill");if(p&&i>0){const h=t/i*100;p.style.width=`${h}%`}}}class E{constructor(){this.preloaded=new Set,this.videoElements=new Map}preload(e){if(!e||this.preloaded.has(e))return;this.preloaded.add(e);const t=document.createElement("video");t.preload="auto",t.muted=!0,t.playsInline=!0,t.src=e,t.load(),this.videoElements.set(e,t)}preloadAll(e){Array.isArray(e)&&e.forEach(t=>this.preload(t))}}const I=new E;document.addEventListener("DOMContentLoaded",()=>{new k("presentation-root",y).render();const e=new m({containerId:"player-pnp-container",pipelineId:"pnp3d",config:c.pnp3d,isDataDriven:!1,showGT:!1}),t=new m({containerId:"player-data-container",pipelineId:"dataDriven",config:c.dataDriven,isDataDriven:!0,defaultMode:c.dataDriven.defaultMode||"CNN",showGT:!1});new $({controlBarId:"sync-control-bar",playerLeft:e,playerRight:t});const i=b();I.preloadAll(i),console.log("6-DoF Tracking Benchmark App initialized successfully.")});
