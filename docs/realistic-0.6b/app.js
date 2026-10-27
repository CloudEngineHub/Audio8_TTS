/* Both the series landing page and the permanent demo use these same assets. */
(() => {
  "use strict";
  const assetBase = new URL(".", document.currentScript.src);
  const root = document.getElementById("demo-root");
  const escape = (text) => String(text ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const data = window.__DATA__;
  if (!data || !Array.isArray(data.samples) || !data.samples.length) {
    root.innerHTML = '<p class="audio-error" role="alert">The listening examples could not load. Please refresh the page.</p>';
    return;
  }
  const player = (sample, column) => {
    const model = data.models[column.key];
    const focus = column.key === "realistic";
    return `<section class="player${focus ? " focus" : ""}${column.key === "reference" ? " reference" : ""}" aria-label="${escape(model.label)} for ${escape(sample.label)}">
      <div class="player-heading"><h4>${escape(model.label)}</h4>${model.size ? `<span class="size">${escape(model.size)}</span>` : ""}${focus ? '<span class="focus-caption">Current release</span>' : ""}</div>
      <div><audio controls preload="none" playsinline aria-label="${escape(sample.label)} — ${escape(model.label)}" src="${escape(new URL(column.audio, assetBase).href)}"></audio><span class="duration">${Number(column.duration).toFixed(2)} s${column.key === "reference" ? " · Reference recording" : ""}</span></div>
    </section>`;
  };
  root.innerHTML = data.samples.map(sample => {
    const reference = {key:"reference",audio:sample.reference_audio,duration:sample.reference_duration};
    const columns = [reference, ...sample.columns.filter(c => c.key !== "realistic"), ...sample.columns.filter(c => c.key === "realistic")];
    return `<article class="sample-card" id="${escape(sample.id)}" data-language="${escape(sample.language)}">
      <div class="sample-head"><div class="sample-meta"><h3>${escape(sample.label)}</h3><span class="language-tag">${sample.language === "zh" ? "Chinese" : "English"}</span><a class="sample-link" href="#${escape(sample.id)}" aria-label="Link to ${escape(sample.label)}">#</a></div>
      <p class="prompt" lang="${escape(sample.language)}">${escape(sample.prompt)}</p>
      <details class="reference-text"><summary>Reference transcript</summary><p>${escape(sample.reference_text)}</p></details></div>
      <div class="players">${columns.map(column => player(sample,column)).join("")}</div>
    </article>`;
  }).join("");
  const audios = [...root.querySelectorAll("audio")];
  audios.forEach(audio => {
    audio.addEventListener("play", () => audios.forEach(other => { if (other !== audio && !other.paused) other.pause(); }));
    audio.addEventListener("error", () => {
      if (audio.parentElement.querySelector(".audio-error")) return;
      const message = document.createElement("p");
      message.className = "audio-error";
      message.textContent = "Audio could not load. Please refresh to try again.";
      message.setAttribute("role", "status");
      audio.after(message);
    });
  });
  const cards = [...root.querySelectorAll(".sample-card")];
  const filters = [...document.querySelectorAll(".filters button")];
  const jump = document.getElementById("sample-jump");
  const count = document.getElementById("result-count");
  function filter(language) {
    filters.forEach(button => button.setAttribute("aria-pressed", String(button.dataset.language === language)));
    cards.forEach(card => {
      card.hidden = language !== "all" && card.dataset.language !== language;
      if (card.hidden) card.querySelectorAll("audio").forEach(audio => audio.pause());
    });
    const visible = data.samples.filter(sample => language === "all" || sample.language === language);
    jump.innerHTML = '<option value="">an example</option>' + visible.map(sample => `<option value="${escape(sample.id)}">${escape(sample.label)} · ${sample.language === "zh" ? "Chinese" : "English"}</option>`).join("");
    count.textContent = language === "all" ? `Showing all ${visible.length} examples` : `Showing ${visible.length} ${language === "zh" ? "Chinese" : "English"} examples`;
  }
  filters.forEach(button => {
    const language = button.dataset.language;
    button.querySelector("span").textContent = data.samples.filter(sample => language === "all" || sample.language === language).length;
    button.addEventListener("click", () => filter(language));
  });
  jump.addEventListener("change", () => { if (jump.value) location.hash = jump.value; });
  const showHash = () => {
    const id = location.hash.slice(1);
    const target = cards.find(card => card.id === id);
    if (target) {
      if (target.hidden) filter("all");
      target.scrollIntoView();
    }
  };
  filter("all");
  window.addEventListener("hashchange", showHash);
  if (location.hash) requestAnimationFrame(showHash);
})();
