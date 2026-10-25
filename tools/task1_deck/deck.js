/* === SLIDE PRESENTATION ===
   Fixed 1920x1080 stage scaled to the window; keyboard, wheel, swipe and
   #N hash navigation; .active toggles visibility (never display:none). */
class SlidePresentation {
    constructor() {
        this.stage = document.querySelector(".deck-stage");
        this.slides = [...document.querySelectorAll(".slide")];
        this.progress = document.querySelector(".progress");
        this.index = this.fromHash();
        this.locked = false;
        this.fit = this.fit.bind(this);
        window.addEventListener("resize", this.fit);
        window.addEventListener("hashchange", () => this.show(this.fromHash(), false));
        document.addEventListener("keydown", (e) => this.onKey(e));
        window.addEventListener("wheel", (e) => this.onWheel(e), { passive: true });
        this.bindTouch();
        this.fit();
        this.show(this.index, false);
    }

    fromHash() {
        const n = parseInt(location.hash.slice(1), 10);
        return Number.isFinite(n) ? Math.min(Math.max(n - 1, 0), this.slides.length - 1) : 0;
    }

    fit() {
        const s = Math.min(innerWidth / 1920, innerHeight / 1080);
        const x = (innerWidth - 1920 * s) / 2;
        const y = (innerHeight - 1080 * s) / 2;
        this.stage.style.transform = `translate(${x}px, ${y}px) scale(${s})`;
    }

    show(i, push = true) {
        this.index = Math.min(Math.max(i, 0), this.slides.length - 1);
        this.slides.forEach((el, k) => {
            const on = k === this.index;
            el.classList.toggle("active", on);
            el.classList.toggle("visible", on);
            el.setAttribute("aria-hidden", on ? "false" : "true");
        });
        this.progress.style.width = `${((this.index + 1) / this.slides.length) * 100}%`;
        if (push) history.replaceState(null, "", `#${this.index + 1}`);
    }

    onKey(e) {
        if (document.body.classList.contains("editing")) return;
        const next = ["ArrowRight", "ArrowDown", " ", "PageDown"];
        const prev = ["ArrowLeft", "ArrowUp", "PageUp"];
        if (next.includes(e.key)) { e.preventDefault(); this.show(this.index + 1); }
        else if (prev.includes(e.key)) { e.preventDefault(); this.show(this.index - 1); }
        else if (e.key === "Home") this.show(0);
        else if (e.key === "End") this.show(this.slides.length - 1);
    }

    onWheel(e) {
        if (this.locked || Math.abs(e.deltaY) < 30 || document.body.classList.contains("editing")) return;
        this.locked = true;
        this.show(this.index + (e.deltaY > 0 ? 1 : -1));
        setTimeout(() => { this.locked = false; }, 700);
    }

    bindTouch() {
        let x0 = null;
        document.addEventListener("touchstart", (e) => { x0 = e.touches[0].clientX; }, { passive: true });
        document.addEventListener("touchend", (e) => {
            if (x0 === null) return;
            const dx = e.changedTouches[0].clientX - x0;
            if (Math.abs(dx) > 50) this.show(this.index + (dx < 0 ? 1 : -1));
            x0 = null;
        });
    }
}

/* === INLINE EDITOR ===
   Hover the top-left corner or press E; edits auto-save to this browser only.
   Saved edits are ignored if the slide count changed since they were saved. */
class InlineEditor {
    constructor(key) {
        this.key = key;
        this.button = document.getElementById("editToggle");
        this.zone = document.querySelector(".edit-hotzone");
        this.targets = [...document.querySelectorAll(".slide h1, .slide h2, .slide .lede, .slide .note")];
        this.hideTimer = null;
        this.restore();
        this.zone.addEventListener("mouseenter", () => this.reveal());
        this.zone.addEventListener("mouseleave", () => this.hideSoon());
        this.button.addEventListener("mouseenter", () => this.reveal());
        this.button.addEventListener("mouseleave", () => this.hideSoon());
        this.button.addEventListener("click", () => this.toggle());
        document.addEventListener("keydown", (e) => this.onKey(e));
    }

    reveal() { clearTimeout(this.hideTimer); this.button.classList.add("show"); }
    hideSoon() { this.hideTimer = setTimeout(() => this.button.classList.remove("show"), 400); }

    toggle() {
        const on = !document.body.classList.contains("editing");
        document.body.classList.toggle("editing", on);
        this.button.classList.toggle("on", on);
        this.button.textContent = on ? "Editing · E to stop" : "Edit text";
        this.targets.forEach((el) => { el.contentEditable = on ? "true" : "false"; });
        if (!on) this.save();
    }

    onKey(e) {
        const typing = document.activeElement && document.activeElement.isContentEditable;
        if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); this.save(); return; }
        if (!typing && (e.key === "e" || e.key === "E")) this.toggle();
    }

    save() {
        try {
            const data = { count: this.targets.length, html: this.targets.map((el) => el.innerHTML) };
            localStorage.setItem(this.key, JSON.stringify(data));
        } catch (err) { console.warn("edit save failed", err); }
    }

    restore() {
        try {
            const data = JSON.parse(localStorage.getItem(this.key) || "null");
            if (!data || data.count !== this.targets.length) return;
            this.targets.forEach((el, i) => { el.innerHTML = data.html[i]; });
        } catch (err) { console.warn("edit restore failed", err); }
    }
}

/* === PROVENANCE TOOLTIPS ===
   Each sourced number carries the printed notebook line as its title. */
document.querySelectorAll("[data-line]").forEach((el) => {
    el.title = `intro_track.ipynb cell ${el.dataset.cell}: ${el.dataset.line}`;
});

new SlidePresentation();
new InlineEditor("task1-deck-edits-v3");
