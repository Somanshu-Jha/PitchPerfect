# =====================================================================
# ULTIMATE NEURAL COMMAND CENTER v8.0 (Zero-Failure Edition)
# =====================================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Dict
import multiprocessing as mp

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ── The 3-Color Protocol ───────────────────────────────────────────
_BG_NAVY      = "#060714"
_BG_PANEL     = "#0c0d21"
_GRID_COLOR   = "#1b1e36"
_GLOW_CYAN    = "#00f2ff"  # [SYSTEM]
_GLOW_MAGENTA = "#ff00ff"  # [LEARNING]
_GLOW_GOLD    = "#ffd700"  # [PRECISION]
_TEXT_DIM     = "#4a5568"
_TEXT_BRIGHT  = "#cbd5e0"

class TrainingVisualizer:
    def __init__(self, title: str = "Neural Command Center", head_names: Optional[List[str]] = None):
        self.title = title
        self.head_names = head_names or ["Clarity", "Completeness", "Structure", "Confidence", "Tech_Depth", "Overall"]
        
        self.epochs = []
        self.losses = []
        self.lrs = []
        self.best_loss = float("inf")
        self.head_maes = {n: [] for n in self.head_names}
        
        self.last_preds = None
        self.last_targets = None
        self.last_embeddings = None
        self.cached_topology = {"names": [], "mags": []}
        self.cached_weights = {"w": [0], "b": [0]}

        self._closed = False
        self._init_window()

    def _init_window(self):
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry("1550x980")
        self.root.configure(bg=_BG_NAVY)
        
        # --- HEADER HUD ---
        hud = tk.Frame(self.root, bg=_BG_NAVY, pady=10)
        hud.pack(fill=tk.X)
        tk.Label(hud, text=self.title.upper(), font=("monospace", 26, "bold"), fg=_GLOW_CYAN, bg=_BG_NAVY).pack()
        
        legend_frame = tk.Frame(hud, bg=_BG_NAVY)
        legend_frame.pack()
        l_items = [("💎 CYAN", "Data Flow", _GLOW_CYAN), ("🌸 MAGENTA", "Training", _GLOW_MAGENTA), ("🟡 GOLD", "Accuracy", _GLOW_GOLD)]
        for icon, desc, color in l_items:
            tk.Label(legend_frame, text=f"{icon}: {desc} | ", font=("monospace", 9), fg=color, bg=_BG_NAVY).pack(side=tk.LEFT)

        self.stats_label = tk.Label(hud, text="SYNCING CORE...", font=("monospace", 10), fg=_TEXT_BRIGHT, bg=_BG_NAVY)
        self.stats_label.pack()

        # --- SCROLLABLE MAINFRAME ---
        container = tk.Frame(self.root, bg=_BG_NAVY)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, bg=_BG_NAVY, highlightthickness=0)
        v_scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=_BG_NAVY)

        self.scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw", width=1530)
        canvas.configure(yscrollcommand=v_scroll.set)

        canvas.pack(side="left", fill="both", expand=True)
        v_scroll.pack(side="right", fill="y")

        # Matplotlib Grid
        self.fig = Figure(figsize=(15, 18), facecolor=_BG_NAVY, dpi=100)
        self.fig.subplots_adjust(hspace=0.45, wspace=0.3, left=0.08, right=0.95, top=0.95, bottom=0.05)
        
        self.ax_loss  = self.fig.add_subplot(4, 2, 1)
        self.ax_brain = self.fig.add_subplot(4, 2, 2)
        self.ax_lr    = self.fig.add_subplot(4, 2, 3)
        self.ax_whist = self.fig.add_subplot(4, 2, 4)
        self.ax_mae   = self.fig.add_subplot(4, 2, 5)
        self.ax_3d    = self.fig.add_subplot(4, 2, 6, projection='3d')
        self.ax_r2    = self.fig.add_subplot(4, 2, (7, 8))

        for ax in [self.ax_loss, self.ax_lr, self.ax_mae, self.ax_r2, self.ax_whist]:
            ax.set_facecolor(_BG_PANEL)
            ax.tick_params(colors=_TEXT_DIM, labelsize=7)
            for spine in ax.spines.values(): spine.set_color(_GRID_COLOR)
        self.ax_3d.set_facecolor(_BG_NAVY)

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.scroll_frame)
        self.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._closed = True
        self.root.destroy()

    def _draw_neural_flux(self):
        ax = self.ax_brain
        ax.clear()
        ax.set_facecolor(_BG_NAVY)
        ax.set_title("NEURAL FLUX ENGINE (TRAINING ACTIVE)", color=_GLOW_CYAN, fontsize=10, fontweight="bold")
        ax.axis("off")

        layers = [8, 12, 12, 6]
        x_space = [0.1, 0.4, 0.6, 0.9]
        mags = self.cached_topology.get("mags", [0.01]*10)
        avg_mag = np.mean(mags) if mags else 0.01

        # Connections - Pulse Flow (FIXED THICKNESS)
        for i in range(len(layers)-1):
            x1, x2 = x_space[i], x_space[i+1]
            y1s, y2s = np.linspace(0.1, 0.9, layers[i]), np.linspace(0.1, 0.9, layers[i+1])
            for y1 in y1s:
                for y2 in y2s:
                    # Pulsing logic with robust baseline
                    pulse = (np.sin(time.time() * 4 + x1 * 8) + 1) / 2
                    alpha = 0.12 + (avg_mag * 8.0) + (pulse * 0.08)
                    lw = 1.2 + (avg_mag * 10)
                    color = _GLOW_MAGENTA if (avg_mag > 0.005 or pulse > 0.92) else _GLOW_CYAN
                    ax.plot([x1, x2], [y1, y2], color=color, alpha=min(alpha, 0.35), lw=lw)

        # Neurons with PERMANENT LABELS
        input_labels = ["Audio", "Tone", "Fluency", "WPM", "Grammar", "Depth"]
        output_labels = self.head_names

        for i, count in enumerate(layers):
            x, ys = x_space[i], np.linspace(0.1, 0.9, count)
            color = _GLOW_GOLD if i==0 else (_GLOW_MAGENTA if i==len(layers)-1 else _GLOW_CYAN)
            ax.scatter([x]*count, ys, s=55, color=color, edgecolors="white", lw=1, zorder=6)
            
            # Add labels for Input and Output layers
            if i == 0:
                for j, y in enumerate(ys[:len(input_labels)]):
                    ax.text(x-0.03, y, input_labels[j], color=_TEXT_BRIGHT, ha="right", va="center", fontsize=8, fontfamily="monospace")
            elif i == len(layers)-1:
                for j, y in enumerate(ys[:len(output_labels)]):
                    ax.text(x+0.03, y, output_labels[j], color=_TEXT_BRIGHT, ha="left", va="center", fontsize=8, fontfamily="monospace")

    def update_ui(self, data: Dict):
        if self._closed: return
        
        # Update shared state
        self.epochs.append(data["epoch"])
        self.losses.append(data["loss"])
        self.lrs.append(data.get("lr", 0))
        if data["loss"] < self.best_loss: self.best_loss = data["loss"]
        if "topology" in data: self.cached_topology = data["topology"]
        if "weights" in data: self.cached_weights = data["weights"]
        if "embeddings" in data: self.last_embeddings = np.array(data["embeddings"])
        if "preds" in data: self.last_preds, self.last_targets = np.array(data["preds"]), np.array(data["targets"])

        # 1. BRAIN ENGINE (Most frequent)
        try: self._draw_neural_flux()
        except: pass

        # 2. LOSS / LR
        try:
            self.ax_loss.clear()
            self.ax_loss.set_title("LOSS_CONVERGENCE", color=_GLOW_CYAN, loc="left", fontsize=9)
            self.ax_loss.plot(self.epochs, self.losses, color=_GLOW_CYAN, lw=2)
            self.ax_loss.grid(color=_GRID_COLOR, alpha=0.3)
            
            self.ax_lr.clear()
            self.ax_lr.set_title("LEARN_SIGNAL_LR", color=_GLOW_GOLD, loc="left", fontsize=9)
            self.ax_lr.plot(self.epochs, self.lrs, color=_GLOW_GOLD, lw=2)
        except: pass

        # 3. WEIGHTS / MAE
        try:
            self.ax_whist.clear()
            self.ax_whist.hist(self.cached_weights.get("w", [0]), bins=30, color=_GLOW_MAGENTA, alpha=0.6)
            self.ax_whist.set_title("PARAM_DISTRIBUTION", color=_GLOW_MAGENTA, fontsize=9)

            if "maes" in data:
                self.ax_mae.clear()
                self.ax_mae.set_title("PER_CATEGORY_ERROR", color=_GLOW_GOLD, fontsize=9)
                colors = ["#5b9bd5", "#4ec9b0", "#e5c07b", "#e06c75", "#c678dd", "#d19a66"]
                for i, (name, vals) in enumerate(data["maes"].items()):
                    if name not in self.head_maes: self.head_maes[name] = []
                    self.head_maes[name].append(vals)
                    self.ax_mae.plot(range(len(self.head_maes[name])), self.head_maes[name], label=name, color=colors[i%6], lw=1)
                if len(self.epochs) < 5: self.ax_mae.legend(fontsize=6, facecolor=_BG_NAVY, labelcolor="white")
        except: pass

        # 4. 3D VECTOR HUB
        try:
            if self.last_embeddings is not None and len(self.last_embeddings) >= 3:
                from sklearn.decomposition import PCA
                self.ax_3d.clear()
                self.ax_3d.set_title("3D NEURAL VECTOR HUB (150 SAMPLES)", color=_GLOW_CYAN, fontsize=10)
                emb_pca = PCA(n_components=3).fit_transform(self.last_embeddings[:150])
                # Brighter map 'spring' for visibility
                c_vals = self.last_targets[:150, -1] if (self.last_targets is not None and len(self.last_targets.shape)>1) else None
                self.ax_3d.scatter(emb_pca[:,0], emb_pca[:,1], emb_pca[:,2], c=c_vals, cmap="spring", s=35, alpha=0.9, edgecolors="white", lw=0.3)
                self.ax_3d.view_init(elev=20, azim=45)
        except: pass

        # 5. R2 PRECISION
        try:
            if self.last_preds is not None:
                self.ax_r2.clear()
                self.ax_r2.set_title("R² PRECISION CORE (0-10 RANGE)", color=_GLOW_GOLD, fontsize=9)
                self.ax_r2.scatter(self.last_targets.flatten(), self.last_preds.flatten(), c=_GLOW_MAGENTA, s=5, alpha=0.5)
                self.ax_r2.plot([0, 10], [0, 10], "--", color=_TEXT_DIM, lw=1)
                self.ax_r2.set_xlim(0, 10); self.ax_r2.set_ylim(0, 10)
        except: pass

        # DRAW Once per update cycle
        try:
            status = "💠 CORE ACTIVE" if (time.time() % 1.0 > 0.5) else "🌀 CORE ACTIVE"
            self.stats_label.config(text=f"{status} | EPOCH: {data['epoch']+1} | LOSS: {data['loss']:.5f} | BEST: {self.best_loss:.5f}")
            self.canvas_widget.draw()
            self.root.update()
        except: pass

def ui_entry_point(queue: mp.Queue):
    viz = TrainingVisualizer()
    while not viz._closed:
        try:
            # Process all pending signals but only redraw once
            last_pulse = None
            while not queue.empty():
                last_pulse = queue.get_nowait()
            
            if last_pulse:
                viz.update_ui(last_pulse)
            else:
                # Still need to tick the UI for "Always-On" animations
                viz._draw_neural_flux()
                viz.canvas_widget.draw()
                viz.root.update()
            
            time.sleep(0.02)
        except: break

if __name__ == "__main__":
    q = mp.Queue()
    p = mp.Process(target=ui_entry_point, args=(q,))
    p.start()
    for i in range(100):
        q.put({"epoch": i, "loss": 0.1/(i+1), "lr": 0.001, "weights": {"w": np.random.randn(1000)}})
        time.sleep(0.5)
    p.join()
