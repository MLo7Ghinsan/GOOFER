import os
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import GOOFER as gf

class SillyEditor:
    def __init__(self, parent, y, sr, init_mask=None, title="Voicing Editor"):
        self.parent = parent
        self.y = np.asarray(y, dtype=np.float32)
        self.sr = int(sr)
        self.N = len(self.y)

        # per sample mask (default voiced)
        if init_mask is None or len(init_mask) != self.N:
            self.mask_full = np.ones(self.N, dtype=np.float32)
        else:
            self.mask_full = np.asarray(init_mask, dtype=np.float32)

        self.zoom = 1.0
        self.scroll_pos = 0.0
        self.ok = False
        self._painting = None
        self.playing = False
        self._play_stream = None
        self._play_job = None 
        self.f0_full = None

        self.edit_mode = tk.StringVar(value="both")  # both | voiced | unvoiced

        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.geometry("970x380")
        self.win.minsize(820, 320)
        self.win.protocol("WM_DELETE_WINDOW", self._cancel)

        main_frame = ttk.Frame(self.win)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left canvas section
        left = ttk.Frame(main_frame)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.canvas = tk.Canvas(left, bg="#0b0b0b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # scrollbar
        style = ttk.Style()
        try:
            style.theme_use("default")
        except Exception:
            pass
        style.configure(
            "Horizontal.TScrollbar",
            troughcolor="#1a1a1a",
            background="#00bfff",
            bordercolor="#333333",
            arrowcolor="#ffffff",
            troughrelief="flat",
            relief="flat",
        )
        self.scrollbar = ttk.Scrollbar(
            left,
            orient=tk.HORIZONTAL,
            command=self._scrollbar_move,
            style="Horizontal.TScrollbar",
        )
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 2))
        self.scrollbar.set(0, 1)

        # zoom bar
        zoom_frame = ttk.Frame(left)
        zoom_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(zoom_frame, text="Zoom").pack(side=tk.LEFT)
        self.zoom_slider = ttk.Scale(zoom_frame, from_=1, to=20, orient=tk.HORIZONTAL, command=self._zoom_changed)
        self.zoom_slider.set(1)
        self.zoom_slider.pack(fill=tk.X, padx=6, expand=True)

        # tools shits
        right = ttk.Frame(main_frame)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        ttk.Button(right, text="Play", command=self._play_audio).pack(fill=tk.X, pady=(0, 4))
        ttk.Button(right, text="Stop", command=self._stop_audio).pack(fill=tk.X, pady=(0, 12))
        ttk.Label(right, text="").pack(pady=4)  # spacer
        ttk.Button(right, text="Apply", command=self._ok).pack(fill=tk.X, pady=(0, 12))
        ttk.Label(right, text="").pack(pady=4)  # spacer
        ttk.Button(right, text="Cancel", command=self._cancel).pack(fill=tk.X)
        ttk.Label(right, text="").pack(pady=8)
        mode_frame = ttk.Frame(right); mode_frame.pack(fill=tk.X)
        ttk.Label(mode_frame, text="Editing:").pack(side=tk.LEFT)
        self.mode_combo = ttk.Combobox(
            mode_frame, textvariable=self.edit_mode,
            values=["both", "voiced", "unvoiced"], state="readonly", width=12
        )
        self.mode_combo.pack(side=tk.LEFT, padx=6)
        self.mode_combo.configure(takefocus=False)
        self.mode_combo.bind("<FocusIn>", lambda e: e.widget.selection_clear())
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _e: (self._rebind_canvas(), self._draw()))

        f0_frame = ttk.Frame(right); f0_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(f0_frame, text="F0 brush (Hz)").pack(side=tk.LEFT)

        self.f0_brush = tk.DoubleVar(value=120.0)
        self.f0_value_lbl = ttk.Label(f0_frame, text="120 Hz")
        self.f0_value_lbl.pack(side=tk.RIGHT)

        def _fmt_f0(v):
            try:
                return f"{int(round(float(v)))} Hz"
            except Exception:
                return "- Hz"

        def _on_f0_changed(v):
            self.f0_value_lbl.config(text=_fmt_f0(v))

        self.f0_slider = ttk.Scale(
            right, from_=50, to=500, orient=tk.HORIZONTAL,
            variable=self.f0_brush, command=_on_f0_changed
        )
        _on_f0_changed(self.f0_brush.get())
        self.f0_slider.set(120)
        self.f0_slider.pack(fill=tk.X, padx=6, pady=(4, 0))

        self.f0_slider.bind("<ButtonRelease-1>", lambda _e: self._apply_f0_after_slider())
        self.f0_slider.bind("<ButtonRelease-2>", lambda _e: self._apply_f0_after_slider())
        self.f0_slider.bind("<ButtonRelease-3>", lambda _e: self._apply_f0_after_slider())
        self.f0_slider.bind("<KeyRelease>", lambda _e: self._apply_f0_after_slider())

        # keyboard mode swaps (1=both, 2=voiced, 3=unvoiced)
        self.win.bind("1", lambda _e: (self.edit_mode.set("both"), self._rebind_canvas(), self._draw()))
        self.win.bind("2", lambda _e: (self.edit_mode.set("voiced"), self._rebind_canvas(), self._draw()))
        self.win.bind("3", lambda _e: (self.edit_mode.set("unvoiced"), self._rebind_canvas(), self._draw()))

        self._set_title()

        self._update_view()
        self.wave_rect = (10, 10, 740, 210)

        self._rebind_canvas()
        self.canvas.bind("<Configure>", self._on_resize)

        self._draw()

    def _apply_f0_after_slider(self):
        if self.f0_full is None:
            return
        a, b = 0, self.N # whole file
        if b <= a:
            return
        f0v = float(self.f0_brush.get())
        if f0v < 50.0: f0v = 50.0
        elif f0v > 500.0: f0v = 500.0

        voiced = self.mask_full[a:b] > 0.5
        seg = self.f0_full[a:b].copy()
        seg[voiced] = f0v
        seg[~voiced] = 0.0
        self.f0_full[a:b] = seg
        self._draw()
        
    # playback
    def _synthesize_preview(self):
        try:
            # unpack
            env0, f0i0, vmask0, forms0, sr0, ylen0 = self.features
            env_dense = gf.decode_env_from_knots(env0) if isinstance(env0, dict) and env0.get("mode") == "knots" else env0

            # visible region (samples)
            start = int(self.start_sample)
            end = int(self.end_sample)
            if end <= start:
                return np.zeros(1, dtype=np.float32)

            # analysis (might change, might not, idk)
            hop_length = 256
            n_fft = 1024

            # per-sample segments (copy)
            vmask_seg = np.array(self.mask_full[start:end], dtype=np.float32, copy=True)
            src_f0 = self.f0_full if (self.f0_full is not None) else f0i0
            f0_seg = np.array(src_f0[start:end], dtype=np.float32, copy=True)

            # If painted voicing (mask=1) where f0==0, fill F0 by nearest/connected values
            need_fill = (vmask_seg > 0.5) & (f0_seg <= 0.0)
            if np.any(need_fill):
                idx = np.arange(len(f0_seg))
                known = f0_seg > 0.0
                if np.any(known):
                    interp = np.interp(
                        idx, idx[known], f0_seg[known],
                        left=float(f0_seg[known][0]),
                        right=float(f0_seg[known][-1])
                    ).astype(np.float32)
                else:
                    # no F0 in the visible segment use nearest global voiced F0 or fall back
                    global_known = f0i0 > 0.0
                    if np.any(global_known):
                        voiced_idx = np.where(global_known)[0]
                        mid = (start + end) // 2
                        nearest_i = voiced_idx[np.argmin(np.abs(voiced_idx - mid))]
                        base = float(f0i0[nearest_i])
                    else:
                        base = 120.0 # 120hz when no F0
                    interp = np.full(len(f0_seg), base, dtype=np.float32)
                f0_seg[need_fill] = interp[need_fill]

            # frame range
            start_f = start // hop_length
            end_f = int(np.ceil(end / hop_length))
            end_f = max(start_f + 1, end_f)

            env_seg = env_dense[:, start_f:end_f]
            forms_seg = {}
            if isinstance(forms0, dict):
                for k, track in forms0.items():
                    t = np.asarray(track, dtype=np.float32)
                    forms_seg[k] = t[start_f:end_f]
            else:
                forms_seg = forms0

            y_len_bool = np.empty(len(vmask_seg), dtype=np.bool_)
            _, harmonic, aper_uv, aper_bre = gf.synthesize(
                env_seg, f0_seg, vmask_seg, y_len_bool, sr0,
                n_fft=n_fft, hop_length=hop_length, formants=forms_seg
            )
            return (harmonic + aper_uv + aper_bre) * 0.5

        except Exception as e:
            logging.warning(f"[PLAYBACK] synth error: {e}")
            return np.zeros(44100, dtype=np.float32)

    def _play_audio(self):
        if self.playing:
            self._stop_audio()

        try:
            y_play = self._synthesize_preview().astype(np.float32, copy=False)
            if y_play.size == 0:
                return

            sr0 = int(self.features[4]) if hasattr(self, "features") else self.sr

            sd.stop()
            sd.play(y_play, sr0)
            self.playing = True

            if self._play_job is not None:
                try:
                    self.win.after_cancel(self._play_job)
                except Exception:
                    pass
            dur_ms = int(1000 * len(y_play) / sr0)
            self._play_job = self.win.after(dur_ms, self._on_play_finished)

        except Exception as e:
            logging.warning(f"[PLAYBACK] Failed to play: {e}")

    def _on_play_finished(self):
        self.playing = False
        self._play_job = None

    def _stop_audio(self):
        try:
            sd.stop()
        except Exception:
            pass
        self.playing = False
        if self._play_job is not None:
            try:
                self.win.after_cancel(self._play_job)
            except Exception:
                pass
            self._play_job = None

    def _update_view(self):
        visible_samples = int(self.N / self.zoom)
        visible_samples = max(200, min(self.N, visible_samples))
        start = int(self.scroll_pos * (self.N - visible_samples))
        start = max(0, min(start, self.N - visible_samples))
        end = min(self.N, start + visible_samples)

        self.start_sample = start
        self.end_sample = end
        self.view_idx = np.arange(start, end, max(1, int((end - start) / 800)), dtype=np.int32)
        if len(self.view_idx) == 0 or self.view_idx[-1] != end - 1:
            self.view_idx = np.append(self.view_idx, end - 1)
        self.view_mask = self.mask_full[self.view_idx].copy()
        self.wave_x, self.wave_y = self._make_wave_points(self.y[start:end], width_px=800, height_px=120)

        page_fraction = visible_samples / self.N
        start_frac = self.scroll_pos
        end_frac = min(1.0, start_frac + page_fraction)
        self.scrollbar.set(start_frac, end_frac)

    def _scrollbar_move(self, *args):
        if args[0] == "moveto":
            self.scroll_pos = float(args[1])
        elif args[0] == "scroll":
            delta = int(args[1])
            self.scroll_pos += delta * 0.05 / self.zoom
        self.scroll_pos = np.clip(self.scroll_pos, 0.0, 1.0)
        self._update_view()
        self._draw()

    def _zoom_changed(self, value):
        self.zoom = float(value)
        self.scroll_pos = np.clip(self.scroll_pos, 0.0, 1.0)
        self._update_view()
        self._draw()

    def _on_resize(self, _event):
        self._draw()

    def _make_wave_points(self, y, width_px=800, height_px=120):
        N = len(y)
        if N == 0:
            return np.array([0, 1], dtype=np.float32), np.array([height_px / 2] * 2, dtype=np.float32)
        max_abs = float(np.max(np.abs(y))) or 1.0
        yn = y / max_abs
        ds = max(1, N // width_px)
        idx = np.arange(0, N, ds)
        y_ds = yn[idx]
        y_plot = (0.5 - 0.45 * y_ds) * height_px
        x_plot = np.linspace(0, width_px - 1, num=len(y_ds))
        return x_plot.astype(np.float32), y_plot.astype(np.float32)

    def _x_to_view_i(self, x_canvas):
        x0, _, x1, _ = self.wave_rect
        W = max(1, int(x1 - x0))
        x = np.clip(x_canvas - x0, 0, W)
        frac = x / W
        return int(round(frac * (len(self.view_idx) - 1)))

    def _apply_paint(self, x0, x1, voiced=True):
        i0 = self._x_to_view_i(min(x0, x1))
        i1 = self._x_to_view_i(max(x0, x1))
        val = 1.0 if voiced else 0.0
        self.view_mask[i0:i1+1] = val
        a = int(self.view_idx[i0])
        b = int(self.view_idx[i1]) + 1
        self.mask_full[a:b] = val
        if self.f0_full is not None:
            if voiced:
                self.f0_full[a:b] = float(self.f0_brush.get())
            else:
                self.f0_full[a:b] = 0.0
        self._draw()

    def _begin_voiced(self, e):
        self._painting = ("voiced", e.x)
        self._apply_paint(e.x, e.x, True)

    def _begin_unvoiced(self, e):
        self._painting = ("unvoiced", e.x)
        self._apply_paint(e.x, e.x, False)

    def _paint_motion(self, e):
        if not self._painting:
            return
        mode, start_x = self._painting
        self._apply_paint(start_x, e.x, voiced=(mode == "voiced"))

    def _end_paint(self, _):
        self._painting = None

    def _reset(self):
        self.mask_full[:] = 1.0
        self._update_view()
        self._draw()

    def _ok(self):
        self.ok = True
        self._stop_audio()
        self.win.destroy()

    def _cancel(self):
        self.ok = False
        self._stop_audio()
        self.win.destroy()

    def _draw(self):
        c = self.canvas
        c.delete("all")
        width = c.winfo_width() or 800
        height = c.winfo_height() or 220
        self.wave_rect = (10, 10, width - 10, height - 10)
        x0, y0, x1, y1 = self.wave_rect
        W = int(x1 - x0)
        H = int(y1 - y0)
        c.create_rectangle(x0, y0, x1, y1, outline="#333")

        for i in range(len(self.view_idx) - 1):
            frac0 = i / (len(self.view_idx) - 1)
            frac1 = (i + 1) / (len(self.view_idx) - 1)
            xx0 = x0 + frac0 * W
            xx1 = x0 + frac1 * W
            val = self.view_mask[i]
            color = "#00bfff" if val > 0.5 else "#333333"
            c.create_rectangle(xx0, y0, xx1, y1, outline="", fill=color)

        wx = x0 + (self.wave_x / (self.wave_x.max() + 1e-9)) * (W - 1)
        wy = y0 + (self.wave_y / (self.wave_y.max() + 1e-9)) * (H - 1)
        pts = []
        for xi, yi in zip(wx, wy):
            pts.extend([xi, yi])
        if len(pts) >= 4:
            c.create_line(*pts, fill="#e6f7ff", width=1)

        sec_start = self.start_sample / self.sr
        sec_end = self.end_sample / self.sr
        hint = ""
        if self.edit_mode.get() == "both":
            hint = "LMB=voiced | RMB=unvoiced"
        else:
            hint = f"Editing: {self.edit_mode.get()} (any click/drag)"

        c.create_text(
            x0 + 8, y0 + 12, anchor="w",
            text=f"{hint} | {sec_start:.2f}s–{sec_end:.2f}s | Zoom={self.zoom:.1f}x",
            fill="#ffffff"
        )

    def _init_tracks(self):
        try:
            _, f0i0, _, _, _, ylen0 = self.features
            target = int(ylen0)
            f0i0 = np.asarray(f0i0, dtype=np.float32)
            if f0i0.shape[0] != target:
                if f0i0.shape[0] > target:
                    f0i0 = f0i0[:target]
                else:
                    f0i0 = np.pad(f0i0, (0, target - f0i0.shape[0]), mode="edge")
            self.f0_full = f0i0.copy()

            f0v = float(self.f0_brush.get())
            if f0v < 50.0: f0v = 50.0
            elif f0v > 500.0: f0v = 500.0

            voiced = (self.mask_full[:target] > 0.5)
            self.f0_full[:target][voiced] = f0v
            self.f0_full[:target][~voiced] = 0.0
        except Exception:
            self.f0_full = None

    def _begin_single(self, e, voiced: bool):
        self._painting = ("voiced" if voiced else "unvoiced", e.x)
        self._apply_paint(e.x, e.x, voiced)

    def _set_title(self):
        self.win.title(f"{self.win.title().split(' - ')[0]} - {self.edit_mode.get().title()}")

    def _rebind_canvas(self):
        # clear old stuff...
        for seq in ("<Button-1>","<B1-Motion>","<ButtonRelease-1>",
                    "<Button-2>","<B2-Motion>","<ButtonRelease-2>",
                    "<Button-3>","<B3-Motion>","<ButtonRelease-3>"):
            self.canvas.unbind(seq)

        mode = self.edit_mode.get()
        if mode == "both":
            # LMB = voiced
            self.canvas.bind("<Button-1>", self._begin_voiced)
            self.canvas.bind("<B1-Motion>", self._paint_motion)
            self.canvas.bind("<ButtonRelease-1>", self._end_paint)
            # RMB = unvoiced
            self.canvas.bind("<Button-3>", self._begin_unvoiced)
            self.canvas.bind("<B3-Motion>", self._paint_motion)
            self.canvas.bind("<ButtonRelease-3>", self._end_paint)
            self.canvas.bind("<Button-2>", self._begin_unvoiced)
            self.canvas.bind("<B2-Motion>", self._paint_motion)
            self.canvas.bind("<ButtonRelease-2>", self._end_paint)

        elif mode == "voiced":
            for b in ("1","2","3"):
                self.canvas.bind(f"<Button-{b}>", lambda e: self._begin_single(e, True))
                self.canvas.bind(f"<B{b}-Motion>", self._paint_motion)
                self.canvas.bind(f"<ButtonRelease-{b}>", self._end_paint)

        else: # unvoiced
            for b in ("1","2","3"):
                self.canvas.bind(f"<Button-{b}>", lambda e: self._begin_single(e, False))
                self.canvas.bind(f"<B{b}-Motion>", self._paint_motion)
                self.canvas.bind(f"<ButtonRelease-{b}>", self._end_paint)

        self._set_title()

def interactive_voicing(y_snippet, sr, init_mask=None, title="Voicing Editor"):
    root = tk.Tk()
    root.withdraw()
    ui = SillyEditor(root, y_snippet, sr, init_mask=init_mask, title=title)
    root.wait_window(ui.win)
    out = ui.mask_full.astype(np.float32) if ui.ok else None
    try:
        root.destroy()
    except Exception:
        pass
    return out

AUDIO_EXTS = [".wav", ".flac", ".aiff", ".aif", ".mp3"]

def write_back_voicing_to_goofy(
    feat_path: str,
    edited_mask_snippet: np.ndarray,
    start_sample: int,
    end_sample: int,
    snippet_was_reversed: bool,
    total_len: int,
):
    env0, f0i0, vmask0, forms0, sr0, ylen0 = gf.load_features(feat_path)
    total_len = int(ylen0)

    a = max(0, min(int(start_sample), total_len))
    b = max(a, min(int(end_sample), total_len))

    if snippet_was_reversed:
        a_orig = total_len - b
        b_orig = total_len - a
        edited_local = edited_mask_snippet[::-1].astype(np.float32, copy=False)
    else:
        a_orig, b_orig = a, b
        edited_local = edited_mask_snippet.astype(np.float32, copy=False)

    span = b_orig - a_orig
    if span <= 0:
        return
    if edited_local.shape[0] != span:
        if edited_local.shape[0] > span:
            edited_local = edited_local[:span]
        else:
            edited_local = np.pad(edited_local, (0, span - edited_local.shape[0]), mode="edge")

    vmask_new = np.array(vmask0, dtype=np.float32, copy=True)
    vmask_new[a_orig:b_orig] = edited_local

    tmp_path = feat_path + ".tmp"
    gf.save_features(tmp_path, env0, f0i0, vmask_new, forms0, sr0, total_len)
    os.replace(tmp_path, feat_path)


def _find_neighbor_audio_for_goofy(goofy_path: Path):
    # features names usually like "{wav}_features.goofy"
    name = goofy_path.name
    base = name[:-len("_features.goofy")] if name.endswith("_features.goofy") else goofy_path.stem
    for ext in AUDIO_EXTS:
        cand = goofy_path.with_name(base + ext)
        if cand.exists() and cand.is_file():
            return cand
    return None

def _load_preview_audio(env0, f0i0, vmask0, forms0, sr0, n_fft: int, hop_length: int):
    env_dense = gf.decode_env_from_knots(env0) if isinstance(env0, dict) and env0.get("mode") == "knots" else env0
    y_len = int(len(vmask0))
    y_len_bool = np.empty(y_len, dtype=np.bool_)
    _, harmonic, aper_uv, aper_bre = gf.synthesize(
        env_dense, vmask0 * np.maximum(f0i0, 0.0), vmask0, y_len_bool, sr0,
        n_fft=n_fft, hop_length=hop_length, normalize=1.0
    )
    return (harmonic + aper_uv + aper_bre) * 0.5


def edit_goofy_files(goofy_paths, n_fft=1024, hop_length=256):
    for path_str in goofy_paths:
        p = Path(path_str)
        if not p.exists() or p.suffix.lower() != ".goofy":
            logging.warning(f"[GOOFY] Skip non-existent or not .goofy: {p}")
            continue

        try:
            logging.info(f"[GOOFY] Opening {p.name}")
            env0, f0i0, vmask0, forms0, sr0, ylen0 = gf.load_features(str(p))

            y_for_ui = None
            audio = _find_neighbor_audio_for_goofy(p)
            if audio is not None:
                try:
                    y_for_ui, sr_a = sf.read(audio)
                    if y_for_ui.ndim > 1:
                        y_for_ui = y_for_ui.mean(axis=1)
                    if sr_a != sr0:
                        x_old = np.linspace(0, len(y_for_ui)/sr_a, num=len(y_for_ui), endpoint=False, dtype=np.float64)
                        x_new = np.linspace(0, len(y_for_ui)/sr0, num=int(round(len(y_for_ui)*sr0/sr_a)), endpoint=False, dtype=np.float64)
                        f = gf.interp1d(x_old, y_for_ui.astype(np.float64), kind="linear", fill_value="extrapolate")
                        y_for_ui = f(x_new).astype(np.float32)
                except Exception as e:
                    logging.warning(f"[GOOFY] Failed to load neighbor audio {audio.name}: {e}")
                    y_for_ui = None

            if y_for_ui is None:
                y_for_ui = _load_preview_audio(env0, f0i0, vmask0, forms0, sr0, n_fft, hop_length)

            init_mask = np.asarray(vmask0, dtype=np.float32)

            root = tk.Tk()
            root.withdraw()
            ui = SillyEditor(root, y_for_ui.astype(np.float32), sr0, init_mask=init_mask, title=f"Voicing: {p.name}")
            ui.features = (env0, f0i0, vmask0, forms0, sr0, ylen0)
            ui._init_tracks()
            root.wait_window(ui.win)
            ui_mask = ui.mask_full.astype(np.float32) if ui.ok else None
            try:
                root.destroy()
            except Exception:
                pass

            if ui_mask is None:
                logging.info(f"[GOOFY] Edit cancelled: {p.name}")
                continue

            target_len = int(ylen0)
            if ui_mask.shape[0] != target_len:
                if ui_mask.shape[0] > target_len:
                    ui_mask = ui_mask[:target_len]
                else:
                    ui_mask = np.pad(ui_mask, (0, target_len - ui_mask.shape[0]), mode="edge")

            out_f0 = np.asarray(f0i0, dtype=np.float32)
            if getattr(ui, "f0_full", None) is not None:
                out_f0 = np.asarray(ui.f0_full, dtype=np.float32)

            if out_f0.shape[0] != target_len:
                if out_f0.shape[0] > target_len:
                    out_f0 = out_f0[:target_len]
                else:
                    out_f0 = np.pad(out_f0, (0, target_len - out_f0.shape[0]), mode="edge")

            tmp_path = str(p) + ".tmp"
            gf.save_features(tmp_path, env0, out_f0, ui_mask, forms0, sr0, target_len)
            os.replace(tmp_path, str(p))
            logging.info(f"[GOOFY] Saved edits {p.name}")

        except Exception:
            logging.exception(f"[GOOFY] Failed to edit {p}")
