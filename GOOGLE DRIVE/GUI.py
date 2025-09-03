import tkinter as tk
from tkinter import messagebox
import tkinter.font as tkfont
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
import textwrap

# ---------- Config ----------
HIST_CSV = "Histogram_Data.csv"   # expects numeric column 'value'
TABLE_CSV = "Table_Data.csv"      # expects 5 columns

# Theme
BG = "#000000"
FG = "#FFFFFF"
HEADER_BG = "#FF8C00"
GRID_COLOR = "#FFFFFF"

# Histogram
BAR_COLOR = "#FF8C00"
LAST_BAR_COLOR = "#00C853"
BINS = 6

# Fonts
TABLE_FONT = ("Verdana", 8)
HEADER_FONT = ("Verdana", 8, "normal")
AXIS_FONT   = ("Verdana", 8, "normal")
TITLE_FONT  = ("Verdana", 12, "normal")
TICK_FONT_SIZE = 9

# Left-panel fonts (buttons + logs)
LOG_TITLE_FONT = ("Verdana", 8, "bold")
LOG_TEXT_FONT  = ("Verdana", 6, "normal")
BUTTON_FONT    = ("Verdana", 12, "bold")

# Table behaviour
WRAP_COLUMN_INDEX = 4
REL_COL_WEIGHTS   = [1, 1, 1, 1, 4]
MIN_COL_WIDTHS    = [50, 50, 50, 50, 100]
MAX_ROW_LINES     = 4
FIXED_WRAP_CHARS  = 80

# ---------- App ----------
root = tk.Tk()
root.title("Analysis Dashboard")
root.configure(bg=BG)

# Fullscreen
try:
    root.state('zoomed')
except Exception:
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{sw}x{sh}+0+0")

# ---------- Root grid: Left | Divider | Right (DEFAULT 40:60) ----------
root.grid_rowconfigure(0, weight=1)
# 40:60 via weights 2:3
root.grid_columnconfigure(0, weight=2, uniform="cols", minsize=400)  # left ~40%
root.grid_columnconfigure(1, weight=0, minsize=2)                    # divider
root.grid_columnconfigure(2, weight=3, uniform="cols", minsize=600)  # right ~60%

# ====== LEFT PANE ======
left = tk.Frame(root, bg=BG)
left.grid(row=0, column=0, sticky="nsew")

left.grid_rowconfigure(0, weight=0)   # title
left.grid_rowconfigure(1, weight=1)   # rows (buttons + logs)
left.grid_columnconfigure(0, weight=1)

left_title = tk.Label(left, text="RAG CV Analysis Tool",
                      font=("Verdana", 14, "bold"), fg=FG, bg=BG)
left_title.grid(row=0, column=0, sticky="nw", padx=16, pady=12)

rows = tk.Frame(left, bg=BG)
rows.grid(row=1, column=0, sticky="nsew", padx=12, pady=(4, 12))
rows.grid_columnconfigure(0, weight=0)
rows.grid_columnconfigure(1, weight=0)
rows.grid_columnconfigure(2, weight=1)

BTN_BG = "#66B2FF"
BTN_BG_ACTIVE = "#4C97E6"
AN_BG = "#0078D7"
AN_BG_ACTIVE = "#005A9E"

BUTTON_WIDTH = 16
ROW_PADY = 14
BOX_HEIGHT = 36

def not_implemented(name):
    messagebox.showinfo("Info", f"{name} clicked. (Hook up your logic here.)")

def make_worklog(parent, row, initial_text=""):
    outer = tk.Frame(parent, bg=FG, height=BOX_HEIGHT)
    outer.grid(row=row, column=2, sticky="ew", padx=(6, 4))
    outer.grid_propagate(False)
    inner = tk.Frame(outer, bg=BG)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    content = tk.Label(inner, text=initial_text, font=LOG_TEXT_FONT,
                       fg=FG, bg=BG, justify="left", anchor="nw",
                       wraplength=700)
    content.pack(anchor="nw", padx=2, pady=1)
    return content

def add_left_row(r, btn_text, btn_bg, btn_active_bg, command, worklog_text=""):
    tk.Button(rows, text=btn_text,
              font=BUTTON_FONT,
              fg="white", bg=btn_bg,
              activebackground=btn_active_bg, activeforeground="white",
              relief="raised", bd=3, padx=6, pady=4,
              width=BUTTON_WIDTH, cursor="hand2",
              command=command
              ).grid(row=r, column=0, sticky="w", pady=ROW_PADY, padx=(4, 8))
    tk.Label(rows, text="Work Log:", font=LOG_TITLE_FONT, fg=FG, bg=BG)\
        .grid(row=r, column=1, sticky="w", pady=ROW_PADY, padx=(0, 4))
    make_worklog(rows, r, initial_text=worklog_text)

# ====== DIVIDER ======
divider = tk.Frame(root, bg="white", width=2)
divider.grid(row=0, column=1, sticky="ns")

# ====== RIGHT PANE (exact 40% plot / 60% table using place) ======
right = tk.Frame(root, bg=BG)
right.grid(row=0, column=2, sticky="nsew")
right.update_idletasks()  # ensure it has a size

# Containers with fixed relative heights
plot_container  = tk.Frame(right, bg=BG)
table_container = tk.Frame(right, bg=BG)
# EXACT same split, just ensure stacking & no rounding overlap
plot_container.place(relx=0, rely=0.00, relwidth=1, relheight=0.55)
table_container.place(relx=0, rely=0.55, relwidth=1, relheight=0.65)

# keep the plot above the table if pixels collide
plot_container.lift()

# Inner frames with padding (so padding doesn't change the ratio)
plot_frame = tk.Frame(plot_container, bg=BG) ##BG
plot_frame.pack(fill="both", expand=True, padx=2, pady=(0, 0))

table_frame = tk.Frame(table_container, bg=BG)
table_frame.pack(fill="both", expand=True, padx=2, pady=(0, 0))

# Ensure the table internals fill horizontally and vertically
table_frame.grid_rowconfigure(0, weight=0)  # title
table_frame.grid_rowconfigure(1, weight=1)  # canvases region
table_frame.grid_columnconfigure(0, weight=1)

# ---------- Matplotlib theme ----------
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Verdana", "Arial", "DejaVu Sans", "Liberation Sans"],
    "text.color": FG,
    "axes.labelcolor": FG,
    "axes.edgecolor": GRID_COLOR,
    "xtick.color": FG,
    "ytick.color": FG,
})

# Placeholders
plot_placeholder = tk.Label(plot_frame, text="Analyse CVs to view results",
                            font=("Verdana", 14, "bold"), fg="white", bg="gray25")
plot_placeholder.place(relx=0.5, rely=0.5, anchor="center")

table_placeholder = tk.Label(table_frame, text="Analyse CVs to view results",
                             font=("Verdana", 14, "bold"), fg="white", bg="gray25")
table_placeholder.place(relx=0.5, rely=0.5, anchor="center")

# ================= Table helpers =================
def compute_col_widths(total_width: int):
    wsum = sum(REL_COL_WEIGHTS)
    widths = [max(m, int(total_width * w / wsum)) for m, w in zip(MIN_COL_WIDTHS, REL_COL_WEIGHTS)]
    extra = sum(widths) - total_width
    if extra > 0:
        order = sorted(range(len(widths)), key=lambda i: widths[i]-MIN_COL_WIDTHS[i], reverse=True)
        i = 0
        while extra > 0 and i < len(order):
            idx = order[i]
            reducible = widths[idx] - MIN_COL_WIDTHS[idx]
            take = min(reducible, extra)
            widths[idx] -= take
            extra -= take
            if reducible - take == 0:
                i += 1
    return widths

def wrap_text_fixed(text: str, width_chars: int, max_lines=None, ellipsis=True):
    if text is None:
        return ""
    if isinstance(text, float):
        try:
            if np.isnan(text): return ""
        except Exception:
            pass
    wrapped = textwrap.fill(str(text), width=width_chars)
    if max_lines is not None:
        lines = wrapped.split("\n")
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            if ellipsis:
                if len(lines[-1]) > 3:
                    lines[-1] = lines[-1][:-3] + "..."
                else:
                    lines[-1] += "..."
        wrapped = "\n".join(lines)
    return wrapped

def wrap_header_two_lines(text: str, width_chars: int = 80):
    words = textwrap.wrap(str(text), width=width_chars)
    if len(words) <= 1:
        return words[0] if words else ""
    if len(words) == 2:
        return "\n".join(words)
    second = " ".join(words[1:])
    sw = textwrap.wrap(second, width=width_chars)
    second_line = sw[0] if sw else ""
    if len(sw) > 1:
        second_line = (second_line[:-3] + "...") if len(second_line) > 3 else (second_line + "...")
    return words[0] + "\n" + second_line

def draw_table(table_df, table_cols, header_canvas, body_canvas, tk_header_font, tk_table_font):
    body_canvas.delete("all")
    header_canvas.delete("all")

    view_w = max(1, body_canvas.winfo_width())
    col_widths = compute_col_widths(view_w)

    df = table_df.copy()
    wrapped_lines = []
    for i, val in enumerate(df.iloc[:, WRAP_COLUMN_INDEX].tolist()):
        wrapped = wrap_text_fixed(val, FIXED_WRAP_CHARS, max_lines=MAX_ROW_LINES, ellipsis=True)
        df.iloc[i, WRAP_COLUMN_INDEX] = wrapped
        wrapped_lines.append(wrapped.count("\n") + 1)
    wrapped_lines = [min(MAX_ROW_LINES, n) for n in wrapped_lines]
    ROW_LINE_HEIGHT = 16
    row_heights = [max(ROW_LINE_HEIGHT, n * ROW_LINE_HEIGHT) for n in wrapped_lines]

    col_x = [0]
    for w in col_widths:
        col_x.append(col_x[-1] + w)
    table_width = col_x[-1]

    # Header
    header_texts, header_lines_counts = [], []
    for i, title in enumerate(table_cols):
        wrapped = wrap_header_two_lines(str(title), width_chars=FIXED_WRAP_CHARS)
        header_texts.append(wrapped)
        header_lines_counts.append(wrapped.count("\n") + 1)
    header_max_lines = max(header_lines_counts) if header_lines_counts else 1
    header_h = max(24, header_max_lines * (tk_header_font.metrics("linespace") + 2) + 6)
    header_canvas.configure(scrollregion=(0, 0, table_width, header_h), height=header_h)

    for i, wrapped_title in enumerate(header_texts):
        x0, x1 = col_x[i], col_x[i+1]
        header_canvas.create_rectangle(x0, 0, x1, header_h, fill=HEADER_BG, outline=GRID_COLOR, width=1)
        anchor = "w" if i == WRAP_COLUMN_INDEX else "center"
        if anchor == "w":
            lines = wrapped_title.split("\n")
            ty = header_h/2 - (len(lines)-1) * (tk_header_font.metrics("linespace")/2)
            for idx, ln in enumerate(lines):
                header_canvas.create_text(x0 + 6, ty + idx*(tk_header_font.metrics("linespace")),
                                          text=ln, fill=FG, font=HEADER_FONT, anchor="w")
        else:
            header_canvas.create_text((x0 + x1)/2, header_h/2,
                                      text=wrapped_title, fill=FG, font=HEADER_FONT, anchor="center")

    # Body
    y = 0
    for r in range(len(df)):
        rh = row_heights[r]
        body_canvas.create_rectangle(0, y, table_width, y + rh, outline=GRID_COLOR, width=1, fill=BG)
        for i, col in enumerate(df.columns):
            x0, x1 = col_x[i], col_x[i+1]
            cell_text = str(df.iloc[r, i])
            if i == WRAP_COLUMN_INDEX:
                lines = cell_text.split("\n")[:MAX_ROW_LINES]
                ty = y + 4
                for ln in lines:
                    body_canvas.create_text(x0 + 6, ty, text=ln, fill=FG, font=TABLE_FONT, anchor="nw")
                    ty += ROW_LINE_HEIGHT
            else:
                body_canvas.create_text((x0 + x1)/2, y + rh/2, text=cell_text, fill=FG, font=TABLE_FONT, anchor="c")
            body_canvas.create_line(x1, y, x1, y + rh, fill=GRID_COLOR)
        body_canvas.create_line(0, y + rh, table_width, y + rh, fill=GRID_COLOR)
        y += rh

    body_canvas.create_line(0, 0, 0, y, fill=GRID_COLOR)
    body_canvas.configure(scrollregion=(0, 0, table_width, y))
    header_canvas.configure(width=body_canvas.winfo_width())

# ====== Analyse CVs action ======
def analyse_cvs():
    # Freeze window size during rebuild to avoid a visible jump
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    root.minsize(w, h)
    root.maxsize(w, h)

    try:
        if plot_placeholder and plot_placeholder.winfo_exists():
            plot_placeholder.destroy()
        if table_placeholder and table_placeholder.winfo_exists():
            table_placeholder.destroy()

        for wdg in plot_frame.winfo_children():
            wdg.destroy()
        for wdg in table_frame.winfo_children():
            wdg.destroy()

        # ---------- Load Data ----------
        try:
            hist_df = pd.read_csv(HIST_CSV)
            hist_col = 'value' if 'value' in hist_df.columns else hist_df.select_dtypes(include=[np.number]).columns.tolist()[0]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {HIST_CSV}:\n{e}")
            hist_df = pd.DataFrame({'value': np.random.randn(200)})
            hist_col = 'value'

        try:
            table_df = pd.read_csv(TABLE_CSV)
        except Exception as e:
            messagebox.showwarning("Warning", f"Failed to load {TABLE_CSV}:\n{e}\nUsing demo data.")
            table_df = pd.DataFrame({
                "Col 1": [f"R{r+1}C1" for r in range(12)],
                "Col 2": [f"R{r+1}C2" for r in range(12)],
                "Col 3": [f"R{r+1}C3" for r in range(12)],
                "Column 4 (very long title example)": [f"R{r+1}C4" for r in range(12)],
                "Details / Description": [f"This is some longer wrapped text example row {r+1}. " * 2 for r in range(12)],
            })

        if table_df.shape[1] < 5:
            for i in range(5 - table_df.shape[1]):
                table_df[f"Col{table_df.shape[1] + i + 1}"] = ""
        elif table_df.shape[1] > 5:
            table_df = table_df.iloc[:, :5]
        table_cols = [str(c) for c in table_df.columns.tolist()]

        # ---------- Histogram ----------
        fig = Figure(dpi=100)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        vals = pd.to_numeric(hist_df[hist_col], errors='coerce').dropna().values
        hist_out = ax.hist(vals, bins=BINS, edgecolor=GRID_COLOR, linewidth=1.0)
        patches = hist_out[2]
        for p in patches: p.set_facecolor(BAR_COLOR)
        if patches: patches[-1].set_facecolor(LAST_BAR_COLOR)

        for spine in ax.spines.values(): spine.set_visible(False)

        ax.set_title("Histogram of Applicants Scores",
                     fontname=TITLE_FONT[0], fontsize=TITLE_FONT[1],
                     fontweight=TITLE_FONT[2], color=FG)
        ax.set_xlabel(hist_col, fontname=AXIS_FONT[0], fontsize=AXIS_FONT[1],
                      fontweight=AXIS_FONT[2], color=FG)
        ax.set_ylabel("Frequency", fontname=AXIS_FONT[0], fontsize=AXIS_FONT[1],
                      fontweight=AXIS_FONT[2], color=FG)

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname(TABLE_FONT[0]); tick.set_fontsize(TICK_FONT_SIZE)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        cw = canvas.get_tk_widget()
        cw.configure(bg=BG, highlightthickness=0, bd=0)
        cw.pack(expand=True)

        def on_canvas_configure(_):
            fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.26)
            canvas.draw_idle()
        cw.bind("<Configure>", on_canvas_configure)

        fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.14)
        canvas.draw()

        # ---------- Table ----------
        title_label = tk.Label(table_frame, text="Top 5 Applicants", font=TITLE_FONT, fg=FG, bg=BG)
        title_label.grid(row=0, column=0, sticky="w", padx=6, pady=(2, 4))

        outer = tk.Frame(table_frame, bg=BG)
        outer.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        inner = tk.Frame(outer, bg=BG)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        header_canvas = tk.Canvas(inner, bg=BG, highlightthickness=0, bd=0)
        body_canvas   = tk.Canvas(inner, bg=BG, highlightthickness=0, bd=0)
        header_canvas.grid(row=0, column=0, sticky="ew")
        body_canvas.grid(row=1, column=0, sticky="nsew")
        inner.grid_rowconfigure(0, weight=0)
        inner.grid_rowconfigure(1, weight=1)
        inner.grid_columnconfigure(0, weight=1)

        tk_table_font  = tkfont.Font(root=root, font=TABLE_FONT)
        tk_header_font = tkfont.Font(root=root, font=HEADER_FONT)

        def trigger_table_draw(*_):
            body_canvas.after_idle(lambda: draw_table(table_df, table_cols, header_canvas, body_canvas, tk_header_font, tk_table_font))

        body_canvas.bind("<Configure>", trigger_table_draw)
        table_frame.update_idletasks()
        trigger_table_draw()

    finally:
        # Unfreeze after the layout settles (prevents visible jump)
        root.after(100, lambda: (root.minsize(0, 0), root.maxsize(1_000_000, 1_000_000)))

# ----- Left rows -----
add_left_row(
    0, "Ingest CVs", BTN_BG, BTN_BG_ACTIVE,
    lambda: not_implemented("Ingest CVs"),
    worklog_text="CV_ID8342: Language not detected as being in english, omitted\n"
                 "CV_ID4326: File type not supported, omitted\n"
                 "CV_ID1234: No CV detected, omitted"
)
add_left_row(1, "Extract Text", BTN_BG, BTN_BG_ACTIVE,
             lambda: not_implemented("Extract Text"))
add_left_row(2, "Clean Text", BTN_BG, BTN_BG_ACTIVE,
             lambda: not_implemented("Clean Text"))
add_left_row(3, "Vectorise", BTN_BG, BTN_BG_ACTIVE,
             lambda: not_implemented("Vectorise"))
add_left_row(4, "Run LLM", BTN_BG, BTN_BG_ACTIVE,
             lambda: not_implemented("Run LLM"))
add_left_row(5, "Analyse CVs", AN_BG, AN_BG_ACTIVE, analyse_cvs)

root.mainloop()
