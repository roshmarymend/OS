# =====================================================================
#  SDG-Based Round Robin Scheduling Visualizer
#  Run in Google Colab or Jupyter Notebook (paste into ONE cell and run)
#  Covers Practical Activity 1 (no overhead) and Activity 2 (with overhead)
# =====================================================================
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

# ----------------------------- DATA -----------------------------------
# EDIT THESE to try your own workloads: (pid, task, SDG, arrival, burst)
ACTIVITIES = {
    "Activity 1 - Municipal Control Centre (no overhead)": {
        "default_q": (2, 3), "default_cs": 0,
        "procs": [
            ("P1", "Water-quality monitoring",   "SDG 6",  0, 5),
            ("P2", "Waste-collection scheduling", "SDG 12", 1, 3),
            ("P3", "Traffic-pollution analysis",  "SDG 11", 2, 4),
            ("P4", "Public energy-use monitoring", "SDG 12", 3, 2),
        ],
    },
    "Activity 2 - Climate Emergency Centre (with overhead)": {
        "default_q": (2, 4), "default_cs": 1,
        "procs": [
            ("P1", "Flood-warning analysis",          "SDG 13", 0, 5),
            ("P2", "Air-quality alert generation",    "SDG 11", 1, 4),
            ("P3", "Emergency evacuation-route update", "SDG 11", 2, 3),
            ("P4", "Renewable-grid status monitoring", "SDG 13", 3, 2),
        ],
    },
}
COLORS = ["#26BDE2", "#FD9D24", "#4C9F38", "#DD1367", "#8E44AD", "#3F7E44"]
CS_COLOR = "#555555"


# ------------------------- SIMULATION ENGINE --------------------------
def round_robin(procs, q, cs):
    """Round Robin. A context switch (cs ms) is inserted only when the CPU
    moves from one process to a DIFFERENT process (none before the first or
    after the last). Arrivals during a slice join the queue BEFORE the
    pre-empted process."""
    procs = sorted(procs, key=lambda p: (p[3], p[0]))
    n = len(procs)
    rem = {p[0]: p[4] for p in procs}
    first, finish, segs = {}, {}, []
    queue, t, i, last, switches = deque(), 0, 0, None, 0

    def admit(upto):
        nonlocal i
        while i < n and procs[i][3] <= upto:
            queue.append(procs[i][0]); i += 1

    admit(t)
    while len(finish) < n:
        if not queue:                       # CPU idle until next arrival
            segs.append((t, procs[i][3], "IDLE", "idle"))
            t = procs[i][3]; admit(t)
        pid = queue.popleft()
        if last is not None and last != pid:
            switches += 1
            if cs > 0:
                segs.append((t, t + cs, "CS", "cs")); t += cs
        run = min(q, rem[pid])
        first.setdefault(pid, t)
        segs.append((t, t + run, pid, "run"))
        t += run; rem[pid] -= run
        admit(t)
        if rem[pid] > 0:
            queue.append(pid)
        else:
            finish[pid] = t
        last = pid

    rows = []
    for pid, task, sdg, at, bt in procs:
        ct = finish[pid]; tat = ct - at
        rows.append({"Process": pid, "SDG": sdg, "Task": task, "Arrival": at,
                     "Burst": bt, "Completion": ct, "Turnaround": tat,
                     "Waiting": tat - bt, "Response": first[pid] - at})
    df = pd.DataFrame(rows)
    makespan = max(finish.values())
    stats = {
        "Quantum (ms)": q,
        "Avg Turnaround (ms)": round(df["Turnaround"].mean(), 2),
        "Avg Waiting (ms)": round(df["Waiting"].mean(), 2),
        "Avg Response (ms)": round(df["Response"].mean(), 2),
        "Max Response (ms)": int(df["Response"].max()),
        "Context Switches": switches,
        "Total CS Overhead (ms)": switches * cs,
        "Total Time (ms)": makespan,
        "CPU Efficiency (%)": round(100 * df["Burst"].sum() / makespan, 2),
    }
    return segs, df, stats


# ---------------------------- VISUALS ---------------------------------
def gantt(segs, procs, title, cs):
    pids = [p[0] for p in procs]
    sdg = {p[0]: p[2] for p in procs}
    color = {pid: COLORS[k % len(COLORS)] for k, pid in enumerate(pids)}
    rows = pids + (["Ctx Switch"] if cs > 0 else [])
    ypos = {r: k for k, r in enumerate(rows)}
    fig, ax = plt.subplots(figsize=(11, 1.0 + 0.7 * len(rows)))
    ticks = set()
    for s, e, lab, kind in segs:
        if kind == "run":
            ax.barh(ypos[lab], e - s, left=s, color=color[lab], edgecolor="black")
            ax.text((s + e) / 2, ypos[lab], f"{lab}\n{e - s}ms", ha="center",
                    va="center", fontsize=8, color="white", weight="bold")
        elif kind == "cs":
            ax.barh(ypos["Ctx Switch"], e - s, left=s, color=CS_COLOR,
                    edgecolor="black", hatch="//")
        ticks.update([s, e])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r} ({sdg[r]})" if r in sdg else r for r in rows])
    ax.set_xticks(sorted(ticks)); ax.set_xlabel("Time (ms)")
    ax.invert_yaxis(); ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.set_title(title, weight="bold")
    plt.tight_layout(); plt.show()


def recommend(all_stats, act_name, threshold):
    names = list(all_stats)
    if "Activity 2" in act_name:
        ok = [k for k in names if all_stats[k]["Max Response (ms)"] <= threshold]
        pool = ok if ok else names
        best = max(pool, key=lambda k: (all_stats[k]["CPU Efficiency (%)"],
                                        -all_stats[k]["Max Response (ms)"]))
        note = (f"Quanta meeting the max-response limit of {threshold} ms: "
                f"{[all_stats[k]['Quantum (ms)'] for k in ok] or 'none'}.")
        why = "highest CPU efficiency (least overhead) while keeping response suitable for emergency services"
    else:
        keys = ["Avg Waiting (ms)", "Avg Response (ms)", "Context Switches"]
        mins = {m: (min(all_stats[k][m] for k in names) or 1) for m in keys}
        score = {k: sum(all_stats[k][m] / mins[m] for m in keys) for k in names}
        best = min(score, key=score.get); note = ""
        why = "best balance of waiting time, fair CPU access (response) and context switching"
    q = all_stats[best]["Quantum (ms)"]
    return f"RECOMMENDATION: Quantum = {q} ms gives the {why}. {note}"


# ------------------------------ UI ------------------------------------
act_dd = widgets.Dropdown(options=list(ACTIVITIES), description="Activity:",
                          layout=widgets.Layout(width="620px"),
                          style={"description_width": "80px"})
q1 = widgets.IntSlider(value=2, min=1, max=8, description="Quantum A:",
                       style={"description_width": "80px"})
q2 = widgets.IntSlider(value=3, min=1, max=8, description="Quantum B:",
                       style={"description_width": "80px"})
cs_sl = widgets.IntSlider(value=0, min=0, max=5, description="Ctx switch:",
                          style={"description_width": "80px"})
thr = widgets.IntSlider(value=3, min=1, max=10, description="Max resp.:",
                        style={"description_width": "80px"})
out = widgets.Output()


def run(_=None):
    with out:
        clear_output(wait=True)
        act = act_dd.value
        procs = ACTIVITIES[act]["procs"]
        cs = cs_sl.value
        print(f"=== {act} | context-switch overhead = {cs} ms ===")
        all_stats = {}
        for label, q in (("A", q1.value), ("B", q2.value)):
            segs, df, stats = round_robin(procs, q, cs)
            all_stats[f"Q={q} ms"] = stats
            gantt(segs, procs, f"Gantt Chart - Quantum {q} ms (Option {label})", cs)
            display(df.style.hide(axis="index").set_caption(
                f"Per-process results (Quantum {q} ms)"))
            print(f"Avg turnaround: {stats['Avg Turnaround (ms)']} ms | "
                  f"Avg waiting: {stats['Avg Waiting (ms)']} ms | "
                  f"Context switches: {stats['Context Switches']}"
                  + (f" | Overhead: {stats['Total CS Overhead (ms)']} ms | "
                     f"CPU efficiency: {stats['CPU Efficiency (%)']}%" if cs else ""))
            print()
        comp = pd.DataFrame(all_stats)
        display(comp.style.set_caption("Comparison table (tool-generated)"))
        comp.loc[["Avg Turnaround (ms)", "Avg Waiting (ms)", "Avg Response (ms)"]] \
            .plot(kind="bar", figsize=(8, 3.5), rot=0, edgecolor="black",
                  title="Quantum comparison")
        plt.ylabel("ms"); plt.tight_layout(); plt.show()
        print(recommend(all_stats, act, thr.value))


def on_activity(_=None):
    cfg = ACTIVITIES[act_dd.value]
    q1.value, q2.value = cfg["default_q"]
    cs_sl.value = cfg["default_cs"]
    run()


act_dd.observe(on_activity, names="value")
for w in (q1, q2, cs_sl, thr):
    w.observe(run, names="value")

display(widgets.VBox([
    widgets.HTML("<h3>SDG Round Robin Scheduler "
                 "(SDG 6 | 11 | 12 | 13)</h3>"
                 "<i>Change any control - charts update instantly.</i>"),
    act_dd, widgets.HBox([q1, q2]), widgets.HBox([cs_sl, thr]), out]))
on_activity() 