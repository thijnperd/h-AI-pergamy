"""A native desktop GUI for the HYPERGAMY models -- no browser, no server.

This is now the default interface started by ``python gui.py``. It uses only the
standard library (``tkinter``) and calls the same :func:`gui.compute` engine as
the legacy browser UI, so the numbers are identical. Profiles never leave the
machine because there is no HTTP layer at all.

The Tk widgets are created lazily so importing this module -- and running the
unit tests -- does not require a display. Only opening the window needs Tk::

  python local_gui.py
  python gui.py              # same desktop window
  python gui.py --serve      # legacy local browser UI
"""

import argparse
import json
import queue
import threading

import adult
import gui
from profile import DEFAULTS

SELF_LABELS = ("looks", "sincerity", "intelligence", "fun", "ambition")
WANT_LABELS = ("looks", "sincerity", "intelligence", "fun", "ambition",
               "shared taste")
MODES = ("regular", "advanced", "adult", "18-25")
GENDERS = ("man", "woman")
PARTNERS = ("women", "men")
ETHNICITIES = ("not specified", "white", "black", "latino", "asian",
               "pacific", "other")
EDUCATIONS = ("", "high school or less", "some college", "bachelor",
              "graduate")


def default_form():
    """Starter values for every desktop control, as strings and booleans."""
    return {
        "mode": "regular",
        "gender": "man",
        "partner_gender": "women",
        "age": "24",
        "income": "32000",
        "goal": "4",
        "self": ["5", "5", "5", "5", "3"],
        "want": ["6", "6", "6", "6", "6", "6"],
        "importance": ["30", "10", "15", "15", "20", "10"],
        "height_cm": "178",
        "body_type": "average",
        "ethnicity": "not specified",
        "education": "",
        "age_gap": "3",
        "same_race": True,
        "kink_match": "3",
        "libido_match": "3",
        "size_cm": "13.1",
        "own_cup": "C",
        "cup": "C",
        "partner_size_cm": "13.1",
        "partner_body": "average",
        "experience": "3",
        "orientation_match": True,
    }


def to_float(value, default, minimum=None, maximum=None, label="value"):
    """Parse one control, falling back to the default only when it is blank."""
    text = "" if value is None else str(value).strip()
    if text == "":
        number = float(default)
    else:
        try:
            number = float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a number, got {text!r}") from exc
        if number != number:
            raise ValueError(f"{label} must not be NaN")
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return float(number)


def to_float_list(values, defaults, minimum, maximum, label):
    """Parse a complete rating row, accepting either a list or CSV text."""
    if isinstance(values, str):
        values = [part for part in values.replace(";", ",").split(",")]
    if len(list(values)) != len(defaults):
        raise ValueError(f"{label} needs {len(defaults)} numbers")
    return [to_float(value, default, minimum, maximum, f"{label}[{i}]")
            for i, (value, default) in enumerate(zip(values, defaults))]


def _choice(value, options, label):
    text = str(value).strip()
    if text not in options:
        raise ValueError(f"{label} must be one of {', '.join(options)}")
    return text


def build_payload(form):
    """Turn desktop-control values into the payload understood by gui.compute."""
    mode = _choice(form.get("mode", "regular"), MODES, "mode")
    gender = _choice(form.get("gender", "man"), GENDERS, "gender")
    partner_gender = _choice(form.get("partner_gender", "women"), PARTNERS,
                             "partner gender")
    payload = {
        "mode": mode,
        "gender": gender,
        "partner_gender": partner_gender,
        "age": to_float(form.get("age"), DEFAULTS["age"], 18.0, 80.0, "age"),
        "income": to_float(form.get("income"), DEFAULTS["income"], 0.0,
                           None, "income"),
        "goal": to_float(form.get("goal"), DEFAULTS["goal"], 1.0, 6.0,
                         "goal"),
        "self": to_float_list(form.get("self", DEFAULTS["self"]),
                              DEFAULTS["self"], 0.0, 10.0, "self ratings"),
        "want": to_float_list(form.get("want", DEFAULTS["want"]),
                              DEFAULTS["want"], 0.0, 10.0, "wanted ratings"),
        "importance": to_float_list(form.get("importance",
                                            DEFAULTS["importance"]),
                                   DEFAULTS["importance"], 0.0, 100.0,
                                   "importance weights"),
        "height_cm": to_float(form.get("height_cm"), DEFAULTS["height_cm"],
                              140.0, 210.0, "height"),
        "body_type": _choice(form.get("body_type", "average"),
                             adult.BODY_TYPES, "body type"),
        "ethnicity": _choice(form.get("ethnicity", "not specified"),
                             ETHNICITIES, "ethnicity"),
        "education": _choice(form.get("education", ""), EDUCATIONS,
                             "education"),
        "age_gap": to_float(form.get("age_gap"), DEFAULTS["age_gap"], 0.0,
                            20.0, "age gap"),
        "same_race": bool(form.get("same_race", True)),
        "kink_match": to_float(form.get("kink_match"),
                               DEFAULTS["kink_match"], 0.0, 5.0,
                               "kink compatibility"),
        "libido_match": to_float(form.get("libido_match"),
                                 DEFAULTS["libido_match"], 0.0, 5.0,
                                 "libido compatibility"),
        "size_cm": to_float(form.get("size_cm"), DEFAULTS["size_cm"], 8.0,
                            18.0, "size"),
        "own_cup": adult.cup_key(form.get("own_cup", "C")),
        "cup": adult.cup_key(form.get("cup", "C")),
        "partner_size_cm": to_float(form.get("partner_size_cm"),
                                    DEFAULTS["partner_size_cm"], 8.0, 18.0,
                                    "preferred size"),
        "partner_body": _choice(form.get("partner_body", "average"),
                                adult.BODY_TYPES, "partner build"),
        "experience": to_float(form.get("experience"),
                               DEFAULTS["experience"], 0.0, 5.0,
                               "experience"),
        "orientation_match": bool(form.get("orientation_match", True)),
    }
    return payload


def _percent(value):
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _money(value):
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def format_result(result):
    """Render one compute() result as plain desktop text with its basis."""
    lines = ["HYPERGAMY local result", ""]
    lines.append(f"Both say yes: {_percent(result.get('mutual'))}")
    lines.append(f"They say yes: {_percent(result.get('p_them'))}")
    lines.append(f"You say yes:  {_percent(result.get('p_you'))}")
    if result.get("routing"):
        lines.append(f"Models: {result['routing']}")
    for warning in result.get("warnings", []) or []:
        lines.append(f"Warning: {warning}")

    hyper = result.get("hypergamy") or {}
    market = result.get("market") or {}
    if hyper:
        lines.append(f"Ambition-vs-looks percentile: {hyper.get('percentile')}")
    if market:
        lines.append(f"Market score: {market.get('score')} "
                     f"(rank {market.get('rank')}, n={market.get('n')})")

    likes = result.get("likes") or {}
    if likes:
        lines.append("")
        lines.append(f"Who likes you most: package {likes.get('best')} at "
                     f"{_percent(likes.get('best_chance'))}")
        lines.append(f"Best mutual package: {likes.get('best_mutual')} at "
                     f"{_percent(likes.get('best_mutual_chance'))}")

    mapped = result.get("map") or {}
    if mapped:
        lines.append("")
        lines.append(f"Map: best package {mapped.get('best_level')} at "
                     f"{_percent(mapped.get('chance'))}; your package "
                     f"{mapped.get('your_level')} at "
                     f"{_percent(mapped.get('chance_your_level'))}")
        lines.append(f"Reaching: {mapped.get('hypergamy')}")
        lines.append(f"Verdict: {mapped.get('verdict')}")
        for band in mapped.get("bands", []) or []:
            lines.append(f"Package {band.get('level')}: "
                         f"{_percent(band.get('mutual'))} mutual -- "
                         f"{band.get('band')}")

    person = result.get("person")
    if person:
        lines.append("")
        lines.append(f"Generated partner: {person.get('gender')} "
                     f"{person.get('age')} (package {person.get('package')})")
        lines.append(f"Partner chance: they {_percent(person.get('p_they'))}, "
                     f"you {_percent(person.get('p_you'))}, mutual "
                     f"{_percent(person.get('mutual'))}")

    partner = result.get("partner") or {}
    if partner.get("available"):
        typical = partner.get("typical", {}) or {}
        lines.append("")
        lines.append("Measured partner context: median partner age "
                     f"{typical.get('age')}, income "
                     f"{_money(typical.get('income'))}")
    elif partner:
        lines.append("")
        lines.append(f"Measured partner context: {partner.get('note')}")

    adjustments = [row for row in result.get("adjustments", []) or []
                   if abs(float(row.get("pp", 0.0))) > 0.05]
    if adjustments:
        lines.append("")
        lines.append("Adjustments:")
        for row in adjustments:
            lines.append(f"- {row.get('label')}: {float(row.get('pp')):+.1f} "
                         f"pp [{row.get('basis')}]")

    models = (result.get("taste_models", []) or []) + \
        (result.get("market_models", []) or [])
    if models:
        lines.append("")
        lines.append("Model files:")
        for model in models:
            cv = model.get("cv")
            cv_text = f", CV {100.0 * float(cv):.1f}%" if cv else ""
            lines.append(f"- {model.get('path')}{cv_text}")
    return "\n".join(lines)


def _tk():
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:
        raise SystemExit(
            "tkinter is required for the desktop window but is not installed "
            "with this Python. On Debian/Ubuntu install the python3-tk "
            "system package, then run python gui.py again."
        ) from exc
    return tk, ttk, messagebox


class DesktopApp:
    """The native window. All model work stays in gui.compute()."""

    def __init__(self):
        tk, ttk, messagebox = _tk()
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.root = tk.Tk()
        self.root.title("HYPERGAMY -- local desktop")
        self.root.geometry("1120x720")
        self.root.minsize(900, 600)
        self.root.bind("<Return>", lambda event: self.on_compute())
        self.results = queue.Queue()
        self.vars = {}
        self.lists = {}
        self._build_layout()
        self._refresh_fields()
        self._poll()

    def _build_layout(self):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="both", expand=True)
        left = ttk.Frame(top)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = ttk.Frame(top)
        right.pack(side="right", fill="both", expand=True)

        canvas = tk.Canvas(left, highlightthickness=0)
        scroll = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self.form = ttk.Frame(canvas, padding=8)
        self.form_window = canvas.create_window((0, 0), window=self.form,
                                                anchor="nw")
        self.form.bind("<Configure>",
                       lambda event: canvas.configure(
                           scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda event: canvas.itemconfig(self.form_window,
                                                    width=event.width))
        self._build_form()

        buttons = ttk.Frame(left, padding=(0, 8, 0, 0))
        buttons.pack(fill="x")
        self.compute_button = ttk.Button(buttons, text="Compute my result",
                                         command=self.on_compute)
        self.compute_button.pack(side="left")
        ttk.Button(buttons, text="Reset",
                   command=self.on_reset).pack(side="left", padx=(8, 0))
        self.status = tk.StringVar(value="Ready. Nothing leaves this machine.")
        ttk.Label(left, textvariable=self.status,
                  wraplength=520).pack(anchor="w", pady=(6, 0))

        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)
        self.summary_text = tk.Text(notebook, wrap="word", width=58)
        notebook.add(self.summary_text, text="Summary")
        map_frame = ttk.Frame(notebook, padding=8)
        notebook.add(map_frame, text="Map")
        self.heat = tk.Canvas(map_frame, width=360, height=360,
                              highlightthickness=0)
        self.heat.pack()
        self.bands_text = tk.Text(map_frame, wrap="word", height=10)
        self.bands_text.pack(fill="both", expand=True, pady=(8, 0))
        self.details_text = tk.Text(notebook, wrap="word", width=58)
        notebook.add(self.details_text, text="Details")

    def _build_form(self):
        ttk = self.ttk
        row = 0
        row = self._section(row, "Profile")
        row = self._combo(row, "Mode", "mode", MODES)
        row = self._combo(row, "You are", "gender", GENDERS)
        row = self._combo(row, "Interested in", "partner_gender", PARTNERS)
        row = self._entry(row, "Age (18-80)", "age")
        row = self._entry(row, "Income ($/year)", "income")
        row = self._entry(row, "Goal (1-6; 4 is serious)", "goal")
        row = self._numbers(row, "How you rate yourself (0-10)", "self",
                            SELF_LABELS)
        row = self._numbers(row, "What you want (0-10)", "want", WANT_LABELS)
        row = self._numbers(row, "Importance weights (0-100)", "importance",
                            WANT_LABELS)
        row = self._section(row, "Advanced")
        row = self._entry(row, "Height in cm (140-210)", "height_cm")
        row = self._combo(row, "Your build", "body_type", adult.BODY_TYPES)
        row = self._combo(row, "Ethnicity (measured context only)",
                          "ethnicity", ETHNICITIES)
        row = self._combo(row, "Education (measured context only)",
                          "education", EDUCATIONS)
        row = self._entry(row, "Age gap in years (0-20)", "age_gap")
        row = self._check(row, "Same race as the date", "same_race")
        row = self._section(row, "Adult layer (research priors)")
        row = self._entry(row, "Kink compatibility (0-5)", "kink_match")
        row = self._entry(row, "Libido compatibility (0-5)", "libido_match")
        row = self._entry(row, "Your size in cm, men (8-18)", "size_cm")
        row = self._combo(row, "Your chest, women", "own_cup",
                          adult.CUP_SIZES)
        row = self._combo(row, "Their chest, dates who are women", "cup",
                          adult.CUP_SIZES)
        row = self._entry(row, "Preferred size in cm, dates who are men",
                          "partner_size_cm")
        row = self._combo(row, "Partner build", "partner_body",
                          adult.BODY_TYPES)
        row = self._entry(row, "Experience (0-5)", "experience")
        row = self._check(row, "Usually dates people like me",
                          "orientation_match")

    def _section(self, row, title):
        self.ttk.Label(self.form, text=title,
                       font=("TkDefaultFont", 10, "bold")).grid(
                           row=row, column=0, columnspan=3, sticky="w",
                           pady=(10, 4))
        return row + 1

    def _variable(self, key, initial):
        if key not in self.vars:
            self.vars[key] = self.tk.StringVar(value=str(initial))
        return self.vars[key]

    def _combo(self, row, label, key, values):
        self.ttk.Label(self.form, text=label).grid(row=row, column=0,
                                                  sticky="w", pady=2)
        variable = self._variable(key, default_form()[key])
        widget = self.ttk.Combobox(self.form, textvariable=variable,
                                   values=list(values), state="readonly",
                                   width=24)
        widget.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        variable.trace_add("write", lambda *args: self._refresh_fields())
        self.vars[key + "__widget"] = widget
        return row + 1

    def _entry(self, row, label, key):
        self.ttk.Label(self.form, text=label).grid(row=row, column=0,
                                                  sticky="w", pady=2)
        variable = self._variable(key, default_form()[key])
        widget = self.ttk.Entry(self.form, textvariable=variable, width=26)
        widget.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        self.vars[key + "__widget"] = widget
        return row + 1

    def _numbers(self, row, title, key, labels):
        self.ttk.Label(self.form, text=title,
                       font=("TkDefaultFont", 9, "bold")).grid(
                           row=row, column=0, columnspan=3, sticky="w",
                           pady=(6, 2))
        row += 1
        variables = []
        for i, name in enumerate(labels):
            self.ttk.Label(self.form, text=name).grid(row=row, column=0,
                                                      sticky="w")
            variable = self.tk.StringVar(value=default_form()[key][i])
            entry = self.ttk.Entry(self.form, textvariable=variable,
                                   width=10)
            entry.grid(row=row, column=1, sticky="w", pady=1)
            variables.append(variable)
            self.vars[f"{key}__{i}"] = entry
            row += 1
        self.lists[key] = variables
        return row

    def _check(self, row, label, key):
        variable = self.tk.BooleanVar(value=default_form()[key])
        widget = self.ttk.Checkbutton(self.form, text=label,
                                      variable=variable)
        widget.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        self.vars[key] = variable
        self.vars[key + "__widget"] = widget
        variable.trace_add("write", lambda *args: self._refresh_fields())
        return row + 1

    def _set_enabled(self, key, enabled):
        widget = self.vars.get(key + "__widget")
        if widget is None:
            return
        try:
            widget.state(["!disabled"] if enabled else ["disabled"])
        except self.tk.TclError:
            pass

    def _refresh_fields(self):
        mode = self.vars["mode"].get()
        gender = self.vars["gender"].get()
        partner = self.vars["partner_gender"].get()
        advanced = mode in ("advanced", "adult")
        adult_mode = mode == "adult"
        for key in ("height_cm", "body_type", "ethnicity", "education",
                    "age_gap", "same_race"):
            self._set_enabled(key, advanced)
        for key in ("kink_match", "libido_match", "partner_body",
                    "experience", "orientation_match"):
            self._set_enabled(key, adult_mode)
        self._set_enabled("size_cm", adult_mode and gender == "man")
        self._set_enabled("own_cup", adult_mode and gender == "woman")
        self._set_enabled("cup", adult_mode and partner == "women")
        self._set_enabled("partner_size_cm", adult_mode and partner == "men")

    def _collect(self):
        form = {
            "mode": self.vars["mode"].get(),
            "gender": self.vars["gender"].get(),
            "partner_gender": self.vars["partner_gender"].get(),
            "same_race": self.vars["same_race"].get(),
            "orientation_match": self.vars["orientation_match"].get(),
        }
        for key in ("age", "income", "goal", "height_cm", "body_type",
                    "ethnicity", "education", "age_gap", "kink_match",
                    "libido_match", "size_cm", "own_cup", "cup",
                    "partner_size_cm", "partner_body", "experience"):
            form[key] = self.vars[key].get()
        for key, variables in self.lists.items():
            form[key] = [variable.get() for variable in variables]
        return build_payload(form)

    def on_reset(self):
        defaults = default_form()
        for key, value in defaults.items():
            if key in self.lists:
                for variable, item in zip(self.lists[key], value):
                    variable.set(item)
            elif isinstance(self.vars.get(key), self.tk.BooleanVar):
                self.vars[key].set(bool(value))
            elif key in self.vars:
                self.vars[key].set(str(value))
        self._refresh_fields()
        self.status.set("Reset. Nothing leaves this machine.")

    def on_compute(self):
        try:
            payload = self._collect()
        except ValueError as exc:
            self.messagebox.showerror("Check one field", str(exc))
            return
        self.compute_button.state(["disabled"])
        self.status.set("Scoring the profile against the saved weights ...")
        thread = threading.Thread(target=self._worker, args=(payload,),
                                  daemon=True)
        thread.start()

    def _worker(self, payload):
        try:
            self.results.put((True, gui.compute(payload)))
        except Exception as exc:  # show engine errors instead of hanging
            self.results.put((False, str(exc)))

    def _poll(self):
        try:
            while True:
                ok, payload = self.results.get_nowait()
                if ok:
                    self._render(payload)
                    self.status.set("Done. This window did the maths "
                                    "on your machine.")
                else:
                    self.status.set(f"Failed: {payload}")
                    self.messagebox.showerror("Compute failed", payload)
                self.compute_button.state(["!disabled"])
        except queue.Empty:
            pass
        self.root.after(120, self._poll)

    def _render(self, result):
        text = format_result(result)
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", text + "\n")
        details = json.dumps(result, indent=2, sort_keys=True)
        self.details_text.delete("1.0", "end")
        self.details_text.insert("end", details + "\n")
        self._draw_heatmap(result.get("map") or {})
        bands = (result.get("map") or {}).get("bands", []) or []
        self.bands_text.delete("1.0", "end")
        for band in bands:
            self.bands_text.insert(
                "end",
                f"Package {band.get('level')}: "
                f"{_percent(band.get('mutual'))} mutual -- "
                f"{band.get('band')}\n")

    def _draw_heatmap(self, mapped):
        matrix = mapped.get("matrix", []) if mapped else []
        self.heat.delete("all")
        if len(matrix) != 10 or any(len(row) != 10 for row in matrix):
            self.heat.create_text(180, 180, text="Map unavailable",
                                  fill="#6b5f66")
            return
        top = max((float(value) for row in matrix for value in row),
                  default=0.0)
        top = top if top > 0 else 1.0
        size, cells = 340, 10
        cell = size / cells
        for i in range(cells):
            for j in range(cells):
                strength = float(matrix[i][j]) / top
                red = round(255 - 61 * strength)
                green = round(255 - 231 * strength)
                blue = round(255 - 164 * strength)
                x0, y0 = 10 + j * cell, 10 + i * cell
                self.heat.create_rectangle(x0, y0, x0 + cell, y0 + cell,
                                           fill=f"#{red:02x}{green:02x}"
                                                f"{blue:02x}",
                                           outline="#e6e6e6")
                self.heat.create_text(x0 + cell / 2, y0 + cell / 2,
                                      text=f"{100 * float(matrix[i][j]):.0f}",
                                      font=("TkDefaultFont", 7))
        your = int(mapped.get("your_level", 0)) - 1
        best = int(mapped.get("best_level", 0)) - 1
        if 0 <= your < 10 and 0 <= best < 10:
            x0, y0 = 10 + best * cell, 10 + your * cell
            self.heat.create_rectangle(x0, y0, x0 + cell, y0 + cell,
                                       outline="#111111", width=3)
        self.heat.create_text(180, 356,
                              text="Rows are men, columns are women",
                              fill="#6b5f66", font=("TkDefaultFont", 8))

    def run(self):
        self.root.mainloop()


def main(argv=None):
    """Open the desktop window."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.parse_args(argv)
    DesktopApp().run()


if __name__ == "__main__":
    main()
