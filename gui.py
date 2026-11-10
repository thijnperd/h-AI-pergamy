"""A local GUI for the whole project -- one page, every model.

No toolkit to install: the interface is plain HTML served by Python's standard
library, and the maths stays on your machine. Run it, the browser opens, fill in
the form, press Compute. Nothing is sent anywhere.

The app routes the profile to the right weights (see route.py):

  age 18-25, or the 18-25 mode -> the gender model and the 18-25 model, averaged
  any other age               -> the gender model only

  python gui.py                 # http://127.0.0.1:8000
  python gui.py --port 9000
  python gui.py --no-browser
"""

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import adult
import route
from profile import DEFAULTS, competitors, have_data, stated_hypergamy

HERE = os.path.dirname(os.path.abspath(__file__))


def compute(payload):
    """Run one profile through the routed models and return a JSON-able result."""
    gender = 1 if payload.get("gender") == "man" else 0
    profile = dict(DEFAULTS)
    for key in ("age", "income", "goal", "height_in", "age_gap", "kink_match",
                "libido_match", "genital", "cup", "experience"):
        if payload.get(key) is not None:
            profile[key] = float(payload[key])
    for key in ("self", "want", "importance"):
        if isinstance(payload.get(key), list) and len(payload[key]) == len(profile[key]):
            profile[key] = [float(v) for v in payload[key]]
    profile["mode"] = payload.get("mode", "regular")
    profile["body_type"] = payload.get("body_type", "average")
    profile["same_race"] = bool(payload.get("same_race", True))
    profile["orientation_match"] = bool(payload.get("orientation_match", True))

    p_you, taste_used = route.predict_you(profile, gender)
    p_them, market_used = route.predict_them(profile, gender)
    d_them, d_you, rows = adult.adjustments(profile, gender)
    p_you = float(np.clip(p_you + d_you / 100.0, 0.01, 0.99))
    p_them = float(np.clip(p_them + d_them / 100.0, 0.01, 0.99))

    result = {
        "gender": "man" if gender else "woman",
        "age": profile["age"],
        "income": profile["income"],
        "p_you": p_you,
        "p_them": p_them,
        "mutual": p_you * p_them,
        "adjustments": [{"label": label, "pp": delta, "basis": basis}
                        for label, delta, basis in rows if abs(delta) > 0.05],
        "taste_models": taste_used,
        "market_models": market_used,
        "routing": route.explain(profile["age"], profile["mode"], gender)["reason"],
        "hypergamy": None,
        "market": None,
    }

    if have_data():
        _, percentile, _ = stated_hypergamy(profile, gender)
        comp = competitors(profile, gender)
        result["hypergamy"] = {
            "percentile": percentile,
            "you_tilt": 100.0 * profile["self"][4]
                        / max(profile["self"][4] + profile["self"][0], 1e-9),
            "partner_tilt": 100.0 * profile["want"][4]
                            / max(profile["want"][4] + profile["want"][0], 1e-9),
        }
        result["market"] = {
            "score": comp["my"],
            "rank": comp["rank"],
            "n": comp["n"],
        }
    return result


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HYPERGAMY</title>
<style>
  :root { --ink:#111; --grey:#6b6b6b; --pink:#c2185b; --blue:#1b9ad6; --rule:#e6e6e6; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         color:var(--ink); background:#fafafa; }
  header { padding:22px 28px 10px; }
  h1 { margin:0; font-size:26px; letter-spacing:.5px; }
  header p { margin:4px 0 0; color:var(--grey); font-size:13px; }
  main { display:grid; grid-template-columns:minmax(360px,1fr) minmax(380px,1fr);
         gap:18px; padding:14px 28px 40px; align-items:start; }
  section { background:#fff; border:1px solid var(--rule); border-radius:10px; padding:16px 18px; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.08em; color:var(--pink);
       margin:0 0 10px; }
  label { display:block; font-size:12px; color:var(--grey); margin:8px 0 2px; }
  input, select { width:100%; padding:6px 8px; border:1px solid #d8d8d8; border-radius:6px;
                  font:inherit; background:#fff; }
  .row { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .row3 { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
  .grid6 { display:grid; grid-template-columns:repeat(6,1fr); gap:6px; }
  .grid6 input { padding:4px; text-align:center; }
  .caption { font-size:11px; color:var(--grey); text-align:center; }
  button { margin-top:16px; width:100%; padding:11px; border:0; border-radius:8px;
           background:var(--pink); color:#fff; font:600 15px/1 inherit; cursor:pointer; }
  button:disabled { opacity:.6; cursor:progress; }
  .big { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:14px; }
  .big div { border:1px solid var(--rule); border-radius:8px; padding:10px; }
  .big b { display:block; font-size:24px; }
  .big span { font-size:11px; color:var(--grey); text-transform:uppercase; letter-spacing:.06em; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  td { padding:3px 0; border-bottom:1px solid #f2f2f2; }
  td:last-child { text-align:right; color:var(--grey); }
  ul { margin:6px 0 0; padding-left:16px; font-size:12.5px; color:var(--grey); }
  .muted { color:var(--grey); font-size:12.5px; }
  .err { color:#b00020; font-size:13px; }
</style></head><body>
<header>
  <h1>HYPERGAMY</h1>
  <p>chance of a mutual yes &middot; hypergamy percentile &middot; competitor rank &mdash;
     all from saved weights, nothing leaves this machine</p>
</header>
<main>
  <section>
    <h2>Your profile</h2>
    <div class="row">
      <div><label for="mode">mode</label>
        <select id="mode">
          <option value="regular">regular &mdash; 10 numbers</option>
          <option value="advanced">advanced &mdash; + height, build, age gap</option>
          <option value="adult">adult &mdash; + kink, libido, size</option>
          <option value="18-25">18-25 &mdash; force the young model</option>
        </select></div>
      <div><label for="gender">you are</label>
        <select id="gender"><option value="man">man</option><option value="woman">woman</option></select></div>
    </div>
    <div class="row3">
      <div><label for="age">age</label><input id="age" type="number" value="24" min="18" max="80"></div>
      <div><label for="income">income $/yr</label><input id="income" type="number" value="32000" step="1000"></div>
      <div><label for="goal">goal (1-6)</label><input id="goal" type="number" value="4" min="1" max="6"></div>
    </div>

    <h2 style="margin-top:18px">How you rate yourself</h2>
    <div class="grid6" id="self"></div>

    <h2 style="margin-top:18px">What you want</h2>
    <div class="grid6" id="want"></div>

    <h2 style="margin-top:18px">How much it matters (%)</h2>
    <div class="grid6" id="importance"></div>

    <div id="advanced" hidden>
      <h2 style="margin-top:18px">Advanced</h2>
      <div class="row3">
        <div><label for="height_in">height (in)</label><input id="height_in" type="number" value="70"></div>
        <div><label for="body_type">build</label>
          <select id="body_type">
            <option>thin</option><option>athletic</option><option selected>average</option>
            <option>curvy</option><option>full figured</option><option>a little extra</option>
          </select></div>
        <div><label for="age_gap">age gap (yrs)</label><input id="age_gap" type="number" value="3"></div>
      </div>
      <label style="margin-top:10px"><input id="same_race" type="checkbox" checked
        style="width:auto"> same race as the date</label>
    </div>

    <div id="adult" hidden>
      <h2 style="margin-top:18px">Adult layer</h2>
      <div class="row3">
        <div><label for="kink_match">kink match 0-5</label><input id="kink_match" type="number" value="3" min="0" max="5"></div>
        <div><label for="libido_match">libido 0-5</label><input id="libido_match" type="number" value="3" min="0" max="5"></div>
        <div><label for="genital">size 1-5</label><input id="genital" type="number" value="3" min="1" max="5"></div>
      </div>
      <div class="row3">
        <div><label for="cup">partner build 1-5</label><input id="cup" type="number" value="3" min="1" max="5"></div>
        <div><label for="experience">experience 0-5</label><input id="experience" type="number" value="3" min="0" max="5"></div>
        <div><label>&nbsp;</label><label><input id="orientation_match" type="checkbox" checked
          style="width:auto"> in their pool</label></div>
      </div>
    </div>

    <button id="run">Compute</button>
    <p class="muted" id="status">Ages 18-25 blend the 18-25 model with the gender model.</p>
  </section>

  <section>
    <h2>Result</h2>
    <div id="result" class="muted">Press Compute.</div>
  </section>
</main>
<script>
const NAMES = ["looks","sincerity","intelligence","fun","ambition"];
const WANT = ["looks","sincerity","intelligence","fun","ambition","shared taste"];
const DEF_SELF = [7,7,7,7,5], DEF_WANT = [8,6,7,8,9,7], DEF_IMP = [30,10,15,15,20,10];

function grid(id, names, values) {
  document.getElementById(id).innerHTML = names.map((n, i) =>
    `<div><div class="caption">${n}</div>
     <input id="${id}_${i}" type="number" min="0" max="100" value="${values[i]}"></div>`).join("");
}
grid("self", NAMES, DEF_SELF);
grid("want", WANT, DEF_WANT);
grid("importance", WANT, DEF_IMP);

document.getElementById("mode").addEventListener("change", (e) => {
  const m = e.target.value;
  document.getElementById("advanced").hidden = !(m === "advanced" || m === "adult");
  document.getElementById("adult").hidden = m !== "adult";
});

const num = (id) => Number(document.getElementById(id).value);

function payload() {
  return {
    mode: document.getElementById("mode").value,
    gender: document.getElementById("gender").value,
    age: num("age"), income: num("income"), goal: num("goal"),
    self: [0,1,2,3,4].map(i => Number(document.getElementById("self_" + i).value)),
    want: [0,1,2,3,4,5].map(i => Number(document.getElementById("want_" + i).value)),
    importance: [0,1,2,3,4,5].map(i => Number(document.getElementById("importance_" + i).value)),
    height_in: num("height_in"), body_type: document.getElementById("body_type").value,
    age_gap: num("age_gap"), same_race: document.getElementById("same_race").checked,
    kink_match: num("kink_match"), libido_match: num("libido_match"),
    genital: num("genital"), cup: num("cup"), experience: num("experience"),
    orientation_match: document.getElementById("orientation_match").checked
  };
}

const pct = (x) => (100 * x).toFixed(1) + "%";

function render(r) {
  const adj = r.adjustments.length ? `<h2 style="margin-top:16px">Adjustments</h2>
    <table>${r.adjustments.map(a =>
      `<tr><td>${a.label}</td><td>${a.pp > 0 ? "+" : ""}${a.pp.toFixed(1)} pp &nbsp;[${a.basis}]</td></tr>`).join("")}</table>` : "";
  const hyper = r.hypergamy ? `<h2 style="margin-top:16px">Hypergamy</h2>
    <p style="margin:0">${r.hypergamy.percentile.toFixed(1)} / 100 among ${r.gender === "man" ? "men" : "women"}.
    Your tilt ${r.hypergamy.you_tilt.toFixed(0)}/100, the partner you want ${r.hypergamy.partner_tilt.toFixed(0)}/100.</p>` : "";
  const market = r.market ? `<h2 style="margin-top:16px">The market</h2>
    <p style="margin:0">Score ${r.market.score.toFixed(0)}/100, top ${(100 - r.market.rank).toFixed(0)}%
    of ${r.market.n.toLocaleString()} competitors whose stated wants resemble your partner.</p>`
    : `<p class="muted" style="margin-top:14px">Competitor rank needs data/ &mdash; run data/build.py.</p>`;
  const models = r.taste_models.concat(r.market_models).map(m =>
    `<li>${m.path}${m.cv ? " &mdash; CV " + (100 * m.cv).toFixed(1) + "%" : ""}
     ${m.arch ? "&mdash; " + m.arch.join("&rarr;") : ""}</li>`).join("");

  document.getElementById("result").innerHTML = `
    <div class="big">
      <div><span>both say yes</span><b>${pct(r.mutual)}</b></div>
      <div><span>they say yes</span><b>${pct(r.p_them)}</b></div>
      <div><span>you say yes</span><b>${pct(r.p_you)}</b></div>
    </div>
    ${adj}${hyper}${market}
    <h2 style="margin-top:16px">Models used</h2>
    <p class="muted" style="margin:0 0 6px">${r.routing}</p>
    <ul>${models}</ul>`;
}

document.getElementById("run").addEventListener("click", async () => {
  const button = document.getElementById("run");
  const status = document.getElementById("status");
  button.disabled = true; status.textContent = "running ...";
  try {
    const response = await fetch("/api/compute", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload())
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "request failed");
    render(data);
    status.textContent = "done";
  } catch (err) {
    document.getElementById("result").innerHTML = `<p class="err">${err.message}</p>`;
    status.textContent = "failed";
  } finally {
    button.disabled = false;
  }
});
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type="application/json"):
        blob = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/compute":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, json.dumps(compute(payload)))
        except FileNotFoundError as exc:
            self._send(400, json.dumps({
                "error": f"missing weights file: {exc}. Train it with the commands in the README."
            }))
        except Exception as exc:
            self._send(400, json.dumps({"error": str(exc)}))

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="local GUI for the HYPERGAMY models")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"HYPERGAMY GUI  ->  {url}")
    print("press Ctrl+C to stop")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
