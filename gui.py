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
import hypergamy_map
import route
from profile import DEFAULTS, competitors, have_data, stated_hypergamy

HERE = os.path.dirname(os.path.abspath(__file__))


def compute(payload):
    """Run one profile through the routed models and return a JSON-able result."""
    gender = 1 if payload.get("gender") == "man" else 0
    profile = dict(DEFAULTS)
    for key in ("age", "income", "goal", "height_cm", "age_gap", "kink_match",
                "libido_match", "size_cm", "experience"):
        if payload.get(key) is not None:
            profile[key] = float(payload[key])
    profile["cup"] = payload.get("cup", "C")
    profile["partner_body"] = payload.get("partner_body", "average")
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

    partner = 0 if payload.get("partner_gender", "women") == "women" else 1
    profile["partner_gender"] = partner
    mapped = hypergamy_map.partner_map(profile, gender, partner)
    result["map"] = {
        "matrix": [[round(float(v), 4) for v in row] for row in mapped["matrix"]],
        "your_probs": [round(float(v), 4) for v in mapped["your_probs"]],
        "best_level": mapped["best_level"],
        "your_level": mapped["your_level"],
        "bands": mapped["bands"],
        "chance": mapped["chance"],
        "chance_your_level": mapped["chance_your_level"],
        "chance_typical": mapped["chance_typical"],
        "percentile": mapped["percentile"],
        "hypergamy": mapped["hypergamy"],
        "verdict": hypergamy_map.verdict(mapped),
        "arrows": hypergamy_map.arrows_svg(mapped),
        "same_gender": mapped["same_gender"],
        "attractiveness": mapped["their_yes"],
        "partner_gender": "men" if partner else "women",
        "goal_rating": float(np.mean(profile["want"][:5])),
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
  body { margin:0; font:16px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
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
  .grid6 { display:grid; grid-template-columns:repeat(3,1fr); gap:8px 10px; }
  .grid6 .caption { font-size:11px; margin-bottom:2px; }
  input[type=range] { padding:0; border:0; background:transparent; accent-color:var(--pink); }
  .val { text-align:center; font-weight:600; font-size:12.5px; margin-top:-2px; }
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
  .maprow { display:grid; grid-template-columns:repeat(10,1fr); gap:2px; margin-bottom:2px; }
  .mapcell { font-size:9px; text-align:center; color:#555; padding:3px 0; border-radius:2px; }
  .mapcell.mine { outline:2px solid #111; font-weight:700; color:#111; }
  .help { font-size:11px; color:#8a8a8a; line-height:1.3; margin-top:3px; }
  .verdict { background:#fdf0f5; border:1px solid #f3d4e1; border-radius:10px;
             padding:13px 15px; font-size:14.5px; margin-bottom:14px; }
  .hero { display:flex; gap:14px; align-items:center; background:#fdf0f5;
          border:1px solid #f3d4e1; border-radius:10px; padding:12px 16px;
          margin-bottom:12px; }
  .hero .num { font-size:42px; font-weight:700; line-height:1; color:var(--pink);
               min-width:92px; }
  .hero b { font-size:11px; text-transform:uppercase; letter-spacing:.08em;
            color:var(--pink); }
  .hero span { display:block; font-size:12px; color:var(--grey); margin-top:4px; }
  .league { border:1px solid #f0f0f0; border-radius:8px; padding:4px 6px;
            margin-bottom:12px; }
  .arrows { margin-top:12px; border:1px solid #f0f0f0; border-radius:8px; padding:4px 6px; }
  .big small { display:block; font-size:10px; color:#8a8a8a; margin-top:2px;
               text-transform:none; letter-spacing:0; }
  details.explain { margin-top:16px; font-size:12.5px; color:#555; }
  details.explain summary { cursor:pointer; color:var(--pink); font-weight:600; }
  details.explain p { margin:8px 0 0; }
</style></head><body>
<header>
  <h1>HYPERGAMY</h1>
  <p>hypergamousness &middot; attractiveness &middot; chance of a mutual yes &mdash;
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
    <div class="row">
      <div><label for="partner_gender">interested in</label>
        <select id="partner_gender"><option value="women" selected>women</option>
          <option value="men">men</option></select></div>
      <div><label>&nbsp;</label>
        <div class="help">the study only has opposite-gender dates; picking your own gender
          still works but is an extrapolation</div></div>
    </div>
    <div class="row3">
      <div><label for="age">age <b id="age_v">24</b></label>
        <input id="age" type="range" min="18" max="80" value="24"
          oninput="document.getElementById('age_v').textContent=this.value"></div>
      <div><label for="income">income $/yr</label><input id="income" type="number" value="32000" step="1000"></div>
      <div><label for="goal">goal <b id="goal_v">4</b> / 6</label>
        <input id="goal" type="range" min="1" max="6" value="4"
          oninput="document.getElementById('goal_v').textContent=this.value">
        <div class="help">the study codes 4 as a serious relationship</div></div>
    </div>

    <h2 style="margin-top:18px">How you rate yourself</h2>
    <p class="help" style="margin:0 0 8px">0 = terrible · 5 = average · 10 = perfect.
      Higher is better: 8 is better than 4.</p>
    <div class="grid6" id="self"></div>

    <h2 style="margin-top:18px">What you want</h2>
    <p class="help" style="margin:0 0 8px">the looks/sincerity/... you would like a date to have,
      same 0-10 scale, 10 = perfect</p>
    <div class="grid6" id="want"></div>

    <h2 style="margin-top:18px">How much each thing matters</h2>
    <p class="help" style="margin:0 0 8px">0% = does not matter, 100% = the only thing that matters.
      The weights are rescaled, so they do not have to add up to 100.</p>
    <div class="grid6" id="importance"></div>

    <div id="advanced" hidden>
      <h2 style="margin-top:18px">Advanced</h2>
      <div class="row3">
        <div><label for="height_cm">height <b id="height_cm_v">178</b> cm</label>
          <input id="height_cm" type="range" min="140" max="210" value="178"
            oninput="document.getElementById('height_cm_v').textContent=this.value">
          <div class="help">178 cm = 5 ft 10</div></div>
        <div><label for="body_type">your build</label>
          <select id="body_type">
            <option>thin</option><option>athletic</option><option selected>average</option>
            <option>curvy</option><option>full figured</option><option>a little extra</option>
          </select>
          <div class="help">thin = narrow · athletic = fit · average = neither · curvy = waist and
            hips · full figured = soft · a little extra = heavier</div></div>
        <div><label for="age_gap">age gap <b id="age_gap_v">3</b> yrs</label>
          <input id="age_gap" type="range" min="0" max="20" value="3"
            oninput="document.getElementById('age_gap_v').textContent=this.value"></div>
      </div>
      <label style="margin-top:10px"><input id="same_race" type="checkbox" checked
        style="width:auto"> same race as the date</label>
    </div>

    <div id="adult" hidden>
      <h2 style="margin-top:18px">Adult layer</h2>
      <div class="row3">
        <div><label for="kink_match">kink match <b id="kink_match_v">3</b> / 5</label>
          <input id="kink_match" type="range" min="0" max="5" value="3"
            oninput="document.getElementById('kink_match_v').textContent=this.value">
          <div class="help">0 = opposite, 5 = identical</div></div>
        <div><label for="libido_match">libido match <b id="libido_match_v">3</b> / 5</label>
          <input id="libido_match" type="range" min="0" max="5" value="3"
            oninput="document.getElementById('libido_match_v').textContent=this.value">
          <div class="help">0 = opposite, 5 = identical</div></div>
        <div><label for="size_cm">your size <b id="size_cm_v">13.1</b> cm</label>
          <input id="size_cm" type="range" min="8" max="18" step="0.1" value="13.1"
            oninput="document.getElementById('size_cm_v').textContent=(+this.value).toFixed(1)">
          <div class="help">13.1 cm is the population median</div></div>
      </div>
      <div class="row3">
        <div><label for="cup">partner chest</label>
          <select id="cup"><option>A</option><option>B</option><option selected>C</option>
            <option>D</option><option>DD/E</option><option>F or larger</option></select>
          <div class="help">A small · C medium · F very large</div></div>
        <div><label for="partner_body">partner build</label>
          <select id="partner_body"><option>thin</option><option>athletic</option>
            <option selected>average</option><option>curvy</option>
            <option>full figured</option><option>a little extra</option></select>
          <div class="help">same scale as your build</div></div>
        <div><label for="experience">experience <b id="experience_v">3</b> / 5</label>
          <input id="experience" type="range" min="0" max="5" value="3"
            oninput="document.getElementById('experience_v').textContent=this.value">
          <div class="help">0 = none, 5 = a lot</div></div>
      </div>
      <label style="margin-top:10px"><input id="orientation_match" type="checkbox" checked
        style="width:auto"> in their pool</label>
      <div class="help">uncheck if you are outside the group they usually date</div>
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
const DEF_SELF = [5,5,5,5,3], DEF_WANT = [6,6,6,6,6,6], DEF_IMP = [30,10,15,15,20,10];

function grid(id, names, values, min, max, step, suffix) {
  document.getElementById(id).innerHTML = names.map((n, i) =>
    `<div><div class="caption">${n}</div>
     <input id="${id}_${i}" type="range" min="${min}" max="${max}" step="${step}"
       value="${values[i]}"
       oninput="document.getElementById('${id}_${i}_v').textContent=this.value+'${suffix}'">
     <div class="val" id="${id}_${i}_v">${values[i]}${suffix}</div></div>`).join("");
}
grid("self", NAMES, DEF_SELF, 0, 10, 1, "");
grid("want", WANT, DEF_WANT, 0, 10, 1, "");
grid("importance", WANT, DEF_IMP, 0, 100, 5, "%");

document.getElementById("gender").addEventListener("change", (e) => {
  document.getElementById("partner_gender").value =
    e.target.value === "man" ? "women" : "men";
});

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
    partner_gender: document.getElementById("partner_gender").value,
    age: num("age"), income: num("income"), goal: num("goal"),
    self: [0,1,2,3,4].map(i => Number(document.getElementById("self_" + i).value)),
    want: [0,1,2,3,4,5].map(i => Number(document.getElementById("want_" + i).value)),
    importance: [0,1,2,3,4,5].map(i => Number(document.getElementById("importance_" + i).value)),
    height_cm: num("height_cm"), body_type: document.getElementById("body_type").value,
    age_gap: num("age_gap"), same_race: document.getElementById("same_race").checked,
    kink_match: num("kink_match"), libido_match: num("libido_match"),
    size_cm: num("size_cm"), cup: document.getElementById("cup").value,
    partner_body: document.getElementById("partner_body").value,
    experience: num("experience"),
    orientation_match: document.getElementById("orientation_match").checked
  };
}

const pct = (x) => (100 * x).toFixed(1) + "%";

function league_svg(r) {
  const youMan = r.gender === "man";
  const xYou = youMan ? 250 : 430, xGoal = youMan ? 430 : 250;
  const y = (v) => 330 - (Math.max(0, Math.min(10, v)) / 10) * 270;
  const mine = Math.max(0, Math.min(10, r.p_them * 10));
  const goal = Math.max(0, Math.min(10, r.map.goal_rating));
  const delta = goal - mine, above = delta > 0.25, below = delta < -0.25;
  const colour = above ? "#c2185b" : below ? "#1b9ad6" : "#111";
  const balls = (x) => Array.from({length: 10}, (_, i) => {
    const v = i + 1, near = x === xGoal && Math.abs(v - goal) <= 0.5;
    return `<circle cx="${x}" cy="${y(v)}" r="${near ? 8 : 6}" fill="${near ? colour : "#e4e4e4"}" stroke="${near ? colour : "#c9c9c9"}"/>`;
  }).join("");
  const dx = xGoal - xYou, dy = y(goal) - y(mine), len = Math.hypot(dx, dy) || 1;
  const sx = xYou + dx / len * 17, sy = y(mine) + dy / len * 17;
  const ex = xGoal - dx / len * 17, ey = y(goal) - dy / len * 17;
  const mx = (sx + ex) / 2, my = (sy + ey) / 2;
  const word = above ? `${delta.toFixed(1)} above your league` :
               below ? `${Math.abs(delta).toFixed(1)} below your league` :
               "level with your league";
  return `<div class="league"><svg viewBox="0 0 660 380" width="100%" role="img"
      aria-label="you and the partner you aim for, by attractiveness">
    <defs><marker id="lg-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6"
      markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z"
      fill="${colour}"/></marker></defs>
    <text x="${xYou}" y="34" font-size="12" font-weight="700" text-anchor="middle"
      fill="#111">${youMan ? "MEN" : "WOMEN"}</text>
    <text x="${xGoal}" y="34" font-size="12" font-weight="700" text-anchor="middle"
      fill="#111">${youMan ? "WOMEN" : "MEN"}</text>
    <text x="150" y="${y(10) + 4}" font-size="10" text-anchor="end" fill="#8a8a8a">10</text>
    <text x="150" y="${y(5) + 4}" font-size="10" text-anchor="end" fill="#8a8a8a">5</text>
    <text x="150" y="${y(0) + 4}" font-size="10" text-anchor="end" fill="#8a8a8a">0</text>
    <text x="150" y="${y(5) - 18}" font-size="10" text-anchor="end" fill="#8a8a8a">attractiveness</text>
    ${balls(xYou)}${balls(xGoal)}
    <circle cx="${xYou}" cy="${y(mine)}" r="15" fill="none" stroke="#c2185b" stroke-width="2.5"/>
    <circle cx="${xYou}" cy="${y(mine)}" r="8.5" fill="#c2185b"/>
    <text x="${xYou + (youMan ? -22 : 22)}" y="${y(mine) + 4}" font-size="11" font-weight="700"
      text-anchor="${youMan ? "end" : "start"}" fill="#c2185b">you ${mine.toFixed(1)}</text>
    <line x1="${sx}" y1="${sy}" x2="${ex}" y2="${ey}" stroke="${colour}" stroke-width="2.4"
      marker-end="url(#lg-arrow)"/>
    <text x="${mx}" y="${my - 8}" font-size="11" font-weight="700" text-anchor="middle"
      fill="${colour}">aim ${goal.toFixed(1)}</text>
    <text x="${mx}" y="${my + 15}" font-size="10.5" text-anchor="middle"
      fill="${colour}">${word}</text>
    <text x="330" y="356" font-size="10.5" text-anchor="middle" fill="#8a8a8a">
      left: men · right: women · balls 1-10 = attractiveness</text>
    <text x="330" y="372" font-size="10.5" text-anchor="middle" fill="#8a8a8a">
      ringed red = you · pink arrow = over your league · blue = under</text>
  </svg></div>`;
}

function render(r) {
  const hero = r.map ? `<div class="hero">
    <div class="num">${r.map.hypergamy >= 0 ? "+" : ""}${r.map.hypergamy.toFixed(1)}</div>
    <div><b>hypergamousness (reaching)</b><span>how far above your own ratings you aim,
    on a -10 to +10 scale. +10 = you rate yourself 0 and want 10s; -10 = you are a 10 who
    wants 0s. Read this first.</span></div>
  </div>` : "";
  const league = r.map ? league_svg(r) : "";
  const adj = r.adjustments.length ? `<h2 style="margin-top:16px">Adjustments</h2>
    <table>${r.adjustments.map(a =>
      `<tr><td>${a.label}</td><td>${a.pp > 0 ? "+" : ""}${a.pp.toFixed(1)} pp &nbsp;[${a.basis}]</td></tr>`).join("")}</table>` : "";
  const hyper = r.hypergamy ? `<h2 style="margin-top:16px">Ambition vs looks</h2>
    <p style="margin:0">${r.hypergamy.percentile.toFixed(1)} / 100 among ${r.gender === "man" ? "men" : "women"}:
    how much importance you put on ambition compared with looks.
    Your tilt ${r.hypergamy.you_tilt.toFixed(0)}/100, the partner you want ${r.hypergamy.partner_tilt.toFixed(0)}/100.</p>` : "";
  const market = r.market ? `<h2 style="margin-top:16px">The market</h2>
    <p style="margin:0">Score ${r.market.score.toFixed(0)}/100, top ${(100 - r.market.rank).toFixed(0)}%
    of ${r.market.n.toLocaleString()} competitors whose stated wants resemble your partner.</p>`
    : `<p class="muted" style="margin-top:14px">Competitor rank needs data/ &mdash; run data/build.py.</p>`;
  const map = r.map ? (() => {
    const max = Math.max(...r.map.matrix.flat(), 1e-9);
    const grid = r.map.matrix.map((row, i) => `<div class="maprow">` + row.map((v, j) => {
      const t = v / max;
      const bg = `rgb(${Math.round(255 - 61 * t)},${Math.round(255 - 231 * t)},${Math.round(255 - 164 * t)})`;
      const mine = (i === r.map.your_level - 1 && j === r.map.best_level - 1) ? " mine" : "";
      return `<div class="mapcell${mine}" style="background:${bg}" title="M${i + 1} x W${j + 1}: ${(v * 100).toFixed(0)}%">${(v * 100).toFixed(0)}</div>`;
    }).join("") + `</div>`).join("");
    const bands = r.map.bands.map(b => `<li>package ${b.level}: ${(b.mutual * 100).toFixed(0)}% mutual &mdash; ${b.band}</li>`).join("");
    return `<h2 style="margin-top:16px">Who you can reach</h2>
      <p class="muted" style="margin:0 0 8px">rows are men, columns are women, each cell is the chance you both say yes. M1-M10 and W1-W10 are ten made-up packages (looks, age and income moving together), not ten real people and not a score out of ten.</p>
      ${grid}
      <div class="arrows">${r.map.arrows}</div>
      <h2 style="margin-top:16px">In plain numbers</h2>
      <table>
        <tr><td><b>attractiveness</b> &mdash; dates who would say yes to you</td>
          <td>${(r.p_them * 100).toFixed(0)}%</td></tr>
        <tr><td>your chance at your own package (${r.map.your_level})</td>
          <td>${(r.map.chance_your_level * 100).toFixed(0)}%</td></tr>
        <tr><td>your best-fitting package (${r.map.best_level}), the easiest to reach</td>
          <td>${(r.map.chance * 100).toFixed(0)}%</td></tr>
        <tr><td>a middle package (5)</td>
          <td>${(r.map.chance_typical * 100).toFixed(0)}%</td></tr>
        <tr><td>reaching &mdash; how far above your own ratings you aim (-10 to +10)</td>
          <td>${r.map.hypergamy >= 0 ? "+" : ""}${r.map.hypergamy.toFixed(1)}</td></tr>
      </table>
      <ul>${bands}</ul>`;
  })() : "";

  const models = r.taste_models.concat(r.market_models).map(m =>
    `<li>${m.path}${m.cv ? " &mdash; CV " + (100 * m.cv).toFixed(1) + "%" : ""}
     ${m.arch ? "&mdash; " + m.arch.join("&rarr;") : ""}</li>`).join("");

  document.getElementById("result").innerHTML = `
    ${hero}${league}
    ${r.map ? `<div class="verdict">${r.map.verdict}</div>` : ""}
    <div class="big">
      <div><span>both say yes</span><b>${pct(r.mutual)}</b><small>the number that matters</small></div>
      <div><span>they say yes</span><b>${pct(r.p_them)}</b><small>your attractiveness score</small></div>
      <div><span>you say yes</span><b>${pct(r.p_you)}</b><small>your side, taste model</small></div>
    </div>
    ${adj}${hyper}${market}${map}
    ${r.map && r.map.same_gender ? `<p class="muted" style="margin-top:12px">Same-gender
      pairing: the study only recorded opposite-gender dates, so read these numbers as an
      extrapolation, not a measured result.</p>` : ""}
    <details class="explain"><summary>What these numbers mean</summary>
      <p><b>both say yes</b> — the chance that you and this kind of person both want a second
      date. <b>they say yes</b> comes from the market model, <b>you say yes</b> from the taste model.</p>
      <p><b>hypergamousness (reaching)</b> — how far above your own ratings you aim, from -10 to +10.
      Rate yourself 0 everywhere and want 10s and it is +10; be a 10 who wants 0s and it is
      -10. It is not a judgement, just the gap between what you offer and what you look for.</p>
      <p><b>the league graph</b> — men on the left, women on the right, attractiveness 0 at the
      bottom to 10 at the top. The ringed red ball is you, the arrow runs from you to the
      attractiveness you say you want, and the gap is printed next to it. Pink means above
      your league, blue means below, black means level.</p>
      <p><b>packages</b> — the ten rows and columns are made-up bundles of looks, age and
      income, numbered 1 to 10. Package 10 is not "the perfect woman", it is just the top
      rung the model builds; every package has roughly 10% of the market.</p>
      <p><b>attractiveness</b> — how many dates in the study pool would say yes to you,
      according to the market model. It is about 64% accurate, so read it as a rough market
      estimate, not as a score of your worth.</p>
      <p><b>bands</b> — every package is compared to your best cell: 60% or more is likely,
      25% to 60% is stretch, below 25% is hard.</p>
    </details>
    <h2 style="margin-top:16px">Models used</h2>
    <p class="muted" style="margin:0 0 6px">${r.routing}</p>
    <ul>${models}</ul>`;
}

document.getElementById("run").addEventListener("click", async () => {
  const button = document.getElementById("run");
  const status = document.getElementById("status");
  button.disabled = true; status.textContent = "running ...";
  try {
    const response = await fetch("/api/compute/" + Date.now(), {
      method: "POST", cache: "no-store", headers: {"Content-Type": "application/json"},
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
        path = self.path.split("?")[0]
        if path in ("/", "/index.html") or path == "/app":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/health":
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if not self.path.split("?")[0].startswith("/api/compute"):
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
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind (use 0.0.0.0 to expose it)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}/"
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
