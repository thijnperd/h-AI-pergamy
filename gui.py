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
import couples
import generate
import hypergamy_map
import likes
import research
import route
from profile import (DEFAULTS, calibration, competitors, have_data,
                      stated_hypergamy, support_warnings)
from partner import who_you_end_up_with

HERE = os.path.dirname(os.path.abspath(__file__))


def _json_safe(value):
    """NaN and Infinity are not JSON; strict clients and the tests reject them."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def compute(payload):
    """Run one profile through the routed models and return a JSON-able result."""
    gender = 1 if payload.get("gender") == "man" else 0
    profile = dict(DEFAULTS)
    for key in ("age", "income", "goal", "height_cm", "age_gap", "kink_match",
                "libido_match", "size_cm", "partner_size_cm", "experience"):
        if payload.get(key) is not None:
            profile[key] = float(payload[key])
    profile["own_cup"] = payload.get("own_cup", "C")
    profile["cup"] = payload.get("cup", "C")
    profile["partner_body"] = payload.get("partner_body", "average")
    for key in ("self", "want", "importance"):
        if isinstance(payload.get(key), list) and len(payload[key]) == len(profile[key]):
            profile[key] = [float(v) for v in payload[key]]
    profile["mode"] = payload.get("mode", "regular")
    profile["body_type"] = payload.get("body_type", "average")
    profile["ethnicity"] = payload.get("ethnicity", "not specified")
    profile["education"] = payload.get("education") or ""
    profile["same_race"] = bool(payload.get("same_race", True))
    profile["orientation_match"] = bool(payload.get("orientation_match", True))
    partner = 0 if payload.get("partner_gender", "women") == "women" else 1
    profile["partner_gender"] = partner

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

    mapped = hypergamy_map.partner_map(profile, gender, partner)
    liked = likes.who_likes_you(profile, gender, partner)
    result["likes"] = {
        "order": [int(k) + 1 for k in liked["order"]],
        "their_yes": [round(float(v), 4) for v in liked["their_yes"]],
        "your_yes": [round(float(v), 4) for v in liked["your_yes"]],
        "mutual": [round(float(v), 4) for v in liked["mutual"]],
        "best": int(liked["best"]),
        "best_chance": round(float(liked["best_chance"]), 4),
        "best_mutual": int(np.argmax(liked["mutual"])) + 1,
        "best_mutual_chance": round(float(np.max(liked["mutual"])), 4),
        "range": liked["range"],
        "range_low": liked["range_low"],
        "range_high": liked["range_high"],
        "range_low_chance": round(float(liked["range_low_chance"]), 4),
        "range_high_chance": round(float(liked["range_high_chance"]), 4),
    }
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
        "attractiveness": mapped["attractiveness"],
        "their_yes_by_package": mapped["their_yes_by_package"],
        "your_yes_by_package": mapped["your_yes_by_package"],
        "partner_gender": "men" if partner else "women",
        "goal_rating": float(np.mean(profile["want"][:5])),
    }
    result["person"] = generate.generate_person(profile, gender, partner)
    result["market_context"] = couples.context(profile, gender, partner)
    result["research"] = research.notes(profile, gender, partner)

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
    result["partner"] = who_you_end_up_with(profile, gender, partner)
    result["warnings"] = support_warnings(profile, gender, partner)
    result["calibration"] = calibration(gender)
    return _json_safe(result)


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
  .mapcell { font-size:11px; text-align:center; color:#555; padding:4px 0; border-radius:2px; }
  .mapcell.mine { outline:2px solid #111; font-weight:700; color:#111; }
  .help { font-size:12.5px; color:#6b5f66; line-height:1.45; margin-top:3px; }
  .verdict { background:#fdf0f5; border:1px solid #f3d4e1; border-radius:10px;
             padding:13px 15px; font-size:14.5px; margin-bottom:14px; }
  .arrows { margin-top:12px; border:1px solid #f0f0f0; border-radius:8px; padding:4px 6px; }
  .big small { display:block; font-size:11px; color:#6b5f66; margin-top:2px;
               text-transform:none; letter-spacing:0; }
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
  .likes { background:#f3f8f3; border:1px solid #d8e8d8; border-radius:10px;
           padding:11px 15px; margin-bottom:14px; font-size:13.5px; }
  .likes b { color:#2e7d32; font-size:14px; }
  .likes span { display:block; color:var(--grey); margin-top:3px; }
  .likes table { margin-top:7px; }
  .person { background:#f3f7fb; border:1px solid #d6e4f0; border-radius:10px;
            padding:12px 15px; margin-bottom:14px; font-size:13.5px; }
  .context { background:#f7f4fb; border:1px solid #e4dcf0; border-radius:10px;
             padding:11px 15px; margin-bottom:14px; font-size:13px; }
  .research { background:#fffaf0; border:1px solid #f0e2c8; border-radius:10px;
              padding:11px 15px; margin-bottom:14px; font-size:13px; }
  .research-item { margin-top:9px; }
  .research-item b { display:block; color:#8a5a00; }
  .research-item span { display:block; color:#333; margin-top:2px; }
  .research-item .help { margin-top:3px; }
  .context table { margin-top:6px; }
  .person b.name { color:#1b5e8a; font-size:15px; }
  .person .facts { display:grid; grid-template-columns:repeat(4,1fr); gap:6px 10px;
                   margin:8px 0; }
  .person .facts div { border-top:1px solid #e3edf5; padding-top:4px; }
  .person .facts span { display:block; font-size:11px; color:var(--grey);
                        text-transform:uppercase; letter-spacing:.05em; }
  .person .facts b { font-size:14px; }
  .person .rates { color:var(--grey); font-size:12px; }
  details.explain { margin-top:16px; font-size:12.5px; color:#555; }
  details.explain summary { cursor:pointer; color:var(--pink); font-weight:600; }
  details.explain p { margin:8px 0 0; }

  /* 2026 refresh: calmer surface, clearer hierarchy, accessible controls. */
  :root { --ink:#241b21; --grey:#6b5f66; --pink:#a62c52; --blue:#246b8e;
          --rule:#e7ddd7; --bg:#f5f0ea; --surface:#fffdfb; --soft:#fbf7f3; }
  body { background:radial-gradient(circle at 12% 0%,#fdf7f1 0,var(--bg) 38%,#f1ebe5 100%); }
  .site-header { max-width:1320px; margin:0 auto; padding:22px 24px 8px;
                 display:flex; align-items:center; justify-content:space-between; gap:18px; }
  .brand { display:flex; align-items:center; gap:12px; }
  .brand-mark { display:grid; place-items:center; width:42px; height:42px; border-radius:13px;
                background:linear-gradient(145deg,#b43a60,#7c2946); color:#fff;
                font:700 20px/1 Georgia,serif; box-shadow:0 8px 18px #7c294633; }
  h1 { font-size:23px; letter-spacing:.12em; }
  header p { max-width:620px; line-height:1.45; }
  .header-meta { display:flex; align-items:center; gap:7px; flex-wrap:wrap; justify-content:flex-end; }
  .pill { display:inline-flex; align-items:center; min-height:26px; padding:3px 9px;
          border:1px solid var(--rule); border-radius:999px; background:#fffaf6;
          color:var(--grey); font-size:11px; font-weight:600; white-space:nowrap; }
  .pill.local { color:#2e6b4f; border-color:#cfe2d7; background:#f2f9f4; }
  main.shell { max-width:1320px; margin:0 auto; padding:12px 24px 56px;
               grid-template-columns:minmax(0,1.02fr) minmax(0,.98fr); gap:20px; }
  .panel { background:var(--surface); border:1px solid var(--rule); border-radius:16px;
           padding:22px 24px; box-shadow:0 10px 30px #3b25220a; }
  .section-head { display:flex; align-items:flex-start; gap:11px; margin-bottom:14px; }
  .section-head h2 { margin:1px 0 2px; font-size:12px; letter-spacing:.12em; }
  .section-head p { margin:0; color:var(--grey); font-size:12px; line-height:1.35; }
  .step { display:grid; place-items:center; width:28px; height:28px; flex:0 0 28px;
          border-radius:9px; background:#f4e8ed; color:var(--pink);
          font:700 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace; }
  .form-panel h2:not(.section-head h2) { margin-top:22px; }
  .actions { display:grid; grid-template-columns:1fr auto; gap:10px; margin-top:20px; }
  #run { margin-top:0; padding:13px 18px; border-radius:10px; font-size:14px;
          box-shadow:0 7px 16px #a62c5226; }
  .ghost { margin-top:0; padding:11px 15px; border:1px solid var(--rule); border-radius:10px;
            background:#fffaf6; color:var(--grey); font:600 13px/1 inherit; }
  .ghost:hover { background:#f7eee9; color:var(--ink); }
  input, select { min-height:38px; border-color:#d9cec7; border-radius:9px; }
  input[type=range] { min-height:26px; cursor:pointer; }
  input:focus-visible, select:focus-visible, button:focus-visible, summary:focus-visible {
    outline:3px solid #246b8e55; outline-offset:2px; }
  .control { min-width:0; }
  .grid6 .caption { text-align:left; font-size:11px; margin:0 0 1px; white-space:nowrap;
                    overflow:hidden; text-overflow:ellipsis; }
  .val { text-align:right; color:var(--pink); font-size:11.5px; margin-top:-1px; }
  .range-limits { display:flex; justify-content:space-between; color:#6b5f66;
                  font-size:11.5px; line-height:1.2; margin-top:3px; }
  .big div { background:var(--soft); }
  .big div:first-child { border-color:#e9c9d4; background:#fff5f8; }
  .big div:first-child b { color:var(--pink); }
  .chip { display:inline-block; padding:2px 6px; border-radius:999px; font-size:11px;
          font-weight:700; letter-spacing:.02em; vertical-align:1px; }
  .chip.trained { background:#e8f3fb; color:#1b5e8a; }
  .chip.measured { background:#eaf6ed; color:#2e6b4f; }
  .chip.research { background:#fdf1dc; color:#8a5a00; }
  .result-panel { align-self:start; position:sticky; top:16px;
                   max-height:calc(100vh - 32px); overflow-y:auto;
                   overscroll-behavior:contain; scrollbar-width:thin; }
  .result-panel .section-head { margin-bottom:12px; }
  #result { min-height:180px; }
  .empty { border:0; border-top:1px solid var(--rule); border-radius:0; padding:18px 2px;
            color:var(--grey); background:none; font-size:13.5px; }
  .loading { display:flex; align-items:center; gap:9px; padding:20px 4px; color:var(--grey); }
  .spinner { width:17px; height:17px; border:2px solid #eadbe2; border-top-color:var(--pink);
             border-radius:50%; animation:spin .8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .explain summary { padding:3px 0; }
  @media (max-width: 980px) {
    main.shell { grid-template-columns:1fr; padding:10px 14px 40px; }
    .site-header { padding:16px 14px 6px; align-items:flex-start; }
    .header-meta { display:none; }
    .panel { padding:18px 16px; border-radius:14px; }
    .result-panel { position:static; max-height:none; overflow:visible; }
    .row3, .grid6 { grid-template-columns:1fr 1fr; }
  }
  @media (max-width: 560px) {
    .row, .row3, .grid6 { grid-template-columns:1fr; }
    .actions { grid-template-columns:1fr; }
    .ghost { width:100%; }
    .person .facts { grid-template-columns:repeat(2,1fr); }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration:.01ms !important; transition:none !important; }
  }

  .ico { width:18px; height:18px; vertical-align:-3px; margin-right:7px;
         stroke:currentColor; fill:none; stroke-width:2; stroke-linecap:round;
         stroke-linejoin:round; }
  .ico-lg { width:25px; height:25px; margin-right:9px; }
  .page-foot { max-width:1080px; margin:0 auto 26px; padding:0 24px;
               color:var(--grey); font-size:11.5px; line-height:1.5; }
  .page-foot a { color:var(--pink); }
</style></head><body>
<header class="site-header">
  <div class="brand">
    <span class="brand-mark" aria-hidden="true">H</span>
    <div><h1><svg class="ico ico-lg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>HYPERGAMY</h1>
      <p>Mutual yes, reaching, and the partner market &mdash; from saved weights.
        Nothing leaves this machine.</p></div>
  </div>
  <div class="header-meta"><span class="pill local">local only</span>
    <span class="pill">v1.4.1</span><span class="pill">trained + measured</span></div>
</header>
<main class="shell">
  <section class="panel form-panel" aria-labelledby="profile-title">
    <div class="section-head"><span class="step" aria-hidden="true">01</span>
      <div><h2 id="profile-title">Your profile</h2>
        <p>Start with the basics; open a deeper mode only when you need it.</p></div></div>
    <div class="row">
      <div><label for="mode">mode</label>
        <select id="mode">
          <option value="regular">regular &mdash; quick profile</option>
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
      <div style="margin-top:10px"><label for="ethnicity">your ethnicity (optional)</label>
        <select id="ethnicity"><option value="not specified" selected>not specified</option>
          <option value="white">white</option><option value="black">black</option>
          <option value="latino">Latino / Hispanic</option><option value="asian">Asian</option>
          <option value="pacific">Pacific Islander</option><option value="other">other / multi</option>
        </select>
        <div class="help">used only for the measured ACS couple context; it is not a
          trait score and it does not change the trained date decision.</div></div>
      <div style="margin-top:10px"><label for="education">education (optional)</label>
        <select id="education"><option value="" selected>not specified</option>
          <option value="high school or less">high school or less</option>
          <option value="some college">some college</option>
          <option value="bachelor">bachelor</option>
          <option value="graduate">graduate</option></select>
        <div class="help">optional: used only for the measured ACS same-education context.
          Leave it as not specified to use the overall table; it never enters the trained
          date decision.</div></div>
      <label style="margin-top:10px"><input id="same_race" type="checkbox" checked
        style="width:auto"> same race as the date</label>
    </div>

    <div id="adult" hidden>
      <h2 style="margin-top:18px">Adult layer</h2>
      <p class="help" style="margin:0 0 10px">Research priors, not trained outcomes.
        The anatomy fields adapt to who you are and who you are interested in.</p>
      <div class="row3">
        <div class="control"><label for="kink_match">kink match <b id="kink_match_v">3</b> / 5</label>
          <input id="kink_match" type="range" min="0" max="5" value="3"
            oninput="document.getElementById('kink_match_v').textContent=this.value">
          <div class="range-limits"><span>opposite</span><span>identical</span></div>
          <div class="help">0 = opposite, 5 = identical</div></div>
        <div class="control"><label for="libido_match">libido match <b id="libido_match_v">3</b> / 5</label>
          <input id="libido_match" type="range" min="0" max="5" value="3"
            oninput="document.getElementById('libido_match_v').textContent=this.value">
          <div class="range-limits"><span>opposite</span><span>identical</span></div>
          <div class="help">0 = opposite, 5 = identical</div></div>
        <div class="control" id="size_group"><label for="size_cm">your size <b id="size_cm_v">13.1</b> cm</label>
          <input id="size_cm" type="range" min="8" max="18" step="0.1" value="13.1"
            oninput="document.getElementById('size_cm_v').textContent=(+this.value).toFixed(1)">
          <div class="range-limits"><span>8 cm</span><span>18 cm</span></div>
          <div class="help">asked only when you are a man; 13.1 cm is the population median</div></div>
      </div>
      <div class="row3">
        <div class="control" id="own_cup_group" hidden><label for="own_cup">your chest (cup size)</label>
          <select id="own_cup"><option>A</option><option>B</option><option selected>C</option>
            <option>D</option><option>DD/E</option><option>F or larger</option></select>
          <div class="help">asked only when you are a woman; a small research prior</div></div>
        <div class="control" id="cup_group"><label for="cup">their chest (cup size)</label>
          <select id="cup"><option>A</option><option>B</option><option selected>C</option>
            <option>D</option><option>DD/E</option><option>F or larger</option></select>
          <div class="help">shown when you are interested in women; A small · C medium · F very large</div></div>
        <div class="control" id="partner_size_group" hidden>
          <label for="partner_size_cm">preferred penis size <b id="partner_size_cm_v">13.1</b> cm</label>
          <input id="partner_size_cm" type="range" min="8" max="18" step="0.1" value="13.1"
            oninput="document.getElementById('partner_size_cm_v').textContent=(+this.value).toFixed(1)">
          <div class="range-limits"><span>8 cm</span><span>18 cm</span></div>
          <div class="help">shown when you are interested in men; a small research prior, not a trained outcome</div></div>
      </div>
      <div class="row3">
        <div class="control"><label for="partner_body">partner build</label>
          <select id="partner_body"><option>thin</option><option>athletic</option>
            <option selected>average</option><option>curvy</option>
            <option>full figured</option><option>a little extra</option></select>
          <div class="help">same scale as your build</div></div>
        <div class="control"><label for="experience">experience <b id="experience_v">3</b> / 5</label>
          <input id="experience" type="range" min="0" max="5" value="3"
            oninput="document.getElementById('experience_v').textContent=this.value">
          <div class="range-limits"><span>none</span><span>a lot</span></div>
          <div class="help">0 = none, 5 = a lot</div></div>
        <div class="control"><label for="orientation_match">in their pool</label>
          <label style="display:flex; align-items:center; gap:8px; margin:8px 0 0; color:var(--ink); font-size:12.5px">
            <input id="orientation_match" type="checkbox" checked style="width:auto; min-height:auto">
            usually dates people like me</label>
          <div class="help">uncheck if you are outside the group they usually date</div></div>
      </div>
    </div>

    <div class="actions">
      <button id="run" type="button">Compute my result</button>
      <button id="reset" class="ghost" type="button">Reset</button>
    </div>
    <p class="muted" id="status" role="status" aria-live="polite">Ages 18-25 blend the
      18-25 model with the gender model. Nothing is uploaded.</p>
  </section>

  <section class="panel result-panel" aria-labelledby="result-title">
    <div class="section-head"><span class="step" aria-hidden="true">02</span>
      <div><h2 id="result-title"><svg class="ico" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/></svg>Result</h2>
        <p>Every number carries its basis: trained, measured, or research.</p></div></div>
    <div id="result" class="muted" aria-live="polite" aria-busy="false">
      <div class="empty">Set your profile, then press <b>Compute</b>. The result
        appears here and stays on this machine.</div>
    </div>
  </section>
</main>
<footer class="page-foot">Icons: <a href="https://lucide.dev">Lucide</a> (ISC).
  The server binds 127.0.0.1; no telemetry, no uploads.</footer>
<script>
const NAMES = ["looks","sincerity","intelligence","fun","ambition"];
const WANT = ["looks","sincerity","intelligence","fun","ambition","shared taste"];
const DEF_SELF = [5,5,5,5,3], DEF_WANT = [6,6,6,6,6,6], DEF_IMP = [30,10,15,15,20,10];

function grid(id, names, values, min, max, step, suffix) {
  document.getElementById(id).innerHTML = names.map((n, i) =>
    `<div class="control"><label class="caption" for="${id}_${i}">${n}</label>
     <input id="${id}_${i}" type="range" min="${min}" max="${max}" step="${step}"
       value="${values[i]}" aria-label="${n}"
       oninput="document.getElementById('${id}_${i}_v').textContent=this.value+'${suffix}'">
     <div class="range-limits"><span>${min}${suffix}</span><span>${max}${suffix}</span></div>
     <output class="val" id="${id}_${i}_v" for="${id}_${i}">${values[i]}${suffix}</output></div>`).join("");
}
grid("self", NAMES, DEF_SELF, 0, 10, 1, "");
grid("want", WANT, DEF_WANT, 0, 10, 1, "");
grid("importance", WANT, DEF_IMP, 0, 100, 5, "%");

function syncAnatomy() {
  const you = document.getElementById("gender").value;
  const partner = document.getElementById("partner_gender").value;
  document.getElementById("size_group").hidden = you !== "man";
  document.getElementById("own_cup_group").hidden = you !== "woman";
  document.getElementById("cup_group").hidden = partner !== "women";
  document.getElementById("partner_size_group").hidden = partner !== "men";
}

document.getElementById("gender").addEventListener("change", (e) => {
  document.getElementById("partner_gender").value =
    e.target.value === "man" ? "women" : "men";
  syncAnatomy();
});

document.getElementById("partner_gender").addEventListener("change", syncAnatomy);

document.getElementById("mode").addEventListener("change", (e) => {
  const m = e.target.value;
  document.getElementById("advanced").hidden = !(m === "advanced" || m === "adult");
  document.getElementById("adult").hidden = m !== "adult";
  syncAnatomy();
});

document.getElementById("reset").addEventListener("click", () => location.reload());
syncAnatomy();

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
    ethnicity: document.getElementById("ethnicity").value,
    education: document.getElementById("education").value,
    age_gap: num("age_gap"), same_race: document.getElementById("same_race").checked,
    kink_match: num("kink_match"), libido_match: num("libido_match"),
    size_cm: num("size_cm"), own_cup: document.getElementById("own_cup").value,
    cup: document.getElementById("cup").value,
    partner_size_cm: num("partner_size_cm"),
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
  const like = r.likes || null;
  const likeY = like ? y(3 + 7 * (like.best - 1) / 9) : 0;
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
    ${like ? `<circle cx="${xGoal}" cy="${likeY}" r="12" fill="none" stroke="#2e7d32"
      stroke-width="2" stroke-dasharray="3 3"/>` : ""}
    ${like ? `<text x="${xGoal + (youMan ? 20 : -20)}" y="${Math.max(26, likeY - 12)}"
      font-size="10.5" font-weight="700" text-anchor="${youMan ? "start" : "end"}"
      fill="#2e7d32">likes you ${(100 * like.best_chance).toFixed(0)}%</text>` : ""}
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
      ringed red = you · green dotted = likes you most · pink/blue = over/under</text>
  </svg></div>`;
}

function likes_panel(r) {
  if (!r.likes) return "";
  const i = r.likes.best - 1;
  const rows = r.likes.order.slice(0, 5).map(k => {
    const j = k - 1;
    return `<tr><td>package ${k}</td>
      <td>${pct(r.likes.their_yes[j])} say yes to you</td>
      <td>${pct(r.likes.mutual[j])} mutual</td></tr>`;
  }).join("");
  return `<div class="likes">
    <b>Who likes you most: package ${r.likes.best}</b>
    <span>${pct(r.likes.their_yes[i])} of that package would say yes to you;
      you would say yes ${pct(r.likes.your_yes[i])}; both ${pct(r.likes.mutual[i])}.</span>
    <span>Best mutual: package ${r.likes.best_mutual}
      (${pct(r.likes.best_mutual_chance)}).</span>
    <span>Range most likely to say yes: packages ${r.likes.range_low}-${r.likes.range_high}
      (${pct(r.likes.range_low_chance)}-${pct(r.likes.range_high_chance)}).</span>
    <table>${rows}</table>
  </div>`;
}

function person_panel(r) {
  if (!r.person) return "";
  const p = r.person;
  const facts = [
    ["age", p.age.toFixed(1)],
    ["income", "$" + Math.round(p.income).toLocaleString()],
    ["build", p.body_type],
    ["height", p.height_cm.toFixed(0) + " cm"],
    [p.cup ? "chest" : "size", p.cup || (p.size_cm.toFixed(1) + " cm")],
    ["kink", p.kink_match.toFixed(0) + "/5"],
    ["libido", p.libido_match.toFixed(0) + "/5"],
    ["experience", p.experience.toFixed(0) + "/5"]
  ];
  const rates = Object.entries(p.rates_you).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(" · ");
  return `<div class="person">
    <b class="name">The single person who likes you most: ${p.gender} ${p.age.toFixed(1)}</b>
    <div class="facts">${facts.map(([k, v]) =>
      `<div><span>${k}</span><b>${v}</b></div>`).join("")}</div>
    <div class="rates">How ${p.gender === "woman" ? "she" : "he"} rates you: ${rates}</div>
    <div style="margin-top:6px"><b>${pct(p.p_they)}</b> say yes to you ·
      <b>${pct(p.p_you)}</b> you say yes · <b>${pct(p.mutual)}</b> mutual</div>
    <div class="help">Package ${p.package} template, generated by the new person model from
      your profile. Age, income and self-ratings come from the package; the body, height,
      chest/size, kink, libido and experience are population/research priors, not trained
      outcomes. The age/rating model is trained, but its cross-validated error is no better
      than predicting the average, so treat this person as a sketch, not a forecast.
      The range most likely to say yes is ${p.range.low}-${p.range.high}
      (${pct(p.range.low_chance)}-${pct(p.range.high_chance)}).</div>
  </div>`;
}

function partner_panel(r) {
  const p = r.partner;
  if (!p) return "";
  if (!p.available) {
    return `<h2 style="margin-top:16px">Who you end up with</h2>
      <div class="context"><p style="margin:0">${p.note || "not available for this pairing."}</p></div>`;
  }
  const t = p.typical;
  const money = v => v == null ? "n/a" : "$" + Math.round(v).toLocaleString();
  const one = (v, s) => v == null ? "n/a" : Number(v).toFixed(1) + (s || "");
  const pct1 = v => v == null ? "n/a" : Number(v).toFixed(1) + "%";
  const rows = p.candidates.map(c => `<tr>
      <td>${c.age.toFixed(0)} yrs, ${money(c.income)}</td>
      <td>${c.age_quantile} age, ${c.income_quantile} income</td></tr>`).join("");
  const ladder = p.ladder_level == null ? "" :
    `<p style="margin:8px 0 0">The typical partner's self-ratings sit at
     <b>${p.ladder_level.toFixed(1)} / 10</b> on the package ladder &mdash; measured from
     real participants (${p.self_basis}), not modelled.</p>`;
  return `<h2 style="margin-top:16px"><svg class="ico" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>Who you most plausibly end up with</h2>
    <div class="context">
      <p style="margin:0 0 4px">People with your age and income pair with a partner whose median
      age is <b>${one(t.age, " yrs")}</b>, income <b>${money(t.income)}</b>, age gap
      <b>${t.age_gap == null ? "n/a" : (t.age_gap > 0 ? "+" : "") + one(t.age_gap, " yrs")}</b>.
      Same ethnicity in <b>${pct1(t.same_ethnicity_pct)}</b> of those couples, same education
      in <b>${pct1(t.same_education_pct)}</b>.</p>
      ${ladder}
      <table>${rows}</table>
      <div class="help">${p.note}<br><br>${p.stability_note}</div>
    </div>`;
}

function market_panel(r) {
  const c = r.market_context;
  if (!c || !c.available) return '';
  const cell = c.cell || c.overall;
  if (!cell) return '';
  const age = cell.partner_age || {};
  const inc = cell.partner_income || {};
  const gap = cell.age_gap || {};
  const ratio = cell.income_ratio || {};
  const money = (v) => v == null ? 'n/a' : '$' + Math.round(v).toLocaleString();
  const one = (v, suffix) => v == null ? 'n/a' : Number(v).toFixed(1) + (suffix || '');
  const pct1 = (v) => v == null ? 'n/a' : Number(v).toFixed(1) + '%';
  const h = c.height || {};
  const eth = c.ethnicity_cell;
  const ethRows = eth ? `<tr><td>same ethnicity, ${c.ethnicity} couples</td>
    <td>${pct1(eth.same_eth_pct)} (n=${eth.n.toLocaleString()})</td></tr>` : '';
  const edu = c.education_cell;
  const eduRows = edu ? `<tr><td>same education, ${c.education} couples</td>
    <td>${pct1(edu.same_edu_pct)} (n=${edu.n.toLocaleString()})</td></tr>` : '';
  const heightRow = h.your_percentile == null ? '' :
    `<tr><td>your measured height percentile</td>
      <td>${h.your_percentile.toFixed(0)} (NHANES, n=${h.n.toLocaleString()})</td></tr>`;
  return `<h2 style="margin-top:16px"><svg class="ico" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>Measured couple market</h2>
    <div class="context">
      <p style="margin:0 0 4px"><b>${c.n_total.toLocaleString()}</b> real opposite-sex couples from
      ${c.states.join(', ')}. This cell: ${c.age_band} age, ${c.income_band} income,
      n=${cell.n.toLocaleString()} (${c.match}).</p>
      <table>
        <tr><td>partner age, median</td><td>${one(age.median, ' yrs')} (IQR ${one(age.p25, '')}-${one(age.p75, '')})</td></tr>
        <tr><td>partner personal income, median</td><td>${money(inc.median)}</td></tr>
        <tr><td>age gap, median</td><td>${gap.median == null ? 'n/a' : (gap.median > 0 ? '+' : '') + one(gap.median, ' yrs')}</td></tr>
        <tr><td>partner income / yours, median</td><td>${ratio.median == null ? 'n/a' : Number(ratio.median).toFixed(2) + 'x'}</td></tr>
        <tr><td>same ethnicity</td><td>${pct1(cell.same_eth_pct)}</td></tr>
        <tr><td>same education</td><td>${pct1(cell.same_edu_pct)}</td></tr>
        ${heightRow}${ethRows}${eduRows}
      </table>
      <div class="help">ACS 2023 1-year PUMS, opposite-sex married or cohabiting couples.
      These are <b>realised couples</b>, not dating decisions and not a prediction of who
      would say yes. Income is personal income, including zeros. Counts are unweighted
      sample counts; this is not a national estimate.</div>
    </div>`;
}

function research_panel(r) {
  const q = r.research;
  if (!q) return '';
  const item = (heading, x) => `<div class="research-item"><b>${heading}</b>
    <span>${x.summary}</span><div class="help">${x.detail} <em>${x.applied}</em></div></div>`;
  const sources = q.sources.map(s => `<li><b>${s.label}</b> &mdash; ${s.finding}</li>`).join('');
  return `<h2 style="margin-top:16px"><svg class="ico" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/></svg>What the research says</h2>
    <div class="research">
      ${item(q.income_age.title, q.income_age)}
      ${item(q.height.title, q.height)}
      ${item(q.types.title, q.types)}
      <details class="explain"><summary>Sources</summary><ul>${sources}</ul></details>
    </div>`;
}

function render(r) {
  const chip = (basis) => `<span class="chip ${basis}">${basis}</span>`;
  const adj = r.adjustments.length ? `<h2 style="margin-top:16px">Adjustments</h2>
    <table>${r.adjustments.map(a =>
      `<tr><td>${a.label}</td><td>${a.pp > 0 ? "+" : ""}${a.pp.toFixed(1)} pp ${chip(a.basis)}</td></tr>`).join("")}</table>` : "";
  const hyper = r.hypergamy ? `<h2 style="margin-top:16px">Ambition vs looks</h2>
    <p style="margin:0">${r.hypergamy.percentile == null ? "n/a" : r.hypergamy.percentile.toFixed(1)} / 100 among ${r.gender === "man" ? "men" : "women"}:
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
    return `<h2 style="margin-top:16px"><svg class="ico" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M14.106 5.553a2 2 0 0 0 1.788 0l3.659-1.83A1 1 0 0 1 21 4.619v12.764a1 1 0 0 1-.553.894l-4.553 2.277a2 2 0 0 1-1.788 0l-4.212-2.106a2 2 0 0 0-1.788 0l-3.659 1.83A1 1 0 0 1 3 19.381V6.618a1 1 0 0 1 .553-.894l4.553-2.277a2 2 0 0 1 1.788 0z"/><path d="M15 5.764v15"/><path d="M9 3.236v15"/></svg>Who you can reach</h2>
      <p class="muted" style="margin:0 0 8px">rows are men, columns are women, each cell is the chance you both say yes. M1-M10 and W1-W10 are ten made-up packages (looks, age and income moving together), not ten real people and not a score out of ten.</p>
      ${grid}
      <div class="arrows">${r.map.arrows}</div>
      <h2 style="margin-top:16px">In plain numbers</h2>
      <table>
        <tr><td><b>attractiveness</b> &mdash; dates who would say yes to you</td>
          <td>${(r.p_them * 100).toFixed(0)}%</td></tr>
        <tr><td>your chance at your own package (${r.map.your_level})</td>
          <td>${(r.map.chance_your_level * 100).toFixed(0)}%</td></tr>
        <tr><td>best mutual package (${r.map.best_level})</td>
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

  const hero = r.map ? `<div class="hero">
    <div class="num">${r.map.hypergamy >= 0 ? "+" : ""}${r.map.hypergamy.toFixed(1)}</div>
    <div><b>reaching</b><span>how far above your own ratings you aim, on a -10 to +10
    scale. +10 = you rate yourself 0 and want 10s; -10 = you are a 10 who wants 0s.
    The three main chances are above; this is the aim gap.</span></div>
  </div>` : "";
  const league = r.map ? league_svg(r) : "";
  const big = `<div class="big">
      <div><span>both say yes</span><b>${pct(r.mutual)}</b><small>the number that matters</small></div>
      <div><span>they say yes</span><b>${pct(r.p_them)}</b><small>your attractiveness score</small></div>
      <div><span>you say yes</span><b>${pct(r.p_you)}</b><small>your side, taste model</small></div>
    </div>`;
  const verdict = r.map ? `<div class="verdict">${r.map.verdict}</div>` : "";

  const warn = r.warnings && r.warnings.length
    ? `<div class="research" style="background:#fff5f8;border-color:#f3d4e1">
        ${r.warnings.map(w => `<div>${w}</div>`).join("")}</div>` : "";
  document.getElementById("result").innerHTML = `
    ${warn}${big}${hero}${verdict}${league}${likes_panel(r)}${person_panel(r)}${partner_panel(r)}
    ${adj}${hyper}${market}${map}${market_panel(r)}${research_panel(r)}
    ${r.map && r.map.same_gender ? `<p class="muted" style="margin-top:12px">Same-gender
      pairing: the study only recorded opposite-gender dates, so read these numbers as an
      extrapolation, not a measured result.</p>` : ""}
    <details class="explain"><summary>What these numbers mean</summary>
      <p><b>both say yes</b> — the chance that you and this kind of person both want a second
      date. <b>they say yes</b> comes from the market model, <b>you say yes</b> from the taste model.</p>
      <p><b>the single person</b> — a new multi-output model reads your full profile and
      picks the package whose age and likelihood fit best; the card fills in every field.
      The body, height, chest/size, kink, libido and experience are population/research
      priors because the speed-dating data has no outcome for them. Age, income and the
      package's self-ratings are the model's template, not a claim about a real person.</p>
      <p><b>who likes you most</b> — the two dyad models score each package on how likely
      it is to say yes to you, using its own ratings as well as its stated importance.
      The green dotted ring is that package; the table shows the top five and the mutual
      chance, which also depends on whether you would say yes.</p>
      <p><b>bands</b> — every level is compared to your best cell: 60% or more is likely,
      25% to 60% is stretch, below 25% is hard.</p>
      <p><b>packages</b> — the ten rows and columns are made-up bundles of looks, age and
      income, numbered 1 to 10. Package 10 is not "the perfect woman", it is just the top
      rung the model builds; every package has roughly 10% of the market.</p>
      <p><b>hypergamousness (reaching)</b> — how far above your own ratings you aim, from -10 to +10.
      Rate yourself 0 everywhere and want 10s and it is +10; be a 10 who wants 0s and it is
      -10. It is not a judgement, just the gap between what you offer and what you look for.</p>
      <p><b>attractiveness</b> — how many dates in the study pool would say yes to you,
      according to the market model. It is about 64% accurate, so read it as a rough market
      estimate, not as a score of your worth. "Rank vs levels" is different: it compares you
      with the ten made-up packages, and it is not a rating of you either.</p>
      <p><b>who you end up with</b> — measured from 250,172 ACS couples: the median partner
      age, income and age gap for people with your age and income. The ACS has no partner
      self-ratings, so those candidates are measured profiles, not dyad-scored. It is not a
      lifetime forecast; no shipped data follows a person to a partner.</p>
      <p><b>calibration</b> — ${r.calibration ? `on ${r.calibration.n.toLocaleString()}
      ${r.gender === "man" ? "male" : "female"} rows the model's own predictions are compared
      with what actually happened: ${r.calibration.taste.slice(0, 5).map(b => `${b.range}
      predicted ${b.predicted}, observed ${b.observed}`).join("; ")}. The ends are compressed,
      so read the headline as a rank, not a frequency.` : "needs data/build.py."}</p>
    </details>
    <h2 style="margin-top:16px"><svg class="ico" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>Models used</h2>
    <p class="muted" style="margin:0 0 6px">${r.routing}</p>
    <ul>${models}</ul>`;
}

document.getElementById("run").addEventListener("click", async () => {
  const button = document.getElementById("run");
  const status = document.getElementById("status");
  const result = document.getElementById("result");
  button.disabled = true; status.textContent = "running ...";
  result.setAttribute("aria-busy", "true");
  result.innerHTML = `<div class="loading"><span class="spinner" aria-hidden="true"></span>
    Scoring the profile against the saved weights...</div>`;
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
    result.innerHTML = `<p class="err" role="alert">${err.message}</p>`;
    status.textContent = "failed";
  } finally {
    result.setAttribute("aria-busy", "false");
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
        elif self.path == "/api/health":
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
