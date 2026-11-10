# Done / needs to be done

Status: 2026-11-10.

## DONE

1. **Joined dataset.** 8,378 dates, 195 variables, income restored from the 15-column
   copy; row alignment proven on 11 columns before writing. `data/build.py`.
2. **18–25 dataset.** 3,625 real dates, trained separately: 79.8% CV / 82.3% holdout
   vs 55.6% baseline.
3. **Deeper models.** Generic MLP, any depth. Taste 16→24→12→1 reaches 80.0% CV
   (81.1% men, 80.3% women), up from 77.4% for the drawn 10×10, which is unchanged.
4. **Market models.** Trained on `dec_o` from self-ratings: 66.1% (men) / 63.9% (women).
   Weak, and reported as weak.
5. **Profile tool.** `regular` / `advanced` / `adult`, adjustment breakdown with
   `measured` / `population` / `research` labels, chance of mutual yes, hypergamy
   percentile, competitor rank.
6. **Hypergamy scale.** `visualize_scale.py`: your position, the position of the
   partner you asked for, and the distribution of competitors targeting them.
7. **Partial-effect chart.** `visualize_chance.py`: looks, ambition and income swept
   through both gender models; 18–25 vs full-sample income slope printed.
8. **Research priors.** Penile size (Veale 2015, Lever 2006), breast size (Swami &
   Tovée 2013, BSSS 2020), height (Stulp 2013), kink prevalence (Holvoet 2017, Joyal
   2017), desire discrepancy (Willoughby 2013), encoded with citations in `adult.py`.
9. **Population priors.** 59,946 OKCupid profiles reduced to height/body/ethnicity
   percentiles, `data/population_priors.json`.
10. **README** rebuilt with Mermaid diagrams, results, modes, sources and limits.

## NEEDS TO BE DONE

1. **No public adult-outcome dataset exists.** Everything adult is a prior. If one ever
   appears, replace `adult.py` constants with trained weights and say so.
2. **Partner income is missing** from the speed-dating data, so "status" is ambition
   plus your own income. A dataset with both partners' incomes (HCMST) would fix it.
3. **Draw layer 2** of the drawn network so the picture shows an output, not just a
   funnel.
4. **Per-gender fly test** — the stimulation experiment still runs on the base model.
5. **Calibration report** — temperature is fitted, but no reliability curve is plotted.
6. **Images and CSVs are not on GitHub.** The integration uploads text only; the
   `with-data` branch is local.

## Constraints

* Every arrow a real weight; every bar a re-run of the network.
* The 10×10 drawing stays 10×10.
* No statistic without a source; no synthetic dataset presented as real.
* Adult effects stay capped and labelled.
