"""What the published research supports, and what it does not.

The app has three different kinds of claim and must not blend them:

* **trained** -- a network fitted to 8,378 speed-date outcomes;
* **measured** -- a descriptive statistic from a real dataset (ACS couples, NHANES
  heights, the speed-date age-gap table);
* **research** -- a published effect used as a small, capped prior, with a citation.

This module is the third bucket.  It also contains the one height interaction used by
`adult.py`: shorter women weight height somewhat more heavily, following the field
experiment by Ong (2014).  It is a prior, not a trained coefficient.

The most important answer here is a negative one: the large lifespan studies do **not**
find a reliable decline in the importance of height with age, and the evidence on
income is mixed.  We therefore do not invent an age x height multiplier.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))


SOURCES = {
    "fales2016": {
        "label": "Fales et al. 2016, Personality and Individual Differences 88:78-87",
        "url": "https://gwern.net/doc/sociology/2016-fales.pdf",
        "finding": "Women's preference for steady income rose from age 18 to about 40, "
                   "then slowly declined; preference for 'makes a lot of money' did "
                   "not vary by age.",
    },
    "botzet2023": {
        "label": "Botzet et al. 2023, Human Nature 34:539-568",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10739319/",
        "finding": "17,254 single heterosexual women aged 18-67 in 147 countries: "
                   "age had negligible effects on financial security, attractiveness, "
                   "education/intelligence and confidence preferences.",
    },
    "schwarz2012": {
        "label": "Schwarz & Hassebrauck 2012, Human Nature 23:447-466",
        "url": "https://link.springer.com/article/10.1007/s12110-012-9152-x",
        "finding": "21,245 heterosexual adults aged 18-65: sex explained much more "
                   "than age; age betas across 82 criteria were small (-0.001 to 0.159).",
    },
    "plos2021": {
        "label": "Whyte et al. 2021, PLOS ONE, n=7,325 online daters",
        "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0250151",
        "finding": "Younger people regarded income as less important than older people; "
                   "the sex difference in income preference was not significant.",
    },
    "stulp2013": {
        "label": "Stulp et al. 2013, Animal Behaviour 86:37-46",
        "url": "https://www.gertstulp.com/pdf/Stulp%20et%20al%202013_Anim%20Behav_The%20height%20of%20choosiness.pdf",
        "finding": "5,782 speed-daters, 128,104 choices: women were most likely to choose "
                   "a man 25 cm taller, men a woman 7 cm shorter, and realised matches "
                   "peaked at a 19 cm gap.",
    },
    "ong2014": {
        "label": "Ong 2014, Height and Income Attraction field experiment",
        "url": "https://english.phbs.pku.edu.cn/info/3361/20601.htm",
        "finding": "360 artificial profiles and nearly 4,000 visits: women preferred "
                   "higher-income men; short women were most willing to trade income "
                   "for height; men were largely indifferent to women's income.",
    },
    "height2022": {
        "label": "2022 Frontiers in Psychology cross-cultural height study",
        "url": "https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2022.937146/full",
        "finding": "536 adults aged 15-77 in Canada, Cuba, Norway and the US: men's height "
                   "preferences were weaker and less consistent than women's; preferences "
                   "were positively assortative, not a single universal ideal.",
    },
    "hitsch2010": {
        "label": "Hitsch, Hortacsu & Ariely 2010, Quantitative Marketing and Economics 8:393-427",
        "url": "https://link.springer.com/article/10.1007/s11129-010-9088-6",
        "finding": "Online daters showed strong same-race preferences; for women these "
                   "did not differ by age, income or education, and for men only slightly.",
    },
    "hwang2013": {
        "label": "Hwang 2013, Race and Social Problems 5:28-40",
        "url": "https://link.springer.com/article/10.1007/s12552-012-9082-6",
        "finding": "2,123 online dating profiles: same-race willingness was generally "
                   "high, out-dating lower, and racial status shaped women's willingness.",
    },
}



SOURCES.update({
    "england2016": {
        "label": "England 2016, Journal of Marriage and Family 78:1184-1194",
        "url": "https://onlinelibrary.wiley.com/doi/abs/10.1111/jomf.12314",
        "finding": "National Survey of Families and Households: age heterogamy itself did "
                   "not raise divorce odds; both men and women were more likely to leave "
                   "when their spouse was older than they were.",
    },
    "kraft2009": {
        "label": "Kraft & Neimann 2009, IZA Discussion Paper 4491",
        "url": "https://docs.iza.org/dp4491.pdf",
        "finding": "German data: the risk of divorce rose with the absolute age difference; "
                   "a difference in education did not affect marital stability.",
    },
    "iza2023": {
        "label": "IZA Discussion Paper 15749 (2022), educational structure of marriage",
        "url": "https://docs.iza.org/dp15749.pdf",
        "finding": "US newlyweds: a four-year within-couple age gap was associated with a "
                   "16 percentage-point higher probability of divorce within four years "
                   "than the same age; educationally homogamous marriages were most stable.",
    },
    "hcmst": {
        "label": "Rosenfeld, Thomas & Falcon, HCMST (Stanford / ICPSR 30103)",
        "url": "https://data.stanford.edu/hcmst",
        "finding": "The public panel that follows couples across waves and records "
                   "breakup and marriage; it is the right data for a lifetime-outcome "
                   "model, needs its own registration, and is not shipped or trained here.",
    },
})

def height_slope_multiplier(gender, height_cm):
    """A small own-height interaction for the height-gap prior.

    Ong (2014) found short women most willing to trade income for height.  This is
    the only defensible direction for a prior, so the multiplier is deliberately
    narrow: 0.90x to 1.15x.  There is no age multiplier because the lifespan studies
    do not support one.
    """
    if gender != 0 or not height_cm:
        return 1.0
    h = float(height_cm)
    if h < 160:
        return 1.15
    if h > 172:
        return 0.90
    return 1.0


def income_age_summary(age):
    age = float(age)
    if age < 25:
        return ("At younger ages, income is usually rated less important; the measured "
                "steady-income preference rises toward about age 40.")
    if age < 40:
        return ("This is the age range where the measured preference for steady income "
                "is near its peak.  The effect is small and not a rule for one person.")
    if age < 55:
        return ("After about 40, the measured preference for steady income slowly "
                "declines; other traits become relatively more important.")
    return ("In older samples, steady income is less central on average, while "
            "companionship and health matter more.  Again, this is a group average.")


def notes(profile, gender, partner_gender):
    """Small, JSON-able evidence notes for the GUI."""
    age = float(profile.get("age", 24.0))
    h = profile.get("height_cm")
    height_mult = height_slope_multiplier(gender, h)
    return {
        "income_age": {
            "title": "Does money matter less with age?",
            "summary": income_age_summary(age),
            "detail": "Fales et al. 2016 found steady-income importance rising to about "
                      "40 then declining. Botzet et al. 2023 (17,254 women, 18-67) found "
                      "negligible overall age effects, and Schwarz & Hassebrauck 2012 "
                      "(21,245 adults) found small age effects. The honest answer is "
                      "'a small average curve, not a law'.",
            "applied": "not applied as a probability shift: the speed-date data has no "
                       "partner-income preference question to train it on.",
        },
        "height": {
            "title": "Does height matter less with age?",
            "summary": "The large lifespan studies do not show a reliable decline in the "
                       "importance of height with age. The robust findings are about "
                       "own height, gender and the height gap, not age.",
            "detail": "Stulp et al. 2013 (5,782 speed-daters) found preferred gaps of "
                      "+25 cm for women and -7 cm for men, with realised matches near "
                      "+19 cm. Ong 2014 found short women most willing to trade income "
                      "for height. The 2022 cross-cultural study found weaker and less "
                      "consistent height preferences in men.",
            "applied": f"a small own-height multiplier of {height_mult:.2f}x on the "
                       "height-gap prior; no age multiplier was added.",
        },
        "types": {
            "title": "Do 'types' like specific people?",
            "summary": "There is good evidence for same-race and same-education sorting, "
                       "and for height/income trade-offs that vary by own height. There "
                       "is no public dataset here that identifies 'goth', 'Latina goth', "
                       "or any subculture, so the app cannot honestly learn that label.",
            "detail": "Hitsch et al. 2010 found strong same-race preferences that did not "
                      "vary much by age, income or education. Hwang 2013 found same-race "
                      "willingness high and out-dating lower. The ACS couple file shows "
                      "82.8% same-ethnicity couples in these five states, but that is "
                      "realised couples, not a dating decision.",
            "applied": "shown as measured same-ethnicity/same-education context; the "
                       "only ethnicity input in the profile is the measured same-race "
                       "flag; the ethnicity dropdown is context, not a model feature.",
        },
        "stability": {
            "title": "Does a big age gap or a different education make a couple break up?",
            "summary": "The published studies agree on a direction, not a number: larger "
                       "age gaps are associated with more dissolution, while the education "
                       "effect is weak or absent in some samples and a stability advantage "
                       "in others.",
            "detail": "Kraft & Neimann 2009 found divorce risk rising with the absolute age "
                      "difference and no education effect. IZA 15749 reports a four-year age "
                      "gap going with a 16 percentage-point higher divorce probability within "
                      "four years among US newlyweds, and a homogamy advantage. England 2016 "
                      "finds the direction is about who is older, not heterogeneity alone.",
            "applied": "used only to order the candidate partners in the 'who you end up "
                       "with' panel; no probability is attached and it says nothing about "
                       "one particular relationship.",
        },
        "sources": list(SOURCES.values()),
    }
