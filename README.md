# UniSentiment — University Sentiment Intelligence System

An aspect-based sentiment analysis tool for student feedback, built with Flask, NLTK, and TextBlob.

> Group 12 · Intro to Intelligent Systems (INF 402) · Dr. Patrick Gyaase
>
> **Status: Work in progress** — still being built out by the team.

## What it does

Instead of giving a single sentiment score for a whole comment, UniSentiment breaks feedback down
by **aspect** (professors, Wi-Fi, parking, library, cafeteria, administration, campus safety,
facilities). A comment like *"the professors are great but the Wi-Fi is terrible"* is scored
positive for professors and negative for Wi-Fi separately, instead of averaging out to neutral.

It also handles:
- **Negation** ("not bad" ≠ full-strength "good")
- **Sarcasm** via pattern matching, which overrides literal word polarity
- **Explainability** — every score comes with the keywords that drove it
- **Auto-generated insights** — Fix First / Keep Doing / Controversial aspect rankings
- **Alerts** — threshold-based, plus a 14-day trend drop alert
- An interactive dashboard (Chart.js + Tabler UI) and CSV export

## Files

| File | What it is |
|---|---|
| `main.py` | Flask app — dashboard, routes, request handling |
| `sentiment_engine.py` | All NLP logic — tokenization, scoring, negation, sarcasm, aspects, insights, alerts |

## Running it

```bash
pip install flask nltk textblob
python main.py
```

The app starts a local Flask server and opens the dashboard in your browser.
