# MAIN.PY - Flask Web Application for UniSentiment

# This file contains the web layer of UniSentiment.
# 
# What it does:
# - Serves the dashboard HTML/CSS/JavaScript
# - Handles API requests (/analyse, /single, /export_csv, /batches, /load_sample)
# - Orchestrates the NLP engine to process comments
# - Returns JSON results or CSV downloads
# - Auto-opens browser when you run it
# 
# This file depends on sentiment_engine.py for all NLP logic.


# IMPORTS 
# io: For handling in-memory file operations (CSV generation without saving to disk)
import io
# csv: For writing CSV export files with proper formatting
import csv
# json: For parsing incoming JSON requests and generating JSON responses from Flask
import json
# webbrowser: For automatically opening the dashboard in the user's default browser
import webbrowser
# threading: For running the browser opener in a separate thread without blocking the Flask server
import threading
# datetime: For generating timestamps for analysis results and CSV filenames
from datetime import datetime
# defaultdict: For aggregating aspect scores without key errors - automatically creates missing keys
from collections import defaultdict
# Flask: Web framework for serving the dashboard HTML and REST API endpoints
from flask import Flask, request, jsonify, Response


# IMPORT SENTIMENT ENGINE - All NLP Logic

# These functions come from sentiment_engine.py
# We only import what we need for the web layer.


# IMPORT SENTIMENT ENGINE 
# Import all the NLP functions from our sentiment_engine.py file
from sentiment_engine import (
    score_sentiment,           # Scores sentiment for a single comment using TextBlob + custom logic
    detect_aspects,            # Detects which aspects (professors, cafeteria, etc.) a comment mentions
    score_aspect_sentiment,    # Scores sentiment for a specific aspect using sentence extraction
    RAW_COMMENTS,              # The original 60 hardcoded realistic student comments (batch1)
    SAMPLE_BATCHES,            # Registry of every loadable sample batch: {id: {name, comments}}
    DEFAULT_BATCH,             # Which batch id loads first when the dashboard opens
    analyse_all_comments,      # Full analysis pipeline (kept for reference but not used directly here)
    nltk_tokenize,             # NLTK word tokenizer that splits text into individual words
    remove_stopwords,          # NLTK stopword filter that removes common words like 'the', 'is', 'at'
)


# INITIALIZE FLASK APP

# Create the Flask web application instance - this is the main app object

# INITIALIZE FLASK APP 
# Create the Flask web application instance - this is the main app object
app = Flask(__name__)


# HTML TEMPLATE - The Entire Frontend

# This is the entire frontend - HTML, CSS, and JavaScript all in one string.
# 
# Why embedded in Python?
# - No separate templates folder needed
# - Single file deployment
# - Everything works offline
# 
# What's inside:
# - Tabler CSS for styling (loaded from CDN)
# - Chart.js for charts (loaded from CDN)
# - Custom gold branding (CSS variables)
# - All dashboard components (metric cards, charts, comment explorer)
# - JavaScript for analysis, filtering, charts, and interactivity


# HTML TEMPLATE 
# This is the entire frontend - HTML, CSS, and JavaScript all in one string

HTML = r"""<!DOCTYPE html>
<html lang="en">
<!-- lang="en" specifies English language for accessibility and SEO -->
<head>
<!-- Character encoding for proper text display - UTF-8 supports all Unicode characters -->
<meta charset="UTF-8"/>
<!-- Viewport for responsive design on mobile devices - scales page to device width -->
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<!-- Browser tab title - shows "UniSentiment Dashboard" in the browser tab -->
<title>UniSentiment Dashboard</title>

<!--TABLER CDN -->
<!-- Tabler is a free, open-source admin dashboard UI kit built on Bootstrap -->
<!-- CDN = Content Delivery Network - loads the CSS from a fast global server -->
<!-- This loads all the CSS styles for the dashboard - layout, colors, typography, etc. -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/core@latest/dist/css/tabler.min.css">
<!-- Tabler Icons - A comprehensive icon set with 1000+ icons (ti-* classes) -->
<!-- The icons are font-based so they scale perfectly and can be colored with CSS -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/dist/tabler-icons.min.css">

<!-- CHART.JS -->
<!-- Chart.js is a lightweight JavaScript library for creating interactive charts -->
<!-- It uses HTML5 Canvas for rendering, making it fast and responsive -->
<!-- We use it for the pie chart (sentiment split) and bar chart (aspect performance) -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

<style>
  /* CUSTOM STYLES  */
  /* These CSS rules override Tabler's default styles to add our gold branding */
  
  /* :root is a CSS pseudo-class that matches the document's root element */
  /* CSS variables defined here are available throughout the entire document */
  /* --tblr-primary is a CSS variable used throughout Tabler for primary color */
  :root {
    --tblr-primary: #206bc4;    /* Tabler's default blue - used for buttons, links, etc. */
    --tblr-primary-rgb: 32, 107, 196;  /* RGB version of the blue for rgba() usage */
  }
  
  /* Gold gradient for the logo badge - creates a smooth transition from light to dark gold */
  /* !important overrides other conflicting styles - ensures our branding appears */
  .bg-gold {
    background: linear-gradient(135deg, #f0d080, #c8a84b) !important;
    /* linear-gradient creates a smooth color transition at 135 degrees angle */
    /* #f0d080 is light gold, #c8a84b is medium gold */
  }
  
  /* Gold text color for branding elements like headers and labels */
  .text-gold {
    color: #c8a84b !important;
  }
  
  /* Gold border color for elements that need gold outlines */
  .border-gold {
    border-color: #c8a84b !important;
  }
  
  /* Custom gold button with gradient background and hover effect */
  .btn-gold {
    background: linear-gradient(135deg, #f0d080, #c8a84b) !important;
    border-color: #c8a84b !important;  /* Border matches the background */
    color: #1a1a00 !important;          /* Dark text for contrast on gold background */
  }
  /* Hover state - when user hovers mouse over the button */
  .btn-gold:hover {
    background: linear-gradient(135deg, #e8c96a, #b89830) !important;
    /* Darker gold on hover for visual feedback */
    border-color: #b89830 !important;
    color: #1a1a00 !important;
  }
  
  /* ── CARD HOVER EFFECT ────────────────────────────────────────────────── */
  /* Cards are content containers - this adds a subtle lift on hover */
  /* transition makes the animation smooth over 0.2 seconds */
  .card {
    transition: transform 0.2s, box-shadow 0.2s;
    /* transform changes position/size, box-shadow changes shadow depth */
  }
  /* When user hovers over a card */
  .card:hover {
    transform: translateY(-2px);     /* Move up 2 pixels for a lifting effect */
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);  /* Add a soft shadow below */
  }
  
  /* ── LOADING OVERLAY ──────────────────────────────────────────────────── */
  /* Full-screen overlay shown during analysis to indicate processing */
  /* position:fixed keeps it in place even when scrolling */
  #loading-overlay {
    position: fixed;           /* Fixed positioning covers entire viewport regardless of scroll */
    inset: 0;                  /* Shorthand for top:0; right:0; bottom:0; left:0; */
    background: rgba(255, 255, 255, 0.8);  /* Semi-transparent white - 80% opacity */
    backdrop-filter: blur(4px);            /* Blur the content underneath by 4 pixels */
    z-index: 9999;             /* Very high z-index to appear on top of everything else */
    display: none;             /* Hidden by default, shown via JavaScript when processing */
    align-items: center;       /* Center child elements vertically using flexbox */
    justify-content: center;   /* Center child elements horizontally using flexbox */
    flex-direction: column;    /* Stack children vertically (spinner above message) */
    gap: 16px;                 /* 16px gap between spinner and message */
  }
  /* Bootstrap spinner with custom blue color */
  /* .spinner-border is a Bootstrap class that creates a spinning ring animation */
  #loading-overlay .spinner-border {
    width: 3rem;               /* 3rem = 48px (relative to root font size) */
    height: 3rem;              /* match width for a perfect circle */
    color: #206bc4;            /* Tabler blue color */
  }
  /* Loading message styling - shows what NLP step is currently running */
  .load-msg {
    color: #206bc4;            /* match the spinner's blue colour */
    font-size: 14px;           /* readable but secondary to the spinner visually */
    font-weight: 500;          /* Medium weight - between normal and bold */
  }
  
  /* ── QUICK RESULT BOX ──────────────────────────────────────────────────── */
  /* Shows the result of a single comment analysis instantly without full batch run */
  #quickRes {
    display: none;             /* Hidden by default, shown when quick analysis runs */
    padding: 10px 14px;        /* Top/bottom 10px, left/right 14px */
    border-radius: 8px;        /* Rounded corners */
    background: #f8fafc;       /* Very light gray-blue background */
    border: 1px solid #e2e8f0; /* Light border for definition */
    font-size: 13px;
    margin-top: 6px;           /* Space above the result box */
  }
  
  /* ── COMMENT ITEMS ────────────────────────────────────────────────────── */
  /* Each comment in the explorer list - displays as a card-like item */
  .comment-item {
    padding: 10px 14px;
    border-left: 3px solid #e2e8f0;  /* Left border color changes based on sentiment */
    cursor: pointer;                  /* Hand cursor indicates clickable to expand details */
    transition: background 0.15s;     /* Smooth background color change on hover */
    background: #ffffff;              /* White background for each comment */
    border-radius: 4px;              /* Small rounded corners */
    margin-bottom: 4px;              /* Small gap between comments */
  }
  /* Hover effect for comment items */
  .comment-item:hover {
    background: #f1f5f9;              /* Light gray-blue background on hover */
  }
  /* Sentiment-specific left border colors - visual indicator of sentiment */
  .comment-item.pos { border-left-color: #2fb344; }   /* Green for positive comments */
  .comment-item.neg { border-left-color: #d63939; }   /* Red for negative comments */
  .comment-item.neu { border-left-color: #4299e1; }   /* Blue for neutral comments */
  
  /* Sentiment badge colors - small labels next to each comment */
  .badge-pos { background: #d4edda; color: #155724; }  /* Light green with dark green text */
  .badge-neg { background: #f8d7da; color: #721c24; }  /* Light red with dark red text */
  .badge-neu { background: #cce5ff; color: #004085; }  /* Light blue with dark blue text */
  
  /* Flag colors for sarcasm and negation detection - shows special flags */
  .cflag { color: #d69e2e; }    /* Amber/gold for sarcasm detection */
  .cflag2 { color: #805ad5; }   /* Purple for negation detection */
  
  /* ── COMMENT DETAIL EXPAND ────────────────────────────────────────────── */
  /* Hidden by default, shown when user clicks a comment to expand it */
  .comment-detail {
    display: none;               /* Hidden initially */
    margin-top: 8px;            /* Space from the comment summary */
    padding-top: 8px;           /* Space inside before the border */
    border-top: 1px solid #e2e8f0;  /* Separator line */
    font-size: 12px;            /* Slightly smaller text for details */
    color: #64748b;             /* Slate gray - secondary text color */
    line-height: 1.7;           /* 1.7x line height for better readability */
  }
  
  /* ── ASPECT PROGRESS BARS ────────────────────────────────────────────── */
  /* Visual representation of positive percentage per aspect */
  /* The track is the empty background, the fill shows the positive percentage */
  .aspect-bar-track {
    height: 6px;                /* Thin bar */
    background: #e2e8f0;        /* Light gray background */
    border-radius: 3px;         /* Rounded ends */
    overflow: hidden;           /* Ensures the fill stays inside the rounded corners */
  }
  /* The fill portion of the progress bar - width changes based on data */
  .aspect-bar-fill {
    height: 100%;               /* Full height of the track */
    border-radius: 3px;         /* Rounded ends to match the track */
    transition: width 0.6s ease;   /* Animate the width change smoothly over 0.6 seconds */
  }
  
  /* ── KEYWORD TAGS ────────────────────────────────────────────────────── */
  /* Small tags showing top keywords from NLTK frequency analysis */
  .kw-tag {
    display: inline-block;      /* Allows padding and margin while staying in line */
    padding: 2px 10px;          /* Small padding inside */
    border-radius: 12px;        /* Pill shape */
    font-size: 11px;            /* Small text */
    background: #fef3c7;        /* Light amber background */
    border: 1px solid #f6e05e;  /* Amber border */
    color: #975a16;             /* Dark amber text */
    margin: 2px;               /* Small gap between tags */
  }
  
  /* ── FILTER TABS ────────────────────────────────────────────────────── */
  /* Clickable tabs for filtering comments by sentiment - like button pills */
  .filter-tab {
    cursor: pointer;           /* Hand cursor indicates clickable */
    padding: 4px 12px;         /* Small padding */
    border-radius: 4px;        /* Small rounded corners */
    font-size: 12px;
    transition: all 0.15s;     /* Smooth transition for all properties */
    border: 1px solid #e2e8f0; /* Light border */
    background: #ffffff;       /* White background */
  }
  /* Hover effect for filter tabs */
  .filter-tab:hover {
    background: #f1f5f9;       /* Light gray on hover */
  }
  /* Active state - currently selected filter */
  .filter-tab.active {
    background: #206bc4;       /* Tabler blue background */
    color: #ffffff;            /* White text */
    border-color: #206bc4;     /* Blue border */
  }
  
  /* ── INSIGHT ROWS ────────────────────────────────────────────────────── */
  /* Each insight (Fix First, Keep Doing, Controversial/Watch) in a row */
  .insight-row {
    display: flex;             /* Flexbox layout for alignment */
    align-items: center;       /* Vertically center items */
    gap: 10px;                /* Space between items */
    padding: 6px 0;           /* Vertical padding only */
    border-bottom: 1px solid #e2e8f0;  /* Separator line between rows */
    font-size: 13px;
  }
  /* Remove border from the last row */
  .insight-row:last-child { border-bottom: none; }
  
  /* ── ALERT ITEMS ────────────────────────────────────────────────────── */
  /* High and medium severity alerts that need attention */
  .alert-item {
    display: flex;
    gap: 10px;
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 4px;
    font-size: 12px;
    border: 1px solid #e2e8f0;  /* Default light border */
  }
  /* High severity alert - red theme */
  .alert-high {
    background: #fef2f2;        /* Very light red background */
    border-color: #fca5a5;      /* Light red border */
  }
  /* Medium severity alert - yellow/amber theme */
  .alert-med {
    background: #fffbeb;        /* Very light amber background */
    border-color: #fcd34d;      /* Light amber border */
  }
  
  /* ── COMMENT LIST SCROLLBAR ──────────────────────────────────────────── */
  /* Custom scrollbar styling for the comment list container */
  .comment-list {
    max-height: 300px;          /* Maximum height before scrolling */
    overflow-y: auto;           /* Vertical scroll when content exceeds max-height */
  }
  /* Width of the scrollbar */
  .comment-list::-webkit-scrollbar {
    width: 4px;                /* Thin scrollbar */
  }
  /* The draggable thumb of the scrollbar */
  .comment-list::-webkit-scrollbar-thumb {
    background: #206bc4;       /* Blue thumb */
    border-radius: 2px;        /* Rounded ends */
  }
  
  /* ── TOAST NOTIFICATIONS ────────────────────────────────────────────── */
  /* Popup notifications that appear at bottom-right */
  .toast-container {
    position: fixed;           /* Fixed position - stays in place when scrolling */
    bottom: 20px;             /* 20px from bottom of viewport */
    right: 20px;              /* 20px from right of viewport */
    z-index: 9999;            /* High z-index to appear on top of everything */
  }
  
  /* ── NAVBAR LOGO ────────────────────────────────────────────────────── */
  /* Custom styling for the UniSentiment logo in the navbar */
  .navbar-brand-logo {
    width: 32px;              /* Square 32x32 */
    height: 32px;
    display: inline-flex;     /* Flex container for centering the "U" */
    align-items: center;      /* Center vertically */
    justify-content: center;  /* Center horizontally */
    font-weight: 900;         /* Very bold - almost black */
    color: #1a1a00;           /* Very dark text for contrast on gold */
    border-radius: 6px;       /* Small rounded corners */
  }
  
  /* ── PAGE HEADER ICON ────────────────────────────────────────────────── */
  /* Makes the chart icon gold in the page header */
  .page-title .ti {
    color: #c8a84b;           /* Gold color */
  }

  /* ── BATCH SELECT ─────────────────────────────────────────────────────── */
  /* Dropdown next to "Load Sample" that picks which comment set to load */
  #batchSelect {
    width: auto;               /* Shrink to fit its content instead of full width */
    max-width: 220px;          /* Cap the width so it doesn't crowd the navbar */
  }
</style>
</head>

<body>
<!-- ── PAGE CONTAINER ───────────────────────────────────────────────────── -->
<!-- Main container div that wraps the entire page content -->
<div class="page">

  <!-- ── LOADING OVERLAY ──────────────────────────────────────────────────── -->
  <!-- Shown when analysis is running to provide visual feedback -->
  <div id="loading-overlay">
    <!-- Bootstrap spinner animation - a rotating ring -->
    <div class="spinner-border" role="status">
      <!-- visually-hidden class hides this text from sighted users but keeps it for screen readers -->
      <span class="visually-hidden">Loading...</span>
    </div>
    <!-- Dynamic loading message that cycles through NLP steps via JavaScript -->
    <div class="load-msg" id="loadMsg">Tokenizing with NLTK …</div>
  </div>

  <!-- ── NAVBAR ───────────────────────────────────────────────────────────── -->
  <!-- Top navigation bar with logo, links, search, and actions -->
  <!-- navbar-expand-md: expands to full navbar on medium screens and above -->
  <!-- navbar-light: light theme for the navbar -->
  <header class="navbar navbar-expand-md navbar-light">
    <div class="container-fluid">
      <!-- Mobile hamburger menu toggler - appears on small screens -->
      <!-- data-bs-toggle and data-bs-target are Bootstrap attributes for collapse functionality -->
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbar-menu">
        <span class="navbar-toggler-icon"></span>
      </button>
      <!-- Brand/logo with gold badge -->
      <a class="navbar-brand" href="#">
        <!-- Span with gold background containing "U" -->
        <span class="navbar-brand-logo bg-gold me-2">U</span>
        UniSentiment
      </a>
      
      <!-- Collapsible navbar menu - hidden on mobile, shown when hamburger clicked -->
      <div class="collapse navbar-collapse" id="navbar-menu">
        <!-- Navigation links - left side of navbar -->
        <ul class="navbar-nav me-auto">
          <!-- me-auto pushes the following items to the right -->
          <li class="nav-item">
            <!-- navTo('dashboard') scrolls to the dashboard section when clicked -->
            <a class="nav-link active" href="#" onclick="navTo('dashboard')">
              <i class="ti ti-dashboard me-2"></i>Dashboard
            </a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="#" onclick="navTo('comments')">
              <i class="ti ti-messages me-2"></i>Comments
            </a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="#" onclick="navTo('aspects')">
              <i class="ti ti-list-details me-2"></i>Aspects
            </a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="#" onclick="navTo('insights')">
              <i class="ti ti-lightbulb me-2"></i>Insights
            </a>
          </li>
        </ul>
        
        <!-- Right-side actions: search, batch picker, load sample, notifications -->
        <div class="d-flex align-items-center gap-2">
          <!-- Search input with magnifying glass icon -->
          <div class="input-icon">
            <!-- input-icon-addon positions the icon inside the input -->
            <span class="input-icon-addon">
              <i class="ti ti-search"></i>
            </span>
            <!-- oninput fires searchComments() every time user types -->
            <input type="text" class="form-control form-control-sm" id="searchIn" placeholder="Search comments…" style="width:160px;" oninput="searchComments()"/>
          </div>
          <!-- Batch picker - lets the user choose which sample comment set to load -->
          <!-- Populated dynamically from /batches so new batches show up automatically -->
          <select class="form-select form-select-sm" id="batchSelect" title="Choose which sample set to load"></select>
          <!-- Load sample comments button - loads whichever batch is selected above -->
          <button class="btn btn-gold btn-sm" onclick="loadSample()" title="Load the selected sample batch">
            <i class="ti ti-plus me-1"></i>Load Sample
          </button>
          <!-- Notification bell - shows toast message on click -->
          <button class="btn btn-ghost-secondary btn-icon btn-sm" onclick="toast('No new notifications')">
            <i class="ti ti-bell"></i>
          </button>
        </div>
      </div>
    </div>
  </header>

  <!-- ── PAGE WRAPPER ─────────────────────────────────────────────────────── -->
  <!-- Wraps the main content area of the page -->
  <div class="page-wrapper">

    <!-- ── PAGE HEADER ────────────────────────────────────────────────────── -->
    <!-- Title and action buttons at top of content area -->
    <div class="page-header d-print-none">
      <div class="container-fluid">
        <div class="row g-2 align-items-center">
          <div class="col">
            <h2 class="page-title">
              <i class="ti ti-chart-bar me-2"></i>Sentiment Dashboard
            </h2>
            <!-- Subtitle showing status or analysis time -->
            <div class="text-secondary" id="m-sub">Ready to analyse student feedback</div>
          </div>
          <div class="col-auto ms-auto d-print-none">
            <!-- Run Analysis button - triggers the full NLP pipeline -->
            <button class="btn btn-primary" id="runBtn" onclick="runAnalysis()">
              <i class="ti ti-player-play me-1"></i>Run Analysis
            </button>
            <!-- Export CSV button - appears after analysis completes -->
            <button class="btn btn-ghost-secondary" id="expBtn" onclick="exportCSV()" style="display:none;">
              <i class="ti ti-file-export me-1"></i>Export CSV
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- ── PAGE BODY ──────────────────────────────────────────────────────── -->
    <!-- Main content area containing all dashboard components -->
    <div class="page-body">
      <div class="container-fluid">

        <!-- ── INPUT ROW ──────────────────────────────────────────────────── -->
        <!-- Card containing Quick Analyse and Batch Comments inputs -->
        <div class="card mb-3" id="section-dashboard">
          <div class="card-body">
            <div class="row g-3">
              <!-- Left column: Quick single comment analysis -->
              <div class="col-md-5">
                <label class="form-label">Quick Analyse</label>
                <div class="input-group">
                  <!-- Input field for single comment -->
                  <input class="form-control" id="quickIn" placeholder='e.g. "Wi-Fi is terrible but professors are great"' onkeydown="if(event.key==='Enter')quickAnalyse()"/>
                  <button class="btn btn-secondary" onclick="quickAnalyse()">Analyse</button>
                </div>
                <!-- Result container for quick analysis output -->
                <div id="quickRes"></div>
              </div>
              <!-- Right column: Batch comments textarea -->
              <div class="col-md-7">
                <label class="form-label">Batch Comments <span class="text-secondary" id="lct">(0 comments)</span></label>
                <div class="input-group">
                  <textarea class="form-control" id="ta" rows="2" placeholder="Paste multiple comments — one per line" oninput="countLines()"></textarea>
                </div>
                <div class="mt-2">
                  <button class="btn btn-ghost-secondary btn-sm" onclick="clearAll()">✕ Clear</button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- ── METRIC CARDS ROW ───────────────────────────────────────────── -->
        <!-- 4 cards showing: Total, Positive, Negative, Neutral sentiment counts -->
        <div class="row g-3 mb-3">
          <!-- Total Comments Card -->
          <div class="col-sm-6 col-lg-3">
            <div class="card">
              <div class="card-body">
                <div class="d-flex align-items-center">
                  <div class="subheader">Total Comments</div>
                  <div class="ms-auto lh-1">
                    <span class="badge bg-gold text-dark">LIVE</span>
                  </div>
                </div>
                <!-- h1 is a large heading - displays the total count -->
                <div class="h1 mb-3" id="m-total" style="color:#c8a84b;">—</div>
                <div class="d-flex mb-2">
                  <span class="text-secondary">Analysed</span>
                </div>
              </div>
            </div>
          </div>
          <!-- Positive Card - green theme -->
          <div class="col-sm-6 col-lg-3">
            <div class="card">
              <div class="card-body">
                <div class="subheader">Positive</div>
                <div class="h1 mb-3 text-green" id="m-pos">—</div>
                <div class="d-flex mb-2">
                  <span class="text-secondary" id="m-posn">comments</span>
                  <span class="ms-auto text-green" id="m-pos-badge" style="display:none;">↑</span>
                </div>
              </div>
            </div>
          </div>
          <!-- Negative Card - red theme -->
          <div class="col-sm-6 col-lg-3">
            <div class="card">
              <div class="card-body">
                <div class="subheader">Negative</div>
                <div class="h1 mb-3 text-red" id="m-neg">—</div>
                <div class="d-flex mb-2">
                  <span class="text-secondary" id="m-negn">comments</span>
                  <span class="ms-auto text-red" id="m-neg-badge" style="display:none;">↓</span>
                </div>
              </div>
            </div>
          </div>
          <!-- Neutral Card - blue theme -->
          <div class="col-sm-6 col-lg-3">
            <div class="card">
              <div class="card-body">
                <div class="subheader">Neutral</div>
                <div class="h1 mb-3 text-azure" id="m-neu">—</div>
                <div class="d-flex mb-2">
                  <span class="text-secondary" id="m-neun">comments</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- ── CHARTS ROW ──────────────────────────────────────────────────── -->
        <div class="row g-3 mb-3" id="section-aspects">
          <!-- Left: Pie Chart - Sentiment Split -->
          <div class="col-lg-5">
            <div class="card">
              <div class="card-header">
                <h3 class="card-title">Sentiment Split</h3>
              </div>
              <div class="card-body">
                <!-- Fixed height container for the chart -->
                <div style="height:200px;">
                  <!-- Canvas element where Chart.js renders the pie chart -->
                  <canvas id="pieC"></canvas>
                </div>
              </div>
            </div>
          </div>
          <!-- Right: Bar Chart - Aspect Performance -->
          <div class="col-lg-7">
            <div class="card">
              <div class="card-header">
                <h3 class="card-title">Aspect Performance</h3>
              </div>
              <div class="card-body">
                <div style="height:200px;">
                  <!-- Canvas element where Chart.js renders the bar chart -->
                  <canvas id="barC"></canvas>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- ── BOTTOM ROW ──────────────────────────────────────────────────── -->
        <div class="row g-3" id="section-comments">
          
          <!-- LEFT: Comment Explorer -->
          <div class="col-lg-7">
            <div class="card">
              <div class="card-header">
                <h3 class="card-title">Comment Explorer</h3>
                <div class="card-actions">
                  <!-- Export button as icon - hidden until analysis runs -->
                  <a href="#" onclick="exportCSV()" class="btn btn-sm btn-ghost-secondary">
                    <i class="ti ti-file-export"></i>
                  </a>
                </div>
              </div>
              <div class="card-body">
                <!-- Filter tabs for sentiment filtering - click to filter comments -->
                <div class="mb-2 d-flex gap-1 flex-wrap">
                  <span class="filter-tab active" onclick="filt('all',this)">All</span>
                  <span class="filter-tab" onclick="filt('positive',this)">✅ Pos</span>
                  <span class="filter-tab" onclick="filt('negative',this)">❌ Neg</span>
                  <span class="filter-tab" onclick="filt('neutral',this)">➖ Neu</span>
                  <span class="filter-tab" onclick="filt('sarcasm',this)">⚠ Sarc</span>
                  <span class="filter-tab" onclick="filt('negation',this)">↩ Neg</span>
                </div>
                <!-- Container for the list of comments - populated by JavaScript -->
                <div class="comment-list" id="clist">
                  <!-- Empty state shown before any analysis is run -->
                  <div class="empty-state text-center py-4 text-secondary">
                    <i class="ti ti-messages icon-lg mb-2 d-block opacity-25"></i>
                    <span>Comments will appear here after analysis</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- RIGHT: Insights + Alerts -->
          <div class="col-lg-5">
            
            <!-- Insights Card - shows Fix First, Keep Doing, Controversial -->
            <div class="card mb-3" id="section-insights">
              <div class="card-header">
                <h3 class="card-title">💡 Insights</h3>
              </div>
              <div class="card-body" id="insList">
                <!-- Empty state before analysis -->
                <div class="empty-state text-center py-3 text-secondary">
                  <i class="ti ti-lightbulb icon-lg mb-1 d-block opacity-25"></i>
                  <span>Run analysis first</span>
                </div>
              </div>
              <div class="card-footer">
                <div class="fw-medium text-secondary" style="font-size:11px;">Keywords (NLTK)</div>
                <!-- Container for keyword tags from NLTK frequency analysis -->
                <div id="kwWrap" class="mt-1"></div>
              </div>
            </div>

            <!-- Alerts Card - shows HIGH and MEDIUM severity alerts -->
            <div class="card" id="section-alerts">
              <div class="card-header">
                <h3 class="card-title">🔔 Alerts</h3>
              </div>
              <div class="card-body" id="alertsList">
                <!-- Empty state before analysis -->
                <div class="empty-state text-center py-3 text-secondary">
                  <i class="ti ti-bell icon-lg mb-1 d-block opacity-25"></i>
                  <span>No alerts yet</span>
                </div>
              </div>
            </div>

            <!-- Aspect Scores Card - shows progress bars for each aspect -->
            <div class="card mt-3">
              <div class="card-header">
                <h3 class="card-title">📊 Aspect Scores</h3>
              </div>
              <div class="card-body" id="aspList">
                <!-- Empty state before analysis -->
                <div class="empty-state text-center py-3 text-secondary">
                  <i class="ti ti-list-details icon-lg mb-1 d-block opacity-25"></i>
                  <span>Run analysis to see aspects</span>
                </div>
              </div>
            </div>

          </div>
        </div>

      </div>
    </div>

    <!-- ── FOOTER ──────────────────────────────────────────────────────────── -->
    <!-- Page footer with version information -->
    <footer class="footer footer-transparent d-print-none">
      <div class="container-fluid">
        <div class="row text-center align-items-center flex-row-reverse">
          <div class="col-lg-auto ms-lg-auto">
            <span class="text-secondary">UniSentiment v2.0 &mdash; Sentiment Intelligence</span>
          </div>
        </div>
      </div>
    </footer>

  </div>
</div>

<!-- ── TOAST CONTAINER ────────────────────────────────────────────────────── -->
<!-- Bootstrap toast component for notification messages -->
<div class="toast-container" id="toastContainer">
  <div class="toast" id="toast" role="alert">
    <div class="toast-body" id="toastBody"></div>
  </div>
</div>

<!-- ── TABLER JAVASCRIPT ────────────────────────────────────────────────── -->
<!-- Tabler's JavaScript for interactive components like dropdowns, toggles, collapse -->
<script src="https://cdn.jsdelivr.net/npm/@tabler/core@latest/dist/js/tabler.min.js"></script>

<script>
// ── GLOBAL VARIABLES ──────────────────────────────────────────────────────
// D: Stores the full analysis results data from the backend
let D = null;
// curFilt: Current sentiment filter ('all', 'positive', 'negative', etc.)
let curFilt = 'all';
// pieI: Chart.js instance for pie chart - for destroying/recreating
let pieI = null;
// barI: Chart.js instance for bar chart - for destroying/recreating
let barI = null;
// currentBatch: id of whichever sample batch is currently selected in the dropdown
let currentBatch = null;

// ── LOADING MESSAGES ──────────────────────────────────────────────────────
// These messages cycle during analysis to show NLP progress to the user
const MSGS = [
  'Tokenizing with NLTK …',        // Step 1: Word tokenization
  'Removing stopwords …',          // Step 2: Filter common words
  'Detecting aspects …',           // Step 3: Find aspect mentions
  'TextBlob scoring …',            // Step 4: Base sentiment scoring
  'Negation check …',              // Step 5: Detect negation words
  'Sarcasm detection …',           // Step 6: Detect sarcastic patterns
  'Building insights …'            // Step 7: Generate insights/alerts
];
// mi: Current message index (which message is shown)
let mi = 0;
// mt: Interval timer for cycling messages
let mt = null;

// ── TOAST NOTIFICATION ────────────────────────────────────────────────────
// Shows a popup notification at bottom-right of screen
// msg: The message text to display
// type: 'info', 'success', or 'error' - determines color scheme
function toast(msg, type = 'info') {
  // Get the toast element and its body
  const toast = document.getElementById('toast');
  const body = document.getElementById('toastBody');
  // Set the message text
  body.textContent = msg;
  // Show the toast (Bootstrap's 'show' class makes it visible)
  toast.className = 'toast show';
  // Apply color classes based on type
  if (type === 'success') toast.classList.add('bg-success', 'text-white');
  else if (type === 'error') toast.classList.add('bg-danger', 'text-white');
  else toast.classList.add('bg-dark', 'text-white');
  // Auto-hide after 3 seconds (3000 milliseconds)
  setTimeout(() => { toast.className = 'toast'; }, 3000);
}

// ── LOADING OVERLAY ──────────────────────────────────────────────────────
// Shows/hides the loading overlay with cycling NLP messages
// on: true to show, false to hide
function loading(on) {
  // Get the loading overlay element
  const el = document.getElementById('loading-overlay');
  // Show or hide based on 'on' parameter
  el.style.display = on ? 'flex' : 'none';
  // Disable the Run Analysis button while loading
  document.getElementById('runBtn').disabled = on;
  if (on) {
    // Start cycling through messages
    mi = 0;  // Start from the first message
    document.getElementById('loadMsg').textContent = MSGS[0];
    // Clear any existing timer
    if (mt) clearInterval(mt);
    // Set up interval to change message every 700ms
    mt = setInterval(() => {
      mi = (mi + 1) % MSGS.length;  // Move to next message, loop back to start
      document.getElementById('loadMsg').textContent = MSGS[mi];
    }, 700);
  } else {
    // Stop the message cycling
    if (mt) clearInterval(mt);
  }
}

// ── COUNT LINES ──────────────────────────────────────────────────────────
// Updates the comment count display as the user types in the textarea
function countLines() {
  // Split textarea value by newlines, filter out empty lines
  const n = document.getElementById('ta').value.split('\n').filter(l => l.trim()).length;
  // Update the count label with pluralization
  document.getElementById('lct').textContent = `(${n} comment${n !== 1 ? 's' : ''})`;
}

// ── CLEAR ALL ─────────────────────────────────────────────────────────────
// Clears the textarea, quick result, and quick input field
function clearAll() {
  document.getElementById('ta').value = '';
  document.getElementById('quickRes').style.display = 'none';
  document.getElementById('quickIn').value = '';
  countLines();  // Update the count to 0
}

// ── LOAD BATCH LIST ───────────────────────────────────────────────────────
// Fetches the list of available sample batches from the backend and
// populates the <select id="batchSelect"> dropdown in the navbar.
// Called once on page load, before the first sample is loaded.
async function loadBatchList() {
  const r = await fetch('/batches');
  const d = await r.json();  // d: { batches: [{id, name, count}], default: 'batch1' }
  const sel = document.getElementById('batchSelect');
  // Build one <option> per batch, showing its name and comment count
  sel.innerHTML = d.batches.map(b => `<option value="${b.id}">${b.name}</option>`).join('');
  sel.value = d.default;        // pre-select the default batch
  currentBatch = d.default;     // remember it as the currently active batch
}

// ── LOAD SAMPLE ──────────────────────────────────────────────────────────
// Fetches the sample comments for whichever batch is selected in the
// dropdown and populates the textarea with them.
async function loadSample() {
  // Read the currently selected batch id from the dropdown (falls back to
  // currentBatch if the dropdown hasn't been populated yet for some reason)
  const sel = document.getElementById('batchSelect');
  const batchId = (sel && sel.value) ? sel.value : (currentBatch || 'batch1');
  currentBatch = batchId;

  const batchName = sel && sel.selectedOptions.length ? sel.selectedOptions[0].textContent : batchId;
  toast(`Loading "${batchName}"…`, 'info');

  // Fetch the sample comments from /load_sample, passing the chosen batch
  const r = await fetch(`/load_sample?batch=${encodeURIComponent(batchId)}`);
  const d = await r.json();
  if (d.error) { toast(d.error, 'error'); return; }

  // Join the comments array with newlines and set as textarea value
  document.getElementById('ta').value = d.comments.join('\n');
  countLines();  // Update the count
  toast(`${d.comments.length} comments loaded from "${d.name}"!`, 'success');
}

// ── QUICK ANALYSE ────────────────────────────────────────────────────────
// Analyses a single comment instantly without running the full batch
// This is called when user clicks "Analyse" or presses Enter
async function quickAnalyse() {
  // Get the comment text and trim whitespace
  const c = document.getElementById('quickIn').value.trim();
  if (!c) { toast('Enter a comment first', 'error'); return; }
  
  // Send the single comment to the /single endpoint
  const r = await fetch('/single', {
    method: 'POST',  // POST method for sending data
    headers: { 'Content-Type': 'application/json' },  // Tell server we're sending JSON
    body: JSON.stringify({ comment: c })  // Convert the comment to JSON string
  });
  // Parse the JSON response
  const d = await r.json();
  const ov = d.overall;  // ov: the overall sentiment result object for this comment
  
  // Determine color based on sentiment label: green=positive, red=negative, blue=neutral
  const col = ov.label === 'positive' ? '#2fb344' : ov.label === 'negative' ? '#d63939' : '#4299e1';
  
  // Build flags for sarcasm and negation detection
  const flags = [];
  if (ov.sarcasm_detected) flags.push('<span class="cflag">⚠ Sarcasm</span>');
  if (ov.negation_detected) flags.push('<span class="cflag2">↩ Negation</span>');
  
  // Build aspect score tags - shows each aspect with its score
  const asps = Object.entries(d.aspect_scores || {}).map(([a, s]) => {
    // a: aspect name string; s: that aspect's sentiment score object
    const c2 = s.label === 'positive' ? '#2fb344' : s.label === 'negative' ? '#d63939' : '#4299e1';  // c2: colour for this aspect's label
    return `<span class="kw-tag" style="color:${c2};border-color:${c2}40;">${a}: ${s.compound > 0 ? '+' : ''}${s.compound}</span>`;  // build one tag span per aspect
  }).join('');
  
  // Display the result in the quickRes container
  const el = document.getElementById('quickRes');
  el.style.display = 'block';
  el.innerHTML = `
    <div class="d-flex align-items-center gap-3 flex-wrap">
      <strong style="color:${col};text-transform:uppercase;font-size:13px;">${ov.label}</strong>
      <span class="text-secondary">Score: <strong>${ov.compound > 0 ? '+' : ''}${ov.compound}</strong></span>
      <span class="text-secondary">Subjectivity: <strong>${ov.subjectivity}</strong></span>
      ${flags.join(' ')}
    </div>
    <div class="mt-1">${asps}</div>
  `;
}

// ── RUN ANALYSIS ─────────────────────────────────────────────────────────
// Main analysis function - processes all comments in the textarea
// This is the primary function called by the "Run Analysis" button
async function runAnalysis() {
  // Get all non-empty lines from the textarea
  const lines = document.getElementById('ta').value.split('\n').filter(l => l.trim());
  if (!lines.length) { toast('Please add comments first', 'error'); return; }
  
  loading(true);  // Show loading overlay
  
  try {
    // Send the comments to /analyse endpoint
    const r = await fetch('/analyse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ comments: lines })
    });
    const d = await r.json();
    if (d.error) { toast(d.error, 'error'); loading(false); return; }
    
    D = d;  // Store data globally for filtering and searching
    loading(false);  // Hide loading overlay
    render(d);  // Render all dashboard components with the data
    document.getElementById('expBtn').style.display = 'inline-flex';  // Show export button
    toast(`${d.total} comments analysed`, 'success');
  } catch (e) {
    loading(false);
    toast('Server error', 'error');
  }
}

// ── RENDER DASHBOARD ─────────────────────────────────────────────────────
// Renders all dashboard components with analysis results
// This is called after successful analysis
function render(d) {
  // ── METRIC CARDS ──────────────────────────────────────────────────────
  // Update the 4 metric cards with the analysis results
  document.getElementById('m-total').textContent = d.total;
  document.getElementById('m-pos').textContent = d.overall.positive_pct + '%';
  document.getElementById('m-neg').textContent = d.overall.negative_pct + '%';
  document.getElementById('m-neu').textContent = d.overall.neutral_pct + '%';
  document.getElementById('m-posn').textContent = d.overall.positive_count + ' comments';
  document.getElementById('m-negn').textContent = d.overall.negative_count + ' comments';
  document.getElementById('m-neun').textContent = d.overall.neutral_count + ' comments';
  // Update the subtitle with the analysis time
  document.getElementById('m-sub').textContent = 'Analysed · ' + d.analysed_at.slice(11, 16);
  // Show the badge counts
  document.getElementById('m-pos-badge').style.display = 'inline';
  document.getElementById('m-neg-badge').style.display = 'inline';
  document.getElementById('m-pos-badge').textContent = '+' + d.overall.positive_count;
  document.getElementById('m-neg-badge').textContent = d.overall.negative_count;

  // ── PIE CHART ──────────────────────────────────────────────────────────
  // Destroy previous chart if it exists (prevents memory leaks)
  if (pieI) pieI.destroy();
  // Create new pie chart using Chart.js
  pieI = new Chart(document.getElementById('pieC'), {
    type: 'doughnut',  // Doughnut chart (pie chart with a hole)
    data: {
      labels: ['Positive', 'Negative', 'Neutral'],
      datasets: [{
        data: [d.overall.positive_pct, d.overall.negative_pct, d.overall.neutral_pct],
        backgroundColor: ['#2fb344', '#d63939', '#4299e1'],  // Green, Red, Blue
        borderWidth: 2,
        borderColor: '#ffffff'  // White border between slices
      }]
    },
    options: {
      responsive: true,           // Resize with container
      maintainAspectRatio: false, // Don't maintain aspect ratio
      cutout: '62%',             // 62% of center is cut out
      plugins: {
        legend: {
          labels: { color: '#64748b', font: { size: 10 }, padding: 12 }
        }
      }
    }
  });

  // ── BAR CHART ──────────────────────────────────────────────────────────
  // Destroy previous chart if it exists
  if (barI) barI.destroy();
  const asp = d.aspect_summary;  // asp: the aspect_summary dict from the response (aspect name -> stats)
  const labels = Object.keys(asp).map(a => a.charAt(0).toUpperCase() + a.slice(1));  // labels: capitalized aspect names for the x-axis
  const pos = Object.values(asp).map(s => s.positive_pct);  // pos: array of positive percentages, one per aspect
  const neg = Object.values(asp).map(s => s.negative_pct);  // neg: array of negative percentages, one per aspect
  barI = new Chart(document.getElementById('barC'), {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        { label: 'Positive', data: pos, backgroundColor: 'rgba(47,179,68,0.7)', borderRadius: 3 },
        { label: 'Negative', data: neg, backgroundColor: 'rgba(214,57,57,0.7)', borderRadius: 3 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#64748b', font: { size: 10 } }
        }
      },
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 9 } }, grid: { color: 'rgba(0,0,0,0.05)' } },  // x-axis: aspect name ticks + faint gridlines
        y: { ticks: { color: '#64748b', font: { size: 9 }, callback: v => v + '%' }, grid: { color: 'rgba(0,0,0,0.05)' }, max: 100 }  // y-axis: 0-100%, ticks shown with "%" suffix
      }
    }
  });

  // ── ASPECT SCORES LIST ────────────────────────────────────────────────
  // Sort aspects by positive percentage (highest first)
  const al = document.getElementById('aspList');
  const sorted = Object.entries(asp).sort((a, b) => b[1].positive_pct - a[1].positive_pct);
  // Generate HTML for each aspect with progress bar
  al.innerHTML = sorted.map(([name, s]) => {
    // Determine color based on performance
    const col = s.positive_pct >= 60 ? '#2fb344' : s.negative_pct >= 50 ? '#d63939' : '#c8a84b';
    return `
      <div class="d-flex align-items-center gap-2 py-1 border-bottom">
        <div style="width:80px;font-size:12px;font-weight:500;">${name.charAt(0).toUpperCase() + name.slice(1)}</div>
        <div class="aspect-bar-track flex-grow-1">
          <div class="aspect-bar-fill" style="width:${s.positive_pct}%;background:${col};"></div>
        </div>
        <div style="font-size:12px;width:32px;text-align:right;color:${col};">${s.positive_pct}%</div>
      </div>
    `;
  }).join('');

  // ── INSIGHTS ────────────────────────────────────────────────────────────
  // Build insight rows from fix_first, keep_doing, and controversial
  const rows = [];
  (d.fix_first || []).forEach(a => rows.push(['fix', a, asp[a]?.negative_pct + '% neg']));      // 'fix' rows: aspects needing urgent attention
  (d.keep_doing || []).forEach(a => rows.push(['keep', a, asp[a]?.positive_pct + '% pos']));     // 'keep' rows: aspects performing well
  (d.controversial || []).forEach(a => rows.push(['watch', a, '50/50 split']));                  // 'watch' rows: aspects with a divided opinion
  document.getElementById('insList').innerHTML = rows.length ? rows.map(([k, a, info]) => `
    <div class="insight-row">
      <span class="badge ${k === 'fix' ? 'bg-danger' : k === 'keep' ? 'bg-success' : 'bg-warning'}">${k === 'fix' ? 'Fix' : k === 'keep' ? 'Keep' : 'Watch'}</span>
      <span>${a.charAt(0).toUpperCase() + a.slice(1)}</span>
      <span class="ms-auto text-secondary" style="font-size:11px;">${info}</span>
    </div>
  `).join('') : '<div class="text-secondary text-center py-2">No insights generated</div>';

  // ── KEYWORDS ────────────────────────────────────────────────────────────
  // Display top 14 keywords from NLTK frequency analysis
  document.getElementById('kwWrap').innerHTML = (d.top_keywords || []).slice(0, 14).map(([w, c]) =>
    `<span class="kw-tag" title="${c}x">${w}</span>`
  ).join('');

  // ── ALERTS ──────────────────────────────────────────────────────────────
  // Display alerts with severity levels
  const al2 = document.getElementById('alertsList');
  al2.innerHTML = (d.alerts && d.alerts.length) ? d.alerts.map(a => `
    <div class="alert-item ${a.severity === 'HIGH' ? 'alert-high' : 'alert-med'}">
      <span>${a.severity === 'HIGH' ? '🚨' : '⚠️'}</span>
      <div>
        <strong style="font-size:12px;">${a.aspect.charAt(0).toUpperCase() + a.aspect.slice(1)}</strong>
        <div class="text-secondary" style="font-size:11px;">${a.message}</div>
      </div>
    </div>
  `).join('') : '<div class="text-secondary text-center py-2">No active alerts</div>';

  // ── COMMENTS ────────────────────────────────────────────────────────────
  renderComments(d.records);
}

// ── RENDER COMMENTS ──────────────────────────────────────────────────────
// Renders the list of comments with expandable details
// recs: Array of comment objects with their sentiment data
function renderComments(recs) {
  const el = document.getElementById('clist');
  if (!recs || !recs.length) {
    el.innerHTML = '<div class="text-secondary text-center py-3"><i class="ti ti-messages icon-lg mb-1 d-block opacity-25"></i>No comments match</div>';
    return;
  }
  // Generate HTML for each comment
  el.innerHTML = recs.map(r => {
    const ov = r.overall;
    // Determine CSS classes based on sentiment
    const cls = ov.label === 'positive' ? 'pos' : ov.label === 'negative' ? 'neg' : 'neu';
    const bc = ov.label === 'positive' ? 'badge-pos' : ov.label === 'negative' ? 'badge-neg' : 'badge-neu';
    
    // Build flags for sarcasm and negation
    const flags = [];
    if (ov.sarcasm_detected) flags.push('<span class="cflag">⚠ Sarcasm</span>');
    if (ov.negation_detected) flags.push('<span class="cflag2">↩ Negation</span>');
    
    const asps = Object.keys(r.aspect_scores || {}).map(a => `<span class="text-secondary">${a}</span>`).join(' · ');
    
    // Build detailed view - shows aspect scores when expanded
    const det = Object.entries(r.aspect_scores || {}).map(([a, s]) =>
      `→ ${a}: <strong>${s.label}</strong> (${s.compound > 0 ? '+' : ''}${s.compound})`
    ).join('<br>');
    
    return `
      <div class="comment-item ${cls}" onclick="toggleDet(${r.id})">
        <div class="d-flex align-items-start justify-content-between gap-2">
          <div class="fw-medium" style="font-size:13px;">${r.comment.slice(0, 110)}${r.comment.length > 110 ? '…' : ''}</div>
          <span class="badge ${bc}">${ov.label}</span>
        </div>
        <div class="text-secondary" style="font-size:11px;">
          <span>Score: <strong>${ov.compound > 0 ? '+' : ''}${ov.compound}</strong></span>
          ${flags.join(' ')} ${asps}
        </div>
        <div class="comment-detail" id="d${r.id}">
          <strong>TextBlob raw:</strong> ${ov.raw_textblob} → adjusted: ${ov.compound}<br>
          <strong>Pos keywords:</strong> ${ov.pos_words.join(', ') || 'none'}<br>
          <strong>Neg keywords:</strong> ${ov.neg_words.join(', ') || 'none'}<br>
          ${det}
        </div>
      </div>
    `;
  }).join('');
}

// ── TOGGLE COMMENT DETAIL ──────────────────────────────────────────────
// Shows/hides the detailed view of a comment when clicked
// id: The comment ID
function toggleDet(id) {
  const el = document.getElementById('d' + id);
  // Toggle display: if it's block, hide it; if hidden, show it
  el.style.display = el.style.display === 'block' ? 'none' : 'block';
}

// ── FILTER COMMENTS ──────────────────────────────────────────────────────
// Applies sentiment filter to the comment list
// f: The filter value ('all', 'positive', 'negative', etc.)
// btn: The clicked button element to set active state
function filt(f, btn) {
  curFilt = f;  // Store the current filter globally
  // Remove active class from all filter tabs
  document.querySelectorAll('.filter-tab').forEach(b => b.classList.remove('active'));
  // Add active class to the clicked tab
  btn.classList.add('active');
  // Apply the filter
  applyFilt();
}

// ── SEARCH COMMENTS ──────────────────────────────────────────────────────
// Filters comments by text search - called when user types in search box
function searchComments() { applyFilt(); }

// ── APPLY FILTER ─────────────────────────────────────────────────────────
// Applies both sentiment filter and text search to the comment list
function applyFilt() {
  if (!D) return;  // No data yet
  const q = document.getElementById('searchIn').value.toLowerCase();
  let r = D.records;
  
  // Apply sentiment filter
  if (curFilt === 'positive') r = r.filter(x => x.overall.label === 'positive');
  else if (curFilt === 'negative') r = r.filter(x => x.overall.label === 'negative');
  else if (curFilt === 'neutral') r = r.filter(x => x.overall.label === 'neutral');
  else if (curFilt === 'sarcasm') r = r.filter(x => x.overall.sarcasm_detected);
  else if (curFilt === 'negation') r = r.filter(x => x.overall.negation_detected);
  
  // Apply text search
  if (q) r = r.filter(x => x.comment.toLowerCase().includes(q));
  
  renderComments(r);
}

// ── EXPORT CSV ────────────────────────────────────────────────────────────
// Exports the current comments as a CSV file download
async function exportCSV() {
  const lines = document.getElementById('ta').value.split('\n').filter(l => l.trim());
  if (!lines.length) { toast('No data to export', 'error'); return; }
  toast('Preparing CSV…', 'info');
  
  // Send to /export_csv endpoint
  const r = await fetch('/export_csv', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ comments: lines })
  });
  
  // Download the CSV file
  const blob = await r.blob();  // Get response as a Blob (binary data)
  const url = URL.createObjectURL(blob);  // Create a URL for the blob
  const a = document.createElement('a');  // Create a hidden link element
  a.href = url;
  a.download = `sentiment_${Date.now()}.csv`;  // Filename with timestamp
  a.click();  // Trigger the download
  URL.revokeObjectURL(url);  // Clean up the URL
  toast('CSV downloaded!', 'success');
}

// ── NAVIGATE TO SECTION ──────────────────────────────────────────────────
// Smooth scrolls to a section and updates nav link active state
// section: The section ID ('dashboard', 'comments', 'aspects', 'insights')
function navTo(section) {
  const target = document.getElementById('section-' + section);
  if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  // Update active state of nav links
  document.querySelectorAll('.navbar-nav .nav-link').forEach(l => l.classList.remove('active'));
  document.querySelectorAll('.navbar-nav .nav-link').forEach(l => {
    if (l.textContent.toLowerCase().includes(section)) l.classList.add('active');
  });
}

// ── KEYBOARD SHORTCUTS ──────────────────────────────────────────────────
// Adds keyboard shortcuts for common actions
document.addEventListener('keydown', (e) => {
  // Ctrl+Enter: Run analysis
  if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); runAnalysis(); }
  // Escape: Hide quick result
  if (e.key === 'Escape') { document.getElementById('quickRes').style.display = 'none'; }
});

// ── AUTO-RUN ON LOAD ─────────────────────────────────────────────────────
// When the page loads, populate the batch dropdown, load the default
// sample comments, and automatically run analysis. This gives a great
// first impression - the dashboard is populated immediately.
window.addEventListener('DOMContentLoaded', async () => {
  await loadBatchList();  // populate the batch dropdown first so loadSample() has a value to read
  await loadSample();     // load the default batch's comments into the textarea
  setTimeout(() => runAnalysis(), 500);  // Run analysis after 500ms
});
</script>
</body>
</html>"""


# FLASK ROUTES - API Endpoints

# These are the HTTP endpoints that the frontend calls.
# Each route handles a specific request from the dashboard.


# ── ROUTE: HOME ─────
# Serves the main dashboard HTML
# When user visits http://127.0.0.1:5000/, this returns the HTML template
# @app.route() is a decorator that tells Flask which URL triggers this function
@app.route("/")
def index():
    # Return the HTML string - Flask automatically sets Content-Type: text/html
    return HTML

# ── ROUTE: BATCHES ────
# Returns the list of available sample batches so the frontend can build
# the "Load Sample" dropdown. Kept separate from /load_sample so the
# dropdown can be populated once on page load without pulling every
# batch's full comment list over the wire.
@app.route("/batches")
def batches():
    # Build a lightweight list: just id, display name, and comment count per batch
    # SAMPLE_BATCHES.items() preserves insertion order (batch1 -> batch5) in Python 3.7+
    batch_list = [
        {"id": bid, "name": info["name"], "count": len(info["comments"])}
        for bid, info in SAMPLE_BATCHES.items()
    ]
    return jsonify({"batches": batch_list, "default": DEFAULT_BATCH})

# ── ROUTE: LOAD SAMPLE ────
# Returns the sample comments for a given batch as JSON.
# Called by the frontend when "Load Sample" button is clicked.
# Accepts an optional ?batch=<id> query param; falls back to DEFAULT_BATCH
# if missing, and to batch1/RAW_COMMENTS if an unknown id is passed.
@app.route("/load_sample")
def load_sample():
    # request.args.get() reads a query string parameter, e.g. /load_sample?batch=batch3
    batch_id = request.args.get("batch", DEFAULT_BATCH)
    # Look up the requested batch; fall back to the default batch if the id is unknown
    info = SAMPLE_BATCHES.get(batch_id, SAMPLE_BATCHES[DEFAULT_BATCH])
    # jsonify() converts Python dict to JSON string with correct Content-Type
    return jsonify({"batch": batch_id, "name": info["name"], "comments": info["comments"]})

# ── ROUTE: SINGLE ANALYSIS ────
# Analyses a single comment using the sentiment_engine
# Called by the "Quick Analyse" feature
@app.route("/single", methods=["POST"])
def single():
    # Get the comment from the request body (JSON)
    # request.get_json() parses the JSON from the POST body
    c = request.get_json().get("comment", "").strip()
    # If empty, return error with 400 Bad Request status
    if not c: return jsonify({"error": "empty"}), 400
    
    # Score the overall sentiment using TextBlob + custom logic
    ov = score_sentiment(c)
    # Detect which aspects are mentioned using keyword matching
    aspects = detect_aspects(c)
    # Score each aspect individually using sentence extraction
    asp_scores = {a: score_aspect_sentiment(c, a) for a in aspects if a != "general"}
    
    # Return results as JSON
    return jsonify({"comment": c, "overall": ov, "aspects": aspects, "aspect_scores": asp_scores})

# PROCESS FUNCTION - Shared Analysis Logic

# Core processing logic - shared between /analyse and /export_csv
# Takes a list of comments, runs the NLP pipeline, returns structured data
# This is a helper function, not a route - it's called by the routes


# ── PROCESS FUNCTION ───
# Core processing logic - shared between /analyse and /export_csv
# Takes a list of comments, runs the NLP pipeline, returns structured data
# This is a helper function, not a route - it's called by the routes
def _process(comments):
    # Filter out empty lines and strip whitespace
    comments = [c.strip() for c in comments if c.strip()]
    if not comments: return None  # Return None if no valid comments
    
    # ─PROCESS EACH COMMENT ─
    records = []
    # enumerate gives us both the index (i) and the comment (c)
    for i, c in enumerate(comments):
        # Detect which aspects are mentioned in this comment
        aspects = detect_aspects(c)
        # Get the overall sentiment score for the entire comment
        overall = score_sentiment(c)
        # Get NLTK tokens and remove stopwords for keyword analysis
        tokens = remove_stopwords(nltk_tokenize(c))
        # Score each aspect individually for this comment
        asp_scores = {a: score_aspect_sentiment(c, a) for a in aspects if a != "general"}
        
        # Store all results for this comment
        records.append({
            "id": i + 1,  # 1-based ID for human readability
            "comment": c,
            "aspects": aspects,
            "aspect_scores": asp_scores,
            "overall": overall,
            "tokens": tokens[:10],  # Store first 10 tokens for keyword analysis
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # ─ AGGREGATE STATISTICS ──
    total = len(records)
    # Count positive, negative, neutral comments
    pos = sum(1 for r in records if r["overall"]["label"] == "positive")
    neg = sum(1 for r in records if r["overall"]["label"] == "negative")
    neu = total - pos - neg
    
    # ─ASPECT AGGREGATION ──
    # defaultdict ensures we don't get KeyError for new aspects
    # It automatically creates new keys with the factory function
    agg = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0, "compounds": [], "count": 0})
    for rec in records:
        # For each aspect in this comment
        for a, s in rec["aspect_scores"].items():
            # Increment the appropriate label count
            agg[a][s["label"]] += 1
            # Store the compound score for averaging later
            agg[a]["compounds"].append(s["compound"])
            # Increment the mention count
            agg[a]["count"] += 1
    
    # ── ASPECT SUMMARY ──
    # Convert the aggregated data into percentage summaries
    asp_sum = {}
    for a, g in agg.items():
        n = g["count"] or 1  # Avoid division by zero
        asp_sum[a] = {
            "positive_pct": round(g["positive"] / n * 100),
            "negative_pct": round(g["negative"] / n * 100),
            "neutral_pct": round(g["neutral"] / n * 100),
            "avg_compound": round(sum(g["compounds"]) / len(g["compounds"]), 3),
            "mention_count": g["count"]
        }
    
    # ── INSIGHTS ─────
    # Sort aspects by average compound score (worst first - most negative)
    # This helps identify which aspects need attention
    ranked = sorted(asp_sum.items(), key=lambda x: x[1]["avg_compound"])
    
    # Fix First: aspects with >= 40% negative sentiment
    # These are the biggest problems that need immediate attention
    fix = [a for a, s in ranked if s["negative_pct"] >= 40][:3]  # Top 3
    
    # Keep Doing: aspects with >= 60% positive sentiment
    # These are the strengths of the university
    keep = [a for a, s in ranked if s["positive_pct"] >= 60][:3]  # Top 3
    
    # Controversial: aspects with near 50/50 split (35-65% positive AND 35-65% negative)
    # These are polarizing - users have strong opinions on both sides
    watch = [a for a, s in ranked if 35 <= s["positive_pct"] <= 65 and 35 <= s["negative_pct"] <= 65][:2]
    
    # ── ALERTS ──────
    # Generate alerts based on negative percentage thresholds
    alerts = []
    for a, s in asp_sum.items():
        if s["negative_pct"] >= 60:
            # HIGH severity: more than 60% negative
            alerts.append({"aspect": a, "severity": "HIGH", "message": f"{s['negative_pct']}% negative mentions"})
        elif s["negative_pct"] >= 40:
            # MEDIUM severity: 40-60% negative
            alerts.append({"aspect": a, "severity": "MEDIUM", "message": f"{s['negative_pct']}% negative mentions"})
    
    # ── KEYWORD FREQUENCY ─────
    # Count how often each word appears across all comments
    # This is done using NLTK tokens (with stopwords removed)
    freq = defaultdict(int)
    for rec in records:
        for t in rec["tokens"]:
            if len(t) > 3:  # Only count meaningful words (length > 3 characters)
                freq[t] += 1
    # Sort by frequency (highest first) and take top 20
    kws = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # ── RETURN RESULT ───
    # Return a structured dictionary with all analysis results
    return {
        "records": records,
        "total": total,
        "overall": {
            "positive_pct": round(pos / total * 100),
            "negative_pct": round(neg / total * 100),
            "neutral_pct": round(neu / total * 100),
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu
        },
        "aspect_summary": asp_sum,
        "fix_first": fix,
        "keep_doing": keep,
        "controversial": watch,
        "alerts": alerts,
        "top_keywords": kws,
        "analysed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ROUTE: ANALYSE - Batch Analysis

# Processes batch comments and returns JSON results
# Called by the "Run Analysis" button

# ── ROUTE: ANALYSE ─────
# Processes batch comments and returns JSON results
# Called by the "Run Analysis" button
@app.route("/analyse", methods=["POST"])
def analyse():
    # Get the comments array from the request body
    # request.get_json() parses the JSON from the POST body
    data = request.get_json()  # <--- FIXED: This line was missing!
    # Process the comments using the shared _process function
    result = _process(data.get("comments", []))
    # If no valid comments, return error
    if not result:
        return jsonify({"error": "No valid comments"}), 400
    # Return the results as JSON
    return jsonify(result)


# ROUTE: EXPORT CSV - CSV Download

# Processes comments and returns a downloadable CSV file


# ── ROUTE: EXPORT CSV ───
# Processes comments and returns a downloadable CSV file
@app.route("/export_csv", methods=["POST"])
def export_csv():
    # Get the comments array from the request body
    # request.get_json() parses the JSON from the POST body
    data = request.get_json()  # <--- FIXED: This line was missing!
    # Process the comments using the shared _process function
    result = _process(data.get("comments", []))
    if not result:
        return jsonify({"error": "No data"}), 400
    
    # ── CREATE CSV IN MEMORY ──────
    # StringIO creates a file-like object in memory (not on disk)
    out = io.StringIO()
    # Create a CSV writer that writes to the StringIO object
    w = csv.writer(out)
    
    # Write the header row with column names
    w.writerow(["ID", "Comment", "Label", "Polarity", "Subjectivity", "Aspects", "Sarcasm", "Negation", "TextBlob Raw"])
    
    # Write each comment's data as a row
    for r in result["records"]:
        ov = r["overall"]
        w.writerow([
            r["id"],
            r["comment"],
            ov["label"],
            ov["compound"],
            ov["subjectivity"],
            ", ".join(r["aspects"]),
            ov["sarcasm_detected"],
            ov["negation_detected"],
            ov["raw_textblob"]
        ])
    
    out.seek(0)  # Rewind to the beginning of the file
    
    # ── RETURN AS CSV DOWNLOAD ─────
    # Generate filename with timestamp for uniqueness
    fn = f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    # Return as a Flask Response with proper CSV headers
    return Response(
        out.getvalue(),  # Get the CSV string from StringIO
        mimetype="text/csv",  # Tell browser it's a CSV file
        headers={"Content-Disposition": f"attachment;filename={fn}"}  # Force download
    )

# ── OPEN BROWSER ────
# Function to automatically open the dashboard in the default browser
def open_browser():
    # webbrowser.open() launches the default browser with the given URL
    webbrowser.open("http://127.0.0.1:5000")

# ── MAIN ENTRY POINT ────
# This code runs when the script is executed directly (not imported)
if __name__ == "__main__":
    # Print startup message to the terminal
    print("\n========================================")
    print("  🎓 UniSentiment — Sentiment Intelligence")
    print("  URL: http://127.0.0.1:5000")
    print("  Press CTRL+C to stop")
    print("========================================\n")
    
    # Open browser after 1.5 seconds (gives Flask time to start)
    # We use a timer so it doesn't block the Flask server
    # threading.Timer creates a thread that runs the function after the delay
    threading.Timer(1.5, open_browser).start()
    
    # Run Flask app on port 5000
    # debug=False for production-like performance (no debugger/reloader)
    # host='127.0.0.1' means only local connections (localhost)
    app.run(debug=False, port=5000)
