# SENTIMENT ENGINE - NLP Brain of UniSentiment

# This file contains ALL the Natural Language Processing logic.
# It has NO dependency on Flask - can be used in any Python script.
# 
# What it does:
# - Takes raw student comments
# - Detects which university aspects are mentioned
# - Scores sentiment (positive/negative/neutral)
# - Handles tricky cases (negation, sarcasm)
# - Generates insights and alerts



# IMPORTS

# Standard library 
import re                                          # re: regex module, used for keyword/pattern matching (aspects, sarcasm)
import random                                      # random: used to generate pseudo-random but reproducible timestamps
from datetime import datetime, timedelta           # datetime: current time object; timedelta: time differences/offsets
from collections import defaultdict                # defaultdict: dict subclass that auto-creates missing keys with a default value

#  NLTK 
import nltk                                         # nltk: the Natural Language Toolkit library itself (needed for .download())
from nltk.tokenize import word_tokenize, sent_tokenize  # word_tokenize: splits text into words; sent_tokenize: splits text into sentences
from nltk.corpus import stopwords                   # stopwords: NLTK's built-in list of common words per language

# TextBlob 
from textblob import TextBlob                       # TextBlob: wraps text and exposes .sentiment, .tags, etc.


# NLTK SETUP - Download Required Data

# These are pre-trained models and corpora that NLTK needs:
# - punkt: Tokenizer models (splits text into words/sentences)
# - stopwords: Common words like "the", "is", "at"
# - wordnet: Dictionary for word meanings
# - averaged_perceptron_tagger: Part-of-speech tagger
# 
# quiet=True prevents console spam if already downloaded.

for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "brown", "conll2000", "movie_reviews"):
    # _pkg: loop variable holding the current package name string on each iteration
    nltk.download(_pkg, quiet=True)                 # download the package; quiet=True suppresses console spam if already installed

STOP_WORDS = set(stopwords.words("english"))        # STOP_WORDS: a set (for O(1) lookup) of English stopwords like "the", "is", "a"


# 1. RAW_COMMENTS - The Sample Dataset

# This is our sample dataset - 60 realistic student comments
# about university life. We created these by hand to include:
# - Positive and negative opinions
# - Mixed feelings (both good and bad in one comment)
# - Sarcastic remarks ("oh wow, another parking ticket")
# - Negation ("not good", "never works")
# 
# Users can also paste their own comments, but these 60 are
# pre-loaded so the dashboard works instantly when you run it.


# DATASET  -  60 realistic student comments with simulated timestamps


RAW_COMMENTS = [                                                                                          # RAW_COMMENTS: the master list of all 60 sample comments
    "The professors here are absolutely amazing and always willing to help after class.",                # comment 1: strongly positive, mentions professors
    "Wi-Fi drops every 5 minutes. Can't even submit assignments online. Totally unacceptable.",            # comment 2: negative, mentions wi-fi, contains negation ("can't")
    "Parking is a nightmare. Spent 45 minutes looking for a spot and was late to my exam.",                 # comment 3: negative, mentions parking
    "I love this university! Professors are knowledgeable, library resources are excellent.",               # comment 4: positive, mentions professors + library (multi-aspect)
    "The cafeteria has improved a lot lately. Fresh food and affordable prices now.",                       # comment 5: positive, mentions cafeteria
    "Administration is completely unresponsive. Sent 3 emails and got zero replies.",                       # comment 6: negative, mentions administration
    "Campus safety is top-notch. Security patrols regularly and I feel safe at night.",                     # comment 7: positive, mentions campus safety
    "Wi-Fi in the dorms is so slow it is practically useless. Very frustrating.",                            # comment 8: negative, mentions wi-fi
    "Oh wow, another parking ticket. Thanks university, very cool. Maybe build more parking.",              # comment 9: sarcastic, mentions parking (contains sarcasm markers "oh wow", "very cool")
    "My professor actually remembered my name and gave personalized feedback. Wonderful.",                  # comment 10: positive, mentions professors
    "The library has an incredible collection and study rooms are always clean.",                           # comment 11: positive, mentions library
    "Facilities are crumbling. Broken AC in classrooms and a leaking roof in building C.",                   # comment 12: negative, mentions facilities
    "Enrolled expecting great things. Average professors, terrible food, Wi-Fi barely works.",               # comment 13: negative/mixed, mentions professors + cafeteria + wi-fi
    "Love the professor office hours system. So accessible and my grades improved a lot.",                   # comment 14: positive, mentions professors
    "The cafeteria vegan options are actually solid now. Much better than last year.",                       # comment 15: positive, mentions cafeteria
    "Parking fees doubled but the lot conditions got worse. Potholes everywhere.",                            # comment 16: negative, mentions parking
    "Campus security responded to a medical emergency in under 2 minutes. Impressive.",                      # comment 17: positive, mentions campus safety
    "The gym facilities are brand new and well-maintained. Best fitness centre I have used.",                 # comment 18: positive, mentions facilities (gym)
    "Administration lost my scholarship paperwork TWICE. Nearly lost my funding over it.",                    # comment 19: negative, mentions administration
    "I expected to hate it here but professors have completely changed my perspective.",                      # comment 20: positive, mentions professors, contains negation ("did not" implied contrast)
    "Wi-Fi is acceptable in most buildings but drops out completely in the east wing.",                       # comment 21: mixed, mentions wi-fi
    "Cafeteria breakfast is good, lunch mediocre, dinner borderline inedible.",                                # comment 22: mixed/negative, mentions cafeteria
    "Parking is what it is on an urban campus. The shuttle service helps a lot though.",                       # comment 23: neutral/mixed, mentions parking
    "My TA was totally unprepared and clearly did not understand the material at all.",                       # comment 24: negative, mentions professors (TA), contains negation ("did not")
    "Beautiful campus and great community. Facilities a bit dated but environment great.",                     # comment 25: mixed/positive, mentions facilities
    "The administration online portal is actually user-friendly now. Big improvement.",                        # comment 26: positive, mentions administration
    "Cannot believe they charge so much for parking and do not plow the lot in winter.",                        # comment 27: negative, mentions parking, contains negation ("cannot", "do not")
    "Professors in the engineering department are world-class. Research opportunities great.",                  # comment 28: positive, mentions professors
    "The library extended its hours for finals week. Thoughtful decision by management.",                       # comment 29: positive, mentions library
    "Cafeteria food made me sick twice this semester. Health and safety must investigate.",                      # comment 30: strongly negative, mentions cafeteria + campus safety
    "Campus security is excellent this year. Escort service after dark is fantastic.",                          # comment 31: positive, mentions campus safety
    "IT help desk resolved my Wi-Fi issue in under an hour. Impressive turnaround.",                             # comment 32: positive, mentions wi-fi
    "Administration flagged my account incorrectly. Three months to fix. Absolutely maddening.",                 # comment 33: negative, mentions administration
    "Professors connect course material to real-world applications. Classes feel relevant.",                     # comment 34: positive, mentions professors
    "Facilities team fixed a broken elevator within 24 hours. Very responsive lately.",                          # comment 35: positive, mentions facilities
    "Oh sure, the university cares about students. That is why heating broke in January.",                        # comment 36: sarcastic, mentions facilities (contains sarcasm marker "oh sure")
    "Overall positive experience. Wi-Fi could be better but professors compensate.",                              # comment 37: mixed/positive, mentions wi-fi + professors
    "Parking situation is genuinely dire. No solutions, no communication, just tickets.",                          # comment 38: negative, mentions parking, contains negation ("no solutions", "no communication")
    "New cafeteria renovation made a huge difference. Quality and variety both improved.",                          # comment 39: positive, mentions cafeteria
    "Campus feels unsafe near the north parking lot at night. Lighting is poor.",                                    # comment 40: negative, mentions campus safety + parking
    "My professor went beyond the syllabus and introduced cutting-edge research. Incredible.",                       # comment 41: positive, mentions professors
    "Administration denied my appeal without any explanation. Zero transparency.",                                   # comment 42: negative, mentions administration, contains negation ("without", "zero")
    "Wi-Fi upgrades this semester have been noticeable. Streaming lectures no longer buffers.",                       # comment 43: positive, mentions wi-fi, contains negation ("no longer")
    "Library staff are knowledgeable and go out of their way to help. Underappreciated.",                              # comment 44: positive, mentions library
    "Cafeteria staff are friendly but food quality is inconsistent day to day.",                                       # comment 45: mixed, mentions cafeteria
    "Parking app crashes every time I try to pay. Technology should not be this unreliable.",                           # comment 46: negative, mentions parking, contains negation ("should not")
    "Campus safety organised a self-defence workshop that was incredibly well-run.",                                     # comment 47: positive, mentions campus safety
    "Older classroom facilities need renovation. Projectors break and acoustics are poor.",                               # comment 48: negative, mentions facilities
    "New student orientation brilliantly organised by administration. Great first week.",                                  # comment 49: positive, mentions administration
    "Not impressed. Everything feels like an afterthought at this university.",                                            # comment 50: negative, general, contains negation ("not impressed")
    "Professor was brilliant but moved too fast. Students clearly struggling, no adjustments.",                            # comment 51: mixed, mentions professors, contains negation ("no adjustments")
    "Campus green spaces are beautiful and well-maintained. Great for studying outdoors.",                                  # comment 52: positive, mentions facilities (campus)
    "Administration processed my transfer credits quickly and accurately. Smooth experience.",                             # comment 53: positive, mentions administration
    "Library digital journal access is phenomenal. Research for my thesis has been seamless.",                              # comment 54: positive, mentions library
    "Wi-Fi in the stadium is surprisingly strong. Unexpected given how bad it is elsewhere.",                                # comment 55: mixed/positive, mentions wi-fi
    "Cafeteria prices are outrageous for the quality provided. Dining plan feels like a rip-off.",                            # comment 56: negative, mentions cafeteria
    "My academic advisor was patient, thorough and genuinely invested in my success.",                                       # comment 57: positive, mentions professors (advisor)
    "Facilities have declined noticeably. Bathrooms dirty, hallways unlit, equipment broken.",                                # comment 58: negative, mentions facilities
    "Professors are hit or miss. Some incredible, others seem like they would rather not be here.",                            # comment 59: mixed, mentions professors, contains negation ("rather not")
    "Campus security stopped a theft attempt near my car. Quick response and followed up.",                                    # comment 60: positive, mentions campus safety
]                                                                                                            # end of RAW_COMMENTS list

RAW_COMMENTS_2 = [                                     # Batch 2: "Fall Semester Intake" - 30 comments, first-year/new-student focus
    "Orientation week was chaotic but the professors who spoke were genuinely inspiring.",
    "New Wi-Fi routers in the freshman dorms have made a night-and-day difference.",
    "First week and already got two parking tickets for signage that isn't even visible.",
    "The library ran a fantastic tour for new students. Staff were patient with all our questions.",
    "Cafeteria lines during orientation week were brutal, easily 40 minutes just to get a tray.",
    "Administration sent my acceptance letter to the wrong email and then blamed me for missing it.",
    "Campus security walked freshmen to their dorms every night during welcome week. Really reassuring.",
    "Wi-Fi password kept changing without notice during the first week. Nobody could get online.",
    "Oh great, another 'temporary' construction fence blocking the only path to class. Very cool planning.",
    "My advisor sat with me for an hour going over my course schedule. Above and beyond.",
    "The freshman library orientation was rushed and honestly forgettable.",
    "Facilities hadn't even finished painting the new dorm wing when we moved in.",
    "Enrolled excited for a fresh start; professors are decent but the food is genuinely inedible.",
    "Professors in the intro seminar made a real effort to learn everyone's names by week two.",
    "Cafeteria added a new international food station and it's actually amazing.",
    "Parking permits for new students cost more than upperclassmen pay. Doesn't add up.",
    "Campus safety gave a great self-defence demo during orientation. Felt genuinely useful.",
    "The new student gym orientation was well organised and the equipment is spotless.",
    "Administration still hasn't processed my housing deposit refund from two months ago.",
    "Didn't expect much from my first lecture but the professor completely won me over.",
    "Wi-Fi in the freshman quad is patchy at best, solid nowhere near the courtyard benches.",
    "Cafeteria breakfast during welcome week was surprisingly good, better than I expected.",
    "Parking for move-in day was actually well organised this year, no complaints there.",
    "My orientation leader clearly hadn't been trained properly and gave us wrong information.",
    "Freshman dorm facilities are dated but the community atmosphere makes up for it.",
    "Administration's new student portal for registration is honestly a huge upgrade.",
    "Cannot believe how disorganised the parking situation was during move-in weekend.",
    "Professors hosting the welcome lectures clearly love what they teach. Great first impression.",
    "The library's new student library card process took less than five minutes. Smooth.",
    "Cafeteria food poisoning scare during orientation week was never properly explained by staff.",
]                                                       # end of RAW_COMMENTS_2

RAW_COMMENTS_3 = [                                     # Batch 3: "Spring Semester Wrap-up" - 30 comments, end-of-year/exam focus
    "Professors during finals week held extra office hours without being asked. Really appreciated it.",
    "Wi-Fi crashed campus-wide during the online final exam window. Absolute disaster.",
    "Parking near the exam hall was impossible to find and I nearly missed my final.",
    "Library extended hours to 3am for finals and it made a genuine difference to my grades.",
    "Cafeteria stayed open late during finals with free coffee. Small thing but it mattered.",
    "Administration lost track of my grade appeal for an entire semester. No apology given.",
    "Campus security added extra night patrols during finals week and I felt much safer walking back.",
    "Wi-Fi in the library basement study rooms is nonexistent during peak finals traffic.",
    "Oh fantastic, the elevator breaks down the exact week everyone needs it for exams. Typical.",
    "My professor curved the final exam generously after seeing how tough it was. Fair and kind.",
    "The library's group study room booking system crashed the week we needed it most.",
    "Facilities left broken heating in the exam hall for the entire cold snap in April.",
    "End of semester, professors seemed burnt out and grading feedback was practically nonexistent.",
    "Thank you to the professor who recorded every lecture, saved me during exam prep.",
    "Cafeteria quality noticeably dropped as the semester wound down. Feels like they stopped trying.",
    "Parking enforcement got aggressive right during reading week, tickets everywhere overnight.",
    "Campus safety escorted stressed students to counselling without judgment during finals. Kind touch.",
    "Facilities finally fixed the gym AC just as the semester ended. Better late than never I suppose.",
    "Administration processed my end-of-year transcript request in under 24 hours. Genuinely impressed.",
    "Never expected finals to go this smoothly but my professors made it manageable.",
    "Wi-Fi held up fine for most of finals week except one disastrous afternoon outage.",
    "Cafeteria ran out of vegetarian options every single day during exam week. Frustrating.",
    "Parking shuttle stopped running early during reading week with zero warning to students.",
    "My TA held a review session that was more useful than the entire semester of lectures.",
    "Facilities never fixed the broken study lounge chairs all semester despite multiple reports.",
    "Administration's end-of-year satisfaction survey felt like it actually led to real changes.",
    "Cannot understand why the library closes early on the Friday before finals begin.",
    "Professors handling the capstone reviews gave incredibly detailed, constructive feedback.",
    "The library's silent study floor was a lifesaver during the last week of the semester.",
    "Cafeteria staff were clearly exhausted by finals week but stayed friendly the whole time.",
]                                                       # end of RAW_COMMENTS_3

RAW_COMMENTS_4 = [                                     # Batch 4: "Graduate & Research Students" - 30 comments, grad-school focus
    "My thesis advisor is incredibly generous with time despite an overloaded schedule.",
    "Wi-Fi in the graduate research lab drops constantly during large data uploads. Maddening.",
    "Parking for grad students working late is nearly impossible after 6pm.",
    "The library's interlibrary loan system got me a rare journal article within two days. Excellent.",
    "Cafeteria hours don't accommodate grad students working odd research hours at all.",
    "Administration took four months to process my graduate stipend, leaving me without pay.",
    "Campus security checks on the research labs late at night, which grad students genuinely appreciate.",
    "Wi-Fi outages during a live conference presentation from my lab were embarrassing and preventable.",
    "Oh sure, cut the grad lounge budget right when we needed a functioning coffee machine most.",
    "My committee chair provided detailed line-by-line feedback on every single draft. Invaluable.",
    "The library's database access for grad research is genuinely world-class here.",
    "Facilities never resolved the mould issue in the basement research offices despite repeated complaints.",
    "Started my PhD excited, but administration bureaucracy for funding renewal is soul-crushing.",
    "Professors in my department treat grad students as colleagues, not just free labor. Refreshing.",
    "Cafeteria added late-night hours near the grad building and usage has clearly gone up.",
    "Parking permits for grad researchers doubled in price with zero improvement to availability.",
    "Campus safety responded within minutes when a lab incident happened after hours. Reassuring professionalism.",
    "Facilities upgraded the wet lab equipment finally after three years of requests.",
    "Administration denied my conference travel reimbursement without any clear explanation. Zero transparency.",
    "Never thought I would say this but the administration's new grant portal is actually intuitive.",
    "Wi-Fi bandwidth throttling during peak hours makes remote data analysis nearly impossible.",
    "Cafeteria food quality near the grad campus building is noticeably worse than main campus.",
    "Parking shuttle for the satellite research campus is unreliable and frequently late.",
    "My co-advisor pushed back my defence date without proper notice, no adjustments offered.",
    "Facilities keep the grad study carrels spotless and well maintained, small thing but appreciated.",
    "Administration handled my visa-related paperwork for a conference smoothly and quickly.",
    "Cannot believe grad students still don't have reserved parking despite years of requests.",
    "Professors on my thesis committee gave conflicting feedback that delayed my defence by months.",
    "The library's 24-hour grad reading room is one of the best resources on this campus.",
    "Cafeteria vending machines near the research building are perpetually empty or broken.",
]                                                       # end of RAW_COMMENTS_4

RAW_COMMENTS_5 = [                                     # Batch 5: "International Students" - 30 comments, international/exchange focus
    "My professor adjusted deadlines to accommodate my visa processing delays. Truly grateful.",
    "Wi-Fi at the international student housing block is unreliable, hard to call family back home.",
    "Parking permits are confusing for international students who don't drive, wasted money on one.",
    "The library's multilingual resource section made research so much easier in my first semester.",
    "Cafeteria finally added halal and kosher options this year, long overdue but appreciated.",
    "Administration's international office lost my transcript evaluation paperwork twice in one term.",
    "Campus security staff learned basic greetings in several languages, small gesture that meant a lot.",
    "Wi-Fi speed drops badly during video calls with family overseas, especially in the evenings.",
    "Oh wonderful, another visa document rejected over a formatting issue nobody explained beforehand.",
    "My academic advisor helped me navigate credit transfer from my home country without any hassle.",
    "The library hosts international student study groups that have been genuinely welcoming.",
    "Facilities in the international dorm block are noticeably older than the rest of campus housing.",
    "Arrived expecting culture shock, but professors here have been remarkably understanding and patient.",
    "Professors in my program go out of their way to explain idioms and cultural references. Thoughtful.",
    "Cafeteria staff always double-check dietary restrictions with international students. Nice attention to detail.",
    "Parking near the international student office is essentially nonexistent during peak hours.",
    "Campus safety's international student safety briefing was thorough and genuinely reassuring.",
    "Facilities upgraded the international dorm kitchens, now they actually have working stoves.",
    "Administration's visa support office took over three months to respond to a simple query.",
    "Never expected such a warm welcome, but the international student orientation exceeded expectations.",
    "Wi-Fi issues make it genuinely difficult to attend virtual family events during exam season.",
    "Cafeteria prices feel steep when converted back to my home currency, hard adjustment.",
    "Parking shuttle to the international dorms runs too infrequently on weekends.",
    "My professor never adjusted grading despite language barriers being clearly acknowledged upfront.",
    "Facilities added prayer and reflection rooms this year, a small but meaningful change for many students.",
    "Administration processed my work-study authorization quickly, faster than I expected honestly.",
    "Cannot get a straight answer from administration about international tuition payment deadlines.",
    "Professors leading the international student seminar made everyone feel included from day one.",
    "The library's international newspaper subscriptions help me stay connected to news from home.",
    "Cafeteria ran out of the international food station options constantly, felt like an afterthought.",
]                                                       # end of RAW_COMMENTS_5


# SAMPLE_BATCHES - Named registry of every loadable comment set

# Powers the "Load Sample" dropdown in the dashboard. Each entry has a
# short id (used in the URL query string), a human-readable name shown
# in the dropdown, and the actual list of comments to load.

SAMPLE_BATCHES = {                                     # SAMPLE_BATCHES: ordered registry of every sample batch, keyed by batch id
    "batch1": {"name": "Original Mix (60)",                       "comments": RAW_COMMENTS},
    "batch2": {"name": "Fall Semester Intake (30)",                "comments": RAW_COMMENTS_2},
    "batch3": {"name": "Spring Semester Wrap-up (30)",             "comments": RAW_COMMENTS_3},
    "batch4": {"name": "Graduate & Research Students (30)",        "comments": RAW_COMMENTS_4},
    "batch5": {"name": "International Students (30)",              "comments": RAW_COMMENTS_5},
}                                                       # end of SAMPLE_BATCHES dict

DEFAULT_BATCH = "batch1"                               # DEFAULT_BATCH: which batch id loads first when the dashboard opens


# 2. ASPECT_KEYWORDS - Aspect Detection Dictionary

# This dictionary maps 8 university aspects to their keywords.
# 
# How it works:
# - Each aspect (like "professors") has a list of words
# - If a comment contains "professor" or "lecturer" or "TA"...
# - We know the comment is talking about professors
# 
# Why this matters:
# - We don't just say "this comment is positive"
# - We say "this comment is positive ABOUT PROFESSORS"
# - This gives much more useful information to university staff



ASPECT_KEYWORDS = {                                  # ASPECT_KEYWORDS: dict mapping each aspect name -> list of trigger keywords
    "professors": [                                  # "professors" aspect: keywords related to teaching staff
        "professor", "professors", "lecturer", "lecturers", "faculty", "teacher",   # singular/plural forms of teaching roles
        "instructor", "teaching", "ta", "tutor", "office hours", "syllabus",        # other teaching-related terms
        "course", "class", "lecture", "academic advisor", "advisor",                # course/class related terms and advisor terms
    ],                                                # end of professors keyword list
    "cafeteria": [                                    # "cafeteria" aspect: keywords related to food services
        "cafeteria", "food", "meal", "dining", "lunch", "breakfast", "dinner",       # meal-related nouns
        "canteen", "eat", "menu", "vegan", "cafeteria staff", "dining plan",         # more food service terms
        "catering", "prices", "quality", "kitchen",                                  # generic food-quality terms
    ],                                                # end of cafeteria keyword list
    "wi-fi": [                                        # "wi-fi" aspect: keywords related to internet/network
        "wi-fi", "wifi", "internet", "network", "connection", "broadband",          # core connectivity terms
        "streaming", "online", "bandwidth", "signal", "it help desk", "it",         # tech-support / streaming terms
        "upload", "download", "connectivity",                                        # data transfer terms
    ],                                                # end of wi-fi keyword list
    "parking": [                                      # "parking" aspect: keywords related to vehicles/parking
        "parking", "park", "lot", "car park", "vehicle", "ticket", "permit",        # core parking terms
        "shuttle", "potholes", "towing", "parking app", "parking fees",             # related parking-infrastructure terms
    ],                                                # end of parking keyword list
    "library": [                                      # "library" aspect: keywords related to the library
        "library", "libraries", "librarian", "book", "books", "study room",         # core library terms
        "study rooms", "journal", "journals", "database", "research", "collection", # research-related terms
        "digital", "thesis", "reading", "resources",                                 # digital/resource terms
    ],                                                # end of library keyword list
    "administration": [                               # "administration" aspect: keywords related to admin offices
        "administration", "admin", "office", "bureaucracy", "paperwork", "email",   # core admin terms
        "portal", "appeal", "enrolment", "financial aid", "scholarship",            # admin-process terms
        "registration", "orientation", "transcript", "credits",                      # more admin-process terms
    ],                                                # end of administration keyword list
    "campus safety": [                                # "campus safety" aspect: keywords related to security
        "safety", "security", "safe", "unsafe", "guard", "guards", "patrol",        # core safety terms
        "escort", "lighting", "crime", "theft", "emergency", "police",              # crime/safety terms
        "self-defence", "self defense", "response",                                  # additional safety-related terms
    ],                                                # end of campus safety keyword list
    "facilities": [                                   # "facilities" aspect: keywords related to physical buildings
        "facilities", "facility", "building", "buildings", "classroom", "classrooms",  # building-related terms
        "gym", "gymnasium", "lab", "laboratory", "equipment", "ac", "air conditioning", # equipment/room terms
        "heating", "elevator", "lift", "bathroom", "bathrooms", "green spaces",      # infrastructure terms
        "campus", "projector", "acoustics",                                           # more facility terms
    ],                                                # end of facilities keyword list
}                                                      # end of ASPECT_KEYWORDS dict


# 3. NEGATION_WORDS - Words That Flip Sentiment

# These words flip the meaning of a sentence.
# 
# Example:
# - "The food is good" → Positive
# - "The food is NOT good" → Negative (should be!)
# 
# Without this list, "not good" would be misclassified as
# positive because "good" is a positive word.
# 
# We also include contractions like "can't" and "don't"
# because students write informally.

NEGATION_WORDS = {                                    # NEGATION_WORDS: a set of words that indicate negation in English
    "not", "no", "never", "neither", "nor", "nobody", "nothing", "nowhere",          # basic negation words
    "hardly", "barely", "scarcely", "without", "cannot", "can't", "won't",           # near-negation adverbs + contracted negatives
    "wouldn't", "shouldn't", "couldn't", "isn't", "aren't", "wasn't",                # more contracted negatives (modal/be verbs)
    "weren't", "doesn't", "don't", "didn't", "hasn't", "haven't", "hadn't",          # contracted negatives (do/have verbs)
}                                                      # end of NEGATION_WORDS set


# 4. SARCASM_PATTERNS - Catch Sarcastic Comments

# These are regex patterns that catch sarcastic comments.
# 
# TextBlob can't detect sarcasm - it sees "very cool" and
# thinks it's positive. But a student saying "very cool"
# about a parking ticket is being sarcastic (actually angry).
# 
# Our patterns catch common sarcastic phrases:
# - "oh wow" + complaint
# - "thanks university" (when something went wrong)
# - "very cool" (used ironically)
# - Lots of exclamation marks !! or ...
# 
# When these match, we force the score to be negative.

SARCASM_PATTERNS = [                                  # SARCASM_PATTERNS: list of regex strings that flag likely sarcastic phrasing
    r"\b(oh\s+sure|oh\s+wow|oh\s+great|oh\s+right)\b",   # common sarcastic interjections ("oh sure", "oh wow", etc.)
    r"\b(thanks\s+a\s+lot|thanks\s+for\s+nothing)\b",     # sarcastic "thanks" phrases
    r"\bvery\s+cool\b",                                   # sarcastic "very cool" phrase
    r"!{2,}",                                             # two or more exclamation marks in a row (often sarcastic emphasis)
    r"\.\.\.",                                            # an ellipsis, often used to imply a sarcastic pause
]                                                      # end of SARCASM_PATTERNS list


# 5. NLTK HELPERS - Tokenization and Stopword Removal

# Simple helper functions using NLTK (Natural Language Toolkit).
# 
# What they do:
# - nltk_tokenize(): Splits text into individual words
#   "The food is good" → ["the", "food", "is", "good"]
# 
# - nltk_sentences(): Splits text into sentences
#   "Food is good. Wi-Fi is bad." → ["Food is good.", "Wi-Fi is bad."]
# 
# - remove_stopwords(): Removes common words like "the", "is", "at"
#   These words carry no meaning for sentiment analysis


# 
# NLTK HELPERS
# 

def nltk_tokenize(text: str) -> list:                 # nltk_tokenize: function that takes raw text and returns lowercase word tokens
    """
    NLTK word tokenizer.
    Splits text into individual word tokens (lowercase).
    """
    return [w.lower() for w in word_tokenize(text)]   # word_tokenize splits the string into tokens; list comp lowercases each one


def nltk_sentences(text: str) -> list:                # nltk_sentences: function that splits a block of text into separate sentences
    """
    NLTK sentence tokenizer.
    Splits a paragraph into individual sentences.
    """
    return sent_tokenize(text)                        # sent_tokenize returns a list of sentence strings


def remove_stopwords(tokens: list) -> list:           # remove_stopwords: filters out common non-meaningful words from a token list
    """
    NLTK stopword filter.
    Removes common words (the, is, at, ...) that carry no sentiment.
    """
    return [t for t in tokens if t not in STOP_WORDS and t.isalpha()]  # keep token only if it's not a stopword AND is purely alphabetic


# 6. TEXTBLOB SENTIMENT SCORING - Base Sentiment Scorer

# TextBlob is a pre-trained library that gives two scores:
# 
# 1. Polarity (-1 to +1):
#    - Negative words → score goes down (towards -1)
#    - Positive words → score goes up (towards +1)
#    - Neutral words → score stays around 0
# 
# 2. Subjectivity (0 to 1):
#    - 0 = factual statement ("The sky is blue")
#    - 1 = opinion ("The food is delicious")
# 
# We use TextBlob as our base scorer, then add custom
# layers on top (negation, sarcasm) to fix its weaknesses.


# 
# TEXTBLOB SENTIMENT SCORING
# 

def textblob_polarity(text: str) -> float:            # textblob_polarity: returns the polarity score of a text string
    """
    TextBlob sentiment polarity.
    Returns a float from -1.0 (very negative) to +1.0 (very positive).
    TextBlob uses a pattern-based approach on its built-in sentiment lexicon.
    """
    return TextBlob(text).sentiment.polarity          # wrap text in TextBlob, access .sentiment, return the .polarity attribute


def textblob_subjectivity(text: str) -> float:        # textblob_subjectivity: returns the subjectivity score of a text string
    """
    TextBlob subjectivity score.
    Returns 0.0 (objective fact) to 1.0 (highly subjective/opinionated).
    Useful for identifying neutral factual statements.
    """
    return TextBlob(text).sentiment.subjectivity      # wrap text in TextBlob, access .sentiment, return the .subjectivity attribute


# 7. NEGATION DETECTION - Custom Layer on Top of TextBlob

# This fixes one of TextBlob's weaknesses.
# 
# How it works:
# 1. Scan the comment for any NEGATION word ("not", "never", etc.)
# 2. If found, look at the next 3 words
# 3. If there's a meaningful word there (not a stopword)...
# 4. ...then we flip the sentiment!
# 
# "not good" → TextBlob sees +0.7 → We flip to -0.7
# "not bad"  → TextBlob sees -0.7 → We flip to +0.35 (half strength)
# 
# Why half strength for "not bad"?
# Because "not bad" is mildly positive, not strongly positive.


# 
# NEGATION DETECTION  (custom, applied on top of TextBlob score)
# 

def detect_negation(tokens: list, window: int = 3) -> bool:   # detect_negation: scans tokens for negation words within a window
    """
    Sliding-window negation detector using NLTK tokens.
    Scans tokens for negation words (not, no, never, can't ...).
    Returns True if any negation word is found within `window` tokens
    before a meaningful word - indicating the sentiment should be flipped.
    """
    for i, tok in enumerate(tokens):                  # i: index of current token; tok: the token string itself
        if tok in NEGATION_WORDS:                     # check if the current token is one of our known negation words
            # Check the next `window` tokens for content words
            upcoming = tokens[i + 1: i + 1 + window]  # upcoming: slice grabbing the next `window` tokens after the negation word
            content = [t for t in upcoming if t not in STOP_WORDS and t.isalpha()]  # content: filter upcoming tokens to meaningful words only
            if content:                               # if there is at least one meaningful word after the negation word
                return True                           # negation context found -> return True immediately
    return False                                      # no negation context found anywhere in the tokens -> return False


def apply_negation(polarity: float, tokens: list) -> float:   # apply_negation: flips/reduces polarity if negation context exists
    """
    If negation is detected in the token list, flip the TextBlob polarity.
    'not good' -> TextBlob gives +0.7, we flip to -0.7.
    'not bad'  -> TextBlob gives -0.7, we flip to +0.35 (partial positive).
    """
    if detect_negation(tokens):                       # only adjust polarity if negation was actually detected
        return -polarity * (0.5 if polarity < 0 else 1.0)  # flip sign; if original was negative, also halve magnitude (e.g. "not bad")
    return polarity                                   # no negation detected -> return polarity unchanged


# 8. SARCASM DETECTION - Custom Regex Layer

# This fixes another TextBlob weakness - it can't detect sarcasm.
# 
# How it works:
# - We check the comment against our SARCASM_PATTERNS
# - If ANY pattern matches → comment is sarcastic
# - We force the score to be NEGATIVE
# 
# Example:
# Comment: "Oh wow, another parking ticket. Thanks university."
# TextBlob sees "wow" and "thanks" → thinks it's positive!
# But it's clearly negative/sarcastic.
# Our regex catches "oh wow" and overrides to negative.
# 
# Known limitation: Only catches patterns we listed.
# Future work: Train a proper sarcasm classifier.
# ============================================================

# SARCASM DETECTION  (custom regex)


def detect_sarcasm(text: str) -> bool:                # detect_sarcasm: checks text against known sarcasm regex patterns
    """
    Heuristic sarcasm detector.
    TextBlob cannot detect sarcasm on its own, so we layer regex patterns
    on top to catch common ironic expressions used by students.
    If sarcasm is detected, the final polarity is forced negative.
    """
    lower = text.lower()                              # lower: lowercase version of the text so patterns match case-insensitively
    return any(re.search(p, lower) for p in SARCASM_PATTERNS)  # return True if ANY sarcasm pattern matches anywhere in the text


# 9. ASPECT DETECTION - Figures Out What Topic is Mentioned

# This figures out WHICH part of the university the comment is about.
# 
# How it works:
# 1. Loop through each aspect (professors, cafeteria, etc.)
# 2. Check if any of that aspect's keywords appear in the comment
# 3. If yes → add that aspect to the list
# 4. One comment can mention MULTIPLE aspects
# 
# Example:
# "The professor is great but the cafeteria food is terrible"
# → Detects: ["professors", "cafeteria"]
# 
# If no aspect is found → defaults to ["general"]


# ASPECT DETECTION


def detect_aspects(text: str) -> list:                # detect_aspects: returns which university aspects a comment mentions
    """
    Keyword-based aspect extractor.
    Uses regex word-boundary matching on the ASPECT_KEYWORDS dictionary.
    Returns list of aspect names found in the comment.
    NLTK sentence tokenization is used first so multi-sentence comments
    are checked sentence-by-sentence for more accurate aspect-sentiment pairing.
    """
    lower = text.lower()                              # lower: lowercase version of the comment for case-insensitive matching
    found = []                                         # found: list that will collect aspect names detected in this comment
    for aspect, keywords in ASPECT_KEYWORDS.items():  # aspect: aspect name string; keywords: list of keywords for that aspect
        for kw in keywords:                            # kw: a single keyword string from the current aspect's keyword list
            if re.search(r'\b' + re.escape(kw) + r'\b', lower):  # \b = word boundary; re.escape avoids special-char issues in kw
                found.append(aspect)                   # keyword matched -> record this aspect as present in the comment
                break                                  # stop checking further keywords for this aspect (one match is enough)
    return found if found else ["general"]            # if no aspects matched, default to ["general"] so callers always get a list


# 10. CORE SCORING - The 5-Stage Pipeline

# This is the HEART of the entire system.
# Every comment passes through these 5 stages:
# 
# STAGE 1: Tokenization (NLTK)
#   - Split text into words, remove stopwords
# 
# STAGE 2: Base Scoring (TextBlob)
#   - Get raw polarity (-1 to +1) and subjectivity (0 to 1)
# 
# STAGE 3: Negation Adjustment (Custom)
#   - Check for "not", "never", "can't" → flip polarity
# 
# STAGE 4: Sarcasm Override (Custom)
#   - Check for sarcastic patterns → force negative
# 
# STAGE 5: Label Assignment
#   - Polarity >= 0.05 → Positive
#   - Polarity <= -0.05 → Negative
#   - Everything in between → Neutral
# 
# Returns: score, label, evidence words, flags


# CORE SCORING  (TextBlob + NLTK + custom layers)


def score_sentiment(text: str) -> dict:               # score_sentiment: the master function scoring a single comment's sentiment
    """
    Full sentiment scoring pipeline for one comment:

    Step 1 - NLTK tokenizes the text into words
    Step 2 - TextBlob scores raw polarity and subjectivity
    Step 3 - Negation check (NLTK tokens) adjusts the polarity
    Step 4 - Sarcasm check (regex) forces polarity negative if irony found
    Step 5 - Polarity is mapped to a label: positive / negative / neutral

    Returns a dict with compound score, label, and explanation flags.
    """
    # Step 1: NLTK tokenization
    tokens = nltk_tokenize(text)                      # tokens: lowercase word tokens for the whole comment (used for negation check)
    clean_tokens = remove_stopwords(tokens)           # clean_tokens: tokens with stopwords removed (used for keyword/aspect evidence)

    # Step 2: TextBlob base scoring
    raw_polarity    = textblob_polarity(text)         # raw_polarity: TextBlob's unmodified polarity score (-1 to +1)
    subjectivity    = textblob_subjectivity(text)     # subjectivity: TextBlob's subjectivity score (0 to 1)

    # Step 3: Negation adjustment
    polarity        = apply_negation(raw_polarity, tokens)   # polarity: raw_polarity adjusted for negation context if present
    negation_found  = detect_negation(tokens)         # negation_found: boolean flag, True if a negation word was detected

    # Step 4: Sarcasm override
    sarcasm_found   = detect_sarcasm(text)            # sarcasm_found: boolean flag, True if a sarcasm pattern matched
    if sarcasm_found:                                  # if sarcasm was detected in this comment
        polarity = -abs(polarity) - 0.2               # force the score strongly negative (take absolute value, negate, then subtract 0.2)
    polarity = max(-1.0, min(1.0, polarity))          # clamp polarity so it always stays within the valid -1.0 to +1.0 range

    # Step 5: Label assignment
    if polarity >= 0.05:                              # if polarity is at or above the positive threshold
        label = "positive"                            # label this comment as positive
    elif polarity <= -0.05:                           # else if polarity is at or below the negative threshold
        label = "negative"                            # label this comment as negative
    else:                                              # otherwise polarity is close enough to zero
        label = "neutral"                             # label this comment as neutral

    # Collect keyword evidence from TextBlob word-level analysis
    blob = TextBlob(text)                              # blob: a fresh TextBlob object used here for POS (part-of-speech) tagging
    pos_words = [w for w, tag in blob.tags if tag.startswith("JJ") and    # w: word; tag: its POS tag; JJ* = adjective tags
                 textblob_polarity(w) > 0.1][:5]       # keep adjectives with polarity > 0.1, limit to first 5 matches
    neg_words = [w for w, tag in blob.tags if tag.startswith("JJ") and    # same logic but for negative adjectives
                 textblob_polarity(w) < -0.1][:5]      # keep adjectives with polarity < -0.1, limit to first 5 matches

    return {                                           # return a dictionary summarising all the scoring results for this comment
        "compound":           round(polarity, 3),     # "compound": final adjusted polarity score, rounded to 3 decimal places
        "raw_textblob":       round(raw_polarity, 3), # "raw_textblob": the original unmodified TextBlob polarity, rounded
        "subjectivity":       round(subjectivity, 3), # "subjectivity": the TextBlob subjectivity score, rounded
        "label":              label,                  # "label": the final positive/negative/neutral classification string
        "pos_words":          pos_words,               # "pos_words": list of positive adjectives found as evidence
        "neg_words":          neg_words,               # "neg_words": list of negative adjectives found as evidence
        "negation_detected":  negation_found,          # "negation_detected": boolean flag for explainability
        "sarcasm_detected":   sarcasm_found,           # "sarcasm_detected": boolean flag for explainability
    }                                                  # end of returned dict


# 11. ASPECT-SPECIFIC SCORING - Scores Each Aspect Separately

# This solves the "mixed sentiment" problem.
# 
# The problem:
# "Professors are great but Wi-Fi is terrible"
# - Overall comment is mixed (neutral)
# - But professors are GREAT, Wi-Fi is TERRIBLE
# - We need separate scores for each!
# 
# Our solution:
# 1. Split comment into sentences
# 2. Keep only sentences mentioning the target aspect
# 3. Score ONLY those sentences
# 
# Result:
# - "professors" → scored from "Professors are great" → POSITIVE
# - "wi-fi" → scored from "Wi-Fi is terrible" → NEGATIVE
# 
# This is what makes our analysis truly useful!


# NLTK sentence split

def score_aspect_sentiment(text: str, aspect: str) -> dict:   # score_aspect_sentiment: scores sentiment for one specific aspect only
    """
    Scores sentiment for a specific aspect.

    Uses NLTK's sentence tokenizer to isolate sentences that mention
    the aspect's keywords, then runs TextBlob + negation/sarcasm on
    just those sentences for a more accurate aspect-level score.
    """
    # NLTK sentence split
    sentences = nltk_sentences(text)                  # sentences: list of individual sentences from the comment
    keywords  = ASPECT_KEYWORDS.get(aspect, [])       # keywords: the keyword list for this aspect (empty list if aspect unknown)

    relevant = [                                       # relevant: list comprehension collecting only sentences that mention this aspect
        s for s in sentences                           # s: a single sentence string from the sentences list
        if any(re.search(r'\b' + re.escape(kw) + r'\b', s.lower()) for kw in keywords)  # keep sentence if ANY keyword matches it
    ]                                                  # end of relevant list comprehension
    target = " ".join(relevant) if relevant else text # target: joined relevant sentences, or fall back to the full comment if none matched

    result          = score_sentiment(target)         # result: run the full sentiment scoring pipeline on just the relevant text
    result["aspect"]         = aspect                  # attach the aspect name to the result dict for reference
    result["relevant_text"]  = target.strip()          # attach the exact text that was scored, trimmed of whitespace, for transparency
    return result                                      # return the enriched result dict


# 12. TIMESTAMP GENERATION - Creates Fake Dates for Trend Chart

# Since our dataset has no real dates, we generate fake ones.
# 
# How it works:
# - Spreads 60 comments across 14 days
# - Random hours between 7am-10pm (realistic posting times)
# - FIXED random seed (42) → same timestamps every run
# 
# Why fixed seed?
# - Makes results REPRODUCIBLE
# - Everyone who runs the code gets the same trend chart
# - Good for grading and demonstrations
# 
# This powers the 14-day trend chart on the dashboard.


# TIMESTAMP GENERATION


def generate_timestamps(n: int, days: int = 14) -> list:   # generate_timestamps: builds n random timestamps within the last `days` days
    """Generate n timestamps spread across the last `days` days."""
    base = datetime.now() - timedelta(days=days)      # base: the earliest possible timestamp (today minus `days` days)
    random.seed(42)                                    # seed the RNG with a fixed value so results are reproducible across runs
    return sorted([                                    # return a sorted (chronological) list of generated timestamps
        base + timedelta(                              # each timestamp = base + a random offset
            days=random.randint(0, days - 1),          # random day offset within the date range
            hours=random.randint(7, 22),               # random hour between 7am and 10pm (realistic comment-posting hours)
            minutes=random.randint(0, 59),             # random minute within the hour
        )
        for _ in range(n)                               # repeat this generation n times (once per comment)
    ])                                                   # end of sorted list


# 13. FULL ANALYSIS PIPELINE - Master Function

# This is the master function that runs everything.
# 
# What it does:
# 1. Generates timestamps for each comment
# 2. For EACH comment:
#    a. Detect which aspects are mentioned
#    b. Score overall sentiment
#    c. Score EACH aspect separately
# 3. Store all results in a "records" list
# 4. Then aggregate (summarize) everything:
#    - Overall counts (positive/negative/neutral %)
#    - Per-aspect statistics
#    - 14-day trend
#    - Generate insights and alerts
# 
# Returns one big dictionary with ALL results.


# FULL ANALYSIS PIPELINE


def analyse_all_comments(comments: list = None) -> dict:   # analyse_all_comments: runs the full pipeline on a given comment list (defaults to RAW_COMMENTS)
    """
    Master pipeline. Processes the given comments (defaults to the built-in
    RAW_COMMENTS dataset if none supplied) and returns a results dict
    containing: per-comment records, overall stats, aspect summaries,
    14-day trend, alerts, and actionable insights.
    """
    comments = comments if comments is not None else RAW_COMMENTS  # comments: the list of comments to analyse; falls back to RAW_COMMENTS
    timestamps = generate_timestamps(len(comments))       # timestamps: one simulated timestamp per comment, same length as comments
    records    = []                                        # records: list that will hold one detailed dict per comment

    for i, (comment, ts) in enumerate(zip(comments, timestamps)):  # i: index; comment: text; ts: matching timestamp
        aspects = detect_aspects(comment)              # aspects: list of aspect names mentioned in this comment
        overall = score_sentiment(comment)             # overall: full sentiment scoring dict for this comment

        aspect_scores = {}                              # aspect_scores: dict that will map aspect name -> its sentiment score dict
        for asp in aspects:                             # asp: a single aspect name from the aspects list
            if asp != "general":                        # skip the fallback "general" aspect (no specific keywords matched)
                aspect_scores[asp] = score_aspect_sentiment(comment, asp)  # compute and store the aspect-specific sentiment

        records.append({                                # append a new dict representing this comment's full analysis
            "id":            i + 1,                     # "id": 1-based sequential ID for this comment
            "comment":       comment,                   # "comment": the original raw comment text
            "timestamp":     ts,                         # "timestamp": the simulated datetime object for this comment
            "aspects":       aspects,                    # "aspects": list of aspect names detected
            "aspect_scores": aspect_scores,              # "aspect_scores": dict of per-aspect sentiment results
            "overall":       overall,                    # "overall": the overall sentiment scoring dict
        })                                               # end of dict appended to records

    
    # 14. OVERALL COUNTS - Summarize All Comments
   
    # Simple math to summarize all comments.
    # 
    # What we calculate:
    # - Total number of comments
    # - How many are Positive
    # - How many are Negative
    # - How many are Neutral
    # - Convert counts to percentages
    # 
    # These numbers power the 4 metric cards at the top
    # of the dashboard (Total, Positive %, Negative %, Neutral %).
   
    total     = len(records)                            # total: total number of comments processed
    pos_count = sum(1 for r in records if r["overall"]["label"] == "positive")  # pos_count: how many comments were labelled positive
    neg_count = sum(1 for r in records if r["overall"]["label"] == "negative")  # neg_count: how many comments were labelled negative
    neu_count = total - pos_count - neg_count            # neu_count: the remainder must be neutral (avoids a third sum() pass)


    # 15. PER-ASPECT AGGREGATION - Summarize Each Aspect
 
    # For each aspect, we calculate:
    # 
    # - positive_pct: % of mentions that were positive
    # - negative_pct: % of mentions that were negative
    # - neutral_pct: % of mentions that were neutral
    # - avg_compound: average polarity score (-1 to +1)
    # - mention_count: how many times this aspect was mentioned
    # 
    # This powers:
    # - The bar chart (shows pos/neg % for each aspect)
    # - The aspect progress bars
    # - The "Fix First", "Keep Doing", "Controversial" insights
    # - The alerts panel
    
    aspect_agg = defaultdict(lambda: {                  # aspect_agg: dict that auto-initialises a fresh counter dict per new aspect key
        "positive": 0, "negative": 0, "neutral": 0,      # counters for how many times this aspect was each sentiment
        "compounds": [], "count": 0,                      # compounds: list of every compound score seen; count: total mentions
    })                                                    # end of defaultdict factory lambda
    for rec in records:                                  # rec: one comment's full record dict
        for asp, sc in rec["aspect_scores"].items():     # asp: aspect name; sc: that aspect's sentiment score dict
            aspect_agg[asp][sc["label"]] += 1            # increment the matching label counter (positive/negative/neutral) for this aspect
            aspect_agg[asp]["compounds"].append(sc["compound"])  # record this aspect's compound score for later averaging
            aspect_agg[asp]["count"]     += 1            # increment the total mention counter for this aspect

    aspect_summary = {}                                   # aspect_summary: final dict that will hold percentage-based stats per aspect
    for asp, agg in aspect_agg.items():                  # asp: aspect name; agg: its raw aggregated counts/compounds
        n = agg["count"] or 1                             # n: total mentions, but use 1 instead of 0 to avoid division by zero
        aspect_summary[asp] = {                           # build the percentage summary dict for this aspect
            "positive_pct":  round(agg["positive"] / n * 100),   # % of this aspect's mentions that were positive
            "negative_pct":  round(agg["negative"] / n * 100),   # % of this aspect's mentions that were negative
            "neutral_pct":   round(agg["neutral"]  / n * 100),   # % of this aspect's mentions that were neutral
            "avg_compound":  round(sum(agg["compounds"]) / len(agg["compounds"]), 3),  # mean compound score for this aspect
            "mention_count": agg["count"],                # total number of times this aspect was mentioned across all comments
        }                                                  # end of this aspect's summary dict


    # 16. 14-DAY TREND - Sentiment Over Time
   
    # We track positive sentiment day-by-day over 14 days.
    # 
    # How it works:
    # 1. Find the oldest date (today - 13 days)
    # 2. For each day in the 14-day window:
    #    a. Count total comments that day
    #    b. Count positive comments that day
    #    c. Calculate % positive
    # 
    # Why this matters:
    # - Shows if sentiment is improving or declining
    # - If positive % drops 20+ points in one day → ALERT!
    # - Helps spot emerging problems early
    
    base_day = (datetime.now() - timedelta(days=13)).date()  # base_day: the earliest date in our 14-day trend window
    daily    = defaultdict(lambda: {"pos": 0, "total": 0})    # daily: dict auto-initialising {pos, total} counters per day offset
    for rec in records:                                   # rec: one comment's full record dict
        offset = (rec["timestamp"].date() - base_day).days   # offset: how many days after base_day this comment's timestamp falls
        if 0 <= offset <= 13:                              # only count the comment if it falls within our 14-day window (0 to 13)
            daily[offset]["total"] += 1                    # increment the total comment counter for that day
            if rec["overall"]["label"] == "positive":      # if this particular comment was positive
                daily[offset]["pos"] += 1                  # increment the positive counter for that day too

    trend = []                                              # trend: list that will hold one dict per day of the 14-day window
    for d in range(14):                                    # d: day offset from 0 (oldest) to 13 (most recent)
        bucket = daily[d]                                   # bucket: the {pos, total} counts for this specific day
        pct    = round(bucket["pos"] / bucket["total"] * 100) if bucket["total"] else 50  # % positive, or 50 default if no data that day
        trend.append({                                      # append this day's trend data point
            "day":          d,                              # "day": the day offset index (0-13)
            "date":         str(base_day + timedelta(days=d)),  # "date": the actual calendar date as a string
            "positive_pct": pct,                            # "positive_pct": percentage of comments that day which were positive
        })                                                   # end of this day's trend dict

    alerts   = _detect_alerts(aspect_summary, trend)        # alerts: list of alert dicts generated from the aspect summary + trend
    insights = _generate_insights(aspect_summary)           # insights: dict of fix_first / keep_doing / controversial lists

    return {                                                 # return the full results dict for the entire analysis run
        "records":        records,                          # "records": list of per-comment detailed analysis dicts
        "total":          total,                            # "total": total number of comments analysed
        "overall": {                                         # "overall": aggregate sentiment percentages across all comments
            "positive_pct": round(pos_count / total * 100), # overall % positive
            "negative_pct": round(neg_count / total * 100), # overall % negative
            "neutral_pct":  round(neu_count / total * 100), # overall % neutral
        },                                                    # end of "overall" sub-dict
        "aspect_summary": aspect_summary,                    # "aspect_summary": per-aspect percentage breakdowns
        "trend":          trend,                             # "trend": 14-day daily positive-sentiment trend data
        "alerts":         alerts,                             # "alerts": list of triggered alert dicts
        "insights":       insights,                           # "insights": fix_first/keep_doing/controversial recommendations
    }                                                          # end of returned results dict


# ALERT GENERATION - Triggers Based on Thresholds

# Alerts fire when negative sentiment crosses thresholds:
# - HIGH severity: 60% or more negative mentions
# - MEDIUM severity: 40% to 59% negative mentions
# 
# Also checks for sudden drops in the 14-day trend:
# - If positive % drops 20+ points in one day → HIGH alert


def _detect_alerts(aspect_summary: dict, trend: list) -> list:   # _detect_alerts: builds alert dicts based on thresholds
    alerts = []                                              # alerts: list that will collect every triggered alert dict
    for asp, stats in aspect_summary.items():                # asp: aspect name; stats: its percentage summary dict
        neg = stats["negative_pct"]                          # neg: this aspect's negative percentage, for threshold comparisons
        if neg >= 60:                                         # if 60% or more of this aspect's mentions are negative
            alerts.append({"aspect": asp, "severity": "HIGH", # append a HIGH severity alert
                           "message": f"{neg}% of {asp} mentions are negative — immediate action needed.",  # human-readable message
                           "negative_pct": neg})              # include the raw negative percentage for reference
        elif neg >= 40:                                       # else if 40-59% of mentions are negative
            alerts.append({"aspect": asp, "severity": "MEDIUM",  # append a MEDIUM severity alert
                           "message": f"{neg}% of {asp} mentions are negative — monitor closely.",  # human-readable message
                           "negative_pct": neg})              # include the raw negative percentage for reference
    for i in range(1, len(trend)):                           # i: index into the trend list, starting from the second day (index 1)
        drop = trend[i - 1]["positive_pct"] - trend[i]["positive_pct"]  # drop: how much positive % fell from yesterday to today
        if drop >= 20:                                        # if positive sentiment dropped by 20 points or more in one day
            alerts.append({"aspect": "overall", "severity": "HIGH",  # append a HIGH severity overall trend alert
                           "message": f"Positive sentiment dropped {drop}% on {trend[i]['date']}.",  # human-readable message
                           "negative_pct": 100 - trend[i]["positive_pct"]})  # the implied negative percentage on that day
    return alerts                                             # return the complete list of alerts


# INSIGHT GENERATION - Actionable Recommendations

# Three categories of actionable insights:
# 
# 1. Fix First: Aspects with 40%+ negative sentiment
#    - These need urgent attention
# 
# 2. Keep Doing: Aspects with 60%+ positive sentiment
#    - These are strengths to maintain
# 
# 3. Controversial: Aspects near 50/50 split
#    - Need qualitative investigation


def _generate_insights(aspect_summary: dict) -> dict:        # _generate_insights: derives actionable recommendations from aspect stats
    ranked = sorted(aspect_summary.items(), key=lambda x: x[1]["avg_compound"])  # ranked: aspects sorted worst (most negative) to best
    return {                                                   # return a dict with three categories of insight
        "fix_first": [                                         # "fix_first": aspects that most need improvement
            {"aspect": a, "negative_pct": s["negative_pct"],    # a: aspect name; s: its stats dict
             "reason": f"{s['negative_pct']}% negative, avg score {s['avg_compound']}"}  # human-readable justification
            for a, s in ranked if s["negative_pct"] >= 40        # only include aspects with 40%+ negative mentions
        ][:3],                                                   # limit to the top 3 worst aspects
        "keep_doing": [                                          # "keep_doing": aspects performing well, worth maintaining
            {"aspect": a, "positive_pct": s["positive_pct"],     # a: aspect name; s: its stats dict
             "reason": f"{s['positive_pct']}% positive, avg score {s['avg_compound']}"}  # human-readable justification
            for a, s in ranked if s["positive_pct"] >= 60         # only include aspects with 60%+ positive mentions
        ][:3],                                                    # limit to the top 3 best aspects
        "controversial": [                                        # "controversial": aspects with a roughly even pos/neg split
            {"aspect": a, "positive_pct": s["positive_pct"], "negative_pct": s["negative_pct"],  # include both percentages
             "reason": "Near 50/50 split — investigate further"}  # generic justification for controversial aspects
            for a, s in ranked if 35 <= s["positive_pct"] <= 65 and 35 <= s["negative_pct"] <= 65  # both percentages near the middle
        ][:3],                                                     # limit to the top 3 most controversial aspects
    }                                                              # end of returned insights dict