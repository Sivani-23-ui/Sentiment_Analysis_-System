"""Creates a small SYNTHETIC dataset so you can test the whole pipeline before downloading a real one.
For your final results use a real dataset (see README): python train_*.py --data data/Tweets.csv"""
import random
import pandas as pd
import config

random.seed(config.RANDOM_STATE)
subjects = ["the phone", "this laptop", "the delivery", "customer support", "the new update", "the app", "this brand",
            "the service", "the camera", "the battery", "the launch event", "the website"]
pos = ["is amazing", "works perfectly", "is really great", "exceeded my expectations", "is fantastic and fast",
       "makes me so happy", "is the best I have used", "looks beautiful", "is worth every penny", "is excellent"]
neg = ["is terrible", "stopped working", "is really disappointing", "is a complete waste of money", "keeps crashing",
       "is awful and slow", "makes me so angry", "is not good at all", "arrived broken", "is the worst I have used"]
neu = ["was announced today", "is available in stores", "will be released next week", "has a new version",
       "is mentioned in the news", "comes in three colors", "was discussed at the meeting", "is priced at the usual rate",
       "has an update scheduled", "is listed on the website"]
extras = ["", " #tech", " honestly", " lol", " https://example.com", " @someone", "!!", "..."]

rows = []
for label, phrases in (("positive", pos), ("negative", neg), ("neutral", neu)):
    for _ in range(1000):
        text = f"{random.choice(subjects)} {random.choice(phrases)}{random.choice(extras)}"
        if random.random() < 0.08:  # a little label noise so the models don't all score 100%
            label_used = random.choice(["positive", "negative", "neutral"])
        else:
            label_used = label
        rows.append((text, label_used))
random.shuffle(rows)
pd.DataFrame(rows, columns=["text", "sentiment"]).to_csv(config.DEFAULT_DATA, index=False)
print(f"Wrote {len(rows)} synthetic rows -> {config.DEFAULT_DATA}")
