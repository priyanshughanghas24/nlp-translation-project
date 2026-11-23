"""Stages 2-5: dataset download, cleaning, preprocessing, EDA.
Run: python prepare_data.py"""
import io, os, re, zipfile, unicodedata, urllib.request
from collections import Counter
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

URL = "https://www.manythings.org/anki/fra-eng.zip"
RAW = os.path.join("data", "fra.txt")
MAX_WORDS, SAMPLE = 12, 40000


def download():
    """Download fra.txt once. A browser User-Agent avoids the HTTP 406 error."""
    if os.path.exists(RAW):
        return
    print("Downloading dataset...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        data = urllib.request.urlopen(req, timeout=60).read()
    except Exception as e:
        raise SystemExit(
            f"Download failed: {e}\nDownload fra-eng.zip in your browser, extract "
            f"fra.txt into the 'data' folder, then run this script again.")
    zipfile.ZipFile(io.BytesIO(data)).extract("fra.txt", "data")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower().strip()
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"([?.!,])", r" \1 ", s)
    s = re.sub(r"[^a-zàâäçéèêëîïôöùûüÿœæ?.!,' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    os.makedirs("data", exist_ok=True); os.makedirs("plots", exist_ok=True)
    download()
    df = pd.read_csv(RAW, sep="\t", header=None, names=["eng", "fra", "meta"],
                     usecols=["eng", "fra"], quoting=3, encoding="utf-8")
    print("Raw pairs:", len(df))
    print(df.head(), "\nMissing:\n", df.isna().sum())

    # ---- Stage 3: cleaning
    df = df.dropna().drop_duplicates()
    df["eng"], df["fra"] = df.eng.map(normalize), df.fra.map(normalize)
    df = df[(df.eng != "") & (df.fra != "")]
    df = df.drop_duplicates(subset="eng").drop_duplicates(subset="fra")

    # ---- Stage 4: preprocessing (stop-words kept on purpose)
    df["n_en"], df["n_fr"] = df.eng.str.split().str.len(), df.fra.str.split().str.len()
    df = df[(df.n_en <= MAX_WORDS) & (df.n_fr <= MAX_WORDS)]
    df = df.sample(min(SAMPLE, len(df)), random_state=42)
    df[["eng", "fra"]].to_csv("data/clean.csv", index=False, encoding="utf-8")
    print("Clean pairs:", len(df))

    # ---- Stage 5: EDA
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(df.n_en, bins=range(1, MAX_WORDS + 2), color="#4c72b0"); ax[0].set_title("English length")
    ax[1].hist(df.n_fr, bins=range(1, MAX_WORDS + 2), color="#dd8452"); ax[1].set_title("French length")
    plt.tight_layout(); plt.savefig("plots/length_dist.png", dpi=120); plt.close()
    for col, name in [("eng", "English"), ("fra", "French")]:
        c = Counter(w for t in df[col] for w in t.split() if w not in {"?", ".", "!", ",", "'"})
        print(f"{name} vocab: {len(c)}")
        w, n = zip(*c.most_common(15))
        plt.figure(figsize=(7, 3.5)); plt.bar(w, n); plt.xticks(rotation=60)
        plt.title(f"Top words - {name}"); plt.tight_layout()
        plt.savefig(f"plots/top_{name.lower()}.png", dpi=120); plt.close()
    print("Done. clean.csv and plots/ saved.")


main()