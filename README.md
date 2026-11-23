# Project 1: Text-to-Text Translation (English → French)

## Run order
```
pip install -r requirements.txt
python prepare_data.py     # download, clean, preprocess, EDA  -> data/clean.csv, plots/
python train.py            # features, model, training, evaluation -> model/
streamlit run app.py       # GUI
```

## Stage map
1. Problem: automatically translate text between languages (chat, travel, support, accessibility).
2. Dataset: Tatoeba/Anki fra-eng pairs (manythings.org/anki), ~230k pairs, 40k sampled.
3. Cleaning: drop NaN, duplicates, empty rows, unusual symbols.
4. Preprocessing: Unicode NFC, lowercase, punctuation spacing, whitespace tokenisation, length filter (≤12 words).
   Stop-words are kept; removing them destroys meaning in translation. Stemming/lemmatisation also not used.
5. EDA: length histograms, top words, vocab size (plots/).
6. Features: integer-id sequences (Keras Tokenizer, 8k vocab) + learned 256-d embeddings; `<start>/<end>` tokens.
7. Model: Encoder LSTM → context state → Decoder LSTM (teacher forcing), greedy decoding.
8. Metrics: BLEU, token accuracy, unigram precision/recall/F1 (translation has no single "correct" answer, so BLEU is primary).
9. GUI: Streamlit, switch between custom model and pretrained MarianMT.
10. Testing checklist: empty input; one word ("hello"); question ("where is the station?");
    unknown word; multi-line input; model files missing (error message shown).

## Improvements
Attention layer, bidirectional encoder, beam search, Transformer, subword tokens (BPE), more data.
