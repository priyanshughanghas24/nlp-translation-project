"""Inference wrapper for the custom Seq2Seq model (used by train.py and app.py)."""
import json, pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from prepare_data import normalize

class Translator:
    def __init__(self, folder="model"):
        m = tf.keras.models.load_model(f"{folder}/seq2seq.keras")
        self.cfg = json.load(open(f"{folder}/config.json"))
        self.src_tok = pickle.load(open(f"{folder}/src_tok.pkl", "rb"))
        self.tgt_tok = pickle.load(open(f"{folder}/tgt_tok.pkl", "rb"))
        self.idx2w = {i: w for w, i in self.tgt_tok.word_index.items()}
        u = self.cfg["units"]
        self.enc = Model(m.input[0], m.get_layer("enc_lstm").output[1:])
        d_in, sh, sc = layers.Input((1,)), layers.Input((u,)), layers.Input((u,))
        x = m.get_layer("dec_emb")(d_in)
        o, h, c = m.get_layer("dec_lstm")(x, initial_state=[sh, sc])
        self.dec = Model([d_in, sh, sc], [m.get_layer("dec_dense")(o), h, c])

    def translate(self, text: str) -> str:
        seq = self.src_tok.texts_to_sequences([normalize(text)])
        seq = pad_sequences(seq, maxlen=self.cfg["max_src"], padding="post")
        h, c = self.enc.predict(seq, verbose=0)
        tok = np.array([[self.tgt_tok.word_index["<start>"]]]); out = []
        for _ in range(self.cfg["max_tgt"]):
            p, h, c = self.dec.predict([tok, h, c], verbose=0)
            i = int(np.argmax(p[0, -1]))
            w = self.idx2w.get(i, "<unk>")
            if w == "<end>": break
            out.append(w); tok = np.array([[i]])
        return " ".join(out).replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",")
