"""Helpers shared by the notebooks: data, the split, vocabularies, the encoder-decoder model, translation and the metric.

Import this module before TensorFlow or Keras (it sets two environment variables that only work if TensorFlow is not loaded yet).
"""

import hashlib
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")     # keep TensorFlow start-up messages quiet
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")    # avoid tiny run-to-run rounding differences

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

SEED = 42
DATA = Path(__file__).resolve().parents[1] / "legacy" / "spa.txt"
ASSETS = Path(__file__).resolve().parents[1] / "assets"
MAX_LEN = 10                       # sentences with more than 10 tokens (words and punctuation marks) are left out
N_TRAIN, N_VALIDATION, N_TEST = 30000, 2000, 2000     # CPU-sized subsets, fixed by the hash order (see ``make_split``)
PAD, UNK, START, END = "<pad>", "<unk>", "<start>", "<end>"


def set_seed(seed=SEED):
    """Fix every source of randomness we use, so reruns give the same numbers on a CPU."""
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def save_figure(fig, name):
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / name, dpi=110, bbox_inches="tight")


# ---------------------------------------------------------------- data, tokens and the split
def load_pairs(path=DATA):
    """All (English, Spanish) pairs of the file, in file order."""
    pairs = []
    with open(path, encoding="utf-8") as handle:
        for line in handle.read().split("\n"):
            if line:
                english, spanish, _attribution = line.split("\t")
                pairs.append((english, spanish))
    return pairs


def tokenize(text):
    """Lower case; words (accents kept) and punctuation marks are separate tokens."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def group_of(english):
    """A number from 0 to 99 that depends only on the English sentence (and not on any random generator)."""
    return int(hashlib.md5(english.lower().encode("utf-8")).hexdigest(), 16) % 100


def make_split(pairs):
    """Split by English sentence: every pair of one English sentence lands in the same part.

    Group numbers 0-79 are train, 80-89 validation and 90-99 test. Only pairs with at most MAX_LEN tokens on each side are kept, and each part is
    cut to its CPU-sized subset by the md5 order of the English sentence (so the choice is fixed, never random). Returns three dicts
    {english: [spanish translations]} for train, validation, test.
    """
    kept = [(e, s) for e, s in pairs if len(tokenize(e)) <= MAX_LEN and len(tokenize(s)) <= MAX_LEN]
    parts = {"train": {}, "validation": {}, "test": {}}
    for english, spanish in kept:
        g = group_of(english)
        part = "train" if g < 80 else ("validation" if g < 90 else "test")
        parts[part].setdefault(english, []).append(spanish)
    def order(e):
        return hashlib.md5(("order" + e).encode("utf-8")).hexdigest()
    limits = {"train": N_TRAIN, "validation": N_VALIDATION, "test": N_TEST}
    result = {}
    for part, groups in parts.items():
        chosen = sorted(groups, key=order)
        if part == "train":                      # for training we count pairs, so cut after N_TRAIN pairs
            taken, count = [], 0
            for e in chosen:
                taken.append(e); count += len(groups[e])
                if count >= limits[part]:
                    break
            chosen = taken
        else:
            chosen = chosen[:limits[part]]
        result[part] = {e: groups[e] for e in chosen}
    return result["train"], result["validation"], result["test"]


def flatten(part):
    """{english: [spanish, ...]} -> list of (english, spanish) pairs."""
    return [(e, s) for e, spanish_list in part.items() for s in spanish_list]


class Vocabulary:
    """Maps tokens to numbers and back. Built from the training sentences only; rare words become <unk>."""

    def __init__(self, token_lists, max_size):
        counts = {}
        for tokens in token_lists:
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
        words = sorted(counts, key=lambda w: (-counts[w], w))[:max_size - 4]
        self.itos = [PAD, UNK, START, END] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, 1) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids if i > 3]          # drops <pad>, <unk>, <start>, <end>


def encode_pairs(pairs, source_vocab, target_vocab):
    """Arrays for training: the English sentence; the Spanish sentence shifted (input of the decoder); the Spanish sentence (what it must predict)."""
    n = len(pairs)
    source = np.zeros((n, MAX_LEN), dtype="int32")
    decoder_in = np.zeros((n, MAX_LEN + 1), dtype="int32")
    decoder_out = np.zeros((n, MAX_LEN + 1), dtype="int32")
    for i, (english, spanish) in enumerate(pairs):
        e = source_vocab.encode(tokenize(english))
        s = target_vocab.encode(tokenize(spanish))
        source[i, :len(e)] = e
        decoder_in[i, :len(s) + 1] = [2] + s                 # <start> first word ...
        decoder_out[i, :len(s) + 1] = s + [3]                # ... last word <end>
    return source, decoder_in, decoder_out


# ---------------------------------------------------------------- the model
def build_seq2seq(source_size, target_size, embedding=128, units=256):
    """An encoder-decoder with LSTMs. Returns (training model, encoder model, decoder step model); the three share their layers."""
    source = keras.Input((MAX_LEN,), dtype="int32", name="english")
    embed_source = keras.layers.Embedding(source_size, embedding, mask_zero=True, name="embed_english")
    encoder_lstm = keras.layers.LSTM(units, return_state=True, name="encoder")
    _, state_h, state_c = encoder_lstm(embed_source(source))

    decoder_input = keras.Input((None,), dtype="int32", name="spanish_so_far")
    embed_target = keras.layers.Embedding(target_size, embedding, mask_zero=True, name="embed_spanish")
    decoder_lstm = keras.layers.LSTM(units, return_sequences=True, return_state=True, name="decoder")
    dropout = keras.layers.Dropout(0.3)
    output_layer = keras.layers.Dense(target_size, activation="softmax", name="next_word")
    decoded, _, _ = decoder_lstm(embed_target(decoder_input), initial_state=[state_h, state_c])
    probabilities = output_layer(dropout(decoded))
    training = keras.Model([source, decoder_input], probabilities, name="seq2seq")

    encoder = keras.Model(source, [state_h, state_c], name="encoder_model")
    step_input = keras.Input((1,), dtype="int32")
    h_in, c_in = keras.Input((units,)), keras.Input((units,))
    step_out, h_out, c_out = decoder_lstm(embed_target(step_input), initial_state=[h_in, c_in])
    decoder = keras.Model([step_input, h_in, c_in], [output_layer(step_out), h_out, c_out], name="decoder_model")
    return training, encoder, decoder


def translate(encoder, decoder, source_array, target_vocab):
    """Greedy translation: at every step the decoder takes the most probable next word; stops at <end> or after MAX_LEN + 1 steps."""
    h, c = encoder.predict(source_array, verbose=0)
    token = np.full((len(source_array), 1), 2, dtype="int32")          # <start>
    finished = np.zeros(len(source_array), dtype=bool)
    outputs = [[] for _ in range(len(source_array))]
    for _ in range(MAX_LEN + 1):
        probabilities, h, c = decoder.predict([token, h, c], verbose=0)
        token = probabilities[:, 0, :].argmax(axis=1).astype("int32")[:, None]
        for i, t in enumerate(token[:, 0]):
            if not finished[i]:
                if t == 3:
                    finished[i] = True
                else:
                    outputs[i].append(int(t))
        if finished.all():
            break
    return [target_vocab.decode(o) for o in outputs]


# ---------------------------------------------------------------- the metric
def corpus_bleu(hypotheses, references):
    """BLEU (sacrebleu) on lower-cased, tokenised text. ``references`` is a list of lists (several valid translations per sentence).

    Scores are NOT comparable with BLEU reported on detokenised text (our texts are split into words and punctuation marks, and sentences are short).
    """
    import sacrebleu
    hyps = [" ".join(h) for h in hypotheses]
    width = max(len(r) for r in references)
    streams = [[" ".join(tokenize(r[min(k, len(r) - 1)])) for r in references] for k in range(width)]
    return sacrebleu.corpus_bleu(hyps, streams, tokenize="none", lowercase=False, force=True).score    # force: our text is tokenised on purpose


# ---------------------------------------------------------------- figures
def plot_seq2seq():
    """The encoder-decoder with teacher forcing, drawn with boxes and arrows (training view)."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    english = ["how", "are", "you", "?"]
    spanish_in = ["<start>", "¿", "cómo", "estás", "?"]
    spanish_out = ["¿", "cómo", "estás", "?", "<end>"]
    for i, word in enumerate(english):
        ax.text(i * 1.3, 0, word, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round", fc="#f2f2f2", ec="#555555"))
        ax.text(i * 1.3, 1.1, "LSTM", ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round", fc="#dbe9f7", ec="#555555"))
        ax.annotate("", xy=(i * 1.3, 0.85), xytext=(i * 1.3, 0.25), arrowprops=dict(arrowstyle="->", color="#555555"))
        if i:
            ax.annotate("", xy=(i * 1.3 - 0.45, 1.1), xytext=((i - 1) * 1.3 + 0.45, 1.1), arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.text(1.95, 1.9, "ENCODER reads the English sentence", ha="center", fontsize=9, color="#1f4e79")
    start = 6.2
    for j, (inp, out) in enumerate(zip(spanish_in, spanish_out)):
        x = start + j * 1.3
        ax.text(x, 0, inp, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round", fc="#f2f2f2", ec="#555555"))
        ax.text(x, 1.1, "LSTM", ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round", fc="#fde7d9", ec="#555555"))
        ax.text(x, 2.2, out, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round", fc="#e8f5e9", ec="#555555"))
        ax.annotate("", xy=(x, 0.85), xytext=(x, 0.25), arrowprops=dict(arrowstyle="->", color="#555555"))
        ax.annotate("", xy=(x, 1.95), xytext=(x, 1.35), arrowprops=dict(arrowstyle="->", color="#555555"))
        if j:
            ax.annotate("", xy=(x - 0.45, 1.1), xytext=(x - 1.3 + 0.45, 1.1), arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.annotate("", xy=(start - 0.45, 1.1), xytext=(3 * 1.3 + 0.45, 1.1),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
    ax.text(5.0, 0.6, "the sentence summary\n(the LSTM state)", ha="center", fontsize=8, color="#c0392b")
    ax.text(start + 2.6, -0.8, "inputs of the DECODER: the correct Spanish words so far (teacher forcing)", ha="center", fontsize=8, color="#444444")
    ax.text(start + 2.6, 2.85, "what the decoder must predict: the next word at every step", ha="center", fontsize=9, color="#2e7d32")
    ax.set_xlim(-0.8, 13); ax.set_ylim(-1.2, 3.2)
    return fig


def plot_training(history):
    """Loss and accuracy per epoch for training and validation."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    epochs = range(1, len(history["loss"]) + 1)
    a.plot(epochs, history["loss"], "--o", ms=3, label="training"); a.plot(epochs, history["val_loss"], "-o", ms=3, label="validation")
    a.set_title("loss (lower is better)"); b.plot(epochs, history["accuracy"], "--o", ms=3, label="training")
    b.plot(epochs, history["val_accuracy"], "-o", ms=3, label="validation"); b.set_title("accuracy of the next word (higher is better)")
    for ax in (a, b):
        ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend()
    return fig


def plot_bars(labels, values, title):
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.bar(labels, values, color="tab:blue")
    for k, v in enumerate(values):
        ax.text(k, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=10); ax.set_ylabel("BLEU")
    return fig
