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
    return sacrebleu.corpus_bleu(hyps, streams, tokenize="none", lowercase=False).score
