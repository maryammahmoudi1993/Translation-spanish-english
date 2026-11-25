"""Quick checks of the helpers; the models are only tried for a few steps."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import translation as tr  # noqa: E402


def test_data_and_tokens_keep_the_accents():
    pairs = tr.load_pairs()
    assert len(pairs) == 139013 and pairs[0] == ("Go.", "Ve.")
    assert tr.tokenize("¿Dónde está la estación?") == ["¿", "dónde", "está", "la", "estación", "?"]
    assert tr.tokenize("Mañana vamos al pingüino!") == ["mañana", "vamos", "al", "pingüino", "!"]


def test_split_is_grouped_by_english_sentence_and_fixed():
    pairs = tr.load_pairs()
    train, validation, test = tr.make_split(pairs)
    assert not (set(train) & set(test)) and not (set(train) & set(validation)) and not (set(validation) & set(test))
    assert len(test) == tr.N_TEST and len(validation) == tr.N_VALIDATION and len(tr.flatten(train)) >= tr.N_TRAIN
    assert all(len(tr.tokenize(e)) <= tr.MAX_LEN for e in test)
    again = tr.make_split(pairs)
    assert list(again[2]) == list(test)                                # no randomness: the same test sentences every time


def test_vocabulary_and_encoding_shift_the_target():
    pairs = [("I run.", "Corro ahora."), ("You run.", "Corres ahora.")]
    source_vocab = tr.Vocabulary([tr.tokenize(e) for e, _ in pairs], 20)
    target_vocab = tr.Vocabulary([tr.tokenize(s) for _, s in pairs], 20)
    source, decoder_in, decoder_out = tr.encode_pairs(pairs, source_vocab, target_vocab)
    assert source.shape == (2, tr.MAX_LEN) and decoder_in.shape == (2, tr.MAX_LEN + 1)
    assert decoder_in[0, 0] == 2 and decoder_out[0, 3] == 3                     # <start> first; <end> after "corro ahora ."
    assert list(decoder_in[0, 1:4]) == list(decoder_out[0, 0:3])               # the input is the output shifted by one
    assert target_vocab.decode(decoder_out[0]) == ["corro", "ahora", "."]


def test_model_trains_a_few_steps_and_translates():
    pairs = [("I run.", "Corro."), ("You run.", "Corres."), ("We run.", "Corremos.")] * 8
    source_vocab = tr.Vocabulary([tr.tokenize(e) for e, _ in pairs], 20)
    target_vocab = tr.Vocabulary([tr.tokenize(s) for _, s in pairs], 20)
    source, decoder_in, decoder_out = tr.encode_pairs(pairs, source_vocab, target_vocab)
    tr.set_seed()
    model, encoder, decoder = tr.build_seq2seq(len(source_vocab), len(target_vocab), embedding=8, units=16)
    model.compile("adam", "sparse_categorical_crossentropy")
    history = model.fit([source, decoder_in], decoder_out, epochs=30, batch_size=8, verbose=0).history
    assert history["loss"][-1] < history["loss"][0]
    out = tr.translate(encoder, decoder, source[:3], target_vocab)
    assert len(out) == 3 and all(isinstance(o, list) for o in out)


def test_bleu_is_100_for_a_perfect_copy_and_low_for_nonsense():
    refs = [["Hola, mundo."], ["¿Cómo estás?", "¿Qué tal?"]]
    perfect = [tr.tokenize("Hola, mundo."), tr.tokenize("¿Qué tal?")]
    assert tr.corpus_bleu(perfect, refs) > 99.9
    assert tr.corpus_bleu([["xx"], ["yy"]], refs) < 1.0
