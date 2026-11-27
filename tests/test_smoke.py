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


def test_attention_model_weights_are_valid_and_ignore_padding():
    pairs = [("I run.", "Corro."), ("You run fast.", "Corres rápido."), ("We run.", "Corremos.")] * 8
    source_vocab = tr.Vocabulary([tr.tokenize(e) for e, _ in pairs], 30)
    target_vocab = tr.Vocabulary([tr.tokenize(s) for _, s in pairs], 30)
    source, decoder_in, decoder_out = tr.encode_pairs(pairs, source_vocab, target_vocab)
    tr.set_seed()
    model, encoder, decoder = tr.build_attention_seq2seq(len(source_vocab), len(target_vocab), embedding=8, units=16)
    model.compile("adam", "sparse_categorical_crossentropy")
    history = model.fit([source, decoder_in], decoder_out, epochs=30, batch_size=8, verbose=0).history
    assert history["loss"][-1] < history["loss"][0]
    translations, maps = tr.translate_with_attention(encoder, decoder, source[:4], target_vocab)
    assert len(translations) == 4
    for i, weights in enumerate(maps):
        real = source[i] != 0
        assert weights.shape[1] == tr.MAX_LEN and np.allclose(weights.sum(axis=1), 1.0, atol=1e-4)    # weights add up to 1
        assert weights[:, ~real].max() < 1e-6                                                          # nothing is attended in the padding


def test_bootstrap_of_identical_systems_is_centred_on_zero():
    refs = [["hola mundo"], ["¿cómo estás?"], ["buenos días"], ["adiós"]] * 10
    hyps = [tr.tokenize("hola mundo"), tr.tokenize("cómo estás"), tr.tokenize("buenos días"), tr.tokenize("adiós")] * 10
    differences = tr.bootstrap_bleu_difference(hyps, hyps, refs, resamples=5)
    assert np.allclose(differences, 0.0)


def test_glove_matrix_fills_known_words_and_the_model_uses_it(tmp_path):
    import gzip
    path = tmp_path / "tiny.gz"
    vectors = {"run": [1.0] * 50, "walk": [2.0] * 50}
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(f"2 50\n" + "".join(f"{w} " + " ".join(str(x) for x in v) + "\n" for w, v in vectors.items()))
    pairs = [("I run.", "Corro."), ("I walk.", "Camino.")] * 6
    source_vocab = tr.Vocabulary([tr.tokenize(e) for e, _ in pairs], 20)
    target_vocab = tr.Vocabulary([tr.tokenize(s) for _, s in pairs], 20)
    matrix, found = tr.glove_source_matrix(path, source_vocab, embedding=64)
    assert tr.glove_source_matrix(path, source_vocab, embedding=64, return_missing=True)[2] == [w for w in source_vocab.itos[4:] if w not in vectors]
    assert found == 2 and matrix.shape == (len(source_vocab), 64)
    assert np.allclose(matrix[source_vocab.stoi["run"], :50], 1.0) and np.allclose(matrix[source_vocab.stoi["walk"], :50], 2.0)
    assert np.allclose(matrix[0], 0.0)                                                       # padding row stays zero
    model, encoder, decoder = tr.build_attention_seq2seq(len(source_vocab), len(target_vocab), embedding=64, units=16, source_embeddings=matrix)
    table = model.get_layer("embed_english").get_weights()[0]
    assert np.allclose(table[source_vocab.stoi["run"], :50], 1.0) and model.get_layer("embed_english").trainable
