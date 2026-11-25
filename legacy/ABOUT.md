# Legacy files

These are my original files from the first version of this project (January 2023, README updated February 2023), kept unchanged:
`Translation spanish english.py` (the script), `spa.txt` (the data, 20 MB), the first `README.md` and the first `.gitignore`.
(The repository's default branch is `master`.)

What I notice today, from reading the script and checking the data:

- **The Spanish text is damaged by the standardisation.** The list of characters to strip is `string.punctuation + "¿" + "¿" + "á" + "é" + "í" + "ó" + "ú" + "ñ" + "ü"`: the accented
  letters are *deleted*, not kept. I ran the same expression on a sample: `¿Dónde está la estación?¡Mañana vamos al pingüino!` becomes `dnde est la estacin¡maana vamos al pingino`
  (and `¡` is not stripped because `¿` was typed twice).
- **The vocabularies were built from the test set.** The function `tokenization(train_pairs)` ignores its argument and adapts both vectorisers on `test_pairs`.
  So the "training" vocabulary comes from the 15% test split and not from the training split, and the test split is used as if it were training data.
- **The random split leaks.** The 139,013 pairs contain only 118,119 different English sentences: 15,406 English sentences appear with two or more Spanish translations
  (one, "You can put it there.", with 68). A random split of *pairs*, without a seed, puts the same English sentence in the training and the test split.
- **The script cannot be run on another computer** (a hard-coded Windows path) and has no random seed.
- **No translation function and no evaluation.** The script ends with `fit(...)`. The README describes a section "Evaluate and Translate" that loads a trained model and prints translations; it does not exist in the code.
  Training metrics were never saved, and no translation quality measure (such as BLEU) was ever computed.
- **The model is large for a CPU:** embeddings of 256, a bidirectional GRU and a GRU decoder with 1,024 units, a 15,000-word softmax, 15 epochs. (It is a GRU, not an LSTM.)
- The README mentions a `LICENSE` file that does not exist and a command (`python translator.py`) for a file with another name; the data has a licence that is stated in every line of the file (see `../data/README.md`).

The new notebooks in `../notebooks/` rebuild the project as a path: an encoder-decoder LSTM explained step by step (part 1), then attention and a comparison with a pretrained model (next parts), always on the same split and with a real metric.
