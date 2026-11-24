# Data

This project uses the English-Spanish sentence pairs that I downloaded in 2023: the file [`../legacy/spa.txt`](../legacy/spa.txt)
(20 MB, 139,013 lines; it was already in this repository, so it is used from there and not copied again).
Each line has three tab-separated fields: the English sentence, the Spanish translation and an attribution text, for example

```
Go.	Ve.	CC-BY 2.0 (France) Attribution: tatoeba.org #2877272 (CM) & #4986655 (cueyayotl)
```

- **Origin and licence:** the sentences come from the Tatoeba project (<https://tatoeba.org>); every line of the file states its own licence, **CC-BY 2.0 (France)**, and names the sentence numbers and
  contributors (attribution). Please keep this attribution if you reuse the file. I could not open the Tatoeba website from the environment in which this was written, so the licence is taken from the
  text inside the file and not re-checked at the source.
- **What is in it:** 139,013 pairs, but only 118,119 different English sentences: 15,406 English sentences have two or more Spanish translations (one has 68).
  Our train / validation / test split therefore groups pairs by their English sentence (see notebook 01).
- **Personal data:** the file contains short everyday sentences; names of the contributors appear only as Tatoeba user names in the attribution field.

Nothing is downloaded and no other data is used.
