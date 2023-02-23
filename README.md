# English-Spanish Language Translator using Encoder-Decoder Architecture
# Maryam Mahmoudi


This code is a deep learning model for translating English sentences to Spanish sentences using an Encoder-Decoder architecture. The code loads an English-Spanish language dataset, preprocesses the text, and trains an Encoder-Decoder model using TensorFlow and Keras.

Installation
Install Python (version 3.7 or higher) from the official website
Install TensorFlow (version 2.5 or higher) using pip: pip install tensorflow
Install Keras (version 2.5 or higher) using pip: pip install keras
Usage
Clone the repository: git clone https://github.com/your_username/translator.git
Navigate to the directory: cd translator
Run the script: python translator.py
Explanation of code
The code is divided into three main sections:

Data and Preprocessing
This section loads the English-Spanish dataset from: https://www.kaggle.com/datasets/tejasurya/eng-spanish, cleans the text, and splits it into training, validation, and test sets. The text is cleaned by removing punctuation and converting all text to lowercase. The training data is tokenized using the TextVectorization class in TensorFlow.

Define Network and Training
This section defines the Encoder-Decoder architecture using Keras. The Encoder processes the English input text and the Decoder generates the Spanish output text. The model is trained using the fit() method in Keras.

Evaluate and Translate
This section loads the trained model, generates translations for sample English sentences, and prints the translations to the console.

Credits
This code was adapted from the tutorial on the TensorFlow website: https://www.tensorflow.org/tutorials/text/nmt_with_attention

License
This code is licensed under the MIT License. See the LICENSE file for details.
