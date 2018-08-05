# Neural-Machine-Translation-using-Keras
This is the sequential Encoder-Decoder implementation of Neural Machine Translation using Keras. This model translates the input German sentence into the corresponding English sentence with a **Bleu Score: 0.509124** on the test set.

**Encoder** - Represents the input text corpus (German text) in the form of embedding vectors and trains the model.

**Decoder** - Translates and predicts the input embedding vectors into one-hot vectors representing English words in the dictionary.

![Model](https://github.com/vibhor98/Neural-Machine-Translation-using-Keras/blob/master/images/model.png)

### Code Requirements
You can install Conda that resolves all the required dependencies for Machine Learning.

Run: `pip install requirements.txt`

### Dataset
We're using dataset containing pairs of English - German sentences and can be downloaded from [here](http://www.manythings.org/anki/).
* This dataset is present in `data/deu.txt` file containing 1,52,820 pairs of English to German phrases.

### Calculating the Bleu scores
On the Training Set:

![Training set Bleu score](https://github.com/vibhor98/Neural-Machine-Translation-using-Keras/blob/master/images/train_bleu.png)

On the Testing Set:

![Testing set Bleu score](https://github.com/vibhor98/Neural-Machine-Translation-using-Keras/blob/master/images/test_bleu.png)

### References:
* [Neural Machine Translation by jointly learning to Align and Translate](https://arxiv.org/pdf/1409.0473v7.pdf)
* [How to develop a Neural Machine Translation System from scratch](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/)
