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

### Preprocessing the dataset
* To preprocess the dataset, run `pre_process.py` to clean the data and then run `prepare_dataset.py` to break it into smaller training and testing dataset.
* After running the above scripts, you'll get `english-german-both.pkl`, `english-german-train.pkl` and `english-german-test.pkl` datasets for training and testing purposes.

The preprocessing of the data involves:
* Removing punctuation marks from the data.
* Converting text corpus into lower case characters.
* Shuffling the sentences as sentences were previously sorted in the increasing order of their length.

### Training the Encoder-Decoder LSTM model
Run `model.py` to train the model. After successful training, the model will be saved as `model.h5` in your current directory.
* This model uses **Encoder-Decoder LSTMs** for NMT. In this architecture, the input sequence is encoded by the front-end model called encoder then, decoded by backend model called decoder.
* It uses Adam Optimizer to train the model using Stochastic Gradient Descent and minimizes the categorical loss function.

### Evaluating the model
Run `evaluate_model.py` to evaluate the accuracy of the model on both train and test dataset.
* It loads the best saved `model.h5` model.
* The model performs pretty well on train set and have been generalized to perform well on test set.
* After prediction, we calculate Bleu scores for the predicted sentences to check how well the model generalizes.

### Calculating the Bleu scores
**BLEU (bilingual evaluation understudy)** is an algorithm for comparing predicted machine translated text with the reference string given by the human. A high BLEU score means the predicted translated sentence is pretty close to the reference string. More information can be found [here](https://en.wikipedia.org/wiki/BLEU). Below are the BLEU scores for both the training set and the testing set along with the predicted and target English sentence corresponding to the given German source sentence.

On the Training Set:

![Training set Bleu score](https://github.com/vibhor98/Neural-Machine-Translation-using-Keras/blob/master/images/train_bleu.png)

On the Testing Set:

![Testing set Bleu score](https://github.com/vibhor98/Neural-Machine-Translation-using-Keras/blob/master/images/test_bleu.png)

### References:
* [Neural Machine Translation by jointly learning to Align and Translate](https://arxiv.org/pdf/1409.0473v7.pdf)
* [How to develop a Neural Machine Translation System from scratch](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/)
