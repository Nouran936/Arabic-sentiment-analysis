# Arabic-sentiment-analysis
Neural networks and deep learning course project:

First: Data preparation & Preprocessing:
a)	Tokenization: the process of converting a sequence of text into smaller parts, known as tokens. This enables the neural networks to process and understand natural language test.

b)	Remove punctuation:
removing any punctuation marks from the text, e.g. (/, , , #, $, ؟, ...etc.).

c)	Removing stop words:
removing stop words by defining a list of common or non-meaningful words that we want to remove from the text, e.g. (على، لو، لكن، حتى، الخ.).

d)	Emojis converter:
converting emojis into English text, then from English to Arabic (إيجابي/سلبي) and replace the emoji itself either by the word ايجابي or the word سلبي.

e)	Stemming: the process of reducing a word to its word stem that affixes to suffixes and prefixes or the roots.

f)	Lemmatization: reducing words to their core meaning, but it will give a complete word that makes sense on its own instead of just a fragment of a word.

g)	Detect language: detecting if the input text is English or Arabic language.

h)	English to Arabic: convert the input English text into Arabic.

i)	Text augmentation: By analyzing data, we have found that the positive class has 19189 samples, the negative class has 11340 samples, and the neutral class has only 1507 samples. Therefore, we have decided to apply text augmentation technique to avoid under sampling of the neutral class.
Before augmentation:
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/ede83137-0594-4f0d-8c27-74707c635bbb)
After augmentation:
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/c3230ed8-17ad-4699-8942-5ccb8295718b)
-------------------------------------------------------------------------------------------------------------------------------------------
Second: Models:
1.	LSTM
2.	Transformer
3.	RNN


1.	LSTM: 
Model Architecture:
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/294447dc-171c-48ab-a907-212de5e46a0c)
•	We have implemented the model according to the following steps:
1.	Fitting the tokenizer in the tokens and converting to sequence.
2.	Adjusting padding sequence based on the desired sequence length.
3.	Converting the labels coming from values of ‘rating’ column to one-hot encoding.
4.	Building the LSTM model using built-in function ‘Sequential’.
5.	Initializing the vocab_size = len(tokenizer.word_index) + 1.
6.	Adjusting dropout rate, setting its value to 0.4.
7.	Adding LSTM layer with number of units = 160.
8.	The last step is adding a dense layer, taking 3 as number of classes and ‘softmax’ as an activation function.

•	The best accuracy for LSTM model was obtained by the following values:
epochs: 4
batch_size: 32
dropout_rate: 0.3
input_size: 15
output_embedding_dimension: 100

•	Testing experiments:
1.	With Lemmatization:
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/20905fbc-865b-479f-957a-42cad5a281b7)

2.	Without stemming & stopwords:
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/7a160908-f961-4b6e-afea-b2826d702052)

3.	With all preprocessing applied:
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/be1fe3c8-bbc6-46ae-89c8-c403ccc17c70)





2.	Transformer:
Model architecture:
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/d9029ece-ce8a-48d8-b764-9cdd89c59fd7)

We have implemented the encoder part in the transformer architecture because our goal is classification, so we don’t need the decoder part that converts to words again.

Our code is divided into two functions. Function to create a transformer block, and a function to build the transformer model.

•	transformer_encoder: a function to create the encoder, and it takes the following hyperparameters:
inputs: input coming out from the embedding layer.
embed_dim: the dimension of the embedding layer. We have set it to 32.
num_heads: the number of heads in MultiHeadAttention layer. We have set it to 10.
ff_dim: number of nodes (dimension) in feedforward layer. We have set it to 32.
rate: dropout rate to avoid overfitting. We have set its value to 0.3.

This function is the implementation of the encoder part in the model architecture. It consists of the input coming from the embedding layer, entering the MultiHeadAttention to give each word a value according to its meaning in the context. 
A dropout function is applied on the output for regularization to avoid overfitting. Then a normalization function is applied.
The output after normalization is then fed to a feedforward layer, with ‘relu’ activation function.
Finally, normalization and regularization are applied again.

•	build_transformer_model: a function for building the transformer model, and it takes the following hyperparameters:
max_len: length of input sequence. We have set its value to 100.
vocab_size: We have set its value to 45000.
embed_dim: the dimension of the embedding layer. We have set it to 32.
num_heads: the number of heads in MultiHeadAttention layer. We have set it to 10.
ff_dim: number of nodes (dimension) in feedforward layer. We have set it to 32.
num_classes: number of classes we want to classify according to, and we have set it to 3 because we are having 3 classes (positive, negative, and neutral). 

Firstly, we initialize the inputs with the max_len value. Then we feed it to the embedding layer that takes hyperparameters (input dimension = vocab_size, and output dimension = embed_dim).

Then we call the transformer_encoder function to build the encoder block.

To reduce the dimension of the features coming from transformer encoder, we have used GlobalAveragePooling1D built in function.

The output coming from this layer is then fed to the feedforward layer with ‘softmax’ activation function.

Finally, a built-in function ‘Model’ takes the inputs and outputs and returns the model.

•	Testing experiments:
1.	Changing num_heads values:
num_heads=10
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/ce89b916-bb3a-48b1-9e75-a0154d3d4233)
num_heads=20
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/cbdc2765-790c-4b44-814c-b8ebad3fdde9)
num_heads=8
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/e995056a-c593-468d-b333-c428fb37cd5b)
2.	Without using stemming & stop words functions:
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/178b84db-bbae-41ff-9803-c9f5e85536dd)
3.	With all the preprocessing applied:
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/162ed48c-5043-4837-a378-19f6a1e33cf6)
4.	With embed_dim: 150, ff_dim: 150:
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/1d0333c4-5957-49d6-b8b3-131079396655)
5.	With max_len = 16
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/bbbef997-05dd-43b8-b485-c67d9f4ccc4e)





3. RNN:
Model Architecture:
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/48a5463c-b2f1-4fb5-ba2f-b670a509aaed)

•	We have implemented the model according to the following steps (SAME AS LSTM):
1.	Fitting the tokenizer in the tokens and converting to sequence.
2.	Adjusting padding sequence based on the desired sequence length.
3.	Converting the labels coming from values of ‘rating’ column to one-hot encoding.
4.	Building the RNN model using built-in function ‘Sequential’.
5.	Initializing the vocab_size = len(tokenizer.word_index) + 1.
6.	Adjusting dropout rate, setting its value to 0.3.
7.	Adding SimpleRNN layer with number of units(hidden cells) = 50.
8.	The last step is adding a dense layer, taking 3 as number of classes and ‘softmax’ as an activation function.

•	Testing Experiments:
1.	Without stop words & stemming:
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/582f7012-1004-46ea-81cb-279b3059abac)
2.	All preprocessing applied:
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/430eab53-2071-4097-85db-1bf448174dc5)
3.	Without stemming & stop words, SimpleRNN parameter (no. of units) = 64:
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/f6dc0fd8-e831-435e-9851-68bb0e6f86f5),
4.	Without stemming & stop words, SimpleRNN parameter (no. of units) = 128:
![image](https://github.com/Nouran936/Arabic-sentiment-analysis/assets/112628931/f083e2bc-3d1e-427c-890a-0a9619331086)





