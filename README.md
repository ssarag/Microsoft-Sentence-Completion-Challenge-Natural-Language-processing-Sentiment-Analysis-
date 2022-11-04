# Microsoft-Sentence-Completion-Challenge

Text prediction algorithms for automatic or semi-automatic sentence completion have been vastly discussed in the Natural Language Processing domain.These algorithms are widely used to enhance communication speed, reduce the time taken to compose the text, and play a pivotal role in developing cognitive abilities to intercept and examine written texts. This project focuses on building and evaluating models for Microsoft Research Sentence Completion Challenge and carriesout a comparison of the effects of these models on the performance. The task of sentence completion, which aims to infer missing text in a given sentence was carried out to access the reading comprehension ability of machines using different algorithms


# Motivation and Methodology

In the Project, two approaches – n-gram modelling and BERT for evaluating the models on the test sentences were used. The n-gram model is extensively used in text mining and Natural Language Processing. The other model used is Bidirectional Encoder Representation from Transformers (BERT) which has caused a stir in the Natural 
Language Processing community by presenting state-of-the-art results in a wide variety of NLP tasks like question answering, natural language inference, etc. BERT’s key 
technical innovation is applying the bidirectional training of Transformer to language modelling. The motivation behind applying these models to the sentence completion challenge data was to evaluate the performance of the n-gram model and BERT and explore the effects of applying different features like smoothing and hyperparameter 
tuning in the development process and gain insights into how the models process natural language.

#Methods

The MSR SCC paper has details about the analysis of 6 different approaches. The highest scoring was the human baseline. Latent Semantic Analysis (LSA) using similarity 
calculations between the cosines of the angles between vector forms of different words and the candidate vector gave the accuracy of 49%. The other approaches were variations of n-gram language modelling which gave the accuracy in the range were variations of n-gram language modelling which gave the accuracy in the range 31% to 39%. The results from these models indicated that the “Holmes” sentence completion set is indeed a challenging problem with a level of difficulty comparable to SAT questions. Simple models such as n-gram do not give the best results. 

![image](https://user-images.githubusercontent.com/103538049/200082664-6386e8a2-2181-46a0-949c-9fde95ee4ff8.png)

![image](https://user-images.githubusercontent.com/103538049/200082692-8cc23cac-091a-4c50-9f40-370ed16407ab.png)


In this investigation, two different approaches are used to analyze the problem. 
1. N-gram Model
2. BERT Model

The first approach is using n-gram modelling. The n-gram modelling was tried with simple unigram, bigram, and trigram models as well as with smoothed unigram, bigram, and trigram models. The smoothed n-gram model shows some progression in the accuracy than the simple n-gram model after adjusting the hyperparameters.




![N-gram model](https://user-images.githubusercontent.com/103538049/200083168-d81f973e-0190-4cd2-b03e-5f4a04e03ff2.png)





The second model used is BERT which is considered to give state-of-the-art results. After applying the BERT model, the accuracy showed a positive increment as the 
model could move across the local information constraint imposed on n-grams and provide global semantic coherence.





![BERT model](https://user-images.githubusercontent.com/103538049/200083207-e40bf0aa-7ab9-43b0-8599-17b22a3ce1d6.jpg)


# Experimental Results

1. N-gram Language Model 

In this model the questions were first tokenized to break the raw text into words and these tokens will be further used to calculate the probabilities of the words. 
__START and __END was added to each sentence in the dataset with the tokenization step. The N-gram model was trained with an increasing number of text files starting with 10 files in the first iteration, and 20 in the next iteration and so on. For calculating the probability of a word in the given sentence, the model gives the token which has the maximum probability among the tokens in that context. The N-gram model handles the Out of Vocabulary (OOV) words by passing a “known” parameter during the initialization. When the model is initialized, it checks for the words whose occurrence is less than the value of known and replaces it with the “UNK” token. For experimenting, the known parameter was iterated with values of known = 2, 3, and 4. The language model was first trained on 10 files with unknown =2. 
“UNK” words are the maximum occurring words among the tokens. As the vocabulary size of the files was less, the model did not perform that well with 10 files. As the number of files increased, the performance of the model increased as well. The probability of a “UNK” word in the unigram model comes out to be 0.01031. The total number of times a “UNK” word has occurred in bigram is 2404 and in trigram is 10521. This will impact the sentence completion because even if the word occurred 
less time, it might have a probability greater than some other words in the token or UNK. 


# Hyperparameter tuning
Kneser-Ney is widely used in speech and language processing and has shown successful performance in the natural language processing domain. Therefore, to tune the 
model, here Kneser-Ney smoothing is used. After applying the Kneser-Ney smoothing to the N-gram model, the accuracy of the model increased by around 5%. The accuracy was also checked by taking OVV thresholds words and the number of files.


![image](https://user-images.githubusercontent.com/103538049/200083863-9f27e1f2-c795-43b7-bbc9-7e18e6a52d4e.png)


2. BERT

The hyperparameters that we choose, have significant impact on the performance of the model. BERT is a pre-trained model where the parameters are tuned already. In this investigation, 𝐵𝐸𝑅𝑇𝐵𝐴𝑆𝐸 model is used. The difference between 𝐵𝐸𝑅𝑇𝐵𝐴𝑆𝐸 and 𝐵𝐸𝑅𝑇𝐿𝑎𝑟𝑔𝑒 is on the number of encoder layers. 𝐵𝐸𝑅𝑇𝐵𝐴𝑆𝐸 uses 12 encoder layes, while 𝐵𝐸𝑅𝑇𝐿𝑎𝑟𝑔𝑒 has 24 encoders. As the number of layers in BERT increase, the parameters increase as well. Fine tuning the BERT model with 𝐵𝐸𝑅𝑇𝐿𝑎𝑟𝑔𝑒 will require a lot of hardware specifications and memory. Therefore, the finetuning of 𝐵𝐸𝑅𝑇𝐵𝐴𝑆𝐸model was not performed practically.

There are other ways in which the BERT model can be fine-tuned. The first method is to set a Baseline with Grid Search over a set of predefined hyperparameters. The search space recommend by BERT Authors can be used. For this investigation per_gpu_batch_size, learning_rate and num_epochs were used. We can then run the model for a total of 18 trials or full training runs for each combination of hyperparameters to increase the accuracy. The Grid Search method increases accuracy by 5%.

The second method is to improve Grid Search with Bayesian optimization. In this approach a Gaussian Process model is fit which predicts the performance of the parameters (loss) and informs future hyperparameters. For this method weight decay and warmup_steps are also added in the grid Search with Table 2: Accuracy of the N-gram Models per_gpu_batch_size, learning_rate and num_epochs. For the sentence completion challenge a total of 60 trails can be run. This method increases the accuracy by 6%. 


# Results

![image](https://user-images.githubusercontent.com/103538049/200084370-c4173e25-5053-472e-ace9-017868b8a0e6.png)



![image](https://user-images.githubusercontent.com/103538049/200084405-92a4e627-7920-4ecd-9b9b-4297a2897d47.png)


![image](https://user-images.githubusercontent.com/103538049/200084436-17c50664-173e-4026-ab82-c6ed0f8e37d8.png)

