# Fake News Detection Using Claim Specific Article Attention
Internet and the web is a huge part of people’s day to day life in collecting valuable information but fake news is one of the biggest problems in moderns era of civilization. Research on fake news detection is growing in a rapid pace, there are many methods that detect fake news using supervised learning which captures the linguistic styles, stance  information of the corresponding article towards the claim etc to asses the credibility of the claim. However, most works do not consider the external evidence to judge the claim, contextual information of the claims, and corresponding evidence. This paper overcomes these limitations by considering external evidence that supports or renounce the claim and the contextual representation of claim and the corresponding article words. To this extent, we capture the contextual representations of claim texts and article words using BiLSTMs, efficient representations of claim and corresponding article sources and use an end to end neural network with claim specific article word attention to access the credibility of the claim. It also derives informative features for generating user-comprehensible explanations that makes the neural network predictions transparent to the end-user

In this work we access the credibility of a claim by looking at different articles that are relevent to the claim .In order
to get effiecient relevence ,context information of both claim text and article word are important.Using this context information of both claim and article words we produce a claim specific article representation that contains the weighted average of article words. where weights are assigned to the word representation based on thier contextual relevence with the whole claim text representation.In the following sections we list the works that relate to this paper and describe our proposed approach.The novelty of the paper lies in:
* Efficient representation of claim source using Bi-LSTM using self attention instead of traditional word2vec or glove embeddings.
* Efficient article word representation that constitute complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). These word vectors are learned functions of the internal states of a deep bidirectional language model (Bi-LSTM)
* Efficient claim source and article source representations using ratings labels given by other sources
* Use of Loung attention on the above given representations.

## Model Architechture
![image of architecture](architecture.PNG)

## Results

![alt-text-1](results_snopes.PNG "Resutls on Snopes dataset") 
![alt-text-2](newstrust_dataset.PNG "Results on NewsTrust datset")
![Attention](attention_importance.PNG "Visualization of Self Attention Weights")

