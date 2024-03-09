# Final Project Cognition and Computation Course

The project explores the impact of GloVe and BERT vector embeddings on language tasks, with a focus on gender bias. The study replicates Pereira et al.'s (2018) Analysis 1, using fMRI imaging data and GloVe embeddings to predict semantic vectors. The project extends this analysis, incorporating BERT embeddings and additional tasks. The open research task quantifies gender bias in both vector embeddings and fMRI data, referencing Garg et al. (2018). Furthermore, we attempt to replicate a debiasing task from Bolukbasi et al. (2016) to assess its effectiveness in mitigating gender bias identified in previous analyses.

### Semantic Vectors To Decode Brain Activation

The project attempts to replicate Pereira et al.'s Analysis 1 using BERT vector representations on 180 concepts. Despite initially poor performance, dimension reduction via PCA improved results, with 133 successful concepts for BERT compared to 134 for GloVe. However, comparison of successful concepts revealed BERT's advantage in handling complex vocabulary, while GloVe excelled with simpler words. BERT achieved slightly better accuracy ranks overall. The study suggests BERT's transformer-based architecture, designed for context, might explain its performance variation compared to GloVe, which will be further explored using sentence-level vectors.

### Testing a pre-Trained Decoder on Sentences
In this section , the pre-trained decoder, initially trained on individual words, is tested on sentences from Experiments 2 and 3 by Pereira et al. (2018) using GloVe vector embeddings. The evaluation is based on rank-based accuracy, considering the ability of the decoder to generalize concepts from sentences linked to broad topics. Experiment 2, with four-sentence passages, showed success in decoding 21 out of 24 broad topics, while Experiment 3, featuring narrative passages, had 19 successful topics out of 24. Notably, Experiment 3 demonstrated better accuracy scores for the top 15 successful concepts. The difference in outcomes is attributed to the inclusion of narrative elements in Experiment 3, introducing more complex linguistic patterns. While both experiments showcased the decoder's ability to decode semantic vectors from brain imaging data based on textual stimuli, the addition of narrative passages in Experiment 3 led to improved accuracy but fewer successfully decoded topics in total.

### Training a Decoder on Sentence Representations
In this section, we repeated the earlier analysis by training the decoder on sentences, using both GloVe and BERT vector embeddings. The evaluation was conducted on sentences from Experiment 2, employing a k-fold cross-evaluation method. Surprisingly, the results initially showed no significant difference between BERT and GloVe. However, after performing PCA decomposition on the vectors, GloVe outperformed BERT, contrary to our expectations of BERT's superior performance with sentences due to its contextual design. This unexpected outcome suggests that the averaging of word vectors into a single representation treats sentences similarly to single words, contributing to the comparable performance of BERT and GloVe across tasks.

Interestingly, the earlier trend observed—BERT excelling with complex topics and GloVe with more 'everyday' ones—was maintained only prior to PCA. After considering the original components, there was a substantial overlap among their top-scoring concepts, indicating that much of the complexity and nuance captured by BERT extends beyond the principal components.

Additionally, we attempted to build a brain-encoder model, reversing the relationship by encoding fMRI voxels using word embedding vectors. Both GloVe and BERT embeddings demonstrated success in accurately capturing voxel activations. The small differences in performance between the embeddings may reflect various factors, requiring further research beyond the scope of this work to fully understand. Since the fMRI data was obtained from a single test subject, the results provide insights into the potential use of both embeddings in encoding fMRI data, but further research is needed for broader generalizations.

### Gender Bias and Debiasing in Language Models and the Brain

In this section, the aim is to explore and quantify gender bias in language embeddings, extending the analysis from Garg et al. (2018). It acknowledges the theoretical ideal of a bias-free language model but highlights the challenge as models are trained on human-generated data, inevitably reflecting human biases.

The project sets out to determine the existence and quantification of biases in language models, comparing GloVe and BERT embeddings. It also investigates potential bias-revealing patterns in fMRI imaging data and discerns the contribution of inherent language factors versus human perception to language bias.

Additionally, the section aims to replicate debiasing methods demonstrated by Bolukbasi et al. (2016) and assess their effectiveness. The project seeks to answer whether it is possible to remove bias from word embeddings while preserving the intrinsic meaning of the words represented.

The section introduces definitions for terms like "gender-neutral," "gender-specific," and "gender-leaning" to categorize words based on their gender associations. It emphasizes a neutral stance, avoiding moral judgments on gender biases in words, and expresses a language modeling perspective on the issue.

Concerning the vector representation of genders, the initial analysis using individual vectors for 'male' and 'female' resulted in erratic outcomes. Following Garg et al. (2018), the approach was refined by creating vectors representing the average of all word vectors associated with or defining each gender.

In the initial attempt to quantify gender bias, the project analyzed vector similarity differences for gender-neutral words, aiming to compare GloVe, BERT, and fMRI embeddings. A set of 114 gender-neutral words was defined as neutral, and bias scores were computed based on the cosine similarity between each word's vector and vectors for 'male' and 'female.' Words with bias scores in the top 10% extremes of the neutral set were considered biased. BERT exhibited the smallest number of biased concepts (15), with 11 biased toward males, while GloVe had 21 biased concepts, with 8 biased toward males. The fMRI embeddings demonstrated higher bias scores for all words, with a total of 16 biased concepts.

Analysis revealed that gender-specific words exhibited high bias toward their respective genders, and stereotypes related to violence and work manifested as male biases, while concepts tied to appearance and timidness reflected female biases. Surprisingly, occupation-related words, heavily emphasized in previous studies, did not exhibit significant bias.

The differences in bias scores were attributed to the nature of the models; BERT's context-based approach allowed for better identification of nuances, while GloVe, being static, relied on co-occurrences and associated words with genders more frequently. The fMRI data exhibited higher variability, reflecting personal associations from a single participant.

Subsequent clustering analysis using the K-Means algorithm aimed to group similar vectors together and identify biases. GloVe separated genders more frequently than BERT, possibly due to BERT's contextual embeddings creating stronger associations between gender vectors. Adjusting the K-Means approach by fixing centroids resulted in opposite bias outcomes, with GloVe showing no bias for gender-neutral words, while BERT exhibited bias.

PCA decomposition attempts on BERT and fMRI vectors had little impact on fMRI biases but increased the number of biased concepts and bias scores for BERT. This suggested that while principal components defined the vector's general meaning, additional components contributed to sentiment, context, and associations, adding nuance and complexity.

In summary, various analyses, including similarity measures and clustering, revealed gender biases in language embeddings, with differences attributed to the nature of models and the complexity of contextual embeddings in BERT. The results underscore the intricate relationship between bias and vector representations in language models.


### Debiasing

In the process of debiasing embeddings, inspired by the work of Bolukbasi et al. (2016), we aimed to reduce or eliminate gender bias in word representations to promote fairness and address gender inequalities in natural language processing applications.

The first step involved identifying a vector subspace, referred to as the 'gender subspace.' We sought to find a k-dimensional vector subspace representing the gender components of word embeddings, allowing the depiction of the positioning of gender-related words in the embedding space. This was achieved by using 'defining sets,' which are subsets of words closely associated with or used to define gender. Singular-value decomposition (SVD) of a matrix defined by the squared distance of each vector from its subset mean yielded the top-k singular values, determining the gender subspace.

The gender subspace was then used for two debiasing tasks: "Neutralize" and "Equalize." The "Neutralize" task ensured that gender-neutral words had zero influence in the gender subspace. This was achieved by subtracting from the original vector the components that existed in the gender subspace, effectively removing any bias associated with those words. The "Equalize" task aimed to force equidistance between gender-specific pairs for components outside the gender subspace. This ensured that the only difference between such vectors was in the gender definition of the word, promoting equal treatment of gender-specific pairs.

In summary, we employed a method based on singular-value decomposition and defined gender subspaces to perform debiasing tasks, emphasizing neutralization and equalization to mitigate gender bias in word embeddings.

### Results

Results of the debiasing process, inspired by Bolukbasi et al. (2016), were assessed by creating neutralized and equalized sets from a large corpus of GLoVe vectors. The effectiveness of the debiasing was evaluated by verifying that the debiased words did not exhibit bias, as measured in previous analyses, while maintaining their semantic meaning.

When applying the similarity method to assess bias, only three words exhibited bias post-debiasing: "king" displayed bias towards males, "lady" showed bias towards females, and "hair" exhibited a minimal bias towards females. The retention of bias in the gender-specific words was considered inherent to their meaning. The overall success of the debiasing method was reflected in the minimal bias observed.

Decoder analysis on the same set of concepts and fMRI data as previous experiments indicated that the contextual relationships and semantic meanings between words were preserved after debiasing. The results demonstrated the effectiveness of the debiasing algorithms in reducing biases in GLoVe word embeddings while maintaining contextual relationships and semantic meanings.

However, the debiasing process on BERT embeddings was less effective, with most of the original bias retained, and new biased concepts introduced. Gender-specific words in BERT became entirely unbiased, suggesting potential challenges in defining a gender subspace within BERT vectors. The performance of the decoder model on BERT vectors was significantly worse post-debiasing, emphasizing the importance of the retained components for the vector's semantic representation.

In conclusion, GLoVe was identified as more biased than BERT, and despite the less successful debiasing on BERT, the overall effectiveness of the process on GLoVe demonstrated significant improvements in mitigating gender bias and promoting equality in language processing tasks.
