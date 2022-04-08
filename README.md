# Implemented Word2Vec model using CBOW algorithm to learn the representations of word sounds

## Overview 
During training we generate the word embeddings which is the weight matrix of the model using Vocabulary V. CBOW algorithm here is used to learn the sequence of observed pronunciation or word sounds which is well suited to learn the factors that influence what sounds occur
where. We can make the distributional assumption for sound in a word just like we make for words in a sentence. Sounds are partly determined by the contexts in which they occur.  '
The smallest units of sound is Phonemes which help in discrimination between words and sounds that are similar and word and sounds that are different. Sounds regularly co-occur with other sounds depending on positions of phonetic representations in articulation of words. 
Hence the sequence of sounds make words and sequence of words make sentence though there are some differences in the way these units are correlated for example vowel sequencing, nasal and non-nasal vowels,etc and nouns, verbs, and other grammer rules but they do have the similarity of context in them.      


## Dataset
### The Buckeye Speech Corpus
The Buckeye Corpus of conversational speech contains high-quality recordings from 40 speakers in Columbus OH conversing freely with an interviewer. The speech has been orthographically transcribed and phonetically labeled. The audio and text files, together with time-aligned phonetic labels, are stored in a format for use with speech analysis software.

# Author 
 * [Harshit Monish](https://github.com/harshitmonish)
