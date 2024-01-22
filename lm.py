# imports go here
import sys
from collections import Counter
import random
import re

"""
Name: Anusha Kalbande
NLP homework 2
"""

# Feel free to implement helper functions

class LanguageModel:
    # constants to define pseudo-word tokens
    # access via self.UNK, for instance
    UNK = "<UNK>"
    SENT_BEGIN = "<s>"
    SENT_END = "</s>"

    def __init__(self, n_gram, is_laplace_smoothing):
        """Initializes an untrained LanguageModel
        Parameters:
          n_gram (int): the n-gram order of the language model to create
          is_laplace_smoothing (bool): whether or not to use Laplace smoothing
        """
        self.n_gram = n_gram
        self.is_laplace_smoothing = is_laplace_smoothing
        self.num_of_tokens = 0
        self.tokens = {}
        self.token_dictionary = {}
        self.Ngrams = {}
        self.lowerNGrams = {}
        self.vocabulary = []


    def readFromFile(self, file_path):
        """
        Method to read the file
        :param file_path: File to read data from
        :return: data read from file
        """
        file = open(file_path)
        data = file.read()
        file.close()
        return data

    def tokenizeData(self, data):
        """
        Method to tokenize the data
        :param data: data to tokenize
        :return: tokenized data
        """
        data = data.replace("\n", " ")
        return data.split()


    def processNGrams(self,tokenized_data,N):
        """
        :param tokenized_data: data to be divided into N grams
        :param N: denotes N grams
        :return: Ngrams
        """
        if N == 1:
            return self.tokens
        else:
            Ngrams = {}
            splitGrams = [tokenized_data[i:i + N] for i in range(len(tokenized_data)-N + 1)]
            for splitGram in splitGrams:
                splitGram = " ".join(splitGram)
                if splitGram in Ngrams.keys():
                    Ngrams[splitGram] += 1
                else:
                    Ngrams[splitGram] = 1
            return Ngrams


    def train(self, training_file_path):
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Parameters:
          training_file_path (str): the location of the training data to read

        Returns:
        None
        """

        train_data = self.readFromFile(training_file_path) #read the training data
        tokenized_data = self.tokenizeData(train_data)    #tokenize the data
        self.tokens = Counter(tokenized_data)
        self.num_of_tokens = len(tokenized_data)
        self.vocabulary = self.tokens.keys()

        processed_data = self.processUNK(train_data)  #process the data for 'UNK'
        tokenized_data = self.tokenizeData(processed_data) #tokenize after processing for UNK
        self.tokens = dict(Counter(tokenized_data))
        self.num_of_tokens = len(tokenized_data)
        self.vocabulary = self.tokens.keys()
        self.Ngrams = self.processNGrams(tokenized_data,self.n_gram) #collect the n grams
        if (self.n_gram > 1):
            self.lowerNGrams = self.processNGrams(tokenized_data, (self.n_gram - 1)) #collect n-1 grams to calculate scores


    def generateNgramSentence(self):
        """
        Generates the required sentence using n gram prediction.
        :return: n gram predicted sentence.
        """
        sentence = []
        endWord = self.SENT_BEGIN
        sentence.append(endWord)
        while endWord != self.SENT_END:
            probableNgrams = []
            for ngram in self.Ngrams:
                if endWord == ngram.split()[0]:
                    probableNgrams.append(ngram)
            endWord = random.choice(probableNgrams)
            endWord = endWord.split()[-1] #change
            sentence.append(endWord)

        sentence = " ".join(sentence)
        return sentence

    def generateUnigramSentence(self):
        """
        Method to generate a sentence using Unigram approach
        :return: unigram generated sentence
        """
        sentence = []
        word = self.SENT_BEGIN
        sentence.append(word)
        while word != self.SENT_END:
            if (word != self.SENT_BEGIN):
                sentence.append(word)
            word = random.choice(list(self.tokens.keys()))

        sentence.append(self.SENT_END)
        return " ".join(sentence)



    def processUNK(self,train_data):
        """
        Method to replace a word occuring less than once in the train data with the token <UNK>
        :param train_data: data to process for UNK
        :return: data with certain words replaced by UNK.
        """
        keysToDel = []
        for key, value in self.tokens.items():
            if value == 1:
                keysToDel.append(key)

        if len(keysToDel) > 0:
            for key in keysToDel:
                regex = r"\b" + re.escape(key) + r"\b"
                train_data = re.sub(regex, self.UNK, train_data)

        return train_data

    def score(self, sentence):
        """Calculates the probability score for a given string representing a single sentence.
        Parameters:
          sentence (str): a sentence with tokens separated by whitespace to calculate the score of

        Returns:
          float: the probability value of the given string for this model
        """
        finalScore = 1.0
        intermediateScore = 0.00
        sentenceTokens = []

        tokens = sentence.split()
        if (self.n_gram == 1):
            sentenceTokens = tokens
        else:
            sentenceTokens = [tokens[i:i + self.n_gram] for i in range(len(tokens) - self.n_gram + 1)]

        if (self.is_laplace_smoothing == True):
            if self.n_gram == 1:
                for sToken in sentenceTokens:
                    if sToken in self.Ngrams.keys():
                        intermediateScore = (self.Ngrams[sToken] + 1) / (
                                    (self.num_of_tokens) + (len(self.vocabulary)))  # remove
                    else:
                        intermediateScore = (self.Ngrams[self.UNK] + 1) / (self.num_of_tokens + (len(self.vocabulary)))
                    finalScore *= intermediateScore
            else:
                for sToken in sentenceTokens:
                    lowerToken = sToken[:self.n_gram - 1]
                    lowerToken = " ".join(lowerToken)
                    sToken = " ".join(sToken)
                    if sToken in self.Ngrams.keys():
                        intermediateScore = (self.Ngrams[sToken] + 1) / (self.lowerNGrams[lowerToken] + len(self.vocabulary))
                    else:
                        if lowerToken in self.lowerNGrams:
                            intermediateScore = (0 + 1) / (self.lowerNGrams[lowerToken] + len(self.vocabulary))
                        else:
                            ngram = str((self.UNK + " ") * self.n_gram).strip()
                            lowergram = str((self.UNK + " ") * (self.n_gram - 1)).strip()
                            intermediateScore = (self.Ngrams[ngram] + 1) / (self.lowerNGrams[lowergram] + len(self.vocabulary))

                    finalScore *= intermediateScore

        else:
            if self.n_gram == 1:
                for sToken in sentenceTokens:
                    if sToken in self.Ngrams.keys():
                        intermediateScore = self.Ngrams[sToken] / self.num_of_tokens
                    else:
                        intermediateScore = self.Ngrams[self.UNK] / self.num_of_tokens
                    finalScore *= intermediateScore
            else:
                for sToken in sentenceTokens:
                    lowerToken = sToken[:self.n_gram-1]
                    lowerToken = " ".join(lowerToken)
                    sToken = " ".join(sToken)
                    if sToken in self.Ngrams.keys():
                        intermediateScore = self.Ngrams[sToken] / self.lowerNGrams[lowerToken]
                    else:
                        intermediateScore = 0 # the numerator for this term becomes zero
                    finalScore *= intermediateScore

        return finalScore

    def generate_sentence(self):
        """Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
          str: the generated sentence
        """
        if self.n_gram == 1:
            return self.generateUnigramSentence()
        else:
            return self.generateNgramSentence()

    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
        Parameters:
          n (int): the number of sentences to generate

        Returns:
          list: a list containing strings, one per generated sentence
        """
        sentences = []
        for i in range(0, n):
            sentences.append(self.generate_sentence())
        return sentences

    def scoreTestSet(self, testFilePath):
        """
        Method to score each sentence in the given test set.
        :param testFilePath: test file to be tested.
        """
        testFile = open(testFilePath)
        testData = testFile.read()
        testFile.close()
        testData = testData.split("\n")

        sentenceScores = []
        for sentence in testData:
            sentenceScores.append(self.score(sentence))

        average = sum(sentenceScores) / len(sentenceScores)
        print("Average of the sentence scores: ", average)
        variance = sum([((x - average) ** 2) for x in sentenceScores]) / len(sentenceScores)
        standardDeviation = variance ** 0.5
        print("Standard deviation of the sentence scores: ", standardDeviation)

    def perplexity(self, test_sequence):
        """
        Method to calculate the perplexity of a test sequence.
        :param test_sequence: the test sequence to be used for calculating perplexity
        :return: perplexity.
        """
        score = self.score(test_sequence)
        inverseScore = 1 / score
        root = self.n_gram
        perplexity = inverseScore ** (1 / root)
        return perplexity


def main():
    # TODO: implement
    training_path = sys.argv[1]
    testing_path1 = sys.argv[2]
    testing_path2 = sys.argv[3]


    one = LanguageModel(1,False)
    one.train(training_path)

    two = LanguageModel(2, False)
    two.train(training_path)


    sentences = one.generate(50)
    print("Sentences generated using Unigram approach:\n")
    for sentence in sentences:
        print(sentence +"\n")


    sentences = two.generate(50)
    print("Sentences generated using Bigram approach:\n")
    for sentence in sentences:
        print(sentence + "\n")
    


    print("\nScoring the provided test set hw2-test.txt using unigram approach: ")
    one.scoreTestSet(testing_path1)
    print("\nScoring the provided test set hw2-test.txt using bigram approach: ")
    two.scoreTestSet(testing_path1)
    

    print("\nScoring the curated test set hw2-my-test.txt using unigram approach: ")
    one.scoreTestSet(testing_path2)

    print("\nScoring the curated test set hw2-my-test.txt using bigram approach: ")
    two.scoreTestSet(testing_path2)
    

    onePerplex = LanguageModel(1,True)
    onePerplex.train(training_path)
    print("\n Calculating perplexity on given test set using unigrams:")
    perplexData = "<s> a vegetarian meal about ten miles and i'm willing to drive ten miles and this will be for dinner are any of these restaurants open for breakfast are there russian restaurants in berkeley between fifteen and twenty dollars can you at least list the nationality of these restaurants can you give me more information on viva taqueria dining </s>"
    print("Perplexity: ",onePerplex.perplexity(perplexData))


    twoPerplex = LanguageModel(2,True)
    twoPerplex.train(training_path)
    print("\n Calculating perplexity on given test set using bigrams:")
    perplexData = "<s> a vegetarian meal about ten miles and i'm willing to drive ten miles and this will be for dinner are any of these restaurants open for breakfast are there russian restaurants in berkeley between fifteen and twenty dollars can you at least list the nationality of these restaurants can you give me more information on viva taqueria dining </s>"
    print("Perplexity: ", twoPerplex .perplexity(perplexData))




if __name__ == '__main__':
    # make sure that they've passed the correct number of command line arguments
    if len(sys.argv) != 4:
        print("Usage:", "python hw2_lm.py training_file.txt testingfile1.txt testingfile2.txt")
        sys.exit(1)

    main()

