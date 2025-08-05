'''
    ngram_score module to provide a mechanism to score
    texts based on a n-gramm lookup table. The lookup
    table has to be created out-of-band.

    Date 11.10.2021
'''

from math import log10



'''
    ngram_score class to calculate the n-gramm score
    of a text based on a lookup table of the most
    common n-gramms in a specific language.
'''
class ngram_score(object):

    def __init__(self, file_name, sep=' '):
        '''
            Construct a new n-gram lookup table from the
            provided file. The assumed file structure is
            <ngram> <number-of-occurance>, separated by
            a whitespace.
        '''

        # reak in raw file
        self.ngrams = {}
        for line in open(file_name, 'r'):
            ngram, count = line.split(sep)
            self.ngrams[ngram] = int(count)
        
        # fix some internal parameters
        self.order = len(ngram)
        self.total_ngrams = sum(self.ngrams.values())

        # calculate propabilities
        for ix in self.ngrams.keys():
            p = log10(float(self.ngrams[ix]) / self.total_ngrams)
            self.ngrams[ix] = p

        # define default propability for n-gramms
        # not occuring in the given lookup table
        self.default_value = log10(0.01/self.total_ngrams)


    def order(self):
        '''
            Return the order of the n-gramm.
        '''
        return self.order


    def score(self, input_text, normalize=False):
        '''
            Calculate the Score of the input text based on
            the lookup table. The option 'normalize' is uesed
            to normalize the score based on the text input
            length. While this is required to compare different
            length texts, it has negative effects on scoring
            texts of same length! Only enable it if required!
        '''
        score = 0
        text = input_text.upper()
        for idx in range(len(text)-self.order+1):
            current_ngram = text[idx:idx+self.order]
            if current_ngram in self.ngrams:
                score += self.ngrams[current_ngram]
            else:
                score += self.default_value

        # normalize the score to the length of the
        # given input text. Otherwise longer texts
        # will automatically have lower score, even
        # if they were better natural language
        if normalize:
            score = score / (len(text)-self.order+1)

        return score
    
