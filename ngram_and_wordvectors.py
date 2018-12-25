import math;
import numpy;

input='Assignment1_resources/Assignment1_resources/development/obama.txt'


def add_to_bigram_dict(bigram, prev_word, word):
    if prev_word in bigram:
        bigram[prev_word]['count'] = 1 + bigram[prev_word]['count']
        bigram[prev_word]['successors'][word] = 1 + bigram[prev_word]['successors'].get(word, 0)
    else:
        bigram[prev_word] = {'count': 1, 'successors': {word: 1}}


def process_corpus(file_name):
    unigram = {}
    bigram = {}
    N = 0
    start_token='<s>'
    end_token='</s>'
    with open(file_name) as input_file:
        file_content = input_file
        for line in file_content:
            prev_word = start_token
            for word in line.split():
                add_to_bigram_dict(bigram, prev_word, word)
                N += 1
                if word == '.' or word == '?' or word == '!':
                    add_to_bigram_dict(bigram, word, end_token)
                    add_to_bigram_dict(bigram, end_token, start_token)
                    word = start_token
                    N += 2
                prev_word = word
    return {
        'N': N,
        'V': len(bigram),
        'bigram': bigram,
    }


def get_bigram_next_distribution(bigram, word):
    if 'next' in bigram[word]:
        return (bigram[word]['next'], bigram[word]['distribution'])
    bigram[word]['next'] = []
    bigram[word]['distribution'] = []
    for next_word in bigram[word]['successors']:
        bigram[word]['next'].append(next_word)
        bigram[word]['distribution'].append(float(bigram[word]['successors'][next_word]) / bigram[word]['count'])
    return (bigram[word]['next'], bigram[word]['distribution'])


def random_sentence_generation_bigram(data, seed=None):
    bigram = data['bigram']
    if not seed:
        sentence = ['<s>']
        word = '<s>'
        length = 0
        while length <= 20 and word != '</s>':
           next, distribution = get_bigram_next_distribution(bigram, word)
           new_word = numpy.random.choice(a=next,p=distribution)
           sentence.append(new_word)
           word = new_word
        return sentence
    sentence = [seed]
    word = seed
    length = 0
    while length <= 20 and word != '</s>':
        next, distribution = get_bigram_next_distribution(bigram, word)
        new_word = numpy.random.choice(a=next,p=distribution)
        sentence.append(new_word)
        word = new_word
    return sentence



def get_unigram_next_distribution(data, total):
    next = []
    distribution = []
    for word in data:
        next.append(word)
        distribution.append(float(data[word]['count']) / total)
    return (next, distribution)


def convert_list_to_sentence(sentence_list):
    sentence = ""
    for word in sentence_list:
        if word == '<s>' or word == '</s>':
            continue
        sentence = "{sentence} {word}".format(sentence=sentence, word=word)
    return sentence


def random_sentence_generation_unigram(data, seed=None):
    next, distribution = get_unigram_next_distribution(data['bigram'], total=data['N'])
    if seed is None:
        sentence = ['<s>']
        word = '<s>'
        length = 0
        while length <= 20 and word != '</s>':
           new_word = numpy.random.choice(a=next,p=distribution)
           sentence.append(new_word)
           word = new_word
        return sentence
    sentence = ['<s>','I','like']
    word = 'like'
    length = 0
    while length <= 20 and word != '</s>':
        new_word = numpy.random.choice(a=next,p=distribution)
        sentence.append(new_word)
        word = new_word
    return sentence


def handle_unknowns(data, threshold = 1, unk_token='<unk>'):
    under_threshold = set()
    smoothed_bigrams = {}
    smoothed_bigrams[unk_token] = {'count': 0, 'successors': {}}
    # Handle preceeding unknowns and populate set of unknown words
    for word in data['bigram']:
        if data['bigram'][word]['count'] <= threshold:
            under_threshold.add(word)
            smoothed_bigrams[unk_token]['count'] += data['bigram'][word]['count']
            for successor in data['bigram'][word]['successors']:
                if successor in smoothed_bigrams[unk_token]['successors']:
                    smoothed_bigrams[unk_token]['successors'][successor] += data['bigram'][word]['successors'][successor]
                else:
                    smoothed_bigrams[unk_token]['successors'][successor] = data['bigram'][word]['successors'][successor]
        else:
            smoothed_bigrams[word] = {'count': data['bigram'][word]['count'], 'successors': data['bigram'][word]['successors']}
    # Handle succeeding unknown words
    for word in smoothed_bigrams:
        unk_count = 0
        smoothed_successors = {}
        for successor in smoothed_bigrams[word]['successors']:
            if successor in under_threshold:
                unk_count += smoothed_bigrams[word]['successors'][successor]
            else:
                smoothed_successors[successor] = smoothed_bigrams[word]['successors'][successor]
        if unk_count > 0:
            smoothed_successors[unk_token] = unk_count
        smoothed_bigrams[word]['successors'] = smoothed_successors
    return smoothed_bigrams




def calculate_bigram_probability_log(smoothed_bigrams, prev_word, word, V, k):
    unk_token = '<unk>'
    #if prev_word in smoothed_bigrams and word in smoothed_bigrams:
    #    if word in smoothed_bigrams[prev_word
    if not prev_word in smoothed_bigrams:
        prev_word = unk_token
    if (not word in smoothed_bigrams):# or not (word in smoothed_bigrams[word]['successors']):
        word = unk_token
    return math.log(float(k + smoothed_bigrams[prev_word]['successors'].get(word, 0)) / (smoothed_bigrams[prev_word]['count'] + k * V))



# Find paragraph wise perplexity from a given validation_file and return summation of all, for a given k value
def find_bigram_perplexity(validation_file, smoothed_bigrams, k):
    file_perplexity = 0.0
    start_token='<s>'
    end_token='</s>'
    V = len(smoothed_bigrams)
    para_count = 0
    with open(validation_file) as input_file:
        file_content = input_file
        for line in file_content:
            para_count += 1
            N = 0
            prev_word = start_token
            paragraph_perplexity = 0.0
            for word in line.split():
                paragraph_perplexity += calculate_bigram_probability_log(smoothed_bigrams, prev_word, word, V, k)
                N += 1
                if word == '.' or word == '?' or word == '!':
                    paragraph_perplexity += calculate_bigram_probability_log(smoothed_bigrams, word, end_token, V, k)
                    paragraph_perplexity += calculate_bigram_probability_log(smoothed_bigrams, end_token, start_token, V, k)
                    N += 2
                    word = start_token
                prev_word = word
            paragraph_perplexity = math.exp(paragraph_perplexity * float(-1)/N)
            file_perplexity += paragraph_perplexity
    return file_perplexity/para_count



def test_bigram_perplexity(test_file, smoothed_bigrams_0, smoothed_bigrams_1, k):
    start_token='<s>'
    end_token='</s>'
    V_0 = len(smoothed_bigrams_0)
    V_1 = len(smoothed_bigrams_1)
    para_count = 0
    prediction=[]
    with open(test_file) as input_file:
        file_content = input_file
        for line in file_content:
            N = 0
            prev_word = start_token
            paragraph_perplexity_0 = 0.0
            paragraph_perplexity_1 = 0.0
            for word in line.split():
                paragraph_perplexity_0 += calculate_bigram_probability_log(smoothed_bigrams_0, prev_word, word, V, k)
                paragraph_perplexity_1 += calculate_bigram_probability_log(smoothed_bigrams_1, prev_word, word, V, k)
                N += 1
                if word == '.' or word == '?' or word == '!':
                    paragraph_perplexity_0 += calculate_bigram_probability_log(smoothed_bigrams_0, word, end_token, V_0, k)
                    paragraph_perplexity_0 += calculate_bigram_probability_log(smoothed_bigrams_0, end_token, start_token, V_0, k)
                    paragraph_perplexity_1 += calculate_bigram_probability_log(smoothed_bigrams_1, word, end_token, V_1, k)
                    paragraph_perplexity_1 += calculate_bigram_probability_log(smoothed_bigrams_1, end_token, start_token, V_1, k)
                    N += 2
                    word = start_token
                prev_word = word
            paragraph_perplexity_0 = math.exp(paragraph_perplexity_0 * float(-1)/N)
            paragraph_perplexity_1 = math.exp(paragraph_perplexity_1 * float(-1)/N)
            prediction.append((para_count, 0 if paragraph_perplexity_0 < paragraph_perplexity_1 else 1))
            para_count += 1
    return prediction







try_k_values(
    input_obama='Assignment1_resources/Assignment1_resources/train/obama.txt',
    validation_obama='Assignment1_resources/Assignment1_resources/development/obama.txt',
    input_trump='Assignment1_resources/Assignment1_resources/train/trump.txt',
    validation_trump='Assignment1_resources/Assignment1_resources/development/trump.txt',
    k_list=[0.1, 0.01, 0.05, 0.5, 1, 0.001, 0.0001, 0.025, 0.25, 0.08],
)

def try_k_values(input_obama, validation_obama, input_trump, validation_trump, k_list):
    data_obama=process_corpus(input_obama)
    smoothed_bigrams_obama=handle_unknowns(data_obama)
    data_trump=process_corpus(input_trump)
    smoothed_bigrams_trump=handle_unknowns(data_trump)
    for k in k_list:
        file_perplexity_obama=find_bigram_perplexity(validation_file=validation_obama, smoothed_bigrams=smoothed_bigrams_obama, k=k)
        file_perplexity_trump=find_bigram_perplexity(validation_file=validation_trump, smoothed_bigrams=smoothed_bigrams_trump, k=k)
        file_perplexity_obama_using_trump_model=find_bigram_perplexity(validation_file=validation_obama, smoothed_bigrams=smoothed_bigrams_trump, k=k)
        file_perplexity_trump_using_obama_model=find_bigram_perplexity(validation_file=validation_trump, smoothed_bigrams=smoothed_bigrams_obama, k=k)
        print('{k}:\t{file_perplexity_obama}\t{file_perplexity_trump}\t{file_perplexity_obama_using_trump_model}\t{file_perplexity_trump_using_obama_model}'.format(
            k=k,
            file_perplexity_obama=file_perplexity_obama,
            file_perplexity_trump=file_perplexity_trump,
            file_perplexity_obama_using_trump_model=file_perplexity_obama_using_trump_model,
            file_perplexity_trump_using_obama_model=file_perplexity_trump_using_obama_model,
        ))




file_perplexity_obama
file_perplexity_trump
file_perplexity_obama_using_trump_model
file_perplexity_trump_using_obama_model


input_obama='Assignment1_resources/Assignment1_resources/train/obama.txt'
validation_obama='Assignment1_resources/Assignment1_resources/development/obama.txt'
input_trump='Assignment1_resources/Assignment1_resources/train/trump.txt'
validation_trump='Assignment1_resources/Assignment1_resources/development/trump.txt'

data_obama=process_corpus(input_obama)
smoothed_bigrams_obama=handle_unknowns(data_obama)
data_trump=process_corpus(input_trump)
smoothed_bigrams_trump=handle_unknowns(data_trump)






predictions=test_bigram_perplexity(test_file, smoothed_bigrams_0, smoothed_bigrams_1, k)
for record in predictions:
    print('{para},{result}'.format(
        para=record[0],
        result=record[1],
    ))


def get_all_words(data_obama, data_trump):
    words = set()
    for word in data_obama['bigram']:
        words.add(word)
    for word in data_trump['bigram']:
        words.add(word)
    return words



def get_vectors_for_matching_words(words, vector_file_name):
    vector_dict={}
    unk_token='<unk>'
    unk_vector=None
    with open(vector_file_name) as vector_file:
        file_content = vector_file
        for line in file_content:
            vector = line.split()
            if vector[0] in words:
                vector_dict[vector[0]] = [float(value) for value in vector[1:]]
            elif not unk_vector:
                unk_vector = [float(value) for value in vector[1:]]
    vector_dict[unk_token]=unk_vector
    return vector_dict


def generate_avg_vector(vector_dict, data):
    unk_token='<unk>'
    accumulated_vector = [0.0 for x in range(0, 300)]
    for word in data['bigram']:
        #word_vector = vector_dict[word] if word in vector_dict else vector_dict[unk_token]
        if not word in vector_dict:
            continue
        word_vector = vector_dict[word]
        count = 0
        while count < data['bigram'][word]['count']:
            accumulated_vector = [ x + y for x, y in zip(accumulated_vector, word_vector)]
            count += 1
    accumulated_vector = [x / data['N'] for x in accumulated_vector]
    return accumulated_vector


def find_magnitude(vector):
    return math.sqrt(sum([x*x for x in vector]))


def find_cosine_similarity(vector_1, vector_2):
    inner_product = sum([x*y for x,y in zip(vector_1,vector_2)])
    return inner_product/(find_magnitude(vector_1) * find_magnitude(vector_2))


def predict_using_cosine_similarity(obama_vector, trump_vector, vector_dict, test_file_name):
    predictions = []
    unk_token='<unk>'
    with open(test_file_name) as test_file:
        file_content = test_file
        para_number = 0
        for line in file_content:
            N = 0
            accumulated_vector = [0.0 for x in range(0, 300)]
            for word in line.split():
                #word_vector = vector_dict[word] if word in vector_dict else vector_dict[unk_token]
                if not word in vector_dict:
                    continue
                word_vector = vector_dict[word]
                accumulated_vector = [ x + y for x, y in zip(accumulated_vector, word_vector)]
                N += 1
            accumulated_vector = [x / N for x in accumulated_vector]
            obama_similarity = find_cosine_similarity(obama_vector, accumulated_vector)
            trump_similarity = find_cosine_similarity(trump_vector, accumulated_vector)
            print('{first} {second}'.format(first=obama_similarity, second=trump_similarity))
            predictions.append((para_number, 0 if obama_similarity > trump_similarity else 1))
            para_number += 1
    return predictions



import gensim;

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

#path = get_tmpfile("word2vec.model")

#model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
#model.save("word2vec.model")

#model = Word2Vec.load("word2vec.model")
#vector = model.wv['computer']  # numpy vector of a word

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

word_vectors = model.wv
del model
word_vectors.vocab.keys()

test_file_name='Assignment1_resources/Assignment1_resources/test/test.txt'
words=get_all_words(data_obama, data_trump)
#vector_file_name='glove.840B.300d.txt'
#vector_dict=get_vectors_for_matching_words(words, vector_file_name)
vector_dict=word_vectors
obama_vector = generate_avg_vector(vector_dict, data_obama)
trump_vector = generate_avg_vector(vector_dict, data_trump)
predictions = predict_using_cosine_similarity(obama_vector, trump_vector, vector_dict, test_file_name)

analogy_vector_file_name='Assignment1_resources/Assignment1_resources/analogy_test.txt'
analogy_vector_dict=get_vectors_for_matching_words(words, vector_file_name=analogy_vector_file_name)

analogy_file='Assignment1_resources/Assignment1_resources/analogy_test.txt'
analogy_words=find_distinct_words(file_name=analogy_file)



def find_distinct_words(file_name):
    words = set()
    with open(file_name) as input_file:
        file_content = input_file
        for line in file_content:
            for word in line.split():
                words.add(word)
    return words



def process_analogies(analogy_file, word_vectors):
    analogy_words=find_distinct_words(file_name=analogy_file)
    attempted_predicitons = 0
    correct_predictions = 0
    with open(analogy_file) as input_file:
        file_content = input_file
        for line in file_content:
            line = line.split()
            if not line[0] in analogy_words or not line[1] in analogy_words or not line[2] in analogy_words or not line[3] in analogy_words:
                continue
            attempted_predicitons += 1
            target_vector = [ x + y - z for x, y, z in zip(word_vectors[line[2]], word_vectors[line[1]], word_vectors[line[0]])]
            if line[3] == find_closest_word(word_vectors, analogy_words, target_vector, line):
                correct_predictions += 1
    return float(correct_predictions) / attempted_predicitons


def find_closest_word(word_vector, analogy_words, target_vector, line):
    closest_word = None
    cosine_similarity = None
    for analogy_word in analogy_words:
        if not closest_word:
            closest_word = analogy_word
            closest_vector = word_vector[analogy_word]
            cosine_similarity = find_cosine_similarity(vector_1=target_vector, vector_2=word_vector[analogy_word])
            continue
        if analogy_word == line[0] or analogy_word == line[1] or analogy_word == line[2]:
            continue
        current_cosine_sililarity = find_cosine_similarity(vector_1=target_vector, vector_2=word_vector[analogy_word])
        if current_cosine_sililarity > cosine_similarity:
            cosine_similarity = current_cosine_sililarity
            closest_word = analogy_word
    return closest_word



process_analogies(analogy_file, word_vectors)

def process_analogies(analogy_file, word_vectors, start_line, end_line):
    analogy_words=find_distinct_words(file_name=analogy_file)
    attempted_predicitons = 0
    correct_predictions = 0
    line_number=0
    with open(analogy_file) as input_file:
        file_content = input_file
        for line in file_content:
            line_number += 1
            if line_number < start_line or line_number > end_line:
                continue
            line = line.split()
            if not line[0] in word_vectors.vocab.keys() or not line[1] in word_vectors.vocab.keys() or not line[2] in word_vectors.vocab.keys() or not line[3] in word_vectors.vocab.keys():
                continue
            attempted_predicitons += 1
            target_vector = [ x + y - z for x, y, z in zip(word_vectors[line[2]], word_vectors[line[1]], word_vectors[line[0]])]
            if line[3] == find_closest_word(word_vectors, analogy_words, target_vector, line):
                correct_predictions += 1
    return (correct_predictions, attempted_predicitons)
