# import pickle as pk
# import gensim
# from collections import Counter
# import PyPDF2
# import nltk
# from nltk.corpus import stopwords
# import re
# from nltk import pos_tag,word_tokenize



# def match(resume , job_decription):
#     model = pk.load(open('model.pkl' , 'rb'))
#     resume = pipeline(resume)
#     job_decription = pipeline(job_decription)
#     nonmatched = []
#     matched = []
#     for key in job_decription:
#         if key not in model.wv.key_to_index:
#             continue
#         match = False
#         for word in resume:
#             if word not in model.wv.key_to_index:
#                 continue
#             if(model.wv.similarity(key , word) >= 0.75):
#                 match = True
#         if not match:
#             nonmatched.append(key)
#         else:
#             matched.append(key)
#     matched_words = Counter(matched)
#     nonmatched_words = Counter(nonmatched)
#     match_score = 1
#     nonmatch_score = 1
#     for tuple in matched_words.items():
#         match_score += tuple[1]
#     for tuple in nonmatched_words.items():
#         nonmatch_score += tuple[1]
    
#     return round(100*((match_score/(match_score+nonmatch_score)) + 0.3) , 2) , matched_words.most_common(min(5 , len(matched_words))) , nonmatched_words.most_common(min(5 , len(nonmatched_words)))

# def gen_text(pdf_file):

#     pdf_reader = PyPDF2.PdfReader(pdf_file)

#     text = ''

#     try:
#         for page_num in range(len(pdf_reader.pages)):

#             page = pdf_reader.pages[page_num]

#             text += ' ' + page.extract_text()
#     except:
#         text = ''
        
#     # pdf_file.close()

#     return text

# def extract_keywords(document):
#     text = preprocessing(document)
#     words = word_tokenize(text)
#     tagged = nltk.pos_tag(words)
#     keywords =  [word for word,pos in tagged if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
#     return " ".join(keywords)

# def preprocessing(text):
#     nltk.download('stopwords')
#     sw = set(stopwords.words('english'))
#     text = text.replace('\n',' ').lower()
#     pattern = re.compile('[^A-Za-z\s]')
#     text = re.sub(pattern , "" , text)
#     words = []
#     for word in text.split():
#         if len(word) == 1 or word in sw:
#             continue
#         words.append(word)
#     return " ".join(words)

# def pipeline(file):
#     nltk.download('averaged_perceptron_tagger')
#     text = gen_text(file)
#     return extract_keywords(text).split()















import pickle as pk
import gensim
from collections import Counter
import PyPDF2
import nltk
from nltk.corpus import stopwords
import re
from nltk import pos_tag,word_tokenize



def match(resume , job_decription):
    model = pk.load(open('model.pkl' , 'rb'))
    resume = pipeline(resume)
    job_decription = pipeline(job_decription)
    nonmatched = []
    matched = []
    for key in job_decription:
        if key not in model.wv.key_to_index:
            continue
        match = False
        for word in resume:
            if word not in model.wv.key_to_index:
                continue
            if(model.wv.similarity(key , word) >= 0.75):
                match = True
        if not match:
            nonmatched.append(key)
        else:
            matched.append(key)
    matched_words = Counter(matched)
    nonmatched_words = Counter(nonmatched)
    match_score = 1
    nonmatch_score = 1
    for tuple in matched_words.items():
        match_score += tuple[1]
    for tuple in nonmatched_words.items():
        nonmatch_score += tuple[1]
    
    return round(100*((match_score/(match_score+nonmatch_score)) + 0.3) , 2) , matched_words.most_common(min(5 , len(matched_words))) , nonmatched_words.most_common(min(5 , len(nonmatched_words)))

def gen_text(pdf_file):

    pdf_reader = PyPDF2.PdfReader(pdf_file)

    text = ''

    try:
        for page_num in range(len(pdf_reader.pages)):

            page = pdf_reader.pages[page_num]

            text += ' ' + page.extract_text()
    except:
        text = ''
        
    # pdf_file.close()

    return text

def extract_keywords(document):
    text = preprocessing(document)
    words = word_tokenize(text)
    tagged = nltk.pos_tag(words)
    keywords =  [word for word,pos in tagged if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    return " ".join(keywords)

def preprocessing(text):
    nltk.download('stopwords')
    sw = set(stopwords.words('english'))
    text = text.replace('\n',' ').lower()
    pattern = re.compile('[^A-Za-z\s]')
    text = re.sub(pattern , "" , text)
    words = []
    for word in text.split():
        if len(word) == 1 or word in sw:
            continue
        words.append(word)
    return " ".join(words)

def pipeline(file):
    nltk.download('averaged_perceptron_tagger')
    text = gen_text(file)
    return extract_keywords(text).split()







