####### REGEX #############
'''
my_string 
Out[8]: "Let's write RegEx!  Won't that be fun?  I sure think so.  Can you find 4 sentences?  Or perhaps, all 19 words?"
'''

# Import the regex module
import re

# Write a pattern to match sentence endings: sentence_endings
sentence_endings = r"[.?!]"

# Split my_string on sentence endings and print the result
print(re.split(sentence_endings, my_string))

# Find all capitalized words in my_string and print the result
capitalized_words = r"[A-Z]\w+"
print(re.findall(capitalized_words, my_string))

# Split my_string on spaces and print the result
spaces = r"\s+"
print(re.split(spaces, my_string))

# Find all digits in my_string and print the result
digits = r"\d+"
print(re.findall(digits, my_string))
--------------------------------------------------------------
##### Tokenization #####
'''
scene_one (text_data)
"SCENE 1: [wind] [clop clop clop] \nKING ARTHUR: 
Whoa there!  [clop clop clop] \nSOLDIER #1: Halt!  
Who goes there?\nARTHUR: It is I, Arthur, son of 
Uther Pendragon, from the castle of Camelot.  King of the Britons, d
efeator of the Saxons, sovereign of all England!\nSOLDIER 
#1: Pull the other one!\nARTHUR: I am, ...  and this is my trusty servant Patsy. 
 We have ridden the length and breadth of the land in search of knights who will join me in my court at Camelot. 
 I must speak with your lord and master.\nSOLDIER #1: What?  Ridden on a horse?\nARTHUR: Yes!\nSOLDIER 
 #1: You're using coconuts!\nARTHUR: What?\nSOLDIER #1: You've got two empty halves of coconut and you're bangin' 'em together.\nARTHUR: So?  
 We have ridden since the snows of winter covered this land, through the kingdom of Mercea, through--\nSOLDIER
 #1: Where'd you get the coconuts?\nARTHUR: We found them.\nSOLDIER #1: Found them?  In Mercea?  The coconut's tropical!\nARTHUR: 
 What do you mean?\nSOLDIER #1: Well, this is a temperate zone.\nARTHUR: The swallow may fly south with the sun or the house martin or the plover may seek warmer climes in winter, yet these are not strangers to our land?\nSOLDIER #1: Are you suggesting coconuts migrate?\nARTHUR: Not at all.  They could be carried.\nSOLDIER #1: What?  A swallow carrying a coconut?\nARTHUR: It could grip it by the husk!\nSOLDIER #1: It's not a question of where he grips it!  It's a simple question of weight ratios!  A five ounce bird could not carry a one pound coconut.\nARTHUR: Well, it doesn't matter.  Will you go and tell your master that Arthur from the Court of Camelot is here.\nSOLDIER #1: Listen.  In order to maintain air-speed velocity, a swallow needs to beat its wings forty-three times every second, right?\nARTHUR: Please!\nSOLDIER #1: Am I right?\nARTHUR: I'm not interested!\nSOLDIER #2: It could be carried by an African swallow!\nSOLDIER #1: Oh, yeah, an African swallow maybe, but not a European swallow.  That's my point.\nSOLDIER #2: Oh, yeah, I agree with that.\nARTHUR: Will you ask your master if he wants to join my court at Camelot?!\nSOLDIER #1: But then of course a-- African swallows are non-migratory.\nSOLDIER #2: Oh, yeah...\nSOLDIER #1: So they couldn't bring a coconut back anyway...  [clop clop clop] \nSOLDIER #2: Wait a minute!  Supposing two swallows carried it together?\nSOLDIER #1: No, they'd have to have it on a line.\nSOLDIER #2: Well, simple!  They'd just use a strand of creeper!\nSOLDIER #1: What, held under the dorsal guiding feathers?\nSOLDIER #2: Well, why not?\n"

'''
# Import necessary modules
from nltk.tokenize import sent_tokenize,word_tokenize,unique_tokens

# Split scene_one into sentences: sentences
sentences = sent_tokenize(scene_one) ### all text into sentences 

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3]) ### breaking 4th sentences into word 

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(scene_one))

# Print the unique tokens result
print(unique_tokens)
----------------------------------------------------
### More Regex Patterns #########################
# Search for the first occurrence of "coconuts" in scene_one: match
match = re.search("coconuts", scene_one)

# Print the start and end indexes of match
print(match.start(), match.end())

# Write a regular expression to search for anything in square brackets: pattern1
pattern1 = r"\[.*\]"

# Use re.search to find the first text in square brackets
print(re.search(pattern1, scene_one))

# Find the script notation at the beginning of the fourth sentence and print it
pattern2 = r"[\w\s]+:"
print(re.match(pattern2, sentences[3]))
-----------------------------------------------------
#How to find #,@ from a tweeter data (Very Important)
----------------------------------------------------
'''
tweets:
['This is the best #nlp exercise ive found online! #python',
 '#NLP is super fun! <3 #learning',
 'Thanks @datacamp :) #nlp #python']
 '''
# Import the necessary modules
from nltk.tokenize import regexp_tokenize,TweetTokenizer
import regex as re

# Define a regex pattern to find hashtags: pattern1
pattern1 = r"#\w+" 

# Use the pattern on the first tweet in the tweets list
regexp_tokenize(tweets[0],pattern1)

# Write a pattern that matches both mentions and hashtags
pattern2 = r"([@|#]\w+)" # @-pattern, #-pattern ,|-or, \w-lettersearch,\w+-greedy search 

# Use the pattern on the last tweet in the tweets list
regexp_tokenize(tweets[-1],pattern2)

# Use the TweetTokenizer to tokenize all tweets into one list
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(t) for t in tweets]
print(all_tokens)
------------------------------
Chart your NLP 
------------------------------
# Split the script into lines: lines
lines = holy_grail.split('\n')

# Replace all script lines for speaker
pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
lines = [re.sub(pattern, '', l) for l in lines]

# Tokenize each line: tokenized_lines
tokenized_lines = [regexp_tokenize(s, "\w+") for s in lines]

# Make a frequency list of lengths: line_num_words
line_num_words = [len(t_line) for t_line in tokenized_lines]

# Plot a histogram of the line lengths
plt.hist(line_num_words)

# Show the plot
plt.show()
-------------------------------------------

####### Bag of Words ######################

# Import Counter
from collections import Counter 

# Tokenize the article: tokens
tokens = word_tokenize(article)

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t.lower() for t in tokens]

# Create a Counter with the lowercase tokens: bow_simple
bow_simple = Counter(lower_tokens)

# Print the 10 most common tokens
print(bow_simple.most_common(10))
-------------------------------------------------
######### Text Processing (craeting a BOW from scratch) ################
# Import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer

# Tokenize the article: tokens
tokens = word_tokenize(article)

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t.lower() for t in tokens]


# Retain alphabetic words: alpha_only
alpha_only = [t for t in lower_tokens if t.isalpha()] 

# Remove all stop words: no_stops
no_stops = [t for t in alpha_only if t not in english_stops]

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

# Create the bag-of-words: bow
bow = Counter(lemmatized)

# Print the 10 most common tokens
print(bow.most_common(10))
--------------------------------------------------------------
############# Simple Topic Identification #####################

# Import Dictionary
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize

# Create a Dictionary from the articles: dictionary
tokenized_doc = [word_tokenize(t.lower()) for t in articles]
dictionary = Dictionary(articles)

# Select the id for "computer": computer_id
computer_id =dictionary.token2id.get("computer")

# Use computer_id with the dictionary to print the word
print(dictionary.get(computer_id))

# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]
#corpus = [dictionary.doc2bow(article) for article in articles]
#Print the first 10 word ids with their frequency counts from the fifth document
print(corpus[4][:10])
-----------------------------------------------
# Save the fifth document: doc
doc = corpus[4]

# Sort the doc for frequency: bow_doc
bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

# Print the top 5 words of the document alongside the count
for word_id, word_count in bow_doc[:5]:
    print(dictionary.get(word_id), word_count)
    
# Create the defaultdict: total_word_count
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count

# Create a sorted list from the defaultdict: sorted_word_count 
sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True) 

# Print the top 5 words across all documents alongside the count
for word_id, word_count in sorted_word_count[:5]:
    print(dictionary.get(word_id), word_count)
------------------------------------------------------------------------
######## TF-IDF ##########
# Import TfidfModel
from gensim.models.tfidfmodel import TfidfModel

# Create a new TfidfModel using the corpus: tfidf
tfidf = TfidfModel(corpus)

# Calculate the tfidf weights of doc: tfidf_weights
tfidf_weights = tfidf[doc]

# Print the first five weights
print(tfidf_weights[:5])

# Sort the weights from highest to lowest: sorted_tfidf_weights
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

# Print the top 5 weighted words
for term_id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(term_id), weight)
-----------------------------------------------------------------------------
# NER 
-----------------------------------------------------------------------------
# Tokenize the article into sentences: sentences
sentences = nltk.sent_tokenize(article)

# Tokenize each sentence into words: token_sentences
token_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences] 

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

# Test for stems of the tree with 'NE' tags
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == "NE":
            print(chunk)
------------------------------------------------------------------------------
#CHARTING LABEL
------------------------------------------------------------------------------
'''
Charting practice
In this exercise, you'll use some extracted named entities and their groupings from a series of newspaper articles to chart the diversity of named entity types in the articles.

You'll use a defaultdict called ner_categories, with keys representing every named entity group type, and values to count the number of each different named entity type. You have a chunked sentence list called chunked_sentences similar to the last exercise, but this time with non-binary category names.

You can use hasattr() to determine if each chunk has a 'label' and then simply use the chunk's .label() method as the dictionary key.
'''
# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1
            
# Create a list from the dictionary keys for the chart labels: labels
labels = list(ner_categories.keys())

# Create a list of the values: values
values = [ner_categories.get(l) for l in labels]

# Create the pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

# Display the chart
plt.show()
#------------------------------------------------------------------
#SPACY-NAMED ENTITY 
#------------------------------------------------------------------
'''
In [11]: article
Out[11]: '\ufeffThe taxi-hailing company Uber brings into very sharp focus the question of 
whether corporations can be said to have a moral character. 
If any human being were to behave with the single-minded and ruthless greed of the company,
 we would consider them sociopathic. Uber wanted to know as much as 
 possible about the people who use its service, and those who don’t. It has an arrangement with unroll.me, 
 a company which offered a free service for unsubscribing from junk mail, to buy the contacts unroll.
 me customers had had with rival taxi companies. 
 Even if their email was notionally anonymised, this use of it was not something the users had bargained for. 
 Beyond that, it keeps track of the phones that have been used to summon its services even after the original 
 owner has sold them, attempting this with Apple’s phones even thought it is forbidden by the company.
 \r\n\r\n\r\nUber has also tweaked its software so that regulatory agencies that the company regarded as hostile would, 
 when they tried to hire a driver, be given false reports about the location of its cars. Uber management booked and then cancelled rides with a rival taxi-hailing company which took their vehicles out of circulation. Uber deny this was the intention. The punishment for this behaviour was negligible. Uber promised not to use this “greyball” software against law enforcement – one wonders what would happen to someone carrying a knife who promised never to stab a policeman with it. Travis Kalanick of Uber got a personal dressing down from Tim Cook, who runs Apple, but the company did not prohibit the use of the app. Too much money was at stake for that.\r\n\r\n\r\nMillions of people around the world value the cheapness and convenience of Uber’s rides too much to care about the lack of drivers’ rights or pay. Many of the users themselves are not much richer than the drivers. The “sharing economy” encourages the insecure and exploited to exploit others equally insecure to the profit of a tiny clique of billionaires. Silicon Valley’s culture seems hostile to humane and democratic values. The outgoing CEO of Yahoo, Marissa Mayer, who is widely judged to have been a failure, is likely to get a $186m payout. This may not be a cause for panic, any more than the previous hero worship should have been a cause for euphoria. Yet there’s an urgent political task to tame these companies, to ensure they are punished when they break the law, that they pay their taxes fairly and that they behave responsibly.
'''
# Import spacy
import spacy

# Instantiate the English model: nlp
nlp = spacy.load('en', tagger=False, parser=False, matcher=False)

# Create a new document: doc
doc = nlp(article)

# Print all of the found entities and their labels
for ent in doc.ents:
    print(ent.label_, ent.text)
'''
ORG Uber
ORG Uber
ORG Apple
ORG Uber
ORG Uber
PERSON Travis Kalanick
ORG Uber
PERSON Tim Cook
ORG Apple
CARDINAL Millions
ORG Uber
GPE drivers’
LOC Silicon Valley’s
ORG Yahoo
PERSON Marissa Mayer
MONEY $186m
'''
#----------------------------------------------------------------------
#PLYGOT LIBRARY -MULTILINGUAL TEXT PROCESSING (NAME ENTITY RECOGNITION)
#-----------------------------------------------------------------------
'''
In [4]: article
Out[4]: "\ufeffédition abonné\r\n\r\n\r\nDans une tribune au « Monde », l’universitaire Charles Cuvelliez estime que le fantasme d’un remplacement de l’homme par l’algorithme et le robot repose sur un malentendu.\r\n\r\n\r\nLe Monde | 10.05.2017 à 06h44 • Mis à jour le 10.05.2017 à 09h47 | Par Charles Cuvelliez (Professeur à l’Ecole polytechnique de l'université libre de Bruxelles)\r\n\r\n\r\nTRIBUNE. L’usage morbide, par certains, de Facebook Live a amené son fondateur à annoncer précipitamment le recrutement de 3 000 modérateurs supplémentaires. Il est vrai que l’intelligence artificielle (IA) est bien en peine de reconnaître des contenus violents, surtout diffusés en direct.\r\n\r\n\r\nLe quotidien affreux de ces modérateurs, contraints de visionner des horreurs à longueur de journée, mériterait pourtant qu’on les remplace vite par des machines !\r\n\r\n\r\nL’IA ne peut pas tout, mais là où elle peut beaucoup, on la maudit, accusée de détruire nos emplois, de remplacer la convivialité humaine. Ce débat repose sur un malentendu.\r\n\r\n\r\nIl vient d’une définition de l’IA qui n’a, dans la réalité, jamais pu être mise en pratique : en 1955, elle était vue comme la création de programmes informatiques qui, quoi qu’on leur confie, le feraient un jour mieux que les humains. On pensait que toute caractéristique de l’intelligence humaine pourrait un jour être si précisément décrite qu’il suffirait d’une machine pour la simuler. Ce n’est pas vrai.\r\n\r\n\r\nAngoisses infondées\r\n\r\n\r\nComme le dit un récent Livre blanc sur la question (Pourquoi il ne faut pas avoir peur de l’Intelligence arti\xadficielle, Julien Maldonato, Deloitte, mars 2017), rien ne pourra remplacer un humain dans sa globalité.\r\n\r\n\r\nL’IA, c’est de l’apprentissage automatique doté d’un processus d’ajustement de modèles statistiques à des masses de données, explique l’auteur. Il s’agit d’un apprentissage sur des paramètres pour lesquels une vision humaine n’explique pas pourquoi ils marchent si bien dans un contexte donné.\r\n\r\n\r\nC’est aussi ce que dit le rapport de l’Office parlementaire d’évaluation des choix scientifiques et technologiques (« Pour une intelligence artificielle maîtrisée, utile et démystifiée », 29 mars 2017), pour qui ce côté « boîte noire » explique des angoisses infondées. Ethiquement, se fonder sur l’IA pour des tâches critiques sans bien comprendre le comment..."
'''
# Create a new text object using Polyglot's Text class: txt
txt = Text(article)

# Print each of the entities found
for ent in txt.entities:
    print(ent)
    
# Print the type of each entity
print(type(ent))
'''
<script.py> output:
    ['Charles', 'Cuvelliez']
    ['Charles', 'Cuvelliez']
    ['Bruxelles']
    ['l’IA']
    ['Julien', 'Maldonato']
    ['Deloitte']
    ['Ethiquement']
    ['l’IA']
    ['.']
    <class 'polyglot.text.Chunk'>
'''

#----------------------------------------------------------------------
#PRINT THE ENTITY NAME WITH ENTITY (ex. ('I-PER', 'Charles Cuvelliez'))
#----------------------------------------------------------------------

# Create the list of tuples: entities
entities = [(ent.tag, ' '.join(ent)) for ent in txt.entities]

# Print the entities
print(entities)

#------------------------------------------------------------------------
#SPANISH PLOYGOT (NAME ENTITY RECOGNITION)
#------------------------------------------------------------------------

'''
txt:
In [6]: txt
Out[6]: 
Text("﻿Lina del Castillo es profesora en el Instituto de Estudios Latinoamericanos Teresa Lozano Long (LLILAS) y el Departamento de Historia de la Universidad de Texas en Austin. Ella será la moderadora del panel “Los Mundos Políticos de Gabriel García Márquez” este viernes, Oct. 30, en el simposio Gabriel García Márquez: Vida y Legado.


LIna del Castillo


Actualmente, sus investigaciones abarcan la intersección de cartografía, disputas a las demandas de tierra y recursos, y la formación del n...el tren de medianoche que lleva a miles y miles de cadáveres uno encima del otro como tantos racimos del banano que acabarán tirados al mar. Ningún recuento periodístico podría provocar nuestra imaginación y nuestra memoria como este relato de García Márquez.


Contenido Relacionado


Lea más artículos sobre el archivo de Gabriel García Márquez


Reciba mensualmente las últimas noticias e información del Harry Ransom Center con eNews, nuestro correo electrónico mensual. ¡Suscríbase hoy!")

'''
# Initialize the count variable: count
count = 0

# Iterate over all the entities
for ent in txt.entities:
    # Check whether the entity contains 'Márquez' or 'Gabo'
    if "Márquez" in ent or "Gabo" in ent:
        # Increment count
        count += 1

# Print count
print(count)

# Calculate the percentage of entities that refer to "Gabo": percentage
percentage = count * 1.0 / len(txt.entities)
print(percentage)

#################################
# FAKE NEWS CLASSIFIER PROJECT
#################################

# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Print the head of df
print(df.head())

# Create a series to store the labels: y
y = df.label

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])

#--------- TfIDf VECTOR ----------------
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])

'''

<script.py> output:
    ['000', '000ft', '000km', '003', '01', '02', '027', '033', '04', '05']
    [[0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.03834063 0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]]

In [1]: 
'''
#--Inspecting the vectors---
'''
To get a better idea of how the vectors work, you'll investigate them by converting them into pandas DataFrames.

Here, you'll use the same data structures you created in the previous two exercises (count_train, count_vectorizer, tfidf_train, tfidf_vectorizer) as well as pandas, which is imported as pd.
'''
# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))
'''
<script.py> output:
       000  10  100  11  114th  11th  12  120  13  133    ...     œexplosiveâ  \
    0    0   0    0   0      0     0   0    0   0    0    ...               0   
    1    0   0    0   0      0     0   0    0   0    0    ...               0   
    2    0   1    0   0      0     0   0    0   1    0    ...               0   
    3    0   1    0   0      0     0   0    0   0    0    ...               0   
    4    0   0    0   0      0     0   0    0   0    0    ...               0   
'''


#--------------------------------------------------
# TRAIN THE MODEL with COUNTVECTORIZER DATAFRAME 
#-------------------------------------------------
# Import the necessary modules
from sklearn import metrics 
from sklearn.naive_bayes import MultinomialNB

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train,y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test,pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test,pred,labels=['FAKE','REAL'])
print(cm)
'''
<script.py> output:
    0.893352462936394
    [[ 865  143]
     [  80 1003]]
'''

#-------------------------------------------
#TRAIN WITH TFIDFVECTORIZER 
#-------------------------------------------
# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)
'''
0.8565279770444764
[[ 739  269]
 [  31 1052]]
 '''
#--------------------------------------------------
#MULTINOMINAL WITH VARIOUS ALPHAS 
#--------------------------------------------------

# Create the list of alphas: alphas
alphas = np.arange(0,1,0.1)

# Define train_and_predict()
def train_and_predict(alpha=alphas):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB()
    # Fit to the training data
    nb_classifier.fit(tfidf_train,y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    # Compute accuracy: score
    score = metrics.accuracy_score(pred,y_test)
    return score

# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()
	
'''
Score:  0.8857006217120995
    
    Alpha:  0.5
    Score:  0.8842659014825442
    
    Alpha:  0.6000000000000001
    Score:  0.874701099952176
    
    Alpha:  0.7000000000000001
    Score:  0.8703969392635102
    
    Alpha:  0.8
    Score:  0.8660927785748446
    
    Alpha:  0.9
    Score:  0.8589191774270684

'''
#---------- Feature Extraction -------------
# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])
'''
FAKE [(-12.641778440826338, '0000'), (-12.641778440826338, '000035'), (-12.641778440826338, '0001'), (-12.641778440826338, '0001pt'), (-12.641778440826338, '000km'), (-12.641778440826338, '0011'), (-12.641778440826338, '006s'), (-12.641778440826338, '007'), (-12.641778440826338, '007s'), (-12.641778440826338, '008s'), (-12.641778440826338, '0099'), (-12.641778440826338, '00am'), (-12.641778440826338, '00p'), (-12.641778440826338, '00pm'), (-12.641778440826338, '014'), (-12.641778440826338, '015'), (-12.641778440826338, '018'), (-12.641778440826338, '01am'), (-12.641778440826338, '020'), (-12.641778440826338, '023')]
REAL [(-6.790929954967984, 'states'), (-6.765360557845786, 'rubio'), (-6.751044290367751, 'voters'), (-6.701050756752027, 'house'), (-6.695547793099875, 'republicans'), (-6.6701912490429685, 'bush'), (-6.661945235816139, 'percent'), (-6.589623788689862, 'people'), (-6.559670340096453, 'new'), (-6.489892292073901, 'party'), (-6.452319082422527, 'cruz'), (-6.452076515575875, 'state'), (-6.397696648238072, 'republican'), (-6.376343060363355, 'campaign'), (-6.324397735392007, 'president'), (-6.2546017970213645, 'sanders'), (-6.144621899738043, 'obama'), (-5.756817248152807, 'clinton'), (-5.596085785733112, 'said'), (-5.357523914504495, 'trump')]
'''
#------------------------------------------------
















