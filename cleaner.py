import re
import lxml.etree

import nltk
from nltk.corpus import stopwords

def parse_xml(file):
	xml_dict = dict()
	tree = lxml.etree.parse(file)

	try: xml_dict["text"] = tree.find("//articletitle").text.strip().lower()
	except: xml_dict["text"] = ""

	try: xml_dict["text"] += " ".join([x.text.strip().lower() for x in tree.findall("//abstracttext")])
	except: xml_dict["text"] += ""

	try: xml_dict["journal_title"] = tree.find("//journal/title").text.strip().lower()
	except: xml_dict["journal_title"] = ""

	try: xml_dict["country"] = tree.find("//country").text.strip().lower()
	except: xml_dict["country"] = ""

	try: xml_dict["language"] = tree.find("//language").text.strip().lower()
	except: xml_dict["language"] = ""

	return xml_dict

def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else '' for i in text])

def remove_punc(text):
    return re.sub(r'[.,:;!?)(-_*^&+=<>/|$%\'\@#\""]', ' ', text)

def remove_digits(text):
    return re.sub(r'[0-9]+', ' ', text)

def remove_whitespace(text):
	return re.sub(r'  +', ' ', text)

def remove_stop_words(text):
	return [word for word in nltk.word_tokenize(text) if word not in set(stopwords.words("english"))]

def stem_text(text):
	stemmer = nltk.PorterStemmer()
	return [stemmer.stem(word) for word in text]

TOTAL_CLEANERS = [
    remove_non_ascii,
    remove_punc,
    remove_digits,
    remove_whitespace
]

LEX_CLEANERS = [
    remove_stop_words,
    stem_text
]

def clean(text, cleaners=[]):
    for action in cleaners:
        text = action(text)

    return text

def total_clean(text):
    return clean(text, TOTAL_CLEANERS)

def lex_clean(text):
	return clean(text, LEX_CLEANERS)