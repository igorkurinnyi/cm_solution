import os
import glob
import cleaner
from collections import defaultdict

import re
import nltk
import artm
import luigi
import pickle
from gensim.models import Word2Vec

import urllib
from bs4 import BeautifulSoup

import firstTask

class ScrapeMedicalTerms(luigi.ExternalTask):
	result_folder = luigi.Parameter("second_task") 

	def run(self):
		medical_terms = set()
		alphabet = "abcdefghijklmnopqrstuvwxyz"

		for letter in alphabet:
			url = "http://users.ugent.be/~rvdstich/eugloss/EN/lijst" + letter + ".html"
			html_text = urllib.urlopen(url).read()
			soup = BeautifulSoup(html_text)

			for row in soup.find_all("li"):
				try: 
					term = row.find("b").contents[0].strip()
					term = term.split(" ")[0].split("\n")[0]
					medical_terms.add(term)
				except: 
					pass

		# terms in 'leftover list'
		medical_terms.add("enterocolitis")
		medical_terms.add("catheter")

		if not os.path.exists(self.result_folder):
			os.makedirs(self.result_folder)

		with self.output().open("w") as f: pickle.dump(medical_terms, f)


	def output(self):
		return luigi.LocalTarget(self.result_folder + "/medical_terms.pickle")


class FindMedicalTerms(luigi.Task):
	data_folder = luigi.Parameter(default="abstracts")
	fisrt_folder = luigi.Parameter(default="first_task")
	second_folder = luigi.Parameter(default="second_task")

	def requires(self):
		return [firstTask.InputText(self.data_folder, self.fisrt_folder),
		        ScrapeMedicalTerms(self.second_folder)]

	def run(self):
		with self.input()[0][1].open("r") as f: papers = pickle.load(f)
		with self.input()[1].open("r") as f: medical_terms = pickle.load(f)

		stemmer = nltk.PorterStemmer()
		medical_terms = set(stemmer.stem(term) for term in medical_terms)
		
		present_medical_terms = list()
		for paper in papers:
			temp = [word[0] for word in paper["text"] if word[0] in medical_terms]
			present_medical_terms.extend(temp)

		present_medical_terms = set(present_medical_terms)
		with self.output().open("w") as f: pickle.dump(present_medical_terms, f)

	def output(self):
		return luigi.LocalTarget(self.second_folder + "/present_medical_terms.pickle")


class GetSentences(luigi.ExternalTask):
	data_folder = luigi.Parameter(default="abstracts")
	result_folder = luigi.Parameter(default="second_task")

	def run(self):
		files = glob.glob(self.data_folder + "/*.xml")

		all_sentences = list()
		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

		for file in files:
			paper = cleaner.parse_xml(file)

			sentences = tokenizer.tokenize(paper["text"]) #split by sentences
			sentences = [nltk.word_tokenize(cleaner.total_clean(x)) for x in sentences]
			all_sentences.extend(sentences)

		with self.output().open("w") as f: pickle.dump(all_sentences, f)

	def output(self):
		return luigi.LocalTarget(self.result_folder + "/sentences.pickle")


class TrainWordVecModel(luigi.Task):
	data_folder = luigi.Parameter(default="abstracts")
	result_folder = luigi.Parameter(default="second_task")

	def requires(self):
		return GetSentences(self.data_folder, self.result_folder)

	def run(self):
		with self.input().open("r") as f: sentences = pickle.load(f)

		num_features = 300                      
		min_word_count = 1                        
		num_workers = 4
		context = 10                                                                                
		downsampling = 1e-3

		model = Word2Vec(sentences, workers=num_workers,
			                      size=num_features, min_count=min_word_count,
                                  window=context, sample=downsampling)
		model.init_sims(replace=True)
		model.save(self.output().path)

	def output(self):
		return luigi.LocalTarget(self.result_folder + "/word2vec_model.pickle")


class FindSynonyms(luigi.Task):
	data_folder = luigi.Parameter(default="abstracts")
	fisrt_folder = luigi.Parameter(default="first_task")
	second_folder = luigi.Parameter(default="second_task")

	def requires(self):
		return [FindMedicalTerms(self.data_folder, self.fisrt_folder, self.second_folder),
		        TrainWordVecModel(self.data_folder, self.second_folder)]

	def run(self):
		with self.input()[0].open("r") as f: present_medical_terms = pickle.load(f)
		model = Word2Vec.load(self.input()[1].path)
		
		synonyms = defaultdict(list)
		for term in present_medical_terms:
			try: 
				synonyms[term].extend([x[0] for x in model.most_similar(term) if x[0] in present_medical_terms])
			except: pass

		with self.output()[0].open("w") as f: pickle.dump(synonyms, f)		

	def output(self):
		return luigi.LocalTarget(self.second_folder + "/synonyms.pickle")


if __name__ == "__main__":
	luigi.run()