import os
import glob
import timeit
import pickle
from pprint import pprint
from itertools import product
from collections import defaultdict

import artm
import nltk
import numpy as np
import pandas as pd

class TopicModel:
	def __init__(self, params, batch_vectorizer, dictionary):
		"""
		Args:
			*model_name (str)      : model name
		    *num_top_tokens (int)  : number of top words to track
		    *num_topics (int)      : number of topics
			*target_folder (str)   : folder to save/load batches
			*data_path (str)       : path to data source
	        *dictionary_path (str) : path to save/load dictionary
		    *class_ids (dict)      : describes all modalities - modality: weight
		    *main_modality (str)   : modality with text
		"""
		self.params = params

		self.topic_names = ["topic_" + str(i) for i in range(self.params["num_topics"] - 1)]
		self.topic_names.append("grb")
		self.batch_vectorizer = batch_vectorizer
		self.dictionary = dictionary

		self.initialize_model()


	def batch_dictionary(self):
		"""
		Create batches and dictionary from data source in vowpal wabbit format
		"""
		self.dictionary = artm.Dictionary(self.params["dictionary_path"])

		self.batch_vectorizer = None
		if len(glob.glob(os.path.join(self.params["target_folder"], "*.batch"))) < 1:
			self.batch_vectorizer = artm.BatchVectorizer(data_path = self.params["data_path"], 
			                                         target_folder = self.params["target_folder"],
                                                     data_format = "vowpal_wabbit", 
                                                     batch_size = 200)
		else:
		    self.batch_vectorizer = artm.BatchVectorizer(data_path = self.params["target_folder"], 
		    	                                         data_format = "batches")

		if not os.path.isfile(self.params["dictionary_path"]):
			self.dictionary.gather(data_path = self.batch_vectorizer.data_path)
			self.dictionary.save(dictionary_path = self.params["dictionary_path"])

		self.dictionary.load(dictionary_path = self.params["dictionary_path"])

		with open("bv.pickle", "w") as f: pickle.dump(self.batch_vectorizer, f)


	def initialize_model(self):
		"""
		Initialize model
		"""
		# self.batch_dictionary()

		self.model = artm.ARTM(num_topics = self.params["num_topics"], 
                               topic_names = self.topic_names,
                               class_ids = self.params["class_ids"],
                               num_document_passes = 1,
                               cache_theta = True)

		self.model.initialize(dictionary = self.dictionary)

		self.add_scores()
		self.add_regularizers()


	def add_scores(self):
		"""
		Add scores
		"""
		self.model.scores.add(artm.PerplexityScore(name = "perplexity_score", 
	                                               use_unigram_document_model = False, 
	                                               dictionary = self.dictionary))

		self.model.scores.add(artm.SparsityPhiScore(name = "sparsity_phi_score", 
	                                                class_id = self.params["main_modality"]))


		self.model.scores.add(artm.TopicKernelScore(name = "topic_kernel_score", 
	                                                class_id = self.params["main_modality"]))

		self.model.scores.add(artm.SparsityThetaScore(name = "sparsity_theta_score"))

		for cl in self.params["class_ids"]:
			self.model.scores.add(artm.TopTokensScore(name = "top_words_" + cl, 
	                                              num_tokens = self.params["num_top_tokens"], 
	                                              class_id = cl))


	def add_regularizers(self):
		"""
		Add regularizers
		"""  
		# --------------------------- first step ---------------------------------
		self.model.regularizers.add(artm.DecorrelatorPhiRegularizer(name = "decorrelate_phi",
	                                                               class_ids = [self.params["main_modality"]],
	                                                               tau = 0))

		self.model.regularizers.add(artm.SmoothSparsePhiRegularizer(name = "smooth_phi_grb", 
	                                                                class_ids = [self.params["main_modality"]], 
	                                                                topic_names = self.topic_names[-1],
	                                                                tau = 0))

		# --------------------------- second step ---------------------------------- 
		for cl in self.params["class_ids"]:
			self.model.regularizers.add(artm.SmoothSparsePhiRegularizer(name = "sparse_phi_sbj_" + cl, 
	                                                                class_ids = [cl],
	                                                                topic_names = self.topic_names[:-1],
	                                                                tau = 0))

		self.model.regularizers.add(artm.SmoothSparseThetaRegularizer(name = "sparse_theta",
	                                                                  topic_names = self.topic_names[:-1],
	                                                                  tau = 0))


	def set_regularization_coefficients(self, coefs):
		"""
		Set regularizers coefficients

		Args:
			*coefs (dict): regularizator_name: coefficient
		"""
		for c in coefs:
			self.model.regularizers[c].tau = coefs[c] 


	def all_top_words_combinations(self, top_words_content):
		"""
		Create dictionary of combination of all top words (for concrete topic)

		Args:
			*top_words_content (dict): top words for all topics

		Return:
			*top_words_combinations (dict): all combinations of top words
		"""
		top_words_combinations = dict()

		for topic in top_words_content:
			top_words = top_words_content[topic]
			for i in range(0, self.params["num_top_tokens"] - 1):
				for j in range(i + 1, self.params["num_top_tokens"]):
					w1, w2 = top_words[i], top_words[j]

					top_words_combinations[w1] = 0
					top_words_combinations[w2] = 0
					top_words_combinations[w1 + "|" + w2] = 0

		return top_words_combinations


	def combinations_count(self, top_words_combinations):
		"""
		Count number of coocurrences of all top words

		Args:
			*top_words_combinations (dict): all combinations of top words

		Return:
			*top_words_combinations (dict): all combinations of top words with counted values of coocurrence
		"""
		with open(self.params["data_path"]) as f:
			for line in f:
				temp = set(nltk.word_tokenize(line.split(" |" + self.params["main_modality"] + " ")[1].split(" |")[0]))

				for w in top_words_combinations:
					if "|" not in w:
						if w in temp: top_words_combinations[w] += 1
					else:
						w1, w2 = w.split("|")
						if (w1 in temp) and (w2 in temp): top_words_combinations[w] += 1

		return top_words_combinations


	def topic_coherence(self, sbj, top_words_content, top_words_combinations, n_documents):
		"""
		Compute coherence for particular topic 'sbj'

		Args:
			*sbj (str): topic for computing coherence
			*top_words_content (dict): top words for all topics
			*top_words_combinations (dict): all combinations of top words
			*n_documents (int): number of documents in data source

		Return:
			*topic_coherence (float): coherence for 'sbj' 
		"""
		top_words = top_words_content[sbj]
		k = len(top_words)

		topic_coherence = 0.0
		for i in range(0, k - 1):
			for j in range(i + 1, k):
				w1, w2 = top_words[i], top_words[j]

				n = top_words_combinations[w1 + "|" + w2]
				n1, n2 = top_words_combinations[w1], top_words_combinations[w2]

				topic_coherence += (n * n_documents + 1e-6) / (n1 * n2)

		topic_coherence = (2 * topic_coherence) / (k * (k - 1))
		return topic_coherence


	def coherence(self):
		"""
		Compute coherence for all topics

		Return:
			*median_cohernce (double): median coherence over all topics
			*collection_coherence (dict): coherence for every topic - topic_name: value
		"""
		top_words_content = self.model.score_tracker["top_words_" + self.params["main_modality"]].last_tokens

		top_words_combinations = self.all_top_words_combinations(top_words_content)
		top_words_combinations = self.combinations_count(top_words_combinations)

		self.n_documents = self.model.get_theta().shape[1]
		collection_coherence = defaultdict(str)

		for sbj in self.topic_names:
			collection_coherence[sbj] = self.topic_coherence(sbj, top_words_content, top_words_combinations, self.n_documents)

		median_coherence = np.median([x[1] for x in collection_coherence.items()])
		return median_coherence, collection_coherence


	def fit(self, num_collection_passes):
		"""
		Train model

		Args:
			*num_collection_passes(int): number of passes through collection
		"""
		self.model.fit_offline(batch_vectorizer = self.batch_vectorizer, num_collection_passes = num_collection_passes)


	def extract_matricies(self):
		"""
		Get matricies Phi (for each modality) and Theta
		"""
		self.matricies = dict()
		for cl in self.params["class_ids"]:
			self.matricies["phi_" + cl] = self.model.get_phi(class_ids = [cl])

		self.matricies["theta"] = self.model.get_theta()


	def modality_map(self, modality):
		"""
		Build data frame that matches each modality to topics

		Args:
			*modality (str): modality to build map for
		"""
		sbj_prob = self.matricies["theta"].sum(axis = 1)
		sbj_prob = (sbj_prob / sum(sbj_prob))

		sbj_prob_sort = (sbj_prob / sum(sbj_prob))
		sbj_prob_sort.sort(ascending = False)

		x1 = np.multiply(self.matricies["phi_" + modality], sbj_prob.values)
		x2 = 1 / x1.sum(axis = 1)

		x = np.multiply(x1, x1.sum(axis = 1)[:, None])
		# emotion_topic = x.values.argsort(axis = 1)
		return x


	def timeline_map(self):
		"""
		Build map for timeline
		"""
		date_document = defaultdict(list)

		with open("data/vw_emails_anthony", "r") as f:
		    for i, line in enumerate(f):
		        date = line.split("|date ")[1].split("|")[0]
		        date_document[date].append(i)


		index = [date for date in date_document] 
		index.sort(key = lambda x: (int(x.split("_")[0]) + int(x.split("_")[1]) / 100.0))

		theme_prob_by_time = pd.DataFrame(index = index, columns = self.topic_names, data = 0.0)
		time_prob_by_theme = pd.DataFrame(index = index, columns = self.topic_names, data = 0.0)

		# text_dicts = read_pickle("data/text_dicts")
		with open("data/text_dicts") as f: text_dicts = pickle.load(f)
		
		f = pd.DataFrame(index = self.matricies["phi_" + self.params["main_modality"]].index, 
			             columns = self.matricies["theta"].columns, 
			             data = np.dot(self.matricies["phi_" + self.params["main_modality"]].values, self.matricies["theta"].values))

		f = f / f.sum(axis = 0).astype(float)
		words = list(f.index)

		for d in f.columns:
		    ndw = [text_dicts[d][word] for word in words]
		    f.ix[:, d] = f.ix[:, d] * ndw
		    
		f = f.sum(axis = 0)
		# print f

		for date in date_document:
		#     time_prob_by_theme.ix[date, :] = theta.ix[:, date_document[date]].sum(axis = 1) / f.values
		    theme_prob_by_time.ix[date, :] = (self.matricies["theta"].ix[:, date_document[date]].sum(axis = 1) / len(date_document[date])).values

		print theme_prob_by_time["topic_5"]


	def scores_values(self):
		"""
		Get all the scores

		Return:
			* (dict): dict with scores and their values
		"""
		# temp = self.coherence()
		return {
			"sparsity_phi"     : self.model.score_tracker["sparsity_phi_score"].last_value,
			"sparsity_theta"   : self.model.score_tracker["sparsity_theta_score"].last_value,
			"kernel_contrast"  : self.model.score_tracker["topic_kernel_score"].last_average_contrast,
			"kernel_purity"    : self.model.score_tracker["topic_kernel_score"].last_average_purity,
			"perplexity"       : self.model.score_tracker["perplexity_score"].last_value,
			# "coherence"        : temp[0],
			# "coherence_classes": temp[1]
		}

	def top_words(self):
		"""
		Extarct top words from score_tracker
		"""
		self.top_words_dict = dict()

		for cl in self.params["class_ids"]:
			self.top_words_dict[cl] = self.model.score_tracker["top_words_" + cl].last_tokens


	def print_scores(self):
		"""
		Print scores and their values
		"""
		temp = self.scores_values()
		for score in temp:
			if score != "coherence_classes":
				print "%30s: %.3f" %(score, temp[score])

	            
	def print_top_words(self, top_words_classes):
		"""
		Print topics and their top words

		Args:
			*top_words_classes (list): modalities for which print top words 
		"""
		try:
			self.top_words_dict
		except:
			self.top_words()

		for cl in top_words_classes:
			print "================== %s ====================\n"
			for topic_name in self.model.topic_names:
				print topic_name, ":",
				for word in self.top_words_dict[cl][topic_name]: print word, "|",
				print "\n"


	def __str__(self):
		"""
		Print fun string
		"""
		return "super cool model"


if __name__ == "__main__":

	params = {
				"model_name"      : "For Anthony",
		        "num_top_tokens"  : 10,
		        "num_topics"      : 30,
			    "target_folder"   : "email_batches",
				"data_path"       : "data/vw_emails_anthony",
				# "data_path"       : "data/vw_small",
	            "dictionary_path" : "email_batches/dictionary.dict",
		        "class_ids"       : {"bicontent": 1, "emotion": 1, "date": 1},
		        "main_modality"   : "bicontent"
	         }

	optimal_reg_first = {
					"smooth_phi_grb"          : 100.0,
					"decorrelate_phi"         : 100000.0
				}

	optimal_reg_second = {					
	                       "sparse_phi_sbj_bicontent" : 0,
					       "sparse_phi_sbj_emotion"   : -10.0,
					       "sparse_phi_sbj_date"      : -1.0,
					       "sparse_theta"             : -1.0
					    }

	model = TopicModel(params)

	model.set_regularization_coefficients(optimal_reg_first)
	model.fit(num_collection_passes=1)

	model.set_regularization_coefficients(optimal_reg_second)
	model.fit(num_collection_passes=1)

	model.top_words()
	model.extract_matricies()

	# model.modality_map("emotion")
	model.timeline_map()

	# model.print_scores()
	# model.print_top_words(["bicontent"])

	# paramSearch = RegularizationParametersSearch(model)
	
	# paramSearch.grid_search()
	# paramSearch.reg_wise_search()

	# print paramSearch
	# print paramSearch


