import os
import glob
import cleaner
from collections import defaultdict

import re
import nltk
import artm
import luigi
import pickle
from topicModel import TopicModel

	
class InputText(luigi.ExternalTask):
	data_folder = luigi.Parameter(default="abstracts")
	result_folder = luigi.Parameter(default="first_task")

	def run(self):
		files = glob.glob(self.data_folder + "/*.xml")
		papers = list()

		simple_stats = {"countries" :defaultdict(int), 
		                "languages" :defaultdict(int), 
		                "journals"  :defaultdict(int), 
		                "pos"       :{"nouns"      :defaultdict(int), 
		                              "adjectives" :defaultdict(int), 
		                              "adverbs"    :defaultdict(int)}}

		i = 0
		for file in files:
			paper = cleaner.parse_xml(file)
			paper["journal_title"] = cleaner.remove_non_ascii(paper["journal_title"])

			simple_stats["countries"][paper["country"]] += 1
			simple_stats["languages"][paper["language"]] += 1
			simple_stats["journals"][paper["journal_title"]] += 1

			paper["text"] = cleaner.total_clean(paper["text"])
			paper["text"] = cleaner.lex_clean(paper["text"])
			paper["text"] = nltk.pos_tag(paper["text"])

			paper["journal_title"] = cleaner.remove_punc(paper["journal_title"]) #required for topics 
			papers.append(paper)

			for x in paper["text"]:
				if x[1] in ("NN", "NNS", "NNP", "NNPS"):
					simple_stats["pos"]["nouns"][x[0]] += 1
				elif x[1] in ("JJ", "JJR", "JJS"):
					simple_stats["pos"]["adjectives"][x[0]] += 1
				elif x[1] in ("RB", "RBS", "RBR"):
					simple_stats["pos"]["adverbs"][x[0]] += 1

			i += 1
			if i % 100 == 0: print i

		if not os.path.exists(self.result_folder):
			os.makedirs(self.result_folder)

		with self.output()[0].open("w") as f: pickle.dump(simple_stats, f)
		with self.output()[1].open("w") as f: pickle.dump(papers, f)


	def output(self):
		return [luigi.LocalTarget(self.result_folder + "/simple_stats.pickle"),
		        luigi.LocalTarget(self.result_folder + "/processed_papers.pickle")]


class PrepareVWFormat(luigi.Task):
	data_folder = luigi.Parameter(default="abstracts")
	result_folder = luigi.Parameter(default="first_task")
	
	def requires(self):
		return InputText(self.data_folder, self.result_folder)

	def run(self):
		with self.input()[1].open("r") as f: papers = pickle.load(f)

		with self.output().open("w") as f:
			for paper in papers:
				paper["text"] = [x[0] for x in paper["text"] if len(x[0]) > 2]
				paper["text"] = ["_".join(x) for x in zip(*[paper["text"][i: ] for i in range(2)])] #bigrams

				line = " |text " + " ".join(x for x in paper["text"]) + \
				       " |journal " + "_".join(paper["journal_title"].split(" ")) + \
				       " |country " + paper["country"] + \
				       " |language " + paper["language"] + \
				       "\n"
				f.write(line.encode("utf-8"))

	def output(self):
		return luigi.LocalTarget(self.result_folder + "/vw_papers")



class BatchDictionary(luigi.Task):
	data_folder = luigi.Parameter(default="abstracts")
	result_folder = luigi.Parameter(default="first_task")
	batches = luigi.Parameter(default="batches")

	def requires(self):
		return PrepareVWFormat(self.data_folder, self.result_folder)

	def run(self):
		self.batches = self.result_folder  + "/" + self.batches
		batch_vectorizer = artm.BatchVectorizer(data_path = self.input().path, 
			                                    target_folder = self.batches,
                                                data_format = "vowpal_wabbit", 
                                                batch_size = 200)

		dictionary = artm.Dictionary()
		dictionary.gather(data_path=batch_vectorizer.data_path)

		if not os.path.exists(self.batches):
		    os.makedirs(self.batches)

		with self.output()[0].open("w") as f: pickle.dump(batch_vectorizer, f)
		if os.path.isfile(self.output()[1].path): os.remove(self.output()[1].path)
		dictionary.save(dictionary_path=self.output()[1].path)

	def output(self):
		return [luigi.LocalTarget(self.batches + "/batch_vectorizer.pickle"),
		        luigi.LocalTarget(self.batches + "/dictionary.dict")]


class TopicModel(luigi.Task):
	data_folder = luigi.Parameter(default="abstracts")
	result_folder = luigi.Parameter(default="first_task")
	batches = luigi.Parameter(default="batches")

	params = luigi.DictParameter()

	def requires(self):
		return BatchDictionary(self.data_folder, self.result_folder, self.batches)

	def run(self):
		self.params = dict(self.params)
		self.params["data_path"] = self.result_folder + "/vw_papers"
		self.params["class_ids"] = {self.params["main_modality"] :1, 
		                            "journal"                    :1, 
		                            "country"                    :1, 
		                            "language"                   :1}

		self.batches = self.result_folder  + "/" + self.batches

		dictionary = artm.Dictionary()
		dictionary.load(dictionary_path=self.input()[1].path)
		with self.input()[0].open("r") as f: batch_vectorizer = pickle.load(f)


		reg_first_step = {
					"smooth_phi_grb"  : 1000.0,
					"decorrelate_phi" : 100000.0
				}

		reg_second_step = {					
	                       "sparse_phi_sbj_text"     : -1.0,
					       "sparse_phi_sbj_journal"  : -5.0,
					       "sparse_phi_sbj_country"  : -0.0,
					       "sparse_phi_sbj_language" : -0.0,
					       "sparse_theta"            : -10.0
					    }

		model = TopicModel(self.params, batch_vectorizer, dictionary)

		model.set_regularization_coefficients(reg_first_step)
		model.fit(num_collection_passes=20)

		model.set_regularization_coefficients(reg_second_step)
		model.fit(num_collection_passes=10)

		model.print_scores()

		model.top_words()
		model.extract_matricies()

		journal_map = model.modality_map("journal")
		country_map = model.modality_map("country")
		language_map = model.modality_map("language")

		top_words = model.top_words_dict
		matricies = model.matricies

		with self.output()[0].open("w") as f: pickle.dump(top_words, f)
		with self.output()[1].open("w") as f: pickle.dump(matricies, f)
		with self.output()[2].open("w") as f: pickle.dump(journal_map, f)
		with self.output()[3].open("w") as f: pickle.dump(country_map, f)
		with self.output()[4].open("w") as f: pickle.dump(language_map, f)


	def output(self):
		return [luigi.LocalTarget("{0}/model/top_words_{1}.pickle"\
			                      .format(self.result_folder, self.params["model_name"])),
		        luigi.LocalTarget("{0}/model/matricies_{1}.pickle"\
		        	              .format(self.result_folder, self.params["model_name"])),
		        luigi.LocalTarget("{0}/model/journal_map_{1}.pickle"\
		        	              .format(self.result_folder, self.params["model_name"])),
		        luigi.LocalTarget("{0}/model/country_map_{1}.pickle"\
		        	              .format(self.result_folder, self.params["model_name"])),
		        luigi.LocalTarget("{0}/model/language_map_{1}.pickle"\
		        	              .format(self.result_folder, self.params["model_name"]))]



if __name__ == '__main__':
	luigi.run()
