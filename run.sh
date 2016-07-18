#!/bin/sh
# python2 firstTask.py \
#         TopicModel \
#         --params '{"model_name": "myModel", "main_modality": "text", "num_top_tokens": 10, "num_topics": 20}' \
#         --local-scheduler

python2 secondTask.py \
	FindSynonyms \
	--local-scheduler