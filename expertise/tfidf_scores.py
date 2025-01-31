import argparse
import json
import os
from collections import OrderedDict
from expertise.config import ModelConfig

from .preprocess.textrank import run_textrank
from .models.tfidf.train_tfidf import train
from .models.tfidf.infer_tfidf import infer

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('config_path', help="a config file for a model")
	args = parser.parse_args()

	config_path = os.path.abspath(args.config_path)

	with open(config_path) as f:
	    data = json.load(f, object_pairs_hook=OrderedDict)
	config = ModelConfig(**data)

	textrank_config = run_textrank(config)
	textrank_config.save(args.config_path)

	trained_config = train(config)
	trained_config.save(args.config_path)

	inferred_config = infer(config)
	inferred_config.save(args.config_path)
