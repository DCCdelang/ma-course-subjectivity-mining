from ml_pipeline import experiment
import argparse

parser = argparse.ArgumentParser(description='run classifier on data')
parser.add_argument('--task', dest='task', default="vua_format") # choose format
# parser.add_argument('--data_dir', dest='data_dir', default="data/test/")
parser.add_argument('--data_dir', dest='data_dir', default="data/") # choose directory
parser.add_argument('--pipeline', dest='pipeline', default='svm_libsvc_embed') # choose pipeline --> see experiments.pipeline(name) or cnn_raw / cnn_prep
parser.add_argument('--print_predictions', dest='print_predictions', default=False)
args = parser.parse_args()

GridSearch = False
ImpFea = False
conf_matrix = True

experiment.run(args.task, args.data_dir, args.pipeline, args.print_predictions, GridSearch, ImpFea,conf_matrix)

