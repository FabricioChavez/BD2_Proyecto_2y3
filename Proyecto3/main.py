import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))

#from Experiments.Time_extractor_KnnSeq import run_knn_seq_experiment
#from Experiments.Time_extractor_KnnLsh import run_knn_lsh_experiment
#from Experiments.Time_extractor_KnnIvf import run_knn_ivf_experiment
from Experiments.time_extractor_KnnSeqRango import run_knn_seq_ran_experiment
if __name__ == "__main__":
    #run_knn_seq_experiment()
    #run_knn_lsh_experiment()
   # run_knn_ivf_experiment()
   run_knn_seq_ran_experiment()