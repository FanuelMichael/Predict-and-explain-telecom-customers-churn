import os,warnings
# 1. Suppress standard Python warnings
warnings.filterwarnings('ignore')
# 2. Suppress TensorFlow C++ logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from numba.core.typing.builtins import Print
from Data_analysis import data_analysis


# ========================Whole project modules run begins from this main module================================
def main(data_path='WA_Fn-UseC_-Telco-Customer-Churn.csv', random_state=42):

    Print("===1. Telecom customer dataset analysis===:\n"+str(data_analysis(file_path=data_path)))

if __name__ == "__main__":
    main()
