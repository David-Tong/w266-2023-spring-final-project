# question-answering-model
Question Answering Model

1. Run SQuAD2_Baseline_And_Improvement.ipynb

Train Baseline Model - distilbert-base-uncased

Train Improved Type 1 Model - albert-base-v2

Train Improved Type 1 Model - SpanBERT/spanbert-large-cased

Train Improved Type 1 Model - t5-small

Get eval_null_odds.json file and eval_nbest_predictions.json file

2. Run SQuAD2_Skim_Read_Model.ipynb

Train Improved Type 2 Model - skim-read-model, skim predictor part

Get skim_null_odds.json file, reuse eval_nbest_predictions.json file in step 1

3. squad2-skim-read-predictor

Run predict-v2.0.py for skim-read model

python predict-v2.0.py --input_null_files input/skim_null_odds.json --input_nbest_files input/nbest_predictions.json --predict_file output/skim_read_predictions.json

Run predict-v2.0.py for elbow effort model, by changing --threshold value

python predict-v2.0.py --input_null_files null_odds.json --input_nbest_files nbest_predictions.json --predict_file elbow_predictions.json --threshold -5

Evaluate and get final result for SQuAD 2.0

python evaluate-v2.0.py dev-v2.0.json output/predictions.json