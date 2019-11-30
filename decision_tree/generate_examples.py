import collections
import json
import sys

TRUE_POSITIVES = "tp"
TRUE_NEGATIVES = "tn"
FALSE_POSITIVES = "fp"
FALSE_NEGATIVES = "fn"

TOKEN_WINDOW = 50

def pair_in_predicted(pair, predicted_clusters):
  antecedent, consequent = pair
  for cluster in predicted_clusters:
    if antecedent in cluster and consequent in cluster:
      return True
  return False

def get_examples_from_clusters(true_clusters, predicted_clusters):
  examples = collections.defaultdict(list)

  # True positives and false negatives
  for cluster in true_clusters:
    for i, true_antecedent in enumerate(cluster):
      for true_consequent in cluster[i+1:]:
        if true_consequent[0] - true_antecedent[0] < TOKEN_WINDOW:
          pair = (true_antecedent, true_consequent)
          if pair_in_predicted(pair, predicted_clusters):
            examples[TRUE_POSITIVES].append(pair)
          else:
            examples[FALSE_NEGATIVES].append(pair)

  # False positives
  for cluster in predicted_clusters:
    for i, predicted_antecedent in enumerate(cluster):
      for predicted_consequent in cluster[i+1:]:
        if predicted_consequent[0] - predicted_antecedent[0] < TOKEN_WINDOW:
          pair = (predicted_antecedent, predicted_consequent)
          if pair not in examples[TRUE_POSITIVES]:
            examples[FALSE_POSITIVES].append(pair)
    
    # True negatives
    not_false_negatives = sum(
      [examples[TRUE_POSITIVES], examples[FALSE_POSITIVES], examples[FALSE_NEGATIVES]], [])
    all_mentions = sum(true_clusters, [])
    for i, maybe_antecedent in enumerate(all_mentions):
      for maybe_consequent in all_mentions[i+1:]:
        if maybe_consequent[0] - maybe_antecedent[0] < TOKEN_WINDOW:
          pair = (maybe_antecedent, maybe_consequent)
          if pair not in not_false_negatives:
            examples[FALSE_NEGATIVES].append(pair)
        
  

def main():
  result_file = sys.argv[1]
  all_dt_examples = []
  with open(result_file, 'r') as f:
    result_examples = [json.loads(line.strip()) for line in f.readlines()]
    dt_examples = get_examples_from_clusters(
      example["doc_key"], example["clusters"], example["predicted_clusters"])


if __name__ == "__main__":
  main()
