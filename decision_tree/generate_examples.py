import collections
import json
import sys
import random
import os

TRUE_POSITIVES = "tp"
TRUE_NEGATIVES = "tn"
FALSE_POSITIVES = "fp"
FALSE_NEGATIVES = "fn"

CATEGORIES = [TRUE_POSITIVES, FALSE_POSITIVES, TRUE_NEGATIVES, FALSE_NEGATIVES]

TOKEN_WINDOW = 50

def pair_in_predicted(pair, predicted_clusters):
  antecedent, consequent = pair
  for cluster in predicted_clusters:
    if antecedent in cluster and consequent in cluster:
      return True
  return False

def pair_within_window(antecedent, consequent, token_window=TOKEN_WINDOW):
  return consequent[0] - antecedent[0] < token_window

def get_examples_from_clusters(true_clusters, predicted_clusters):
  examples = collections.defaultdict(list)

  # True positives and false negatives
  for cluster in true_clusters:
    for i, true_antecedent in enumerate(cluster):
      for true_consequent in cluster[i+1:]:
        if pair_within_window(true_antecedent, true_consequent):
          pair = (true_antecedent, true_consequent)
          if pair_in_predicted(pair, predicted_clusters):
            examples[TRUE_POSITIVES].append(pair)
          else:
            examples[FALSE_NEGATIVES].append(pair)

  # False positives
  for cluster in predicted_clusters:
    for i, predicted_antecedent in enumerate(cluster):
      for predicted_consequent in cluster[i+1:]:
        if pair_within_window(predicted_antecedent, predicted_consequent):
          pair = (predicted_antecedent, predicted_consequent)
          if pair not in examples[TRUE_POSITIVES]:
            examples[FALSE_POSITIVES].append(pair)
    
  # True negatives
  not_true_negatives = sum(
    [examples[TRUE_POSITIVES], examples[FALSE_POSITIVES], examples[FALSE_NEGATIVES]], [])
  true_negative_candidates = []
  all_mentions = sum(true_clusters, [])
  for i, maybe_antecedent in enumerate(all_mentions):
    for maybe_consequent in all_mentions[i+1:]:
      if pair_within_window(maybe_antecedent, maybe_consequent):
        pair = (maybe_antecedent, maybe_consequent)
        if pair not in not_true_negatives:
          true_negative_candidates.append(pair)

  if len(true_negative_candidates) < len(not_true_negatives):
    examples[TRUE_NEGATIVES] = true_negative_candidates
  else:
    examples[TRUE_NEGATIVES] = random.sample(true_negative_candidates, len(not_true_negatives))

  
  return examples

class Model(object):
  SPANBERT = "spanbert"
  BERS = "bers"

class Boundaries(object):
  GOLD = "gold"
  PREDICTED = "predicted"

class Split(object):
  TRAIN = "train"
  TEST = "test"

def get_output_file_name(output_dir, model, boundaries, dataset, split):
  return os.path.join(output_dir,
    "".join([model, "_", boundaries, "_", dataset, "_dt_", split, ".txt"]))

def split_docs(doc_keys):
  doc_key_list = list(doc_keys)
  random.shuffle(doc_key_list)
  boundary = int(0.8 * len(doc_key_list))
  return doc_key_list[:boundary], doc_key_list[boundary:]


def write_file(filename, all_examples, keys):
  with open(filename, 'w') as f:
    for doc_key in keys:
      for label, examples in all_examples[doc_key].items():
        for example in examples:
          f.write("\t".join([doc_key] + [str(i) for i in sum(example, [])] + [label]) + "\n")

def main():
  random.seed(43)
  result_file, model, boundaries, dataset = sys.argv[1:]
  output_dir = "./"
  all_dt_examples = {}
  with open(result_file, 'r') as f:
    docs = [json.loads(line.strip()) for line in f.readlines()]
  for doc in docs:
    all_dt_examples[doc["doc_key"]] = get_examples_from_clusters(doc["clusters"], doc["predicted_clusters"])


  doc_keys = {}
  doc_keys[Split.TRAIN], doc_keys[Split.TEST] = split_docs(all_dt_examples.keys())

  for split in [Split.TRAIN, Split.TEST]:
    filename = get_output_file_name(output_dir, model, boundaries, dataset, split)
    write_file(filename, all_dt_examples, doc_keys[split])

if __name__ == "__main__":
  main()
