from sklearn import tree
import json
import graphviz 
import numpy as np
import sys

TRUE_POSITIVES = "tp"
TRUE_NEGATIVES = "tn"
FALSE_POSITIVES = "fp"
FALSE_NEGATIVES = "fn"

CATEGORIES = [TRUE_POSITIVES, FALSE_POSITIVES, TRUE_NEGATIVES, FALSE_NEGATIVES]

def get_mention_list(dt_examples):
  mention_set = set()
  for example in dt_examples:
    mention_set.update([example.antecedent, example.consequent])
  return mention_set

def flatten_parse_spans(parse_spans, token_sentences):
  overall_parse_span_map = {}
  token_count = 0
  for sentence, sentence_parse_spans in zip(token_sentences, parse_spans):
    for start, end, label in sentence_parse_spans:
      overall_span = token_count + start, token_count + end
      assert overall_span not in overall_parse_span_map
      overall_parse_span_map[overall_span] = label.replace("*", "")
    token_count += len(sentence)

  return overall_parse_span_map
    

def get_onehot(value_list, target):
  onehot = [0] * len(value_list)
  for i, value in enumerate(value_list):
    if value == target:
      onehot[i] = 1
      return onehot
    
def featurize(mentions, json_examples, all_parse_labels):
  mention_feature_map = {}
  for doc_key, (str_start, str_end) in mentions:
    start = int(str_start)
    end = int(str_end)
    doc = json_examples[doc_key]
    parse_spans = flatten_parse_spans(doc["parse_spans"], doc["token_sentences"])
    mention_feature_map[(doc_key, (start, end))] = [end - start] + get_onehot(all_parse_labels, parse_spans.get((start, end), "-"))

  return mention_feature_map, sorted(list(all_parse_labels))

class DecisionTreeExample(object):
  def __init__(self, doc_key, antecedent, consequent, label):
    self.antecedent = (doc_key, antecedent)
    self.consequent = (doc_key, consequent)
    self.label = label


def get_all_parse_labels(json_examples):
  all_parse_labels = set()
  for example in json_examples.values():
    for parse_span_map in example["parse_spans"]:
      all_parse_labels.update(x[2].replace("*", "") for x in parse_span_map)
  return sorted(list(all_parse_labels)) + ["-"]


def main():
  example_file, jsonline_file = sys.argv[1:]

  json_examples = {}
  with open(jsonline_file, 'r') as f:
    for line in f:
      example = json.loads(line.strip())
      json_examples[example["doc_key"]] = example
  
  dt_examples = []
  with open(example_file, 'r') as f:
    for line in f:
      (doc_key, label, antecedent_start, antecedent_end, consequent_start,
       consequent_end) = line.strip().split()
      dt_examples.append(DecisionTreeExample(doc_key, (int(antecedent_start), int(antecedent_end)),
                                        (int(consequent_start), int(consequent_end)), label))
  
  examples = []
  labels = []
  
  mentions = get_mention_list(dt_examples)
  all_parse_labels = get_all_parse_labels(json_examples)
  mention_feature_map, all_parse_labels= featurize(mentions, json_examples, all_parse_labels)

  for example in dt_examples:
    pair_features = mention_feature_map[example.antecedent] + mention_feature_map[example.consequent]
    examples.append(pair_features)
    labels.append(CATEGORIES.index(example.label))

  examples, labels = np.array(examples), np.array(labels)

  feature_names = ["antecedent_len"] + ["antecedent_" + span for span in all_parse_labels]
  feature_names += ["consequent_len"] + ["consequent_" + span for span in all_parse_labels]
  
  clf = tree.DecisionTreeClassifier(max_depth=5)
  clf = clf.fit(examples, labels)
  tree.plot_tree(clf) 
  dot_data = tree.export_graphviz(clf, filled=True, rounded=True, feature_names=feature_names, class_names=CATEGORIES, out_file=None) 
  graph = graphviz.Source(dot_data) 
  graph.render("iris")



if __name__ == "__main__":
  main()
