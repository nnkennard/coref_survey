from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegressionCV
import json
import graphviz 
import numpy as np
import sys
import h2o
from tree_categorical_variables import *

h2o.init()

TRUE_POSITIVES = "tp"
TRUE_NEGATIVES = "tn"
FALSE_POSITIVES = "fp"
FALSE_NEGATIVES = "fn"

CATEGORIES = [TRUE_POSITIVES, FALSE_POSITIVES, TRUE_NEGATIVES, FALSE_NEGATIVES]


def get_top_n_RFE_features(model, n, examples, labels):
  feature_indices = []

  for i in range(n):
    rfe = RFE(model, n+1)
    fit = rfe.fit(examples, labels)
    for i, included in enumerate(fit.support_):
      if included and i not in feature_indices:
        feature_indices.append(i)
  
  return feature_indices
    

def regroup_labels(labels):
  replication_positives = [TRUE_POSITIVES, FALSE_POSITIVES]
  detection_positives = [TRUE_POSITIVES, TRUE_NEGATIVES]
  
  return [int(i in replication_positives) for i in labels], [int(i in detection_positives) for i in labels]
  

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
  
def _get_feature_importances(estimator, norm_order=1):
    """Retrieve or aggregate feature importances from estimator"""
    importances = getattr(estimator, "feature_importances_", None)

    coef_ = getattr(estimator, "coef_", None)
    if importances is None and coef_ is not None:
        if estimator.coef_.ndim == 1:
            importances = np.abs(coef_)

        else:
            importances = np.linalg.norm(coef_, axis=0,
                                         ord=norm_order)

    elif importances is None:
        raise ValueError(
            "The underlying estimator %s has no `coef_` or "
            "`feature_importances_` attribute. Either pass a fitted estimator"
            " to SelectFromModel or call fit before calling transform."
            % estimator.__class__.__name__)

    return importances  

def get_onehot(value_list, target):
  onehot = [0] * len(value_list)
  for i, value in enumerate(value_list):
    if value == target:
      onehot[i] = 1
      assert sum(onehot) == 1
      return onehot
  assert False
    
def get_parse_onehot(span, parse_spans, all_parse_labels):
  return get_onehot(all_parse_labels, parse_spans.get(span, "-"))

def featurize(mentions, json_examples, all_parse_labels):
  mention_feature_map = {}
  for doc_key, (start, end) in mentions:
    doc = json_examples[doc_key]
    parse_spans = flatten_parse_spans(doc["parse_spans"], doc["token_sentences"])
    #mention_feature_map[
    #  (doc_key, (start, end))] = [end - start] + get_parse_onehot((start, end), parse_spans, all_parse_labels)
    mention_feature_map[
      (doc_key, (start, end))] = [end - start] + [parse_spans.get((start, end), "-")]

  return mention_feature_map

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
      (doc_key, antecedent_start, antecedent_end, consequent_start,
       consequent_end, label) = line.strip().split()
      dt_examples.append(DecisionTreeExample(doc_key, (int(antecedent_start), int(antecedent_end)),
                                        (int(consequent_start), int(consequent_end)), label))
  
  examples = []
  labels = []
  
  mentions = get_mention_list(dt_examples)
  all_parse_labels = get_all_parse_labels(json_examples)
  mention_feature_map = featurize(mentions, json_examples, all_parse_labels)

  for example in dt_examples:
    pair_features = mention_feature_map[example.antecedent] + mention_feature_map[example.consequent]#+ [example.label]
    examples.append(pair_features)
    labels.append(CATEGORIES.index(example.label))

  replication_labels, error_detection_labels = regroup_labels(labels)
  feature_names =[
      "antecedent_len", "antecedent_span_label",
      "consequent_len", "consequent_span_label", "label"]
  target_col = "label"
  h2o_examples = h2o.H2OFrame.from_python(examples, column_names=feature_names)
  pd_examples = pd.DataFrame(examples, columns=feature_names)
  print(pd_examples)
  h2ofr = h2o.H2OFrame(pd_examples)
  h2ofr.col_names = list(feature_names)
  model=H2ORandomForestEstimator(ntrees=100, categorical_encoding="sort_by_response", max_depth=100)
  model.train(x=feature_names,
                    y=target_col,
                    training_frame=h2ofr)
  print(dir(model))
  print(model.categorical_encoding)
  print(model.model_performance(test_data=h2ofr))

  from h2o.tree import H2OTree

  first_tree = H2OTree(model=model, tree_number = 0, tree_class='fp')
  print(first_tree)
  print(first_tree.root_node)

 
if __name__ == "__main__":
  main()
