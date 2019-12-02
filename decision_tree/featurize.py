from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegressionCV
import json
import graphviz 
import numpy as np
import sys

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
  replication_positives = [CATEGORIES.index(TRUE_POSITIVES), CATEGORIES.index(FALSE_POSITIVES)]
  detection_positives = [CATEGORIES.index(TRUE_POSITIVES), CATEGORIES.index(TRUE_NEGATIVES)]
  
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
    mention_feature_map[
      (doc_key, (start, end))] = [end - start] + get_parse_onehot((start, end), parse_spans, all_parse_labels)

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
    pair_features = mention_feature_map[example.antecedent] + mention_feature_map[example.consequent]
    examples.append(pair_features)
    labels.append(CATEGORIES.index(example.label))
    if len(examples) == 1000:
      break

  examples, labels = np.array(examples), np.array(labels)
  print(examples.shape)
  print(labels.shape)

  replication_labels, error_detection_labels = regroup_labels(labels)

  for x, y in zip(replication_labels, error_detection_labels):
    print(x, y)

  feature_names = ["antecedent_len"] + ["antecedent_" + span for span in all_parse_labels]
  feature_names += ["consequent_len"] + ["consequent_" + span for span in all_parse_labels]
  

  # Replication
  clf = LogisticRegressionCV(cv=5, random_state=0, max_iter=1000,
          multi_class='multinomial').fit(examples, replication_labels)
  sfm = SelectFromModel(clf, 10)
  print(dir(sfm))
  print(dir(clf))
  print(_get_feature_importances(clf))
  print(clf.coef_)

  
  print(list(feature_names[i] for i in get_top_n_RFE_features(clf, 15, examples, replication_labels)))
  
  clf = LogisticRegressionCV(cv=5, random_state=0, max_iter=1000,
          multi_class='multinomial').fit(examples, error_detection_labels)
  print(list(feature_names[i] for i in get_top_n_RFE_features(clf, 15, examples, error_detection_labels)))
  
if __name__ == "__main__":
  main()
