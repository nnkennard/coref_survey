from sklearn import tree
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

def featurize(mentions):
  mention_feature_map = {}
  for doc, (start, end) in mentions:
    mention_feature_map[(doc, (start, end))] = [int(end) - int(start)]
  return mention_feature_map

class DecisionTreeExample(object):
  def __init__(self, doc_key, antecedent, consequent, label):
    self.antecedent = (doc_key, antecedent)
    self.consequent = (doc_key, consequent)
    self.label = label

def main():
  example_file, jsonline_file = sys.argv[1:]
  
  dt_examples = []
  with open(example_file, 'r') as f:
    for line in f:
      (doc_key, label, antecedent_start, antecedent_end, consequent_start,
       consequent_end) = line.strip().split()
      dt_examples.append(DecisionTreeExample(doc_key, (antecedent_start, antecedent_end),
                                        (consequent_start, consequent_end), label))
  
  examples = []
  labels = []
  
  mentions = get_mention_list(dt_examples)
  mention_feature_map = featurize(mentions)
  for example in dt_examples:
    pair_features = mention_feature_map[example.antecedent] + mention_feature_map[example.consequent]
    examples.append(pair_features)
    labels.append(CATEGORIES.index(example.label))


  examples, labels = np.array(examples), np.array(labels)
  
  clf = tree.DecisionTreeClassifier()
  clf = clf.fit(examples, labels)
  tree.plot_tree(clf) 
  dot_data = tree.export_graphviz(clf, out_file=None) 
  graph = graphviz.Source(dot_data) 
  graph.render("iris")



if __name__ == "__main__":
  main()
