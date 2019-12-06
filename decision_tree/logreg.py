import collections
import json
import random
from sklearn import preprocessing
import sys
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics#, cross_validation
import pandas as pd
import numpy as np

TRUE_POSITIVES = "tp"
TRUE_NEGATIVES = "tn"
FALSE_POSITIVES = "fp"
FALSE_NEGATIVES = "fn"

CATEGORIES = [TRUE_POSITIVES, FALSE_POSITIVES, TRUE_NEGATIVES, FALSE_NEGATIVES]


def regroup_labels(labels):
  replication_positives = [TRUE_POSITIVES, FALSE_POSITIVES]
  detection_positives = [TRUE_POSITIVES, TRUE_NEGATIVES]
  
  return [str(i in replication_positives) for i in labels], [str(i in detection_positives) for i in labels]
  

def remove_repeats(sequence):
  print(sequence)
  new_sequence = [sequence[0]]
  for i, item in enumerate(sequence[1:]):
    if item == sequence[i]:
      continue
    else:
      new_sequence.append(item)
  return tuple(new_sequence)


def get_mentions(data_frame):
  mentions = set()
  for _, example in data_frame.iterrows():
    mentions.add((example["doc_key"], 
                  example["ant_start"],
                  example["ant_end"]))
    mentions.add((example["doc_key"], 
                  example["con_start"],
                  example["con_end"]))
  return sorted(list(mentions))


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




def get_features(mentions, data_file):
  docs = {}
  with open(data_file, 'r') as f:
    for line in f:
      obj = json.loads(line.strip())
      docs[obj["doc_key"]] = obj

  features = {}
  for doc_key, start, end in mentions:

    doc = docs[doc_key]
    parse_spans = flatten_parse_spans(doc["parse_spans"], doc["token_sentences"])
    features[(doc_key, start, end)] = [(end - start) % 3, 
                     parse_spans.get((start, end), "-"),
                     "_".join(remove_repeats(sum(doc["pos"], [])[start:end+1])),
                    ]

  return features

def featurize(example, features):
  ant = (example["doc_key"], example["ant_start"], example["ant_end"])
  con = (example["doc_key"], example["con_start"], example["con_end"])
  #return features[ant] + features[con] + [(int(example["con_start"]) - int(example["ant_start"])) % 5]
  return features[con] + [(int(example["con_start"]) - int(example["ant_start"])) % 5]

def main():

  data_names = ["doc_key", "ant_start", "ant_end", "con_start", "con_end", "label"]
  data_frame = pd.read_csv(sys.argv[1], sep="\t", names=data_names)
  mentions = get_mentions(data_frame)
  features = get_features(mentions, sys.argv[2])

  feature_df_builder = []
  label_builder = []
  for i, example in data_frame.iterrows():
    feature_df_builder.append(featurize(example, features))
    label_builder.append(example["label"])

  replication_labels, detection_labels = regroup_labels(label_builder)
  print(collections.Counter(label_builder))
  print(collections.Counter(replication_labels))
  print(collections.Counter(detection_labels))


  for label_builder in [replication_labels, detection_labels]:
    joint_list = list(zip(feature_df_builder, label_builder)) 
    random.shuffle(joint_list)
    feature_df_builder, label_builder = zip(*joint_list)

    feature_df_builder = feature_df_builder
    label_builder = label_builder

    boundary = int(0.9 * len(label_builder))

    X_train, X_test = feature_df_builder[:boundary], feature_df_builder[boundary:]
    y_train, y_test = label_builder[:boundary], label_builder[boundary:]

    print(collections.Counter(y_train))
    print(collections.Counter(y_test))
    
    enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
    enc.fit(feature_df_builder)
    X_train_one_hot = enc.transform(X_train)
    y_train = np.array(y_train).reshape((len(y_train),))

    clf = LogisticRegressionCV(class_weight = 'balanced',
          cv=5, random_state=0, max_iter=10000).fit(X_train_one_hot, y_train)

    predicted = clf.predict(enc.transform(X_test))
      
    print(metrics.accuracy_score(y_test, predicted))
    print(metrics.classification_report(y_test, predicted))

    coefs=clf.coef_[0]
    top_three = np.argpartition(coefs, -15)[-15:]
    top_three_sorted=top_three[np.argsort(coefs[top_three])]
    print(enc.get_feature_names()[top_three_sorted])

    print()
    bot_three = np.argpartition(coefs, 15)[:15]
    bot_three_sorted=bot_three[np.argsort(coefs[bot_three])]
    print(enc.get_feature_names()[bot_three_sorted])

if __name__ == "__main__":
  main()
