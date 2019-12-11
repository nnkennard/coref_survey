import numpy as np
import pandas as pd
import seaborn as sns
import collections
import sys
import json
import matplotlib.pyplot as plt

def normalize(df):
  result = df.copy()
  for feature_name in df.columns:
    total = df[feature_name].sum()
    result[feature_name] = df[feature_name]/total
  return result

pd.set_option('display.max_columns', 20)

HF_spans = ["Span_(NP)", "Span_(NML)"]
HF_pos = ["POS_PRP$", "POS_NNP"]
OTHER = ["Other"]
MED_FREQ = ["POS_VB", "POS_VBD", "Span_(TOP)", "POS_VBN", "POS_VBG", "POS_VBZ", "POS_VBP", "Other",
        #"Span_(S)", "Span_(VP)", "POS_NN", "POS_PRP", "POS_CD", "POS_JJ", "Span_(ADJP)", "Span_(ADVP)"
        ]

def read_file(filename):
  detected_mention_list = []
  with open(filename, 'r') as f:
    for line in f:
      for doc_key, values in json.loads(line.strip()).items():
        detected_mention_list += [(doc_key.replace("/", "-"), mention["start"], mention["end"]) for mention in values]
  return detected_mention_list


def read_data_from_files(data_path):
  detected_mentions = {}
  for model, span_ratio in [("spanbert_base", "0.4"), ("spanbert_large", "0.4"), ("spanbert_large", "0.8")]:
    filename = "".join([data_path, "/detected-mentions_train_", model, "_conll12_", span_ratio, ".jsonl"])
    detected_mentions[model + "-" + span_ratio] = read_file(filename)

  detected_mentions["bers"] = read_file(data_path + "/bers_mentions_conll-dev.jsonl")
  return detected_mentions

def read_actual_mentions(mention_summary_file, mention_jsonl_file):

  gold_counts = collections.Counter()

  examples = {}
  with open(mention_jsonl_file, 'r') as f:
    for line in f:
      example = json.loads(line.strip())
      examples[example["doc_key"]] = example

  mention_map = {}
  with open(mention_summary_file, 'r') as f:
    for line in f:
      doc, part, entity, sent, start, span, pos, tokens, ner, idk = line.strip().split("\t")
      doc_key = doc[8:].replace("/", "-") + "_" + part
      sentence_offset = len(sum(examples[doc_key]["token_sentences"][:int(sent)], []))
      span_len = len(pos.split())
      if not span.startswith("~"):
        label = (span, None)
        gold_counts["Span_" + span] += 1
      elif span_len == 1:
        label = (None, pos)
        gold_counts["POS_" + pos] += 1
      else:
        label = (None, None)
        gold_counts["Other"] += 1
      token_start = sentence_offset + int(start)
      token_end = sentence_offset + int(start) + span_len - 1
      mention_map[(doc_key , token_start, token_end)] = label
  return mention_map, {"gold": gold_counts}
  
def get_key_from_details(span, pos):    
  if span is not None:
    key = "Span_" + span
  elif pos is not None:
    key = "POS_" + pos
  else:
    key = "Other"
  return key

def main():

  data_path, gold_summary_file, json_file = sys.argv[1:]

  detected_mentions = read_data_from_files(data_path)
  actual_mentions, gold_counts= read_actual_mentions(gold_summary_file, json_file)

  #true_positive_map = collections.defaultdict(lambda : collections.Counter())
  false_negative_map = collections.defaultdict(lambda : collections.Counter())

  # True positives
  #for dataset, mention_values in detected_mentions.items():
  #  for mention in mention_values: 
  #    details = actual_mentions.get(mention, None)
  #    if details is not None:
  #      key = get_key_from_details(*details)
  #      if key in HF_pos + HF_spans :
  #        continue
  #      true_positive_map[dataset][key] += 1

  # False negatives
  for dataset, mention_values in detected_mentions.items():
    false_negatives = set(actual_mentions.keys()) - set(mention_values)
    for mention in false_negatives:
      key = get_key_from_details(*actual_mentions[mention])
      if key in HF_spans + HF_pos:
        continue
      if key not in MED_FREQ:
        continue
      false_negative_map[dataset][key] += 1

  table_totals = collections.defaultdict(lambda:collections.Counter())
  table_tp = collections.defaultdict(lambda:collections.Counter())

  for model, labels in gold_counts.iteritems():
    for label, count in labels.iteritems():
      if label in HF_pos + HF_spans:
        table_totals[model][label] += count
      elif label in OTHER:
        table_totals[model]["Other"] += count
      else:
        table_totals[model]["low_freq"] += count

  print(pd.DataFrame(table_totals).to_csv(sep="\t"))
 
  #for model, labels in true_positive_map.iteritems():
  #  for label, count in labels.iteritems():
  #    if label in HF_pos + HF_spans:
  #      table_tp[model][label] += count
  #    elif label in OTHER:
  #      table_tp[model]["Other"] += count
  #    else:
  #      table_tp[model]["low_freq"] += count

  for label in HF_spans + HF_pos:
    del gold_counts["gold"][label]
  
  for label in gold_counts["gold"].keys():
    if label not in MED_FREQ:
      del gold_counts["gold"][label]


  gold_counts = pd.DataFrame(gold_counts) 
  #true_positive_df = pd.DataFrame(true_positive_map)
  #print(true_positive_df)
  #true_positive_df = pd.concat([true_positive_df, gold_counts], axis = 1)
  #print(true_positive_df)


  false_negative_df = pd.DataFrame(false_negative_map)
  false_negative_df = normalize(pd.concat([gold_counts, false_negative_df], axis=1, sort=True)).sort_values(by=["gold"])
  #false_negative_df.sort_values(by=["gold"])

  print(false_negative_df)

  #from matplotlib.colors import ListedColormap

  sns.set()


    
  fig = plt.figure()
  ax = plt.subplot(111)

  plt.title("Distribution of false negatives")
    
  false_negative_df.T.plot(kind="bar", stacked=True, ax=ax)

    
  handles, labels = ax.get_legend_handles_labels()
  chartBox = ax.get_position()
  ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
  ax.legend(reversed(handles), reversed(labels), loc='upper left', bbox_to_anchor=(1.45, 1.1), shadow=True, ncol=1)

    
  bars = ax.patches
  hatches = ''.join(h*15 for h in 'x/O.')

  #for bar, hatch in zip(bars, hatches):
  #    bar.set_hatch(hatch)

  plt.xticks(rotation=20)
  #true_positive_df.T.plot(kind="bar", stacked=True)
  #plt.legend(loc=8, ncol=5)
  #plt.legend(bbox_to_anchor=(1,0), borderaxespad=0.1, loc="center right", ncol=2, )
  plt.show()


if __name__ == "__main__":
  main()
