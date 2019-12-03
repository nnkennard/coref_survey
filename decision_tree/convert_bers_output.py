import ontonotes_converter
import json
import sys
import collections
import os

class LabelSequences(object):
  WORD = "WORD"
  COREF = "COREF"


def coref_to_spans(coref_col, offset):
  span_starts = collections.defaultdict(list)
  complete_spans = []
  for i, orig_label in enumerate(coref_col):
    if orig_label == '-':
      continue
    else:
      labels = orig_label.split("|")
      for label in labels:
        if label.startswith("("):
          if label.endswith(")"):
            complete_spans.append((i, i, label[1:-1]))
          else:
            span_starts[label[1:]].append(i)
        elif label.endswith(")"):
          ending_cluster = label[:-1]
          assert len(span_starts[ending_cluster]) in [1, 2]
          maybe_start_idx = span_starts[ending_cluster].pop(-1)
          complete_spans.append((maybe_start_idx, i, ending_cluster))

  span_dict = collections.defaultdict(list)
  for start, end, cluster in complete_spans:
    span_dict[cluster].append((offset + start, offset + end))

  return span_dict

def ldd_append(ldd, to_append):
  for k, v in to_append.items():
    ldd[k] += v
  return ldd

def get_lines_from_file(filename):
  with open(filename, 'r') as f:
    return f.readlines()

def add_sentence(curr_sent, doc_spans, sentence_offset):
  coref_spans = coref_to_spans(
      curr_sent[LabelSequences.COREF], sentence_offset)
  doc_spans = ldd_append(doc_spans, coref_spans)

  return doc_spans, sentence_offset + len(curr_sent[LabelSequences.WORD])


def get_bers_clusters(filename, field_map):

  doc_to_clusters = {}
 
  document_counter = 0
  sentence_offset = 0

  curr_doc = None
  curr_doc_id = None
  curr_sent = collections.defaultdict(list)
  doc_spans = collections.defaultdict(list)

  for line in get_lines_from_file(filename):

    if line.startswith("#begin"):
      part = str(int(line.split()[-1]))
      curr_doc_id = line.split()[2][1:-2].replace("/", "-") + "_" + part
      sentence_offset = 0
    
    elif line.startswith("#end"):
      doc_to_clusters[curr_doc_id] = list(doc_spans.values())
      doc_spans = collections.defaultdict(list)
      del(curr_doc_id)

    elif not line.strip():
      if curr_sent:
        doc_spans, sentence_offset = add_sentence(curr_sent, doc_spans, sentence_offset)
        curr_sent = collections.defaultdict(list)

    else:
      fields = line.split()
      for field_name, field_index in field_map.items():
        curr_sent[field_name].append(fields[field_index])

  return doc_to_clusters


BERS_FIELD_MAP = {
  LabelSequences.WORD: 3,
  LabelSequences.COREF: -1,
}

def main():
  bers_output_file, spanbert_output_file = sys.argv[1:]
    
  bers_clusters = get_bers_clusters(bers_output_file, BERS_FIELD_MAP)

  with open(spanbert_output_file, 'r') as f:
    with open(spanbert_output_file.replace("spanbert-base", "bers"), 'w') as g:
      for line in f:
        spanbert_obj = json.loads(line.strip())
        new_obj = json.loads(line.strip())
        new_obj["predicted_clusters"] = bers_clusters[spanbert_obj["doc_key"]]
        g.write(json.dumps(new_obj) + "\n")


if __name__ == "__main__":
  main()
