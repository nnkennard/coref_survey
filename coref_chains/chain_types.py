import conll_lib
import convert_lib
import collections
import os
import sys

CONLL12 = convert_lib.DatasetName.conll12

def get_lines_from_file(filename):
  with open(filename, 'r') as f:
    return f.readlines()

def create_dataset(filename):
 
  dataset = convert_lib.Dataset(CONLL12)

  document_counter = 0
  sentence_offset = 0

  curr_doc = None
  curr_doc_name = None
  #curr_sent = []
  curr_sent_orig_labels = []
  all_spans = collections.defaultdict(list)
  sentence_idx = 0

  for line in get_lines_from_file(filename):

    if line.startswith("#"):
      continue
    if not line.strip():
      # add sentence
      if curr_sent_orig_labels:
        (parts, tokens, pos, parse, speakers, ner, coref) = zip(*curr_sent_orig_labels)
        #add_sentence(curr_doc, curr_sent)
        coref_spans = conll_lib.get_spans_from_conll(coref, 0)
        parse_spans = conll_lib.get_parse_spans_from_conll(parse)

        for entity, cluster in coref_spans.items():
          for start, inclusive_end in cluster:
            assert len(set(parts)) == 1
            assert len(set(speakers)) == 1
            end = inclusive_end + 1 
            parse_label = parse_spans.get((start, inclusive_end), "_")
            print("\t".join([curr_doc_id,
              parts[0],
              entity, str(sentence_idx), str(start),
              parse_label.replace("*", ""),
              " ".join(pos[start:end]),
              " ".join(tokens[start:end]),
              " ".join(ner[start:end]).replace("*", "").replace(" ", ""),
              speakers[0]
            ]))
        sentence_idx += 1

      curr_sent = []
      curr_sent_orig_labels = []
    else:
      fields = line.split()
      # check for new doc
      doc_name, part = line.split()[:2]
      doc_name = doc_name.replace("/", "-")
      if not doc_name == curr_doc_name:
        if curr_doc is not None:
          curr_doc.clusters = list(all_spans.values())
          all_spans = collections.defaultdict(list)
          dataset.documents.append(curr_doc)
        curr_doc_name = doc_name
        curr_doc_id = convert_lib.make_doc_id(CONLL12, doc_name)
        curr_doc = convert_lib.Document(curr_doc_id, part)
        sentence_idx = 0
        sentence_offset = 0
      

      doc, part, token_idx, token, pos, parse, _, _, _, speaker, ner = fields[:11]
      coref = fields[-1]

      curr_sent_orig_labels.append((part, token, pos, parse, speaker, ner, coref))
      #curr_sent.append((word, convert_lib.NO_SPEAKER))
        
  curr_doc.clusters = list(all_spans.values())
  dataset.documents.append(curr_doc)
  return dataset


def main():
  _ = create_dataset(sys.argv[1])



if __name__ == "__main__":
  main()
