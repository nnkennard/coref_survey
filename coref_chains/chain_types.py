import conll_lib
import convert_lib
import collections
import os
import sys

CONLL12 = convert_lib.DatasetName.conll12

def get_lines_from_file(filename):
  with open(filename, 'r') as f:
    return f.readlines()

class Mention(object):
  def __init__(self, doc, part, entity, sentence, start, parse, tokens,
                pos, ner, speaker):
    self.doc = doc
    self.part = part
    self.entity = entity
    self.mention_id = "_".join([doc, part, entity])

    self.sentence = sentence
    self.start = start
    self.tokens = tokens

    self.parse = parse
    self.pos = pos
    self.ner = ner
    self.speaker = speaker

  def __str__(self):
    return "\t".join([self.doc,
              self.part,
              self.entity, str(self.sentence), str(self.start),
              #"".join(self.parse).replace("*", ""),
              self.parse.replace("*", ""),
              " ".join(self.pos),
              " ".join(self.tokens),
              " ".join(self.ner).replace("*", "").replace(" ", ""),
              self.speaker
            ])

def create_dataset(filename):

  mentions_map = collections.defaultdict(list)
 
  dataset = convert_lib.Dataset(CONLL12)

  document_counter = 0
  sentence_offset = 0

  curr_doc = None
  curr_doc_name = None
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
        coref_spans = conll_lib.get_spans_from_conll(coref, 0)
        parse_spans = conll_lib.get_parse_spans_from_conll(parse)

        for entity, cluster in coref_spans.items():
          for start, inclusive_end in cluster:
            assert len(set(parts)) == 1
            assert len(set(speakers)) == 1
            end = inclusive_end + 1 

            parse_label = parse_spans.get((start, inclusive_end), "_")
            mention_obj = Mention(curr_doc_id, parts[0], entity, sentence_idx,
              start, parse_label, tokens[start:end], pos[start:end], ner[start:end],
              speakers[0])
  
            mentions_map[mention_obj.mention_id].append(mention_obj)

            
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
        
  curr_doc.clusters = list(all_spans.values())
  dataset.documents.append(curr_doc)


  i = 0
  for entity, mentions in mentions_map.items():
    print(entity)
    print(all_surface_forms(mentions))
    print(canonical_mention(mentions))
    i += 1
    print()

    if i == 1000:
      break


  return dataset

def all_surface_forms(mentions):
  #return [" ".join(mention.tokens) for mention in mentions]
  return ([" ".join(mention.tokens) for mention in mentions],
          [" ".join(mention.ner) for mention in mentions],
          [mention.parse for mention in mentions])

def canonical_mention(mentions):
  selected = 0, mentions[0]
  for i, mention in enumerate(mentions):
    ner_string = "".join(mention.ner)
    if ner_string.startswith("(") and ner_string.endswith(")"):
      selected = (i, mention)
      break

  return selected[0], " ".join(selected[1].tokens)


def main():
  _ = create_dataset(sys.argv[1])


if __name__ == "__main__":
  main()
