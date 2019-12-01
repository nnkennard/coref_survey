import convert_lib
import collections
import os

CONLL12 = convert_lib.DatasetName.conll12

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

def split_parse_label(label):
  curr_chunk = ""
  chunks = []
  for c in label:
    if c in "()":
      if curr_chunk:
        chunks.append(curr_chunk)
      curr_chunk = c
    else:
      curr_chunk += c
  chunks.append(curr_chunk)
  return chunks


def parse_to_spans(parse_col):
  span_starts = collections.defaultdict(list)
  stack = []
  label_map = {}
  for i, orig_label in enumerate(parse_col):
    labels = split_parse_label(orig_label)
    for label in labels:
      if label.endswith(")"):
        span_start, start_idx = stack.pop(0)
        assert (span_start, i) not in label_map
        label_map[(start_idx, i)] = span_start + label
      elif label.startswith("("):
        stack.insert(0, [label, i])
      else:
        stack[0][0] += label
  return [(k[0], k[1], v) for k,v in label_map.items()]

def ldd_append(ldd, to_append):
  for k, v in to_append.items():
    ldd[k] += v
  return ldd

def get_lines_from_file(filename):
  with open(filename, 'r') as f:
    return f.readlines()

def add_sentence(curr_doc, curr_sent, doc_spans, sentence_offset):
  curr_doc.speakers.append(curr_sent[convert_lib.LabelSequences.SPEAKER])
  curr_doc.sentences.append(curr_sent[convert_lib.LabelSequences.WORD])
  coref_spans = coref_to_spans(
      curr_sent[convert_lib.LabelSequences.COREF], sentence_offset)
  doc_spans = ldd_append(doc_spans, coref_spans)
  parse_spans = parse_to_spans(
      curr_sent[convert_lib.LabelSequences.PARSE])
  curr_doc.parse_spans.append(parse_spans)
  curr_doc.pos.append(curr_sent[convert_lib.LabelSequences.POS])


  return doc_spans, sentence_offset + len(curr_sent[convert_lib.LabelSequences.WORD])


def create_dataset(filename, field_map):
 
  dataset = convert_lib.Dataset(CONLL12)

  document_counter = 0
  sentence_offset = 0

  curr_doc = None
  curr_doc_id = None
  curr_sent = collections.defaultdict(list)
  doc_spans = collections.defaultdict(list)

  for line in get_lines_from_file(filename):

    if line.startswith("#begin"):
      assert curr_doc is None 
      curr_doc_id = line.split()[2][1:-2].replace("/", "-")
      part = str(int(line.split()[-1]))
      curr_doc = convert_lib.Document(curr_doc_id, part)
      sentence_offset = 0
    
    elif line.startswith("#end"):
      curr_doc.clusters = list(doc_spans.values())
      dataset.documents.append(curr_doc)
      doc_spans = collections.defaultdict(list)
      curr_doc = None

    elif not line.strip():
      if curr_sent:
        doc_spans, sentence_offset = add_sentence(curr_doc, curr_sent, doc_spans, sentence_offset)
        curr_sent = collections.defaultdict(list)

    else:
      fields = line.split()
      for field_name, field_index in field_map.items():
        curr_sent[field_name].append(fields[field_index])

  return dataset


ONTONOTES_FIELD_MAP = {
  convert_lib.LabelSequences.WORD: 3,
  convert_lib.LabelSequences.POS: 4, 
  convert_lib.LabelSequences.PARSE: 5, 
  convert_lib.LabelSequences.SPEAKER: 9, 
  convert_lib.LabelSequences.COREF: -1,
}

def convert(data_home):
  ontonotes_directory = os.path.join(data_home, "original", "CoNLL12/flat/")
  output_directory = os.path.join(data_home, "processed", CONLL12)
  convert_lib.create_processed_data_dir(output_directory)
  ontonotes_datasets = {}
  for split in [convert_lib.DatasetSplit.train, convert_lib.DatasetSplit.dev]:
    input_filename = ''.join([ontonotes_directory, split, ".", convert_lib.FormatName.txt])
    converted_dataset = create_dataset(input_filename, ONTONOTES_FIELD_MAP)
    convert_lib.write_converted(converted_dataset, output_directory + "/" + split)
