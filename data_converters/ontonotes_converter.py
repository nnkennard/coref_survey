import convert_lib
import conll_lib
import collections
import os

CONLL12 = convert_lib.DatasetName.conll12

def ldd_append(ldd, to_append):
  for k, v in to_append.items():
    ldd[k] += v
  return ldd

def get_lines_from_file(filename):
  with open(filename, 'r') as f:
    return f.readlines()

def add_sentence(curr_doc, curr_sent):
  # This is definitely passed by reference right
  words, speakers = zip(*curr_sent)
  curr_doc.sentences.append(words)
  curr_doc.speakers.append(speakers)


def create_dataset(filename):
 
  dataset = convert_lib.Dataset(CONLL12)

  document_counter = 0
  sentence_offset = 0

  curr_doc = None
  curr_doc_name = None
  curr_sent = []
  curr_sent_orig_coref_labels = []
  all_spans = collections.defaultdict(list)

  for line in get_lines_from_file(filename):

    if not line.strip():
      # add sentence
      if curr_sent:
        add_sentence(curr_doc, curr_sent)
        coref_spans = conll_lib.get_spans_from_conll(curr_sent_orig_coref_labels,
            sentence_offset)
        all_spans = ldd_append(all_spans, coref_spans)
        sentence_offset += len(curr_sent)
      curr_sent = []
      curr_sent_orig_coref_labels = []
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
        sentence_offset = 0
      
      word = fields[3]
      coref_label = fields[-1]
      curr_sent_orig_coref_labels.append(coref_label)
      curr_sent.append((word, convert_lib.NO_SPEAKER))
        
  curr_doc.clusters = list(all_spans.values())
  dataset.documents.append(curr_doc)
  return dataset


def convert(data_home):
  ontonotes_directory = os.path.join(data_home, "original", "CoNLL12/conll2012-")
  output_directory = os.path.join(data_home, "processed", CONLL12)
  convert_lib.create_processed_data_dir(output_directory)
  ontonotes_datasets = {}
  for split in [convert_lib.DatasetSplit.train, convert_lib.DatasetSplit.dev]:
    input_filename = ''.join([ontonotes_directory, split, ".", convert_lib.FormatName.txt])
    converted_dataset = create_dataset(input_filename)
    convert_lib.write_converted(converted_dataset, output_directory + "/" + split)
