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
  curr_doc.speakers.append(curr_sent[convert_lib.LabelSequences.SPEAKER])
  curr_doc.sentences.append(curr_sent[convert_lib.LabelSequences.WORD])


def create_dataset(filename, field_map):
 
  dataset = convert_lib.Dataset(CONLL12)

  document_counter = 0
  sentence_offset = 0

  curr_doc = None
  curr_doc_name = None
  #curr_sent = []
  curr_sent_labels = collections.defaultdict(list)
  all_spans = collections.defaultdict(list)

  for line in get_lines_from_file(filename):

    if line.startswith("#"):
      continue

    if not line.strip():
      # add sentence
      if curr_sent_labels:
        add_sentence(curr_doc, curr_sent_labels)
        coref_spans = conll_lib.get_spans_from_conll(
            curr_sent_labels[convert_lib.LabelSequences.COREF],
            sentence_offset)
        all_spans = ldd_append(all_spans, coref_spans)
        sentence_offset += len(curr_sent_labels[convert_lib.LabelSequences.WORD])
      #curr_sent = []
      curr_sent_labels = collections.defaultdict(list)
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

      for field_name, field_index in field_map.items():
        curr_sent_labels[field_name].append(fields[field_index])

  curr_doc.clusters = list(all_spans.values())
  dataset.documents.append(curr_doc)
  return dataset


ONTONOTES_FIELD_MAP = {
  convert_lib.LabelSequences.WORD: 3,
  convert_lib.LabelSequences.COREF: -1,
  convert_lib.LabelSequences.POS: 4, 
  convert_lib.LabelSequences.PARSE: 5, 
  convert_lib.LabelSequences.SPEAKER: 9, 
}

def convert(data_home):
  ontonotes_directory = os.path.join(data_home, "original", "CoNLL12/flat/")
  output_directory = os.path.join(data_home, "processed", CONLL12)
  convert_lib.create_processed_data_dir(output_directory)
  ontonotes_datasets = {}
  for split in [convert_lib.DatasetSplit.train, convert_lib.DatasetSplit.dev]:
    input_filename = ''.join([ontonotes_directory, split, ".", convert_lib.FormatName.txt])
    converted_dataset = create_dataset(input_filename, ONTONOTES_FIELD_MAP)
    print(dir(converted_dataset))
    convert_lib.write_converted(converted_dataset, output_directory + "/" + split)
