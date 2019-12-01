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

def add_sentence(curr_doc, curr_sent, all_spans, sentence_offset):
  curr_doc.speakers.append(curr_sent[convert_lib.LabelSequences.SPEAKER])
  curr_doc.sentences.append(curr_sent[convert_lib.LabelSequences.WORD])
  coref_spans = conll_lib.get_spans_from_conll(
      curr_sent[convert_lib.LabelSequences.COREF],
      sentence_offset)
  all_spans = ldd_append(all_spans, coref_spans)

  return all_spans, sentence_offset + len(curr_sent[convert_lib.LabelSequences.WORD])


def create_dataset(filename, field_map):
 
  dataset = convert_lib.Dataset(CONLL12)

  document_counter = 0
  sentence_offset = 0

  curr_doc = None
  curr_doc_id = None
  curr_sent = collections.defaultdict(list)
  all_spans = collections.defaultdict(list)

  for line in get_lines_from_file(filename):

    if line.startswith("#begin"):
      assert curr_doc is None 
      curr_doc_id = line.split()[2][1:-2].replace("/", "-")
      part = str(int(line.split()[-1]))
      curr_doc = convert_lib.Document(curr_doc_id, part)
      sentence_offset = 0
    
    elif line.startswith("#end"):
      curr_doc.clusters = list(all_spans.values())
      dataset.documents.append(curr_doc)
      all_spans = collections.defaultdict(list)
      curr_doc = None

    elif not line.strip():
      if curr_sent:
        all_spans, sentence_offset = add_sentence(curr_doc, curr_sent, all_spans, sentence_offset)
        curr_sent = collections.defaultdict(list)

    else:
      fields = line.split()
      for field_name, field_index in field_map.items():
        curr_sent[field_name].append(fields[field_index])
        #print(field_name, fields[field_index])

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
