class Dataset(object):
  def __init__(self, dataset_id):
    self.documents = []
    pass

  def dump_to_jsonl(self, ):
    pass

  def dump_to_bert_jsonl(self, ):
    pass



class Document(object):
  def __init__(self):
    self.sentences = []
    self.clusters = []


  def dump_to_jsonl(self, ):
    pass

  def dump_to_bert_jsonl(self, ):
    pass
