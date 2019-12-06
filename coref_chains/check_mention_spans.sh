python check_mention_spans.py /mnt/nfs/work1/mccallum/coref/conll_train_coref_chains.txt | cut -f-2| sort | uniq -c | awk '{print $1"\t"$2"\t"$3}'
