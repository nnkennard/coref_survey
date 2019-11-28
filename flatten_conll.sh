find ~/ood_coref/data/original/CoNLL12/train/ \
  -name "*v4_gold_conll" -type f | \
  xargs cat > ~/ood_coref/data/original/CoNLL12/flat/train.txt

find ~/ood_coref/data/original/CoNLL12/dev/ \
  -name "*v4_gold_conll" -type f | \
  xargs cat > ~/ood_coref/data/original/CoNLL12/flat/dev.txt

find ~/ood_coref/data/original/CoNLL12/test/ \
  -name "*v4_gold_conll" -type f | \
  xargs cat > ~/ood_coref/data/original/CoNLL12/flat/test.txt

