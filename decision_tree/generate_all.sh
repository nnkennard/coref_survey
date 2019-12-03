for spans in gold predicted
do
  for split in train test
  do
    for model in spanbert-base bers
    do

    args="outputs/"$model"_"$spans"_conll-dev.jsonl outputs/conll-dev_dt_"$split".keys "$model" "$spans" conll-dev "$split

    python generate_examples.py $args

    done
  done
done
