import transformers
import torch
import numpy as np
from datasets import load_dataset, load_metric

from transformers import (AutoTokenizer, 
                          AutoModelForSeq2SeqLM, 
                          DataCollatorForSeq2Seq, 
                          Seq2SeqTrainingArguments, 
                          Seq2SeqTrainer)
def main():
  dataset = load_dataset('opus_infopankki', 'en-zh', split='train')
  dataset = dataset.train_test_split(test_size=0.1)

  metric=load_metric('sacrebleu')

  model_name = 'Helsinki-NLP/opus-mt-zh-en'
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

  max_input_length = 128
  max_target_length = 128
  source_lang = 'zh'
  target_lang = 'en'

  def preprocess_function(examples):
    inputs = [example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
      labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs 

  tokenized_datasets = dataset.map(preprocess_function, batched=True)

  batch_size = 16

  args = Seq2SeqTrainingArguments(
      'translation',
      evaluation_strategy='epoch',
      learning_rate=2e-5,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      weight_decay=0.01,
      save_total_limit=3,
      num_train_epochs=1,
      predict_with_generate=True)

  data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

  def postprocess_text(preds,labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds,labels

  def compute_metrics(eval_preds):
    preds,labels = eval_preds
    if isinstance(preds, tuple):
      preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels!=-100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds,decoded_labels)
    result = metric.compute(predictions = decoded_preds,references=decoded_labels)
    result = {'bleu':result['score']}

    prediction_lens = [np.count_nonzero(pred!=tokenizer.pad_token_id) for pred in preds]
    result['gen_len'] = np.mean(prediction_lens)
    result = {k:round(v,4) for k,v in result.items()}
    return result

  trainer = Seq2SeqTrainer(
      model,
      args,
      train_dataset=tokenized_datasets['train'].select(range(200)),
      eval_dataset=tokenized_datasets['test'].select(range(50)),
      data_collator=data_collator,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics)

  trainer.train()
  trainer.save_model('./translation_checkpoint')

  trainer.evaluate()

  predict_dataset = tokenized_datasets['test'].select(range(100))
  predict_results = trainer.predict(predict_dataset,
                                    metric_key_prefix='predict',
                                    max_length=max_target_length)

  predictions = tokenizer.batch_decode(predict_results.predictions,
                                       skip_special_tokens=True,
                                       clean_up_tokenization_spaces=True)

  predictions = [pred.strip() for pred in predictions]

if __name__ == '__main__':
    main()

